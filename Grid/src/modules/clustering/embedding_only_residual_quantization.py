import logging
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from src.data.loading.components.interfaces import ItemData
from src.models.components.interfaces import OneKeyPerPredictionOutput


class EmbeddingOnlyResidualQuantization(LightningModule):
    """
    Encoder/decoder-only counterpart of
    :class:`src.modules.clustering.residual_quantization.ResidualQuantization`.

    The module wraps two MLPs (an ``encoder`` that compresses the input to a latent
    representation and a ``decoder`` that expands it back to the input dimensionality)
    and is trained to reconstruct its input. It exposes a separate :meth:`encode`
    method so the latent representation can be reused downstream (e.g. as the input
    to a quantization stack).

    The training/validation/test/predict API matches ``ResidualQuantization`` so the
    module can be dropped into the same Lightning training script and Hydra configs
    (such as ``configs/experiment/autoencoder_train_flat.yaml``).
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        normalization_layer: nn.Module = nn.Identity(),
        encoder: nn.Module = nn.Identity(),
        decoder: nn.Module = nn.Identity(),
        reconstruction_loss_function: Optional[nn.Module] = None,
        reconstruction_loss_weight: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            input_dim: Expected feature dimension of the input embeddings. Kept for
                config parity with ``ResidualQuantization``; not used at runtime
                because the encoder/decoder define their own shapes.
            normalization_layer: Module applied to the raw input embeddings before
                encoding. The reconstruction target is the output of this layer.
            encoder: Module that compresses normalized inputs to the latent space.
            decoder: Module that maps the latent representation back to the
                normalized input space.
            reconstruction_loss_function: Loss used to compare the decoder output
                with the normalized input. Defaults to ``nn.MSELoss(reduction="mean")``.
            reconstruction_loss_weight: Scalar weight applied to the reconstruction
                loss when computing the final training loss.
            optimizer: Partial optimizer constructor (e.g. produced by Hydra with
                ``_partial_: true``) called with the module parameters in
                :meth:`configure_optimizers`.
            scheduler: Optional partial LR scheduler constructor.
            verbose: Whether to log progress during training.
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimizer",
                "scheduler",
                "normalization_layer",
                "encoder",
                "decoder",
                "reconstruction_loss_function",
            ],
        )

        self.input_dim = input_dim
        self.normalization_layer = normalization_layer
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = (
            reconstruction_loss_function
            if reconstruction_loss_function is not None
            else nn.MSELoss(reduction="mean")
        )
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.verbose = verbose
        self.log_if_true("Verbose mode enabled", self.verbose)

        self.train_loss = MeanMetric()
        self.train_reconstruction_loss = MeanMetric()
        self.train_mse = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_mse = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_mse = MeanMetric()

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of ``embeddings``.

        Args:
            embeddings: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Latent tensor of shape ``(batch_size, latent_dim)``.
        """
        return self.encoder(self.normalization_layer(embeddings))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Map ``latents`` back to the normalized input space."""
        return self.decoder(latents)

    def forward(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full encode/decode pass.

        Args:
            embeddings: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            normalized_embeddings: Output of ``normalization_layer``, used as the
                reconstruction target.
            latents: Encoder output (the latent representation).
            reconstructed_embeddings: Decoder output, the model's reconstruction
                of ``normalized_embeddings``.
        """
        normalized_embeddings = self.normalization_layer(embeddings)
        latents = self.encoder(normalized_embeddings)
        reconstructed_embeddings = self.decoder(latents)
        return normalized_embeddings, latents, reconstructed_embeddings

    def model_step(
        self, model_input: ItemData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode/decode a batch and compute the reconstruction loss.

        Args:
            model_input: ``ItemData`` whose ``transformed_features`` contains an
                ``"input_embedding"`` tensor of shape ``(batch_size, input_dim)``.

        Returns:
            latents: Latent representation of the batch.
            reconstructed_embeddings: Decoder output.
            reconstruction_loss: Scalar reconstruction loss.
        """
        input_embeddings = model_input.transformed_features["input_embedding"].to(
            self.device
        )
        normalized_embeddings, latents, reconstructed_embeddings = self.forward(
            input_embeddings
        )
        reconstruction_loss = self.reconstruction_loss_function(
            reconstructed_embeddings, normalized_embeddings
        )
        return latents, reconstructed_embeddings, reconstruction_loss

    def training_step(self, batch: Tuple[ItemData]) -> torch.Tensor:
        """Perform a single training step on a batch of data."""
        model_input: ItemData = batch[0]
        latents, reconstructed_embeddings, reconstruction_loss = self.model_step(
            model_input
        )

        loss = self.reconstruction_loss_weight * reconstruction_loss
        self.train_loss(loss)
        self.train_reconstruction_loss(reconstruction_loss)

        train_dict_to_log = {
            "train/loss": self.train_loss,
            "train/reconstruction_loss": self.train_reconstruction_loss,
        }

        with torch.no_grad():
            if self.verbose and self.global_step % self.trainer.log_every_n_steps == 0:
                normalized_embeddings = self.normalization_layer(
                    model_input.transformed_features["input_embedding"].to(self.device)
                )
                mse = torch.mean(
                    (reconstructed_embeddings - normalized_embeddings) ** 2
                )
                self.train_mse(mse)
                train_dict_to_log["train/mse"] = self.train_mse

        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train_loss.reset()
        self.train_reconstruction_loss.reset()
        if self.verbose:
            self.train_mse.reset()

    def eval_step(
        self,
        batch: ItemData,
        loss_to_aggregate: MeanMetric,
        mse_metric: MeanMetric,
    ) -> None:
        """Shared evaluation step used for both validation and test."""
        _, reconstructed_embeddings, reconstruction_loss = self.model_step(batch)
        loss = self.reconstruction_loss_weight * reconstruction_loss
        loss_to_aggregate(loss)

        normalized_embeddings = self.normalization_layer(
            batch.transformed_features["input_embedding"].to(self.device)
        )
        mse = torch.mean((reconstructed_embeddings - normalized_embeddings) ** 2)
        mse_metric(mse)

    def validation_step(self, batch: ItemData, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data."""
        self.eval_step(batch, self.val_loss, self.val_mse)
        self.log_dict(
            {
                "val/loss": self.val_loss,
                "val/mse": self.val_mse,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_start(self) -> None:
        """Lightning hook that is called when validation begins."""
        self.val_loss.reset()
        self.val_mse.reset()

    def test_step(self, batch: ItemData, batch_idx: int) -> None:
        """Perform a single test step on a batch of data."""
        self.eval_step(batch, self.test_loss, self.test_mse)
        self.log_dict(
            {
                "test/loss": self.test_loss,
                "test/mse": self.test_mse,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_loss.reset()
        self.test_mse.reset()

    def predict_step(self, batch: ItemData) -> OneKeyPerPredictionOutput:
        """Return latent representations for each item in ``batch``.

        Mirrors the ``predict_step`` signature of ``ResidualQuantization`` so the
        same prediction writer (e.g. ``local_pickle_writer``) can be used.
        """
        latents, _, _ = self.model_step(batch)

        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]

        return OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=latents,
            key_name="item_id",
            prediction_name="latent",
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and (optionally) the LR scheduler."""
        if self.optimizer is None:
            return {}

        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def log_if_true(self, message: str, condition: bool) -> None:
        """Log a message if ``condition`` is True."""
        if condition:
            logging.info(f"Device {self.device}: {message}")
