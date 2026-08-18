from typing import Any, Optional
from torchmetrics.metric import Metric

import torch
from torchmetrics.utilities.distributed import gather_all_tensors

from src.components.eval_metrics import CustomRetrievalMetric, SIDRetrievalEvaluator


class Coverage(CustomRetrievalMetric):
    """Coverage@K = |unique items appearing in top-K predictions across the whole eval set| / num_total_items.

    `num_total_items` must be passed explicitly: it cannot be inferred from `preds`,
    because when negatives are sampled per query `preds.shape[1] == num_negatives + 1`,
    not the real catalog size.

    `update` accepts an optional `candidate_ids` via kwargs:
      - shape `(batch_size, num_candidates)` — flat catalog ids;
      - shape `(batch_size, num_candidates, num_hierarchies)` — hierarchical SIDs.
    When provided, real item ids of the top-K predictions are accumulated. Otherwise
    we fall back to raw top-K indices from `preds`, which is meaningful only when
    those indices already represent real catalog item ids (e.g. full-vocab scoring
    without per-query negative sampling). For TIGER/SID with beam search you MUST
    pass `candidate_ids`.

    Uniqueness is computed via `torch.unique(..., dim=0)` over a 2D state tensor
    `(N, H)`, so hierarchical SIDs are compared as full tuples and no packing /
    `codebook_size` is required.
    """

    _warned_about_local_indices: bool = False
    _warned_about_denominator: bool = False

    def __init__(
        self,
        num_total_items: int,
        valid_sids_path: Optional[str] = None,
        use_cfg_total_items = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_total_items <= 0:
            raise ValueError(f"num_total_items must be > 0, got {num_total_items}")
        self.num_total_items = int(num_total_items)
        self.add_state("candidates", default=[], dist_reduce_fx="cat")
        self.use_cfg_total_items = use_cfg_total_items
        # Optional catalog of *valid* SIDs. When provided, `compute` keeps only the
        # generated SID tuples that actually exist in the catalog, so hallucinated
        # beam-search outputs cannot inflate the numerator (or push coverage > 1).
        # The file is expected to be an int tensor of shape `(H, num_items)` (same
        # orientation as `model.codebooks`) or `(num_items, H)`; the orientation is
        # resolved lazily in `compute` against the runtime hierarchy size.
        valid_sids: Optional[torch.Tensor] = None
        if valid_sids_path is not None:
            loaded = torch.load(valid_sids_path, map_location="cpu")
            if not isinstance(loaded, torch.Tensor):
                raise TypeError(
                    f"valid_sids_path must contain a torch.Tensor, got {type(loaded)}"
                )
            if loaded.dim() != 2:
                raise ValueError(
                    f"valid_sids must be a 2D tensor, got shape {tuple(loaded.shape)}"
                )
            valid_sids = loaded.detach().to(torch.long)
        # `register_buffer` so the table follows the metric across `.to(device)`.
        self.register_buffer("valid_sids", valid_sids, persistent=False)

    def _oriented_valid_sids(self, num_hierarchies: int) -> torch.Tensor:
        """Return the valid-SID catalog as `(M, H)`, transposing if it was stored
        as `(H, M)`. Orientation is inferred from the runtime hierarchy size."""
        vs = self.valid_sids
        if vs.shape[-1] == num_hierarchies:
            oriented = vs
        elif vs.shape[0] == num_hierarchies:
            oriented = vs.t()
        else:
            raise ValueError(
                f"valid_sids shape {tuple(vs.shape)} is incompatible with "
                f"num_hierarchies={num_hierarchies}: neither dim matches."
            )
        return oriented.unique(dim=0)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        num_candidates = preds.shape[1]

        k = min(self.top_k, num_candidates)
        topk_idx = torch.topk(preds, k=k, dim=1).indices  # (batch_size, k)

        if candidate_ids is not None:
            ids = candidate_ids
            if ids.dim() == 2:
                ids = ids.unsqueeze(-1)  # (B, N, 1) — unify with hierarchical case
            elif ids.dim() != 3:
                raise ValueError(
                    f"candidate_ids must be 2D or 3D, got shape {tuple(ids.shape)}"
                )
            num_hierarchies = ids.shape[-1]
            expanded_idx = topk_idx.unsqueeze(-1).expand(-1, -1, num_hierarchies)
            predicted = torch.gather(ids, dim=1, index=expanded_idx)  # (B, k, H)
        else:
            if not Coverage._warned_about_local_indices:
                print(
                    "[Coverage] WARNING: `candidate_ids` was not provided. Falling "
                    "back to raw top-K indices from `preds`. This is only correct "
                    "when those indices are real catalog item ids (full-vocab "
                    "scoring). For TIGER/SID/beam-search this metric will be "
                    "meaningless until the evaluator forwards real ids."
                )
                Coverage._warned_about_local_indices = True
            predicted = topk_idx.unsqueeze(-1)  # (B, k, 1)

        flat = predicted.reshape(-1, predicted.shape[-1]).detach().to(torch.long)
        self.candidates.append(flat)

    def compute(self) -> torch.Tensor:
        if not self.candidates:
            return torch.tensor(-1.0, device=self.device)

        candidates = torch.cat(self.candidates, dim=0)  # (N, H)

        # Mirrors CustomMeanReductionMetric: metrics here are created with
        # sync_on_compute=False, so torchmetrics will not auto-gather. Do it manually.
        # `valid_sids` is identical on every rank, so we gather only `candidates`
        # and filter afterwards.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = gather_all_tensors(candidates)
            candidates = torch.cat(gathered, dim=0)

        unique_predicted = candidates.unique(dim=0)  # (U, H)

        if self.valid_sids is not None:
            num_hierarchies = unique_predicted.shape[-1]
            valid = self._oriented_valid_sids(num_hierarchies).to(unique_predicted.device)
            # Count-trick membership test (no packing into a single int, so it is
            # safe for arbitrary hierarchy depth / codebook size): a predicted row
            # that also exists in the catalog appears twice in the concatenation.
            both = torch.cat([valid, unique_predicted], dim=0)
            _, counts = both.unique(dim=0, return_counts=True)
            num_unique = int((counts >= 2).sum().item())
            denominator = valid.shape[0]
            if (
                denominator != self.num_total_items
                and not Coverage._warned_about_denominator
                and not self.use_cfg_total_items
            ):
                print(
                    f"[Coverage] WARNING: number of unique valid SIDs "
                    f"({denominator}) != num_total_items ({self.num_total_items}). "
                    f"Using the unique-valid-SID count as the denominator (likely "
                    f"SID collisions in the catalog)."
                )
                Coverage._warned_about_denominator = True
        else:
            num_unique = unique_predicted.shape[0]
            denominator = self.num_total_items

        if self.use_cfg_total_items:
            denominator = self.num_total_items

        return torch.tensor(
            num_unique / float(denominator),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self) -> None:
        super().reset()
        self.candidates = []


class SIDRetrievalEvaluatorWithCoverage(SIDRetrievalEvaluator):
    """Same as `SIDRetrievalEvaluator`, but also forwards `candidate_ids=generated_ids`
    to every metric's `update(...)`.

    `Coverage` requires real beam item ids (full hierarchical SIDs) to compute the
    fraction of unique catalog items the model surfaces; without them it can only
    see local beam indices `0..top_k-1` and produces a meaningless near-constant.
    Other metrics (`NDCG`, `Recall`, ...) accept `**kwargs` in their `update`
    signature and silently ignore the extra tensor, so this evaluator is a safe
    drop-in replacement for `SIDRetrievalEvaluator` in TIGER/SID configs.
    """

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        reshaped_labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        matched_id_coord = torch.all((generated_ids == reshaped_labels), dim=2).nonzero()

        target = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)

        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        device = preds.device
        candidate_ids = generated_ids.to(device)
        target = target.to(device)
        expanded_indexes = expanded_indexes.to(device)

        for metric_object in self.metrics.values():
            metric_object.update(
                preds,
                target,
                indexes=expanded_indexes,
                candidate_ids=candidate_ids,
            )
