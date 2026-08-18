# Preference-Aligned Semantic IDs
for Generative Recommendation

The code in this repository is built on top of:

- [SplitLight](https://github.com/monkey0head/SplitLight) — dataset splitting and preprocessing
- [GRID](https://github.com/snap-research/GRID) — semantic ID generation and TIGER training/inference
- [sasrec-bert4rec-recsys23](https://github.com/antklen/sasrec-bert4rec-recsys23) — SASRec training

## Installation

This project is set up for **Python 3.10**

Install dependencies:

```bash
pip install -r requirements.txt
```

## Creating Splits

1. Download the zip archive containing the preprocessed **Sports** dataset from [Sports_upd.zip](https://drive.google.com/file/d/1iN2BpkpSPFBJsfMBH-xgkcd02uTkfMXe/view?usp=sharing). Extract it to `data/`.
2. Create the folders `data/Sports/grid` and `data/Sports/sasrec` (if you want to obtain SASRec or aligned embeddings using our method).
3. Set environment variables and run the dataset split:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export SEQ_SPLITS_DATA_PATH="$(pwd)/data"

python SplitLight/runs/split.py \
  dataset=sports \
  split_type=global_timesplit \
  split_params.quantile=0.9 \
  split_params.validation_type=last_train_item \
  split_params.target_type=last \
  split_params.remove_cold_items=True
```

1. When using other datasets, add your own config file similar to `SplitLight/runs/configs/dataset/sports.yaml`.

## Creating PCA Embeddings

1. Open the notebook `SplitLight/PCA_embed.ipynb` and run all cells.
2. When using the notebook on other datasets, edit the following variables at the top of the notebook:

```python
core_5_path = "../data/Sports/"
out_torch_path = "../data/Sports/grid"
PCA_DIM = 128
```

## Creating SASRec and Aligned Embeddings (Our Method)

1. Open the notebook `SplitLight/GTS_to_sasrec_and_grid_data.ipynb` and run all cells.
2. When using the notebook on other datasets, edit the following variables at the top of the notebook:

```python
split_folder = "../data/Sports/GTS-q09-val_last_train_item-target_last-no_cold_items/"
path_to_embeddings = "../data/Sports/embeddings.parquet"
out_folder = "../data/Sports/"
preproc_path = "../data/Sports/preprocessed.csv"
```

3a. **Creating SASRec Embeddings**

3a.1. Run the following command:

```bash
cd TrainSASRec/src
python run.py --config-name=SASRec_amazon_sports
cd ../..
```

3a.2. When using other datasets, edit `TrainSASRec/src/configs/SASRec_amazon_sports.yaml` (or create a new config) and update `--config-name` in the command above.

3a.3. Open the notebook `TrainSASRec/notebooks/embed_SASRec.ipynb`, set the checkpoint path and GPU index, then run all cells. Checkpoints are saved under `TrainSASRec/src/lightning_logs/version_<N>/checkpoints/`:

```python
cuda_vd = 3
CONFIG_PATH = ROOT / "src/configs/SASRec_amazon_sports.yaml"
CKPT_PATH = ROOT / "src/lightning_logs/version_1/checkpoints/epoch=3-step=1024.ckpt"
OUT_PATH = ROOT / "../data/Sports/grid"
OUT_PT_NAME = "merged_predictions_tensor_SASRec_item_emb.pt"
```

3b. **Creating Aligned Embeddings**

3b.1. Run the following command:

```bash
cd TrainSASRec/src
python run.py --config-name=ESASRecStageV2___amazon_sports
cd ../..
```

3b.2. When using other datasets, edit `TrainSASRec/src/configs/ESASRecStageV2___amazon_sports.yaml` (or create a new config) and update `--config-name` in the command above.

3b.3. Open the notebook `TrainSASRec/notebooks/embed_Our_method.ipynb`, set the checkpoint path and GPU index, then run all cells. Checkpoints are saved under `TrainSASRec/src/lightning_logs/version_<N>/checkpoints/`:

```python
cuda_vd = 3
CONFIG_PATH = ROOT / "src/configs/ESASRecStageV2___amazon_sports.yaml"
CKPT_PATH = ROOT / "src/lightning_logs/version_2/checkpoints/epoch=80-step=20736.ckpt"
OUT_PATH = ROOT / "../data/Sports/grid"
OUT_PT_NAME = "new_embeddings_amazon_toys_after_adapter.pt"
```

## Description of Obtained Data

After completing all steps above, the expected structure in the `data/` folder is:

```
data/
|--- Sports/
|    |--- embeddings.parquet
|    |--- item_metadata.parquet
|    |--- preprocessed.csv
|    |--- pca_embeddings.parquet
|    |
|    |--- GTS-q09-val_last_train_item-target_last-no_cold_items/
|    |    |--- train.csv
|    |    |--- validation_input.csv
|    |    |--- validation_target.csv
|    |    |--- test_input.csv
|    |    |--- test_target.csv
|    |
|    |--- sasrec/
|    |    |--- sasrec_inter.txt
|    |    |--- embeddings_dict.pkl
|    |
|    |--- grid/
|         |--- pca_embed.pt                                    # embeddings after PCA
|         |--- raw_item_embed_tensor.pt                        # raw content embeddings
|         |--- merged_predictions_tensor_SASRec_item_emb.pt     # embeddings after SASRec
|         |--- new_embeddings_amazon_toys_after_adapter.pt     # embeddings after our method (aligned)
|         |
|         |--- data/
|              |--- training/
|              |    |--- partition_*.tfrecord.gz
|              |--- evaluation/
|              |    |--- partition_*.tfrecord.gz
|              |--- testing/
|              |    |--- partition_*.tfrecord.gz
|              |--- items/
|                   |--- data_*.tfrecord.gz
```

## Generating SIDs

All configs referenced below should be adapted for your own datasets and embedding types (PCA, raw, SASRec, aligned).

Go to the `Grid` folder:

```bash
cd Grid
```

Run training with:

```bash
python -m src.train experiment=rkmeans_train_amazon_sports_PCA
```

Open `Grid/configs/experiment/rkmeans_inference_flat_PCA_sports.yaml` and set the checkpoint path in `ckpt_path`:

```yaml
ckpt_path: ../Grid/logs/train/runs/2026-06-07/00-21-40-SIDs_train_PCA_emb_sports/checkpoints/checkpoint_000_000010.ckpt
seed: 42
id: ${now:%Y-%m-%d}/${now:%H-%M-%S}-SIDs_inference_PCA_emb_sports
tags:
```

Checkpoints are saved under `Grid/logs/train/runs/<date>/<run_id>/checkpoints/`.

Run inference with:

```bash
python -m src.inference experiment=rkmeans_inference_flat_PCA_sports
```

If you encounter an error, try running with:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python -m src.inference experiment=rkmeans_inference_flat_PCA_sports
```

## Training TIGER

Open `Grid/configs/experiment/tiger_train_flat_PCA_sports.yaml` and set the path to semantic IDs in `semantic_id_path`:

```yaml
semantic_id_path: ../Grid/logs/inference/runs/2026-06-07/00-38-11-SIDs_inference_PCA_emb_sports/pickle/merged_predictions_tensor.pt
```

Semantic IDs are saved under `Grid/logs/inference/runs/<date>/<run_id>/pickle/merged_predictions_tensor.pt`.

Run training with:

```bash
python -m src.train experiment=tiger_train_flat_PCA_sports
```

## TIGER Inference

Open `Grid/configs/experiment/tiger_inference_flat_PCA_sports.yaml` and set the path to semantic IDs in `semantic_id_path` and the checkpoint path in `ckpt_path`:

```yaml
semantic_id_path: ../Grid/logs/inference/runs/2026-06-07/00-38-11-SIDs_inference_PCA_emb_sports/pickle/merged_predictions_tensor.pt
ckpt_path: ../Grid/logs/train/runs/2026-06-07/00-53-29-TIGER_train_PCA_emb_sports/checkpoints/checkpoint_epoch=000_step=000001.ckpt
```

Checkpoints are saved under `Grid/logs/train/runs/<date>/<run_id>/checkpoints/`.

Run inference with:

```bash
python -m src.inference experiment=tiger_inference_flat_PCA_sports
```

If you encounter an error, try running with:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python -m src.inference experiment=tiger_inference_flat_PCA_sports
```

