# CAFA-5 Protein Function Prediction — MLOps Pipeline

A production-ready MLOps pipeline for the [Kaggle CAFA-5 competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction), predicting Gene Ontology (GO) terms from protein language model embeddings (ESM-2, ProtBERT, T5).

## Project Objectives

| Objective | Metric | Target |
|---|---|---|
| Predict GO terms per protein | F1-Score (micro) | > 0.10 baseline |
| Multi-label classification | Val F1 | Best checkpoint saved |
| Reproducible pipeline | Seeded splits | seed=42 |

## Architecture

```
data/embeddings/          data/cafa-5-.../Train/
       │                          │
       ▼                          ▼
  preprocess ──────────► label_matrix_top500/
       │
       ▼
    train ──────────────► outputs/checkpoints/best_model.pt
       │
       ▼
   predict ─────────────► outputs/submission.tsv
       │
       ▼
  inference API ──────────► http://localhost:8000
```

## Project Structure

```
CAFA-5-MLOps/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── docker/
│   ├── Dockerfile.amd           # AMD ROCm (RX 9060 XT, gfx1200)
│   ├── Dockerfile.nvidia        # NVIDIA CUDA
│   ├── Dockerfile.cpu           # CPU-only fallback
│   └── Dockerfile.api           # Lightweight inference API
├── scripts/
│   ├── preprocess.py            # Build binary label matrix
│   ├── train.py                 # Train model
│   └── predict.py               # Generate submission.tsv
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI inference API
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── preprocessing.py    # Label matrix builder
│   ├── inference/
│   │   └── predictor.py        # Load checkpoint + predict
│   ├── models/
│   │   ├── mlp.py               # MultiLayerPerceptron
│   │   └── cnn1d.py             # 1D Convolutional Network
│   ├── training/
│   │   └── trainer.py           # Training loop + checkpointing
│   ├── config.py                # YAML config loader
│   └── utils.py                 # Seed, device, logger
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### Prerequisites
- Docker + Docker Compose
- AMD GPU (ROCm) / NVIDIA GPU (CUDA) / CPU

### 1. Clone and prepare data

```bash
git clone https://github.com/SimonCardia/CAFA-5-MLOps.git
cd CAFA-5-MLOps
mkdir -p data/embeddings/esm2 data/cafa-5-protein-function-prediction outputs

# Download from Kaggle
kaggle datasets download viktorfairuschin/cafa-5-ems-2-embeddings-numpy \
  --unzip -p data/embeddings/esm2/
kaggle datasets download siddhvr/train-targets-top500 \
  --unzip -p outputs/label_matrix_top500/
# Place train_terms.tsv in data/cafa-5-protein-function-prediction/Train/
```

### 2. Build and run

```bash
# AMD GPU
docker compose --profile amd build
docker compose --profile amd up

# NVIDIA GPU
docker compose --profile nvidia build
docker compose --profile nvidia up

# CPU only
docker compose --profile cpu build
docker compose --profile cpu up
```

### 3. Start inference API

```bash
docker compose --profile amd build api
docker compose --profile amd up api
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Models

| Model | Params | Description |
|---|---|---|
| `cnn1d` | 2.6M | Two 1D conv layers + max pooling + FC |
| `mlp` | ~1.2M | Three fully-connected layers with ReLU |

Both output raw logits — `BCEWithLogitsLoss` applies sigmoid internally.

## Results (Baseline)

| Epoch | Train F1 | Val F1 |
|---|---|---|
| 1 | 0.019 | 0.047 |
| 2 | 0.069 | 0.079 |
| 3 | 0.097 | 0.108 |
| 4 | 0.117 | 0.117 |
| 5 | 0.137 | **0.129** |

Model: CNN1D, Embeddings: ESM2, 5 epochs, AMD RX 9060 XT

## Configuration

All parameters in `configs/config.yaml`:

```yaml
data:
  embeddings_source: "ESM2"   # ESM2 | ProtBERT | T5
  num_labels: 500
  train_val_split: 0.9
model:
  type: "cnn1d"               # mlp | cnn1d
training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001
  seed: 42
```

## License

MIT
