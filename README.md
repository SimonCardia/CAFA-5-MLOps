# CAFA-5 Protein Function Prediction — MLOps Serving System

## Overview

This project implements a bioinformatics inference system for protein function prediction (CAFA-5 context), built with a clean MLOps architecture.

The system separates:
- Offline pipeline (embedding, training, evaluation)
- Online serving (API-based inference)

Focus:
- reproducibility
- modularity
- deployability

---

## Architecture

### Serving Stack (MVP)

- cafa-api (FastAPI)
  - validates input embeddings
  - loads trained model
  - performs inference
  - logs metadata to MLflow

- mlflow
  - tracks inference metadata
  - enables experiment comparison

- (next step) nginx
  - reverse proxy
  - authentication layer

---

### Offline Pipeline

Executed locally / on GPU:

- scripts/preprocess.py
- scripts/split_train_holdout.py
- scripts/embed_sequences.py
- scripts/train.py
- scripts/evaluate_holdout.py

GPU (ROCm) is used for:
- embedding generation
- training

---

## Project Structure

src/                 core ML logic  
services/            API + service logic  
scripts/             pipeline scripts  
configs/             configuration  
data/                raw + processed data  
outputs/             experimental artifacts  
models/              serving artifacts (stable)  
docs/architecture/   decisions + tasks  

---

## Artifact Strategy

outputs/ = experimental artifacts (training pipeline)  
models/ = production-ready artifacts (serving)

The serving API reads **only from models/**.

Required files:

models/
- best_model.pt
- term_names.npy
- model_meta.json

---

## Running the System

### Build

docker compose -f compose.api.yml build

### Start

docker compose -f compose.api.yml up

---

## API Endpoints

### Health

GET /health

Example response:

{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1"
}

---

### Prediction

POST /predict

Request:

{
  "embedding": [...1280 values...],
  "top_k": 5
}

Response:

{
  "model_version": "v1",
  "top_k": 5,
  "predictions": [
    {"go_term": "...", "score": 0.81}
  ]
}

---

## MLflow

MLflow runs as a separate service:

http://localhost:5000

Logged metadata:
- model_version
- top_k
- runtime_ms
- prediction_count

Logging is best effort and does not affect API responses.

---

## Current Status

Completed:
- FastAPI inference service
- Dockerized API
- MLflow integration
- Microservice architecture (API + MLflow)

In progress:
- DVC-based model versioning
- Nginx + authentication
- production hardening

---

## Key Design Decisions

- training is separated from serving
- models are NOT stored in Docker images
- MLflow is NOT part of the request path
- API is stateless and lightweight
- GPU is only used in offline pipeline

---

## Tech Stack

- Python
- PyTorch (ROCm)
- FastAPI
- Docker / Docker Compose
- MLflow

---

## Next Steps

- add Nginx reverse proxy
- implement Basic Auth
- integrate DVC artifact flow
- add monitoring

---

## Notes

This project focuses on building a clean, minimal, and production-oriented bioinformatics service.
