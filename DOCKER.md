# Docker — Schnellstart

## Voraussetzungen
- Docker >= 24
- Docker Compose >= 2.20
- Daten liegen unter `data/` (siehe Projektstruktur)

## Verzeichnisstruktur erwartet

```
data/
├── cafa-5-protein-function-prediction/
│   └── Train/
│       └── train_terms.tsv
└── embeddings/
    └── esm2/
        ├── train_embeddings.npy
        ├── train_ids.npy
        ├── test_embeddings.npy
        └── test_ids.npy
```

## Komplette Pipeline ausführen

```bash
# 1. Image bauen
docker compose build

# 2. Alle drei Schritte der Reihe nach ausführen
docker compose up preprocess   # → outputs/label_matrix_top500/
docker compose up train        # → outputs/checkpoints/best_model.pt
docker compose up predict      # → outputs/submission.tsv
```

Oder alles auf einmal (Reihenfolge durch depends_on gesichert):
```bash
docker compose up
```

## Einzelne Schritte neu starten

```bash
# Nur Inferenz neu laufen lassen (Modell bereits trainiert)
docker compose up predict

# Training mit anderem Modell (config.yaml anpassen, dann:)
docker compose up train predict
```

## Logs ansehen

```bash
docker compose logs -f train
```

## GPU aktivieren (optional)

In `Dockerfile` die erste Zeile ersetzen:
```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
```

In `docker-compose.yml` unter `train:` ergänzen:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```
