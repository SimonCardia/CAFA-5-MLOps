"""PyTorch Dataset for protein sequence embeddings.

Reconstructed from the original notebook (ProteinSequenceDataset class)
and adapted to work with the Config dataclass and the saved label matrix.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config

logger = logging.getLogger("cafa5")

# Embedding file naming conventions per source
_EMBED_FILES = {
    "esm2":     ("train_embeddings.npy", "train_ids.npy",
                 "test_embeddings.npy",  "test_ids.npy"),
    "protbert": ("train_embeddings.npy", "train_ids.npy",
                 "test_embeddings.npy",  "test_ids.npy"),
    "t5":       ("train_embeds.npy",     "train_ids.npy",
                 "test_embeds.npy",      "test_ids.npy"),
}


class ProteinSequenceDataset(Dataset):
    """Dataset that pairs protein embeddings with GO-term label vectors.

    Args:
        config:            Project configuration.
        datatype:          ``"train"`` or ``"test"``.
        label_matrix_dir:  Directory containing ``label_matrix.npy``,
                           ``protein_ids.npy``, ``term_names.npy``.
                           Required when ``datatype="train"``.
    """

    def __init__(
        self,
        config: Config,
        datatype: str = "train",
        label_matrix_dir: Path | None = None,
    ) -> None:
        super().__init__()

        if datatype not in ("train", "test"):
            raise ValueError(f"datatype must be 'train' or 'test', got '{datatype}'")

        self.datatype = datatype
        source = config.data.get("embeddings_source", "ESM2").lower()

        if source not in _EMBED_FILES:
            raise ValueError(f"Unknown embeddings_source '{source}'")

        embed_file, id_file, test_embed_file, test_id_file = _EMBED_FILES[source]
        embeds_dir = Path(config.data.get("embeddings_dir", "data/embeddings")) / source

        # ── Load embeddings ───────────────────────────────────────────────────
        if datatype == "train":
            embed_path = embeds_dir / embed_file
            id_path    = embeds_dir / id_file
        else:
            embed_path = embeds_dir / test_embed_file
            id_path    = embeds_dir / test_id_file

        logger.info("Loading %s embeddings from %s", datatype, embed_path)
        embeds = np.load(embed_path)          # (n_proteins, embed_dim)
        ids    = np.load(id_path, allow_pickle=True)

        self.df = pd.DataFrame({
            "EntryID": ids,
            "embed":   list(embeds),
        })

        # ── Load labels (train only) ──────────────────────────────────────────
        if datatype == "train":
            if label_matrix_dir is None:
                label_matrix_dir = (
                    config.output_dir / f"label_matrix_top{config.num_labels}"
                )
            label_matrix_dir = Path(label_matrix_dir)

            label_matrix = np.load(
                label_matrix_dir / "label_matrix.npy", allow_pickle=True
            )
            protein_ids = np.load(
                label_matrix_dir / "protein_ids.npy", allow_pickle=True
            )

            # Align embeddings with label matrix rows via EntryID
            label_df = pd.DataFrame({
                "EntryID":     protein_ids,
                "labels_vect": list(label_matrix),
            })
            self.df = self.df.merge(label_df, on="EntryID", how="inner")
            logger.info(
                "Dataset ready: %d proteins (train), %d labels",
                len(self.df),
                config.num_labels,
            )
        else:
            logger.info("Dataset ready: %d proteins (test)", len(self.df))

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        embed = torch.tensor(row["embed"], dtype=torch.float32)

        if self.datatype == "train":
            targets = torch.tensor(row["labels_vect"], dtype=torch.float32)
            return embed, targets

        # test: return embedding + protein ID for submission
        return embed, row["EntryID"]
