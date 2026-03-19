"""Build a binary label matrix from raw train_terms.tsv.

Reconstructed from notebook logic and script imports.
Called by:
  - scripts/preprocess.py  (standalone)
  - scripts/train.py       (on-the-fly if matrix not present)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger("cafa5")


def build_label_matrix(
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read train_terms.tsv and build a binary label matrix.

    Steps:
      1. Load all (EntryID, term) pairs from train_terms.tsv.
      2. Select the top-N most frequent GO terms.
      3. Build a binary matrix  [n_proteins × num_labels].

    Returns:
        label_matrix : np.ndarray, shape (n_proteins, num_labels), dtype float32
        protein_ids  : np.ndarray of EntryID strings, shape (n_proteins,)
        term_names   : np.ndarray of GO-term strings,  shape (num_labels,)
    """
    data_dir = Path(config.data.get("data_dir", "data/cafa-5-protein-function-prediction"))
    labels_path = data_dir / "Train" / "train_terms.tsv"

    if not labels_path.exists():
        raise FileNotFoundError(f"train_terms.tsv not found at {labels_path}")

    logger.info("Loading labels from %s", labels_path)
    df = pd.read_csv(labels_path, sep="\t")  # columns: EntryID, term, aspect

    # ── Top-N GO terms ────────────────────────────────────────────────────────
    num_labels = config.num_labels
    top_terms: np.ndarray = (
        df.groupby("term")["EntryID"]
        .count()
        .sort_values(ascending=False)
        .head(num_labels)
        .index.values
    )
    logger.info("Selected top %d GO terms", len(top_terms))

    # ── Protein list (preserves order) ────────────────────────────────────────
    protein_ids: np.ndarray = df["EntryID"].unique()
    n_proteins = len(protein_ids)
    protein_index = {pid: i for i, pid in enumerate(protein_ids)}
    term_index = {term: j for j, term in enumerate(top_terms)}

    # ── Binary matrix ─────────────────────────────────────────────────────────
    label_matrix = np.zeros((n_proteins, num_labels), dtype=np.float32)

    df_filtered = df[df["term"].isin(term_index)]
    for _, row in df_filtered.iterrows():
        i = protein_index[row["EntryID"]]
        j = term_index[row["term"]]
        label_matrix[i, j] = 1.0

    sparsity = 1.0 - label_matrix.mean()
    logger.info(
        "Label matrix built: %s proteins × %s terms  (sparsity %.1f%%)",
        n_proteins,
        num_labels,
        sparsity * 100,
    )

    return label_matrix, protein_ids, top_terms


def save_label_matrix(
    config: Config,
    label_matrix: np.ndarray,
    protein_ids: np.ndarray,
    term_names: np.ndarray,
) -> Path:
    """Persist label matrix and metadata to disk.

    Saves three .npy files under:
        <output_dir>/label_matrix_top<num_labels>/
            label_matrix.npy   – float32 binary matrix
            protein_ids.npy    – EntryID strings
            term_names.npy     – GO-term strings

    Returns:
        Path to the output directory.
    """
    out_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "label_matrix.npy", label_matrix)
    np.save(out_dir / "protein_ids.npy", protein_ids)
    np.save(out_dir / "term_names.npy", term_names)

    logger.info("Label matrix saved → %s", out_dir)
    return out_dir
