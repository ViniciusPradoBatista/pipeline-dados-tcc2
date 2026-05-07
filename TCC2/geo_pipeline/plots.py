"""Visualizações: PCA scatter antes/depois do ComBat por batch e por classe."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

log = logging.getLogger("geo_pipeline")


def _pca_scatter(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    color_col: str,
    title: str,
    save_path: Path,
) -> None:
    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    common = [s for s in gsm_cols if s in sample_annot["sample_id"].values]
    X = np.nan_to_num(expr_df.set_index("Probe_ID")[common].values.T, nan=0.0)
    pc = PCA(n_components=2, random_state=42).fit_transform(X)
    labels = sample_annot.set_index("sample_id").loc[common][color_col].values
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("husl", len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            pc[mask, 0],
            pc[mask, 1],
            label=label,
            alpha=0.7,
            s=50,
            color=palette[i % len(palette)],
        )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    expr_before: pd.DataFrame,
    expr_after: Optional[pd.DataFrame],
    sample_annot: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Gera scatter PCA por batch e por classe, antes e (se fornecido) depois do ComBat."""
    _pca_scatter(
        expr_before,
        sample_annot,
        "batch",
        "PCA - Before ComBat (Batch)",
        output_dir / "pca_before_batch.png",
    )
    _pca_scatter(
        expr_before,
        sample_annot,
        "class_label",
        "PCA - Before ComBat (Class)",
        output_dir / "pca_before_class.png",
    )
    if expr_after is not None:
        _pca_scatter(
            expr_after,
            sample_annot,
            "batch",
            "PCA - After ComBat (Batch)",
            output_dir / "pca_after_batch.png",
        )
        _pca_scatter(
            expr_after,
            sample_annot,
            "class_label",
            "PCA - After ComBat (Class)",
            output_dir / "pca_after_class.png",
        )
