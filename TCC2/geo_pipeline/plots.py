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

_N_PCA_COMPONENTS = 4


def _impute_median_cols(X: np.ndarray) -> np.ndarray:
    """Imputa NaN pela mediana de cada coluna. Não usa 0."""
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        col = X_out[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any() and not np.all(nan_mask):
            col[nan_mask] = np.nanmedian(col)
        elif np.all(nan_mask):
            col[:] = 0.0
        X_out[:, j] = col
    return X_out


def _pca_scatter(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    color_col: str,
    title: str,
    save_path: Path,
    pc_x: int = 0,
    pc_y: int = 1,
) -> None:
    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    common = [s for s in gsm_cols if s in sample_annot["sample_id"].values]
    X_raw = expr_df.set_index("Probe_ID")[common].values.T.astype(float)
    X = _impute_median_cols(X_raw)

    n_components = min(_N_PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
    if n_components < max(pc_x, pc_y) + 1:
        log.warning(f"Não há componentes suficientes para PC{pc_x+1} vs PC{pc_y+1}. Pulando.")
        return

    pcs = PCA(n_components=n_components, random_state=42).fit_transform(X)
    labels = sample_annot.set_index("sample_id").loc[common][color_col].values
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("husl", len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            pcs[mask, pc_x],
            pcs[mask, pc_y],
            label=label,
            alpha=0.7,
            s=50,
            color=palette[i % len(palette)],
        )
    ax.set_xlabel(f"PC{pc_x + 1}")
    ax.set_ylabel(f"PC{pc_y + 1}")
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
    """Gera scatter PCA (PC1-PC2 e PC3-PC4) por batch e por classe, antes e depois do ComBat."""
    for color_col, label_tag in (("batch", "Batch"), ("class_label", "Class")):
        _pca_scatter(
            expr_before, sample_annot, color_col,
            f"PCA PC1-PC2 - Before ComBat ({label_tag})",
            output_dir / f"pca_before_{color_col}_pc12.png",
            pc_x=0, pc_y=1,
        )
        _pca_scatter(
            expr_before, sample_annot, color_col,
            f"PCA PC3-PC4 - Before ComBat ({label_tag})",
            output_dir / f"pca_before_{color_col}_pc34.png",
            pc_x=2, pc_y=3,
        )

        if expr_after is not None:
            _pca_scatter(
                expr_after, sample_annot, color_col,
                f"PCA PC1-PC2 - After ComBat ({label_tag})",
                output_dir / f"pca_after_{color_col}_pc12.png",
                pc_x=0, pc_y=1,
            )
            _pca_scatter(
                expr_after, sample_annot, color_col,
                f"PCA PC3-PC4 - After ComBat ({label_tag})",
                output_dir / f"pca_after_{color_col}_pc34.png",
                pc_x=2, pc_y=3,
            )
