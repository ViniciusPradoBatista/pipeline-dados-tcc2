"""Métricas de validação cross-platform: PurityB (batch) e PurityD (disease)."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger("geo_pipeline")


def calculate_purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Calcula pureza de cluster (fração da maioria dominante somada ao total)."""
    n = len(cluster_labels)
    if n == 0:
        return 0.0
    cluster_ids = np.unique(cluster_labels)
    true_ids = np.unique(true_labels)
    total = 0
    for k in cluster_ids:
        cluster_mask = cluster_labels == k
        max_overlap = 0
        for j in true_ids:
            class_mask = true_labels == j
            overlap = int(np.sum(cluster_mask & class_mask))
            if overlap > max_overlap:
                max_overlap = overlap
        total += max_overlap
    return total / n


def compute_purity_metrics(
    expr_before: pd.DataFrame,
    expr_after: Optional[pd.DataFrame],
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
) -> pd.DataFrame:
    """Calcula PurityB (batch) e PurityD (disease) antes e depois do ComBat."""
    gsm_cols = [c for c in expr_before.columns if c.startswith("GSM")]
    common = [s for s in gsm_cols if s in sample_annot["sample_id"].values]
    annot = sample_annot.set_index("sample_id").loc[common]

    batch_enc = LabelEncoder().fit_transform(annot[batch_col].values)
    class_enc = LabelEncoder().fit_transform(annot[class_col].values)
    n_batches, n_classes = len(np.unique(batch_enc)), len(np.unique(class_enc))

    results = {}
    X_before = np.nan_to_num(
        expr_before.set_index("Probe_ID")[common].values.T, nan=0.0
    )
    results["PurityB_before"] = calculate_purity(
        KMeans(n_clusters=n_batches, random_state=42, n_init=10).fit_predict(X_before),
        batch_enc,
    )
    results["PurityD_before"] = calculate_purity(
        KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit_predict(X_before),
        class_enc,
    )

    if expr_after is not None:
        X_after = np.nan_to_num(
            expr_after.set_index("Probe_ID")[common].values.T, nan=0.0
        )
        results["PurityB_after"] = calculate_purity(
            KMeans(n_clusters=n_batches, random_state=42, n_init=10).fit_predict(
                X_after
            ),
            batch_enc,
        )
        results["PurityD_after"] = calculate_purity(
            KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit_predict(
                X_after
            ),
            class_enc,
        )

    return pd.DataFrame([results])
