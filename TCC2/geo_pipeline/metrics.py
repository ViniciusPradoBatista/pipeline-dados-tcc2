"""Métricas de validação cross-platform: PurityB/D e Silhouette antes/depois do ComBat."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger("geo_pipeline")

_PCA_COMPONENTS = 50


def _impute_median(X: np.ndarray) -> np.ndarray:
    """Imputa NaN pela mediana de cada coluna (feature/probe). Não usa 0."""
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
        max_overlap = max(
            int(np.sum(cluster_mask & (true_labels == j))) for j in true_ids
        )
        total += max_overlap
    return total / n


def _reduce_and_cluster(X_raw: np.ndarray, n_clusters: int) -> np.ndarray:
    """Reduz dimensionalidade com PCA e aplica KMeans no espaço reduzido."""
    n_components = min(_PCA_COMPONENTS, X_raw.shape[0] - 1, X_raw.shape[1])
    if n_components < 1:
        n_components = 1
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X_raw)
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_pca), X_pca


def compute_purity_metrics(
    expr_before: pd.DataFrame,
    expr_after: Optional[pd.DataFrame],
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
) -> pd.DataFrame:
    """
    Calcula PurityB (batch) e PurityD (disease) antes e depois do ComBat,
    usando PCA para redução dimensional antes do KMeans.

    Também calcula Silhouette Score, que é menos enviesado pelo número de
    clusters e pelo desbalanceamento de classes do que a pureza.

    Interpretação:
    - PurityB_before alto → batches fortemente separados (ruim, indica batch effect).
    - PurityB_after baixo → ComBat misturou os batches (bom).
    - PurityD deve permanecer alto após ComBat (sinal biológico preservado).
    - SilhouetteB_after próximo de 0 ou negativo → batches bem misturados (bom).
    - SilhouetteD_after positivo → classes biologicamente separadas (bom).
    """
    gsm_cols = [c for c in expr_before.columns if c.startswith("GSM")]
    common = [s for s in gsm_cols if s in sample_annot["sample_id"].values]
    annot = sample_annot.set_index("sample_id").loc[common]

    batch_enc = LabelEncoder().fit_transform(annot[batch_col].values)
    class_enc = LabelEncoder().fit_transform(annot[class_col].values)
    n_batches = len(np.unique(batch_enc))
    n_classes = len(np.unique(class_enc))

    results = {}

    X_raw_before = expr_before.set_index("Probe_ID")[common].values.T.astype(float)
    X_before = _impute_median(X_raw_before)

    labels_b, X_before_pca = _reduce_and_cluster(X_before, n_batches)
    labels_d, _ = _reduce_and_cluster(X_before, n_classes)

    results["PurityB_before"] = calculate_purity(labels_b, batch_enc)
    results["PurityD_before"] = calculate_purity(labels_d, class_enc)

    try:
        results["SilhouetteB_before"] = (
            float(silhouette_score(X_before_pca, batch_enc)) if n_batches >= 2 else np.nan
        )
        results["SilhouetteD_before"] = (
            float(silhouette_score(X_before_pca, class_enc)) if n_classes >= 2 else np.nan
        )
    except Exception:
        results["SilhouetteB_before"] = np.nan
        results["SilhouetteD_before"] = np.nan

    if expr_after is not None:
        X_raw_after = expr_after.set_index("Probe_ID")[common].values.T.astype(float)
        X_after = _impute_median(X_raw_after)

        labels_b_a, X_after_pca = _reduce_and_cluster(X_after, n_batches)
        labels_d_a, _ = _reduce_and_cluster(X_after, n_classes)

        results["PurityB_after"] = calculate_purity(labels_b_a, batch_enc)
        results["PurityD_after"] = calculate_purity(labels_d_a, class_enc)

        try:
            results["SilhouetteB_after"] = (
                float(silhouette_score(X_after_pca, batch_enc)) if n_batches >= 2 else np.nan
            )
            results["SilhouetteD_after"] = (
                float(silhouette_score(X_after_pca, class_enc)) if n_classes >= 2 else np.nan
            )
        except Exception:
            results["SilhouetteB_after"] = np.nan
            results["SilhouetteD_after"] = np.nan

    return pd.DataFrame([results])
