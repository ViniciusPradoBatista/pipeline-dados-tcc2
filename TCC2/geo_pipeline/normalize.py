"""Normalização de expressão: z-score por probe e correção de batch via ComBat."""

import logging
import warnings

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

log = logging.getLogger("geo_pipeline")


def zscore_by_probe(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normaliza cada probe (linha) ao longo das amostras de um dataset."""
    result = expr_df.copy()
    gsm_cols = [c for c in result.columns if c.startswith("GSM")]

    data = result[gsm_cols].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(data, axis=1, keepdims=True)
        stds = np.nanstd(data, axis=1, keepdims=True, ddof=0)
        stds[stds == 0] = 1.0
        z_data = (data - means) / stds

    result[gsm_cols] = z_data
    log.info(f"Z-score: {data.shape[0]} probes × {len(gsm_cols)} samples")
    return result


def apply_combat(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
) -> pd.DataFrame:
    """Aplica correção de batch ComBat via neuroCombat."""
    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    annot_ids = set(sample_annot["sample_id"])
    common_samples = [s for s in gsm_cols if s in annot_ids]

    annot_aligned = (
        sample_annot.set_index("sample_id").loc[common_samples].reset_index()
    )
    expr_matrix = expr_df.set_index("Probe_ID")[common_samples].values.astype(float)

    valid_mask = ~np.all(np.isnan(expr_matrix), axis=1)
    expr_clean = expr_matrix[valid_mask].copy()
    probe_ids = expr_df["Probe_ID"].values[valid_mask]

    for i in range(expr_clean.shape[0]):
        row = expr_clean[i]
        if np.isnan(row).any():
            row[np.isnan(row)] = np.nanmean(row)
            expr_clean[i] = row

    covars = pd.DataFrame(
        {
            batch_col: annot_aligned[batch_col].values,
            class_col: annot_aligned[class_col].values,
        }
    )

    result = neuroCombat(
        dat=expr_clean,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=[class_col],
    )
    corrected = result["data"]

    out_df = pd.DataFrame(corrected, columns=common_samples)
    out_df.insert(0, "Probe_ID", probe_ids)
    return out_df
