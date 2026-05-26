"""Normalização de expressão: z-score por probe e correção de batch via ComBat."""

import logging
import warnings

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

log = logging.getLogger("geo_pipeline")

# Probes com taxa de NaN acima deste limiar são removidas antes do ComBat.
_MAX_NAN_RATE_COMBAT = 0.20


def zscore_by_probe(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normaliza cada probe (linha) ao longo das amostras.

    Usa ddof=1 (desvio padrão amostral), que é o estimador correto para
    dados biológicos com N finito.
    """
    result = expr_df.copy()
    gsm_cols = [c for c in result.columns if c.startswith("GSM")]

    data = result[gsm_cols].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(data, axis=1, keepdims=True)
        stds = np.nanstd(data, axis=1, keepdims=True, ddof=1)  # ddof=1: estimador amostral
        stds[stds == 0] = 1.0
        z_data = (data - means) / stds

    result[gsm_cols] = z_data
    log.info(f"Z-score (ddof=1): {data.shape[0]} probes × {len(gsm_cols)} amostras")
    return result


def apply_combat(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
) -> pd.DataFrame:
    """Aplica correção de batch ComBat via neuroCombat.

    Melhorias em relação à versão anterior:
    - Probes com >20% de valores ausentes são removidas antes do ComBat
      (ao invés de imputação por média da linha, que viola pressupostos do modelo).
    - Os valores ausentes restantes são imputados pela mediana da probe
      (mais robusta que a média para dados assimétricos).
    """
    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    annot_ids = set(sample_annot["sample_id"])
    common_samples = [s for s in gsm_cols if s in annot_ids]

    annot_aligned = (
        sample_annot.set_index("sample_id").loc[common_samples].reset_index()
    )
    expr_matrix = expr_df.set_index("Probe_ID")[common_samples].values.astype(float)

    # --- Passo 1: remover probes inteiramente ausentes ---
    valid_mask = ~np.all(np.isnan(expr_matrix), axis=1)
    expr_clean = expr_matrix[valid_mask].copy()
    probe_ids = expr_df["Probe_ID"].values[valid_mask]

    # --- Passo 2: remover probes com taxa de NaN > limiar ---
    nan_rates = np.isnan(expr_clean).mean(axis=1)
    low_nan_mask = nan_rates <= _MAX_NAN_RATE_COMBAT
    n_removed = (~low_nan_mask).sum()
    if n_removed > 0:
        log.warning(
            f"ComBat: removendo {n_removed} probes com >{_MAX_NAN_RATE_COMBAT*100:.0f}% "
            f"de valores ausentes (podem introduzir viés no modelo linear de batch)."
        )
    expr_clean = expr_clean[low_nan_mask]
    probe_ids = probe_ids[low_nan_mask]

    # --- Passo 3: imputar NaN restantes pela mediana da probe ---
    for i in range(expr_clean.shape[0]):
        row = expr_clean[i]
        nan_mask = np.isnan(row)
        if nan_mask.any():
            median_val = np.nanmedian(row)
            row[nan_mask] = median_val if not np.isnan(median_val) else 0.0
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
