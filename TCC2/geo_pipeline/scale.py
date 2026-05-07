"""Detecção de plataforma GEO e inferência da escala dos dados (log2/raw/RMA)."""

import logging
import re
from typing import Tuple

import numpy as np
import pandas as pd

from geo_pipeline.constants import KNOWN_PLATFORMS

log = logging.getLogger("geo_pipeline")


def detect_platform(meta_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detecta a plataforma do microarray a partir dos metadados.

    Returns:
        (platform_id, platform_name)
    """
    platform_cols = [
        c for c in meta_df.columns if any(k in c.lower() for k in ("platform_id",))
    ]

    platform_id = "Unknown"
    for col in platform_cols:
        vals = meta_df[col].astype(str).unique()
        vals = [v for v in vals if v and v not in ("", "nan")]
        if vals:
            platform_id = vals[0]
            break

    # Fallback: scan all columns for GPL pattern
    if platform_id == "Unknown":
        for col in meta_df.columns:
            for val in meta_df[col].astype(str):
                m = re.search(r"(GPL\d+)", str(val))
                if m:
                    platform_id = m.group(1)
                    break
            if platform_id != "Unknown":
                break

    platform_name = KNOWN_PLATFORMS.get(platform_id, platform_id)
    log.info(f"Platform: {platform_id} ({platform_name})")
    return platform_id, platform_name


def infer_scale(
    meta_df: pd.DataFrame,
    platform_id: str,
    expr_num: pd.DataFrame,
) -> str:
    """
    Infere a escala dos dados a partir de keywords de processing,
    regras de plataforma e distribuição dos valores.

    Returns one of:
        "already_log2"      – RMA ou log2 explícito
        "already_processed" – normalizado mas escala ambígua
        "needs_log2"        – intensidades raw que precisam de log2(x+1)
        "unknown"           – não foi possível determinar
    """

    processing_text = ""
    for col in meta_df.columns:
        if "data_processing" in col.lower():
            vals = meta_df[col].dropna().unique()
            processing_text = " ".join(str(v).lower() for v in vals)
            break

    has_log2 = any(k in processing_text for k in ("log2", "log 2", "log-transformed"))
    has_rma = (
        bool(re.search(r"\brma\b", processing_text))
        or bool(re.search(r"\bmas5\b", processing_text))
        or bool(re.search(r"\bgcrma\b", processing_text))
    )
    has_qnorm = "quantile" in processing_text
    has_norm = "normalized" in processing_text or "normalisation" in processing_text

    gsm_cols = [c for c in expr_num.columns if c.startswith("GSM")]
    if gsm_cols:
        flat = expr_num[gsm_cols].to_numpy().ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) > 0:
            v_min, v_med, v_max = (
                np.nanmin(flat),
                np.nanmedian(flat),
                np.nanmax(flat),
            )
        else:
            v_min = v_med = v_max = 0.0
    else:
        v_min = v_med = v_max = 0.0

    log.info(f"Value stats: min={v_min:.3f}, median={v_med:.3f}, max={v_max:.3f}")

    plat_name = KNOWN_PLATFORMS.get(platform_id, "").lower()
    is_affy = "affymetrix" in plat_name or "affy" in platform_id.lower()
    is_3dgene = "3d-gene" in plat_name or "toray" in plat_name

    if has_rma:
        log.info("Scale: RMA keyword found → already_log2")
        return "already_log2"

    if has_log2:
        log.info("Scale: log2 keyword found → already_log2")
        return "already_log2"

    if v_max < 25 and v_min >= -10 and v_med > 0:
        if has_qnorm or has_norm:
            log.info("Scale: low range + normalization keyword → already_log2")
            return "already_log2"
        if is_affy or is_3dgene:
            log.info("Scale: low range + known platform → already_log2")
            return "already_log2"
        log.info("Scale: low range → already_processed (verify manually)")
        return "already_processed"

    if v_max > 1000:
        log.info("Scale: high values → needs_log2")
        return "needs_log2"

    log.warning("Scale: could not determine → unknown")
    return "unknown"
