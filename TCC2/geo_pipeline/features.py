"""Harmonização de Probe IDs cross-platform e construção de anotação de amostras."""

import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from geo_pipeline.conditions import extract_sample_condition, normalize_condition
from geo_pipeline.constants import HEALTHY_SYNONYMS_BROAD, HEALTHY_SYNONYMS_STRICT

log = logging.getLogger("geo_pipeline")


def canonicalize_probe_id(probe_id: str, platform_id: str) -> Tuple[str, bool]:
    """
    Canonicaliza um Probe ID seguindo regras específicas da plataforma.

    Args:
        probe_id:    string original do Probe_ID
        platform_id: accession GPL

    Returns:
        (canonical_id, is_ambiguous)
    """
    pid = probe_id.strip()

    affy_platforms = {"GPL19117", "GPL18402", "GPL16384", "GPL8786"}
    if platform_id in affy_platforms:
        canonical = re.sub(r"_st$", "", pid, flags=re.IGNORECASE)
        return canonical, False

    toray_platforms = {"GPL18941", "GPL21263"}
    if platform_id in toray_platforms:
        if "," in pid:
            # Múltiplos MIMATs → probe ambíguo: usa o primeiro e flag
            parts = [p.strip() for p in pid.split(",")]
            return parts[0], True
        return pid, False

    return pid, False


def build_feature_map(
    expr_df: pd.DataFrame,
    platform_id: str,
) -> pd.DataFrame:
    """
    Constrói mapeamento Probe → ID canônico com flag de ambiguidade.

    Returns DataFrame:
        Probe_ID  |  Probe_ID_Canonical  |  Probe_ID_Ambiguous
    """
    records = []
    for pid in expr_df["Probe_ID"].unique():
        canonical, ambiguous = canonicalize_probe_id(pid, platform_id)
        records.append(
            {
                "Probe_ID": pid,
                "Probe_ID_Canonical": canonical,
                "Probe_ID_Ambiguous": ambiguous,
            }
        )

    fmap = pd.DataFrame(records)
    n_ambig = fmap["Probe_ID_Ambiguous"].sum()
    log.info(f"Feature map: {len(fmap)} probes, {n_ambig} ambiguous")
    return fmap


def build_sample_annotation(
    meta_df: pd.DataFrame,
    gsm_cols: List[str],
    dataset_id: str,
    platform_id: str,
    platform_name: str,
    class_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Constrói DataFrame padronizado de anotação de amostras.

    Columns:
        sample_id, dataset_id, batch, platform_id, platform_name,
        condition_raw, condition_normalized, class_label
    """
    gsm_col: Optional[str] = None
    for col in meta_df.columns:
        if "sample_geo_accession" in col.lower().replace(" ", "_"):
            gsm_col = col
            break
    if gsm_col is None:
        for col in meta_df.columns:
            if "geo_accession" in col.lower():
                if meta_df[col].astype(str).str.match(r"^GSM\d+").any():
                    gsm_col = col
                    break
    if gsm_col is None:
        for col in meta_df.columns:
            if meta_df[col].astype(str).str.match(r"^GSM\d+").any():
                gsm_col = col
                break

    all_columns = list(meta_df.columns)
    records: List[dict] = []

    effective_map = {
        "pancreatic cancer": "PDAC",
        "pdac": "PDAC",
        "healthy control": "Control",
        "normal": "Control",
    }
    if class_map:
        effective_map.update(class_map)

    for gsm_id in gsm_cols:
        condition_raw = ""

        if gsm_col and gsm_id in meta_df[gsm_col].values:
            row = meta_df[meta_df[gsm_col] == gsm_id].iloc[0]
            condition_raw = extract_sample_condition(row, all_columns)

        condition_normalized = normalize_condition(condition_raw)

        class_label = condition_normalized

        matched = False
        for pattern, label in effective_map.items():
            if pattern.lower() in condition_normalized.lower():
                class_label = label
                matched = True
                break

        if not matched:
            all_healthy = HEALTHY_SYNONYMS_STRICT + HEALTHY_SYNONYMS_BROAD
            if any(term in condition_normalized.lower() for term in all_healthy):
                class_label = "Control"

        records.append(
            {
                "sample_id": gsm_id,
                "dataset_id": dataset_id,
                "batch": dataset_id,
                "platform_id": platform_id,
                "platform_name": platform_name,
                "condition_raw": condition_raw,
                "condition_normalized": condition_normalized,
                "class_label": class_label,
            }
        )

    annot = pd.DataFrame(records)
    class_counts = annot["class_label"].value_counts().to_dict()
    log.info(f"Sample annotation: {len(annot)} samples, classes: {class_counts}")
    return annot
