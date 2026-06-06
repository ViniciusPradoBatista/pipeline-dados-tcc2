"""Processamento per-dataset e merge cross-platform pelos miRNAs em comum."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geo_pipeline.conditions import (
    auto_include_healthy_controls,
    extract_conditions,
    filter_samples_by_conditions,
    select_conditions_cli,
)
from geo_pipeline.expression import read_expression
from geo_pipeline.features import build_feature_map, build_sample_annotation
from geo_pipeline.io_geo import parse_series_metadata_tabular
from geo_pipeline.scale import detect_platform, infer_scale

log = logging.getLogger("geo_pipeline")


def extract_dataset_id(path: str) -> str:
    """Extrai o accession GSE de um nome de arquivo."""
    m = re.search(r"(GSE\d+)", Path(path).stem, re.IGNORECASE)
    return m.group(1) if m else Path(path).stem


def process_single_dataset(
    path: str,
    output_root: Path,
    no_interactive: bool = False,
    condition_filter: Optional[List[str]] = None,
    class_map: Optional[Dict[str, str]] = None,
    auto_add_healthy_control: bool = True,
    strict_control_only: bool = True,
) -> Optional[Path]:
    """Roda o pipeline completo de processamento em um Series Matrix."""
    dataset_id = extract_dataset_id(path)
    out_dir = output_root / f"out_{dataset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 60)
    log.info(f"  DATASET: {dataset_id}")
    log.info(f"  File:    {path}")
    log.info(f"  Output:  {out_dir.absolute()}")
    log.info("=" * 60)

    # ── STEP 1: Metadata ─────────────────────────────────────────
    log.info("── Step 1: Metadata extraction ──")
    meta_df = parse_series_metadata_tabular(path)
    if meta_df.empty:
        log.error(f"Failed to parse metadata from {path}")
        return None

    platform_id, platform_name = detect_platform(meta_df)

    # ── STEP 2: Condition selection ──────────────────────────────
    log.info("── Step 2: Condition filtering ──")
    conditions, condition_cols = extract_conditions(meta_df)
    selected = select_conditions_cli(
        conditions,
        condition_filter,
        no_interactive,
    )

    if selected is not None and auto_add_healthy_control:
        log.info(f"Selected by user: {selected}")
        final_conditions = auto_include_healthy_controls(
            selected, conditions, strict_control_only
        )
        if final_conditions != selected:
            log.info(f"Final conditions used for filtering: {final_conditions}")
        selected = final_conditions

    meta_filtered = filter_samples_by_conditions(
        meta_df,
        selected,
        condition_cols,
    )

    meta_df.to_csv(out_dir / "metadata_full.csv", index=False)
    if selected:
        label = "_".join(
            re.sub(r"[^\w\s-]", "", c).strip().replace(" ", "_") for c in selected
        )
        meta_filtered.to_csv(
            out_dir / f"metadata_{label[:50]}.csv",
            index=False,
        )

    # ── STEP 3: Expression reading ───────────────────────────────
    log.info("── Step 3: Expression reading ──")
    expr_text, expr_num, gsm_cols = read_expression(path)

    # ── STEP 4: Cross-reference with filtered samples ────────────
    log.info("── Step 4: Cross-referencing samples ──")
    filtered_gsms: set = set()
    for col in meta_filtered.columns:
        vals = meta_filtered[col].astype(str)
        # Limitar busca de GSMs a colunas reconhecidamente de ID de amostra,
        # evitando incluir GSMs mencionados em campos de texto de descrição.
        if any(k in col.lower() for k in ("geo_accession", "sample", "gsm")):
            filtered_gsms.update(v for v in vals if v.startswith("GSM"))

    # Fallback: buscar GSMs em todas as colunas se colunas específicas não retornaram nada
    if not filtered_gsms:
        for col in meta_filtered.columns:
            vals = meta_filtered[col].astype(str)
            filtered_gsms.update(v for v in vals if v.startswith("GSM"))

    if selected and filtered_gsms:
        keep_cols = [c for c in gsm_cols if c in filtered_gsms]
        if not keep_cols:
            log.warning("No matching GSM columns after filter; using ALL columns")
            keep_cols = gsm_cols
    else:
        keep_cols = gsm_cols

    log.info(f"Samples retained: {len(keep_cols)}")

    expr_text_filt = expr_text[["Probe_ID"] + keep_cols].copy()
    expr_num_filt = expr_num[["Probe_ID"] + keep_cols].copy()

    # ── STEP 5: Scale inference ──────────────────────────────────
    log.info("── Step 5: Scale inference ──")
    scale = infer_scale(meta_df, platform_id, expr_num_filt)

    expr_ready = expr_num_filt.copy()
    if scale == "needs_log2":
        log.info("Applying log2(x + 1) transformation")
        for c in keep_cols:
            n_neg = int((expr_ready[c] < 0).sum())
            if n_neg > 0:
                log.warning(
                    f"Coluna {c}: {n_neg} valores negativos encontrados antes do log2. "
                    f"Estes valores serão convertidos para 0 via clip(lower=0). "
                    f"Verifique se a normalização do dataset introduziu artefatos de background."
                )
            expr_ready[c] = np.log2(expr_ready[c].clip(lower=0) + 1)
    elif scale in ("already_log2", "already_processed"):
        log.info(f"Scale='{scale}' → no transformation applied")
    else:
        log.warning(f"Scale='{scale}' → keeping values as-is. Please verify manually.")

    # ── STEP 6: Probe ID harmonization ───────────────────────────
    log.info("── Step 6: Probe ID harmonization ──")
    feature_map = build_feature_map(expr_num_filt, platform_id)

    fmap_clean = feature_map[~feature_map["Probe_ID_Ambiguous"]].copy()

    expr_merge = expr_ready.merge(
        fmap_clean[["Probe_ID", "Probe_ID_Canonical"]],
        on="Probe_ID",
        how="inner",
    )
    expr_merge.drop(columns=["Probe_ID"], inplace=True)
    expr_merge.rename(
        columns={"Probe_ID_Canonical": "Probe_ID"},
        inplace=True,
    )

    if expr_merge["Probe_ID"].duplicated().any():
        n_dup = expr_merge["Probe_ID"].duplicated().sum()
        log.info(
            f"Colapsando {n_dup} Probe IDs canônicos duplicados usando mediana "
            f"(mediana é mais robusta que média em escala log2 para probes heterogêneas)"
        )
        expr_merge = expr_merge.groupby("Probe_ID", as_index=False)[keep_cols].median()

    expr_merge = expr_merge[["Probe_ID"] + keep_cols]

    # NOTA: O z-score NÃO é calculado aqui por dataset individual.
    # O z-score global (calculado sobre todos os datasets integrados pós-ComBat)
    # é produzido em geo_mirna_pipeline.py. Calcular z-score por dataset e
    # depois concatenar produziria escalas incompatíveis entre datasets.

    # ── STEP 7: Sample annotation ────────────────────────────────
    log.info("── Step 7: Sample annotation ──")
    sample_annot = build_sample_annotation(
        meta_filtered,
        keep_cols,
        dataset_id,
        platform_id,
        platform_name,
        class_map,
    )

    # ── SAVE ALL OUTPUTS ─────────────────────────────────────────
    log.info("── Saving outputs ──")
    expr_text_filt.to_csv(out_dir / "expression_text_preservado.csv", index=False)
    expr_num_filt.to_csv(out_dir / "expression_numerico.csv", index=False)
    expr_ready.to_csv(out_dir / "expression_analysis_ready.csv", index=False)
    feature_map.to_csv(out_dir / "feature_map.csv", index=False)
    expr_merge.to_csv(out_dir / "expression_merge_ready.csv", index=False)
    sample_annot.to_csv(out_dir / "sample_annotation.csv", index=False)

    return out_dir


def merge_datasets(
    dataset_dirs: List[Path],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combina múltiplos datasets pelos miRNAs em comum (escala log2).

    Retorna apenas os dados em escala log2 (sem z-score por dataset).
    O z-score global é calculado em geo_mirna_pipeline.py após o ComBat,
    garantindo que a normalização ocorra sobre a matriz integrada completa.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  MERGE: Combining datasets")
    log.info("=" * 60)

    all_raw: List[pd.DataFrame] = []
    all_annot: List[pd.DataFrame] = []
    all_mirna_sets: List[set] = []

    for d in dataset_dirs:
        raw_path = d / "expression_merge_ready.csv"
        annot_path = d / "sample_annotation.csv"

        if not raw_path.exists() or not annot_path.exists():
            log.warning(f"Dataset {d.name}: arquivos esperados não encontrados, pulando.")
            continue

        raw = pd.read_csv(raw_path)
        annot = pd.read_csv(annot_path)

        mirnas = set(raw["Probe_ID"].unique())
        all_mirna_sets.append(mirnas)
        all_raw.append(raw)
        all_annot.append(annot)

    if len(all_raw) < 2:
        if all_raw:
            log.warning(
                "Apenas 1 dataset disponível para merge. Retornando sem merge. "
                "ComBat não será necessário (sem efeito de batch a corrigir)."
            )
            return all_raw[0], all_annot[0]
        return pd.DataFrame(), pd.DataFrame()

    common: set = all_mirna_sets[0]
    for s in all_mirna_sets[1:]:
        common = common.intersection(s)

    if not common:
        log.error("Nenhum miRNA em comum entre os datasets. Verifique a harmonização de Probe IDs.")
        return pd.DataFrame(), pd.DataFrame()

    log.info(f"miRNAs em comum entre {len(all_raw)} datasets: {len(common)}")

    def _merge_list(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        parts = []
        for df in dfs:
            sub = df[df["Probe_ID"].isin(common)].copy()
            sub = sub.set_index("Probe_ID")
            parts.append(sub)
        return pd.concat(parts, axis=1).reset_index()

    merged_raw = _merge_list(all_raw)

    # Fail-loud: GSMs devem ser únicos por dataset e datasets distintos têm GSMs
    # distintos. Colunas de amostra duplicadas após o concat indicam o MESMO dataset
    # passado 2× ou colisão de IDs — pd.concat(axis=1) não deduplica.
    dup_cols = merged_raw.columns[merged_raw.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(
            f"merge_datasets: colunas de amostra duplicadas entre datasets: "
            f"{dup_cols[:10]}. Verifique se o mesmo dataset foi passado mais de uma vez."
        )

    merged_annot = pd.concat(all_annot, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_raw.to_csv(output_dir / "merged_expression_raw.csv", index=False)
    merged_annot.to_csv(output_dir / "merged_sample_annotation.csv", index=False)

    log.info(
        f"Samples by class after merge: "
        f"{merged_annot['class_label'].value_counts().to_dict()}"
    )

    return merged_raw, merged_annot
