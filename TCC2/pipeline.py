#!/usr/bin/env python3
"""
===================================================================
  Pipeline de integração cross-platform de miRNA (GEO Series Matrix)
  para detecção precoce de câncer de pâncreas (PDAC).

  Evolução do pipeline original (app.py):
    - leitura universal de Series Matrix (Affymetrix, 3D-Gene, etc.)
    - inferência de escala (log2 / processado / raw / unknown)
    - harmonização de IDs de miRNA (canonicalização p/ MIMATxxxx)
    - z-score por probe dentro de cada dataset
    - merge cross-platform pelos miRNAs em comum
    - correção de batch com ComBat (neuroCombat OU inmoose)
    - validação por PurityB / PurityD + PCA antes/depois
===================================================================
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---- dependências opcionais --------------------------------------------
try:
    import chardet  # type: ignore
    _HAS_CHARDET = True
except ImportError:
    _HAS_CHARDET = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ========================================================================
# Logging
# ========================================================================
logger = logging.getLogger("mirna_pipeline")


def setup_logging(level: str = "INFO") -> None:
    """Configura logging padrão."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ========================================================================
# Constantes
# ========================================================================
KNOWN_PLATFORMS: Dict[str, str] = {
    "GPL21263": "3D-Gene Human miRNA (Toray)",
    "GPL18941": "3D-Gene Human miRNA V21 (Toray)",
    "GPL19117": "Affymetrix miRNA 4.0",
    "GPL18402": "Affymetrix miRNA 4.0",
    "GPL16384": "Affymetrix miRNA 3.0",
    "GPL8786":  "Affymetrix miRNA 1.0",
    "GPL10850": "Agilent Human miRNA V2",
    "GPL18058": "Agilent Human miRNA V3",
    "GPL11487": "Agilent Human miRNA V4",
    "GPL7731":  "Agilent Human miRNA V1",
    "GPL8179":  "Illumina Human v2",
    "GPL6480":  "Agilent Whole Human Genome",
    "GPL570":   "Affymetrix HG-U133 Plus 2.0",
    "GPL96":    "Affymetrix HG-U133A",
}

# Plataformas conhecidas por já entregarem dados em log2 no Series Matrix
LOG2_DEFAULT_PLATFORMS = {"GPL19117", "GPL18402", "GPL16384", "GPL8786"}
# Plataformas 3D-Gene – frequentemente normalizadas mas NEM SEMPRE em log2
THREEDGENE_PLATFORMS = {"GPL18941", "GPL21263"}

DEFAULT_PDAC_KEYWORDS = [
    "pdac", "pancreatic cancer", "pancreatic adenocarcinoma",
    "pancreatic ductal",
]
DEFAULT_CONTROL_KEYWORDS = ["healthy", "control", "normal"]

MIMAT_RE = re.compile(r"MIMAT\d+", re.IGNORECASE)
GSM_RE = re.compile(r"^GSM\d+$")
GSE_RE = re.compile(r"(GSE\d+)", re.IGNORECASE)

CONDITION_FIELD_KEYWORDS = [
    "source_name", "characteristics", "title", "description",
    "disease", "tissue", "cell_type", "treatment",
]

SCALE_ALREADY_LOG2 = "already_log2"
SCALE_ALREADY_PROCESSED = "already_processed"
SCALE_NEEDS_LOG2 = "needs_log2"
SCALE_UNKNOWN = "unknown"

PROCESSING_HINTS_LOG2 = [
    "log2", "log 2", "log-transformed", "log transformed",
    "log-2", "rma", "vst", "vsn",
]
PROCESSING_HINTS_NORMALIZED = [
    "quantile", "normalized", "normalised", "processed",
    "median normalized", "robust", "scaled", "background",
]


# ========================================================================
# 1. Leitura de Series Matrix
# ========================================================================
def detect_encoding(path: Path) -> str:
    """Detecta encoding usando chardet (se disponível) com fallback."""
    if _HAS_CHARDET:
        with open(path, "rb") as f:
            raw = f.read(2_000_000)
        detected = chardet.detect(raw).get("encoding")
        if detected:
            return detected
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(100_000)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def parse_series_metadata_tabular(path: Path) -> pd.DataFrame:
    """Lê o cabeçalho do Series Matrix e retorna DataFrame com amostras em linhas.

    Chaves duplicadas (ex.: `Sample_characteristics_ch1`) são renomeadas
    `chave`, `chave__1`, `chave__2`, ... para não perderem informação.
    """
    enc = detect_encoding(path)
    meta_lines: List[str] = []
    with open(path, "r", encoding=enc, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("!series_matrix_table_begin"):
                break
            if line.strip().startswith("!"):
                meta_lines.append(line.rstrip("\r\n"))

    parsed: List[Tuple[str, List[str]]] = []
    for raw in meta_lines:
        parts = re.split(r"\t+|\s{2,}", raw)
        if len(parts) > 1:
            key = parts[0].lstrip("!").strip()
            vals = [v.strip().strip('"') for v in parts[1:] if v.strip() != ""]
            parsed.append((key, vals))

    if not parsed:
        logger.warning("Nenhum metadado encontrado em %s", path)
        return pd.DataFrame()

    counts: Dict[str, int] = defaultdict(int)
    uniq: List[Tuple[str, List[str]]] = []
    for k, v in parsed:
        new_k = k if counts[k] == 0 else f"{k}__{counts[k]}"
        counts[k] += 1
        uniq.append((new_k, v))

    max_cols = max(len(v) for _, v in uniq)
    data = {k: v + [""] * (max_cols - len(v)) for k, v in uniq}
    df = pd.DataFrame(data)
    logger.info("Metadados lidos: %d amostras × %d campos", df.shape[0], df.shape[1])
    return df


def detect_platform(meta_df: pd.DataFrame) -> Tuple[str, str]:
    """Extrai GPL ID e nome amigável a partir dos metadados."""
    if meta_df.empty:
        return "Desconhecida", "Desconhecida"

    platform_id = "Desconhecida"
    priority = [c for c in meta_df.columns if "platform_id" in c.lower()]
    generic = [c for c in meta_df.columns if "platform" in c.lower() and c not in priority]
    for col in priority + generic:
        vals = [v for v in meta_df[col].astype(str).unique()
                if v and v.lower() != "nan"]
        if vals:
            platform_id = vals[0]
            break

    platform_name = KNOWN_PLATFORMS.get(platform_id.upper(), platform_id)
    logger.info("Plataforma detectada: %s (%s)", platform_id, platform_name)
    return platform_id, platform_name


def smart_float(x: object, broken_decimal: bool = False) -> float:
    """Converte string → float tratando NA, aspas e formato locale-quebrado.

    Quando `broken_decimal=True`, o arquivo exportou os dígitos concatenados
    com pontos inseridos como separador de milhar (ex.: valor real
    `1.129491157` aparece como `1.129.491.157`; valor real `5.97201`
    aparece como `597.201`). Regra de reconstrução: strip de todos os
    pontos, o primeiro dígito vira a parte inteira e o resto a decimal.
    """
    s = str(x).strip().strip('"')
    if s == "" or s.lower() in {"na", "nan", "null", "none", "--", "n/a"}:
        return np.nan

    if broken_decimal:
        sign = -1.0 if s.startswith("-") else 1.0
        digits = s.lstrip("-").replace(".", "").replace(",", "")
        if not digits.isdigit():
            return np.nan
        if len(digits) <= 1:
            return sign * float(digits)
        return sign * float(digits[0] + "." + digits[1:])

    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _detect_broken_decimal_format(path: Path, encoding: str) -> bool:
    """Detecta formato locale-quebrado (dígitos com >1 ponto) no arquivo.

    Retorna True se a amostra inicial de valores numéricos contém algum
    valor com mais de um ponto — sinal inequívoco de export com
    separador-de-milhar em dot-locale aplicado ao stream de dígitos.
    """
    found_header = False
    scanned = 0
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if not found_header:
                if "ID_REF" in line:
                    found_header = True
                continue
            if line.startswith("!"):
                break
            for v in line.rstrip().split("\t")[1:]:
                v = v.strip().strip('"')
                if v.count(".") > 1:
                    return True
            scanned += 1
            if scanned >= 100:
                break
    return False


def read_expression_from_txt(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Lê a matriz de expressão do Series Matrix.

    Retorna (expr_text, expr_num, gsm_cols) – SEM aplicar log2.
    Detecta automaticamente export com formato locale-quebrado.
    """
    enc = detect_encoding(path)
    header_idx: Optional[int] = None
    with open(path, "r", encoding=enc, errors="ignore") as f:
        for i, line in enumerate(f):
            if "ID_REF" in line:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Cabeçalho ID_REF não encontrado em {path}")

    broken = _detect_broken_decimal_format(path, enc)
    if broken:
        logger.warning("[parse] formato locale-quebrado detectado em %s "
                       "— aplicando reconstrução dígito-a-dígito", path.name)

    df = pd.read_csv(
        path, sep="\t", skiprows=header_idx, header=0,
        dtype=str, encoding=enc, engine="python",
    )
    df = df[df["ID_REF"].notna()]
    df = df[~df["ID_REF"].astype(str).str.startswith("!")]
    df.rename(columns={"ID_REF": "Probe_ID"}, inplace=True)
    df["Probe_ID"] = df["Probe_ID"].astype(str).str.strip().str.strip('"')

    gsm_cols = [c for c in df.columns if GSM_RE.match(str(c).strip())]
    expr_text = df[["Probe_ID"] + gsm_cols].copy()
    expr_num = expr_text.copy()
    for c in gsm_cols:
        expr_num[c] = expr_num[c].apply(lambda v: smart_float(v, broken_decimal=broken))

    nan_frac = expr_num[gsm_cols].isna().to_numpy().mean()
    logger.info("Expressão carregada: %d probes × %d amostras (NaN frac=%.4f)",
                expr_num.shape[0], len(gsm_cols), nan_frac)
    return expr_text, expr_num, gsm_cols


# ========================================================================
# 2. Inferência de escala
# ========================================================================
def _collect_processing_text(meta_df: pd.DataFrame) -> str:
    """Concatena campos de descrição de processamento em string única."""
    if meta_df.empty:
        return ""
    wanted = ["data_processing", "overall_design", "description",
              "value_definition", "summary"]
    chunks: List[str] = []
    for col in meta_df.columns:
        cl = col.lower()
        if any(w in cl for w in wanted):
            for v in meta_df[col].astype(str).unique():
                if v and v.lower() != "nan":
                    chunks.append(v)
    return " ".join(chunks).lower()


def infer_scale(
    meta_df: pd.DataFrame,
    platform_id: str,
    expr_num: pd.DataFrame,
) -> str:
    """Decide entre already_log2 / already_processed / needs_log2 / unknown.

    Usa três sinais:
      1. Texto de `!Series_data_processing` / overall_design
      2. Default por plataforma (Affymetrix miRNA 4.0 → log2 RMA)
      3. Estatística da matriz (faixa numérica, valores negativos)
    """
    text = _collect_processing_text(meta_df)
    has_log2_hint = any(h in text for h in PROCESSING_HINTS_LOG2)
    has_norm_hint = any(h in text for h in PROCESSING_HINTS_NORMALIZED)

    gsm_cols = [c for c in expr_num.columns if GSM_RE.match(str(c).strip())]
    if not gsm_cols:
        return SCALE_UNKNOWN
    sample = expr_num[gsm_cols].to_numpy(dtype=float)
    flat = sample[~np.isnan(sample)]
    if flat.size == 0:
        return SCALE_UNKNOWN

    vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))
    vmedian = float(np.nanmedian(flat))
    has_negative = vmin < 0.0

    platform_up = (platform_id or "").upper()

    # 1) Hint textual explícito vence
    if has_log2_hint:
        logger.info("[escala] hint de log2 encontrado nos metadados → already_log2")
        return SCALE_ALREADY_LOG2

    # 2) Default por plataforma (Affymetrix miRNA)
    if platform_up in LOG2_DEFAULT_PLATFORMS:
        if vmax <= 32 and (has_negative or vmedian < 20):
            logger.info("[escala] plataforma %s + faixa compatível → already_log2",
                        platform_up)
            return SCALE_ALREADY_LOG2
        # Mesmo assim, se texto indica "normalized" sem log2, tratamos como processed
        if has_norm_hint:
            return SCALE_ALREADY_PROCESSED

    # 3) 3D-Gene: avaliar por faixa
    if platform_up in THREEDGENE_PLATFORMS:
        if has_negative or vmax <= 32:
            return SCALE_ALREADY_LOG2
        if has_norm_hint or vmedian < 1000:
            return SCALE_ALREADY_PROCESSED
        return SCALE_NEEDS_LOG2

    # 4) Heurística geral
    if has_negative:
        return SCALE_ALREADY_LOG2
    if vmax <= 32:
        return SCALE_ALREADY_LOG2
    if vmax > 1000:
        return SCALE_NEEDS_LOG2
    if has_norm_hint:
        return SCALE_ALREADY_PROCESSED
    return SCALE_UNKNOWN


def prepare_analysis_ready(
    expr_num: pd.DataFrame,
    scale_decision: str,
    gsm_cols: Sequence[str],
) -> pd.DataFrame:
    """Aplica transformação de escala conforme decisão."""
    df = expr_num.copy()
    if scale_decision == SCALE_NEEDS_LOG2:
        logger.info("[escala] aplicando log2(x+1) em dados raw")
        for c in gsm_cols:
            vals = df[c].astype(float).clip(lower=0)
            df[c] = np.log2(vals + 1)
    else:
        logger.info("[escala] decisão = %s → preservando valores originais",
                    scale_decision)
    return df


# ========================================================================
# 3. Condições e filtros
# ========================================================================
def normalize_condition(value: str) -> str:
    """Remove IDs individuais de amostra do final do valor.

    Ex: 'pancreatic cancer P75' → 'pancreatic cancer'
    """
    if not value:
        return ""
    norm = re.sub(r"\s+[A-Z]{0,2}\d{1,4}\s*$", "", value.strip())
    return norm.strip()


def _condition_columns(meta_df: pd.DataFrame) -> List[str]:
    return [c for c in meta_df.columns
            if any(k in c.lower() for k in CONDITION_FIELD_KEYWORDS)]


def extract_conditions(meta_df: pd.DataFrame) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Agrupa condições normalizadas e conta amostras."""
    cols = _condition_columns(meta_df)
    counter: Counter = Counter()
    for col in cols:
        for v in meta_df[col].astype(str):
            v = v.strip().strip('"')
            if not v or v.lower() in {"nan", "none"}:
                continue
            norm = normalize_condition(v)
            if norm:
                counter[norm] += 1
    grouped = sorted(counter.items(), key=lambda x: x[0].lower())
    return grouped, cols


def auto_select_conditions(
    grouped: Sequence[Tuple[str, int]],
    pdac_keywords: Sequence[str],
    control_keywords: Sequence[str],
) -> List[Tuple[str, str]]:
    """Seleção automática: devolve (condição, class_label) para PDAC/Control."""
    result: List[Tuple[str, str]] = []
    for cond, _ in grouped:
        c = cond.lower()
        if any(k in c for k in pdac_keywords):
            result.append((cond, "PDAC"))
        elif any(k in c for k in control_keywords):
            result.append((cond, "Control"))
    return result


def select_conditions_interactive(
    grouped: Sequence[Tuple[str, int]],
) -> List[str]:
    """Permite seleção interativa de múltiplas condições."""
    if not grouped:
        return []
    print("\n" + "=" * 60)
    print("  Condições encontradas:")
    print("=" * 60)
    for i, (cond, count) in enumerate(grouped, 1):
        print(f"  [{i}] {cond}  ({count} amostras)")
    print("  [0] Todas (sem filtro)")
    print("=" * 60)
    try:
        raw = input("Selecione (vírgulas, ex: 1,3) ou 0: ").strip()
    except (EOFError, KeyboardInterrupt):
        return []
    if raw == "0" or raw == "":
        return []
    picks: List[str] = []
    for tok in raw.split(","):
        try:
            idx = int(tok.strip())
            if 1 <= idx <= len(grouped):
                picks.append(grouped[idx - 1][0])
        except ValueError:
            continue
    return picks


# ========================================================================
# 4. Canonicalização de probes
# ========================================================================
def canonicalize_probe_id(
    probe_id: str,
    platform_id: str,
) -> Tuple[str, bool]:
    """Devolve (canonical_id, is_ambiguous).

    - Affymetrix: remove sufixo `_st` (ex.: MIMAT0000062_st → MIMAT0000062).
    - 3D-Gene: probes podem conter múltiplos MIMAT separados por vírgula.
    - Ambos: se houver >1 MIMAT no identificador, flag ambíguo.
    """
    if probe_id is None:
        return ("", True)
    pid = str(probe_id).strip().strip('"')
    if not pid:
        return ("", True)

    platform_up = (platform_id or "").upper()
    cleaned = pid

    # Affymetrix: remover sufixo _st (tanto em MIMAT_st quanto em hsa-miR-xxx_st)
    if platform_up in {"GPL19117", "GPL18402", "GPL16384", "GPL8786"}:
        cleaned = re.sub(r"_st$", "", cleaned, flags=re.IGNORECASE)

    mimats = MIMAT_RE.findall(cleaned)
    if len(mimats) == 1:
        return (mimats[0].upper(), False)
    if len(mimats) > 1:
        # 3D-Gene e casos Affymetrix onde múltiplos MIMATs aparecem separados por vírgula
        return (mimats[0].upper(), True)

    # Sem MIMAT: retorna o id limpo e NÃO marca ambíguo
    # (apenas o heurístico "multi-miRNA" constitui ambiguidade real).
    return (cleaned, False)


def build_feature_map(
    probe_ids: Iterable[str],
    platform_id: str,
) -> pd.DataFrame:
    """Gera DataFrame com Probe_ID, Probe_ID_Canonical, Probe_ID_Ambiguous."""
    rows = []
    for pid in probe_ids:
        canon, ambig = canonicalize_probe_id(pid, platform_id)
        rows.append({
            "Probe_ID": pid,
            "Probe_ID_Canonical": canon,
            "Probe_ID_Ambiguous": ambig,
        })
    return pd.DataFrame(rows)


def prepare_merge_ready(
    expr_analysis: pd.DataFrame,
    feature_map: pd.DataFrame,
    gsm_cols: Sequence[str],
) -> pd.DataFrame:
    """Remove probes ambíguos, agrega duplicatas por canonical (média).

    Retorna DataFrame indexado por Probe_ID_Canonical (linhas = miRNA,
    colunas = GSM).
    """
    df = expr_analysis.merge(feature_map, on="Probe_ID", how="left")
    df = df[~df["Probe_ID_Ambiguous"]]
    df = df[df["Probe_ID_Canonical"].astype(str).str.len() > 0]
    # Se sobrou probe sem MIMAT, ele permanece com o ID limpo como canonical
    agg = df.groupby("Probe_ID_Canonical")[list(gsm_cols)].mean()
    agg.index.name = "Probe_ID_Canonical"
    return agg


# ========================================================================
# 5. Imputação e Z-score por probe
# ========================================================================
def impute_by_probe_median(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN com a mediana do probe (linha). Preserva índice/colunas."""
    values = expr_df.to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=1, keepdims=True)
    mask = np.isnan(values)
    if not mask.any():
        return expr_df
    # Probes totalmente NaN: preenche com 0 (fallback neutro)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = np.where(mask, np.broadcast_to(medians, values.shape), values)
    out = pd.DataFrame(filled, index=expr_df.index, columns=expr_df.columns)
    n_filled = int(mask.sum())
    logger.info("[impute] %d células NaN preenchidas com mediana do probe", n_filled)
    return out


def zscore_by_probe(expr_df: pd.DataFrame) -> pd.DataFrame:
    """z = (x - mean_row) / std_row, NaN-safe, dropa probes com std=0."""
    values = expr_df.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, ddof=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - mean) / std
    z_df = pd.DataFrame(z, index=expr_df.index, columns=expr_df.columns)
    finite_std = np.squeeze(std) > 0
    z_df = z_df.loc[finite_std]
    logger.info("[z-score] %d probes mantidos (std>0)", z_df.shape[0])
    return z_df


# ========================================================================
# 6. Anotação de amostras
# ========================================================================
def _find_gsm_column(meta_df: pd.DataFrame) -> Optional[str]:
    for c in meta_df.columns:
        if "geo_accession" in c.lower():
            if meta_df[c].astype(str).str.match(r"^GSM\d+$").any():
                return c
    for c in meta_df.columns:
        if meta_df[c].astype(str).str.match(r"^GSM\d+$").any():
            return c
    return None


def _classify(
    condition_text: str,
    pdac_keywords: Sequence[str],
    control_keywords: Sequence[str],
) -> str:
    c = condition_text.lower()
    if any(k in c for k in pdac_keywords):
        return "PDAC"
    if any(k in c for k in control_keywords):
        return "Control"
    return "Other"


def build_sample_annotation(
    meta_df: pd.DataFrame,
    dataset_id: str,
    platform_id: str,
    platform_name: str,
    pdac_keywords: Sequence[str],
    control_keywords: Sequence[str],
) -> pd.DataFrame:
    """Constrói tabela de anotação de amostras (sample_id em linhas)."""
    if meta_df.empty:
        return pd.DataFrame()
    gsm_col = _find_gsm_column(meta_df)
    cond_cols = _condition_columns(meta_df)

    rows: List[Dict[str, str]] = []
    for _, row in meta_df.iterrows():
        gsm: Optional[str] = None
        if gsm_col is not None:
            val = str(row[gsm_col]).strip()
            if GSM_RE.match(val):
                gsm = val
        if gsm is None:
            # fallback: varre linha em busca de GSM
            for c in meta_df.columns:
                val = str(row[c]).strip()
                if GSM_RE.match(val):
                    gsm = val
                    break
        if gsm is None:
            continue

        parts: List[str] = []
        for c in cond_cols:
            v = str(row[c]).strip().strip('"')
            if v and v.lower() not in {"nan", "none", ""}:
                parts.append(v)
        condition_raw = " | ".join(parts)
        primary = parts[0] if parts else ""
        condition_normalized = normalize_condition(primary)
        class_label = _classify(condition_raw, pdac_keywords, control_keywords)

        rows.append({
            "sample_id": gsm,
            "dataset_id": dataset_id,
            "batch": dataset_id,
            "platform_id": platform_id,
            "platform_name": platform_name,
            "condition_raw": condition_raw,
            "condition_normalized": condition_normalized,
            "class_label": class_label,
        })
    return pd.DataFrame(rows)


# ========================================================================
# 7. Pipeline por dataset
# ========================================================================
def process_dataset(
    path: Path,
    output_root: Path,
    pdac_keywords: Sequence[str],
    control_keywords: Sequence[str],
    interactive: bool,
    include_all: bool,
) -> Dict[str, object]:
    """Executa o pipeline completo para UM arquivo Series Matrix.

    Retorna dict com paths dos arquivos gerados e metadados do dataset.
    """
    logger.info("=" * 60)
    logger.info("Processando: %s", path)
    logger.info("=" * 60)

    stem = path.stem
    gse_match = GSE_RE.search(stem)
    dataset_id = gse_match.group(1).upper() if gse_match else stem
    out_dir = output_root / f"out_{dataset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_series_metadata_tabular(path)
    if meta.empty:
        raise RuntimeError(f"Metadados vazios em {path}")

    platform_id, platform_name = detect_platform(meta)

    grouped, _cond_cols = extract_conditions(meta)
    logger.info("[condições] %d grupos encontrados", len(grouped))

    # Construir anotação completa antes de filtrar – útil para debug
    annot_full = build_sample_annotation(
        meta, dataset_id, platform_id, platform_name,
        pdac_keywords, control_keywords,
    )

    # Decidir quais amostras manter
    if include_all:
        annot = annot_full.copy()
    elif interactive:
        picks = select_conditions_interactive(grouped)
        if not picks:
            annot = annot_full.copy()
        else:
            mask = annot_full["condition_normalized"].str.lower().isin(
                [p.lower() for p in picks]
            ) | annot_full["condition_raw"].str.lower().apply(
                lambda s: any(p.lower() in s for p in picks)
            )
            annot = annot_full.loc[mask].copy()
    else:
        # auto: mantém apenas PDAC + Control (class_label != "Other")
        auto = auto_select_conditions(grouped, pdac_keywords, control_keywords)
        logger.info("[auto] condições selecionadas: %s",
                    [c for c, _ in auto] or "nenhuma (fallback: todas)")
        annot = annot_full[annot_full["class_label"].isin(["PDAC", "Control"])].copy()
        if annot.empty:
            logger.warning("[auto] nenhuma amostra PDAC/Control detectada — usando TODAS")
            annot = annot_full.copy()

    logger.info("Amostras após filtro: %d", annot.shape[0])

    # --- Ler expressão -------------------------------------------------
    expr_text, expr_num, gsm_cols = read_expression_from_txt(path)

    # --- Inferir escala ------------------------------------------------
    scale = infer_scale(meta, platform_id, expr_num)
    expr_ready = prepare_analysis_ready(expr_num, scale, gsm_cols)

    # --- Restringir às amostras filtradas -----------------------------
    kept_samples = [g for g in gsm_cols if g in set(annot["sample_id"])]
    if not kept_samples:
        raise RuntimeError(
            f"Nenhuma coluna GSM do arquivo corresponde às amostras filtradas em {dataset_id}"
        )
    annot = annot[annot["sample_id"].isin(kept_samples)].reset_index(drop=True)

    expr_text_f = expr_text[["Probe_ID"] + kept_samples].copy()
    expr_num_f = expr_num[["Probe_ID"] + kept_samples].copy()
    expr_ready_f = expr_ready[["Probe_ID"] + kept_samples].copy()

    # --- Canonicalização + merge-ready --------------------------------
    feature_map = build_feature_map(expr_ready_f["Probe_ID"].tolist(), platform_id)
    expr_merge_ready = prepare_merge_ready(expr_ready_f, feature_map, kept_samples)
    expr_merge_ready = impute_by_probe_median(expr_merge_ready)
    expr_merge_zscore = zscore_by_probe(expr_merge_ready)

    # --- Salvar --------------------------------------------------------
    expr_text_f.to_csv(out_dir / "expression_text_preservado.csv", index=False)
    expr_num_f.to_csv(out_dir / "expression_numerico.csv", index=False)
    expr_ready_f.to_csv(out_dir / "expression_analysis_ready.csv", index=False)
    feature_map.to_csv(out_dir / "feature_map.csv", index=False)
    expr_merge_ready.to_csv(out_dir / "expression_merge_ready.csv")
    expr_merge_zscore.to_csv(out_dir / "expression_merge_ready_zscore.csv")
    annot.to_csv(out_dir / "sample_annotation.csv", index=False)
    meta.to_csv(out_dir / "metadata_full.csv", index=False)

    logger.info("[%s] arquivos gerados em %s", dataset_id, out_dir)

    return {
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "platform_name": platform_name,
        "scale_decision": scale,
        "out_dir": out_dir,
        "expr_merge_ready": out_dir / "expression_merge_ready.csv",
        "expr_merge_ready_zscore": out_dir / "expression_merge_ready_zscore.csv",
        "sample_annotation": out_dir / "sample_annotation.csv",
        "feature_map": out_dir / "feature_map.csv",
        "n_samples": annot.shape[0],
        "n_probes_merge_ready": expr_merge_ready.shape[0],
    }


# ========================================================================
# 8. Merge cross-platform
# ========================================================================
def _merge_matrices(
    paths: Sequence[Path],
    annot_paths: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper: carrega CSVs, mantém interseção de índices e concatena colunas."""
    expr_dfs: List[pd.DataFrame] = []
    annot_dfs: List[pd.DataFrame] = []
    for p, ap in zip(paths, annot_paths):
        expr = pd.read_csv(p, index_col=0)
        annot = pd.read_csv(ap)
        common = [c for c in expr.columns if c in set(annot["sample_id"])]
        expr_dfs.append(expr[common])
        annot_dfs.append(annot[annot["sample_id"].isin(common)])

    common_idx = set(expr_dfs[0].index)
    for df in expr_dfs[1:]:
        common_idx &= set(df.index)
    common_idx = sorted(common_idx)
    if not common_idx:
        raise RuntimeError("Interseção de miRNAs vazia — revisar canonicalização")

    aligned = [df.loc[common_idx] for df in expr_dfs]
    merged_expr = pd.concat(aligned, axis=1)
    merged_annot = pd.concat(annot_dfs, ignore_index=True)
    merged_annot = merged_annot[merged_annot["sample_id"].isin(merged_expr.columns)]
    merged_annot = merged_annot.drop_duplicates("sample_id").reset_index(drop=True)
    merged_expr = merged_expr[list(merged_annot["sample_id"])]
    return merged_expr, merged_annot


def merge_datasets(
    dataset_results: Sequence[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combina matrizes dos datasets pela interseção de miRNAs.

    Retorna (merged_raw, merged_zscore, merged_annotation).
    - merged_raw: concatenado de `expression_merge_ready.csv` (pré-z-score)
    - merged_zscore: concatenado de `expression_merge_ready_zscore.csv`
    - merged_annotation: anotação concatenada
    """
    if not dataset_results:
        raise RuntimeError("Nenhum dataset para merge")

    zscore_paths = [r["expr_merge_ready_zscore"] for r in dataset_results]
    raw_paths = [r["expr_merge_ready"] for r in dataset_results]
    annot_paths = [r["sample_annotation"] for r in dataset_results]

    merged_z, merged_annot = _merge_matrices(zscore_paths, annot_paths)
    merged_raw, _ = _merge_matrices(raw_paths, annot_paths)
    # Restringe raw para mesmo conjunto de miRNAs e amostras do z-scored
    merged_raw = merged_raw.loc[
        [i for i in merged_raw.index if i in set(merged_z.index)],
        list(merged_annot["sample_id"]),
    ]
    merged_raw = merged_raw.loc[list(merged_z.index)]

    logger.info("[merge] matriz final: %d miRNAs × %d amostras (em %d datasets)",
                merged_z.shape[0], merged_z.shape[1], len(dataset_results))
    return merged_raw, merged_z, merged_annot


# ========================================================================
# 9. ComBat
# ========================================================================
def apply_combat(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
) -> pd.DataFrame:
    """Aplica ComBat preservando a classe biológica como covariável.

    Ordem de tentativa:
      1) neuroCombat
      2) inmoose.pycombat.pycombat_norm
    """
    annot = sample_annot.set_index("sample_id")
    samples = [s for s in expr_df.columns if s in annot.index]
    X = expr_df[samples].copy()
    batch = annot.loc[samples, batch_col].astype(str)
    klass = annot.loc[samples, class_col].astype(str)

    if batch.nunique() < 2:
        logger.warning("[combat] apenas 1 batch — nada a corrigir, retornando cópia")
        return X

    # Safety net: ComBat/neuroCombat não lidam bem com NaN
    if X.isna().any().any():
        logger.warning("[combat] NaN detectado na entrada — imputando mediana do probe")
        X = impute_by_probe_median(X)

    # --- 1) neuroCombat ------------------------------------------------
    try:
        from neuroCombat import neuroCombat  # type: ignore
        logger.info("[combat] usando neuroCombat")
        covars = pd.DataFrame({batch_col: batch.values, class_col: klass.values})
        result = neuroCombat(
            dat=X.values,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=[class_col],
        )
        corrected = pd.DataFrame(result["data"], index=X.index, columns=X.columns)
        return corrected
    except ImportError:
        logger.debug("[combat] neuroCombat indisponível")

    # --- 2) inmoose.pycombat ------------------------------------------
    try:
        from inmoose.pycombat import pycombat_norm  # type: ignore
        logger.info("[combat] usando inmoose.pycombat.pycombat_norm")
        # mod = design matrix de covariáveis (one-hot da classe, sem intercepto)
        mod = pd.get_dummies(klass, drop_first=True).astype(float)
        if mod.shape[1] == 0:
            mod = None
        corrected = pycombat_norm(
            X, batch=list(batch.values), mod=mod,
        )
        if isinstance(corrected, np.ndarray):
            corrected = pd.DataFrame(corrected, index=X.index, columns=X.columns)
        return corrected
    except ImportError:
        logger.debug("[combat] inmoose indisponível")

    raise ImportError(
        "Nenhuma biblioteca ComBat encontrada. "
        "Instale uma destas: `pip install neuroCombat` ou `pip install inmoose`."
    )


# ========================================================================
# 10. Validação por pureza
# ========================================================================
def calculate_purity(
    cluster_labels: Sequence,
    true_labels: Sequence,
) -> float:
    """Purity = (1/N) * Σ_k max_j |w_k ∩ c_j|."""
    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)
    if len(cluster_labels) != len(true_labels):
        raise ValueError("cluster_labels e true_labels com tamanhos diferentes")
    N = len(cluster_labels)
    if N == 0:
        return 0.0
    total = 0
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if not mask.any():
            continue
        counts = Counter(true_labels[mask])
        total += max(counts.values())
    return total / N


def _reduce_for_clustering(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """PCA para clusterização — evita que outliers em alta dimensão dominem
    o KMeans. Usa no máximo min(n_components, n_samples-1, n_features)."""
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_comp < 2:
        return X
    return PCA(n_components=n_comp, random_state=42).fit_transform(X)


def _kmeans_labels(X: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, X.shape[0]))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


def _purity_pair(X: np.ndarray, batches: np.ndarray, classes: np.ndarray) -> Tuple[float, float]:
    """Calcula PurityB e PurityD em um mesmo embedding PCA-reduzido."""
    X_red = _reduce_for_clustering(X, n_components=10)
    nb = max(2, len(set(batches)))
    nc = max(2, len(set(classes)))
    pb = calculate_purity(_kmeans_labels(X_red, nb), batches)
    pd_ = calculate_purity(_kmeans_labels(X_red, nc), classes)
    return pb, pd_


def purity_validation(
    expr_before: pd.DataFrame,
    expr_after: pd.DataFrame,
    sample_annot: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
    expr_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calcula PurityB / PurityD antes e depois do ComBat.

    Clusterização é feita após redução por PCA(10) para mitigar ruído de
    alta dimensionalidade. Se `expr_raw` for passado (merged pré-z-score),
    também reporta um estágio de baseline.
    """
    annot = sample_annot.set_index("sample_id")
    samples = [s for s in expr_before.columns if s in annot.index and s in expr_after.columns]
    batches = annot.loc[samples, batch_col].astype(str).to_numpy()
    classes = annot.loc[samples, class_col].astype(str).to_numpy()

    rows: List[Dict[str, object]] = []

    if expr_raw is not None:
        raw_samples = [s for s in samples if s in expr_raw.columns]
        if len(raw_samples) == len(samples):
            X_raw = expr_raw[samples].fillna(0).to_numpy().T
            pb, pdd = _purity_pair(X_raw, batches, classes)
            rows.append({"stage": "merged_raw", "PurityB": pb, "PurityD": pdd})
            logger.info("[purity] merged_raw           PurityB=%.3f  PurityD=%.3f",
                        pb, pdd)

    X_before = expr_before[samples].fillna(0).to_numpy().T
    X_after = expr_after[samples].fillna(0).to_numpy().T
    pb_b, pd_b = _purity_pair(X_before, batches, classes)
    pb_a, pd_a = _purity_pair(X_after, batches, classes)
    rows.append({"stage": "before_combat", "PurityB": pb_b, "PurityD": pd_b})
    rows.append({"stage": "after_combat",  "PurityB": pb_a, "PurityD": pd_a})

    logger.info("[purity] before_combat (zscore)   PurityB=%.3f  PurityD=%.3f",
                pb_b, pd_b)
    logger.info("[purity] after_combat            PurityB=%.3f  PurityD=%.3f",
                pb_a, pd_a)
    logger.info("[purity] ΔPurityB (after-before)=%+.3f   (esperado: ≤0)",
                pb_a - pb_b)
    logger.info("[purity] ΔPurityD (after-before)=%+.3f   (esperado: ≥0)",
                pd_a - pd_b)
    return pd.DataFrame(rows)


# ========================================================================
# 11. Visualizações (PCA)
# ========================================================================
def plot_pca(
    expr_df: pd.DataFrame,
    sample_annot: pd.DataFrame,
    color_by: str,
    out_path: Path,
    title: str,
) -> None:
    """Gera scatter PCA (2 PCs) colorido pela coluna indicada."""
    annot = sample_annot.set_index("sample_id")
    samples = [s for s in expr_df.columns if s in annot.index]
    X = expr_df[samples].fillna(0).to_numpy().T  # samples × features
    if X.shape[0] < 2 or X.shape[1] < 2:
        logger.warning("[pca] dados insuficientes para %s", out_path.name)
        return
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    labels = annot.loc[samples, color_by].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(sorted(set(labels))):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=lbl, alpha=0.75, s=45,
            color=cmap(i % 10),
            edgecolor="black", linewidth=0.4,
        )
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(title=color_by, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("[pca] salvo: %s", out_path)


def generate_all_pca_plots(
    expr_before: pd.DataFrame,
    expr_after: pd.DataFrame,
    sample_annot: pd.DataFrame,
    output_root: Path,
) -> None:
    plot_pca(expr_before, sample_annot, "batch",
             output_root / "pca_before_batch.png",
             "PCA antes de ComBat — colorido por batch")
    plot_pca(expr_before, sample_annot, "class_label",
             output_root / "pca_before_class.png",
             "PCA antes de ComBat — colorido por classe")
    plot_pca(expr_after, sample_annot, "batch",
             output_root / "pca_after_batch.png",
             "PCA após ComBat — colorido por batch")
    plot_pca(expr_after, sample_annot, "class_label",
             output_root / "pca_after_class.png",
             "PCA após ComBat — colorido por classe")


# ========================================================================
# 12. CLI
# ========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline cross-platform de miRNA para PDAC (GEO Series Matrix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="Um ou mais arquivos *_series_matrix.txt")
    p.add_argument("--output-root", type=Path, default=Path("./out"),
                   help="Diretório raiz de saída")
    p.add_argument("--interactive", action="store_true",
                   help="Seleciona condições interativamente (por dataset)")
    p.add_argument("--include-all", action="store_true",
                   help="Mantém todas as amostras (sem filtro de condição)")
    p.add_argument("--pdac-keywords", nargs="+",
                   default=DEFAULT_PDAC_KEYWORDS,
                   help="Palavras-chave para classificar como PDAC")
    p.add_argument("--control-keywords", nargs="+",
                   default=DEFAULT_CONTROL_KEYWORDS,
                   help="Palavras-chave para classificar como Control")
    p.add_argument("--no-combat", action="store_true",
                   help="Pula correção de batch com ComBat")
    p.add_argument("--no-plots", action="store_true",
                   help="Pula geração de PCA")
    p.add_argument("--skip-merge", action="store_true",
                   help="Pula merge mesmo com múltiplos inputs")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Processa cada dataset ---------------------------------------
    results: List[Dict[str, object]] = []
    for inp in args.inputs:
        if not inp.exists():
            logger.error("Arquivo não encontrado: %s", inp)
            return 2
        try:
            res = process_dataset(
                path=inp,
                output_root=output_root,
                pdac_keywords=args.pdac_keywords,
                control_keywords=args.control_keywords,
                interactive=args.interactive,
                include_all=args.include_all,
            )
            results.append(res)
        except Exception as e:
            logger.exception("Falha processando %s: %s", inp, e)
            return 3

    # --- Merge -------------------------------------------------------
    if args.skip_merge or len(results) < 2:
        logger.info("Merge pulado (skip_merge=%s, n_datasets=%d)",
                    args.skip_merge, len(results))
        return 0

    merged_raw, merged_expr, merged_annot = merge_datasets(results)
    merged_raw.to_csv(output_root / "merged_expression_raw.csv")
    merged_expr.to_csv(output_root / "merged_expression_zscore.csv")
    merged_annot.to_csv(output_root / "merged_sample_annotation.csv", index=False)
    logger.info("[merge] salvos merged_expression_{raw,zscore}.csv / merged_sample_annotation.csv")

    # --- ComBat ------------------------------------------------------
    if args.no_combat:
        logger.info("ComBat pulado (--no-combat)")
        return 0

    try:
        merged_combat = apply_combat(merged_expr, merged_annot,
                                     batch_col="batch", class_col="class_label")
    except ImportError as e:
        logger.error(str(e))
        return 0
    except Exception as e:
        logger.exception("ComBat falhou: %s", e)
        return 4

    merged_combat.to_csv(output_root / "merged_expression_combat.csv")
    logger.info("[combat] salvo merged_expression_combat.csv")

    # --- Validação ---------------------------------------------------
    purity_df = purity_validation(
        expr_before=merged_expr,
        expr_after=merged_combat,
        sample_annot=merged_annot,
        expr_raw=merged_raw,
    )
    purity_df.to_csv(output_root / "purity_metrics.csv", index=False)
    logger.info("[purity] salvo purity_metrics.csv")

    # --- PCAs --------------------------------------------------------
    if not args.no_plots:
        generate_all_pca_plots(
            expr_before=merged_expr,
            expr_after=merged_combat,
            sample_annot=merged_annot,
            output_root=output_root,
        )

    logger.info("Pipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
