"""Leitura de tabelas de expressão de arquivos GEO Series Matrix (.txt e .xlsx)."""

import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openpyxl
import pandas as pd

from geo_pipeline.io_geo import detect_encoding
from geo_pipeline.parsing import smart_float

log = logging.getLogger("geo_pipeline")


def _detect_broken_decimal(values: List[str], max_scan: int = 100) -> bool:
    """
    Detecta formato locale-quebrado: valor com mais de um ponto é sinal
    inequívoco de export com separador-de-milhar em dot-locale.

    Scan dos primeiros ``max_scan`` valores não-vazios.
    """
    scanned = 0
    for v in values:
        s = str(v).strip().strip('"')
        if not s or s.lower() in ("na", "nan", "null", "none", "--", "n/a"):
            continue
        if s.count(".") > 1:
            return True
        scanned += 1
        if scanned >= max_scan:
            break
    return False


def read_expression(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Lê dados de expressão de um arquivo GEO Series Matrix.

    Auto-detecta formato (.txt vs .xlsx) pela extensão.

    Returns:
        expr_text – DataFrame com valores originais em texto preservados
        expr_num  – DataFrame com valores numéricos parseados
        gsm_cols  – lista de colunas GSM (amostras)
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return _read_expression_xlsx(path)
    return _read_expression_txt(path)


def _read_expression_txt(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Lê tabela de expressão de um Series Matrix .txt (tab-delimited)."""
    enc = detect_encoding(path)

    header_idx = None
    with open(path, "r", encoding=enc, errors="ignore") as fh:
        for i, line in enumerate(fh):
            if "ID_REF" in line:
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("❌ Header containing 'ID_REF' not found in file")

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_idx,
        header=0,
        dtype=str,
        encoding=enc,
        engine="python",
    )

    df = df[df.iloc[:, 0].notna()]
    df = df[~df.iloc[:, 0].astype(str).str.startswith("!")]

    first_col = df.columns[0]
    df.rename(columns={first_col: "Probe_ID"}, inplace=True)
    df["Probe_ID"] = df["Probe_ID"].astype(str).str.strip().str.strip('"')

    gsm_cols_raw = [c for c in df.columns if re.match(r'^"?GSM\d+', str(c))]
    rename_map = {c: c.strip('"') for c in gsm_cols_raw}
    df.rename(columns=rename_map, inplace=True)
    gsm_cols = [rename_map[c] for c in gsm_cols_raw]

    expr_text = df[["Probe_ID"] + gsm_cols].copy()
    expr_num = expr_text.copy()

    sample_vals = expr_text[gsm_cols].head(20).values.flatten().tolist()
    broken = _detect_broken_decimal(sample_vals)
    if broken:
        log.warning(
            "Locale-broken numeric format detected — applying digit-by-digit "
            "reconstruction (first digit = integer part)"
        )

    for c in gsm_cols:
        expr_num[c] = expr_num[c].apply(lambda v: smart_float(v, broken_decimal=broken))

    log.info(f"Expression loaded: {expr_num.shape[0]} probes × {len(gsm_cols)} samples")
    return expr_text, expr_num, gsm_cols


def _read_expression_xlsx(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Lê tabela de expressão de um Series Matrix .xlsx."""
    log.warning(
        "⚠️  Reading from .xlsx — numeric values may have locale issues. "
        "Prefer the original .txt Series Matrix from GEO."
    )

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    header_row = None
    rows_data: List[tuple] = []
    in_table = False

    for row in ws.iter_rows(values_only=True):
        first = str(row[0]) if row[0] else ""
        if "series_matrix_table_begin" in first.lower():
            in_table = True
            continue
        if "series_matrix_table_end" in first.lower():
            break
        if in_table:
            if header_row is None:
                header_row = [str(v).strip().strip('"') if v else "" for v in row]
            else:
                rows_data.append(row)

    wb.close()

    if header_row is None:
        raise ValueError("Expression table not found in xlsx")

    df = pd.DataFrame(rows_data, columns=header_row)
    df.rename(columns={df.columns[0]: "Probe_ID"}, inplace=True)
    df["Probe_ID"] = df["Probe_ID"].astype(str).str.strip().str.strip('"')

    gsm_cols = [c for c in df.columns if re.match(r"^GSM\d+", str(c))]

    expr_text = df[["Probe_ID"] + gsm_cols].copy()
    expr_num = expr_text.copy()

    sample_vals = expr_text[gsm_cols].head(20).values.flatten().tolist()
    broken = _detect_broken_decimal(sample_vals)
    if broken:
        log.warning(
            "Locale-broken numeric format detected in xlsx — applying "
            "digit-by-digit reconstruction"
        )

    for c in gsm_cols:
        expr_num[c] = expr_num[c].apply(lambda v: smart_float(v, broken_decimal=broken))

    log.info(f"Expression (xlsx): {expr_num.shape[0]} probes × {len(gsm_cols)} samples")
    return expr_text, expr_num, gsm_cols
