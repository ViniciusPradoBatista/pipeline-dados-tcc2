"""Leitura de baixo nível de arquivos GEO Series Matrix: encoding e helpers."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import chardet
import openpyxl
import pandas as pd

log = logging.getLogger("geo_pipeline")


def detect_encoding(file_path: str) -> str:
    """Detecta o encoding do arquivo via chardet."""
    with open(file_path, "rb") as fh:
        raw = fh.read(2_000_000)
    result = chardet.detect(raw)
    enc = result.get("encoding") or "latin-1"
    conf = result.get("confidence", "?")
    log.info(f"Encoding detected: {enc} (confidence: {conf})")
    return enc


def parse_series_metadata_tabular(path: str) -> pd.DataFrame:
    """
    Parseia metadados (sample-level) de um arquivo GEO Series Matrix.

    Trata tanto .txt (tab-delimited) quanto .xlsx.
    Corrige o bug de chave-duplicada em que múltiplas linhas
    !Sample_characteristics_ch1 sobrescreviam umas às outras.
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return _parse_metadata_xlsx(path)
    return _parse_metadata_txt(path)


def _make_keys_unique(data: List[Tuple[str, list]]) -> List[Tuple[str, list]]:
    """Garante chaves únicas anexando _2, _3, ... (corrige bug de characteristics_ch1)."""
    seen: Dict[str, int] = {}
    unique_data: List[Tuple[str, list]] = []
    for key, vals in data:
        if key in seen:
            seen[key] += 1
            unique_key = f"{key}_{seen[key]}"
        else:
            seen[key] = 1
            unique_key = key
        unique_data.append((unique_key, vals))
    return unique_data


def _build_metadata_df(data: List[Tuple[str, list]]) -> pd.DataFrame:
    """Converte lista (key, values) em DataFrame samples × fields."""
    if not data:
        log.warning("No metadata key/value pairs found")
        return pd.DataFrame()

    data = _make_keys_unique(data)
    max_cols = max(len(v) for _, v in data)

    df = pd.DataFrame({k: v + [""] * (max_cols - len(v)) for k, v in data}).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Field"}, inplace=True)
    df = df.set_index("Field").T.reset_index(drop=True)

    log.info(f"Metadata parsed: {df.shape[0]} samples × {df.shape[1]} fields")
    return df


def _parse_metadata_txt(path: str) -> pd.DataFrame:
    """Parseia metadados de um Series Matrix .txt."""
    enc = detect_encoding(path)

    meta_lines: List[str] = []
    with open(path, "r", encoding=enc, errors="ignore") as fh:
        for line in fh:
            if line.lower().startswith("!series_matrix_table_begin"):
                break
            if line.strip().startswith("!"):
                meta_lines.append(line.strip())

    data: List[Tuple[str, list]] = []
    for raw_line in meta_lines:
        parts = raw_line.split("\t")
        if len(parts) > 1:
            key = parts[0].lstrip("!").strip()
            vals = [v.strip().strip('"') for v in parts[1:] if v.strip()]
            data.append((key, vals))

    return _build_metadata_df(data)


def _parse_metadata_xlsx(path: str) -> pd.DataFrame:
    """Parseia metadados de um Series Matrix .xlsx (via openpyxl)."""

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        log.error(f"No active worksheet in {path}")
        wb.close()
        return pd.DataFrame()

    data: List[Tuple[str, list]] = []
    for row in ws.iter_rows(values_only=True):
        first_cell = str(row[0]) if row[0] else ""
        if first_cell.lower().startswith("!series_matrix_table_begin"):
            break
        if first_cell.startswith("!"):
            key = first_cell.lstrip("!").strip()
            vals = [
                str(v).strip().strip('"')
                for v in row[1:]
                if v is not None and str(v).strip()
            ]
            data.append((key, vals))

    wb.close()
    return _build_metadata_df(data)
