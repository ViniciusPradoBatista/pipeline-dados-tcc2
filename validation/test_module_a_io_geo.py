"""
Módulo A — io_geo.py — Validação técnica.
Cada teste reporta PASSOU/FALHOU com evidência (file:line, valor encontrado).
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

from geo_pipeline.io_geo import detect_encoding, parse_series_metadata_tabular  # noqa: E402


REPO = Path(__file__).parent.parent
DATA = REPO / "TCC2" / "data"
GSE85589 = str(DATA / "GSE85589_series_matrix.txt")
GSE59856 = str(DATA / "GSE59856_series_matrix (1).txt")


def report(name: str, passed: bool, details: str) -> None:
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── A.1: detect_encoding com UTF-8 ─────────────────────────────────
def test_a1_utf8():
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "wb") as f:
        f.write("!Sample_title\t\"hello\"\nID_REF\tGSM1\n".encode("utf-8"))
    enc = detect_encoding(path)
    os.unlink(path)
    passed = enc.lower().startswith(("utf", "ascii"))
    report(
        "A.1 detect_encoding em arquivo UTF-8",
        passed,
        f"Encoding retornado: {enc!r} (esperado utf-8 ou ascii). "
        f"io_geo.py:detect_encoding linha 14",
    )


# ─── A.2: detect_encoding com Latin-1 ───────────────────────────────
def test_a2_latin1():
    fd, path = tempfile.mkstemp(suffix=".txt")
    # Caracteres tipicamente latin-1 (não-UTF-8): "café" em latin-1
    text = "!Sample_title\t\"caf\xe9\"\nID_REF\tGSM1\n"
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("latin-1"))
    enc = detect_encoding(path)
    os.unlink(path)
    # chardet pode reportar "ISO-8859-1", "Windows-1252", "Latin-1" etc.
    passed = enc is not None and "utf" not in enc.lower()
    report(
        "A.2 detect_encoding em arquivo Latin-1",
        passed,
        f"Encoding retornado: {enc!r} (esperado uma família ISO-8859/Windows-12xx, NÃO utf-8). "
        f"io_geo.py:detect_encoding linha 14",
    )


# ─── A.3: parser separa metadados de tabular ────────────────────────
def test_a3_metadata_separation():
    """Metadados vão até !series_matrix_table_begin; tabular vem depois."""
    meta = parse_series_metadata_tabular(GSE85589)
    cols = list(meta.columns)
    # Esperado: pelo menos !Sample_geo_accession, !Sample_title, !Sample_platform_id
    has_accession = any("geo_accession" in c.lower() for c in cols)
    has_platform = any("platform" in c.lower() for c in cols)
    n_samples = meta.shape[0]
    passed = has_accession and has_platform and 0 < n_samples < 1000
    report(
        "A.3 parse_series_metadata_tabular separa metadados de tabular",
        passed,
        f"Metadados parseados: {meta.shape} (samples × fields). "
        f"has_geo_accession_col={has_accession}, has_platform_col={has_platform}. "
        f"io_geo.py:parse_series_metadata_tabular linha 28",
    )


# ─── A.4: localização de ID_REF (não depende de posição fixa) ───────
def test_a4_id_ref_robust():
    """O parser de expressão (em expression.py) varre o arquivo procurando ID_REF.

    Verificamos comportamento real: ID_REF aparece em linhas variadas dependendo
    do número de !Sample_* lines no header.
    """
    from geo_pipeline.expression import _read_expression_txt

    expr_text, expr_num, gsm_cols = _read_expression_txt(GSE85589)
    # Sucesso = encontrou e leu corretamente
    passed = expr_text.shape[0] > 100 and len(gsm_cols) > 0
    report(
        "A.4 ID_REF localizado robustamente (varredura linha-por-linha)",
        passed,
        f"_read_expression_txt: {expr_text.shape[0]} probes × {len(gsm_cols)} GSMs lidos. "
        f"expression.py:_read_expression_txt linhas 51-65 (busca scan no `for i, line in enumerate(fh)`)",
    )


# ─── A.5: separador não-tab (ex: múltiplos espaços) ─────────────────
def test_a5_separator_robustness():
    """O parser de tabular é hardcoded para sep='\\t'. O que acontece com espaços?"""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("!Sample_geo_accession\t\"GSM1\"\t\"GSM2\"\n")
        f.write("!series_matrix_table_begin\n")
        f.write("ID_REF    GSM1    GSM2\n")  # 4 espaços em vez de TAB
        f.write("probe1    1.5    2.3\n")
        f.write("!series_matrix_table_end\n")
    from geo_pipeline.expression import _read_expression_txt
    error_msg = None
    n_probes = -1
    n_gsms = -1
    try:
        expr_text, expr_num, gsm_cols = _read_expression_txt(path)
        n_probes = expr_text.shape[0]
        n_gsms = len(gsm_cols)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
    os.unlink(path)
    # Comportamento esperado: ou parseia corretamente ou levanta erro descritivo.
    # Comportamento real: provavelmente lê 1 coluna gigante (sem split).
    if error_msg:
        verdict = f"Levanta exceção: {error_msg}"
        passed = "ID_REF" in str(error_msg) or "header" in str(error_msg).lower()
    else:
        verdict = f"Lido como {n_probes} probes × {n_gsms} GSMs (provavelmente 0 GSMs = falha silenciosa)"
        passed = n_gsms > 0  # se conseguiu separar, ok; senão falhou silenciosamente
    report(
        "A.5 Comportamento com separador de espaços (esperado: tab)",
        passed,
        f"{verdict}. expression.py:_read_expression_txt linha 49 (`sep='\\t'` hardcoded)",
    )


# ─── A.6: arquivo inexistente ───────────────────────────────────────
def test_a6_missing_file():
    bogus = "/nao/existe/arquivo_bogus.txt"
    error_type = None
    error_msg = None
    try:
        parse_series_metadata_tabular(bogus)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
    descriptive = error_type == "FileNotFoundError" or "no such file" in (error_msg or "").lower()
    report(
        "A.6 Erro descritivo para arquivo inexistente",
        descriptive,
        f"Exceção: {error_type}: {error_msg!r}. "
        f"Esperado FileNotFoundError com nome do arquivo. "
        f"io_geo.py:detect_encoding linha 16 (`open(file_path, 'rb')`) — propaga FileNotFoundError nativo",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO A — io_geo.py")
    print("=" * 70)
    test_a1_utf8()
    test_a2_latin1()
    test_a3_metadata_separation()
    test_a4_id_ref_robust()
    test_a5_separator_robustness()
    test_a6_missing_file()
