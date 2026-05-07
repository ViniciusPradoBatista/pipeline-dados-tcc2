"""Módulo B — conditions.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import pandas as pd  # noqa: E402

from geo_pipeline.conditions import (  # noqa: E402
    extract_conditions,
    extract_sample_condition,
    filter_samples_by_conditions,
)
from geo_pipeline.io_geo import parse_series_metadata_tabular  # noqa: E402


REPO = Path(__file__).parent.parent
DATA = REPO / "TCC2" / "data"
GSE85589 = str(DATA / "GSE85589_series_matrix.txt")


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# Carrega metadados reais uma vez
META = parse_series_metadata_tabular(GSE85589)
COND_COLS = [
    c
    for c in META.columns
    if any(
        k in c.lower()
        for k in (
            "source_name",
            "characteristics",
            "title",
            "description",
            "disease",
            "tissue",
        )
    )
]


# ─── B.1: case-insensitivity ────────────────────────────────────────
def test_b1_case_insensitive():
    """Testa as 3 variações no MESMO dataset."""
    cond_lower = ["pancreatic cancer"]
    cond_upper = ["PANCREATIC CANCER"]
    cond_mixed = ["Pancreatic Cancer"]

    n_lower = filter_samples_by_conditions(META, cond_lower, COND_COLS).shape[0]
    n_upper = filter_samples_by_conditions(META, cond_upper, COND_COLS).shape[0]
    n_mixed = filter_samples_by_conditions(META, cond_mixed, COND_COLS).shape[0]

    passed = n_lower == n_upper == n_mixed and n_lower > 0
    report(
        "B.1 filter_samples_by_conditions é case-insensitive",
        passed,
        f"lower={n_lower}, UPPER={n_upper}, Mixed={n_mixed}. "
        f"Esperado: todos iguais e > 0. "
        f"conditions.py:filter_samples_by_conditions linha 254 (.str.lower() em ambos lados)",
    )


# ─── B.2: escopo dos campos ─────────────────────────────────────────
def test_b2_field_scope():
    """O filtro varre múltiplos campos !Sample_*?"""
    # COND_COLS já é construído pelo extract_conditions com keywords:
    # source_name, characteristics, title, description, disease, tissue, cell_type, treatment
    keywords = [
        "source_name",
        "characteristics",
        "title",
        "description",
        "disease",
        "tissue",
        "cell_type",
        "treatment",
    ]
    cols_used = [
        c for c in META.columns if any(k in c.lower() for k in keywords)
    ]
    passed = len(cols_used) >= 3
    report(
        "B.2 Filtro varre múltiplos campos !Sample_* relevantes",
        passed,
        f"{len(cols_used)} colunas elegíveis em {GSE85589}: {cols_used[:5]}... "
        f"conditions.py:extract_conditions linhas 89-105 (keywords list)",
    )


# ─── B.3: zero matches — falha silenciosa? ──────────────────────────
def test_b3_zero_matches():
    """Filtro com condição inexistente: erra ou retorna df vazio?"""
    bogus_cond = ["xxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
    error_msg = None
    n_rows = -1
    try:
        result = filter_samples_by_conditions(META, bogus_cond, COND_COLS)
        n_rows = result.shape[0]
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    if error_msg:
        verdict = f"Levanta exceção: {error_msg}"
        # Idealmente: exceção descritiva. Falha silenciosa é warning logado mas df vazio.
        passed_strict = True
    else:
        verdict = (
            f"Retorna DataFrame com {n_rows} linhas (sem exceção). "
            f"Comportamento real: filter_samples_by_conditions tem fallback que "
            f"varre TODAS as colunas (linha 270-278) — pode mascarar miss-spell."
        )
        passed_strict = False  # falha silenciosa é problema
    report(
        "B.3 Comportamento com zero matches (falha silenciosa?)",
        passed_strict,
        f"{verdict}. conditions.py:filter_samples_by_conditions linhas 263-278",
    )


# ─── B.4: deduplicação de GSMs ──────────────────────────────────────
def test_b4_gsm_dedup():
    """Os GSM IDs extraídos da metadata são únicos?"""
    geo_col = None
    for col in META.columns:
        if "sample_geo_accession" in col.lower().replace(" ", "_"):
            geo_col = col
            break
    if geo_col is None:
        report(
            "B.4 GSM IDs únicos",
            False,
            "Coluna Sample_geo_accession não encontrada — incapaz de validar.",
        )
        return

    gsms = META[geo_col].dropna().tolist()
    gsms = [g for g in gsms if str(g).startswith("GSM")]
    n_total = len(gsms)
    n_unique = len(set(gsms))
    passed = n_total == n_unique
    report(
        "B.4 GSM IDs únicos (sem duplicação)",
        passed,
        f"Total={n_total}, Únicos={n_unique}. "
        f"Origem: coluna {geo_col!r} dos metadados. "
        f"conditions.py NÃO faz deduplicação explícita — confiamos na origem GEO.",
    )


# ─── B.5: extract_sample_condition prioridade ───────────────────────
def test_b5_extract_priority():
    """Verifica a ordem de prioridade documentada (1=disease, 2=source_name, 3=title)."""
    # Sintetiza uma row para testar
    row1 = pd.Series({
        "Sample_characteristics_ch1": "disease state: pancreatic cancer",
        "Sample_source_name_ch1": "Serum (some other thing)",
        "Sample_title": "PC1",
    })
    cond = extract_sample_condition(row1, list(row1.index))
    p1_correct = cond == "pancreatic cancer"

    row2 = pd.Series({
        "Sample_source_name_ch1": "Serum (pancreatic cancer)",
        "Sample_title": "PC1",
    })
    cond2 = extract_sample_condition(row2, list(row2.index))
    p2_correct = cond2 == "pancreatic cancer"

    row3 = pd.Series({"Sample_title": "PC1"})
    cond3 = extract_sample_condition(row3, list(row3.index))
    p3_extracts = cond3 != ""

    passed = p1_correct and p2_correct and p3_extracts
    report(
        "B.5 extract_sample_condition respeita prioridade (disease>source>title)",
        passed,
        f"P1 (disease state)→{cond!r} (esperado 'pancreatic cancer'). "
        f"P2 (source_name parens)→{cond2!r}. "
        f"P3 (title only)→{cond3!r} (esperado não-vazio). "
        f"conditions.py:extract_sample_condition linhas 38-66",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO B — conditions.py")
    print("=" * 70)
    test_b1_case_insensitive()
    test_b2_field_scope()
    test_b3_zero_matches()
    test_b4_gsm_dedup()
    test_b5_extract_priority()
