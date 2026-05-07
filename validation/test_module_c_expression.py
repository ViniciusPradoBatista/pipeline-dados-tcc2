"""Módulo C — expression.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import numpy as np  # noqa: E402

from geo_pipeline.expression import _read_expression_txt, read_expression  # noqa: E402
from geo_pipeline.io_geo import parse_series_metadata_tabular  # noqa: E402


REPO = Path(__file__).parent.parent
DATA = REPO / "TCC2" / "data"
GSE85589 = str(DATA / "GSE85589_series_matrix.txt")
GSE59856 = str(DATA / "GSE59856_series_matrix (1).txt")


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── C.1: estrutura de índice (txt) ─────────────────────────────────
def test_c1_structure_txt():
    expr_text, expr_num, gsm_cols = _read_expression_txt(GSE85589)
    # Estrutura: primeira coluna é "Probe_ID", restantes são GSM*
    first_col = expr_num.columns[0]
    other_cols_are_gsm = all(c.startswith("GSM") for c in expr_num.columns[1:])
    passed = first_col == "Probe_ID" and other_cols_are_gsm and len(gsm_cols) > 0
    report(
        "C.1 Estrutura .txt: 1ª col = Probe_ID, demais = GSM*",
        passed,
        f"first_col={first_col!r}, all_others_GSM={other_cols_are_gsm}, "
        f"shape={expr_num.shape}, n_gsm_cols={len(gsm_cols)}. "
        f"expression.py:_read_expression_txt linhas 78-82 (rename + filter)",
    )


# ─── C.2: alinhamento expressão↔metadados ───────────────────────────
def test_c2_metadata_alignment():
    """Os GSMs da matriz de expressão estão TODOS presentes nos metadados?"""
    meta = parse_series_metadata_tabular(GSE85589)
    expr_text, expr_num, gsm_cols = _read_expression_txt(GSE85589)

    geo_col = None
    for col in meta.columns:
        if "sample_geo_accession" in col.lower().replace(" ", "_"):
            geo_col = col
            break
    if geo_col is None:
        report("C.2 Alinhamento", False, "coluna geo_accession não encontrada")
        return

    meta_gsms = set(meta[geo_col].dropna().astype(str))
    expr_gsms = set(gsm_cols)
    in_expr_only = expr_gsms - meta_gsms
    in_meta_only = meta_gsms - expr_gsms
    passed = len(in_expr_only) == 0
    report(
        "C.2 GSMs da expressão estão TODOS nos metadados",
        passed,
        f"expr_gsms={len(expr_gsms)}, meta_gsms={len(meta_gsms)}, "
        f"in_expr_only={len(in_expr_only)} (devem ser 0), "
        f"in_meta_only={len(in_meta_only)}. "
        f"NOTA: o pipeline NÃO faz validação explícita — alinhamento é resolvido em "
        f"dataset.py:process_single_dataset 'Cross-referencing' linhas 87-95",
    )


# ─── C.3: tratamento de não-numéricos ───────────────────────────────
def test_c3_non_numeric_handling():
    """smart_float é aplicado APÓS leitura como str (dtype=str em read_csv).

    Verifica: NaN final na matriz (não strings residuais).
    """
    expr_text, expr_num, gsm_cols = _read_expression_txt(GSE85589)
    # expr_num deve ter apenas float em colunas GSM
    dtypes = expr_num[gsm_cols].dtypes.unique()
    all_float = all("float" in str(d).lower() for d in dtypes)

    nan_count = expr_num[gsm_cols].isna().sum().sum()
    total = expr_num[gsm_cols].size
    passed = all_float
    report(
        "C.3 Não-numéricos viram NaN (não strings)",
        passed,
        f"dtypes em colunas GSM: {[str(d) for d in dtypes]} (esperado só float). "
        f"NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%). "
        f"expression.py:_read_expression_txt linhas 49 (dtype=str na leitura) → "
        f"linha 84-85 (smart_float aplicado depois)",
    )


# ─── C.4: txt vs xlsx mesma estrutura (não temos xlsx para testar) ──
def test_c4_txt_xlsx_consistency():
    """Não temos xlsx no projeto. Inspeciona o código pra confirmar
    que ambos retornam mesma assinatura."""
    import inspect

    from geo_pipeline.expression import _read_expression_txt, _read_expression_xlsx

    sig_txt = inspect.signature(_read_expression_txt)
    sig_xlsx = inspect.signature(_read_expression_xlsx)
    same_signature = (
        sig_txt.return_annotation == sig_xlsx.return_annotation
        and len(sig_txt.parameters) == len(sig_xlsx.parameters)
    )
    report(
        "C.4 Assinatura txt e xlsx idêntica (Tuple[expr_text, expr_num, gsm_cols])",
        same_signature,
        f"txt return: {sig_txt.return_annotation}, "
        f"xlsx return: {sig_xlsx.return_annotation}. "
        f"NOTA: não há .xlsx em data/ para teste empírico. "
        f"expression.py linhas 47 e 95",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO C — expression.py")
    print("=" * 70)
    test_c1_structure_txt()
    test_c2_metadata_alignment()
    test_c3_non_numeric_handling()
    test_c4_txt_xlsx_consistency()
