"""Módulo F — normalize.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.normalize import apply_combat, zscore_by_probe  # noqa: E402


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── F.1: zscore por probe (linha) vs por amostra (coluna)? ─────────
def test_f1_zscore_axis():
    """O zscore_by_probe normaliza CADA probe ao longo das amostras (linhas).

    Para microarray miRNA, a convenção comum é:
    - z-score por feature (gene/probe): feature ~ N(0,1) ao longo das amostras
    - z-score por amostra: amostra ~ N(0,1) ao longo das features

    O nome `zscore_by_probe` sugere o primeiro. Verificar empiricamente.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=(10, 5))
    df = pd.DataFrame(data, columns=["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"])
    df.insert(0, "Probe_ID", [f"p{i}" for i in range(10)])

    z = zscore_by_probe(df)
    z_data = z.iloc[:, 1:].values

    # Se z-score por LINHA (probe ao longo de samples): cada linha tem mean≈0, std≈1
    row_means = z_data.mean(axis=1)
    row_stds = z_data.std(axis=1)

    # Se z-score por COLUNA (sample ao longo de probes): cada coluna tem mean≈0, std≈1
    col_means = z_data.mean(axis=0)
    col_stds = z_data.std(axis=0)

    by_row = np.allclose(row_means, 0, atol=1e-9) and np.allclose(row_stds, 1, atol=0.01)
    by_col = np.allclose(col_means, 0, atol=1e-9) and np.allclose(col_stds, 1, atol=0.01)

    is_by_probe = by_row and not by_col
    print(f"\n  por LINHA (probe): mean={row_means.mean():.4f}, std={row_stds.mean():.4f}")
    print(f"  por COLUNA (sample): mean={col_means.mean():.4f}, std={col_stds.mean():.4f}")

    # CIENTIFICAMENTE: para Müller et al. 2016 (ComBat workflow), a normalização
    # padrão é por feature (probe), porque centraliza cada miRNA na sua própria
    # média. Isso é o esperado.
    report(
        "F.1 zscore_by_probe normaliza por LINHA (probe ao longo de samples)",
        is_by_probe,
        f"by_row={by_row}, by_col={by_col}. "
        f"normalize.py:zscore_by_probe linhas 21-26 (np.nanmean/std com axis=1, keepdims=True). "
        f"DECISÃO CIENTÍFICA: para microarray miRNA, z-score por PROBE é a convenção "
        f"em ComBat workflows (Müller 2016) — cada miRNA é re-centrado em sua própria média.",
    )


# ─── F.2: batch label exato no ComBat ───────────────────────────────
def test_f2_combat_batch():
    """Inspeciona a chamada do ComBat para ver qual coluna vira batch."""
    src = inspect.getsource(apply_combat)
    print("\n[fonte de apply_combat] (trecho relevante):")
    for line in src.split("\n"):
        line_l = line.strip().lower()
        if "batch_col" in line_l or "neurocombat(" in line_l or "covars" in line_l:
            print(f"    {line}")
    # batch_col default = "batch"
    sig = inspect.signature(apply_combat)
    default_batch = sig.parameters["batch_col"].default
    default_class = sig.parameters["class_col"].default
    passed = default_batch == "batch" and default_class == "class_label"
    report(
        "F.2 ComBat usa batch_col='batch' e class_col='class_label'",
        passed,
        f"default batch_col={default_batch!r}, class_col={default_class!r}. "
        f"O CONTEÚDO da coluna 'batch' é o dataset_id (GSE85589, GSE59856) "
        f"definido em features.py:build_sample_annotation linha 92 ('batch': dataset_id). "
        f"normalize.py:apply_combat linhas 67-72 (chamada neuroCombat com covars dict)",
    )


# ─── F.3: ComBat com batch de 1 amostra ─────────────────────────────
def test_f3_combat_single_sample_batch():
    """Se um batch tiver apenas 1 amostra, ComBat falha?"""
    rng = np.random.default_rng(0)
    n_probes = 50
    expr_data = rng.uniform(2, 12, size=(n_probes, 5))
    expr = pd.DataFrame(expr_data, columns=["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"])
    expr.insert(0, "Probe_ID", [f"p{i}" for i in range(n_probes)])

    # batch tem 1 amostra apenas no batch_B
    annot = pd.DataFrame({
        "sample_id": ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"],
        "batch": ["A", "A", "A", "A", "B"],  # batch B com 1 amostra
        "class_label": ["PDAC", "PDAC", "Control", "Control", "Control"],
    })

    error_msg = None
    out = None
    try:
        out = apply_combat(expr, annot)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    if error_msg:
        verdict = f"FALHA com {error_msg}"
        passed = "batch" in error_msg.lower() or "sample" in error_msg.lower()
    else:
        # Se não falhou, verificar se output é razoável
        out_data = out.iloc[:, 1:].values
        has_nan = np.isnan(out_data).any()
        verdict = f"NÃO falhou; output shape {out.shape}, has_nan={has_nan}"
        passed = not has_nan

    report(
        "F.3 ComBat com batch de 1 amostra",
        passed,
        f"{verdict}. "
        f"normalize.py:apply_combat NÃO valida tamanho de batch antes de chamar neuroCombat. "
        f"Confiança recai na biblioteca neuroCombat (linha 67-72).",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO F — normalize.py")
    print("=" * 70)
    test_f1_zscore_axis()
    test_f2_combat_batch()
    test_f3_combat_single_sample_batch()
