"""Módulo F — normalize.py (API fit/apply, sem vazamento) — Validação técnica.

As funções legadas apply_combat/zscore_by_probe (que ajustavam sobre a matriz
inteira) foram REMOVIDAS. Este módulo valida a API fit/apply que estima parâmetros
só no treino.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.normalize import (  # noqa: E402
    apply_zscore,
    combat_apply,
    combat_fit,
    fit_zscore,
)


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


def _expr(n_probes, cols, seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=5.0, scale=2.0, size=(n_probes, len(cols)))
    return pd.DataFrame(data, index=pd.Index([f"p{i}" for i in range(n_probes)],
                                             name="Probe_ID"), columns=cols)


# ─── F.0: as funções legadas que vazavam NÃO existem mais ────────────
def test_f0_legacy_removed():
    import geo_pipeline.normalize as nz
    has_legacy = hasattr(nz, "apply_combat") or hasattr(nz, "zscore_by_probe")
    report(
        "F.0 funções legadas (apply_combat/zscore_by_probe) removidas",
        not has_legacy,
        f"apply_combat presente={hasattr(nz, 'apply_combat')}, "
        f"zscore_by_probe presente={hasattr(nz, 'zscore_by_probe')}. "
        f"Ambas devem estar ausentes — vaziam ao ajustar sobre a matriz inteira.",
    )


# ─── F.1: fit_zscore/apply_zscore normalizam por probe (LINHA) ───────
def test_f1_zscore_axis():
    """fit_zscore calcula mu/sd por probe (linha) ao longo das amostras de treino;
    apply_zscore reaplica → cada probe fica com mean≈0, std≈1 no treino."""
    train = _expr(10, [f"GSM{i}" for i in range(5)], seed=42)

    mu, sd = fit_zscore(train)
    z = apply_zscore(train, mu, sd).values

    row_means, row_stds = z.mean(axis=1), z.std(axis=1, ddof=1)
    col_means, col_stds = z.mean(axis=0), z.std(axis=0, ddof=1)

    by_row = np.allclose(row_means, 0, atol=1e-9) and np.allclose(row_stds, 1, atol=0.01)
    by_col = np.allclose(col_means, 0, atol=1e-9) and np.allclose(col_stds, 1, atol=0.01)

    # mu/sd devem ser Series indexadas por Probe_ID (reaplicação por nome, não posição)
    indexed_by_probe = list(mu.index) == list(train.index) and mu.index.name == "Probe_ID"

    passed = by_row and not by_col and indexed_by_probe
    report(
        "F.1 fit_zscore/apply_zscore normalizam por LINHA (probe), mu/sd por Probe_ID",
        passed,
        f"by_row={by_row}, by_col={by_col}, indexado_por_Probe_ID={indexed_by_probe}. "
        f"z-score por PROBE é a convenção em ComBat workflows (cada miRNA re-centrado "
        f"na própria média), com mu/sd aprendidos só no treino.",
    )


# ─── F.2: ComBat usa batch_col='batch' e class_col='class_label' ─────
def test_f2_combat_batch():
    sig = inspect.signature(combat_fit)
    default_batch = sig.parameters["batch_col"].default
    default_class = sig.parameters["class_col"].default

    src = inspect.getsource(combat_fit)
    uses_categorical = "categorical_cols=[class_col]" in src.replace(" ", "")

    passed = default_batch == "batch" and default_class == "class_label" and uses_categorical
    report(
        "F.2 combat_fit usa batch_col='batch', class_col='class_label' (mod=classe)",
        passed,
        f"default batch_col={default_batch!r}, class_col={default_class!r}, "
        f"categorical_cols=[class_col]={uses_categorical}. "
        f"O conteúdo de 'batch' é o dataset_id (features.py:build_sample_annotation).",
    )


# ─── F.3: ComBat fit no treino + apply no teste (batch presente) ─────
def test_f3_combat_fit_apply():
    """Fit no treino (2 batches) e apply no teste cujo batch existe no treino.
    Saída do teste deve ter shape correto e sem NaN."""
    cols_tr = [f"GSM{i}" for i in range(10)]
    cols_te = [f"GSM{i}" for i in range(10, 16)]
    train = _expr(50, cols_tr, seed=0)
    test = _expr(50, cols_te, seed=1)

    annot_tr = pd.DataFrame({
        "sample_id": cols_tr,
        "batch": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "class_label": ["PDAC", "Control"] * 5,
    })
    annot_te = pd.DataFrame({
        "sample_id": cols_te,
        "batch": ["A", "A", "A", "B", "B", "B"],  # batches já vistos no treino
        "class_label": ["PDAC", "Control", "PDAC", "Control", "PDAC", "Control"],
    })

    error_msg = None
    try:
        train_corr, estimates, train_med = combat_fit(train, annot_tr)
        test_corr = combat_apply(test, annot_te, train_corr.index, estimates, train_med)
        ok_shape = test_corr.shape == (train_corr.shape[0], len(cols_te))
        has_nan = bool(np.isnan(test_corr.values).any())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        ok_shape = has_nan = False

    passed = error_msg is None and ok_shape and not has_nan
    report(
        "F.3 combat_fit(treino) + combat_apply(teste) — batch visto no treino",
        passed,
        (f"FALHA: {error_msg}" if error_msg else
         f"shape teste OK={ok_shape}, has_nan={has_nan}. "
         f"Teste harmonizado via neuroCombatFromTraining com estimativas do treino."),
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO F — normalize.py (fit/apply sem vazamento)")
    print("=" * 70)
    test_f0_legacy_removed()
    test_f1_zscore_axis()
    test_f2_combat_batch()
    test_f3_combat_fit_apply()
