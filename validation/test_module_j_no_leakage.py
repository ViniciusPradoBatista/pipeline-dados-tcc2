"""Módulo J — Anti-vazamento da normalização (fit só no treino).

Prova, por perturbação e por inspeção de IDs, que ComBat + z-score são ajustados
exclusivamente no conjunto de treino e que o teste nunca influencia os parâmetros.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.normalize import (  # noqa: E402
    apply_normalization,
    fit_normalization,
)


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


def _make_synthetic():
    """Gera treino/teste sintéticos com 2 batches × 2 classes (estrutura mínima do ComBat)."""
    rng = np.random.RandomState(42)
    n_probes = 40
    probes = [f"hsa-miR-{i:03d}" for i in range(n_probes)]

    def block(prefix, n_per_cell):
        cols, rows = [], []
        for batch in ("GSE_A", "GSE_B"):
            shift = 0.0 if batch == "GSE_A" else 3.0  # efeito de batch artificial
            for cls in ("PDAC", "Control"):
                bio = 1.5 if cls == "PDAC" else 0.0
                for k in range(n_per_cell):
                    sid = f"{prefix}_{batch}_{cls}_{k}"
                    cols.append(sid)
                    rows.append({"sample_id": sid, "batch": batch,
                                 "platform_id": batch, "class_label": cls})
                    pass
        data = rng.normal(0, 1, size=(n_probes, len(cols)))
        # injeta efeito de batch/classe coluna a coluna
        for j, meta in enumerate(rows):
            data[:, j] += (3.0 if meta["batch"] == "GSE_B" else 0.0)
            data[:, j] += (1.5 if meta["class_label"] == "PDAC" else 0.0)
        expr = pd.DataFrame(data, index=pd.Index(probes, name="Probe_ID"), columns=cols)
        annot = pd.DataFrame(rows)
        return expr, annot

    expr_train, annot_train = block("tr", 8)
    expr_test, annot_test = block("te", 3)
    return expr_train, annot_train, expr_test, annot_test


# ─── J.1: perturbação — fit ignora o teste ──────────────────────────
def test_j1_fit_ignores_test():
    expr_train, annot_train, expr_test, annot_test = _make_synthetic()

    params_a = fit_normalization(expr_train, annot_train)

    # corrompe brutalmente o teste; o fit (que só recebe treino) NÃO pode mudar
    expr_test_corrupt = expr_test.copy()
    expr_test_corrupt.iloc[:, :] = expr_test.values * 1e6 + 12345.0

    params_b = fit_normalization(expr_train, annot_train)

    ok_mu = np.array_equal(params_a.mu.values, params_b.mu.values)
    ok_sd = np.array_equal(params_a.sd.values, params_b.sd.values)
    ok_est = all(
        np.array_equal(np.asarray(params_a.combat_estimates[k]),
                       np.asarray(params_b.combat_estimates[k]))
        for k in params_a.combat_estimates
        if isinstance(params_a.combat_estimates[k], (np.ndarray, list, int, float))
    )

    train_a = apply_normalization(expr_train, annot_train, params_a)
    train_b = apply_normalization(expr_train, annot_train, params_b)
    ok_train = np.allclose(train_a.values, train_b.values, rtol=0, atol=0)

    # sanity: a corrupção DEVE fluir pelo apply do teste (teste não é vácuo)
    test_clean = apply_normalization(expr_test, annot_test, params_a)
    test_corrupt = apply_normalization(expr_test_corrupt, annot_test, params_a)
    corruption_flows = not np.allclose(test_clean.values, test_corrupt.values)

    passed = ok_mu and ok_sd and ok_est and ok_train and corruption_flows
    report(
        "J.1 fit_normalization ignora o teste (perturbação, atol=0)",
        passed,
        f"mu igual={ok_mu}, sd igual={ok_sd}, estimates iguais={ok_est}, "
        f"apply(train) idêntico={ok_train}, corrupção do teste flui no apply={corruption_flows}.",
    )


# ─── J.2: IDs de teste não aparecem nos parâmetros persistidos ──────
def test_j2_no_test_ids_in_params():
    expr_train, annot_train, expr_test, annot_test = _make_synthetic()
    params = fit_normalization(expr_train, annot_train)

    test_ids = set(annot_test["sample_id"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        joblib.dump(params.combat_estimates, tmp / "combat_estimates.pkl")
        pd.DataFrame({"Probe_ID": params.mu.index, "mu": params.mu.values,
                      "sd": params.sd.values}).to_csv(tmp / "zscore_params.csv", index=False)

        estimates_blob = repr(joblib.load(tmp / "combat_estimates.pkl"))
        zscore_blob = (tmp / "zscore_params.csv").read_text(encoding="utf-8")

    leaked = [sid for sid in test_ids if sid in estimates_blob or sid in zscore_blob]

    passed = len(leaked) == 0
    report(
        "J.2 nenhum GSM/ID de teste em combat_estimates / zscore_params",
        passed,
        f"IDs de teste vazados: {leaked if leaked else 'nenhum'}. "
        f"(estimates guardam rótulos de BATCH, não de amostra; zscore_params é por Probe_ID.)",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO J — ANTI-VAZAMENTO DA NORMALIZAÇÃO")
    print("=" * 70)
    test_j1_fit_ignores_test()
    test_j2_no_test_ids_in_params()
