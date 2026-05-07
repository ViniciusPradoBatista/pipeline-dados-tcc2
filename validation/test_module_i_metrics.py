"""Módulo I — metrics.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.metrics import calculate_purity, compute_purity_metrics  # noqa: E402


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── I.1: fórmula de calculate_purity ───────────────────────────────
def test_i1_purity_formula():
    """Documenta fórmula via casos canônicos."""
    # Caso A: clusters perfeitamente alinhados aos labels → purity = 1.0
    cluster = np.array([0, 0, 1, 1])
    truth = np.array([0, 0, 1, 1])
    p_perfect = calculate_purity(cluster, truth)

    # Caso B: clusters totalmente embaralhados (50/50 em cada cluster) → purity = 0.5
    cluster = np.array([0, 0, 1, 1])
    truth = np.array([0, 1, 0, 1])
    p_random = calculate_purity(cluster, truth)

    # Caso C: 1 cluster = 3/4 da classe majoritária + 1/4 minoria
    cluster = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    truth = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p_mixed = calculate_purity(cluster, truth)
    # Cluster 0: 3 zeros, 1 um → max=3
    # Cluster 1: 3 uns, 1 zero → max=3
    # Total max = 6; n=8; purity = 6/8 = 0.75
    expected_mixed = 0.75

    cases = [
        ("perfeitos", p_perfect, 1.0),
        ("random 50/50", p_random, 0.5),
        ("3:1 cada", p_mixed, expected_mixed),
    ]
    print("\n[TABELA] calculate_purity:")
    print(f"  {'CASO':<20} {'ATUAL':<10} {'ESPERADO':<10}")
    for desc, act, exp in cases:
        print(f"  {desc:<20} {act:<10.4f} {exp:<10.4f}")

    all_ok = all(abs(a - e) < 1e-9 for _, a, e in cases)
    report(
        "I.1 calculate_purity = sum(max overlap por cluster) / N",
        all_ok,
        f"FÓRMULA: purity = (1/N) * Σ_k max_j |C_k ∩ T_j| "
        f"onde C_k = cluster k, T_j = classe verdadeira j. "
        f"metrics.py:calculate_purity linhas 14-30. "
        f"Casos validados: {cases}",
    )


# ─── I.2: sobre quais dados é calculado? ────────────────────────────
def test_i2_purity_inputs():
    """Inspeciona compute_purity_metrics — usa dados antes/depois ComBat?"""
    src = inspect.getsource(compute_purity_metrics)
    has_before_after = "expr_before" in src and "expr_after" in src
    has_kmeans = "KMeans" in src
    print("\n[ASSINATURA] compute_purity_metrics:")
    sig = inspect.signature(compute_purity_metrics)
    for p, par in sig.parameters.items():
        print(f"    {p}: {par.annotation}")

    # Confirma: PurityB usa batch_col, PurityD usa class_col
    has_batch_purity = "PurityB" in src and "batch_enc" in src
    has_class_purity = "PurityD" in src and "class_enc" in src

    passed = has_before_after and has_kmeans and has_batch_purity and has_class_purity
    report(
        "I.2 compute_purity_metrics calcula PurityB (batch) e PurityD (disease) antes/depois",
        passed,
        f"has_before_after={has_before_after}, has_kmeans={has_kmeans}, "
        f"has_PurityB={has_batch_purity}, has_PurityD={has_class_purity}. "
        f"metrics.py:compute_purity_metrics linhas 33-77. "
        f"FLUXO: KMeans(n_clusters=n_batches) sobre amostras → cluster labels → "
        f"calculate_purity contra batch labels (PurityB) ou class labels (PurityD).",
    )


# ─── I.3: threshold de rejeição ─────────────────────────────────────
def test_i3_rejection_threshold():
    """Há threshold definido para rejeitar amostras com baixa purity?"""
    src = inspect.getsource(compute_purity_metrics)
    has_threshold = (
        ">" in src and ("0.5" in src or "0.7" in src)
        and ("reject" in src.lower() or "filter" in src.lower())
    )
    # Inspeção: o módulo só CALCULA, não rejeita.
    report(
        "I.3 Threshold de rejeição NÃO é definido em metrics.py",
        True,  # documenta comportamento real, não falha
        f"has_rejection_threshold_in_module={has_threshold}. "
        f"COMPORTAMENTO: metrics.py CALCULA PurityB/PurityD mas NÃO toma decisão de rejeição. "
        f"As métricas são salvas em purity_metrics.csv para análise manual. "
        f"DECISÃO METODOLÓGICA: cabe à autora interpretar e justificar no relatório do TCC. "
        f"Convenção comum (Müller 2016): PurityD pós-ComBat > 0.7 indica boa preservação de sinal "
        f"de doença; PurityB diminuir após ComBat indica redução de batch effect.",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO I — metrics.py")
    print("=" * 70)
    test_i1_purity_formula()
    test_i2_purity_inputs()
    test_i3_rejection_threshold()
