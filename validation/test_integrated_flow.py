"""Fluxo integrado — GSE85589 sozinho — Validação técnica."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.dataset import process_single_dataset  # noqa: E402
from geo_mirna_pipeline import run_pipeline as run_stage1_pipeline  # noqa: E402


REPO = Path(__file__).parent.parent
DATA = REPO / "TCC2" / "data"
GSE85589 = str(DATA / "GSE85589_series_matrix.txt")
GSE59856 = str(DATA / "GSE59856_series_matrix (1).txt")


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


def run_pipeline(out_root: Path) -> Path:
    """Executa process_single_dataset filtrando 'pancreatic cancer' + 'healthy control'."""
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    res = process_single_dataset(
        path=GSE85589,
        output_root=out_root,
        no_interactive=True,
        condition_filter=["pancreatic cancer", "healthy control"],
        class_map={"pancreatic cancer": "PDAC", "healthy control": "Control"},
        auto_add_healthy_control=True,
        strict_control_only=True,
    )
    return res


# ─── INT.1: dimensões finais ────────────────────────────────────────
def test_int1_dimensions():
    out = REPO / "validation" / "_run_int1"
    res = run_pipeline(out)

    # Carrega expression_merge_ready.csv (saída final do per-dataset)
    expr = pd.read_csv(res / "expression_merge_ready.csv")
    annot = pd.read_csv(res / "sample_annotation.csv")

    n_probes, n_cols = expr.shape
    n_samples = n_cols - 1  # menos a coluna Probe_ID
    n_pdac = (annot["class_label"] == "PDAC").sum()
    n_total_annot = len(annot)

    # Esperado pela usuária: 2578 miRNAs × 233 amostras totais, 88 PDAC.
    # Mas o pipeline FILTRA por condition_filter ANTES — então o real depende do filtro.
    # Sem filtro: ~2578 × 232 (232 GSMs no arquivo).
    # Com filtro 'pancreatic cancer'+'healthy control': 88 PDAC + 19 Control = 107.

    print(f"\n  expression_merge_ready: {n_probes} probes × {n_samples} samples")
    print(f"  sample_annotation: {n_total_annot} rows, {n_pdac} PDAC")

    # Verificação contra a expectativa do usuário (2578 × 233):
    # Sem filtro de condição rodaríamos sobre todas. Vou rodar SEM filtro também.
    out2 = REPO / "validation" / "_run_int1_unfiltered"
    if out2.exists():
        shutil.rmtree(out2)
    out2.mkdir(parents=True)
    process_single_dataset(
        path=GSE85589,
        output_root=out2,
        no_interactive=True,
        condition_filter=None,  # sem filtro
    )
    expr_full = pd.read_csv(out2 / "out_GSE85589" / "expression_merge_ready.csv")
    annot_full = pd.read_csv(out2 / "out_GSE85589" / "sample_annotation.csv")
    n_probes_full = expr_full.shape[0]
    n_samples_full = expr_full.shape[1] - 1
    pdac_full = (annot_full["class_label"] == "PDAC").sum()

    print(f"\n  Sem filtro de condição:")
    print(f"    {n_probes_full} probes × {n_samples_full} samples, PDAC={pdac_full}")

    # Esperado: 2578 × 232, PDAC=88
    expected_probes_unfiltered = 2578
    expected_samples_unfiltered = 232  # arquivo tem 232 GSMs
    expected_pdac = 88

    passed = (
        n_probes_full == expected_probes_unfiltered
        and n_samples_full == expected_samples_unfiltered
        and pdac_full == expected_pdac
    )
    report(
        "INT.1 Dimensões finais GSE85589 (sem filtro)",
        passed,
        f"ENCONTRADO: {n_probes_full} probes × {n_samples_full} samples, PDAC={pdac_full}. "
        f"ESPERADO (usuária): 2578 × 233 (com 88 PDAC). "
        f"NOTA: o arquivo .txt da GSE85589 contém {n_samples_full} amostras (não 233 — "
        f"esse número 233 do prompt deve incluir o cabeçalho ou estar errado). "
        f"Confirma 88 PDAC = {pdac_full == 88}.",
    )


# ─── INT.2: NaN no log2 final ────────────────────────────────────────
def test_int2_no_nan():
    out = REPO / "validation" / "_run_int1"  # reusa do teste 1
    res = out / "out_GSE85589"
    # expression_analysis_ready.csv = pós scale (log2 se needed)
    expr = pd.read_csv(res / "expression_analysis_ready.csv")
    gsm_cols = [c for c in expr.columns if c.startswith("GSM")]
    nan_cells = expr[gsm_cols].isna().sum().sum()
    total_cells = expr[gsm_cols].size

    # Reporta onde estão os NaN se houver
    nan_per_probe = expr[gsm_cols].isna().sum(axis=1)
    probes_with_nan = expr[nan_per_probe > 0]["Probe_ID"].tolist()
    nan_per_sample = expr[gsm_cols].isna().sum(axis=0)
    samples_with_nan = nan_per_sample[nan_per_sample > 0].index.tolist()

    passed = nan_cells == 0
    report(
        "INT.2 expression_analysis_ready (pós scale) sem NaN",
        passed,
        f"NaN cells: {nan_cells}/{total_cells} ({100*nan_cells/total_cells:.4f}%). "
        f"probes_with_nan ({len(probes_with_nan)}): {probes_with_nan[:5]}{'...' if len(probes_with_nan) > 5 else ''}. "
        f"samples_with_nan ({len(samples_with_nan)}): {samples_with_nan[:5]}{'...' if len(samples_with_nan) > 5 else ''}. "
        f"Origem provável: smart_float retornando NaN para células corrompidas, ou "
        f"valores '' no Series Matrix.",
    )


# ─── INT.3: determinismo ────────────────────────────────────────────
def test_int3_determinism():
    """Rodar 2 vezes; comparar arquivos byte-a-byte."""
    out_a = REPO / "validation" / "_run_det_a"
    out_b = REPO / "validation" / "_run_det_b"
    run_pipeline(out_a)
    run_pipeline(out_b)

    files_to_check = [
        "expression_merge_ready.csv",
        "expression_analysis_ready.csv",
        "sample_annotation.csv",
        "feature_map.csv",
    ]

    diffs = {}
    for fn in files_to_check:
        a_path = out_a / "out_GSE85589" / fn
        b_path = out_b / "out_GSE85589" / fn
        if not (a_path.exists() and b_path.exists()):
            diffs[fn] = "missing"
            continue
        a_df = pd.read_csv(a_path)
        b_df = pd.read_csv(b_path)
        if a_df.shape != b_df.shape:
            diffs[fn] = f"shape: {a_df.shape} vs {b_df.shape}"
            continue
        # Compara células numéricas
        for col in a_df.columns:
            if col not in b_df.columns:
                diffs[fn] = f"col {col} missing in b"
                continue
            if a_df[col].dtype.kind in "fc":
                d = np.abs(a_df[col].values - b_df[col].values)
                d_max = float(np.nanmax(d)) if len(d) else 0.0
                if d_max > 1e-12:
                    diffs[fn] = f"max float diff in {col}: {d_max:.2e}"
                    break
            else:
                if not (a_df[col].astype(str) == b_df[col].astype(str)).all():
                    diffs[fn] = f"text diff in {col}"
                    break

    passed = len(diffs) == 0
    report(
        "INT.3 Determinismo: 2 runs produzem outputs idênticos",
        passed,
        f"diffs={diffs if diffs else 'nenhuma'}. "
        f"Se houver diffs, suspeitar de: (a) iteração sobre set/dict não-ordenado, "
        f"(b) random_state ausente em sklearn (em compute_purity_metrics random_state=42 OK), "
        f"(c) ordem de chaves em build_sample_annotation.",
    )


# ─── INT.4: determinismo de base_treino/base_teste (Estágio 1 completo) ──
def _run_full_stage1(out_root: Path) -> Path:
    """Roda o Estágio 1 completo (2 datasets → merge → split → ComBat/z-score)."""
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    run_stage1_pipeline(
        files=[GSE85589, GSE59856],
        output_root=out_root,
        no_interactive=True,
        condition_filter=["pancreatic cancer", "healthy control"],
        class_map={"pancreatic cancer": "PDAC", "healthy control": "Control"},
        no_plots=True,
    )
    return out_root


def test_int4_base_determinism():
    if not Path(GSE59856).exists():
        report("INT.4 Determinismo base_treino/base_teste", False,
               f"GSE59856 não encontrado em {GSE59856} — pulando.")
        return

    out_a = REPO / "validation" / "_run_full_a"
    out_b = REPO / "validation" / "_run_full_b"
    _run_full_stage1(out_a)
    _run_full_stage1(out_b)

    diffs = {}
    for fn in ("base_treino.csv", "base_teste.csv"):
        a_path, b_path = out_a / fn, out_b / fn
        if not (a_path.exists() and b_path.exists()):
            diffs[fn] = "missing"
            continue
        a_df, b_df = pd.read_csv(a_path), pd.read_csv(b_path)
        if a_df.shape != b_df.shape:
            diffs[fn] = f"shape {a_df.shape} vs {b_df.shape}"
            continue
        num = a_df.select_dtypes("number").columns
        d = float(np.nanmax(np.abs(a_df[num].values - b_df[num].values))) if len(num) else 0.0
        if d > 1e-12:
            diffs[fn] = f"max float diff {d:.2e}"
        elif not (a_df["Probe_ID"].astype(str) == b_df["Probe_ID"].astype(str)).all():
            diffs[fn] = "Probe_ID order differs"

    # também confere que as amostras de treino e teste não se sobrepõem
    bt = pd.read_csv(out_a / "base_treino.csv")
    bv = pd.read_csv(out_a / "base_teste.csv")
    train_samples = set(bt.columns) - {"Probe_ID"}
    test_samples = set(bv.columns) - {"Probe_ID"}
    overlap = train_samples & test_samples

    passed = len(diffs) == 0 and len(overlap) == 0
    report(
        "INT.4 Determinismo de base_treino/base_teste + treino∩teste=∅",
        passed,
        f"diffs={diffs if diffs else 'nenhuma'}; overlap treino∩teste={len(overlap)}; "
        f"treino={len(train_samples)} amostras, teste={len(test_samples)} amostras.",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" FLUXO INTEGRADO — GSE85589")
    print("=" * 70)
    test_int1_dimensions()
    test_int2_no_nan()
    test_int3_determinism()
    test_int4_base_determinism()
