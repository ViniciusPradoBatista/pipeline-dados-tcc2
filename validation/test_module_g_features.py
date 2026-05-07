"""Módulo G — features.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import inspect  # noqa: E402

import pandas as pd  # noqa: E402

from geo_pipeline.features import (  # noqa: E402
    build_feature_map,
    build_sample_annotation,
    canonicalize_probe_id,
)


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── G.1: mapeamento probe→nome biológico ───────────────────────────
def test_g1_probe_mapping():
    """canonicalize_probe_id mapeia probe IDs para nomes biológicos?"""
    # Affymetrix GPL19117: strip suffix '_st'
    # 3D-Gene GPL18941: handle vírgulas (multi-MIMAT) → primeiro
    cases = [
        ("MIMAT0000062_st", "GPL19117", "MIMAT0000062"),  # affy strip _st
        ("MIMAT0001_st", "GPL19117", "MIMAT0001"),
        ("MIMAT0001,MIMAT0002", "GPL18941", "MIMAT0001"),  # toray, primeiro
        ("MIMAT0001", "GPL18941", "MIMAT0001"),
        ("hsa-miR-21", "GPL_DESCONHECIDO", "hsa-miR-21"),  # default passthrough
    ]
    print("\n[TABELA] canonicalize_probe_id:")
    print(f"  {'INPUT':<25} {'PLATFORM':<12} {'EXPECTED':<25} {'ACTUAL':<25}")
    print("  " + "-" * 90)
    all_ok = True
    for inp, plat, expected in cases:
        actual, ambiguous = canonicalize_probe_id(inp, plat)
        ok = actual == expected
        all_ok = all_ok and ok
        print(f"  {inp:<25} {plat:<12} {expected:<25} {actual:<25}")

    report(
        "G.1 canonicalize_probe_id mapeia por regras de plataforma",
        all_ok,
        f"all_ok={all_ok}. "
        f"FONTE DO MAPEAMENTO: regras hardcoded por GPL — NÃO há mapeamento "
        f"para nomes hsa-miR-* (apenas reduz IDs MIMAT à canônica). "
        f"O probe_id do GEO já é o ID biológico (MIMAT) na maioria das plataformas. "
        f"features.py:canonicalize_probe_id linhas 18-39",
    )


# ─── G.2: anotação de amostras (PDAC vs Control) ────────────────────
def test_g2_sample_annotation():
    """build_sample_annotation atribui class_label corretamente?"""
    meta = pd.DataFrame({
        "Sample_geo_accession": ["GSM1", "GSM2", "GSM3", "GSM4"],
        "Sample_characteristics_ch1": [
            "disease state: pancreatic cancer",
            "disease state: pancreatic cancer",
            "disease state: healthy control",
            "disease state: normal",
        ],
    })
    gsm_cols = ["GSM1", "GSM2", "GSM3", "GSM4"]
    annot = build_sample_annotation(
        meta_df=meta,
        gsm_cols=gsm_cols,
        dataset_id="GSE_test",
        platform_id="GPL19117",
        platform_name="Affy",
    )
    expected_classes = ["PDAC", "PDAC", "Control", "Control"]
    actual_classes = annot["class_label"].tolist()
    passed = actual_classes == expected_classes
    print(f"\n  Mapeamento default em build_sample_annotation:")
    print(f"    'pancreatic cancer' → 'PDAC'")
    print(f"    'pdac' → 'PDAC'")
    print(f"    'healthy control' → 'Control'")
    print(f"    'normal' → 'Control'")
    report(
        "G.2 build_sample_annotation atribui class_label correto",
        passed,
        f"actual={actual_classes}, expected={expected_classes}. "
        f"features.py:build_sample_annotation linhas 88-94 (effective_map default), "
        f"+ fallback HEALTHY_SYNONYMS linhas 117-122. "
        f"RESPONSABILIDADE: a CLASSIFICAÇÃO acontece em features.py, mas a EXTRAÇÃO "
        f"da string raw vem de conditions.py (extract_sample_condition).",
    )


# ─── G.3: risco de desalinhamento expr↔annot ────────────────────────
def test_g3_alignment_risk():
    """build_sample_annotation itera sobre gsm_cols (passados como argumento).
    Se chamado com lista correta, alinhamento é garantido. Mas o pipeline garante isso?"""
    src = inspect.getsource(build_sample_annotation)
    iterates_gsm_cols = "for gsm_id in gsm_cols" in src

    # No fluxo real (dataset.py:process_single_dataset), gsm_cols vem do
    # cross-referencing das amostras filtradas. O risco é se houver bug nessa etapa.
    # Vamos verificar: se passar gsm_cols com IDs ausentes nos metadados, o que acontece?
    meta = pd.DataFrame({
        "Sample_geo_accession": ["GSM1", "GSM2"],
        "Sample_characteristics_ch1": ["disease state: pancreatic cancer"] * 2,
    })
    gsm_cols = ["GSM1", "GSM2", "GSM3"]  # GSM3 NÃO está nos metadados
    annot = build_sample_annotation(
        meta_df=meta,
        gsm_cols=gsm_cols,
        dataset_id="GSE_test",
        platform_id="GPL19117",
        platform_name="Affy",
    )
    # Para GSM3, condition_raw fica vazio e class_label vira ""
    gsm3_row = annot[annot["sample_id"] == "GSM3"].iloc[0]
    has_empty_class = gsm3_row["class_label"] == ""
    report(
        "G.3 Risco de desalinhamento se gsm_cols incluir IDs sem metadata",
        not has_empty_class,  # Idealmente, deveria pular ou avisar
        f"iterates_gsm_cols={iterates_gsm_cols}. "
        f"Para GSM3 (sem metadata): class_label={gsm3_row['class_label']!r}, "
        f"condition_raw={gsm3_row['condition_raw']!r}. "
        f"COMPORTAMENTO: cria entrada com class_label='' silenciosamente. "
        f"RISCO: amostras 'fantasma' poluem o annot e podem entrar em pipelines "
        f"downstream com class_label vazio (não filtra explicitamente). "
        f"features.py:build_sample_annotation linhas 96-127",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO G — features.py")
    print("=" * 70)
    test_g1_probe_mapping()
    test_g2_sample_annotation()
    test_g3_alignment_risk()
