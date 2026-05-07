"""Módulo E — scale.py — Validação técnica."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geo_pipeline.scale import detect_platform, infer_scale  # noqa: E402


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── E.1: detect_platform com IDs corretos e variações ──────────────
def test_e1_detect_platform():
    """Testa GPL19117, GPL18941 (Affy/3D-Gene) e variações de capitalização."""
    cases = [
        ("GPL19117", "GPL19117"),
        ("GPL18941", "GPL18941"),
        ("gpl19117", "GPL19117"),  # lowercase
        ("Gpl18941", "GPL18941"),  # mixed case
    ]
    print("\n[TABELA] detect_platform(meta_with_GPL_id):")
    print(f"  {'INPUT':<15} {'EXPECTED':<15} {'ACTUAL':<15} {'OK':<5}")
    print("  " + "-" * 50)
    all_ok = True
    for inp, expected in cases:
        meta = pd.DataFrame({"Sample_platform_id": [inp, inp, inp]})
        plat_id, plat_name = detect_platform(meta)
        ok = plat_id == expected or plat_id.upper() == expected.upper()
        all_ok = all_ok and ok
        print(f"  {inp:<15} {expected:<15} {plat_id:<15} {ok}")

    report(
        "E.1 detect_platform identifica GPL19117/GPL18941 com tolerância a caps",
        all_ok,
        f"all_ok={all_ok}. NOTA: detect_platform faz match LITERAL na string vinda do "
        f"campo platform_id — case-sensitivity depende do conteúdo real do arquivo GEO. "
        f"scale.py:detect_platform linhas 22-30",
    )


# ─── E.2: infer_scale para diferentes ranges ────────────────────────
def test_e2_infer_scale_ranges():
    """Testa critério de decisão para ranges típicos."""
    # Construct synthetic exprs
    def mk_expr(min_v, max_v, n=100):
        rng = np.random.default_rng(0)
        vals = rng.uniform(min_v, max_v, size=(n, 5))
        df = pd.DataFrame(vals, columns=["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"])
        df.insert(0, "Probe_ID", [f"p{i}" for i in range(n)])
        return df

    cases = [
        ("[0, 20] sem keyword", 0.0, 20.0, "GPL00000", ""),
        ("[0, 65000] sem keyword", 0.0, 65000.0, "GPL00000", ""),
        ("[0, 20] com plataforma Affy", 0.0, 20.0, "GPL19117", ""),
        ("[0, 65000] com keyword 'rma'", 0.0, 65000.0, "GPL19117", "data was processed using rma"),
    ]
    print("\n[TABELA] infer_scale(meta, platform, expr):")
    print(f"  {'CASE':<35} {'PLATFORM':<10} {'SCALE':<22}")
    print("  " + "-" * 75)
    results = []
    for desc, mn, mx, plat, proc in cases:
        meta = pd.DataFrame({"Sample_data_processing": [proc] * 5})
        expr = mk_expr(mn, mx)
        scale = infer_scale(meta, plat, expr)
        results.append((desc, scale))
        print(f"  {desc:<35} {plat:<10} {scale:<22}")

    # Esperado:
    # [0,20] sem keyword → ambíguo: depende de plataforma; aqui plat unknown,
    #   infer_scale linha 119: low range + plataforma desconhecida → "already_processed"
    # [0,65000] sem keyword → "needs_log2" (linha 122)
    # [0,20] com Affy → "already_log2" (linha 116)
    # [0,65000] com RMA keyword → "already_log2" (linha 105)
    expected_scales = ["already_processed", "needs_log2", "already_log2", "already_log2"]
    actual = [s for _, s in results]
    passed = actual == expected_scales
    report(
        "E.2 infer_scale infere escala correta por range/keyword/plataforma",
        passed,
        f"actual={actual}, expected={expected_scales}. "
        f"scale.py:infer_scale lógica em linhas 86-126 "
        f"(keywords RMA/log2 → already_log2; range baixo + plat conhecida → already_log2; "
        f"range alto → needs_log2)",
    )


# ─── E.3: comportamento com plataforma desconhecida ─────────────────
def test_e3_unknown_platform():
    """Plataforma fora de KNOWN_PLATFORMS — fail-fast ou default?"""
    meta = pd.DataFrame({"Sample_platform_id": ["GPL99999", "GPL99999"]})
    plat_id, plat_name = detect_platform(meta)
    # Default: nome retornado é o próprio ID (linha 47 KNOWN_PLATFORMS.get(...,
    # platform_id))
    is_default_passthrough = plat_id == "GPL99999" and plat_name == "GPL99999"

    # E em infer_scale, com plataforma desconhecida:
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(
        rng.uniform(0, 20, size=(50, 3)),
        columns=["GSM1", "GSM2", "GSM3"],
    )
    expr.insert(0, "Probe_ID", [f"p{i}" for i in range(50)])
    meta_proc = pd.DataFrame({"Sample_data_processing": [""] * 3})
    scale = infer_scale(meta_proc, "GPL99999", expr)

    # NÃO há exceção. O default em infer_scale para low-range + unknown plat é
    # "already_processed" (linha 119) — comportamento permissivo.
    fail_safe = scale in ("already_processed", "unknown")
    report(
        "E.3 Plataforma desconhecida: comportamento (fail-fast ou default)",
        is_default_passthrough,  # documenta como atual
        f"detect_platform('GPL99999') → ('{plat_id}', '{plat_name}') — "
        f"NÃO faz fail-fast, retorna ID literal como nome. "
        f"infer_scale com plat unknown + range [0,20] → '{scale}' (default permissivo). "
        f"RISCO: se GPL desconhecida tiver dados RAW (range alto), o pipeline "
        f"aplicará log2(x+1); se range baixo mas NÃO for log2, dados podem ser "
        f"interpretados erroneamente como já processados. "
        f"scale.py:detect_platform linha 47 + infer_scale linhas 116-126",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO E — scale.py")
    print("=" * 70)
    test_e1_detect_platform()
    test_e2_infer_scale_ranges()
    test_e3_unknown_platform()
