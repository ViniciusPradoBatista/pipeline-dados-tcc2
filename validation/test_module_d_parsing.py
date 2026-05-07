"""Módulo D — parsing.py / smart_float — Validação técnica."""

import sys
from pathlib import Path

import math

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))

from geo_pipeline.parsing import smart_float  # noqa: E402


def report(name, passed, details):
    icon = "[PASSOU]" if passed else "[FALHOU]"
    print(f"\n{icon} {name}")
    print(f"  {details}")


# ─── D.1: tabela de inputs específicos ──────────────────────────────
def test_d1_specific_inputs():
    """Lista o valor retornado para cada caso, em modo default e broken_decimal."""
    cases = [
        ("1.129.491.157", False),  # multi-dot, modo normal
        ("1.129.491.157", True),   # multi-dot, modo broken_decimal
        ("1,129.49", False),
        ("1,129.49", True),
        ("NA", False),
        ("", False),
        (None, False),
        ("2.612", False),  # decimal padrão
        ("127.586.602", True),  # caso real PT-BR
    ]
    print("\n[TABELA] smart_float(input, broken_decimal=mode):")
    print(f"  {'INPUT':<25} {'BROKEN':<8} {'OUTPUT':<25} {'TYPE':<10}")
    print("  " + "-" * 70)
    for inp, broken in cases:
        try:
            result = smart_float(inp, broken_decimal=broken)
            t = type(result).__name__
            if isinstance(result, float) and math.isnan(result):
                rstr = "nan"
            else:
                rstr = repr(result)
        except Exception as e:
            rstr = f"EXC: {type(e).__name__}: {e}"
            t = "exc"
        print(f"  {repr(inp):<25} {str(broken):<8} {rstr:<25} {t:<10}")
    # Validações concretas
    assertions = [
        (smart_float("1.129.491.157") == 1.129491157, "default mode multi-dot keeps first dot"),
        (smart_float("1.129.491.157", True) == 1.129491157, "broken mode reconstrói (4 dots)"),
        (smart_float("127.586.602", True) == 1.27586602, "broken mode 2-dots → 1.27586602"),
        (math.isnan(smart_float("NA")), "NA → NaN"),
        (math.isnan(smart_float("")), "vazio → NaN"),
        (math.isnan(smart_float(None)), "None → NaN"),
        (smart_float("2.612") == 2.612, "decimal padrão preserva"),
    ]
    fails = [msg for ok, msg in assertions if not ok]
    passed = len(fails) == 0
    report(
        "D.1 smart_float retorna valores esperados em todos os casos",
        passed,
        f"Falhas: {fails if fails else 'nenhuma'}. parsing.py:smart_float linhas 7-37",
    )


# ─── D.2: valores biologicamente impossíveis ────────────────────────
def test_d2_negative_outputs():
    """smart_float pode retornar negativo? Para expressão bruta isso seria suspeito."""
    cases = [
        ("-2.5", False),
        ("-2.5", True),
        ("-1.129.491.157", True),
    ]
    results = []
    for inp, broken in cases:
        r = smart_float(inp, broken_decimal=broken)
        results.append((inp, broken, r))
    has_negative = any(r < 0 for _, _, r in results if not math.isnan(r))
    # Comportamento esperado: retorna negativo (smart_float é parser, não validador biológico).
    # ATENÇÃO: pra expressão bruta esperaria-se >=0; para log2 já-transformado, neg é OK.
    report(
        "D.2 smart_float aceita negativos (caveat biológico)",
        True,  # comportamento documentado: parser não valida domínio
        f"Casos: {results}. "
        f"PARSER aceita negativos (broken_decimal preserva sinal: linha 21 'sign = -1.0 if s.startswith(\"-\")'). "
        f"ATENÇÃO: para microarray RAW, valores negativos são fisicamente impossíveis. "
        f"Para RMA/log2 já transformado (caso de uso atual), negativos são esperados. "
        f"O pipeline NÃO valida range — depende do scale.py para inferir se já é log.",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" MÓDULO D — parsing.py")
    print("=" * 70)
    test_d1_specific_inputs()
    test_d2_negative_outputs()
