"""Parsing numérico robusto para valores de expressão de Series Matrix."""

import numpy as np


def smart_float(x, broken_decimal: bool = False) -> float:
    """
    Converte valor de expressão para float.

    Quando ``broken_decimal=True``, o arquivo exportou os dígitos com pontos
    inseridos como separador de milhar (ex: valor real ``1.129491157``
    aparece como ``1.129.491.157``; valor real ``1.27586602`` aparece como
    ``127.586.602``). Regra de reconstrução: strip de todos os pontos, o
    primeiro dígito vira a parte inteira e o resto a decimal.

    Em arquivos GEO Series Matrix .txt legítimos (sem locale-quebrado), o
    ponto é o separador decimal e a conversão direta é usada.
    """
    s = str(x).replace('"', "").strip()

    if not s or s.lower() in ("na", "nan", "null", "none", "--", "n/a"):
        return np.nan

    if broken_decimal:
        sign = -1.0 if s.startswith("-") else 1.0
        digits = s.lstrip("-").replace(".", "").replace(",", "")
        if not digits.isdigit():
            return np.nan
        if len(digits) <= 1:
            return sign * float(digits)
        return sign * float(digits[0] + "." + digits[1:])

    s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return np.nan
