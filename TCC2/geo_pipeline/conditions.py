"""Extração e filtragem de condições (doença/controle) a partir de metadados GEO."""

import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

import pandas as pd

from geo_pipeline.constants import (
    HEALTHY_SYNONYMS_BROAD,
    HEALTHY_SYNONYMS_STRICT,
    PATHOLOGICAL_SYNONYMS,
)

log = logging.getLogger("geo_pipeline")


def normalize_condition(value: str) -> str:
    """
    Remove sufixos de IDs individuais de uma string de condição.

    Examples:
        "pancreatic cancer P75"     → "pancreatic cancer"
        "healthy control E001"      → "healthy control"
        "biliary tract cancer B101" → "biliary tract cancer"
    """
    normalized = re.sub(r"\s+[A-Z]{0,2}\d{1,4}\s*$", "", value.strip())
    return normalized.strip()


def extract_sample_condition(
    meta_row: pd.Series,
    all_columns: List[str],
) -> str:
    """
    Extrai o rótulo de doença/condição mais informativo de uma linha de metadados.

    Prioridade:
        1. Coluna characteristics contendo "disease state: …"
        2. source_name com condição entre parênteses
        3. title (normalizado)
    """
    for col in all_columns:
        if "characteristics" not in col.lower():
            continue
        val = str(meta_row.get(col, ""))
        if "disease" in val.lower() and ":" in val:
            return re.sub(r"^[^:]+:\s*", "", val).strip().strip('"')

    for col in all_columns:
        if "source_name" not in col.lower():
            continue
        val = str(meta_row.get(col, "")).strip().strip('"')
        m = re.search(r"\(([^)]+)\)", val)
        if m:
            return m.group(1).strip()
        if val and val.lower() not in ("", "nan", "none", "serum", "plasma", "blood"):
            return val

    for col in all_columns:
        if "title" not in col.lower():
            continue
        val = str(meta_row.get(col, "")).strip().strip('"')
        if val and val.lower() not in ("", "nan", "none"):
            return normalize_condition(val)

    return ""


def extract_conditions(
    meta_df: pd.DataFrame,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Extrai condições únicas dos metadados, agrupadas por doença/categoria.

    Usa a lógica per-sample (extract_sample_condition) para agrupar amostras
    pela categoria real em vez de mostrar entradas individuais por paciente.

    Returns:
        grouped    – lista de (nome_condição, contagem) ordenada por contagem desc
        cond_cols  – nomes de colunas que contêm informação de condição
    """
    condition_keywords = [
        "source_name",
        "characteristics",
        "title",
        "description",
        "disease",
        "tissue",
        "cell_type",
        "treatment",
    ]

    cond_cols: List[str] = []
    for col in meta_df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in condition_keywords):
            cond_cols.append(col)

    all_columns = list(meta_df.columns)
    conditions: List[str] = []

    for _, row in meta_df.iterrows():
        cond = extract_sample_condition(row, all_columns)
        normalized = normalize_condition(cond) if cond else ""
        if normalized:
            conditions.append(normalized)

    counts: Counter = Counter(conditions)
    grouped = sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))
    return grouped, cond_cols


def select_conditions_cli(
    grouped_conditions: List[Tuple[str, int]],
    condition_filter: Optional[List[str]] = None,
    no_interactive: bool = False,
) -> Optional[List[str]]:
    """
    Seleciona quais condições manter.

    Prioridade:
        1. --condition-filter da CLI (se fornecido)
        2. Prompt interativo (se permitido)
        3. None (manter todas as amostras)
    """
    if condition_filter:
        log.info(f"Using CLI condition filter: {condition_filter}")
        return condition_filter

    if no_interactive:
        log.info("Non-interactive mode: using ALL samples (no filter)")
        return None

    if not grouped_conditions:
        log.warning("No conditions found in metadata")
        return None

    print("\n" + "=" * 60)
    print("  Conditions / categories found in dataset:")
    print("=" * 60)
    for i, (cond, cnt) in enumerate(grouped_conditions, 1):
        print(f"  [{i}] {cond}  ({cnt} amostras)")
    print(f"  [0] Use ALL samples (no filter)")
    print("=" * 60)

    while True:
        try:
            raw = input(
                "\n🎯 Enter condition numbers separated by comma "
                "(e.g. 1,3) or 0 for all: "
            ).strip()

            if raw == "0":
                return None

            if not raw:
                continue

            indices = [int(x.strip()) for x in raw.split(",")]
            selected: List[str] = []
            for idx in indices:
                if 1 <= idx <= len(grouped_conditions):
                    selected.append(grouped_conditions[idx - 1][0])
                else:
                    print(f"  ❌ Invalid number: {idx}")

            if selected:
                seen = set()
                dedup = []
                for s in selected:
                    if s not in seen:
                        dedup.append(s)
                        seen.add(s)

                for s in dedup:
                    cnt = dict(grouped_conditions).get(s, 0)
                    print(f"  ✅ '{s}' ({cnt} amostras)")
                return dedup

        except ValueError:
            print("  ❌ Enter numbers only.")
        except (EOFError, KeyboardInterrupt):
            print("\n  ⚠️ Using all samples (no filter).")
            return None


def auto_include_healthy_controls(
    selected_conditions: List[str],
    grouped_conditions: List[Tuple[str, int]],
    strict_control_only: bool = True,
) -> List[str]:
    """
    Detecta e inclui automaticamente grupos de controle saudável quando uma
    condição patológica foi selecionada.

    Args:
        selected_conditions: condições escolhidas pela usuária.
        grouped_conditions: todas as condições disponíveis no dataset.
        strict_control_only: se True, só aceita rótulos explícitos
            'healthy/normal'.

    Returns:
        Lista de condições atualizada incluindo controles auto-detectados.
    """
    final_selection = list(selected_conditions)
    all_available = [c[0] for c in grouped_conditions]

    is_pathological = False
    for sel in selected_conditions:
        sel_lower = sel.lower()
        if any(term in sel_lower for term in PATHOLOGICAL_SYNONYMS):
            is_pathological = True
            break

    if not is_pathological:
        return final_selection

    control_already_selected = False
    all_healthy_terms = HEALTHY_SYNONYMS_STRICT + (
        [] if strict_control_only else HEALTHY_SYNONYMS_BROAD
    )

    for sel in selected_conditions:
        sel_lower = sel.lower()
        if any(term == sel_lower for term in all_healthy_terms):
            control_already_selected = True
            break
        if any(term in sel_lower for term in all_healthy_terms):
            control_already_selected = True
            break

    if control_already_selected:
        return final_selection

    found_control = None
    search_list = (
        HEALTHY_SYNONYMS_STRICT
        if strict_control_only
        else (HEALTHY_SYNONYMS_STRICT + HEALTHY_SYNONYMS_BROAD)
    )

    for term in search_list:
        for available in all_available:
            if term == available.lower():
                found_control = available
                break
            if term in available.lower():
                found_control = available
                break
        if found_control:
            break

    if found_control:
        log.info(f"Auto-added healthy control: ['{found_control}']")
        if found_control not in final_selection:
            final_selection.append(found_control)
    else:
        log.info("No healthy control group found for automatic inclusion.")

    return final_selection


def filter_samples_by_conditions(
    meta_df: pd.DataFrame,
    conditions: Optional[List[str]],
    condition_cols: List[str],
) -> pd.DataFrame:
    """
    Filtra linhas de metadados que casam com qualquer das condições selecionadas
    (case-insensitive, substring match).
    """
    if conditions is None:
        log.info(f"Using ALL {meta_df.shape[0]} samples (no filter)")
        return meta_df

    mask = pd.Series(False, index=meta_df.index)

    for cond in conditions:
        cond_lower = cond.lower()
        for col in condition_cols:
            col_mask = (
                meta_df[col]
                .astype(str)
                .str.lower()
                .str.contains(re.escape(cond_lower), na=False)
            )
            mask = mask | col_mask

    if mask.sum() == 0:
        log.warning("Primary columns had no matches; searching all columns")
        df_str = meta_df.astype(str).apply(lambda x: x.str.lower())
        for cond in conditions:
            cond_lower = cond.lower()
            mask = mask | df_str.apply(
                lambda row: cond_lower in " ".join(row.values), axis=1
            )

    filtered = meta_df.loc[mask].copy()
    log.info(f"Filtered: {filtered.shape[0]} samples for conditions {conditions}")
    return filtered
