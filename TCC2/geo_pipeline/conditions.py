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
    Remove sufixos de IDs individuais de paciente de uma string de condição.

    Requer pelo menos 1 letra maiúscula antes dos dígitos para evitar remover
    informação biológica como estadiamentos numéricos ("Stage 4" é preservado,
    "pancreatic cancer P75" perde o "P75").

    Examples:
        "pancreatic cancer P75"     → "pancreatic cancer"
        "healthy control E001"      → "healthy control"
        "biliary tract cancer B101" → "biliary tract cancer"
        "Stage 4"                   → "Stage 4"  (preservado)
    """
    normalized = re.sub(r"\s+[A-Z]{1,2}\d{1,4}\s*$", "", value.strip())
    return normalized.strip()


def _word_boundary_contains(text: str, term: str) -> bool:
    """Verifica se `term` aparece em `text` com word boundaries (não como substring de outra palavra)."""
    return bool(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text))


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
        4. description (fallback — útil quando title é genérico, e.g. GSE85589)
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
        if val and val.lower() not in ("", "nan", "none", "serum", "plasma", "blood",
                                       "rna", "tissue", "cfdna", "cfrna"):
            return val

    title_result = ""
    for col in all_columns:
        if "title" not in col.lower():
            continue
        val = str(meta_row.get(col, "")).strip().strip('"')
        if val and val.lower() not in ("", "nan", "none"):
            title_result = normalize_condition(val)
            break

    # 4. description — fallback when title is uninformative (e.g. "miRNA from N1")
    for col in all_columns:
        if "description" not in col.lower():
            continue
        val = str(meta_row.get(col, "")).strip().strip('"')
        if val and val.lower() not in ("", "nan", "none"):
            desc_norm = normalize_condition(val)
            # Title is considered "descriptive" (and preferred) only when:
            # - it is longer than 3 words
            # - does NOT look like a sample descriptor ("miRNA from ...", "RNA from ...")
            # - does NOT contain " from " (which signals a sample name, not a condition)
            title_looks_descriptive = bool(
                title_result
                and len(title_result.split()) > 3
                and " from " not in title_result.lower()
                and not title_result.lower().startswith(("mirna", "rna", "cdna", "serum", "plasma"))
            )
            if title_looks_descriptive:
                return title_result
            return desc_norm

    return title_result


def extract_conditions(
    meta_df: pd.DataFrame,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Extrai condições únicas dos metadados, agrupadas por doença/categoria.

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

    Usa word-boundary matching para evitar falsos positivos como
    "non-cancer control" sendo detectado ao buscar "cancer".
    """
    final_selection = list(selected_conditions)
    all_available = [c[0] for c in grouped_conditions]

    is_pathological = False
    for sel in selected_conditions:
        sel_lower = sel.lower()
        if any(_word_boundary_contains(sel_lower, term) for term in PATHOLOGICAL_SYNONYMS):
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
        if any(_word_boundary_contains(sel_lower, term) for term in all_healthy_terms):
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
            avail_lower = available.lower()
            if term == avail_lower:
                found_control = available
                break
            if _word_boundary_contains(avail_lower, term):
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
    Filtra linhas de metadados que casam com qualquer das condições selecionadas.

    Estratégia (em ordem de prioridade):
    1. Match exato da condição normalizada (mais seguro).
    2. Word-boundary match (evita "cancer" casar com "non-cancer control").
    3. Fallback: busca em todas as colunas com word boundaries.
    """
    if conditions is None:
        log.info(f"Using ALL {meta_df.shape[0]} samples (no filter)")
        return meta_df

    mask = pd.Series(False, index=meta_df.index)

    for cond in conditions:
        cond_lower = cond.lower()
        cond_pattern = r"(?<![a-z])" + re.escape(cond_lower) + r"(?![a-z])"

        for col in condition_cols:
            col_vals = meta_df[col].astype(str)

            # 1. Match exato da condição normalizada
            exact_mask = col_vals.apply(
                lambda v: normalize_condition(v.strip().strip('"')).lower() == cond_lower
            )
            # 2. Word-boundary substring match
            boundary_mask = col_vals.str.lower().str.contains(
                cond_pattern, na=False, regex=True
            )
            mask = mask | exact_mask | boundary_mask

    if mask.sum() == 0:
        log.warning(
            "Nenhuma amostra encontrada nas colunas de condição. "
            "Buscando em todas as colunas com word boundaries..."
        )
        for cond in conditions:
            cond_lower = cond.lower()
            cond_pattern = r"(?<![a-z])" + re.escape(cond_lower) + r"(?![a-z])"
            mask = mask | meta_df.astype(str).apply(
                lambda row: bool(re.search(cond_pattern, " ".join(row.values).lower())),
                axis=1,
            )

    filtered = meta_df.loc[mask].copy()

    # Fail-loud: zero matches (mesmo após o fallback que varre todas as colunas)
    # quase sempre indica condição mal-digitada ou ausente neste dataset. Erguer
    # erro descritivo é melhor que devolver DataFrame vazio silenciosamente.
    if filtered.empty:
        raise ValueError(
            f"Nenhuma amostra corresponde às condições {conditions} neste dataset "
            f"(nem no fallback de varredura completa). Verifique a grafia das condições "
            f"ou se elas existem neste dataset."
        )

    log.info(f"Filtered: {filtered.shape[0]} samples for conditions {conditions}")
    return filtered
