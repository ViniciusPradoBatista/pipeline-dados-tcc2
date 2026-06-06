"""
===================================================================
  GEO miRNA Cross-Platform Integration Pipeline
  For PDAC (Pancreatic Ductal Adenocarcinoma) Research

  Features:
    - Universal GEO Series Matrix reader (.txt and .xlsx)
    - Automatic platform detection and scale inference
    - Probe ID harmonization across platforms
    - Cross-dataset merge by common miRNAs (log2 scale)
    - ComBat batch correction (preserving biological signal)
    - Global z-score normalization AFTER ComBat (not per-dataset)
    - PurityB / PurityD / Silhouette validation
    - PCA visualization (PC1-PC2 and PC3-PC4)
    - Automatic inclusion of healthy controls

  Supported platforms:
    Affymetrix (GPL19117, GPL18402, etc.)
    3D-Gene / Toray (GPL18941, GPL21263)
    Agilent, Illumina, and others

  Usage:
    python geo_mirna_pipeline.py GSE85589_series_matrix.txt \\
        GSE59856_series_matrix.txt --output-root ./out

  Author: TCC Pipeline (evolved from app.py)
===================================================================
"""

import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split

from geo_pipeline.cli import (
    build_cli,
    interactive_file_picker,
    interactive_output_picker,
)
from geo_pipeline.dataset import merge_datasets, process_single_dataset
from geo_pipeline.metrics import compute_purity_metrics
from geo_pipeline.normalize import (
    apply_zscore,
    combat_apply,
    combat_fit,
    fit_zscore,
)
from geo_pipeline.plots import generate_all_plots

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("geo_pipeline")

# =====================================================================
# Split estratificado — fronteira do Estágio 1 (antes de ComBat/z-score)
# =====================================================================


def stratified_split_ids(
    merged_annot: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Divide as amostras em treino/teste estratificando por classe × plataforma.

    Estratificar pela chave combinada garante que AMBAS as plataformas (batches)
    apareçam no treino e no teste — requisito do ComBat (≥2 batches no treino) e
    do ``neuroCombatFromTraining`` (batch do teste precisa existir no treino).

    Faz fallback gracioso (classe × plataforma → classe → sem estratificação)
    quando alguma célula é pequena demais para o split estratificado.
    """
    ids = merged_annot["sample_id"].astype(str).tolist()

    strat_full = (
        merged_annot["class_label"].astype(str)
        + "|"
        + merged_annot["platform_id"].astype(str)
    ).tolist()
    strat_class = merged_annot["class_label"].astype(str).tolist()

    for strat, label in ((strat_full, "classe × plataforma"), (strat_class, "classe"), (None, "sem estratificação")):
        try:
            train_ids, test_ids = train_test_split(
                ids,
                test_size=test_size,
                random_state=random_state,
                stratify=strat,
            )
            log.info(f"Split estratificado por {label}: treino={len(train_ids)}, teste={len(test_ids)}")
            return train_ids, test_ids
        except ValueError as exc:
            log.warning(f"Estratificação por {label} falhou ({exc}); tentando fallback.")

    # último recurso — não deve acontecer
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
    return train_ids, test_ids


# Abaixo deste nº de amostras, uma célula plataforma×classe no TESTE não sustenta
# métrica estratificada confiável (variância alta demais).
_MIN_TEST_CELL = 10


def _report_split_composition(merged_annot: pd.DataFrame, output_root: Path) -> None:
    """Persiste a composição do split e ADVERTE sobre células de teste frágeis.

    Caso conhecido (GSE85589+GSE59856 / PDAC vs Control): Affymetrix·Control tem
    ~4 amostras no teste — pequeno demais para reportar métrica isolada dessa célula.
    O aviso fica no log E em split_composition.csv para chegar à banca, não só ao chat.
    """
    comp = (
        merged_annot.groupby(["platform_id", "class_label", "split"])
        .size()
        .reset_index(name="n")
        .sort_values(["platform_id", "class_label", "split"])
    )
    comp.to_csv(output_root / "split_composition.csv", index=False)
    log.info("Composição do split (platform × class × split):")
    for _, r in comp.iterrows():
        log.info(f"    {r['platform_id']} · {r['class_label']} · {r['split']}: {r['n']}")

    test_cells = comp[comp["split"] == "test"]
    fragile = test_cells[test_cells["n"] < _MIN_TEST_CELL]
    for _, r in fragile.iterrows():
        log.warning(
            f"AVISO: célula de TESTE {r['platform_id']} × {r['class_label']} tem só "
            f"{r['n']} amostras (< {_MIN_TEST_CELL}). Métrica estratificada nessa célula "
            f"NÃO é confiável — não reportar isoladamente."
        )


# =====================================================================
# Pipeline (reutilizável por main() e pelos testes)
# =====================================================================


def run_pipeline(
    files: List[str],
    output_root: Path,
    no_interactive: bool = False,
    condition_filter: Optional[List[str]] = None,
    class_map: Optional[Dict[str, str]] = None,
    auto_add_healthy_control: bool = True,
    strict_control_only: bool = True,
    no_combat: bool = False,
    no_plots: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Executa o Estágio 1 completo: por-dataset → merge → split → ComBat/z-score.

    Ordem (anti-vazamento):
        merge miRNAs comuns
        → split estratificado (classe × plataforma)   [reserva o teste AQUI]
        → ComBat: fit(treino) → apply(teste)
        → z-score: fit(treino corrigido) → apply(treino, teste)
        → persiste combat_estimates.pkl + zscore_params.csv + base_treino/teste.csv
    """
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = []
    for filepath in files:
        if not Path(filepath).exists():
            continue
        res = process_single_dataset(
            path=filepath,
            output_root=output_root,
            no_interactive=no_interactive,
            condition_filter=condition_filter,
            class_map=class_map,
            auto_add_healthy_control=auto_add_healthy_control,
            strict_control_only=strict_control_only,
        )
        if res:
            dataset_dirs.append(res)

    if len(dataset_dirs) < 2:
        log.warning(
            "Menos de 2 datasets processados — sem merge, sem ComBat e sem split. "
            "O Estágio 2 requer base_treino/base_teste gerados a partir de >= 2 datasets."
        )
        return

    # merge_datasets retorna (merged_raw_log2, merged_annot) — escala log2, SEM normalização.
    merged_raw, merged_annot = merge_datasets(dataset_dirs, output_root)
    if merged_raw.empty:
        log.error("Merge vazio — abortando antes da normalização.")
        return

    # ── SPLIT na fronteira do Estágio 1 (antes de qualquer ComBat/z-score) ──
    train_ids, test_ids = stratified_split_ids(merged_annot, test_size, random_state)
    train_set, test_set = set(train_ids), set(test_ids)

    merged_annot["split"] = merged_annot["sample_id"].astype(str).map(
        lambda s: "train" if s in train_set else "test"
    )
    merged_annot.to_csv(output_root / "merged_sample_annotation.csv", index=False)

    annot_train = merged_annot[merged_annot["split"] == "train"].copy()
    annot_test = merged_annot[merged_annot["split"] == "test"].copy()

    # ── Composição do split por plataforma × classe + aviso de células frágeis ──
    _report_split_composition(merged_annot, output_root)

    gsm_cols = [c for c in merged_raw.columns if c != "Probe_ID"]
    train_cols = [c for c in gsm_cols if c in train_set]
    test_cols = [c for c in gsm_cols if c in test_set]

    expr_train = merged_raw.set_index("Probe_ID")[train_cols]
    expr_test = merged_raw.set_index("Probe_ID")[test_cols]

    # ── Interseção de probes treino∩teste ANTES do fit (política: descartar + logar) ──
    common_probes = expr_train.index.intersection(expr_test.index)
    n_only_train = len(expr_train.index) - len(common_probes)
    n_only_test = len(expr_test.index) - len(common_probes)
    if n_only_train or n_only_test:
        log.warning(
            f"Probes fora da interseção treino∩teste DESCARTADAS: "
            f"só-treino={n_only_train}, só-teste={n_only_test} "
            f"(mantendo {len(common_probes)} probes comuns)."
        )
    expr_train = expr_train.loc[common_probes]
    expr_test = expr_test.loc[common_probes]

    # ── ComBat: fit no treino, apply no teste ──
    if not no_combat:
        train_corr, estimates, train_med = combat_fit(expr_train, annot_train)
        test_corr = combat_apply(
            expr_test, annot_test, train_corr.index, estimates, train_med
        )
        joblib.dump(estimates, output_root / "combat_estimates.pkl")
    else:
        log.info("--no-combat: pulando ComBat (apenas z-score fit/apply).")
        train_corr = expr_train
        test_corr = expr_test.reindex(index=expr_train.index)

    # ── z-score: fit no treino corrigido, apply no treino e no teste ──
    mu, sd = fit_zscore(train_corr)
    base_treino = apply_zscore(train_corr, mu, sd)
    base_teste = apply_zscore(test_corr, mu, sd)

    pd.DataFrame(
        {"Probe_ID": mu.index, "mu": mu.values, "sd": sd.values}
    ).to_csv(output_root / "zscore_params.csv", index=False)

    base_treino.reset_index().to_csv(output_root / "base_treino.csv", index=False)
    base_teste.reset_index().to_csv(output_root / "base_teste.csv", index=False)

    log.info(
        f"base_treino: {base_treino.shape[0]} probes × {base_treino.shape[1]} amostras | "
        f"base_teste: {base_teste.shape[0]} probes × {base_teste.shape[1]} amostras"
    )

    # ── Artefato: matriz integrada corrigida (treino+teste), SÓ para PCA/inspeção ──
    # NÃO alimenta o Estágio 2 (esse consome base_treino/base_teste).
    full_corr = (
        pd.concat([train_corr, test_corr], axis=1).reset_index()
        if not no_combat
        else None
    )
    if full_corr is not None:
        full_corr.to_csv(output_root / "merged_expression_combat.csv", index=False)

    compute_purity_metrics(merged_raw, full_corr, merged_annot).to_csv(
        output_root / "purity_metrics.csv", index=False
    )
    if not no_plots:
        generate_all_plots(merged_raw, full_corr, merged_annot, output_root)


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    args = build_cli()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_map = None
    if args.class_map:
        class_map = {
            item.split("=")[0].strip(): item.split("=")[1].strip()
            for item in args.class_map
            if "=" in item
        }

    if not args.files:
        args.files = interactive_file_picker()
        if not args.files:
            sys.exit(0)
        args.output_root = interactive_output_picker()
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        files=args.files,
        output_root=output_root,
        no_interactive=args.no_interactive,
        condition_filter=args.condition_filter,
        class_map=class_map,
        auto_add_healthy_control=args.auto_add_healthy_control,
        strict_control_only=args.strict_control_only,
        no_combat=args.no_combat,
        no_plots=args.no_plots,
    )

    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()
