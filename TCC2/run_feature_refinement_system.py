"""
===================================================================
  Feature Refinement Orchestrator
  GEO miRNA Pipeline — PDAC Feature Selection System

  Orchestrates two downstream pipelines:
    • Boruta  (refine_features_pdac.py)
    • LASSO   (refine_features_lasso.py)

  Modes:
    Interactive : python run_feature_refinement_system.py
    CLI direct  : python run_feature_refinement_system.py \
                    --expr-path ... --annot-path ... \
                    --output-dir ... --mode both

  Author: TCC Pipeline Orchestrator
===================================================================
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding for box-drawing/unicode (─ ═) — mesmo guard do Estágio 1.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── optional tkinter ──────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


# =====================================================================
# Constants
# =====================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
BORUTA_SCRIPT = SCRIPT_DIR / "refine_features_pdac.py"
LASSO_SCRIPT  = SCRIPT_DIR / "refine_features_lasso.py"

# O Estágio 1 entrega o split treino/teste já normalizado sem vazamento.
# A fonte preferencial do Estágio 2 passa a ser base_treino.csv (+ base_teste.csv).
# merged_expression_combat.csv permanece apenas como artefato de PCA — NÃO usar
# como entrada do Estágio 2 (contém treino+teste juntos).
_EXPR_PRIORITY = [
    "base_treino.csv",
    "merged_expression_raw.csv",  # legado / fallback (split interno)
]
_TEST_NAME = "base_teste.csv"
_ANNOT_NAME = "merged_sample_annotation.csv"

SEPARATOR = "─" * 56
THICK_SEP = "═" * 56


# =====================================================================
# 1. Tkinter helpers
# =====================================================================

def _tk_pick_file(title: str) -> Optional[str]:
    if not _TK_AVAILABLE:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def _tk_pick_folder(title: str) -> Optional[str]:
    if not _TK_AVAILABLE:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or None
    except Exception:
        return None


# =====================================================================
# 2. Interactive UI helpers
# =====================================================================

def _print_header(title: str) -> None:
    print(f"\n{THICK_SEP}")
    print(f"  {title}")
    print(THICK_SEP)


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Operação cancelada.")
        sys.exit(0)
    return val if val else default


def _confirm(msg: str = "Confirmar e executar?") -> bool:
    resp = _prompt(f"{msg} (s/n)", default="s").lower()
    return resp in ("s", "sim", "y", "yes", "")


def _suggest_expression_file(search_dir: Optional[Path]) -> Optional[Path]:
    """Return the highest-priority expression file found near search_dir."""
    if search_dir is None:
        return None
    for name in _EXPR_PRIORITY:
        candidate = search_dir / name
        if candidate.exists():
            return candidate
    return None


def _suggest_annotation_file(search_dir: Optional[Path]) -> Optional[Path]:
    if search_dir is None:
        return None
    candidate = search_dir / _ANNOT_NAME
    return candidate if candidate.exists() else None


def interactive_select_file(label: str, hint_dir: Optional[Path] = None) -> str:
    """Prompt user to select a CSV file interactively."""
    suggestion = None
    if hint_dir:
        if "express" in label.lower():
            suggestion = _suggest_expression_file(hint_dir)
        elif "annot" in label.lower():
            suggestion = _suggest_annotation_file(hint_dir)

    print(f"\n{SEPARATOR}")
    print(f"  Selecionar: {label}")
    if suggestion:
        print(f"  Sugestao  : {suggestion.name}")
    print(SEPARATOR)

    options = []
    if suggestion:
        options.append(f"[1] Usar sugestao: {suggestion.name}")
        options.append("[2] Abrir seletor de arquivos" if _TK_AVAILABLE else "[2] Digitar caminho manualmente")
        options.append("[3] Digitar caminho manualmente")
        default_choice = "1"
    else:
        options.append("[1] Abrir seletor de arquivos" if _TK_AVAILABLE else "[1] Digitar caminho manualmente")
        options.append("[2] Digitar caminho manualmente")
        default_choice = "1"

    for opt in options:
        print(f"  {opt}")
    print(SEPARATOR)

    while True:
        choice = _prompt("Escolha", default=default_choice)

        if suggestion and choice == "1":
            print(f"  OK: {suggestion.name}")
            return str(suggestion)

        # Determine if this choice maps to "picker" or "manual"
        pick_choice = "2" if suggestion else "1"
        manual_choice = "3" if suggestion else "2"

        if choice == pick_choice and _TK_AVAILABLE:
            path = _tk_pick_file(f"Selecione: {label}")
            if path:
                print(f"  OK: {Path(path).name}")
                return path
            print("  Seletor indisponivel. Tente digitar o caminho.")
            choice = manual_choice  # fall through to manual

        if choice == manual_choice or (not _TK_AVAILABLE and choice == pick_choice):
            path = _prompt("Caminho do arquivo").strip('"').strip("'")
            if path and Path(path).exists():
                print(f"  OK: {Path(path).name}")
                return path
            print("  Arquivo nao encontrado. Tente novamente.")


def interactive_select_output_dir() -> str:
    """Prompt user to select an output root directory."""
    print(f"\n{SEPARATOR}")
    print("  Onde salvar os resultados?")
    print(SEPARATOR)
    print("  [1] Pasta atual   (.)")
    if _TK_AVAILABLE:
        print("  [2] Escolher via seletor de pastas")
    print(f"  {'[3]' if _TK_AVAILABLE else '[2]'} Digitar caminho manualmente")
    print(SEPARATOR)

    while True:
        choice = _prompt("Escolha", default="1")

        if choice == "1":
            return "."
        if choice == "2" and _TK_AVAILABLE:
            path = _tk_pick_folder("Selecione pasta raiz de saida")
            return path if path else "."
        manual = "3" if _TK_AVAILABLE else "2"
        if choice == manual or (not _TK_AVAILABLE and choice == "2"):
            path = _prompt("Caminho da pasta").strip('"').strip("'")
            return path if path else "."


def interactive_select_lasso_mode() -> str:
    print(f"\n{SEPARATOR}")
    print("  Modo LASSO:")
    print(SEPARATOR)
    print("  [1] logistic_l1  — Logistic Regression L1 (recomendado para classificacao)")
    print("  [2] lasso_cv     — LassoCV regression (ranking alternativo)")
    print(SEPARATOR)
    choice = _prompt("Escolha", default="1")
    return "lasso_cv" if choice == "2" else "logistic_l1"


def interactive_select_mode() -> str:
    _print_header("Fluxo de Refinamento de Features")
    print("  [1] Boruta")
    print("  [2] LASSO")
    print("  [3] Boruta + LASSO (recomendado)")
    print(THICK_SEP)
    choice = _prompt("Escolha", default="3")
    mapping = {"1": "boruta", "2": "lasso", "3": "both"}
    mode = mapping.get(choice, "both")
    return mode


def interactive_yes_no(question: str, default: bool = False) -> bool:
    default_str = "s" if default else "n"
    resp = _prompt(question + " (s/n)", default=default_str).lower()
    return resp in ("s", "sim", "y", "yes")


# =====================================================================
# 3. CLI Argument Parsing
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_feature_refinement_system.py",
        description="Orchestrator for PDAC miRNA Feature Selection (Boruta + LASSO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemplos:
              # Modo interativo:
              python run_feature_refinement_system.py

              # CLI direto — ambos os pipelines:
              python run_feature_refinement_system.py \\
                --expr-path merged_expression_combat.csv \\
                --annot-path merged_sample_annotation.csv \\
                --output-dir ./results_feature_selection \\
                --mode both

              # Somente LASSO com modo lasso_cv:
              python run_feature_refinement_system.py \\
                --expr-path merged_expression_combat.csv \\
                --annot-path merged_sample_annotation.csv \\
                --output-dir ./results_lasso \\
                --mode lasso --lasso-mode lasso_cv
        """),
    )

    # Paths
    parser.add_argument("--expr-path",  default=None, help="Treino: base_treino.csv (CSV)")
    parser.add_argument("--test-path",  default=None, help="Teste: base_teste.csv (CSV). Se omitido, deriva ao lado do --expr-path.")
    parser.add_argument("--annot-path", default=None, help="Arquivo de annotation (CSV)")
    parser.add_argument("--output-dir", default=None, help="Pasta raiz de saida")

    # Mode
    parser.add_argument(
        "--mode", default=None,
        choices=["boruta", "lasso", "both"],
        help="Pipeline a executar (boruta / lasso / both)",
    )

    # Data flags
    parser.add_argument("--use-combat",  action="store_true", help="Dados sao ComBat-corrected")
    parser.add_argument("--use-zscore",  action="store_true", help="Dados sao Z-scored")

    # Class config
    parser.add_argument("--target-col",      default="class_label", help="Coluna alvo na annotation")
    parser.add_argument("--positive-class",  default="PDAC",        help="Classe positiva")
    parser.add_argument("--negative-class",  default="Control",     help="Classe negativa")

    # Statistical thresholds
    parser.add_argument("--p-val-thresh",  default=0.05, type=float, help="FDR p-value threshold (Step A)")
    parser.add_argument("--effect-thresh", default=1.0,  type=float, help="Effect size threshold (Step A)")

    # ML parameters
    parser.add_argument("--random-state", default=42, type=int, help="Random state para reproducibilidade")
    parser.add_argument("--cv-folds",     default=5,  type=int, help="Folds para cross-validation (LASSO)")
    parser.add_argument(
        "--lasso-mode", default="logistic_l1",
        choices=["logistic_l1", "lasso_cv"],
        help="Modo LASSO: logistic_l1 ou lasso_cv",
    )

    return parser


# =====================================================================
# 4. Config dataclass (simple dict-based)
# =====================================================================

def build_config_from_args(args: argparse.Namespace) -> dict:
    return {
        "expr_path":      args.expr_path,
        "test_path":      args.test_path,
        "annot_path":     args.annot_path,
        "output_dir":     args.output_dir,
        "mode":           args.mode,
        "use_combat":     args.use_combat,
        "use_zscore":     args.use_zscore,
        "target_col":     args.target_col,
        "positive_class": args.positive_class,
        "negative_class": args.negative_class,
        "p_val_thresh":   args.p_val_thresh,
        "effect_thresh":  args.effect_thresh,
        "random_state":   args.random_state,
        "cv_folds":       args.cv_folds,
        "lasso_mode":     args.lasso_mode,
    }


def _validate_annotation_file(path: str) -> Optional[str]:
    """Return an error message if the annotation CSV is invalid, else None."""
    import csv as _csv
    try:
        with open(path, encoding="utf-8") as f:
            header = next(_csv.reader(f))
    except Exception as exc:
        return f"Nao foi possivel ler o arquivo: {exc}"
    cols = [c.strip().lower() for c in header]
    if "sample_id" not in cols:
        return f"Coluna 'sample_id' nao encontrada. Colunas detectadas: {header[:8]}"
    if "class_label" not in cols:
        return f"Coluna 'class_label' nao encontrada. Colunas detectadas: {header[:8]}"
    return None


def fill_config_interactively(cfg: dict) -> dict:
    """Fill missing config values through interactive prompts."""

    if not cfg["mode"]:
        cfg["mode"] = interactive_select_mode()

    # ── Expression file ──
    hint_dir = None
    if not cfg["expr_path"]:
        # Suggest from SCRIPT_DIR/data or current dir
        for candidate_dir in [SCRIPT_DIR / "data", Path(".")]:
            suggestion = _suggest_expression_file(candidate_dir)
            if suggestion:
                hint_dir = candidate_dir
                break

        print(f"\n{THICK_SEP}")
        print("  DICA — Arquivo de TREINO suportado (entrada do Estágio 2):")
        print("    • base_treino.csv     ← prioridade 1 (split do Estágio 1, sem vazamento)")
        print("    • merged_expression_raw.csv  ← legado (faz split interno 80/20)")
        print(THICK_SEP)
        cfg["expr_path"] = interactive_select_file("Arquivo de TREINO (base_treino.csv)", hint_dir)

        name = Path(cfg["expr_path"]).name.lower()
        # base_treino.csv vem do Estágio 1: ComBat + z-score já aplicados sem vazamento.
        if "base_treino" in name:
            cfg["use_combat"] = True
            cfg["use_zscore"] = True
        else:
            if "zscore" in name and not cfg["use_zscore"]:
                if interactive_yes_no("  Arquivo parece ser Z-scored. Ativar --use-zscore?", default=True):
                    cfg["use_zscore"] = True
            if "combat" in name and not cfg["use_combat"]:
                cfg["use_combat"] = True

    # ── Test file (base_teste.csv) — deriva ao lado do treino ──
    if not cfg.get("test_path") and cfg["expr_path"]:
        sibling = Path(cfg["expr_path"]).parent / _TEST_NAME
        if Path(cfg["expr_path"]).name.lower().startswith("base_treino") and sibling.exists():
            cfg["test_path"] = str(sibling)
            print(f"  Teste detectado: {sibling.name} (split do Estágio 1)")

    # ── Annotation file ──
    if not cfg["annot_path"]:
        expr_parent = Path(cfg["expr_path"]).parent if cfg["expr_path"] else None
        while True:
            cfg["annot_path"] = interactive_select_file("Arquivo de annotation (CSV)", expr_parent)
            err = _validate_annotation_file(cfg["annot_path"])
            if err is None:
                break
            print(f"\n  [ATENCAO] Arquivo invalido: {err}")
            print(f"  O arquivo de annotation deve conter a coluna 'sample_id' e 'class_label'.")
            print(f"  Arquivo correto sugerido: merged_sample_annotation.csv\n")

    # ── Output directory ──
    if not cfg["output_dir"]:
        cfg["output_dir"] = interactive_select_output_dir()

    # ── LASSO mode ──
    if cfg["mode"] in ("lasso", "both"):
        print(f"\n{SEPARATOR}")
        print(f"  Modo LASSO atual: {cfg['lasso_mode']}")
        if interactive_yes_no("  Alterar modo LASSO?", default=False):
            cfg["lasso_mode"] = interactive_select_lasso_mode()

    # ── Advanced parameters ──
    print(f"\n{SEPARATOR}")
    print("  Parametros avancados (Enter para manter padrao):")
    print(SEPARATOR)

    p_str = _prompt(f"p-val threshold (Step A)", default=str(cfg["p_val_thresh"]))
    e_str = _prompt(f"effect threshold (Step A)", default=str(cfg["effect_thresh"]))
    r_str = _prompt(f"random state", default=str(cfg["random_state"]))

    try:
        cfg["p_val_thresh"]  = float(p_str)
        cfg["effect_thresh"] = float(e_str)
        cfg["random_state"]  = int(r_str)
    except ValueError:
        log.warning("Parametro invalido detectado. Mantendo valores padrao.")

    return cfg


def print_config_summary(cfg: dict) -> None:
    """Print a formatted configuration summary."""
    _print_header("Resumo da Configuracao")
    mode_label = {"boruta": "Boruta", "lasso": "LASSO", "both": "Boruta + LASSO"}
    print(f"  Modo            : {mode_label.get(cfg['mode'], cfg['mode'])}")
    if cfg.get("test_path"):
        print(f"  Treino          : {Path(cfg['expr_path']).name}")
        print(f"  Teste           : {Path(cfg['test_path']).name}  (split do Estágio 1)")
    else:
        print(f"  Expressao       : {Path(cfg['expr_path']).name}  (split interno 80/20)")
    print(f"  Annotation      : {Path(cfg['annot_path']).name}")
    print(f"  Pasta de saida  : {Path(cfg['output_dir']).resolve()}")
    print(f"  ComBat          : {'Sim' if cfg['use_combat'] else 'Nao'}")
    print(f"  Z-Score         : {'Sim' if cfg['use_zscore'] else 'Nao'}")
    print(f"  Target col      : {cfg['target_col']}")
    print(f"  Classe +        : {cfg['positive_class']}")
    print(f"  Classe -        : {cfg['negative_class']}")
    print(f"  p_val thresh    : {cfg['p_val_thresh']}")
    print(f"  effect thresh   : {cfg['effect_thresh']}")
    print(f"  random state    : {cfg['random_state']}")
    if cfg["mode"] in ("lasso", "both"):
        print(f"  LASSO mode      : {cfg['lasso_mode']}")
        print(f"  CV folds        : {cfg['cv_folds']}")
    print(THICK_SEP)


# =====================================================================
# 5. Subprocess Execution
# =====================================================================

def _stream_subprocess(cmd: List[str], label: str) -> Tuple[int, List[str]]:
    """
    Run a subprocess, streaming its combined stdout+stderr to the console
    in real time. Returns (returncode, list_of_output_lines).
    """
    print(f"\n{SEPARATOR}")
    print(f"  Executando: {label}")
    print(f"  Comando: {' '.join(cmd)}")
    print(SEPARATOR)

    output_lines: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            print(f"  {line}")
            output_lines.append(line)
        proc.wait()
        return proc.returncode, output_lines
    except FileNotFoundError as exc:
        msg = f"Script nao encontrado: {exc}"
        print(f"  [ERRO] {msg}")
        log.error(msg)
        return 1, [msg]
    except Exception as exc:
        msg = f"Erro inesperado: {exc}"
        print(f"  [ERRO] {msg}")
        log.error(msg)
        return 1, [msg]


def _build_common_args(cfg: dict) -> List[str]:
    """Build CLI args shared by both downstream scripts.

    Passa o split do Estágio 1 (--train-path/--test-path). Mantém compatibilidade
    com o modo legado (--expr-path) caso test_path não esteja disponível.
    """
    if cfg.get("test_path"):
        args = [
            "--train-path", cfg["expr_path"],
            "--test-path",  cfg["test_path"],
            "--annot-path", cfg["annot_path"],
        ]
    else:
        args = [
            "--expr-path",  cfg["expr_path"],
            "--annot-path", cfg["annot_path"],
        ]
    args += [
        "--target-col",     cfg["target_col"],
        "--positive-class", cfg["positive_class"],
        "--negative-class", cfg["negative_class"],
        "--p-val-thresh",   str(cfg["p_val_thresh"]),
        "--effect-thresh",  str(cfg["effect_thresh"]),
        "--random-state",   str(cfg["random_state"]),
    ]
    if cfg["use_combat"]:
        args.append("--use-combat")
    if cfg["use_zscore"]:
        args.append("--use-zscore")
    return args


def run_boruta(cfg: dict, boruta_out: Path) -> Tuple[bool, dict]:
    """Execute the Boruta pipeline subprocess. Returns (success, summary_dict)."""
    _validate_script_exists(BORUTA_SCRIPT)
    boruta_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(BORUTA_SCRIPT),
        "--output-dir", str(boruta_out),
    ] + _build_common_args(cfg)

    rc, _ = _stream_subprocess(cmd, "Boruta Pipeline")

    if rc != 0:
        log.error(f"Boruta pipeline FALHOU com codigo {rc}.")
        return False, {}

    summary = _read_json_summary(boruta_out / "feature_selection_summary.json")
    log.info(f"Boruta concluido com sucesso. Saida: {boruta_out}")
    return True, summary


def run_lasso(cfg: dict, lasso_out: Path) -> Tuple[bool, dict]:
    """Execute the LASSO pipeline subprocess. Returns (success, summary_dict)."""
    _validate_script_exists(LASSO_SCRIPT)
    lasso_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(LASSO_SCRIPT),
        "--output-dir",  str(lasso_out),
        "--lasso-mode",  cfg["lasso_mode"],
        "--cv-folds",    str(cfg["cv_folds"]),
    ] + _build_common_args(cfg)

    rc, _ = _stream_subprocess(cmd, "LASSO Pipeline")

    if rc != 0:
        log.error(f"LASSO pipeline FALHOU com codigo {rc}.")
        return False, {}

    summary = _read_json_summary(lasso_out / "feature_selection_summary.json")
    log.info(f"LASSO concluido com sucesso. Saida: {lasso_out}")
    return True, summary


# =====================================================================
# 6. Comparison Report
# =====================================================================

def _read_json_summary(path: Path) -> dict:
    if not path.exists():
        log.warning(f"Summary JSON nao encontrado: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_features_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    import csv
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["miRNA"] for row in reader if row.get("miRNA")]


def _validate_script_exists(script: Path) -> None:
    if not script.exists():
        log.error(f"Script nao encontrado: {script}")
        log.error("Certifique-se de que os scripts downstream estao na mesma pasta que o orquestrador.")
        sys.exit(1)


def generate_comparison(
    boruta_out: Path,
    lasso_out: Path,
    comparison_out: Path,
    boruta_summary: dict,
    lasso_summary: dict,
) -> None:
    """Generate comparison files between Boruta and LASSO results."""
    comparison_out.mkdir(parents=True, exist_ok=True)

    boruta_features = _read_features_list(boruta_out / "selected_miRNAs_step_b.csv")
    lasso_features  = _read_features_list(lasso_out  / "selected_miRNAs_step_b.csv")

    set_b = set(boruta_features)
    set_l = set(lasso_features)
    shared   = sorted(set_b & set_l)
    only_b   = sorted(set_b - set_l)
    only_l   = sorted(set_l - set_b)
    all_feat = sorted(set_b | set_l)

    # ── comparison_summary.json ──
    comp_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "expression_file": str(Path(boruta_summary.get("parameters", {}).get("expr_path", "N/A"))),
        "dataset": {
            "samples_total":   boruta_summary.get("samples_total", "N/A"),
            "samples_pdac":    boruta_summary.get("samples_pdac",  "N/A"),
            "samples_control": boruta_summary.get("samples_control", "N/A"),
            "features_initial": boruta_summary.get("features_initial", "N/A"),
        },
        "boruta": {
            "features_step_a": boruta_summary.get("features_after_step_a", len(boruta_features)),
            "features_step_b": boruta_summary.get("features_after_step_b", len(boruta_features)),
            "final_features":  boruta_features,
            "parameters": boruta_summary.get("parameters", {}),
        },
        "lasso": {
            "features_step_a": lasso_summary.get("features_after_step_a", len(lasso_features)),
            "features_step_b": lasso_summary.get("features_after_step_b", len(lasso_features)),
            "final_features":  lasso_features,
            "lasso_meta":      lasso_summary.get("lasso", {}),
            "parameters":      lasso_summary.get("parameters", {}),
        },
        "overlap": {
            "n_shared":    len(shared),
            "n_only_boruta": len(only_b),
            "n_only_lasso":  len(only_l),
            "shared":      shared,
            "only_boruta": only_b,
            "only_lasso":  only_l,
        },
    }

    json_path = comparison_out / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comp_summary, f, indent=4, ensure_ascii=False)

    # ── selected_features_overlap.csv ──
    import csv
    overlap_path = comparison_out / "selected_features_overlap.csv"
    with open(overlap_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["miRNA", "in_boruta", "in_lasso", "shared"])
        writer.writeheader()
        for feat in all_feat:
            writer.writerow({
                "miRNA":     feat,
                "in_boruta": "1" if feat in set_b else "0",
                "in_lasso":  "1" if feat in set_l else "0",
                "shared":    "1" if feat in set_b and feat in set_l else "0",
            })

    # ── comparison_report.txt ──
    n_initial  = boruta_summary.get("features_initial",   "N/A")
    n_samples  = boruta_summary.get("samples_total",      "N/A")
    n_pdac     = boruta_summary.get("samples_pdac",       "N/A")
    n_ctrl     = boruta_summary.get("samples_control",    "N/A")
    b_step_a   = boruta_summary.get("features_after_step_a", len(boruta_features))
    b_step_b   = boruta_summary.get("features_after_step_b", len(boruta_features))
    l_step_a   = lasso_summary.get("features_after_step_a",  len(lasso_features))
    l_step_b   = lasso_summary.get("features_after_step_b",  len(lasso_features))
    lasso_meta = lasso_summary.get("lasso", {})
    lasso_mode = lasso_meta.get("mode", "N/A")
    best_reg   = lasso_meta.get("best_C") or lasso_meta.get("best_alpha", "N/A")

    shared_block  = "\n".join(f"  - {f}" for f in shared)  if shared  else "  (nenhuma)"
    only_b_block  = "\n".join(f"  - {f}" for f in only_b)  if only_b  else "  (nenhuma)"
    only_l_block  = "\n".join(f"  - {f}" for f in only_l)  if only_l  else "  (nenhuma)"

    report_lines = f"""\
================================
Feature Selection Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================

Dataset
-------
Initial features : {n_initial}
Samples          : {n_samples}
PDAC             : {n_pdac}
Control          : {n_ctrl}

Boruta
------
Step A (t-test + FDR) : {b_step_a} features
Step B (Boruta)       : {b_step_b} features
Final list:
{chr(10).join(f'  - {f}' for f in boruta_features) if boruta_features else '  (nenhuma)'}

LASSO ({lasso_mode})
{'-' * (7 + len(lasso_mode))}
Step A (t-test + FDR) : {l_step_a} features
Step B (LASSO)        : {l_step_b} features
Regularizacao         : {best_reg}
Final list:
{chr(10).join(f'  - {f}' for f in lasso_features) if lasso_features else '  (nenhuma)'}

Overlap
-------
Total features union   : {len(all_feat)}
Shared features        : {len(shared)}
Somente Boruta         : {len(only_b)}
Somente LASSO          : {len(only_l)}

Shared list:
{shared_block}

Somente Boruta:
{only_b_block}

Somente LASSO:
{only_l_block}
"""

    report_path = comparison_out / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_lines)

    # Print to console
    print(f"\n{THICK_SEP}")
    print(report_lines)
    print(THICK_SEP)
    log.info(f"Comparacao salva em: {comparison_out.resolve()}")


# =====================================================================
# 7. Orchestration Logic
# =====================================================================

def orchestrate(cfg: dict) -> None:
    """Main orchestration: run selected pipelines and generate comparison."""
    root_out = Path(cfg["output_dir"]).resolve()
    boruta_out     = root_out / "boruta"
    lasso_out      = root_out / "lasso"
    comparison_out = root_out / "comparison"

    mode = cfg["mode"]
    results: Dict[str, Tuple[bool, dict]] = {}

    log.info(THICK_SEP)
    log.info("  Feature Refinement Orchestrator — Iniciando")
    log.info(THICK_SEP)
    log.info(f"  Modo     : {mode}")
    log.info(f"  Saida    : {root_out}")
    log.info(f"  Expressao: {cfg['expr_path']}")
    log.info(f"  Annotation: {cfg['annot_path']}")
    log.info(THICK_SEP)

    # ── Run Boruta ──
    if mode in ("boruta", "both"):
        success, summary = run_boruta(cfg, boruta_out)
        results["boruta"] = (success, summary)
        status = "OK" if success else "FALHOU"
        log.info(f"  [Boruta] Status: {status}")

    # ── Run LASSO ──
    if mode in ("lasso", "both"):
        success, summary = run_lasso(cfg, lasso_out)
        results["lasso"] = (success, summary)
        status = "OK" if success else "FALHOU"
        log.info(f"  [LASSO] Status : {status}")

    # ── Comparison ──
    if mode == "both":
        b_ok, b_summary = results.get("boruta", (False, {}))
        l_ok, l_summary = results.get("lasso",  (False, {}))

        if b_ok and l_ok:
            log.info("  Gerando relatorio de comparacao...")
            generate_comparison(boruta_out, lasso_out, comparison_out, b_summary, l_summary)
        else:
            log.warning("  Comparacao ignorada pois um ou mais pipelines falharam.")

    # ── Final status ──
    _print_header("Execucao Finalizada")
    all_ok = all(ok for ok, _ in results.values())
    if all_ok:
        print(f"  SUCESSO — todos os pipelines concluidos.")
    else:
        print(f"  ATENCAO — um ou mais pipelines falharam. Verifique os logs acima.")

    print(f"\n  Resultados salvos em:")
    if "boruta" in results:
        status = "OK" if results["boruta"][0] else "FALHOU"
        print(f"    boruta/     [{status}] -> {boruta_out}")
    if "lasso" in results:
        status = "OK" if results["lasso"][0] else "FALHOU"
        print(f"    lasso/      [{status}] -> {lasso_out}")
    if mode == "both" and all_ok:
        print(f"    comparison/ [OK] -> {comparison_out}")
    print(THICK_SEP)


# =====================================================================
# 8. Entrypoint
# =====================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)

    # Determine if we are in CLI mode (all required fields provided) or interactive
    cli_mode = all([cfg["expr_path"], cfg["annot_path"], cfg["output_dir"], cfg["mode"]])

    if not cli_mode:
        # ── Interactive mode ──
        _print_header("Feature Refinement System — PDAC miRNA")
        print("  Pipeline de selecao de atributos para miRNA em PDAC.")
        print("  Use Ctrl+C a qualquer momento para cancelar.")
        print(THICK_SEP)

        cfg = fill_config_interactively(cfg)
        print_config_summary(cfg)

        if not _confirm("Confirmar e iniciar execucao?"):
            print("  Cancelado pelo usuario.")
            sys.exit(0)
    else:
        # ── CLI mode ──
        log.info("Modo CLI detectado — pulando prompts interativos.")
        # Deriva o teste ao lado do treino quando não informado explicitamente.
        if not cfg.get("test_path") and cfg["expr_path"]:
            sibling = Path(cfg["expr_path"]).parent / _TEST_NAME
            if Path(cfg["expr_path"]).name.lower().startswith("base_treino") and sibling.exists():
                cfg["test_path"] = str(sibling)
                log.info(f"Teste derivado: {sibling} (split do Estágio 1)")
        # base_treino vem normalizado do Estágio 1 → marcar combat/zscore.
        if cfg["expr_path"] and Path(cfg["expr_path"]).name.lower().startswith("base_treino"):
            cfg["use_combat"] = True
            cfg["use_zscore"] = True
        print_config_summary(cfg)

    # ── Validate input files ──
    files_to_check = [("expr_path", "treino"), ("annot_path", "annotation")]
    if cfg.get("test_path"):
        files_to_check.append(("test_path", "teste"))
    for key, label in files_to_check:
        path = Path(cfg[key])
        if not path.exists():
            log.error(f"Arquivo de {label} nao encontrado: {path}")
            sys.exit(1)

    if not cfg.get("test_path"):
        log.warning(
            "Sem --test-path / base_teste.csv: usando modo LEGADO (split interno 80/20). "
            "Para o fluxo sem vazamento, aponte --expr-path para base_treino.csv do Estágio 1."
        )

    # ── Validate downstream scripts ──
    for script, name in [(BORUTA_SCRIPT, "Boruta"), (LASSO_SCRIPT, "LASSO")]:
        if cfg["mode"] in ("both", "boruta" if name == "Boruta" else "lasso"):
            if not script.exists():
                log.error(f"Script {name} nao encontrado em: {script}")
                log.error("O orquestrador deve estar na mesma pasta que refine_features_pdac.py e refine_features_lasso.py.")
                sys.exit(1)

    orchestrate(cfg)


if __name__ == "__main__":
    main()
