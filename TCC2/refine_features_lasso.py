"""
===================================================================
  Feature Refinement with LASSO (L1 Regularization)
  For PDAC miRNA Classification

  Step A: Welch t-test + FDR (Benjamini-Hochberg)
  Step B: LASSO via LogisticRegressionCV (L1) ou LassoCV

  IMPORTANTE — Data Leakage Prevention:
    Steps A e B aplicados EXCLUSIVAMENTE no conjunto de TREINO
    (80% das amostras, split estratificado). O conjunto de TESTE
    permanece completamente oculto durante a seleção de features.

    O StandardScaler é aplicado via Pipeline do sklearn, garantindo
    que a normalização seja aprendida apenas nos dados de treino
    dentro de cada fold do cross-validation interno.

  Author: TCC Pipeline Downstream - LASSO variant
===================================================================
"""

import argparse
import json
import logging
import sys
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lasso_refinement")


# =====================================================================
# 1. Interactive File Selection
# =====================================================================

def _try_file_picker(title: str, filetypes: list) -> Optional[str]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path if path else None
    except Exception:
        return None


def _try_folder_picker(title: str) -> Optional[str]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path if path else None
    except Exception:
        return None


def interactive_select_file(label: str) -> str:
    print(f"\n{'─'*50}")
    print(f"  Selecionar: {label}")
    print(f"{'─'*50}")
    print("  [1] Abrir seletor de arquivos")
    print("  [2] Digitar caminho manualmente")
    print(f"{'─'*50}")

    while True:
        try:
            choice = input("  Escolha (1/2) [padrao: 1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if choice == "1":
            ft = [("CSV files", "*.csv"), ("All files", "*.*")]
            path = _try_file_picker(f"Selecione: {label}", ft)
            if path:
                print(f"  OK: {Path(path).name}")
                return path
            print("  Seletor nao disponivel. Digite o caminho:")
            choice = "2"

        if choice == "2":
            try:
                path = input("  Caminho do arquivo: ").strip().strip('"')
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if path and Path(path).exists():
                print(f"  OK: {Path(path).name}")
                return path
            print("  Arquivo nao encontrado. Tente novamente.")


def interactive_select_output() -> str:
    print(f"\n{'─'*50}")
    print("  Onde salvar os resultados?")
    print(f"{'─'*50}")
    print("  [1] Pasta atual (.)")
    print("  [2] Escolher pasta via seletor")
    print("  [3] Digitar caminho manualmente")
    print(f"{'─'*50}")

    while True:
        try:
            choice = input("  Escolha (1/2/3) [padrao: 1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            return "."

        if choice == "1":
            return "."
        elif choice == "2":
            path = _try_folder_picker("Selecione pasta de saida")
            return path if path else "."
        elif choice == "3":
            try:
                path = input("  Caminho da pasta: ").strip().strip('"')
            except (EOFError, KeyboardInterrupt):
                return "."
            return path if path else "."


# =====================================================================
# 2. CLI Parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature Selection with LASSO for PDAC miRNA Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-path", default=None, type=str,
                        help="base_treino.csv (expression CSV, Probe_ID × samples).")
    parser.add_argument("--test-path", default=None, type=str,
                        help="base_teste.csv (expression CSV, Probe_ID × samples).")
    parser.add_argument("--expr-path", default=None, type=str,
                        help="[LEGADO] Matriz única; split interno. Use --train-path/--test-path.")
    parser.add_argument("--annot-path", default=None, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--use-combat", action="store_true")
    parser.add_argument("--use-zscore", action="store_true")
    parser.add_argument("--target-col", default="class_label", type=str)
    parser.add_argument("--positive-class", default="PDAC", type=str)
    parser.add_argument("--negative-class", default="Control", type=str)
    parser.add_argument("--p-val-thresh", default=0.05, type=float)
    parser.add_argument("--effect-thresh", default=1.0, type=float)
    parser.add_argument("--lasso-mode", default="logistic_l1", type=str,
                        choices=["logistic_l1", "lasso_cv"])
    parser.add_argument("--cv-folds", default=5, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float,
                        help="Fracao de amostras reservada para o conjunto de teste (default: 0.2).")
    return parser.parse_args()


# =====================================================================
# 3. Data Loading and Alignment
# =====================================================================

def load_and_align_data(
    expr_path: str, annot_path: str,
    target_col: str, pos_class: str, neg_class: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    log.info(f"Loading expression: {expr_path}")
    expr_df = pd.read_csv(expr_path)

    log.info(f"Loading annotation: {annot_path}")
    annot_df = pd.read_csv(annot_path)

    if "Probe_ID" not in expr_df.columns:
        raise ValueError("Expression CSV must have a 'Probe_ID' column.")
    if "sample_id" not in annot_df.columns:
        raise ValueError("Annotation CSV must have a 'sample_id' column.")
    if target_col not in annot_df.columns:
        raise ValueError(f"Annotation CSV must have '{target_col}' column.")

    log.info("Transposing expression matrix (samples x features)...")
    expr_df = expr_df.set_index("Probe_ID")
    X_raw = expr_df.T
    X_raw.index.name = "sample_id"
    X_raw.reset_index(inplace=True)

    log.info("Aligning by sample_id...")
    merged = pd.merge(
        annot_df[["sample_id", target_col]], X_raw,
        on="sample_id", how="inner",
    )
    merged = merged[merged[target_col].isin([pos_class, neg_class])].copy()

    if merged.empty:
        raise ValueError("No matching samples after alignment + class filter.")

    merged.set_index("sample_id", inplace=True)
    y = merged[target_col].map({neg_class: 0, pos_class: 1})
    X = merged.drop(columns=[target_col])

    log.info(f"Aligned: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y


def load_presplit_data(
    train_path: str, test_path: str, annot_path: str,
    target_col: str, pos_class: str, neg_class: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the Stage-1 train/test split (no internal re-split)."""
    X_train, y_train = load_and_align_data(
        train_path, annot_path, target_col, pos_class, neg_class
    )
    X_test, y_test = load_and_align_data(
        test_path, annot_path, target_col, pos_class, neg_class
    )
    return X_train, X_test, y_train, y_test


def validate_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    log.info("Validating data...")

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        log.warning(f"Filling {nan_count} NaN values with column median.")
        X = X.fillna(X.median())

    if len(y.unique()) < 2:
        raise ValueError(f"Need 2 classes, found: {y.unique().tolist()}")

    zero_var = X.columns[X.var() == 0]
    if len(zero_var) > 0:
        log.warning(f"Removing {len(zero_var)} zero-variance features.")
        X = X.drop(columns=zero_var)

    if X.columns.duplicated().any():
        n_dup = X.columns.duplicated().sum()
        log.warning(f"Removing {n_dup} duplicate columns.")
        X = X.loc[:, ~X.columns.duplicated()]

    log.info(f"Validation done: {X.shape}")
    return X


# =====================================================================
# 4. Step A: Statistical Filtering
# =====================================================================

def calculate_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    if v1 == 0 and v2 == 0:
        return 0.0
    ps = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0.0


def differential_expression_filter(
    X: pd.DataFrame, y: pd.Series,
    p_thresh: float, eff_thresh: float, use_zscore: bool,
) -> pd.DataFrame:
    """
    Step A: Welch t-test + FDR Benjamini-Hochberg.

    IMPORTANTE: recebe apenas o CONJUNTO DE TREINO.
    """
    log.info("── Step A: Statistical Filtering (apenas dados de TREINO) ──")
    log.info(f"  Thresholds: p_adj < {p_thresh}, |effect| > {eff_thresh}")

    pdac_mask = y == 1
    ctrl_mask = y == 0
    rows = []

    for feat in X.columns:
        pv = X.loc[pdac_mask, feat].values
        cv = X.loc[ctrl_mask, feat].values
        _, p_val = stats.ttest_ind(pv, cv, equal_var=False, nan_policy="omit")
        mp, mc = np.nanmean(pv), np.nanmean(cv)

        if use_zscore:
            eff = calculate_cohens_d(pv, cv)
            eff_col = "Cohen_d"
        else:
            eff = mp - mc
            eff_col = "delta_expression"

        rows.append({
            "miRNA": feat, "p_value": p_val,
            "mean_pdac": mp, "mean_control": mc,
            eff_col: eff, "abs_effect": abs(eff),
        })

    df = pd.DataFrame(rows)
    df["p_value"] = df["p_value"].fillna(1.0)
    _, p_adj, _, _ = multipletests(df["p_value"], alpha=p_thresh, method="fdr_bh")
    df.insert(2, "p_adj", p_adj)
    df["selected_step_a"] = (df["p_adj"] < p_thresh) & (df["abs_effect"] > eff_thresh)

    n_sel = df["selected_step_a"].sum()
    log.info(f"  Step A selected {n_sel} / {len(X.columns)} features.")
    return df


# =====================================================================
# 5. Step B: LASSO Selection
# =====================================================================

def lasso_logistic_l1(
    X: pd.DataFrame, y: pd.Series,
    cv_folds: int, random_state: int,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """
    LASSO via LogisticRegressionCV com penalidade L1.

    O StandardScaler é integrado em um Pipeline do sklearn, garantindo
    que a média e o desvio padrão sejam aprendidos apenas nos dados de
    treino de cada fold interno do cross-validation — sem data leakage.

    IMPORTANTE: recebe apenas o CONJUNTO DE TREINO.
    """
    log.info("── Step B: Logistic Regression L1 (LASSO) — apenas dados de TREINO ──")
    log.info(f"  CV folds: {cv_folds}, solver: saga")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            cv=cv_folds,
            class_weight="balanced",
            random_state=random_state,
            max_iter=10000,
            scoring="roc_auc",
            Cs=20,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X.values, y.values)

    model = pipeline.named_steps["clf"]
    best_C = float(model.C_[0])
    coefs = model.coef_[0]
    log.info(f"  Best C (inverse regularization): {best_C:.6f}")

    coef_df = pd.DataFrame({
        "miRNA": X.columns,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    nonzero_mask = coef_df["abs_coefficient"] > 0
    selected = coef_df.loc[nonzero_mask, "miRNA"].tolist()
    n_zero = int((~nonzero_mask).sum())
    n_pos = int((coef_df["coefficient"] > 0).sum())
    n_neg = int((coef_df["coefficient"] < 0).sum())

    log.info(f"  Non-zero coefficients: {len(selected)}")
    log.info(f"  Zero coefficients: {n_zero}")
    log.info(f"  Positive: {n_pos}, Negative: {n_neg}")

    meta = {
        "mode": "logistic_l1",
        "best_C": best_C,
        "nonzero": len(selected),
        "zero": n_zero,
        "positive": n_pos,
        "negative": n_neg,
    }
    return selected, coef_df, meta


def lasso_cv_regression(
    X: pd.DataFrame, y: pd.Series,
    cv_folds: int, random_state: int,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """
    Alternativa: LassoCV (regressão linear).

    AVISO CIENTIFICO: LassoCV é um método de REGRESSÃO LINEAR. Para
    classificação binária (PDAC vs Control), o LogisticRegressionCV com
    L1 (logistic_l1) é cientificamente mais adequado, pois respeita a
    natureza binária da variável resposta. Use este modo apenas para
    ranking exploratório de features, NUNCA para avaliação de modelos.

    O StandardScaler é aplicado via Pipeline para evitar data leakage.

    IMPORTANTE: recebe apenas o CONJUNTO DE TREINO.
    """
    log.info("── Step B: LassoCV (modo regressão) — apenas dados de TREINO ──")
    log.warning("  AVISO CIENTIFICO: LassoCV e um metodo de regressao linear.")
    log.warning("  Para classificacao binaria, logistic_l1 e mais adequado.")
    log.warning("  Use este modo apenas para ranking exploratorio de features.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv_folds, random_state=random_state, max_iter=10000, n_jobs=-1)),
    ])

    pipeline.fit(X.values, y.values)

    model = pipeline.named_steps["lasso"]
    best_alpha = float(model.alpha_)
    coefs = model.coef_
    log.info(f"  Best alpha: {best_alpha:.6f}")

    coef_df = pd.DataFrame({
        "miRNA": X.columns,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    nonzero_mask = coef_df["abs_coefficient"] > 0
    selected = coef_df.loc[nonzero_mask, "miRNA"].tolist()

    log.info(f"  Non-zero coefficients: {len(selected)}")

    meta = {
        "mode": "lasso_cv",
        "best_alpha": best_alpha,
        "nonzero": len(selected),
        "zero": int((~nonzero_mask).sum()),
    }
    return selected, coef_df, meta


# =====================================================================
# 6. Main
# =====================================================================

def main():
    args = parse_args()

    use_presplit = bool(args.train_path and args.test_path)

    if not use_presplit and not args.expr_path:
        print("\n" + "=" * 55)
        print("  LASSO Feature Selection for PDAC miRNA")
        print("=" * 55)
        args.expr_path = interactive_select_file("Arquivo de expressao (CSV)")

    if not args.annot_path:
        args.annot_path = interactive_select_file("Arquivo de annotation (CSV)")

    if not args.output_dir:
        args.output_dir = interactive_select_output()

    # Validar existência dos arquivos de entrada
    paths_to_check = (
        [args.train_path, args.test_path] if use_presplit else [args.expr_path]
    ) + [args.annot_path]
    for p in paths_to_check:
        if not p or not Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    if use_presplit:
        print(f"  Train     : {Path(args.train_path).name}")
        print(f"  Test      : {Path(args.test_path).name}")
    else:
        print(f"  Expressao : {Path(args.expr_path).name}")
    print(f"  Annotation: {Path(args.annot_path).name}")
    print(f"  Saida     : {out_dir.absolute()}")
    print(f"  LASSO mode: {args.lasso_mode}")
    print(f"  CV folds  : {args.cv_folds}")
    print(f"  Z-Score   : {args.use_zscore}")
    print(f"{'='*55}")

    log.info("=" * 60)
    log.info("  LASSO Feature Refinement Pipeline")
    log.info("=" * 60)

    # ── 1. Load ──
    if use_presplit:
        log.info("Usando split do Estágio 1 (sem re-split interno).")
        X_train_raw, X_test_raw, y_train, y_test = load_presplit_data(
            args.train_path, args.test_path, args.annot_path,
            args.target_col, args.positive_class, args.negative_class,
        )
        n_pdac = int((y_train == 1).sum()) + int((y_test == 1).sum())
        n_ctrl = int((y_train == 0).sum()) + int((y_test == 0).sum())
        initial_features = X_train_raw.shape[1]
        log.info(f"Classes: PDAC={n_pdac}, Control={n_ctrl}")

        X_train = validate_data(X_train_raw, y_train)
        X_test = X_test_raw.reindex(columns=X_train.columns)
    else:
        X_raw, y = load_and_align_data(
            args.expr_path, args.annot_path,
            args.target_col, args.positive_class, args.negative_class,
        )
        initial_features = X_raw.shape[1]
        n_pdac = int((y == 1).sum())
        n_ctrl = int((y == 0).sum())
        log.info(f"Classes: PDAC={n_pdac}, Control={n_ctrl}")

        X = validate_data(X_raw, y)

        log.info("── Split estratificado 80/20 (evita data leakage) ──")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size,
            random_state=args.random_state, stratify=y,
        )
    log.info(
        f"Train: {X_train.shape[0]} amostras "
        f"(PDAC={int((y_train==1).sum())}, Control={int((y_train==0).sum())})"
    )
    log.info(
        f"Test:  {X_test.shape[0]} amostras "
        f"(PDAC={int((y_test==1).sum())}, Control={int((y_test==0).sum())})"
    )

    # ── 4. Step A (SOMENTE nos dados de treino) ──
    stats_df = differential_expression_filter(
        X_train, y_train, args.p_val_thresh, args.effect_thresh, args.use_zscore,
    )
    features_a = stats_df.loc[stats_df["selected_step_a"], "miRNA"].tolist()

    if len(features_a) == 0:
        log.error("=" * 60)
        log.error("FALHA NO STEP A: Nenhuma feature passou o filtro FDR + efeito!")
        log.error("Possíveis causas:")
        log.error("  - Dados insuficientes (poucos samples no treino)")
        log.error("  - Threshold muito restritivo (tente reduzir p-val-thresh ou effect-thresh)")
        log.error("  - Ausência de sinal biológico real neste dataset")
        log.error("FALLBACK: usando Top-50 features por p_adj (NÃO são estatisticamente significativas).")
        log.error("Os resultados downstream devem ser interpretados com extrema cautela.")
        log.error("=" * 60)
        features_a = stats_df.sort_values("p_adj")["miRNA"].head(50).tolist()

    X_train_a = X_train[features_a]
    log.info(f"Step A output: {len(features_a)} features")

    # ── 5. Step B: LASSO (SOMENTE nos dados de treino) ──
    if args.lasso_mode == "logistic_l1":
        features_b, coef_df, lasso_meta = lasso_logistic_l1(
            X_train_a, y_train, args.cv_folds, args.random_state,
        )
    else:
        features_b, coef_df, lasso_meta = lasso_cv_regression(
            X_train_a, y_train, args.cv_folds, args.random_state,
        )

    if len(features_b) == 0:
        log.warning("LASSO selecionou 0 features. Usando features do Step A como fallback.")
        features_b = features_a

    # ── 6. Aplicar features ao treino E ao teste ──
    X_train_final = X_train[features_b].copy()
    X_test_final = X_test[features_b].copy()

    # ── 7. Save outputs ──
    log.info("── Saving Outputs ──")

    stats_df.to_csv(out_dir / "differential_expression_results.csv", index=False)
    pd.Series(features_a, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_a.csv", index=False)
    pd.Series(features_b, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_b.csv", index=False)
    coef_df.to_csv(out_dir / "lasso_coefficients.csv", index=False)

    # Conjunto de treino
    train_ds = X_train_final.copy()
    train_ds["target"] = y_train.values
    train_ds["split"] = "train"
    train_ds.to_csv(out_dir / "base_treino.csv", index=True)

    # Conjunto de teste (mantido oculto durante seleção de features)
    test_ds = X_test_final.copy()
    test_ds["target"] = y_test.values
    test_ds["split"] = "test"
    test_ds.to_csv(out_dir / "base_teste.csv", index=True)

    # Base combinada
    full_ds = pd.concat([train_ds, test_ds])
    full_ds.to_csv(out_dir / "base_pronta_para_treinamento.csv", index=True)

    with open(out_dir / "selected_feature_names.txt", "w") as f:
        f.write("\n".join(features_b))

    reg_label = lasso_meta.get("best_C") or lasso_meta.get("best_alpha", "N/A")
    report = (
        f"Feature Selection Report (LASSO)\n"
        f"================================\n"
        f"Initial miRNAs        : {initial_features}\n"
        f"PDAC Samples (total)  : {n_pdac}\n"
        f"Control Samples (total): {n_ctrl}\n"
        f"Train samples         : {X_train.shape[0]}\n"
        f"Test samples          : {X_test.shape[0]}\n"
        f"Remaining after Step A: {len(features_a)}\n"
        f"Remaining after Step B: {len(features_b)}\n"
        f"LASSO mode            : {args.lasso_mode}\n"
        f"Chosen regularization : {reg_label}\n"
        f"Non-zero coefficients : {lasso_meta['nonzero']}\n"
        f"\nNOTA: Step A e Step B foram aplicados APENAS no conjunto de treino.\n"
        f"O conjunto de teste em base_teste.csv deve ser usado para avaliacao\n"
        f"nao-enviesada do classificador downstream.\n"
    )
    with open(out_dir / "feature_counts_report.txt", "w") as f:
        f.write(report)

    summary = {
        "samples_total": int(X_train.shape[0] + X_test.shape[0]),
        "samples_pdac": n_pdac,
        "samples_control": n_ctrl,
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(X_test.shape[0]),
        "features_initial": initial_features,
        "features_after_step_a": len(features_a),
        "features_after_step_b": len(features_b),
        "lasso": lasso_meta,
        "parameters": {
            "p_val_thresh": args.p_val_thresh,
            "effect_thresh": args.effect_thresh,
            "use_zscore": args.use_zscore,
            "use_combat": args.use_combat,
            "lasso_mode": args.lasso_mode,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "test_size": args.test_size,
        },
    }
    with open(out_dir / "feature_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    log.info(report)
    log.info(f"Arquivos salvos em: {out_dir.absolute()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
