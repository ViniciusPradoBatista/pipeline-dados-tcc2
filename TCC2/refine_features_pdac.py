"""
===================================================================
  GEO miRNA Cross-Platform Pipeline - Downstream Feature Refinement
  For PDAC (Pancreatic Ductal Adenocarcinoma) Research

  This script performs a 2-step feature selection on the output of
  the GEO miRNA integration pipeline:
    Step A: Statistical Filtering (Welch's t-test + FDR correction)
    Step B: Machine Learning Selection (Boruta algorithm)

  IMPORTANTE — Data Leakage Prevention:
    Os Steps A e B são aplicados EXCLUSIVAMENTE no conjunto de TREINO
    (80% das amostras, split estratificado). O conjunto de TESTE (20%)
    é mantido completamente oculto durante a seleção de features.
    Isso garante que métricas de desempenho downstream sejam válidas
    e generalizáveis.

  Inputs expected:
    - Expression matrix (features in rows, samples in columns)
    - Sample annotation file

  Author: TCC Pipeline Downstream
===================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_refinement")


# =====================================================================
# 1. CLI Parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature Selection for PDAC miRNA Data")

    parser.add_argument("--expr-path", required=True, type=str,
                        help="Path to the merged expression CSV file.")
    parser.add_argument("--annot-path", required=True, type=str,
                        help="Path to the sample annotation CSV file.")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory to save the results.")

    parser.add_argument("--use-combat", action="store_true")
    parser.add_argument("--use-zscore", action="store_true")

    parser.add_argument("--target-col", default="class_label", type=str)
    parser.add_argument("--positive-class", default="PDAC", type=str)
    parser.add_argument("--negative-class", default="Control", type=str)

    parser.add_argument("--p-val-thresh", default=0.05, type=float)
    parser.add_argument("--effect-thresh", default=1.0, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float,
                        help="Fraction of samples reserved for test set (default: 0.2 = 20%%).")

    return parser.parse_args()


# =====================================================================
# 2. Data Loading and Alignment
# =====================================================================

def load_and_align_data(
    expr_path: str,
    annot_path: str,
    target_col: str,
    pos_class: str,
    neg_class: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads expression and annotation data, transposes expression matrix,
    aligns them by sample_id, and encodes the target variable.

    Returns:
        X (pd.DataFrame): Samples x Features matrix.
        y (pd.Series): Binary target vector (0=Control, 1=PDAC).
    """
    log.info(f"Loading expression data from {expr_path}")
    expr_df = pd.read_csv(expr_path)

    log.info(f"Loading annotation data from {annot_path}")
    annot_df = pd.read_csv(annot_path)

    probe_col = next(
        (c for c in ("Probe_ID", "Probe_ID_Canonical") if c in expr_df.columns),
        None,
    )
    if probe_col is None:
        raise ValueError("Expression matrix must contain a 'Probe_ID' or 'Probe_ID_Canonical' column.")

    if "sample_id" not in annot_df.columns:
        raise ValueError("Annotation file must contain a 'sample_id' column.")
    if target_col not in annot_df.columns:
        raise ValueError(f"Annotation file must contain target column '{target_col}'.")

    log.info("Transposing expression matrix to (samples x features)...")
    expr_df = expr_df.set_index(probe_col)
    X_raw = expr_df.T
    X_raw.index.name = "sample_id"
    X_raw.reset_index(inplace=True)

    log.info("Aligning expression and annotation data by 'sample_id'...")
    merged_df = pd.merge(annot_df[["sample_id", target_col]], X_raw, on="sample_id", how="inner")

    valid_classes = [pos_class, neg_class]
    merged_df = merged_df[merged_df[target_col].isin(valid_classes)].copy()

    if merged_df.empty:
        raise ValueError("No matching samples found between expression and annotation for the specified classes.")

    merged_df.set_index("sample_id", inplace=True)
    y_raw = merged_df[target_col]
    X = merged_df.drop(columns=[target_col])
    y = y_raw.map({neg_class: 0, pos_class: 1})

    log.info(f"Initial alignment complete. {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Performs consistency checks and basic cleaning on the data."""
    log.info("Running data consistency checks...")

    if X.isna().sum().sum() > 0:
        log.warning("Found NaN values in expression matrix. Filling with column median.")
        X = X.fillna(X.median())

    if len(y.unique()) < 2:
        raise ValueError(f"Target vector must have at least 2 classes. Found: {y.unique()}")

    variances = X.var()
    zero_var_cols = variances[variances == 0].index
    if len(zero_var_cols) > 0:
        log.warning(f"Found {len(zero_var_cols)} features with zero variance. Removing them.")
        X = X.drop(columns=zero_var_cols)

    if X.columns.duplicated().any():
        dup_cols = X.columns[X.columns.duplicated()].tolist()
        log.warning(f"Found duplicate feature columns: {dup_cols}. Keeping the first instance.")
        X = X.loc[:, ~X.columns.duplicated()]

    log.info(f"Validation complete. Matrix shape: {X.shape}")
    return X


# =====================================================================
# 3. Step A: Statistical Filtering
# =====================================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size (pooled std, ddof=1)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if var1 == 0 and var2 == 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def differential_expression_filter(
    X: pd.DataFrame,
    y: pd.Series,
    p_val_thresh: float,
    effect_thresh: float,
    use_zscore: bool
) -> pd.DataFrame:
    """
    Step A: Welch's t-test + Benjamini-Hochberg FDR.

    IMPORTANTE: Esta função deve receber apenas o CONJUNTO DE TREINO.
    Aplicar o filtro no dataset completo constitui data leakage.
    """
    log.info("── Step A: Statistical Filtering (apenas em dados de TREINO) ──")
    log.info(f"Parameters: p_adj < {p_val_thresh}, |effect| > {effect_thresh}")

    mask_pdac = y == 1
    mask_control = y == 0

    X_pdac = X[mask_pdac]
    X_control = X[mask_control]

    results = []

    for feature in X.columns:
        pdac_vals = X_pdac[feature].values
        control_vals = X_control[feature].values

        t_stat, p_val = stats.ttest_ind(pdac_vals, control_vals, equal_var=False, nan_policy='omit')

        mean_p = np.nanmean(pdac_vals)
        mean_c = np.nanmean(control_vals)

        if use_zscore:
            effect_size = calculate_cohens_d(pdac_vals, control_vals)
            effect_name = "Cohen_d"
        else:
            effect_size = mean_p - mean_c
            effect_name = "delta_expression"

        results.append({
            "miRNA": feature,
            "p_value": p_val,
            "mean_pdac": mean_p,
            "mean_control": mean_c,
            effect_name: effect_size,
            "abs_effect": abs(effect_size)
        })

    res_df = pd.DataFrame(results)
    res_df["p_value"] = res_df["p_value"].fillna(1.0)

    _, p_adj, _, _ = multipletests(res_df["p_value"], alpha=p_val_thresh, method="fdr_bh")
    res_df.insert(2, "p_adj", p_adj)

    res_df["selected_step_a"] = (
        (res_df["p_adj"] < p_val_thresh) & (res_df["abs_effect"] > effect_thresh)
    )

    selected_count = res_df["selected_step_a"].sum()
    log.info(f"Statistical filtering selected {selected_count} / {len(X.columns)} features.")

    return res_df


# =====================================================================
# 4. Step B: Boruta Selection
# =====================================================================

def boruta_selection_step_b(
    X_filtered: pd.DataFrame,
    y: pd.Series,
    random_state: int
) -> List[str]:
    """
    Step B: Boruta feature selection.

    IMPORTANTE: Esta função deve receber apenas o CONJUNTO DE TREINO.

    Usa max_depth=None conforme recomendado pela literatura do BorutaPy:
    árvores sem restrição de profundidade capturam melhor interações
    complexas entre features.
    """
    log.info("── Step B: Boruta Machine Learning Selection (apenas em dados de TREINO) ──")

    if X_filtered.shape[1] == 0:
        log.error("No features passed to Boruta!")
        return []

    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
        max_depth=None,  # sem restrição de profundidade (recomendado pelo BorutaPy)
    )

    boruta_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        random_state=random_state,
        max_iter=100
    )

    log.info(f"Running Boruta on {X_filtered.shape[1]} features, {X_filtered.shape[0]} training samples...")
    boruta_selector.fit(X_filtered.values, y.values)

    selected_features = X_filtered.columns[boruta_selector.support_].tolist()
    tentative_features = X_filtered.columns[boruta_selector.support_weak_].tolist()

    log.info(f"Boruta Confirmed features: {len(selected_features)}")
    log.info(f"Boruta Tentative features: {len(tentative_features)}")

    return selected_features


# =====================================================================
# 5. Main Workflow
# =====================================================================

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Starting Downstream Feature Refinement Pipeline")
    log.info("=" * 60)
    log.info(f"Expression: {args.expr_path}")
    log.info(f"Annotation: {args.annot_path}")
    log.info(f"Output Dir: {out_dir}")
    log.info(f"Data Mode:  Z-Score={args.use_zscore}, ComBat={args.use_combat}")
    log.info(f"Test size:  {args.test_size*100:.0f}% (holdout estratificado)")
    log.info("=" * 60)

    # ── 1. Load Data ──
    X_raw, y = load_and_align_data(
        args.expr_path,
        args.annot_path,
        args.target_col,
        args.positive_class,
        args.negative_class
    )

    initial_features = X_raw.shape[1]
    pdac_count = int((y == 1).sum())
    control_count = int((y == 0).sum())
    log.info(f"Class distribution -> PDAC (1): {pdac_count}, Control (0): {control_count}")

    # ── 2. Validate ──
    X = validate_data(X_raw, y)

    # ── 3. Holdout Split (ANTES de qualquer seleção de features) ──
    log.info("── Split estratificado 80/20 (evita data leakage) ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    log.info(
        f"Train: {X_train.shape[0]} amostras "
        f"(PDAC={int((y_train==1).sum())}, Control={int((y_train==0).sum())})"
    )
    log.info(
        f"Test:  {X_test.shape[0]} amostras "
        f"(PDAC={int((y_test==1).sum())}, Control={int((y_test==0).sum())})"
    )

    # ── 4. Step A: Stat Filter (SOMENTE nos dados de treino) ──
    stats_df = differential_expression_filter(
        X_train, y_train,
        p_val_thresh=args.p_val_thresh,
        effect_thresh=args.effect_thresh,
        use_zscore=args.use_zscore
    )

    selected_a_mask = stats_df["selected_step_a"]
    features_step_a = stats_df.loc[selected_a_mask, "miRNA"].tolist()

    if len(features_step_a) == 0:
        log.error("=" * 60)
        log.error("FALHA NO STEP A: Nenhuma feature passou o filtro FDR + efeito!")
        log.error("Possíveis causas:")
        log.error("  - Dados insuficientes (poucos samples no treino)")
        log.error("  - Threshold muito restritivo (tente reduzir p-val-thresh ou effect-thresh)")
        log.error("  - Ausência de sinal biológico real neste dataset")
        log.error("FALLBACK: usando Top-50 features por p_adj (NÃO são estatisticamente significativas).")
        log.error("Os resultados downstream devem ser interpretados com cautela.")
        log.error("=" * 60)
        features_step_a = stats_df.sort_values(by="p_adj")["miRNA"].head(50).tolist()

    X_train_a = X_train[features_step_a]

    # ── 5. Step B: Boruta (SOMENTE nos dados de treino) ──
    features_step_b = boruta_selection_step_b(X_train_a, y_train, args.random_state)

    if len(features_step_b) == 0:
        log.warning("Boruta selecionou 0 features. Usando features do Step A como fallback.")
        features_step_b = features_step_a

    # ── 6. Aplicar features selecionadas ao treino E ao teste ──
    X_train_final = X_train[features_step_b].copy()
    X_test_final = X_test[features_step_b].copy()

    # ── 7. Generate Outputs ──
    log.info("── Saving Outputs ──")

    # Resultados estatísticos
    stats_df.to_csv(out_dir / "differential_expression_results.csv", index=False)

    # Listas de features selecionadas
    pd.Series(features_step_a, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_a.csv", index=False)
    pd.Series(features_step_b, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_b.csv", index=False)

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

    # Base combinada (com coluna 'split' indicando treino/teste)
    full_ds = pd.concat([train_ds, test_ds])
    full_ds.to_csv(out_dir / "base_pronta_para_treinamento.csv", index=True)

    # Nomes das features selecionadas
    with open(out_dir / "selected_feature_names.txt", "w") as f:
        f.write("\n".join(features_step_b))

    report = (
        f"Feature Selection Report\n"
        f"========================\n"
        f"Initial miRNAs        : {initial_features}\n"
        f"PDAC Samples (total)  : {pdac_count}\n"
        f"Control Samples (total): {control_count}\n"
        f"Train samples         : {X_train.shape[0]}\n"
        f"Test samples          : {X_test.shape[0]}\n"
        f"Remaining after Step A: {len(features_step_a)}\n"
        f"Remaining after Step B: {len(features_step_b)}\n"
        f"\nNOTA: Step A e Step B foram aplicados APENAS no conjunto de treino.\n"
        f"O conjunto de teste em base_teste.csv deve ser usado para avaliação\n"
        f"não-enviesada do classificador downstream.\n"
    )
    with open(out_dir / "feature_counts_report.txt", "w") as f:
        f.write(report)

    summary = {
        "samples_total": int(X.shape[0]),
        "samples_pdac": pdac_count,
        "samples_control": control_count,
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(X_test.shape[0]),
        "features_initial": int(initial_features),
        "features_after_step_a": int(len(features_step_a)),
        "features_after_step_b": int(len(features_step_b)),
        "parameters": {
            "p_val_thresh": args.p_val_thresh,
            "effect_thresh": args.effect_thresh,
            "use_zscore": args.use_zscore,
            "use_combat": args.use_combat,
            "random_state": args.random_state,
            "test_size": args.test_size,
        }
    }
    with open(out_dir / "feature_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    log.info(report)
    log.info(f"Arquivos salvos em: {out_dir.absolute()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
