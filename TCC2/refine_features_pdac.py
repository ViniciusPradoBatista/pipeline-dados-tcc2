"""
===================================================================
  GEO miRNA Cross-Platform Pipeline - Downstream Feature Refinement
  For PDAC (Pancreatic Ductal Adenocarcinoma) Research

  This script performs a 2-step feature selection on the output of 
  the GEO miRNA integration pipeline:
    Step A: Statistical Filtering (Welch's t-test + FDR correction)
    Step B: Machine Learning Selection (Boruta algorithm)

  Inputs expected:
    - Expression matrix (features in rows, samples in columns)
    - Sample annotation file

  Author: TCC Pipeline Downstream
===================================================================
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier

try:
    from boruta import BorutaPy
except ImportError:
    print("❌ ERROR: 'boruta' package not found. Please install it using: pip install boruta")
    sys.exit(1)

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
                        help="Path to the merged expression CSV file (e.g., merged_expression_combat.csv).")
    parser.add_argument("--annot-path", required=True, type=str,
                        help="Path to the sample annotation CSV file (e.g., merged_sample_annotation.csv).")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory to save the results.")
    
    parser.add_argument("--use-combat", action="store_true",
                        help="Flag indicating if the data is ComBat-corrected (affects logging/naming).")
    parser.add_argument("--use-zscore", action="store_true",
                        help="Flag indicating if the data is Z-scored (changes effect size to Cohen's d).")
    
    parser.add_argument("--target-col", default="class_label", type=str,
                        help="Column in annotation file that contains the classes (default: class_label).")
    parser.add_argument("--positive-class", default="PDAC", type=str,
                        help="Name of the positive class in the target column (default: PDAC).")
    parser.add_argument("--negative-class", default="Control", type=str,
                        help="Name of the negative class in the target column (default: Control).")
    
    parser.add_argument("--p-val-thresh", default=0.05, type=float,
                        help="FDR adjusted p-value threshold for Step A (default: 0.05).")
    parser.add_argument("--effect-thresh", default=1.0, type=float,
                        help="Absolute effect size threshold for Step A (default: 1.0).")
    
    parser.add_argument("--random-state", default=42, type=int,
                        help="Random state for reproducibility (default: 42).")
    
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
    
    # ── Verify expression format ──
    if "Probe_ID" not in expr_df.columns:
        raise ValueError("Expression matrix must contain a 'Probe_ID' column.")
    
    # ── Verify annotation format ──
    if "sample_id" not in annot_df.columns:
        raise ValueError("Annotation file must contain a 'sample_id' column.")
    if target_col not in annot_df.columns:
        raise ValueError(f"Annotation file must contain target column '{target_col}'.")
    
    # Transpose expression matrix: features in columns, samples in rows
    log.info("Transposing expression matrix to (samples x features)...")
    expr_df = expr_df.set_index("Probe_ID")
    X_raw = expr_df.T
    X_raw.index.name = "sample_id"
    X_raw.reset_index(inplace=True)
    
    # Align by sample_id
    log.info("Aligning expression and annotation data by 'sample_id'...")
    merged_df = pd.merge(annot_df[["sample_id", target_col]], X_raw, on="sample_id", how="inner")
    
    # Filter valid classes
    valid_classes = [pos_class, neg_class]
    merged_df = merged_df[merged_df[target_col].isin(valid_classes)].copy()
    
    if merged_df.empty:
        raise ValueError("No matching samples found between expression and annotation for the specified classes.")
    
    # Create X and y
    merged_df.set_index("sample_id", inplace=True)
    y_raw = merged_df[target_col]
    X = merged_df.drop(columns=[target_col])
    
    # Convert y to binary
    y = y_raw.map({neg_class: 0, pos_class: 1})
    
    log.info(f"Initial alignment complete. Extracted {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series):
    """
    Performs consistency checks on the loaded data.
    """
    log.info("Running data consistency checks...")
    
    # Check for missing values
    if X.isna().sum().sum() > 0:
        log.warning("Found NaN values in expression matrix. Filling with median.")
        X.fillna(X.median(), inplace=True)
        
    # Check for multiple classes
    if len(y.unique()) < 2:
        raise ValueError(f"Target vector must have at least 2 classes. Found: {y.unique()}")
        
    # Check for zero variance features
    variances = X.var()
    zero_var_cols = variances[variances == 0].index
    if len(zero_var_cols) > 0:
        log.warning(f"Found {len(zero_var_cols)} features with zero variance. Removing them.")
        X.drop(columns=zero_var_cols, inplace=True)
        
    # Check for duplicate columns
    if X.columns.duplicated().any():
        dup_cols = X.columns[X.columns.duplicated()].tolist()
        log.warning(f"Found duplicate feature columns: {dup_cols}. Keeping the first instance.")
        X = X.loc[:, ~X.columns.duplicated()]
        
    log.info(f"Validation complete. Matrix shape ready for analysis: {X.shape}")
    return X


# =====================================================================
# 3. Step A: Statistical Filtering
# =====================================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d for standardized effect size."""
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
    Performs Step A filtering using Welch's t-test and FDR correction.
    """
    log.info("── Step A: Statistical Filtering ──")
    log.info(f"Parameters: p_adj < {p_val_thresh}, |effect| > {effect_thresh}")
    
    mask_pdac = y == 1
    mask_control = y == 0
    
    X_pdac = X[mask_pdac]
    X_control = X[mask_control]
    
    results = []
    
    for feature in X.columns:
        pdac_vals = X_pdac[feature].values
        control_vals = X_control[feature].values
        
        # Welch's t-test
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
    
    # Handle NaNs in p_value before FDR
    res_df["p_value"].fillna(1.0, inplace=True)
    
    # Benjamini-Hochberg FDR correction
    _, p_adj, _, _ = multipletests(res_df["p_value"], alpha=p_val_thresh, method="fdr_bh")
    res_df.insert(2, "p_adj", p_adj)
    
    # Filtering logic
    effect_name_col = "Cohen_d" if use_zscore else "delta_expression"
    
    res_df["selected_step_a"] = (res_df["p_adj"] < p_val_thresh) & (res_df["abs_effect"] > effect_thresh)
    
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
    Performs Step B filtering using Boruta.
    """
    log.info("── Step B: Boruta Machine Learning Selection ──")
    
    if X_filtered.shape[1] == 0:
        log.error("No features passed to Boruta!")
        return []
        
    rf = RandomForestClassifier(
        n_jobs=-1, 
        class_weight="balanced", 
        random_state=random_state,
        max_depth=5 # Limits depth for Boruta stability
    )
    
    boruta_selector = BorutaPy(
        rf, 
        n_estimators='auto', 
        verbose=0, 
        random_state=random_state,
        max_iter=100
    )
    
    log.info(f"Running Boruta on {X_filtered.shape[1]} features and {X_filtered.shape[0]} samples...")
    boruta_selector.fit(X_filtered.values, y.values)
    
    # Support holds boolean array of confirmed features
    selected_features = X_filtered.columns[boruta_selector.support_].tolist()
    
    # Tentative features
    tentative_features = X_filtered.columns[boruta_selector.support_weak_].tolist()
    
    log.info(f"Boruta Confirmed features: {len(selected_features)}")
    log.info(f"Boruta Tentative features: {len(tentative_features)}")
    
    # We return only confirmed features
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
    pdac_count = (y == 1).sum()
    control_count = (y == 0).sum()
    
    log.info(f"Class distribution -> PDAC (1): {pdac_count}, Control (0): {control_count}")
    
    # ── 2. Validate ──
    X = validate_data(X_raw, y)
    
    # ── 3. Step A: Stat Filter ──
    stats_df = differential_expression_filter(
        X, y, 
        p_val_thresh=args.p_val_thresh, 
        effect_thresh=args.effect_thresh, 
        use_zscore=args.use_zscore
    )
    
    selected_a_mask = stats_df["selected_step_a"]
    features_step_a = stats_df.loc[selected_a_mask, "miRNA"].tolist()
    
    if len(features_step_a) == 0:
        log.warning("⚠️ Step A resulted in 0 features! Applying fallback: Top 100 features by p_adj.")
        stats_df_sorted = stats_df.sort_values(by="p_adj")
        features_step_a = stats_df_sorted["miRNA"].head(100).tolist()
        
    X_step_a = X[features_step_a]
    
    # ── 4. Step B: Boruta ──
    features_step_b = boruta_selection_step_b(X_step_a, y, args.random_state)
    
    if len(features_step_b) == 0:
        log.warning("⚠️ Boruta selected 0 features. Falling back to Step A features.")
        features_step_b = features_step_a
        
    X_final = X[features_step_b].copy()
    
    # ── 5. Generate Outputs ──
    log.info("── Saving Outputs ──")
    
    # 5.1. Stat results
    stats_csv_path = out_dir / "differential_expression_results.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    
    # 5.2. Selected features A
    pd.Series(features_step_a, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_a.csv", index=False)
    
    # 5.3. Selected features B
    pd.Series(features_step_b, name="miRNA").to_csv(out_dir / "selected_miRNAs_step_b.csv", index=False)
    
    # 5.4. Training ready base
    final_dataset = X_final.copy()
    final_dataset["target"] = y.values
    final_csv_path = out_dir / "base_pronta_para_treinamento.csv"
    final_dataset.to_csv(final_csv_path, index=True) # Keep sample_id as index
    
    # 5.5. Text files
    with open(out_dir / "selected_feature_names.txt", "w") as f:
        f.write("\n".join(features_step_b))
        
    report = (
        f"Feature Selection Report\n"
        f"========================\n"
        f"Initial miRNAs        : {initial_features}\n"
        f"PDAC Samples          : {pdac_count}\n"
        f"Control Samples       : {control_count}\n"
        f"Remaining after Step A: {len(features_step_a)}\n"
        f"Remaining after Step B: {len(features_step_b)}\n"
    )
    with open(out_dir / "feature_counts_report.txt", "w") as f:
        f.write(report)
        
    # 5.6. JSON Summary
    summary = {
        "samples_total": int(X.shape[0]),
        "samples_pdac": int(pdac_count),
        "samples_control": int(control_count),
        "features_initial": int(initial_features),
        "features_after_step_a": int(len(features_step_a)),
        "features_after_step_b": int(len(features_step_b)),
        "parameters": {
            "p_val_thresh": args.p_val_thresh,
            "effect_thresh": args.effect_thresh,
            "use_zscore": args.use_zscore,
            "use_combat": args.use_combat,
            "random_state": args.random_state
        }
    }
    with open(out_dir / "feature_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    log.info(report)
    log.info(f"✅ All files saved to: {out_dir.absolute()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
