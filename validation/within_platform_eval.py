"""Avaliação within-platform (Toray/GSE59856) — discriminação PDAC vs. saudável imune ao confound.

Motivação: o platform_confound_check mostrou que a plataforma continua detectável pós-ComBat
(acc 1.00; PCA(2)→0.93). Como plataforma confunde com classe (Affymetrix 82% PDAC, Toray 40%
PDAC), a performance do modelo na matriz FUNDIDA pode estar inflada por "plataforma = atalho
para classe". Aqui avaliamos DENTRO de uma única plataforma (Toray), onde plataforma é
constante e não pode ser o atalho — uma estimativa diagnóstica imune a esse confound.

Disciplina anti-vazamento (idêntica ao Estágio 1):
- Usa SOMENTE amostras Toray (platform_id).
- SEM ComBat (uma plataforma só, sem batch cross-platform).
- Parte da matriz pós-merge ANTES do ComBat (merged_expression_raw.csv) — não arrasta
  parâmetros estimados com a Affymetrix presente.
- z-score por probe e seleção de features são fitados DENTRO de cada fold de treino.

NÃO altera o Estágio 1 nem seus artefatos — apenas os lê.

Uso:
    python validation/within_platform_eval.py --output-root ./out [--n-repeats 5]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import warnings
from pathlib import Path

# t-test em probes quase-constantes (pós z-score) gera aviso de precisão benigno.
warnings.filterwarnings("ignore", message="Precision loss occurred")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("within_platform")

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_REPEATS = 5
P_THRESH = 0.05          # FDR (Benjamini-Hochberg)
EFFECT_THRESH = 1.0      # |Cohen's d| (paridade com o Estágio 2)
STABILITY_PI = 0.5       # frequência mínima de seleção p/ um miRNA entrar num balde

TORAY_PLATFORM = "GPL18941"
POS, NEG = "PDAC", "Control"


# =====================================================================
# Seleção de features (mesma lógica do Estágio 2), por fold de treino
# =====================================================================

def _cohens_d_vec(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    n1, n2 = pos.shape[0], neg.shape[0]
    v1, v2 = pos.var(axis=0, ddof=1), neg.var(axis=0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    pooled[pooled == 0] = np.nan
    d = (pos.mean(axis=0) - neg.mean(axis=0)) / pooled
    return np.nan_to_num(d, nan=0.0)


def step_a_mask(Xz: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Welch t-test + FDR(BH) + |Cohen's d| > thresh. Recebe matriz JÁ z-scorada (treino)."""
    pos, neg = Xz[y == 1], Xz[y == 0]
    with np.errstate(all="ignore"):
        _, p = stats.ttest_ind(pos, neg, axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    d = _cohens_d_vec(pos, neg)
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p, alpha=P_THRESH, method="fdr_bh")
    mask = (p_adj < P_THRESH) & (np.abs(d) > EFFECT_THRESH)
    if mask.sum() == 0:  # fallback: top-50 por p_adj (como o Estágio 2)
        order = np.argsort(p_adj)[:50]
        mask = np.zeros_like(mask)
        mask[order] = True
    return mask


def boruta_select(Xz: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Índices selecionados por Boruta (sobre o subconjunto já filtrado pelo Step A)."""
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                max_depth=5, random_state=RANDOM_STATE)
    sel = BorutaPy(rf, n_estimators="auto", random_state=RANDOM_STATE, max_iter=40, verbose=0)
    sel.fit(Xz, y)
    return np.where(sel.support_)[0]


def lasso_select(Xz: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Índices com coeficiente não-nulo no LASSO (LogReg L1), sobre o subconjunto do Step A."""
    clf = LogisticRegressionCV(penalty="l1", solver="saga", cv=3, Cs=10,
                               class_weight="balanced", scoring="roc_auc",
                               max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xz, y)
    return np.where(np.abs(clf.coef_[0]) > 0)[0]


# =====================================================================
# Modelos (estruturado p/ aceitar SVM/XGBoost depois)
# =====================================================================

def make_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "LogReg_L2": LogisticRegression(
            penalty="l2", class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE),
    }


def _metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": sens,
        "specificity": spec,
    }


# =====================================================================
# Núcleo: CV within-Toray
# =====================================================================

def run_within_toray(X: np.ndarray, y: np.ndarray, feature_names: list, n_repeats: int):
    """CV repetida; z-score e seleção fitados por fold. Retorna (df_metrics, freq_boruta, freq_lasso)."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    models = make_models()
    rows = []
    n_feat = len(feature_names)
    sel_boruta = np.zeros(n_feat, dtype=int)
    sel_lasso = np.zeros(n_feat, dtype=int)
    n_folds = 0

    for fold, (tr, va) in enumerate(rskf.split(X, y)):
        n_folds += 1
        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]

        # 1. imputação (mediana do TREINO do fold) + z-score (μ/σ do TREINO do fold)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i, Xva_i = imp.transform(Xtr), imp.transform(Xva)
        sc = StandardScaler().fit(Xtr_i)
        Xtr_z, Xva_z = sc.transform(Xtr_i), sc.transform(Xva_i)

        # 2. Step A (no treino do fold)
        mask_a = step_a_mask(Xtr_z, ytr)
        a_idx = np.where(mask_a)[0]

        # 2b. estabilidade: Boruta e LASSO sobre o subconjunto do Step A
        try:
            b_local = boruta_select(Xtr_z[:, a_idx], ytr)
            sel_boruta[a_idx[b_local]] += 1
        except Exception as exc:
            log.warning(f"Boruta falhou no fold {fold}: {exc}")
        try:
            l_local = lasso_select(Xtr_z[:, a_idx], ytr)
            sel_lasso[a_idx[l_local]] += 1
        except Exception as exc:
            log.warning(f"LASSO falhou no fold {fold}: {exc}")

        # 3. métricas: classificador treinado no painel do Step A (estável, não-wrapper)
        for mname, model in make_models().items():
            model.fit(Xtr_z[:, a_idx], ytr)
            proba = model.predict_proba(Xva_z[:, a_idx])[:, 1]
            pred = model.predict(Xva_z[:, a_idx])
            m = _metrics(yva, proba, pred)
            m.update({"scenario": "within_toray", "model": mname, "fold": fold,
                      "n_features": len(a_idx)})
            rows.append(m)

    df = pd.DataFrame(rows)
    freq_boruta = sel_boruta / n_folds
    freq_lasso = sel_lasso / n_folds
    return df, freq_boruta, freq_lasso


# =====================================================================
# Controle negativo: permutação de rótulos (prova ausência de vazamento)
# =====================================================================

def permutation_control(X: np.ndarray, y: np.ndarray, n_perm: int = 5):
    """CV com rótulos REAIS vs. EMBARALHADOS. Se a CV é limpa, o embaralhado cai p/ ~0.5.
    Usa LogReg + Step A (sem Boruta) para ser rápido. Mesma disciplina fit-no-fold."""
    def _cv(yv: np.ndarray) -> float:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr, va in skf.split(X, yv):
            imp = SimpleImputer(strategy="median").fit(X[tr])
            Xtr, Xva = imp.transform(X[tr]), imp.transform(X[va])
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
            idx = np.where(step_a_mask(Xtr, yv[tr]))[0]
            clf = LogisticRegression(penalty="l2", class_weight="balanced",
                                     max_iter=5000, random_state=RANDOM_STATE)
            clf.fit(Xtr[:, idx], yv[tr])
            aucs.append(roc_auc_score(yv[va], clf.predict_proba(Xva[:, idx])[:, 1]))
        return float(np.mean(aucs))

    real = _cv(y)
    rng = np.random.RandomState(RANDOM_STATE)
    perms = []
    for _ in range(n_perm):
        ys = y.copy()
        rng.shuffle(ys)
        perms.append(_cv(ys))
    return real, float(np.mean(perms)), perms


# =====================================================================
# Referência: modelo FUNDIDO (held-out em base_teste)
# =====================================================================

def run_fused_reference(output_root: Path):
    """Treina nos base_treino (combat+zscore, fit-no-treino) e avalia no base_teste held-out.
    Retorna (df_metrics, fused_selected_set)."""
    bt = pd.read_csv(output_root / "base_treino.csv").set_index("Probe_ID")
    bv = pd.read_csv(output_root / "base_teste.csv").set_index("Probe_ID")
    annot = pd.read_csv(output_root / "merged_sample_annotation.csv").set_index("sample_id")

    def xy(mat):
        X = mat.T  # samples × probes
        y = annot.loc[X.index, "class_label"].map({NEG: 0, POS: 1})
        keep = y.notna()
        return X.loc[keep].values.astype(float), y[keep].values.astype(int), X.columns.tolist()

    Xtr, ytr, feats = xy(bt)
    Xte, yte, _ = xy(bv)
    # base_* já são combat+zscore (fit no treino) — sem re-normalizar. Imputa eventual NaN.
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)

    mask_a = step_a_mask(Xtr, ytr)  # já z-scorado pelo Estágio 1
    a_idx = np.where(mask_a)[0]

    rows = []
    for mname, model in make_models().items():
        model.fit(Xtr[:, a_idx], ytr)
        proba = model.predict_proba(Xte[:, a_idx])[:, 1]
        pred = model.predict(Xte[:, a_idx])
        m = _metrics(yte, proba, pred)
        m.update({"scenario": "fused_heldout", "model": mname, "fold": "heldout",
                  "n_features": len(a_idx)})
        rows.append(m)

    # conjunto selecionado na matriz fundida (p/ marcar also_in_fused)
    fused_selected = set()
    try:
        fused_selected |= {feats[a_idx[i]] for i in boruta_select(Xtr[:, a_idx], ytr)}
    except Exception as exc:
        log.warning(f"Boruta (fundido) falhou: {exc}")
    try:
        fused_selected |= {feats[a_idx[i]] for i in lasso_select(Xtr[:, a_idx], ytr)}
    except Exception as exc:
        log.warning(f"LASSO (fundido) falhou: {exc}")
    return pd.DataFrame(rows), fused_selected


# =====================================================================
# Agregação, saídas, gráfico
# =====================================================================

def _agg(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]
    out = []
    for mname, g in df.groupby("model"):
        n = len(g)
        for met in metrics:
            mean, std = g[met].mean(), g[met].std(ddof=1)
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            out.append({"model": mname, "metric": met, "mean": mean, "std": std,
                        "ci95_low": mean - ci, "ci95_high": mean + ci, "n_folds": n})
    return pd.DataFrame(out)


def _plot(agg: pd.DataFrame, fused: pd.DataFrame, save_path: Path) -> None:
    models = sorted(agg["model"].unique())
    metrics = ["roc_auc", "pr_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6), sharey=True)
    for ax, met in zip(axes, metrics):
        x = np.arange(len(models))
        means = [agg[(agg.model == m) & (agg.metric == met)]["mean"].values[0] for m in models]
        cis = [agg[(agg.model == m) & (agg.metric == met)]["ci95_high"].values[0] -
               agg[(agg.model == m) & (agg.metric == met)]["mean"].values[0] for m in models]
        ax.bar(x, means, yerr=cis, capsize=6, color="#2980b9", label="within-Toray (CV, IC95%)")
        for i, m in enumerate(models):
            fv = fused[(fused.model == m) & (fused.scenario == "fused_heldout")][met]
            if len(fv):
                ax.scatter([x[i]], [fv.values[0]], color="#c0392b", zorder=5, s=90,
                           marker="D", label="fundido (held-out)" if i == 0 else None)
        # linha de acaso para PR-AUC = prevalência de PDAC no Toray (~0.40)
        if met == "pr_auc":
            ax.axhline(100 / 250, ls=":", color="gray", label="acaso PR (0.40)")
        if met == "roc_auc":
            ax.axhline(0.5, ls=":", color="gray", label="acaso ROC (0.50)")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
        ax.set_title(met.upper()); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.suptitle("Discriminação PDAC vs. saudável: within-Toray (imune ao confound) vs. fundido")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Salvo: {save_path}")


def run(output_root: Path, output_dir: Path, n_repeats: int) -> None:
    raw = pd.read_csv(output_root / "merged_expression_raw.csv")
    annot = pd.read_csv(output_root / "merged_sample_annotation.csv")

    # ── SOMENTE Toray ──
    tor = annot[annot["platform_id"] == TORAY_PLATFORM]
    tor = tor[tor["class_label"].isin([POS, NEG])]
    batches = tor["dataset_id"].unique().tolist()
    if len(batches) > 1:
        log.warning(f"Toray tem múltiplos dataset_id (sub-lotes?): {batches} — logado, não imputado.")
    log.info(f"Toray: {len(tor)} amostras | classes: {tor['class_label'].value_counts().to_dict()} "
             f"| batch(es): {batches}")

    tor_ids = [c for c in raw.columns if c != "Probe_ID" and c in set(tor["sample_id"])]
    # Garantia: nenhuma amostra Affymetrix entra
    affy_ids = set(annot[annot["platform_id"] != TORAY_PLATFORM]["sample_id"])
    assert not (set(tor_ids) & affy_ids), "VAZAMENTO: amostra não-Toray no subconjunto!"

    expr = raw.set_index("Probe_ID")[tor_ids]
    feature_names = expr.index.tolist()
    X = expr.values.astype(float).T  # samples × probes
    y = tor.set_index("sample_id")["class_label"].reindex(tor_ids).map({NEG: 0, POS: 1}).values.astype(int)
    log.info(f"Matriz within-Toray: {X.shape[0]} amostras × {X.shape[1]} probes "
             f"(PDAC={int(y.sum())}, Control={int((y == 0).sum())})")

    # ── within-Toray CV ──
    log.info(f"── CV within-Toray (RepeatedStratifiedKFold {N_SPLITS}×{n_repeats}) ──")
    df_within, freq_b, freq_l = run_within_toray(X, y, feature_names, n_repeats)

    # ── controle negativo de vazamento (permutação de rótulos) ──
    log.info("── Controle negativo: permutação de rótulos (esperado: ~0.5) ──")
    perm_real, perm_shuf_mean, perm_list = permutation_control(X, y, n_perm=5)
    log.info(f"   AUC real={perm_real:.3f}  |  rótulos embaralhados (média)={perm_shuf_mean:.3f}")

    # ── fundido (referência held-out) ──
    log.info("── Referência: modelo fundido (held-out em base_teste) ──")
    df_fused, fused_selected = run_fused_reference(output_root)

    # ── métricas ──
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_within = _agg(df_within)
    df_perm = pd.DataFrame([
        {"scenario": "permutation_real", "model": "LogReg_L2", "fold": "cv", "roc_auc": perm_real},
        {"scenario": "permutation_shuffled", "model": "LogReg_L2", "fold": "cv_mean",
         "roc_auc": perm_shuf_mean},
    ])
    metrics_csv = pd.concat([df_within, df_fused, df_perm], ignore_index=True)
    metrics_csv.to_csv(output_dir / "within_toray_metrics.csv", index=False)
    agg_within.to_csv(output_dir / "within_toray_metrics_aggregated.csv", index=False)
    log.info(f"Salvo: {output_dir/'within_toray_metrics.csv'} (+ agregado)")

    # ── estabilidade de features (3 baldes; boruta_only preservado) ──
    in_b = freq_b >= STABILITY_PI
    in_l = freq_l >= STABILITY_PI
    buckets = []
    for i, name in enumerate(feature_names):
        if not (in_b[i] or in_l[i]):
            continue
        if in_b[i] and in_l[i]:
            bucket = "shared"
        elif in_b[i]:
            bucket = "boruta_only"
        else:
            bucket = "lasso_only"
        buckets.append({"miRNA": name, "freq_boruta": round(float(freq_b[i]), 3),
                        "freq_lasso": round(float(freq_l[i]), 3), "bucket": bucket,
                        "also_in_fused": int(name in fused_selected)})
    stab = pd.DataFrame(buckets).sort_values(
        ["bucket", "freq_boruta", "freq_lasso"], ascending=[True, False, False])
    stab.to_csv(output_dir / "within_toray_feature_stability.csv", index=False)
    log.info(f"Salvo: {output_dir/'within_toray_feature_stability.csv'}")

    # ── gráfico ──
    _plot(agg_within, df_fused, output_dir / "within_toray_auc_comparison.png")

    # ── relatório honesto ──
    def grab(df, model, met, col="mean"):
        r = df[(df.model == model) & (df.metric == met)]
        return float(r[col].values[0]) if len(r) else float("nan")

    print("\n" + "=" * 72)
    print("  WITHIN-TORAY (imune ao confound de PLATAFORMA)  vs.  FUNDIDO (held-out)")
    print("=" * 72)
    for model in sorted(df_within["model"].unique()):
        w_auc = grab(agg_within, model, "roc_auc")
        w_lo = grab(agg_within, model, "roc_auc", "ci95_low")
        w_hi = grab(agg_within, model, "roc_auc", "ci95_high")
        w_pr = grab(agg_within, model, "pr_auc")
        f_auc = df_fused[(df_fused.model == model)]["roc_auc"]
        f_auc = float(f_auc.values[0]) if len(f_auc) else float("nan")
        f_pr = df_fused[(df_fused.model == model)]["pr_auc"]
        f_pr = float(f_pr.values[0]) if len(f_pr) else float("nan")
        print(f"  {model}:")
        print(f"     within-Toray ROC-AUC = {w_auc:.3f} (IC95% {w_lo:.3f}-{w_hi:.3f}) | PR-AUC = {w_pr:.3f}")
        print(f"     fundido      ROC-AUC = {f_auc:.3f} (held-out)               | PR-AUC = {f_pr:.3f}")
        gap = f_auc - w_auc
        if gap > w_hi - w_auc + 0.03:
            print(f"     -> fundido ACIMA do within-Toray (delta={gap:+.3f}): excesso provavel = confound de PLATAFORMA.")
        else:
            print(f"     -> fundido <= within-Toray (delta={gap:+.3f}): confound de PLATAFORMA nao inflou.")
    n_shared = int((stab.bucket == "shared").sum()) if len(stab) else 0
    n_bonly = int((stab.bucket == "boruta_only").sum()) if len(stab) else 0
    n_lonly = int((stab.bucket == "lasso_only").sum()) if len(stab) else 0
    n_fused = int(stab.also_in_fused.sum()) if len(stab) else 0
    print(f"  Controle de vazamento: AUC real={perm_real:.3f} vs rotulos embaralhados="
          f"{perm_shuf_mean:.3f} -> {'OK (sem vazamento)' if perm_shuf_mean < 0.6 else 'ALERTA'}")
    print(f"  Features estaveis (freq>={STABILITY_PI}): {n_shared} shared, "
          f"{n_bonly} boruta_only, {n_lonly} lasso_only ({n_fused} tambem no fundido).")
    print("  ESCOPO: esta analise neutraliza o confound de PLATAFORMA (Toray constante).")
    print("  RESSALVA CRITICA (confound de coleta): em GSE59856, casos e controles foram")
    print("    processados em LOTES DE ARRAY separados (0 lotes mistos) -> scan-batch")
    print("    perfeitamente confundido com a classe, INSEPARAVEL nestes dados. Logo o AUC")
    print("    nao se atribui a biologia pura. Ver validation/collection_batch_check.py +")
    print("    validacao biologica (concordancia com biomarcadores conhecidos).")
    print("  within-Toray e CV (mean+/-IC); fundido e held-out unico -> comparacao DIRECIONAL.")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação within-platform (Toray) imune ao confound")
    ap.add_argument("--output-root", required=True, help="Pasta com artefatos do Estágio 1")
    ap.add_argument("--output-dir", default=None, help="Onde salvar saídas (padrão: --output-root)")
    ap.add_argument("--n-repeats", type=int, default=DEFAULT_REPEATS, help="Repeats do RepeatedStratifiedKFold")
    args = ap.parse_args()
    out_root = Path(args.output_root)
    out_dir = Path(args.output_dir) if args.output_dir else out_root
    run(out_root, out_dir, args.n_repeats)


if __name__ == "__main__":
    main()
