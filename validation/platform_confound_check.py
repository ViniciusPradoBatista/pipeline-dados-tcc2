"""Validação de confundimento de plataforma — prova que o ComBat removeu o sinal de scanner.

Treina um classificador para prever a PLATAFORMA (platform_id), não a classe, a partir
das features de expressão — ANTES e DEPOIS do ComBat. Se o ComBat removeu o efeito de
lote, a plataforma deve ficar (quase) indetectável depois (acurácia ~acaso).

Princípios:
- Usa SOMENTE o conjunto de treino (split=='train'); o teste nunca é tocado.
- Pré-ComBat  = matriz integrada (log2) do treino, pós-merge/interseção.
- Pós-ComBat  = train_corr, a saída de combat_fit (recomputada a partir dos artefatos,
  sem alterar o pipeline do Estágio 1 — apenas chama a função pura combat_fit).
- RandomForest (random_state=42), StratifiedKFold(5). Métricas: acurácia balanceada e AUC.
- Baseline de acaso = proporção da plataforma majoritária no treino.

Uso:
    python validation/platform_confound_check.py --output-root ./out
    (./out deve conter merged_expression_raw.csv e merged_sample_annotation.csv do Estágio 1)
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))
from geo_pipeline.normalize import combat_fit  # noqa: E402

# Fix Windows console encoding (mesmo guard do pipeline).
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("platform_confound")

RANDOM_STATE = 42
# Acima deste nível de acurácia balanceada, a plataforma ainda é detectável pós-ComBat
# (chance = 0.5). Dispara o aviso honesto de "batch não totalmente removido".
_STILL_DETECTABLE = 0.60


def _impute_train_median(mat: np.ndarray) -> np.ndarray:
    """Imputa NaN pela mediana de cada probe (linha) — sobre o TREINO apenas (sem vazamento)."""
    out = mat.copy()
    for i in range(out.shape[0]):
        row = out[i]
        m = np.isnan(row)
        if m.any():
            med = np.nanmedian(row)
            row[m] = med if not np.isnan(med) else 0.0
            out[i] = row
    return out


def _evaluate(X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """RandomForest + StratifiedKFold(5). Retorna acurácia balanceada e AUC médias."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    bal_accs, aucs = [], []
    for tr_idx, va_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        )
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        bal_accs.append(balanced_accuracy_score(y[va_idx], pred))
        try:
            if n_classes == 2:
                proba = clf.predict_proba(X[va_idx])[:, 1]
                aucs.append(roc_auc_score(y[va_idx], proba))
            else:
                proba = clf.predict_proba(X[va_idx])
                aucs.append(roc_auc_score(y[va_idx], proba, multi_class="ovr"))
        except ValueError:
            aucs.append(float("nan"))  # fold sem ambas as classes
    return {
        "balanced_accuracy": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "roc_auc": float(np.nanmean(aucs)),
    }


def run(output_root: Path, output_dir: Path) -> pd.DataFrame:
    raw_path = output_root / "merged_expression_raw.csv"
    annot_path = output_root / "merged_sample_annotation.csv"
    for p in (raw_path, annot_path):
        if not p.exists():
            log.error(f"Artefato do Estágio 1 não encontrado: {p}")
            sys.exit(1)

    merged_raw = pd.read_csv(raw_path)
    annot = pd.read_csv(annot_path)

    if "split" not in annot.columns:
        log.error("merged_sample_annotation.csv não tem coluna 'split'. Rode o Estágio 1 atualizado.")
        sys.exit(1)

    # ── SOMENTE treino ──
    annot_train = annot[annot["split"] == "train"].copy()
    train_ids = set(annot_train["sample_id"])
    train_cols = [c for c in merged_raw.columns if c != "Probe_ID" and c in train_ids]
    # Garantia anti-vazamento: nenhuma amostra de teste entra aqui.
    test_ids = set(annot[annot["split"] == "test"]["sample_id"])
    assert not (set(train_cols) & test_ids), "VAZAMENTO: amostra de teste no conjunto de treino!"

    expr_train = merged_raw.set_index("Probe_ID")[train_cols]
    log.info(f"Treino: {expr_train.shape[0]} probes × {expr_train.shape[1]} amostras")

    # ── Pós-ComBat: train_corr = saída de combat_fit (recomputada, treino apenas) ──
    train_corr, _, _ = combat_fit(expr_train, annot_train)
    probe_index = train_corr.index  # probes que sobreviveram ao filtro de NaN do ComBat

    # ── Pré-ComBat: mesma matriz/treino, MESMO conjunto de probes, antes do ComBat ──
    expr_pre = expr_train.reindex(index=probe_index)

    # Alvo = plataforma, alinhado às colunas (amostras)
    plat_by_sample = annot_train.set_index("sample_id")["platform_id"]
    y_labels = plat_by_sample.reindex(train_cols).values
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    n_classes = len(le.classes_)
    log.info(f"Plataformas no treino: {dict(zip(le.classes_, np.bincount(y)))}")

    # Baseline de acaso = proporção da plataforma majoritária
    counts = np.bincount(y)
    majority_prop = float(counts.max() / counts.sum())
    log.info(f"Baseline (proporção da plataforma majoritária): {majority_prop:.3f}")

    # X = amostras × features (probes). RandomForest é invariante a transformações
    # monotônicas por feature, então o z-score posterior não afeta este teste — por isso
    # comparamos raw (pré) vs train_corr (pós), isolando o efeito do ComBat.
    X_pre = _impute_train_median(expr_pre.values.astype(float)).T
    X_post = train_corr.values.astype(float).T

    log.info("── Avaliando PRÉ-ComBat ──")
    res_pre = _evaluate(X_pre, y, n_classes)
    log.info(f"   bal_acc={res_pre['balanced_accuracy']:.3f}  auc={res_pre['roc_auc']:.3f}")
    log.info("── Avaliando PÓS-ComBat ──")
    res_post = _evaluate(X_post, y, n_classes)
    log.info(f"   bal_acc={res_post['balanced_accuracy']:.3f}  auc={res_post['roc_auc']:.3f}")

    # ── CSV ──
    df = pd.DataFrame(
        [
            {"scenario": "pre_combat", **res_pre},
            {"scenario": "post_combat", **res_post},
            {
                "scenario": "baseline_chance",
                "balanced_accuracy": 0.5,          # acaso para acurácia balanceada
                "balanced_accuracy_std": 0.0,
                "roc_auc": 0.5,                     # acaso para AUC
            },
        ]
    )
    df["majority_platform_proportion"] = majority_prop
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "platform_confound.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Salvo: {csv_path}")

    _plot(res_pre, res_post, majority_prop, output_dir / "platform_confound.png")

    # ── Veredito honesto ──
    print("\n" + "=" * 64)
    print("  CONFUNDIMENTO DE PLATAFORMA — ComBat")
    print("=" * 64)
    print(f"  Pré-ComBat : bal_acc={res_pre['balanced_accuracy']:.3f}  auc={res_pre['roc_auc']:.3f}")
    print(f"  Pós-ComBat : bal_acc={res_post['balanced_accuracy']:.3f}  auc={res_post['roc_auc']:.3f}")
    print(f"  Acaso      : bal_acc=0.500  (majoritária={majority_prop:.3f})")
    print("=" * 64)
    if res_post["balanced_accuracy"] > _STILL_DETECTABLE:
        print("  [ACHADO] A plataforma AINDA é detectável pós-ComBat "
              f"(bal_acc={res_post['balanced_accuracy']:.3f} > {_STILL_DETECTABLE}).")
        print("  O efeito de lote NÃO foi totalmente removido. Resultado honesto — investigar.")
    else:
        print("  [OK] Pós-ComBat a plataforma caiu para ~acaso: efeito de lote removido.")
    print("=" * 64)
    return df


def _plot(res_pre: dict, res_post: dict, majority_prop: float, save_path: Path) -> None:
    metrics = ["balanced_accuracy", "roc_auc"]
    labels = ["Acurácia balanceada", "AUC"]
    pre = [res_pre[m] for m in metrics]
    post = [res_post[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - w / 2, pre, w, label="Pré-ComBat", color="#c0392b")
    b2 = ax.bar(x + w / 2, post, w, label="Pós-ComBat", color="#27ae60")
    ax.axhline(0.5, ls="--", color="gray", lw=1.5, label="Acaso (0.5)")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (5-fold CV, treino)")
    ax.set_title("Detectabilidade da plataforma antes vs. depois do ComBat\n"
                 "(quanto mais perto do acaso pós-ComBat, melhor)")
    ax.legend(loc="lower center")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Salvo: {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Confundimento de plataforma — validação do ComBat")
    ap.add_argument("--output-root", required=True,
                    help="Pasta com os artefatos do Estágio 1 (merged_expression_raw.csv + merged_sample_annotation.csv)")
    ap.add_argument("--output-dir", default=None,
                    help="Onde salvar platform_confound.csv/.png (padrão: --output-root)")
    args = ap.parse_args()
    out_root = Path(args.output_root)
    out_dir = Path(args.output_dir) if args.output_dir else out_root
    run(out_root, out_dir)


if __name__ == "__main__":
    main()
