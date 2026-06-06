"""run_all.py — executa Estágio 1 + Estágio 2 sob UM único diretório de saída.

Conveniência: orquestra os pontos de entrada existentes (NÃO reimplementa lógica) e
organiza tudo num output-root com subpastas claras, deixando as bases prontas para o
treino dos modelos em `bases_modelo/`. Gera um MANIFEST.md descrevendo cada artefato.

Estrutura produzida:
    <output-root>/
    ├── MANIFEST.md
    ├── 1_integracao/        (Estágio 1: base_treino/teste, anotação, ComBat, PCA…)
    ├── 2_selecao_features/  (Estágio 2: boruta/, lasso/, comparison/)
    └── bases_modelo/        (bases prontas p/ o treino: completas e por seletor)

A sequência de dados (merge → split → ComBat → z-score, tudo fit-no-treino) é a do
Estágio 1 — este script só define ONDE os arquivos são escritos.

Uso:
    python TCC2/run_all.py --output-root ./saida_pipeline
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from geo_mirna_pipeline import run_pipeline  # ponto de entrada existente do Estágio 1

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_DATA = SCRIPT_DIR / "data"
_DEFAULT_FILES = [
    str(_DATA / "GSE85589_series_matrix.txt"),
    str(_DATA / "GSE59856_series_matrix.txt"),
]
_DEFAULT_CONDITION_FILTER = ["pancreatic cancer", "healthy control"]
_DEFAULT_CLASS_MAP = {"pancreatic cancer": "PDAC", "healthy control": "Control"}


def _dims(path: Path) -> str:
    """Retorna 'linhas × colunas' de um CSV, ou '—' se não aplicável."""
    try:
        df = pd.read_csv(path)
        return f"{df.shape[0]} × {df.shape[1]}"
    except Exception:
        return "—"


def _stage1(integ: Path, files, condition_filter, class_map, no_plots: bool) -> None:
    print(f"\n=== ESTÁGIO 1 → {integ} ===")
    run_pipeline(
        files=files,
        output_root=integ,
        no_interactive=True,
        condition_filter=condition_filter,
        class_map=class_map,
        no_plots=no_plots,
    )
    # organiza os PCA pngs sob 1_integracao/pca/ (só relocação de arquivo)
    pngs = list(integ.glob("pca_*.png"))
    if pngs:
        pca_dir = integ / "pca"
        pca_dir.mkdir(exist_ok=True)
        for p in pngs:
            shutil.move(str(p), str(pca_dir / p.name))


def _stage2(integ: Path, feat: Path, mode: str) -> bool:
    print(f"\n=== ESTÁGIO 2 → {feat} (mode={mode}) ===")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_feature_refinement_system.py"),
        "--expr-path", str(integ / "base_treino.csv"),
        "--annot-path", str(integ / "merged_sample_annotation.csv"),
        "--output-dir", str(feat),
        "--mode", mode,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[AVISO] Estágio 2 retornou código {rc}. Veja os logs acima.")
    return rc == 0


def _assemble_bases(integ: Path, feat: Path, bases: Path) -> dict:
    """Junta as bases prontas para o treino num único lugar (cópias)."""
    bases.mkdir(parents=True, exist_ok=True)
    copied = {}
    # bases completas (todas as features) — orientação Probe_ID × amostras
    for name in ("base_treino.csv", "base_teste.csv", "merged_sample_annotation.csv"):
        src = integ / name
        if src.exists():
            shutil.copy2(src, bases / name)
            copied[name] = bases / name
    # bases por seletor (só features selecionadas) — orientação amostras × features (+target,+split)
    for sel in ("boruta", "lasso"):
        for split in ("treino", "teste"):
            src = feat / sel / f"base_{split}.csv"
            if src.exists():
                dst = bases / f"base_{split}_{sel}.csv"
                shutil.copy2(src, dst)
                copied[dst.name] = dst
    return copied


def _write_manifest(root: Path, integ: Path, feat: Path, bases: Path, stage2_ok: bool) -> None:
    lines = []
    lines.append("# MANIFEST — saída do pipeline\n")
    lines.append("Gerado por `run_all.py`. Lista os artefatos, dimensões e papel de cada um.\n")
    lines.append("> **Para treinar os modelos, use `bases_modelo/`.** A versão completa tem todas")
    lines.append("> as 2540 features (orientação Probe_ID × amostras — transponha para treinar e")
    lines.append("> junte os rótulos de `merged_sample_annotation.csv` por `sample_id`). A versão")
    lines.append("> selecionada (`*_boruta`/`*_lasso`) tem só os miRNAs escolhidos, já na orientação")
    lines.append("> amostras × features com colunas `target` e `split`. A escolha entre completa e")
    lines.append("> selecionada — e entre Boruta e LASSO — é decisão da equipe (ver discussão no TCC).\n")

    def section(title, base_dir, entries):
        lines.append(f"\n## {title}\n")
        lines.append("| Arquivo | Dimensões | Papel |")
        lines.append("|---|---|---|")
        for rel, role in entries:
            p = base_dir / rel
            if p.exists():
                dims = _dims(p) if p.suffix == ".csv" else "—"
                lines.append(f"| `{rel}` | {dims} | {role} |")

    section("1_integracao/ — Estágio 1", integ, [
        ("base_treino.csv", "treino, TODAS as features (Probe_ID × amostras)"),
        ("base_teste.csv", "teste, TODAS as features (Probe_ID × amostras)"),
        ("merged_sample_annotation.csv", "anotação das amostras (class_label, batch, platform, split)"),
        ("merged_expression_raw.csv", "matriz integrada log2, sem correção"),
        ("merged_expression_combat.csv", "matriz corrigida (treino+teste) — só para PCA"),
        ("combat_estimates.pkl", "estimativas do ComBat (treino) — joblib"),
        ("zscore_params.csv", "μ/σ por probe (treino)"),
        ("purity_metrics.csv", "métricas Purity/Silhouette antes/depois do ComBat"),
        ("split_composition.csv", "composição treino/teste por plataforma × classe"),
    ])

    if stage2_ok:
        section("2_selecao_features/ — Estágio 2", feat, [
            ("boruta/selected_miRNAs_step_b.csv", "miRNAs selecionados pelo Boruta"),
            ("lasso/selected_miRNAs_step_b.csv", "miRNAs selecionados pelo LASSO"),
            ("comparison/selected_features_overlap.csv", "shared / só-Boruta / só-LASSO"),
            ("comparison/comparison_report.txt", "relatório comparativo (texto)"),
        ])

    section("bases_modelo/ — BASES PRONTAS PARA O TREINO", bases, [
        ("base_treino.csv", "treino completo (todas as features)"),
        ("base_teste.csv", "teste completo (todas as features)"),
        ("merged_sample_annotation.csv", "rótulos + split (para as bases completas)"),
        ("base_treino_boruta.csv", "treino, só features do Boruta (amostras × features + target/split)"),
        ("base_teste_boruta.csv", "teste, só features do Boruta"),
        ("base_treino_lasso.csv", "treino, só features do LASSO"),
        ("base_teste_lasso.csv", "teste, só features do LASSO"),
    ])

    lines.append("\n---\n")
    lines.append("Observação: `1_integracao/out_<GSE>/` contém artefatos intermediários por dataset.\n")
    (root / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMANIFEST gerado: {root / 'MANIFEST.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Roda Estágios 1+2 num único output-root e gera MANIFEST.")
    ap.add_argument("--output-root", required=True, help="Diretório-raiz único de saída")
    ap.add_argument("--files", nargs="*", default=_DEFAULT_FILES, help="Series Matrix (default: os 2 do projeto)")
    ap.add_argument("--mode", default="both", choices=["boruta", "lasso", "both"], help="Estágio 2")
    ap.add_argument("--no-plots", action="store_true", help="Pular gráficos PCA do Estágio 1")
    ap.add_argument("--no-stage2", action="store_true", help="Rodar só o Estágio 1")
    args = ap.parse_args()

    root = Path(args.output_root)
    integ = root / "1_integracao"
    feat = root / "2_selecao_features"
    bases = root / "bases_modelo"
    root.mkdir(parents=True, exist_ok=True)

    _stage1(integ, args.files, _DEFAULT_CONDITION_FILTER, _DEFAULT_CLASS_MAP, args.no_plots)

    stage2_ok = False
    if not args.no_stage2:
        stage2_ok = _stage2(integ, feat, args.mode)

    _assemble_bases(integ, feat, bases)
    _write_manifest(root, integ, feat, bases, stage2_ok)

    print(f"\n✅ Concluído. Tudo sob: {root.resolve()}")
    print(f"   → bases prontas para o treino em: {bases.resolve()}")


if __name__ == "__main__":
    main()
