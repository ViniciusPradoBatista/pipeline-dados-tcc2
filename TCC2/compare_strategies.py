#!/usr/bin/env python3
"""Compara as estratégias de normalização do pipeline.

Roda após o pipeline ter sido executado 2+ vezes com `--output-root` distintos
e consolida `purity_metrics.csv` de cada uma. Gera:
  - `strategy_comparison.png` — PurityB e PurityD por estágio/estratégia
  - Resumo em stdout com os valores finais (after_combat).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_DIRS: Dict[str, str] = {
    "zscore_first": "./out_zscore_first",
    "combat_first": "./out_combat_first",
    "combat_first+zscore": "./out_combat_zscore",
    "combat_first+quantile": "./out_combat_quantile",
}


def load_purity(out_dir: Path) -> pd.DataFrame | None:
    csv = out_dir / "purity_metrics.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def plot_comparison(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for strategy, df in results.items():
        axes[0].plot(df["stage"], df["PurityB"], marker="o", label=strategy)
        axes[1].plot(df["stage"], df["PurityD"], marker="o", label=strategy)

    axes[0].set_title("PurityB (batch separation)\nmenor é melhor")
    axes[0].set_ylabel("Purity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("PurityD (disease separation)\nmaior é melhor")
    axes[1].set_ylabel("Purity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[info] salvo {out_path}")


def print_summary(results: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print("  Comparação de estratégias (after_combat / final)")
    print("=" * 60)
    header = f"{'estratégia':<26} {'PurityB':>9} {'PurityD':>9}"
    print(header)
    print("-" * len(header))
    for strategy, df in results.items():
        final_row = df.iloc[-1]
        print(f"{strategy:<26} {final_row['PurityB']:>9.3f} {final_row['PurityD']:>9.3f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", nargs="*", default=None,
        help="Pares label=path (ex: zscore_first=./out_zscore_first)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("strategy_comparison.png"),
        help="Caminho de saída da figura comparativa",
    )
    args = parser.parse_args()

    if args.dir:
        dirs: Dict[str, str] = {}
        for spec in args.dir:
            if "=" not in spec:
                print(f"[warn] ignorando '{spec}' (formato esperado: label=path)")
                continue
            label, path = spec.split("=", 1)
            dirs[label.strip()] = path.strip()
    else:
        dirs = DEFAULT_DIRS

    results: Dict[str, pd.DataFrame] = {}
    for label, path in dirs.items():
        df = load_purity(Path(path))
        if df is None:
            print(f"[skip] {label}: {path}/purity_metrics.csv não encontrado")
            continue
        results[label] = df

    if not results:
        print("[erro] nenhum purity_metrics.csv carregado — nada a comparar.")
        return 1

    plot_comparison(results, args.out)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
