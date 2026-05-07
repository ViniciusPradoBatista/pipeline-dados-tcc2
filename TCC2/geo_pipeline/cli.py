"""CLI argparse e seletores interativos (tkinter) para o pipeline."""

import argparse
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import List

log = logging.getLogger("geo_pipeline")


def build_cli() -> argparse.Namespace:
    """Constrói e parseia argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "GEO miRNA Cross-Platform Integration Pipeline "
            "for PDAC (Pancreatic Cancer) Research"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (opens file picker):
  python geo_mirna_pipeline.py

  # Process two datasets, merge, apply ComBat
  python geo_mirna_pipeline.py GSE85589_series_matrix.txt \\
      GSE59856_series_matrix.txt --output-root ./out

  # Non-interactive mode with condition filter
  python geo_mirna_pipeline.py GSE85589.txt GSE59856.txt \\
      --no-interactive \\
      --condition-filter "pancreatic cancer" "healthy control" \\
      --class-map "pancreatic cancer=PDAC" "healthy control=Control"

  # Single dataset, no merge
  python geo_mirna_pipeline.py GSE85589_series_matrix.txt \\
      --output-root ./results
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="GEO Series Matrix files (.txt or .xlsx). If omitted, a file picker dialog opens.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root output directory (default: current directory)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive prompts; use all samples if no filter given",
    )
    parser.add_argument(
        "--condition-filter",
        nargs="*",
        default=None,
        help=(
            "Conditions to keep (substring match, applied to all datasets). "
            'E.g.: --condition-filter "pancreatic cancer" "healthy control"'
        ),
    )
    parser.add_argument(
        "--class-map",
        nargs="*",
        default=None,
        help=(
            "Map condition names to class labels. "
            'E.g.: --class-map "pancreatic cancer=PDAC" "healthy control=Control"'
        ),
    )
    parser.add_argument(
        "--no-combat",
        action="store_true",
        help="Skip ComBat batch correction",
    )
    parser.add_argument(
        "--zscore-only",
        action="store_true",
        help="Only do z-score merge, skip ComBat",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--auto-add-healthy-control",
        action="store_true",
        default=True,
        help=(
            "Automatically include healthy controls if a pathological condition "
            "is selected (default: True)"
        ),
    )
    parser.add_argument(
        "--no-auto-add-healthy-control",
        action="store_false",
        dest="auto_add_healthy_control",
        help="Disable automatic inclusion of healthy controls",
    )
    parser.add_argument(
        "--strict-control-only",
        action="store_true",
        default=True,
        help="Only accept explicitly healthy/normal labels as controls (default: True)",
    )
    parser.add_argument(
        "--no-strict-control-only",
        action="store_false",
        dest="strict_control_only",
        help="Accept broader synonyms like 'control' for automatic inclusion",
    )

    return parser.parse_args()


def interactive_file_picker() -> List[str]:
    """
    Abre um diálogo nativo (tkinter) para seleção de arquivos GEO Series Matrix.

    Suporta múltiplas rodadas de seleção para a usuária pegar arquivos de
    pastas diferentes (ex: GSE85589 de uma pasta e GSE59856 de outra).

    Returns:
        Lista de paths absolutos dos arquivos selecionados.
    """
    all_files: List[str] = []

    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if not all_files:
            print("\n  Abrindo seletor de arquivos...")
            print("  (Selecione um ou mais arquivos GEO Series Matrix)")
            print("  Dica: Segure Ctrl para selecionar varios arquivos!\n")
        else:
            print(f"\n  Abrindo seletor para adicionar mais arquivos...")
            print(f"  ({len(all_files)} arquivo(s) ja selecionado(s))\n")

        file_paths = filedialog.askopenfilenames(
            title=(
                f"Selecione arquivos GEO Series Matrix "
                f"({len(all_files)} ja selecionados)"
            ),
            filetypes=[
                ("GEO Series Matrix", "*.txt *.xlsx *.xls"),
                ("Arquivos de texto", "*.txt"),
                ("Planilhas Excel", "*.xlsx *.xls"),
                ("Todos os arquivos", "*.*"),
            ],
        )

        root.destroy()

        if file_paths:
            for f in file_paths:
                if f not in all_files:
                    all_files.append(f)
                    print(f"  + {Path(f).name}")

        if not all_files:
            return []

        print(f"\n  Total: {len(all_files)} arquivo(s) selecionado(s)")

        try:
            more = (
                input("  Deseja adicionar mais arquivos de outra pasta? (s/N): ")
                .strip()
                .lower()
            )
            if more not in ("s", "sim", "y", "yes"):
                break
        except (EOFError, KeyboardInterrupt):
            break

    return all_files


def interactive_output_picker() -> str:
    """Pergunta no console onde salvar a saída."""
    print("\n" + "-" * 50)
    print("  Onde salvar os resultados?")
    print("-" * 50)
    print("  [1] Pasta atual (.)")
    print("  [2] Escolher pasta via seletor")
    print("  [3] Digitar caminho manualmente")
    print("-" * 50)

    while True:
        try:
            choice = input("\n  Escolha (1/2/3) [padrao: 1]: ").strip()
            if choice in ("", "1"):
                return "."
            elif choice == "2":
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)

                folder = filedialog.askdirectory(
                    title="Selecione a pasta para salvar os resultados",
                )
                root.destroy()

                if folder:
                    return folder
                print("  Nenhuma pasta selecionada. Usando pasta atual.")
                return "."
            elif choice == "3":
                path = input("  Caminho da pasta: ").strip()
                return path if path else "."
            else:
                print("  Opcao invalida. Digite 1, 2 ou 3.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Usando pasta atual.")
            return "."
