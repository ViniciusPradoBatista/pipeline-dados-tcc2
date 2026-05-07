"""
===================================================================
  GEO miRNA Cross-Platform Integration Pipeline
  For PDAC (Pancreatic Ductal Adenocarcinoma) Research

  Features:
    - Universal GEO Series Matrix reader (.txt and .xlsx)
    - Automatic platform detection and scale inference
    - Probe ID harmonization across platforms
    - Z-score normalization per dataset
    - Cross-dataset merge by common miRNAs
    - ComBat batch correction (preserving biological signal)
    - PurityB / PurityD validation
    - PCA visualization
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

import matplotlib

matplotlib.use("Agg")

from geo_pipeline.cli import (
    build_cli,
    interactive_file_picker,
    interactive_output_picker,
)
from geo_pipeline.dataset import merge_datasets, process_single_dataset
from geo_pipeline.metrics import compute_purity_metrics
from geo_pipeline.normalize import apply_combat, zscore_by_probe
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

    dataset_dirs = []
    for filepath in args.files:
        if not Path(filepath).exists():
            continue
        res = process_single_dataset(
            path=filepath,
            output_root=output_root,
            no_interactive=args.no_interactive,
            condition_filter=args.condition_filter,
            class_map=class_map,
            auto_add_healthy_control=args.auto_add_healthy_control,
            strict_control_only=args.strict_control_only,
        )
        if res:
            dataset_dirs.append(res)

    if len(dataset_dirs) >= 2:
        merged_raw, merged_zscore, merged_annot = merge_datasets(
            dataset_dirs, output_root
        )
        expr_combat = (
            apply_combat(merged_raw, merged_annot) if not args.no_combat else None
        )
        if expr_combat is not None:
            expr_combat.to_csv(
                output_root / "merged_expression_combat.csv", index=False
            )
            zscore_by_probe(expr_combat).to_csv(
                output_root / "merged_expression_combat_zscore.csv", index=False
            )
        compute_purity_metrics(merged_raw, expr_combat, merged_annot).to_csv(
            output_root / "purity_metrics.csv", index=False
        )
        if not args.no_plots:
            generate_all_plots(merged_raw, expr_combat, merged_annot, output_root)

    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()
