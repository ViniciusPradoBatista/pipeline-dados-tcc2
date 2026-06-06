"""Confound de COLETA/scan-batch — checagem direta nos metadados do GEO.

Pergunta: dentro de um mesmo dataset/plataforma, casos (PDAC) e controles foram
processados nos MESMOS lotes de array/scan, ou em lotes separados? Se separados, o
scan-batch fica confundido com a classe e a discriminação pode ser parcialmente
artefato de lote — mesmo numa única plataforma (limitação do desenho do estudo).

Lê os arquivos GEO Series Matrix originais (NÃO os artefatos do pipeline), extrai:
  - data de submissão/atualização por amostra (confound temporal);
  - prefixo do ID da amostra (ex.: P/E/S — coortes) a partir do título;
  - código de array/scan embutido no supplementary_file (ex.: 3D-Gene 'SH3X..');
e cruza com a classe (PDAC vs. saudável).

Uso:
    python validation/collection_batch_check.py \
        --series-matrix "TCC2/data/GSE59856_series_matrix (1).txt" \
                        "TCC2/data/GSE85589_series_matrix.txt" \
        --output-dir ./out
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "TCC2"))
from geo_pipeline.dataset import extract_dataset_id  # noqa: E402
from geo_pipeline.io_geo import parse_series_metadata_tabular  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("collection_batch")

# Padrões de código de array/scan-batch conhecidos (extensível).
_ARRAY_PATTERNS = [
    r"(SH3X[A-Z]\d+)",          # 3D-Gene / Toray
    r"_([A-Z]{2,}\d{2,})\.",    # genérico: token tipo lote antes da extensão
]


def _classify(row, cols) -> str:
    blob = " ".join(str(row.get(c, "")) for c in cols
                    if any(k in c.lower() for k in ("characteristics", "source", "title", "description"))).lower()
    if "pancreatic cancer" in blob or "pancreatic ductal" in blob or "pdac" in blob:
        return "PDAC"
    if "healthy" in blob or "normal" in blob:
        return "Control"
    return "other"


def _extract_prefix(text: str) -> str:
    m = re.search(r"\b([A-Za-z]{1,3})\d{1,4}\b", str(text))
    return m.group(1).upper() if m else "?"


def _extract_array(text: str) -> str:
    for pat in _ARRAY_PATTERNS:
        m = re.search(pat, str(text))
        if m:
            return m.group(1)
    return "?"


def check_dataset(path: str, output_dir: Path) -> dict:
    gse = extract_dataset_id(path)
    m = parse_series_metadata_tabular(path)
    cols = list(m.columns)
    m = m.copy()
    m["cls"] = m.apply(lambda r: _classify(r, cols), axis=1)
    sub = m[m["cls"].isin(["PDAC", "Control"])].copy()

    print(f"\n{'='*72}\n  {gse}  —  {len(sub)} amostras PDAC/Control "
          f"({sub['cls'].value_counts().to_dict()})\n{'='*72}")

    # ── 1. Confound temporal (datas por classe) ──
    for date_col in ("Sample_submission_date", "Sample_last_update_date"):
        if date_col in sub.columns:
            ct = pd.crosstab(sub[date_col], sub["cls"])
            n_dates = ct.shape[0]
            pure = ((ct.get("PDAC", 0) == 0) | (ct.get("Control", 0) == 0)).sum()
            status = "OK (mesma data)" if n_dates == 1 else (
                f"ATENCAO: {pure}/{n_dates} datas sao de uma classe só")
            print(f"  {date_col}: {n_dates} data(s) distinta(s) -> {status}")

    # ── 2. Prefixo do ID (coorte) por classe ──
    title_src = sub.get("Sample_title", pd.Series([""] * len(sub)))
    supp_src = sub.get("Sample_supplementary_file", pd.Series([""] * len(sub)))
    sub["prefix"] = [_extract_prefix(t) for t in title_src]
    pref_ct = pd.crosstab(sub["prefix"], sub["cls"])
    pref_pure = ((pref_ct.get("PDAC", 0) == 0) | (pref_ct.get("Control", 0) == 0)).all()
    print(f"  Prefixo do ID por classe (puro por classe? {pref_pure}):")
    print("    " + pref_ct.to_string().replace("\n", "\n    "))

    # ── 3. Código de array/scan-batch por classe (o tell principal) ──
    sub["array"] = [_extract_array(s) for s in supp_src]
    if (sub["array"] == "?").all():
        # tenta no título como fallback
        sub["array"] = [_extract_array(t) for t in title_src]

    result = {"dataset": gse, "n": len(sub)}
    if (sub["array"] == "?").all():
        print("  Array/scan-batch: GEO NAO expoe codigo de array p/ este dataset "
              "-> confound de scan-batch NAO avaliavel via metadados.")
        result.update({"array_available": False})
    else:
        ct = pd.crosstab(sub["array"], sub["cls"])
        n_arrays = ct.shape[0]
        avg_per_array = len(sub) / n_arrays
        # Se ~1 amostra por "lote", o token é um barcode POR AMOSTRA (ex.: .CEL Affymetrix),
        # não um lote multi-amostra — lotes singleton são puros por construção (não informativo).
        if avg_per_array < 1.5:
            print(f"  Array/scan-batch: {n_arrays} tokens p/ {len(sub)} amostras "
                  f"(~{avg_per_array:.1f}/token) -> parecem barcodes POR AMOSTRA, nao lotes. "
                  f"Confound de scan-batch NAO avaliavel via metadados.")
            result.update({"array_available": False, "reason": "per-sample tokens"})
            return result
        pdac = ct.get("PDAC", pd.Series(0, index=ct.index))
        ctrl = ct.get("Control", pd.Series(0, index=ct.index))
        only_pdac = int(((pdac > 0) & (ctrl == 0)).sum())
        only_ctrl = int(((ctrl > 0) & (pdac == 0)).sum())
        shared = int(((pdac > 0) & (ctrl > 0)).sum())
        n_pure_samples = int(pdac[ctrl == 0].sum() + ctrl[pdac == 0].sum())
        frac_pure = n_pure_samples / len(sub)
        ct.to_csv(output_dir / f"collection_batch_{gse}.csv")
        print(f"  Array/scan-batch: {n_arrays} lotes (~{avg_per_array:.1f} amostras/lote) -> "
              f"{only_pdac} so-PDAC, {only_ctrl} so-Control, {shared} MISTOS.")
        if shared == 0:
            print(f"  [ALERTA] scan-batch PERFEITAMENTE confundido com a classe "
                  f"({frac_pure*100:.0f}% das amostras em lotes puros). INSEPARAVEL.")
            print( "           A discriminacao NAO pode ser atribuida a biologia pura "
                   "neste dataset.")
        elif frac_pure > 0.8:
            print(f"  [ATENCAO] scan-batch fortemente confundido ({frac_pure*100:.0f}% "
                  f"em lotes puros). Interpretar metricas com cautela.")
        else:
            print(f"  [OK] lotes razoavelmente mistos ({frac_pure*100:.0f}% em lotes puros).")
        result.update({"array_available": True, "n_arrays": n_arrays,
                       "arrays_only_pdac": only_pdac, "arrays_only_control": only_ctrl,
                       "arrays_shared": shared, "frac_samples_in_pure_batches": round(frac_pure, 3)})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Confound de coleta/scan-batch nos metadados GEO")
    ap.add_argument("--series-matrix", nargs="+", required=True,
                    help="Um ou mais arquivos GEO Series Matrix (.txt) originais")
    ap.add_argument("--output-dir", default=".", help="Onde salvar os CSVs de confound")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.series_matrix:
        if not Path(p).exists():
            log.error(f"Arquivo nao encontrado: {p}")
            continue
        rows.append(check_dataset(p, out_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "collection_batch_summary.csv", index=False)
    print(f"\nResumo salvo em: {out_dir/'collection_batch_summary.csv'}")


if __name__ == "__main__":
    main()
