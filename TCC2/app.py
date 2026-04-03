"""
===================================================================
  Leitor Universal de Series Matrix (GEO)
  Suporta múltiplas plataformas: Affymetrix, 3D-Gene, Agilent,
  Illumina e qualquer outra plataforma GEO.
===================================================================
"""

import re
import os
import chardet
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys


# =====================================================================
# 0️⃣ Setup inicial
# =====================================================================
print("=" * 60)
print("  Leitor Universal de Series Matrix (GEO)")
print("  Suporta: Affymetrix, 3D-Gene, Agilent, Illumina, etc.")
print("=" * 60)


# =====================================================================
# 1️⃣ Seleção do arquivo Series Matrix
# =====================================================================
def get_file_path():
    """Abre janela para selecionar o arquivo ou aceita via linha de comando."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    print("\nPor favor, selecione o arquivo Series Matrix na janela que foi aberta...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Selecione o arquivo Series Matrix (.txt)",
        filetypes=[("Text/CSV files", "*.txt *.csv *.tsv"), ("All files", "*.*")]
    )
    root.destroy()
    return path


PATH_TXT = get_file_path()

if not PATH_TXT:
    print("❌ Nenhum arquivo selecionado. Saindo...")
    sys.exit(0)

print(f"📄 Arquivo carregado: {PATH_TXT}")

# Nome dinâmico da pasta de saída baseado no nome do arquivo
file_stem = Path(PATH_TXT).stem  # ex: "GSE85589_series_matrix" -> "GSE85589_series_matrix"
# Tenta pegar só o GSE ID
gse_match = re.search(r"(GSE\d+)", file_stem, re.IGNORECASE)
out_name = f"out_{gse_match.group(1)}" if gse_match else f"out_{file_stem}"
OUT_DIR = Path(out_name)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Pasta de saída: {OUT_DIR.absolute()}")


# =====================================================================
# 2️⃣ Extração de metadados e detecção automática de plataforma
# =====================================================================
def detect_encoding(path_txt):
    """Detecta o encoding do arquivo."""
    with open(path_txt, "rb") as f:
        enc = chardet.detect(f.read(2000000))['encoding'] or 'latin-1'
    return enc


def parse_series_metadata_tabular(path_txt):
    """Lê a parte inicial do TXT e extrai metadados estruturados."""
    enc = detect_encoding(path_txt)

    meta_lines = []
    with open(path_txt, "r", encoding=enc, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("!series_matrix_table_begin"):
                break
            if line.strip().startswith("!"):
                meta_lines.append(line.strip())

    data = []
    for l in meta_lines:
        parts = re.split(r"\t+|\s{2,}", l)
        if len(parts) > 1:
            key = parts[0].lstrip("!").strip()
            vals = [v.strip().strip('"') for v in parts[1:] if v.strip()]
            data.append((key, vals))

    if not data:
        print("⚠️ Nenhum metadado encontrado.")
        return pd.DataFrame()

    max_cols = max(len(v) for _, v in data)
    df = pd.DataFrame({k: v + [""] * (max_cols - len(v)) for k, v in data}).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Field"}, inplace=True)
    df = df.set_index("Field").T.reset_index(drop=True)

    print(f"✅ Metadados lidos: {df.shape[0]} amostras × {df.shape[1]} campos")
    return df


def detect_platform(meta_df):
    """Detecta automaticamente a plataforma a partir dos metadados."""
    platform_cols = [c for c in meta_df.columns
                     if any(k in c.lower() for k in ["platform_id", "platform"])]

    platform_info = "Desconhecida"
    for col in platform_cols:
        vals = meta_df[col].astype(str).unique()
        vals = [v for v in vals if v and v not in ["", "nan"]]
        if vals:
            platform_info = vals[0]
            break

    # Mapeamento de plataformas conhecidas
    known_platforms = {
        "GPL21263": "3D-Gene Human miRNA (Toray)",
        "GPL18402": "Affymetrix miRNA 4.0",
        "GPL16384": "Affymetrix miRNA 3.0",
        "GPL8786":  "Affymetrix miRNA 1.0",
        "GPL10850": "Agilent Human miRNA V2",
        "GPL18058": "Agilent Human miRNA V3",
        "GPL11487": "Agilent Human miRNA V4",
        "GPL7731":  "Agilent Human miRNA V1",
        "GPL8179":  "Illumina Human v2",
        "GPL6480":  "Agilent Whole Human Genome",
        "GPL570":   "Affymetrix HG-U133 Plus 2.0",
        "GPL96":    "Affymetrix HG-U133A",
    }

    platform_name = known_platforms.get(platform_info, platform_info)
    print(f"🔬 Plataforma detectada: {platform_info} ({platform_name})")
    return platform_info, platform_name


def normalize_condition(value):
    """Remove IDs individuais de amostra do final do valor.
    Ex: 'pancreatic cancer P75' -> 'pancreatic cancer'
        'healthy control E001' -> 'healthy control'
        'biliary tract cancer B101' -> 'biliary tract cancer'
        'colon cancer CC50' -> 'colon cancer'
    """
    # Remove sufixos como P75, B101, CC50, E001, I03, HC01, SC44, S092, etc.
    normalized = re.sub(r'\s+[A-Z]{0,2}\d{1,4}\s*$', '', value.strip())
    return normalized.strip()


def extract_conditions(meta_df):
    """Extrai condições/doenças únicas, agrupando por categoria base."""
    condition_keywords = [
        "source_name", "characteristics", "title", "description",
        "disease", "tissue", "cell_type", "treatment"
    ]

    raw_values = []
    condition_cols = []

    for col in meta_df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in condition_keywords):
            condition_cols.append(col)
            vals = meta_df[col].astype(str).unique()
            for v in vals:
                v = v.strip().strip('"')
                if v and v.lower() not in ["", "nan", "none"]:
                    raw_values.append(v)

    # Agrupar por nome normalizado e contar amostras
    from collections import Counter
    normalized_counts = Counter()
    for v in raw_values:
        norm = normalize_condition(v)
        if norm:
            normalized_counts[norm] += 1

    # Retornar lista ordenada de (nome_normalizado, contagem)
    grouped = sorted(normalized_counts.items(), key=lambda x: x[0].lower())
    return grouped, condition_cols


def select_condition_interactive(grouped_conditions):
    """Permite ao usuário selecionar a condição de interesse no console."""
    if not grouped_conditions:
        print("⚠️ Nenhuma condição/doença encontrada nos metadados.")
        return None

    print("\n" + "=" * 60)
    print("  Condições/categorias encontradas no dataset:")
    print("=" * 60)
    for i, (cond, count) in enumerate(grouped_conditions, 1):
        print(f"  [{i}] {cond}  ({count} amostras)")
    print(f"  [0] Usar TODAS as amostras (sem filtro)")
    print("=" * 60)

    while True:
        try:
            choice = input("\n🎯 Digite o número da condição que deseja filtrar: ").strip()
            choice = int(choice)
            if choice == 0:
                return None  # sem filtro
            if 1 <= choice <= len(grouped_conditions):
                selected = grouped_conditions[choice - 1][0]
                count = grouped_conditions[choice - 1][1]
                print(f"✅ Condição selecionada: '{selected}' ({count} amostras)")
                return selected
            print("❌ Número inválido. Tente novamente.")
        except ValueError:
            print("❌ Digite apenas o número. Tente novamente.")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ Usando todas as amostras (sem filtro).")
            return None


def filter_samples_by_condition(meta_df, condition, condition_cols):
    """Filtra amostras que correspondem à condição selecionada."""
    if condition is None:
        print(f"📋 Usando TODAS as {meta_df.shape[0]} amostras (sem filtro).")
        return meta_df

    # Busca case-insensitive da condição nos metadados
    cond_lower = condition.lower()
    mask = pd.Series([False] * len(meta_df), index=meta_df.index)

    for col in condition_cols:
        col_mask = meta_df[col].astype(str).str.lower().str.contains(
            re.escape(cond_lower), na=False
        )
        mask = mask | col_mask

    # Se não encontrou nada nas colunas de condição, busca em todas as colunas
    if mask.sum() == 0:
        df_str = meta_df.astype(str).apply(lambda x: x.str.lower())
        mask = df_str.apply(
            lambda row: cond_lower in " ".join(row.values), axis=1
        )

    filtered = meta_df.loc[mask].copy()
    print(f"🎯 Amostras encontradas para '{condition}': {filtered.shape[0]}")
    return filtered


# --- Executar Passo 2 ---
print("\n" + "-" * 60)
print("  PASSO 2: Extração de metadados")
print("-" * 60)

meta = parse_series_metadata_tabular(PATH_TXT)

if meta.empty:
    print("❌ Não foi possível extrair metadados. Verifique o arquivo.")
    sys.exit(1)

# Detectar plataforma
platform_id, platform_name = detect_platform(meta)

# Extrair condições e permitir seleção
conditions, condition_cols = extract_conditions(meta)
selected_condition = select_condition_interactive(conditions)
meta_filtered = filter_samples_by_condition(meta, selected_condition, condition_cols)

# Busca colunas que contenham GSMs
gsm_candidates = []
for c in meta.columns:
    if any(k in c.lower() for k in ["geo_accession", "sample_geo", "gsm"]):
        gsm_candidates.append(c)

if len(gsm_candidates) == 0 and not meta.empty:
    gsm_candidates = [c for c in meta.columns
                      if meta[c].astype(str).str.contains("GSM").any()]

print(f"🔎 Colunas potenciais de GSM: {gsm_candidates[:5]}")

# Salvar metadados
filter_label = re.sub(r'[^\w\s-]', '', selected_condition or "all").strip().replace(" ", "_")
meta_filtered.to_csv(OUT_DIR / f"metadata_{filter_label}.csv", index=False)
meta.to_csv(OUT_DIR / "metadata_full.csv", index=False)


# =====================================================================
# 3️⃣ Ler expressão (com preservação, numérico e log₂)
# =====================================================================
def read_expression_from_txt(path_txt):
    """Lê os dados de expressão do arquivo Series Matrix (universal)."""
    enc = detect_encoding(path_txt)

    header_idx = None
    with open(path_txt, "r", encoding=enc, errors="ignore") as f:
        for i, line in enumerate(f):
            if "ID_REF" in line:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("❌ Cabeçalho ID_REF não encontrado no arquivo.")

    df = pd.read_csv(path_txt, sep="\t", skiprows=header_idx, header=0,
                     dtype=str, encoding=enc, engine="python")

    df = df[df["ID_REF"].notna()]
    df = df[~df["ID_REF"].astype(str).str.startswith("!")]
    df.rename(columns={"ID_REF": "Probe_ID"}, inplace=True)
    df["Probe_ID"] = df["Probe_ID"].astype(str).str.strip().str.strip('"')

    gsm_cols = [c for c in df.columns if re.match(r"^GSM\d+", str(c))]
    expr_text = df[["Probe_ID"] + gsm_cols].copy()
    expr_num = expr_text.copy()

    def smart_float(x):
        """Converte valores de expressão em float.
        Em dados GEO Series Matrix, o ponto SEMPRE é decimal (ex: 3.431 = três ponto quatro três um).
        Nunca existe formatação de milhar nesse tipo de arquivo.
        """
        s = str(x).replace('"', '').strip()
        if s == "" or s.lower() in ["na", "nan", "null", "none", "--", "n/a"]:
            return np.nan
        # Remove vírgulas caso existam (nunca são decimais em GEO)
        s = s.replace(',', '')
        try:
            return float(s)
        except ValueError:
            return np.nan

    print("🔢 Formato numérico: decimal direto (ponto = decimal, sem milhar)")

    for c in gsm_cols:
        expr_num[c] = expr_num[c].apply(smart_float)

    expr_log2 = expr_num.copy()
    for c in gsm_cols:
        expr_log2[c] = np.log2(expr_log2[c].clip(lower=0) + 1)

    expr_text.to_csv(OUT_DIR / "expression_text_preservado.csv", index=False)
    expr_num.to_csv(OUT_DIR / "expression_numerico.csv", index=False)
    expr_log2.to_csv(OUT_DIR / "expression_log2.csv", index=False)

    print(f"✅ Expressão carregada: {expr_num.shape[0]} probes × {len(gsm_cols)} amostras")
    return expr_text, expr_num, expr_log2, gsm_cols


print("\n" + "-" * 60)
print("  PASSO 3: Leitura de dados de expressão")
print("-" * 60)
expr_text, expr_num, expr_log2, gsm_cols = read_expression_from_txt(PATH_TXT)


# =====================================================================
# 4️⃣ Cruzar expressão com amostras filtradas
# =====================================================================
print("\n" + "-" * 60)
print("  PASSO 4: Cruzamento expressão × amostras filtradas")
print("-" * 60)

filtered_ids = []

for col in meta_filtered.columns:
    vals = meta_filtered[col].astype(str)
    filtered_ids += [v for v in vals if v.startswith("GSM")]

# Fallback: se não achou GSMs na tabela filtrada, tenta busca genérica
if len(filtered_ids) == 0 and selected_condition and not meta.empty:
    print("⚠️ Nenhuma amostra com GSM encontrada diretamente — tentando forçar busca...")
    cond_lower = selected_condition.lower()
    for col in meta.columns:
        vals = meta[col].astype(str)
        if vals.str.lower().str.contains(re.escape(cond_lower), na=False).any():
            filtered_ids += [v for v in vals if v.startswith("GSM")]

filtered_ids = list(set(filtered_ids))
print(f"🎯 Total de GSMs detectados para a condição: {len(filtered_ids)}")

if selected_condition is None:
    # Sem filtro: exportar tudo
    expr_text.to_csv(OUT_DIR / "expression_all_text.csv", index=False)
    expr_num.to_csv(OUT_DIR / "expression_all_num.csv", index=False)
    expr_log2.to_csv(OUT_DIR / "expression_all_log2.csv", index=False)
    print(f"✅ Todas as {len(gsm_cols)} amostras exportadas (sem filtro).")
else:
    cols = ["Probe_ID"] + [c for c in expr_text.columns if c in filtered_ids]
    if len(cols) == 1:
        print("❌ Nenhuma coluna correspondente aos GSMs filtrados encontrada.")
    else:
        expr_filt_text = expr_text[cols].copy()
        expr_filt_num = expr_num[cols].copy()
        expr_filt_log = expr_log2[cols].copy()

        expr_filt_text.to_csv(OUT_DIR / f"expression_{filter_label}_text.csv", index=False)
        expr_filt_num.to_csv(OUT_DIR / f"expression_{filter_label}_num.csv", index=False)
        expr_filt_log.to_csv(OUT_DIR / f"expression_{filter_label}_log2.csv", index=False)

        print(f"✅ Amostras exportadas para '{selected_condition}': {len(cols) - 1}")


# =====================================================================
# 5️⃣ Resumo final e abertura da pasta
# =====================================================================
print("\n" + "=" * 60)
print("  🎉 Processamento concluído com sucesso!")
print("=" * 60)
print(f"  📄 Arquivo de entrada:  {Path(PATH_TXT).name}")
print(f"  🔬 Plataforma:         {platform_id} ({platform_name})")
print(f"  🎯 Condição filtrada:  {selected_condition or 'Todas (sem filtro)'}")
print(f"  📁 Pasta de saída:     {OUT_DIR.absolute()}")
print(f"  📊 Arquivos gerados:")
for f in sorted(OUT_DIR.glob("*.csv")):
    size_kb = f.stat().st_size / 1024
    print(f"       • {f.name} ({size_kb:.1f} KB)")
print("=" * 60)

try:
    if os.name == 'nt':
        os.startfile(OUT_DIR.absolute())
        print("📂 Pasta aberta no Windows Explorer!")
except Exception:
    pass
