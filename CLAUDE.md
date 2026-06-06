# CLAUDE.md

Referência técnica do projeto para o Claude Code. Descreve arquitetura, fluxo de dados e convenções deste pipeline de TCC.

## O que é

Pipeline de bioinformática em Python que **integra dados de expressão de miRNA de múltiplos datasets do GEO** (Gene Expression Omnibus), de **plataformas diferentes**, e **seleciona um conjunto enxuto de miRNAs** (biomarcadores candidatos) capaz de distinguir **PDAC** (adenocarcinoma ductal pancreático / câncer de pâncreas) de controles saudáveis.

Não há sistema de build. São scripts Python executados diretamente. Dependências em `TCC2/requirements.txt`.

```
pip install -r TCC2/requirements.txt
```

Principais libs: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `boruta`, `neuroCombat`, `chardet`, `openpyxl`, `matplotlib`, `seaborn`.

## Arquitetura — dois estágios

### Estágio 1 — Integração cross-platform

Entrada: arquivos GEO **Series Matrix** (`.txt` ou `.xlsx`). Orquestrador: `TCC2/geo_mirna_pipeline.py`. Lógica no pacote `TCC2/geo_pipeline/`.

```
geo_mirna_pipeline.py            # main(): processa cada dataset, faz merge, ComBat, z-score, métricas, plots
geo_pipeline/
  cli.py          # argparse + seletores interativos (tkinter)
  constants.py    # KNOWN_PLATFORMS, sinônimos de condição (healthy/pathological)
  io_geo.py       # leitura de baixo nível: encoding (chardet), parse de metadados .txt/.xlsx
  parsing.py      # smart_float — parsing numérico robusto (incl. locale quebrado)
  expression.py   # leitura da tabela de expressão (probes × amostras)
  scale.py        # detect_platform + infer_scale (log2 / raw / RMA / MAS5)
  conditions.py   # extração e filtragem de condição clínica; auto-inclusão de controles
  features.py     # canonicalize_probe_id (harmonização cross-platform) + build_sample_annotation
  dataset.py      # process_single_dataset (orquestra steps 1-7) + merge_datasets
  normalize.py    # fit/apply SEM vazamento: combat_fit/combat_apply, fit_zscore/apply_zscore,
                  #   fit_normalization/apply_normalization. (legados apply_combat/zscore_by_probe REMOVIDOS — vaziavam)
  metrics.py      # PurityB/PurityD + Silhouette antes/depois do ComBat
  plots.py        # scatter PCA (PC1-2, PC3-4) por batch e por classe
```

Fluxo de `process_single_dataset` (em `dataset.py`):
1. **Metadata** — `parse_series_metadata_tabular` (io_geo)
2. **Condições** — `extract_conditions` + `select_conditions_cli` + `auto_include_healthy_controls` (conditions)
3. **Expressão** — `read_expression` (expression)
4. **Cross-reference** — casa amostras filtradas (GSMs) com colunas de expressão
5. **Escala** — `infer_scale`; aplica `log2(x+1)` se necessário (scale)
6. **Harmonização de Probe IDs** — `build_feature_map` + `canonicalize_probe_id`; colapsa duplicatas canônicas pela **mediana** (features)
7. **Anotação de amostras** — `build_sample_annotation` → `sample_annotation.csv` com `class_label`

Depois, em `geo_mirna_pipeline.run_pipeline` (chamado por `main`): `merge_datasets` (interseção dos miRNAs comuns) → **split estratificado por classe × plataforma** (`stratified_split_ids`, `test_size=0.2`, `random_state=42`) → **ComBat fit no treino + apply no teste** (`combat_fit`/`combat_apply` via `neuroCombatFromTraining`) → **z-score fit no treino + apply** (`fit_zscore`/`apply_zscore`) → `compute_purity_metrics` → `generate_all_plots`.

> **Anti-vazamento (decisão central — não reabrir):** o split é feito na FRONTEIRA do Estágio 1, ANTES de qualquer ComBat/z-score. ComBat e z-score estimam parâmetros SOMENTE no treino e os reaplicam ao teste. Estratifica-se por classe × plataforma para que ambas as plataformas existam no treino (requisito do `neuroCombatFromTraining` e do ComBat). Generalização para plataformas não vistas está fora de escopo (limitação intrínseca do ComBat) — é a "Estratégia B (fusão)".

Saídas por dataset em `out_<GSE>/`; saídas do merge na raiz de saída:
- `base_treino.csv` / `base_teste.csv` — **entradas do Estágio 2** (Probe_ID × amostras, ComBat+z-score, sem vazamento).
- `combat_estimates.pkl` (joblib) + `zscore_params.csv` (μ/σ por Probe_ID) — parâmetros do treino, persistidos.
- `merged_sample_annotation.csv` — anotação com coluna `split` (train/test).
- `merged_expression_raw.csv` — matriz integrada log2 (antes de normalizar).
- `merged_expression_combat.csv` — matriz corrigida (treino+teste), **apenas artefato de PCA — NÃO alimenta o Estágio 2**.
- `purity_metrics.csv`, PNGs de PCA.

`TCC2/app.py` é a **versão legada monolítica** — substituída por `geo_mirna_pipeline.py`. Não usar para processamento real.

### Estágio 2 — Seleção de features (Boruta + LASSO)

Orquestrador: `TCC2/run_feature_refinement_system.py` (interativo ou CLI). Roda como **subprocessos**:
- `refine_features_pdac.py` — **Boruta** sobre Random Forest
- `refine_features_lasso.py` — **LASSO** (`logistic_l1` via LogisticRegressionCV, ou `lasso_cv`)

Desenho idêntico nos dois (com **prevenção de data leakage** como princípio central):
1. **Split**: NÃO é mais feito aqui. O Estágio 1 entrega `base_treino.csv` / `base_teste.csv` já splitados e normalizados sem vazamento. Os scripts recebem `--train-path`/`--test-path` (passados pelo orquestrador) e apenas consomem. *(Modo legado `--expr-path` com split interno 80/20 ainda existe para retrocompatibilidade.)*
2. **Step A** — filtro estatístico: t-test de Welch + correção FDR (Benjamini-Hochberg) + tamanho de efeito (Cohen's d se z-scored, senão delta de média). **Só no treino.** Fallback: top-50 por p_adj se nada passar.
3. **Step B** — Boruta ou LASSO. **Só no treino.** No LASSO o `StandardScaler` está num `Pipeline` sklearn para não vazar entre folds.
4. Aplica features selecionadas ao treino e ao teste; salva `base_treino.csv`, `base_teste.csv` (agora só com as features selecionadas), `base_pronta_para_treinamento.csv`, listas de miRNAs e `feature_selection_summary.json`.

O orquestrador resolve a entrada via `_EXPR_PRIORITY` (`base_treino.csv` em primeiro) e deriva `base_teste.csv` ao lado; passa ambos como `--train-path`/`--test-path` aos scripts downstream.

No modo `both`, o orquestrador gera `comparison/` com `comparison_report.txt` e `selected_features_overlap.csv` (miRNAs compartilhados / só-Boruta / só-LASSO).

## Dados

`TCC2/data/`: `GSE85589_series_matrix.txt` e `GSE59856_series_matrix (1).txt` — datasets reais de miRNA de PDAC do GEO.

## Testes / validação

`validation/` — scripts standalone (não usam pytest; rodam via `python validation/test_*.py`):
- `test_module_a_io_geo.py` … `test_module_i_metrics.py` — um por módulo. **`test_module_f_normalize.py`** valida a API fit/apply (F.0 prova que as legadas que vaziavam foram removidas; F.1 z-score por probe; F.2 ComBat usa batch/class; F.3 fit-treino/apply-teste).
- `test_module_j_no_leakage.py` — **anti-vazamento**: J.1 prova por perturbação que `fit_normalization` ignora o teste (corromper o teste não muda mu/sd nem `estimates`, `atol=0`); J.2 confirma que nenhum ID de teste aparece em `combat_estimates`/`zscore_params`.
- `test_integrated_flow.py` — fluxo ponta-a-ponta. INT.1–3 com GSE85589; **INT.4** roda o Estágio 1 completo (GSE85589+GSE59856) 2× e compara `base_treino.csv`/`base_teste.csv` byte-a-byte + verifica treino∩teste=∅.

> Falhas pré-existentes (não relacionadas ao fluxo fit/apply): A.5, B.3, D.1, F.1, F.3, INT.2. Já falhavam antes desta refatoração.

Padrão dos testes: imprimem `[PASSOU]`/`[FALHOU]` com detalhes; inserem `TCC2/` no `sys.path`.

## Comandos úteis

```bash
# Estágio 1 — integração (interativo abre seletor tkinter)
python TCC2/geo_mirna_pipeline.py

# Estágio 1 — não-interativo, dois datasets
python TCC2/geo_mirna_pipeline.py \
  "TCC2/data/GSE85589_series_matrix.txt" \
  "TCC2/data/GSE59856_series_matrix (1).txt" \
  --output-root ./out --no-interactive \
  --condition-filter "pancreatic cancer" "healthy control" \
  --class-map "pancreatic cancer=PDAC" "healthy control=Control"

# Estágio 2 — seleção de features (Boruta + LASSO)
# --expr-path = base_treino.csv; base_teste.csv é derivado ao lado automaticamente
python TCC2/run_feature_refinement_system.py \
  --expr-path ./out/base_treino.csv \
  --annot-path ./out/merged_sample_annotation.csv \
  --output-dir ./results_feature_selection --mode both

# Validação
python validation/test_integrated_flow.py
```

## Convenções e cuidados do código

- **Idioma**: comentários e logs em português; nomes de símbolos em inglês. Manter esse padrão.
- **Probe_ID** é a coluna-chave da matriz de expressão; amostras são colunas `GSM\d+`.
- **z-score**: nunca por dataset antes do merge. É **fit no treino (pós-ComBat) e apply no treino+teste** (`fit_zscore`/`apply_zscore`), com μ/σ por probe persistidos em `zscore_params.csv`. Nunca recalcular μ/σ usando o teste.
- **Anti data-leakage (fronteira no Estágio 1)**: o split ocorre logo após o merge, ANTES de ComBat/z-score. ComBat (`combat_fit`) e z-score (`fit_zscore`) recebem APENAS o treino — a assinatura impede passar o teste (garantia type-level). O teste é harmonizado via `combat_apply`/`apply_zscore` com os parâmetros do treino. Step A/B do Estágio 2 também só veem o treino. **Não** reintroduzir ComBat/z-score sobre a matriz inteira.
- **ComBat fit/apply**: usa `neuroCombat` + `neuroCombatFromTraining` (batch do teste precisa existir no treino → por isso o split estratifica por classe × plataforma). Não migrar para `inmoose`.
- **Probes ausentes**: no `apply`, alinhar teste por `Probe_ID` à ordem do treino; probes do treino ausentes no teste são imputadas pela mediana do treino, probes extras do teste são descartadas (ambas logadas).
- **Determinismo**: `random_state=42` no split e em todo sklearn. INT.3 e INT.4 comparam saídas byte-a-byte — preservar.
- **Encoding**: leitura de Series Matrix tem fallback robusto (chardet → utf-8 → latin-1). `app.py`/pipeline ajustam stdout para utf-8 no Windows.
- Plataformas de **mRNA** (GPL6480, GPL570, GPL96) disparam aviso — o pipeline é para **miRNA**.

## Plataforma

Windows 11, PowerShell. Caminhos com espaço (ex: `Downloads\TCC II`) — citar entre aspas.
