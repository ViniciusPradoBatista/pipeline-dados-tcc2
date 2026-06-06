# Pipeline de Dados — Integração de miRNA do GEO para Detecção de PDAC

Pipeline de bioinformática que integra dados de expressão de **miRNA** de múltiplos estudos públicos do [GEO](https://www.ncbi.nlm.nih.gov/geo/) (de plataformas de microarray diferentes) e seleciona um conjunto enxuto de miRNAs candidatos a **biomarcadores** para distinguir **PDAC** (adenocarcinoma ductal pancreático — câncer de pâncreas) de controles saudáveis.

Trabalho de Conclusão de Curso (TCC).

---

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como funciona](#como-funciona)
- [Estágio 1 — Integração cross-platform](#estágio-1--integração-cross-platform)
- [Estágio 2 — Seleção de features](#estágio-2--seleção-de-features)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Dados](#dados)
- [Validação](#validação)

---

## Pré-requisitos

- Python 3.9+
- Os pacotes de `TCC2/requirements.txt`

## Instalação

```bash
pip install -r TCC2/requirements.txt
```

Principais bibliotecas: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `boruta`, `neuroCombat`, `chardet`, `openpyxl`, `matplotlib`, `seaborn`.

## Como funciona

O pipeline tem **dois estágios**, cada um com seu orquestrador:

```
  Series Matrix (.txt/.xlsx)
            │
            ▼
  ┌──────────────────────────────────────┐
  │ ESTÁGIO 1 — Integração               │  geo_mirna_pipeline.py
  │ leitura → escala → harmoniza → merge │  (pacote geo_pipeline/)
  │ → SPLIT (treino/teste)               │
  │ → ComBat fit(treino)/apply(teste)    │
  │ → z-score fit(treino)/apply          │
  └──────────────────────────────────────┘
            │  base_treino.csv + base_teste.csv
            │  merged_sample_annotation.csv
            ▼
  ┌──────────────────────────────────────┐
  │ ESTÁGIO 2 — Seleção de features      │  run_feature_refinement_system.py
  │ recebe treino/teste já splitados     │  (Boruta + LASSO)
  │ → t-test+FDR → Boruta / LASSO        │
  └──────────────────────────────────────┘
            │
            ▼
   miRNAs selecionados + comparação Boruta×LASSO
```

## Estágio 1 — Integração cross-platform

Lê arquivos GEO **Series Matrix**, detecta a plataforma e a escala dos dados, **harmoniza os IDs de sonda** entre plataformas distintas e combina os datasets pelos miRNAs em comum. Em seguida **separa treino/teste** (estratificado por classe × plataforma) e só então corrige efeito de lote (**ComBat**) e normaliza (**z-score**) — ambos **estimados apenas no treino e reaplicados ao teste**, sem vazamento de dados. Por fim, valida a integração com métricas (Purity, Silhouette) e gráficos de PCA.

**Modo interativo** (abre seletor de arquivos):

```bash
python TCC2/geo_mirna_pipeline.py
```

**Modo não-interativo** (linha de comando):

```bash
python TCC2/geo_mirna_pipeline.py \
  "TCC2/data/GSE85589_series_matrix.txt" \
  "TCC2/data/GSE59856_series_matrix.txt" \
  --output-root ./out --no-interactive \
  --condition-filter "pancreatic cancer" "healthy control" \
  --class-map "pancreatic cancer=PDAC" "healthy control=Control"
```

Saídas principais (na pasta `--output-root`):

| Arquivo | Conteúdo |
|---|---|
| `out_<GSE>/` | resultados por dataset (metadados, expressão, feature map, anotação) |
| `base_treino.csv` / `base_teste.csv` | **entradas do Estágio 2** — matrizes treino/teste já com ComBat + z-score (sem vazamento) |
| `combat_estimates.pkl` / `zscore_params.csv` | parâmetros aprendidos no treino (para reaplicar/auditar) |
| `merged_sample_annotation.csv` | anotação de amostras (classe, batch, plataforma) + coluna `split` |
| `merged_expression_raw.csv` | matriz integrada (log2) sem correção |
| `merged_expression_combat.csv` | matriz corrigida (treino+teste) — **apenas artefato de PCA, não alimenta o Estágio 2** |
| `purity_metrics.csv` | métricas de validação antes/depois do ComBat |
| `pca_*.png` | gráficos de PCA por batch e por classe |

Plataformas suportadas: Affymetrix, 3D-Gene/Toray, Agilent, Illumina, entre outras.

## Estágio 2 — Seleção de features

Aplica duas abordagens de seleção e as compara, com **prevenção rigorosa de _data leakage_**: recebe o split treino/teste já feito no Estágio 1 (sem re-split) e executa todos os passos (filtro estatístico e ML) **apenas no conjunto de treino**.

1. **Step A** — filtro estatístico: t-test de Welch + correção FDR (Benjamini-Hochberg) + tamanho de efeito.
2. **Step B** — seleção por ML: **Boruta** (Random Forest) e/ou **LASSO** (regressão logística L1).

```bash
# --expr-path = base_treino.csv; base_teste.csv é localizado ao lado automaticamente
python TCC2/run_feature_refinement_system.py \
  --expr-path ./out/base_treino.csv \
  --annot-path ./out/merged_sample_annotation.csv \
  --output-dir ./results_feature_selection \
  --mode both
```

`--mode` aceita `boruta`, `lasso` ou `both`. Sem argumentos, entra em modo interativo.

Saídas (em `--output-dir`): `boruta/`, `lasso/` e (no modo `both`) `comparison/` com `comparison_report.txt` e `selected_features_overlap.csv`. Cada pipeline gera `base_treino.csv`, `base_teste.csv`, listas de miRNAs selecionados e um resumo em JSON.

## Estrutura do repositório

```
TCC2/
  geo_mirna_pipeline.py              # orquestrador do Estágio 1
  geo_pipeline/                      # módulos do Estágio 1
  run_feature_refinement_system.py   # orquestrador do Estágio 2
  refine_features_pdac.py            # seleção via Boruta
  refine_features_lasso.py           # seleção via LASSO
  app.py                             # versão legada (não usar)
  data/                              # datasets GEO (GSE85589, GSE59856)
  requirements.txt
validation/                          # testes por módulo + fluxo integrado
CLAUDE.md                            # referência técnica detalhada
```

## Código legado

`TCC2/app.py` é a versão monolítica inicial do pipeline, anterior à refatoração
modular. Está fora do fluxo atual — nenhum módulo o importa e ele não é executado
pelo pipeline. É mantido apenas como registro histórico da evolução do projeto.

➡️ O pipeline atual é o pacote `geo_pipeline/` orquestrado por
`geo_mirna_pipeline.py`. Use esse, não o `app.py`.

## Dados

`TCC2/data/` contém dois Series Matrix reais de miRNA de PDAC: **GSE85589** e **GSE59856**.

## Validação

Os scripts de teste rodam de forma standalone (sem pytest) e imprimem `[PASSOU]`/`[FALHOU]`:

```bash
python validation/test_integrated_flow.py
python validation/test_module_a_io_geo.py   # ... até o módulo i
```

O fluxo integrado inclui checagem de **determinismo** (duas execuções produzem saídas idênticas) e ausência de NaN nos dados processados.
