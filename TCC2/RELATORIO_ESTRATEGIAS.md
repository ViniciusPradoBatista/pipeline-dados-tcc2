# Relatório — Comparação de Estratégias de Normalização Cross-Platform

**Branch:** `feature/cross-platform-pipeline`
**Script:** `TCC2/pipeline.py` + `TCC2/compare_strategies.py`
**Data do teste:** 2026-04-17
**Datasets:** GSE85589 (GPL19117, Affymetrix miRNA 4.0) + GSE59856 (GPL18941, 3D-Gene Toray)
**Amostras finais:** 357 (169 Control / 188 PDAC) · **miRNAs comuns:** 2 540

---

## 1. Motivação

O relatório anterior (`RELATORIO_TESTE.md`) documentou que a estratégia original **z-score por dataset → merge → ComBat** tinha duas propriedades indesejáveis:

1. ComBat sem efeito incremental (métricas before/after idênticas).
2. PurityD = 0,527 — separação biológica PDAC/Control fraca para treino supervisionado.

A literatura (Müller et al., 2016; DOI `10.1371/journal.pone.0156594`) recomenda a ordem inversa: **merge raw log₂ → ComBat → normalização final opcional**. ComBat foi desenhado para operar na escala de expressão, não em z-scores já centrados por dataset.

Este relatório documenta a implementação da estratégia alternativa como flag CLI e a comparação quantitativa das duas ordens.

---

## 2. Implementação

### 2.1. Flags novas na CLI

```
--strategy {zscore_first,combat_first}        (default: combat_first)
--final-normalization {none,zscore_global,quantile}  (default: none)
```

- `zscore_first`: z-score por probe dentro de cada dataset → merge → ComBat (baseline do TCC2, comportamento legado).
- `combat_first`: merge dos `expression_merge_ready.csv` (log₂, pré-z-score) → ComBat → normalização final opcional.
- `--final-normalization` só é aplicada em `combat_first`; em `zscore_first` o flag é ignorado com warning.

### 2.2. Funções adicionadas em `TCC2/pipeline.py`

| Função | Papel |
|---|---|
| `zscore_global(expr_df)` | Z-score por probe sobre a matriz já corrigida por ComBat (estatísticas calculadas no merged global, não por dataset). |
| `quantile_normalize_global(expr_df)` | Quantile normalization via `sklearn.preprocessing.quantile_transform` (distribuição alvo normal, por amostra). |

### 2.3. Branch em `main()`

```python
if args.strategy == "combat_first":
    combat_input = merged_raw   # log2 concatenado pré-z-score
else:  # zscore_first
    combat_input = merged_expr  # z-score por dataset concatenado
```

`merge_datasets` e `process_dataset` **não foram alterados** — ambos já geravam `merged_expression_raw.csv` e `merged_expression_zscore.csv`; bastou escolher qual vai para o ComBat. Backward compatibility preservada.

A tabela `purity_metrics.csv` ganhou duas colunas (`strategy`, `final_normalization`) para facilitar agregação cross-runs.

### 2.4. Script comparativo `TCC2/compare_strategies.py`

Consome `purity_metrics.csv` de múltiplos `--output-root`, plota `strategy_comparison.png` (PurityB e PurityD por estágio/estratégia) e imprime resumo tabular no stdout.

---

## 3. Protocolo de teste

Quatro execuções end-to-end com os mesmos dois Series Matrix:

```bash
python TCC2/pipeline.py "TCC2/data/GSE85589_series_matrix.txt" \
  "TCC2/data/GSE59856_series_matrix (1).txt" \
  --strategy zscore_first --output-root ./out_zscore_first

python TCC2/pipeline.py ... --strategy combat_first --output-root ./out_combat_first
python TCC2/pipeline.py ... --strategy combat_first \
  --final-normalization zscore_global --output-root ./out_combat_zscore
python TCC2/pipeline.py ... --strategy combat_first \
  --final-normalization quantile --output-root ./out_combat_quantile

python TCC2/compare_strategies.py
```

Todas concluíram com exit code 0. Artefatos gerados em cada `./out_*/` são os mesmos do pipeline original (`merged_expression_*.csv`, `pca_*.png`, `purity_metrics.csv`) e, para `combat_first + final`, um arquivo adicional `merged_expression_final_{zscore_global|quantile}.csv`.

---

## 4. Resultados

### 4.1. Métricas de purity finais

| Estratégia | PurityB (↓ melhor) | PurityD (↑ melhor) | ΔPurityD vs baseline |
|---|---:|---:|---:|
| `zscore_first` (baseline) | 0,700 | 0,527 | — |
| **`combat_first`** | 0,714 | **0,936** | **+0,409** |
| `combat_first + zscore_global` | 0,700 | 0,527 | +0,000 |
| `combat_first + quantile` | 0,700 | 0,706 | +0,179 |

### 4.2. Evolução por estágio

- `zscore_first` — `merged_raw` PurityB = 1,00 → `before_combat` 0,700 → `after_combat` 0,700 (ComBat sem ganho incremental, como no relatório anterior).
- `combat_first` puro — `before_combat` PurityB = 1,00 / PurityD = 0,667 → `after_combat` 0,714 / **0,936**. ComBat remove efeito de plataforma e ao mesmo tempo deixa classes biológicas mais separadas em 10 PCs.
- `combat_first + zscore_global` — retorna exatamente às métricas do `zscore_first` (PurityD = 0,527). **Confirma quantitativamente que o z-score pós-ComBat destrói o sinal biológico** que o ComBat havia preservado.
- `combat_first + quantile` — posição intermediária. Preserva parte do ganho (PurityD = 0,706) mas fica abaixo de `combat_first` puro.

### 4.3. Figura comparativa

`strategy_comparison.png` mostra em um painel `PurityB` e `PurityD` por estágio (`before_combat` / `after_combat`) para as quatro configurações. A separação entre curvas de `combat_first` e `zscore_first` no eixo `PurityD` é visível a olho.

---

## 5. Interpretação

1. **ComBat precisa da escala original.** Aplicá-lo após z-score por dataset é ineficaz porque o efeito linear de batch já foi (parcialmente) removido; o ComBat não encontra estrutura residual para corrigir.
2. **Z-score por dataset é destrutivo para esta dupla de plataformas.** Ele colapsa a variabilidade biológica inter-sample dentro de cada dataset — daí a degradação de PurityD de 0,667 (merged_raw) para 0,527.
3. **ComBat preserva e realça a separação biológica** quando alimentado com log₂ bruto (PurityD 0,667 → 0,936) — evidência empírica de que a covariável `class_label` passada ao ComBat funcionou como protetor de sinal.
4. **Normalização final não é necessária neste dataset.** Tanto `zscore_global` quanto `quantile` degradam o resultado — `zscore_global` revertendo ao estado do baseline.

---

## 6. Consequências para as próximas etapas do TCC

- **Default do pipeline alterado** para `--strategy combat_first --final-normalization none`, refletindo a melhor configuração empírica e alinhada à literatura.
- O modo `zscore_first` continua disponível e deve ser reportado no TCC como **baseline metodológico comparativo**, não como o pipeline principal.
- Com PurityD = 0,936 na matriz de entrada, modelos supervisionados (Random Forest, Decision Tree, XGBoost) têm ponto de partida muito melhor do que com a configuração original. Ainda assim, recomenda-se `class_weight="balanced"` dado o desbalanceamento 169/188.

---

## 7. Artefatos gerados

```
./out_zscore_first/           # baseline (legado)
./out_combat_first/            # estratégia recomendada
  merged_expression_raw.csv
  merged_expression_combat.csv       ← matriz para modelagem
  merged_expression_zscore.csv       (gerada, não usada pelo ComBat nessa rota)
  merged_sample_annotation.csv
  purity_metrics.csv
  pca_{before,after}_{batch,class}.png
./out_combat_zscore/           # combat_first + zscore_global
  merged_expression_final_zscore_global.csv
./out_combat_quantile/         # combat_first + quantile
  merged_expression_final_quantile.csv

./strategy_comparison.png      # figura comparativa global
```

---

## 8. Próximos passos sugeridos

1. Treinar Random Forest / Decision Tree / XGBoost sobre `out_combat_first/merged_expression_combat.csv` com split estratificado por `class_label`, validação cruzada 5-fold e métricas AUC/F1.
2. Repetir o experimento incluindo um terceiro dataset (ex.: GSE106817 ou GSE113486) para validar que o ganho de PurityD se sustenta com 3+ plataformas.
3. Feature selection por variância inter-class nos miRNAs de `merged_expression_combat.csv` antes da modelagem, para reduzir dimensionalidade e ruído técnico residual.
4. Confirmar os achados com t-SNE / UMAP além do PCA linear — o PurityD alto em 10 PCs é indicativo, mas uma visualização não linear fortalece o argumento no texto do TCC.
