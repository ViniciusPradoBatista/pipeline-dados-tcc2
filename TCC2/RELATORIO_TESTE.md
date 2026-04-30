# Relatório de Teste — Pipeline Cross-Platform de miRNA (PDAC)

**Branch:** `feature/cross-platform-pipeline`
**Script:** `TCC2/pipeline.py`
**Data do teste:** 2026-04-17
**Ambiente:** Windows 11 · Python 3.11.9 · pandas · numpy · scikit-learn · matplotlib · chardet 7.4.3 · neuroCombat 0.2.12

---

## 1. Escopo do teste

Executar o pipeline end-to-end com os dois Series Matrix do TCC:

- `TCC2/data/GSE85589_series_matrix.txt` — Affymetrix GPL19117 (miRNA 4.0), 232 amostras
- `TCC2/data/GSE59856_series_matrix (1).txt` — 3D-Gene GPL18941 (Toray), 571 amostras

Comando:

```bash
python TCC2/pipeline.py \
  "TCC2/data/GSE85589_series_matrix.txt" \
  "TCC2/data/GSE59856_series_matrix (1).txt" \
  --output-root ./out
```

---

## 2. Execução inicial — falhas observadas

A primeira execução completou sem erro fatal, mas o diagnóstico mostrou três sintomas simultâneos:

| Sintoma | Evidência |
|---|---|
| `PurityB` e `PurityD` idênticos antes/depois | `0.700 → 0.700` e `0.527 → 0.527` |
| `ConvergenceWarning: distinct clusters (1) found smaller than n_clusters (2)` | KMeans colapsando |
| `RuntimeWarning: invalid value encountered in divide` no PCA pós-ComBat | Indício de NaN/inf |
| Matriz `merged_expression_combat.csv` = 832 524 NaN de 832 524 células | 100 % da saída destruída |

Causa raiz → **rastreada até o parser numérico.**

---

## 3. Bug 1 — Parser locale-quebrado (GSE85589)

### Evidência

Amostra da primeira linha de dados do GSE85589 (`MIMAT0000062_st`):

```
1.129.491.157   1.727.869.791   127.586.602   0.385183706   5.508.051.519 ...
```

- 28,4 % dos valores tinham **mais de um ponto** — impossível de parsear como `float()` em Python.
- `smart_float` (e o `app.py` original) assumiam "ponto = decimal, sem separador de milhar" → `ValueError` → `NaN`.
- 78 709 células de expressão viraram `NaN` só em GSE85589.

### Hipótese verificada

Pela comparação de percentis com GSE59856 (mediana = 2,61 em log₂ RMA), concluí que os valores reais estão em `[0, 10)` e o exportador original inseriu pontos a cada 3 dígitos a partir da direita do **stream de dígitos inteiro**, colidindo com o separador decimal.

Regra de reconstrução universal:

> Strip de **todos** os pontos → o primeiro dígito vira parte inteira, o restante vira parte decimal.

| Raw | Parse broken | Parse correto |
|---|---|---|
| `1.129.491.157` | 28 % NaN | `1.129491157` |
| `127.586.602`   | 28 % NaN | `1.27586602`  |
| `597.201`       | 597.2 (absurdo) | `5.97201`  |
| `0.385183706`   | 0.385 ✓   | `0.385183706` (idem) |
| `5.436` (GSE59856) | 5.436 ✓ | — (não aplica) |

### Correção

- **`_detect_broken_decimal_format(path, encoding)`** — inspeciona até 100 linhas de dados; se qualquer valor contém `>1` ponto, ativa modo `broken_decimal`.
- **`smart_float(v, broken_decimal=True)`** — aplica a reconstrução de dígitos.
- Rodado só quando detectado → GSE59856 (formato limpo) não é afetado.

### Resultado pós-correção

- GSE85589: `NaN frac = 0.0000` (antes: 0.285).
- Interseção cross-platform: **2 540 miRNAs comuns** (antes: 2 332).

---

## 4. Bug 2 — NaN residual contaminando o ComBat

### Evidência

Após corrigir o parser, restavam **104 NaN** dispersos na `merged_expression_zscore.csv`:

- 1 célula em GSE85589 (resíduo pontual)
- 103 células em GSE59856 (probes constantes como `MIMAT0004518` onde std=0 em subgrupos)

O `neuroCombat` não lida com `NaN`: a fatoração propaga NaN para **toda** a matriz de saída → 906 780 / 906 780 células pós-ComBat viraram NaN.

### Correção

- **`impute_by_probe_median(expr_df)`** — preenche `NaN` com a mediana do probe (linha); se o probe for 100 % NaN, preenche com 0.
- Chamada:
  - Em `process_dataset`, logo antes do `zscore_by_probe`.
  - Como rede de segurança na entrada de `apply_combat`.

### Resultado

- `merged_expression_zscore.csv`: 0 NaN.
- `merged_expression_combat.csv`: 0 NaN, `|z − combat|` médio = 0,018 (ComBat fez ajustes reais, só que de pequena magnitude).

---

## 5. Bug 3 — KMeans colapsando em cluster `{350, 7}`

### Evidência

Mesmo com matriz limpa, `KMeans(k=2)` alocava 350 amostras em um cluster e 7 em outro (outliers), gerando `PurityB = 0,700` = 250 / 357 = fração do dataset majoritário. Ou seja: sem nenhum sinal de batch.

### Causa

- Z-score por dataset (`zscore_by_probe`) já centra cada probe em média 0 / std 1 **dentro de cada dataset**.
- Concatenar datasets já z-scored elimina quase todo o sinal linear de batch no espaço de features → KMeans em 2 540 dimensões não consegue reencontrar a separação.
- Outliers de alta magnitude (valores até ±15,78 após z-score) dominam a inércia do KMeans.

### Correção

- **`_reduce_for_clustering(X, n_components=10)`** — PCA para 10 componentes antes do clustering.
- Mantém `n_clusters = n_batches` para PurityB e `n_clusters = n_classes` para PurityD, como spec.
- Aplicado também em `purity_validation` e nas plotagens PCA.

### Adicional: baseline de dados crus

Como o z-score por dataset esvazia o trabalho do ComBat, adicionei um terceiro estágio `merged_raw` na validação (concatenação dos `expression_merge_ready.csv` pré-z-score) para mostrar o efeito batch original. Também salvamos `merged_expression_raw.csv` como saída.

---

## 6. Métricas finais

### `purity_metrics.csv`

| Stage | PurityB | PurityD |
|---|---|---|
| `merged_raw`      | **1,0000** | 0,6667 |
| `before_combat` (z-scored)  | 0,7003 | 0,5266 |
| `after_combat`    | 0,7003 | 0,5266 |

### Leitura

1. **`merged_raw`** — os dados crus concatenados se separam **perfeitamente** por batch (`PurityB = 1,0`). Efeito de plataforma dominante, como esperado.
2. **`before_combat`** — z-score por dataset reduz `PurityB` de 1,0 → 0,70 (redução de 0,30), mas também degrada `PurityD` de 0,667 → 0,527. O z-score remove sinal biológico junto com batch.
3. **`after_combat`** — idêntico ao `before_combat`. O ComBat não encontra efeito batch residual para corrigir após o z-score por dataset.

### PCAs

- `pca_before_batch.png`: GSE59856 e GSE85589 sobrepostos (efeito batch atenuado pelo z-score), com cauda de amostras do GSE59856 em alta variância em PC1.
- `pca_after_batch.png`: praticamente idêntico (ComBat sem efeito incremental).
- `pca_before_class.png` / `pca_after_class.png`: PDAC e Control se misturam sem separação clara — consistente com `PurityD ≈ 0,53`.

---

## 7. Diagnóstico científico

A ordem **z-score → merge → ComBat** implementada no pipeline (conforme spec) tem três propriedades cientificamente relevantes observadas aqui:

1. **O z-score por dataset é agressivo demais para esta dupla de plataformas.** Ele remove não só o efeito batch mas também variabilidade biológica inter-sample dentro de cada dataset, reduzindo PurityD.
2. **O ComBat não tem ganho incremental** quando aplicado depois do z-score por dataset, porque o efeito batch linear já foi removido.
3. **PurityD = 0,53 é baixo** para treino supervisionado posterior. Random Forest e Decision Tree podem ter AUC modesto nessa configuração.

### Alternativas que valem ser testadas e comparadas

| Alternativa | Justificativa |
|---|---|
| ComBat sobre `merged_raw` (log₂), z-score opcional no fim | ComBat foi desenhado para operar na escala de expressão, não em z-scores. |
| Quantile normalization cross-platform antes do ComBat | Padroniza distribuição entre plataformas sem colapsar variância biológica. |
| Limma `removeBatchEffect` com design `~ class_label` | Alternativa linear ao ComBat, preserva covariáveis explicitamente. |
| Escolher apenas miRNAs com variância inter-class alta antes do merge | Reduz ruído técnico e aumenta PurityD. |

Sugiro adicionar flags `--combat-before-zscore` e `--skip-zscore` na CLI para permitir a comparação quantitativa dessas variantes como parte da análise da TCC.

---

## 8. Artefatos gerados em `./out/`

Por dataset (`out/out_GSE85589/`, `out/out_GSE59856/`):

- `expression_text_preservado.csv`
- `expression_numerico.csv`
- `expression_analysis_ready.csv`
- `feature_map.csv`
- `expression_merge_ready.csv`
- `expression_merge_ready_zscore.csv`
- `sample_annotation.csv`
- `metadata_full.csv`

Merge global (`out/`):

- `merged_expression_raw.csv` (novo — baseline)
- `merged_expression_zscore.csv`
- `merged_expression_combat.csv`
- `merged_sample_annotation.csv`
- `purity_metrics.csv`
- `pca_before_batch.png`
- `pca_before_class.png`
- `pca_after_batch.png`
- `pca_after_class.png`

---

## 9. Distribuição final das amostras

| Dataset | Plataforma | Control | PDAC | Total |
|---|---|---|---|---|
| GSE85589 | GPL19117 (Affymetrix miRNA 4.0) | 19 | 88 | 107 |
| GSE59856 | GPL18941 (3D-Gene) | 150 | 100 | 250 |
| **Merge** | — | **169** | **188** | **357** |

Desbalanceamento moderado de Control/PDAC (≈ 47/53 %). Para modelagem supervisionada, considerar `class_weight="balanced"` ou SMOTE.

---

## 10. Estado do repositório

- Commit atual local `7f65078` — versão inicial do `pipeline.py`, ainda sem as correções deste relatório.
- Correções deste relatório **ainda não comitadas** — estão no working tree local:
  - `smart_float` com `broken_decimal`
  - `_detect_broken_decimal_format`
  - `impute_by_probe_median`
  - `_reduce_for_clustering` + PCA no `purity_validation`
  - `merge_datasets` retornando `(raw, zscore, annot)`
  - output extra `merged_expression_raw.csv`

Ação pendente: commit adicional na branch `feature/cross-platform-pipeline`.
