# Fluxograma do Pipeline

Reflete o código atual (pós-correção de vazamento). Etapas mapeadas para arquivo/função.

## Visão ASCII

```
═══════════════════════════════════════════════════════════════════════════════
  ENTRADA: arquivos GEO Series Matrix (.txt/.xlsx)
  GSE85589 (Affymetrix GPL19117)   +   GSE59856 (3D-Gene/Toray GPL18941)
═══════════════════════════════════════════════════════════════════════════════
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                        ▼
  ESTÁGIO 1A — POR DATASET (dataset.process_single_dataset), em loop:
   │ 1. Metadados ......... io_geo.parse_series_metadata_tabular (encoding robusto)
   │ 2. Plataforma ........ scale.detect_platform (avisa se for mRNA, não miRNA)
   │ 3. Condições ......... conditions.* (filtro; FAIL-LOUD se 0 matches)
   │ 4. Expressão ......... expression.read_expression + smart_float
   │                        (FAIL-LOUD se 0 colunas GSM)
   │ 5. Cross-reference ... casa GSMs filtrados × colunas de expressão
   │ 6. Escala ............ scale.infer_scale → log2(x+1) se necessário
   │ 7. Harmonização ...... features.build_feature_map + canonicalize_probe_id
   │                        (colapsa Probe_IDs duplicados pela MEDIANA)
   │ 8. Anotação .......... features.build_sample_annotation
   │                        (class_label PDAC/Control/Unknown, batch, platform_id)
   ▼
  out_<GSE>/ (expression_merge_ready.csv, sample_annotation.csv, ...)
        │
        ▼
  ESTÁGIO 1B — INTEGRAÇÃO + SPLIT + NORMALIZAÇÃO (geo_mirna_pipeline.run_pipeline)
   │ 9.  MERGE ........... dataset.merge_datasets (interseção dos miRNAs comuns;
   │                       FAIL-LOUD se colunas GSM duplicadas)
   │                       → merged_expression_raw.csv + merged_sample_annotation.csv
   │ 10. SPLIT ◄════ FRONTEIRA ANTI-VAZAMENTO ════
   │       stratified_split_ids → 80/20 estratificado por CLASSE × PLATAFORMA (rs=42)
   │       → coluna 'split'; split_composition.csv
   │       [AVISO: célula de teste frágil <10 → ex. Affy·Control=4]
   │ 11. INTERSEÇÃO probes treino∩teste (descartar + logar; ANTES do fit)
   │            ┌───────────── TREINO ───────────┬──────── TESTE (intocado) ────┐
   │ 12. ComBat   combat_fit(treino)            combat_apply(teste)
   │              → train_corr, estimates ──────► neuroCombatFromTraining → test_corr
   │              (→ combat_estimates.pkl)
   │ 13. Z-score  fit_zscore(train_corr) → μ,σ ─► apply_zscore(treino e teste)
   │              (→ zscore_params.csv)
   │       → base_treino.csv                       → base_teste.csv
   │ 14. Validação da integração:
   │       merged_expression_combat.csv (treino+teste corrigidos — SÓ p/ PCA)
   │       purity_metrics.csv (PurityB/D, Silhouette) + PCA antes/depois
        │
        ▼
  ESTÁGIO 2 — SELEÇÃO DE FEATURES (run_feature_refinement_system, modo both)
  consome base_treino.csv / base_teste.csv (SEM re-split — split veio do Estágio 1)
   │   Boruta (refine_features_pdac)        LASSO (refine_features_lasso)
   │     Step A: Welch t-test + FDR(BH) + efeito   [TREINO]
   │     Step B: RandomForest (Boruta)  |  LogReg L1 (LASSO)   [TREINO]
   ▼
  comparison/ (shared / só-Boruta / só-LASSO)
   │
   ▼
  ► MODELO final (colega): treina em base_treino, avalia em base_teste intocado.

═══════════════════════════════════════════════════════════════════════════════
  VALIDAÇÃO TÉCNICA (não altera o pipeline; lê artefatos)
═══════════════════════════════════════════════════════════════════════════════
  • validation/test_module_a..j + test_integrated_flow → suíte 40/40
       J: anti-vazamento (perturbação atol=0 + IDs)
       INT.4: determinismo base_treino/teste + treino∩teste=∅
═══════════════════════════════════════════════════════════════════════════════
```

## Versão Mermaid

```mermaid
flowchart TD
    IN["Entrada: GEO Series Matrix<br/>GSE85589 (Affymetrix) + GSE59856 (Toray)"]

    subgraph S1A["ESTÁGIO 1A — por dataset (process_single_dataset)"]
        direction TB
        M1["1. Metadados (io_geo)"] --> M2["2. Plataforma (scale.detect_platform)"]
        M2 --> M3["3. Condições + filtro<br/>(conditions; fail-loud se 0 matches)"]
        M3 --> M4["4. Expressão (read_expression + smart_float)<br/>fail-loud se 0 GSM"]
        M4 --> M5["5. Cross-reference GSMs"]
        M5 --> M6["6. Escala → log2(x+1) se preciso"]
        M6 --> M7["7. Harmonização Probe_ID<br/>+ colapso por mediana"]
        M7 --> M8["8. Anotação (class_label, batch, platform)"]
    end

    subgraph S1B["ESTÁGIO 1B — integração + split + normalização (run_pipeline)"]
        direction TB
        G9["9. MERGE (interseção miRNAs comuns)<br/>fail-loud se GSM duplicado"]
        G9 --> G10["10. SPLIT 80/20 estratificado classe x plataforma<br/>(rs=42) — FRONTEIRA anti-vazamento<br/>split_composition.csv + aviso celula fragil"]
        G10 --> G11["11. Interseção probes treino e teste (descarta+loga)"]
        G11 --> G12T["12. ComBat fit (TREINO) -> train_corr, estimates"]
        G11 --> G12V["12. ComBat apply (TESTE) via neuroCombatFromTraining"]
        G12T --> G13T["13. fit_zscore(train_corr) -> mu, sigma"]
        G12T --> G12V
        G13T --> B_TR["base_treino.csv"]
        G13T --> G13V["13. apply_zscore (teste)"]
        G12V --> G13V
        G13V --> B_TE["base_teste.csv"]
        G12T --> ART["14. Validação integração:<br/>purity_metrics.csv + PCA (antes/depois)<br/>merged_expression_combat.csv (so PCA)"]
    end

    subgraph S2["ESTÁGIO 2 — seleção de features (sem re-split)"]
        direction TB
        BOR["Boruta: t-test+FDR -> RandomForest (TREINO)"]
        LAS["LASSO: t-test+FDR -> LogReg L1 (TREINO)"]
        BOR --> CMP["comparison: shared / so-Boruta / so-LASSO"]
        LAS --> CMP
    end

    MODEL["Modelo final (colega):<br/>treina em base_treino, avalia em base_teste"]
    VAL["Validacao tecnica:<br/>suite 40/40 (modulos A-J + INT.1-4)<br/>anti-vazamento (J.1/J.2) + determinismo (INT.4)"]

    IN --> S1A --> G9
    B_TR --> S2
    B_TE --> S2
    CMP --> MODEL
    S1B -.-> VAL
    S2 -.-> VAL
```

## Notas de fidelidade

- A **fronteira anti-vazamento** é o passo 10 (split antes de qualquer ComBat/z-score).
- ComBat e z-score são sempre **fit no treino / apply no teste**; mediana de imputação é train-only.
- `merged_expression_combat.csv` é **só artefato de PCA** — não alimenta o Estágio 2.
