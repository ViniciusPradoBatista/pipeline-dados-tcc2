"""Normalização de expressão sem vazamento de dados.

API fit/apply: ``combat_fit`` / ``combat_apply`` / ``fit_zscore`` / ``apply_zscore``
e os wrappers ``fit_normalization`` / ``apply_normalization``. Todas estimam os
parâmetros (estimativas do ComBat e μ/σ do z-score) SOMENTE no conjunto de treino
e os reaplicam ao teste — usadas por ``geo_mirna_pipeline.run_pipeline``.

NOTA HISTÓRICA: as funções legadas ``apply_combat`` e ``zscore_by_probe``, que
ajustavam ComBat/z-score sobre a matriz inteira (treino + teste juntos), foram
REMOVIDAS por causarem vazamento de dados. Não reintroduzir uma normalização que
veja o teste durante o fit.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat, neuroCombatFromTraining

log = logging.getLogger("geo_pipeline")

# Probes com taxa de NaN acima deste limiar são removidas antes do ComBat.
_MAX_NAN_RATE_COMBAT = 0.20


# =====================================================================
# API fit/apply — SEM vazamento (parâmetros estimados só no treino)
# =====================================================================
#
# Convenção de orientação: todas as funções abaixo recebem/retornam matrizes
# de expressão como DataFrame com índice = Probe_ID e colunas = amostras (GSM).


def combat_fit(
    expr_train: pd.DataFrame,
    annot_train: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
):
    """Ajusta o ComBat APENAS no conjunto de treino.

    Args:
        expr_train: DataFrame (Probe_ID × amostras de treino), índice = Probe_ID.
        annot_train: anotação das amostras de treino (precisa de sample_id,
            ``batch_col`` e ``class_col``).

    Returns:
        (train_corrected, estimates, train_medians)
        - train_corrected: DataFrame corrigido (Probe_ID × treino).
        - estimates: dict de estimativas do ComBat, para reaplicar via
          ``neuroCombatFromTraining``.
        - train_medians: Series (indexada por Probe_ID) com a mediana de cada
          probe NO TREINO, usada para imputar NaN do teste sem vazamento.

    O ``class_label`` (mod biológico a preservar) vem SOMENTE do treino — os
    rótulos do teste nunca entram aqui.
    """
    samples = list(expr_train.columns)
    annot_ids = set(annot_train["sample_id"])
    common = [s for s in samples if s in annot_ids]
    annot_aligned = annot_train.set_index("sample_id").loc[common].reset_index()

    mat = expr_train[common].values.astype(float)  # (probes × amostras)
    probe_ids = np.asarray(expr_train.index.values)

    # --- Passo 1: remover probes inteiramente ausentes ---
    valid_mask = ~np.all(np.isnan(mat), axis=1)
    mat = mat[valid_mask]
    probe_ids = probe_ids[valid_mask]

    # --- Passo 2: remover probes com taxa de NaN > limiar ---
    nan_rates = np.isnan(mat).mean(axis=1)
    low_nan_mask = nan_rates <= _MAX_NAN_RATE_COMBAT
    n_removed = int((~low_nan_mask).sum())
    if n_removed > 0:
        log.warning(
            f"ComBat fit: removendo {n_removed} probes com >{_MAX_NAN_RATE_COMBAT*100:.0f}% "
            f"de valores ausentes no treino."
        )
    mat = mat[low_nan_mask]
    probe_ids = probe_ids[low_nan_mask]

    # --- Medianas do TREINO (antes de imputar), para reuso no teste ---
    train_medians = np.nanmedian(mat, axis=1)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

    # --- Passo 3: imputar NaN restantes pela mediana da probe (no treino) ---
    nan_mask = np.isnan(mat)
    if nan_mask.any():
        mat = np.where(nan_mask, train_medians[:, None], mat)

    covars = pd.DataFrame(
        {
            batch_col: annot_aligned[batch_col].values,
            class_col: annot_aligned[class_col].values,
        }
    )

    result = neuroCombat(
        dat=mat,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=[class_col],
    )

    probe_index = pd.Index(probe_ids, name="Probe_ID")
    train_corrected = pd.DataFrame(result["data"], index=probe_index, columns=common)
    medians = pd.Series(train_medians, index=probe_index)

    log.info(
        f"ComBat fit: {train_corrected.shape[0]} probes × {train_corrected.shape[1]} "
        f"amostras de treino; {len(np.unique(annot_aligned[batch_col]))} batches."
    )
    return train_corrected, result["estimates"], medians


def combat_apply(
    expr_test: pd.DataFrame,
    annot_test: pd.DataFrame,
    probe_index: pd.Index,
    estimates: dict,
    train_medians: pd.Series,
    batch_col: str = "batch",
) -> pd.DataFrame:
    """Reaplica o ComBat ao teste usando as estimativas do treino.

    Usa ``neuroCombatFromTraining`` (requer que cada batch do teste já exista no
    treino — garantido pelo split estratificado por classe × plataforma).

    Política de probes (decidida no projeto): a interseção treino∩teste é feita
    ANTES do ``combat_fit`` (em ``run_pipeline``), então ``probe_index`` (probes do
    treino que sobreviveram ao filtro de NaN) já é subconjunto das probes do teste.
    - Probes do treino ausentes no teste → ERRO (não imputa: violaria a premissa do
      ``neuroCombatFromTraining`` de linhas idênticas e mascararia um bug a montante).
    - Probes extras do teste (as que o ComBat descartou no treino) → descartadas + logadas.
    A imputação por mediana do treino abaixo trata apenas VALORES NaN de células de
    probes presentes — não probes ausentes.
    """
    samples = list(expr_test.columns)
    annot_ids = set(annot_test["sample_id"])
    common = [s for s in samples if s in annot_ids]
    annot_aligned = annot_test.set_index("sample_id").loc[common].reset_index()

    test_probes = set(expr_test.index)
    train_probes = set(probe_index)
    missing_in_test = [p for p in probe_index if p not in test_probes]
    extra_in_test = [p for p in expr_test.index if p not in train_probes]
    if missing_in_test:
        raise ValueError(
            f"combat_apply: {len(missing_in_test)} probes do treino ausentes no teste "
            f"(ex: {missing_in_test[:3]}). Faça a interseção treino∩teste ANTES do "
            f"combat_fit. neuroCombatFromTraining exige linhas idênticas às do treino."
        )
    if extra_in_test:
        log.warning(
            f"ComBat apply: {len(extra_in_test)} probes do teste ausentes no conjunto "
            f"do treino (descartadas pelo filtro de NaN do ComBat) — removidas do teste."
        )

    aligned = expr_test.reindex(index=probe_index)[common]
    mat = aligned.values.astype(float)

    # Imputa apenas VALORES NaN de células (probe presente, valor faltante) com a
    # mediana do treino — sem vazamento. Não há probes ausentes aqui (erro acima).
    med = train_medians.reindex(probe_index).values
    nan_mask = np.isnan(mat)
    if nan_mask.any():
        mat = np.where(nan_mask, med[:, None], mat)
        mat = np.where(np.isnan(mat), 0.0, mat)  # caso a mediana do treino seja NaN

    corrected = neuroCombatFromTraining(
        dat=mat,
        batch=np.asarray(annot_aligned[batch_col].values),
        estimates=estimates,
    )["data"]

    log.info(
        f"ComBat apply: {corrected.shape[0]} probes × {corrected.shape[1]} amostras de teste."
    )
    return pd.DataFrame(corrected, index=probe_index, columns=common)


def fit_zscore(corrected_train: pd.DataFrame):
    """Calcula μ/σ por probe (linha) ao longo das amostras de TREINO.

    Returns:
        (mu, sd) — duas Series indexadas por Probe_ID. σ usa ddof=1 (amostral);
        σ == 0 ou NaN viram 1.0 para evitar divisão por zero.
    """
    data = corrected_train.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(data, axis=1)
        sd = np.nanstd(data, axis=1, ddof=1)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    log.info(f"Z-score fit (ddof=1): {data.shape[0]} probes a partir do treino.")
    return (
        pd.Series(mu, index=corrected_train.index),
        pd.Series(sd, index=corrected_train.index),
    )


def apply_zscore(expr_df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """Aplica z-score por probe usando μ/σ do treino (alinhados por Probe_ID)."""
    mu_a = mu.reindex(expr_df.index).values[:, None]
    sd_a = sd.reindex(expr_df.index).values[:, None]
    z = (expr_df.values - mu_a) / sd_a
    return pd.DataFrame(z, index=expr_df.index, columns=expr_df.columns)


@dataclass
class NormParams:
    """Parâmetros de normalização aprendidos SOMENTE no treino."""

    combat_estimates: dict
    mu: pd.Series
    sd: pd.Series
    probe_index: pd.Index
    train_medians: pd.Series
    use_combat: bool = True


def fit_normalization(
    expr_train: pd.DataFrame,
    annot_train: pd.DataFrame,
    batch_col: str = "batch",
    class_col: str = "class_label",
    use_combat: bool = True,
) -> NormParams:
    """Ajusta ComBat (opcional) + z-score por probe usando APENAS o treino.

    A assinatura aceita exclusivamente dados de treino — é impossível passar o
    teste por engano. Esta é a garantia "type-level" anti-vazamento.
    """
    if use_combat:
        train_corr, estimates, medians = combat_fit(
            expr_train, annot_train, batch_col, class_col
        )
    else:
        train_corr = expr_train.copy()
        estimates = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(expr_train.values.astype(float), axis=1)
        medians = pd.Series(np.where(np.isnan(med), 0.0, med), index=expr_train.index)

    mu, sd = fit_zscore(train_corr)
    return NormParams(
        combat_estimates=estimates,
        mu=mu,
        sd=sd,
        probe_index=train_corr.index,
        train_medians=medians,
        use_combat=use_combat,
    )


def apply_normalization(
    expr: pd.DataFrame,
    annot: pd.DataFrame,
    params: NormParams,
    batch_col: str = "batch",
) -> pd.DataFrame:
    """Reaplica os parâmetros de treino a um conjunto qualquer (treino ou teste)."""
    if params.use_combat:
        corr = combat_apply(
            expr, annot, params.probe_index, params.combat_estimates,
            params.train_medians, batch_col,
        )
    else:
        corr = expr.reindex(index=params.probe_index)
    return apply_zscore(corr, params.mu, params.sd)
