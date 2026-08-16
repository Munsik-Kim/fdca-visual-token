"""Model-free P1M incremental predictive-value analysis utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg
from scipy.stats import rankdata, spearmanr


RHO_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
RIDGE_ALPHA = 1.0e-6
BOOTSTRAP_RESAMPLES = 20_000
CONTINUOUS_M1 = [
    "actual_rho", "unresolved_fraction", "remaining_steps_fraction",
    "normalized_source_commit_step", "source_x_fraction", "source_y_fraction",
    "logit_alternative_probability", "source_selection_fallback",
]
PROFILE_MAX = ["B_total_max", "B_desc_max", "max_cap_rate"]
PROFILE_Q95 = ["B_total_q95", "B_desc_q95", "q95_cap_rate"]


def semantic_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"FDCA-P1M-BOOTSTRAP|{label}".encode()).digest()[:8], "big")


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return 0.0
    return float(spearmanr(x[mask], y[mask]).statistic)


def source_quartile(step: int) -> str:
    if step < 8:
        return "oldest"
    if step < 16:
        return "early_middle"
    if step < 24:
        return "late_middle"
    return "newest"


def build_analysis_rows(
    terminal: pd.DataFrame,
    anchor: pd.DataFrame,
    source: pd.DataFrame,
    envelope: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scientific = terminal[terminal.control_type.eq("single_shock")].copy()
    key = ["block_id", "anchor_id", "cell_id", "source_type", "replicate_id"]
    if scientific.duplicated(key).any():
        raise ValueError("duplicate scientific terminal key")
    anchor_key = ["block_id", "anchor_id"]
    source_key = ["block_id", "anchor_id", "source_type"]
    envelope_key = ["block_id", "anchor_id", "cell_id", "source_type"]
    if anchor.duplicated(anchor_key).any() or source.duplicated(source_key).any() or envelope.duplicated(envelope_key).any():
        raise ValueError("nonunique join source")
    anchor_columns = anchor_key + ["class_id", "sample_seed_uint64", "actual_stable_count"]
    source_columns = source_key + [
        "target_quartile", "quartile_fallback", "source_position", "source_row", "source_col",
        "source_commit_step", "actual_token", "alternative_token", "alternative_probability",
    ]
    envelope_columns = envelope_key + [
        "q95_missing_lookup_count", "max_missing_lookup_count", "direct_innovation_b",
        "independent_recursion_max_discrepancy",
    ]
    rows = scientific.merge(anchor[anchor_columns], on=anchor_key, how="left", validate="many_to_one")
    rows = rows.merge(source[source_columns], on=source_key, how="left", validate="many_to_one")
    rows = rows.merge(envelope[envelope_columns], on=envelope_key, how="left", validate="many_to_one")
    rows["rho"] = rows.anchor_target_rho.astype(float)
    rows["unresolved_count"] = rows.actual_unresolved_count.astype(int)
    rows["remaining_steps"] = 32 - rows.anchor_step.astype(int)
    rows["unresolved_fraction"] = rows.unresolved_count / 256.0
    rows["remaining_steps_fraction"] = rows.remaining_steps / 32.0
    rows["normalized_source_commit_step"] = rows.source_commit_step / 31.0
    rows["source_x_fraction"] = rows.source_col / 15.0
    rows["source_y_fraction"] = rows.source_row / 15.0
    probability = rows.alternative_probability.clip(1.0e-12, 1.0 - 1.0e-12)
    rows["logit_alternative_probability"] = np.log(probability / (1.0 - probability))
    rows["source_selection_fallback"] = rows.quartile_fallback.astype(int)
    rows["source_commit_quartile"] = rows.source_commit_step.map(lambda value: source_quartile(int(value)))
    rows["source_center_distance"] = np.sqrt((rows.source_row - 7.5) ** 2 + (rows.source_col - 7.5) ** 2)
    rows["descendant_fraction_within_5"] = np.where(
        rows.descendant_count > 0, rows.descendants_within_distance_5 / rows.descendant_count, 0.0,
    )
    rows["extinction_indicator"] = rows.propagation_extinction.astype(int)
    # Retained metadata availability is explicit; these are never imputed or used.
    for column in ["actual_token_probability", "alternative_token_rank", "commit_time_margin", "commit_time_entropy", "log_probability_ratio"]:
        rows[column] = np.nan
    required = [
        "class_id", "sample_seed_uint64", "alternative_probability", "D_total", "D_desc_total",
        "D_desc_rate", "B_total_max", "B_total_q95", "B_desc_max", "B_desc_q95",
    ]
    missing = {column: int(rows[column].isna().sum()) for column in required}
    audit = {
        "schema": "FDCA_VIS_P1M_JOIN_AND_KEY_AUDIT_V1",
        "terminal_total_rows": len(terminal), "zero_shock_rows_excluded": int(terminal.control_type.eq("zero_shock").sum()),
        "scientific_analysis_rows": len(rows), "w8_rows": int(rows.cell_id.eq("W8_L1.00").sum()),
        "w4_rows": int(rows.cell_id.eq("W4_L1.00").sum()),
        "duplicate_primary_keys": int(rows.duplicated(key).sum()), "missing_required": missing,
        "fallback_source_rows_unique": int(source.quartile_fallback.sum()),
        "unavailable_retained_metadata": ["actual_token_probability", "alternative_token_rank", "commit_time_margin", "commit_time_entropy", "log_probability_ratio"],
        "pass": not rows.duplicated(key).any() and max(missing.values()) == 0,
    }
    return rows.sort_values(key, kind="mergesort").reset_index(drop=True), audit


def aggregate_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["block_id", "cell_id", "rho", "source_type"]
    numeric = [
        "actual_rho", "anchor_step", "unresolved_count", "remaining_steps", "unresolved_fraction",
        "remaining_steps_fraction", "normalized_source_commit_step", "source_x_fraction", "source_y_fraction",
        "source_row", "source_col", "source_commit_step", "alternative_probability",
        "logit_alternative_probability", "source_selection_fallback", "source_center_distance",
        "D_total", "D_desc_total", "D_desc_rate", "extinction_indicator", "maximum_manhattan_radius",
        "connected_component_count", "descendant_fraction_within_5", "B_stable_total", "B_seed",
        "B_desc_q95", "B_desc_max", "B_total_q95", "B_total_max", "B_combined_q95",
        "B_combined_max", "q95_prediction_slack", "max_prediction_slack", "q95_cap_rate", "max_cap_rate",
    ]
    grouped = rows.groupby(key, as_index=False)[numeric].mean()
    counts = rows.groupby(key, as_index=False).size().rename(columns={"size": "replicate_count"})
    grouped = grouped.merge(counts, on=key, validate="one_to_one")
    block_anchor = grouped.groupby(["block_id", "cell_id", "rho"], as_index=False)[numeric].mean()
    return grouped.sort_values(key).reset_index(drop=True), block_anchor.sort_values(["block_id", "cell_id", "rho"]).reset_index(drop=True)


def categorical_design(frame: pd.DataFrame, include_source: bool, interactions: bool) -> tuple[list[np.ndarray], list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for rho in RHO_LEVELS[1:]:
        arrays.append(np.isclose(frame.rho.to_numpy(float), rho).astype(float))
        names.append(f"rho_{rho:.1f}")
    if include_source:
        source = frame.source_type.eq("newest_source").to_numpy(float)
        arrays.append(source)
        names.append("source_newest")
        if interactions:
            for rho in RHO_LEVELS[1:]:
                arrays.append(source * np.isclose(frame.rho.to_numpy(float), rho).astype(float))
                names.append(f"rho_{rho:.1f}:source_newest")
    return arrays, names


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "rho_only": {"rho": True, "source": False, "interactions": False, "continuous": []},
    "remaining_steps_only": {"rho": False, "source": False, "interactions": False, "continuous": ["remaining_steps_fraction"]},
    "unresolved_only": {"rho": False, "source": False, "interactions": False, "continuous": ["unresolved_fraction"]},
    "stable_only": {"rho": False, "source": False, "interactions": False, "continuous": ["B_stable_total"]},
    "rho_source_age": {"rho": True, "source": True, "interactions": False, "continuous": []},
    "rho_shock_severity": {"rho": True, "source": False, "interactions": False, "continuous": ["logit_alternative_probability"]},
    "profile_only": {"rho": False, "source": False, "interactions": False, "continuous": PROFILE_MAX},
    "M0": {"rho": True, "source": True, "interactions": True, "continuous": []},
    "M1": {"rho": True, "source": True, "interactions": True, "continuous": CONTINUOUS_M1},
    "M2": {"rho": True, "source": True, "interactions": True, "continuous": CONTINUOUS_M1 + PROFILE_MAX},
    "M3": {"rho": True, "source": True, "interactions": True, "continuous": CONTINUOUS_M1 + PROFILE_Q95},
}


def design_train_test(train: pd.DataFrame, test: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train_arrays = [np.ones(len(train))]
    test_arrays = [np.ones(len(test))]
    names = ["intercept"]
    if spec["rho"]:
        tr, cat_names = categorical_design(train, bool(spec["source"]), bool(spec["interactions"]))
        te, _ = categorical_design(test, bool(spec["source"]), bool(spec["interactions"]))
        train_arrays.extend(tr); test_arrays.extend(te); names.extend(cat_names)
    elif spec["source"]:
        train_arrays.append(train.source_type.eq("newest_source").to_numpy(float))
        test_arrays.append(test.source_type.eq("newest_source").to_numpy(float))
        names.append("source_newest")
    processing = {}
    for column in spec["continuous"]:
        train_value = train[column].to_numpy(float)
        test_value = test[column].to_numpy(float)
        median = float(np.nanmedian(train_value))
        train_value = np.where(np.isfinite(train_value), train_value, median)
        test_value = np.where(np.isfinite(test_value), test_value, median)
        mean = float(train_value.mean())
        scale = float(train_value.std(ddof=0))
        if scale <= 1e-15:
            scale = 1.0
        train_arrays.append((train_value - mean) / scale)
        test_arrays.append((test_value - mean) / scale)
        names.append(column)
        processing[column] = {"median": median, "mean": mean, "scale": scale}
    return np.column_stack(train_arrays), np.column_stack(test_arrays), names, processing


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=float) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    augmented_x = np.vstack([X, penalty])
    augmented_y = np.concatenate([y, np.zeros(X.shape[1])])
    return np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]


def ridge_fit_independent(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=float) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    augmented_x = np.vstack([X, penalty])
    augmented_y = np.concatenate([y, np.zeros(X.shape[1])])
    return scipy_linalg.lstsq(augmented_x, augmented_y, lapack_driver="gelsy")[0]


def lobo_predictions(frame: pd.DataFrame, model_name: str, independent: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = MODEL_SPECS[model_name]
    blocks = sorted(frame.block_id.unique())
    predictions = []
    fold_audits = []
    for held in blocks:
        train = frame[frame.block_id.ne(held)]
        test = frame[frame.block_id.eq(held)]
        X_train, X_test, names, processing = design_train_test(train, test, spec)
        y_train = train.D_desc_rate.to_numpy(float)
        beta = (ridge_fit_independent if independent else ridge_fit)(X_train, y_train)
        predicted = X_test @ beta
        piece = test[["block_id", "cell_id", "rho", "source_type", "D_desc_rate"]].copy()
        piece["model"] = model_name
        piece["prediction"] = predicted
        predictions.append(piece)
        fold_audits.append({
            "held_out_block": held, "training_blocks": len(set(train.block_id)), "test_rows": len(test),
            "parameter_count": len(names), "feature_names": names, "processing": processing,
            "held_out_used_for_processing": False,
            "condition_number": float(np.linalg.cond(X_train)),
        })
    result = pd.concat(predictions, ignore_index=True).sort_values(["block_id", "rho", "source_type"]).reset_index(drop=True)
    return result, {"model": model_name, "folds": fold_audits, "complete": len(result) == len(frame), "unique_held_blocks": len(blocks)}


def within_rho_residual_spearman(frame: pd.DataFrame, observed: str, predicted: str) -> float:
    left = frame[observed] - frame.groupby("rho")[observed].transform("mean")
    right = frame[predicted] - frame.groupby("rho")[predicted].transform("mean")
    return safe_spearman(left, right)


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    error = predictions.prediction - predictions.D_desc_rate
    block_mae = predictions.assign(abs_error=np.abs(error)).groupby("block_id").abs_error.mean()
    return {
        "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Spearman": safe_spearman(predictions.prediction, predictions.D_desc_rate),
        "within_rho_residual_Spearman": within_rho_residual_spearman(predictions, "D_desc_rate", "prediction"),
        "block_level_MAE": float(block_mae.mean()),
    }


def bootstrap_mean(values: Mapping[str, float], label: str, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    blocks = sorted(values)
    array = np.asarray([values[block] for block in blocks], dtype=float)
    rng = np.random.default_rng(semantic_seed(label))
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    draws = array[indices].mean(axis=1)
    return {
        "block_count": len(array), "point_estimate": float(array.mean()),
        "ci95_lower": float(np.quantile(draws, .025, method="higher")),
        "ci95_upper": float(np.quantile(draws, .975, method="higher")),
        "semantic_seed": semantic_seed(label),
    }


def fixed_rank_block_bootstrap(frame: pd.DataFrame, x: str, y: str, label: str, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    work = frame[["block_id", x, y]].dropna().copy()
    work["rx"] = rankdata(work[x].to_numpy(float), method="average")
    work["ry"] = rankdata(work[y].to_numpy(float), method="average")
    blocks = sorted(work.block_id.unique())
    stats = []
    for block in blocks:
        part = work[work.block_id.eq(block)]
        rx, ry = part.rx.to_numpy(), part.ry.to_numpy()
        stats.append([len(part), rx.sum(), ry.sum(), np.square(rx).sum(), np.square(ry).sum(), (rx * ry).sum()])
    stats = np.asarray(stats, dtype=float)
    rng = np.random.default_rng(semantic_seed(label))
    indices = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    totals = stats[indices].sum(axis=1)
    n, sx, sy, sx2, sy2, sxy = totals.T
    numerator = sxy - sx * sy / n
    denominator = np.sqrt(np.maximum((sx2 - sx * sx / n) * (sy2 - sy * sy / n), 0))
    draws = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    point = safe_spearman(work[x], work[y])
    return {
        "point_estimate": point, "ci95_lower": float(np.quantile(draws, .025, method="higher")),
        "ci95_upper": float(np.quantile(draws, .975, method="higher")), "block_count": len(blocks),
        "bootstrap_method": "block_resample_fixed_original_ranks", "semantic_seed": semantic_seed(label),
    }


def block_improvement(m1: pd.DataFrame, m2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["block_id", "cell_id", "rho", "source_type"]
    left = m1.rename(columns={"prediction": "prediction_M1"})
    right = m2[keys + ["prediction"]].rename(columns={"prediction": "prediction_M2"})
    merged = left.merge(right, on=keys, validate="one_to_one")
    merged["abs_error_M1"] = np.abs(merged.D_desc_rate - merged.prediction_M1)
    merged["abs_error_M2"] = np.abs(merged.D_desc_rate - merged.prediction_M2)
    per_block = merged.groupby("block_id", as_index=False).agg(MAE_M1=("abs_error_M1", "mean"), MAE_M2=("abs_error_M2", "mean"))
    per_block["improvement"] = per_block.MAE_M1 - per_block.MAE_M2
    audit = bootstrap_mean(dict(zip(per_block.block_id, per_block.improvement)), "M2-versus-M1-block-MAE")
    return per_block, audit


def empirical_rank_prediction_mae(profile: Sequence[float], observed: Sequence[float]) -> float:
    """MAE after deterministic monotone empirical-quantile rank rescaling."""
    x, y = np.asarray(profile, dtype=float), np.asarray(observed, dtype=float)
    ranks = rankdata(x, method="average")
    quantile_index = np.clip(np.rint((ranks - 1) / max(len(ranks) - 1, 1) * (len(y) - 1)).astype(int), 0, len(y) - 1)
    prediction = np.sort(y)[quantile_index]
    return float(np.mean(np.abs(prediction - y)))


def within_rho_profile_table(block_anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w8 = block_anchor[block_anchor.cell_id.eq("W8_L1.00")]
    features = ["B_total_max", "B_desc_max", "B_total_q95", "B_desc_q95"]
    for rho in RHO_LEVELS:
        part = w8[np.isclose(w8.rho, rho)]
        for feature in features:
            interval = fixed_rank_block_bootstrap(part, feature, "D_desc_rate", f"within-rho-{rho:.1f}-{feature}")
            rows.append({
                "cell_id": "W8_L1.00", "rho": rho, "profile_feature": feature,
                "spearman": interval["point_estimate"], "ci95_lower": interval["ci95_lower"],
                "ci95_upper": interval["ci95_upper"],
                "monotone_rank_prediction_MAE": empirical_rank_prediction_mae(part[feature], part.D_desc_rate),
                "block_count": len(part), "analysis_kind": "post-hoc model-free diagnostic",
            })
    result = pd.DataFrame(rows)
    return result


def within_rho_combined(within: pd.DataFrame, feature: str) -> dict[str, Any]:
    values = within[within.profile_feature.eq(feature)].sort_values("rho").spearman.to_numpy(float)
    clipped = np.clip(values, -0.999999, 0.999999)
    return {
        "feature": feature, "equal_rho_mean_spearman": float(values.mean()),
        "fisher_z_mean_spearman": float(np.tanh(np.arctanh(clipped).mean())),
        "positive_rho_count": int((values > 0).sum()),
    }


def trajectory_adequacy(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    group_columns = ["cell_id", "rho", "source_type"]
    for keys, part in rows.groupby(group_columns):
        for profile in ["q95", "max"]:
            bound = part[f"B_combined_{profile}"].to_numpy(float)
            slack = part.D_total.to_numpy(float) - bound
            positive = slack[slack > 0]
            ratios = np.divide(bound, part.D_total, out=np.full(len(part), np.nan), where=part.D_total.to_numpy(float) > 0)
            block_event = part.assign(exceed=slack > 0).groupby("block_id").exceed.any()
            for stratum, subset in [("all", np.ones(len(part), dtype=bool)), ("extinct", part.extinction_indicator.eq(1).to_numpy()), ("propagated", part.extinction_indicator.eq(0).to_numpy())]:
                sub_slack = slack[subset]
                output.append({
                    "cell_id": keys[0], "rho": keys[1], "source_type": keys[2], "profile": profile,
                    "extinction_stratum": stratum, "trajectory_count": int(subset.sum()),
                    "exceedance_fraction": float(np.mean(sub_slack > 0)) if len(sub_slack) else np.nan,
                    "maximum_positive_exceedance": float(max(np.max(sub_slack), 0)) if len(sub_slack) else np.nan,
                    "q90_positive_exceedance": float(np.quantile(positive, .90, method="higher")) if len(positive) else 0.0,
                    "q95_positive_exceedance": float(np.quantile(positive, .95, method="higher")) if len(positive) else 0.0,
                    "mean_slack": float(np.mean(sub_slack)) if len(sub_slack) else np.nan,
                    "median_slack": float(np.median(sub_slack)) if len(sub_slack) else np.nan,
                    "median_profile_over_actual_nonzero": float(np.nanmedian(ratios[subset])) if np.isfinite(ratios[subset]).any() else np.nan,
                    "mean_cap_rate": float(part.loc[subset, f"{profile}_cap_rate"].mean()) if subset.sum() else np.nan,
                    "block_any_exceedance_fraction": float(block_event.mean()) if stratum == "all" else np.nan,
                    "analysis_kind": "post-hoc model-free diagnostic",
                })
    return pd.DataFrame(output)


def paired_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    pair_key = ["block_id", "cell_id", "rho", "replicate_id"]
    metrics = [
        "D_desc_rate", "D_total", "maximum_manhattan_radius", "connected_component_count",
        "alternative_probability", "logit_alternative_probability", "source_commit_step",
        "source_row", "source_col", "source_center_distance", "source_selection_fallback",
    ]
    old = rows[rows.source_type.eq("oldest_source")][pair_key + metrics].rename(columns={m: f"oldest_{m}" for m in metrics})
    new = rows[rows.source_type.eq("newest_source")][pair_key + metrics].rename(columns={m: f"newest_{m}" for m in metrics})
    paired = old.merge(new, on=pair_key, validate="one_to_one")
    for metric in metrics[:-1]:
        paired[f"delta_{metric}"] = paired[f"oldest_{metric}"] - paired[f"newest_{metric}"]
    paired["both_intended_quartile"] = paired.oldest_source_selection_fallback.eq(0) & paired.newest_source_selection_fallback.eq(0)
    paired["fallback_affected"] = ~paired.both_intended_quartile
    return paired


def paired_contrast(paired: pd.DataFrame, subset: str, cell: str = "W8_L1.00") -> dict[str, Any]:
    part = paired[paired.cell_id.eq(cell)]
    if subset == "no_fallback":
        part = part[part.both_intended_quartile]
    elif subset == "fallback_affected":
        part = part[part.fallback_affected]
    elif subset != "full":
        raise ValueError(subset)
    block = part.groupby("block_id").delta_D_desc_rate.mean().to_dict()
    result = bootstrap_mean(block, f"source-age-{cell}-{subset}")
    result.update({"cell_id": cell, "subset": subset, "pair_rows": len(part), "mean_delta": float(part.delta_D_desc_rate.mean()) if len(part) else np.nan})
    return result


def bootstrap_ridge_coefficient(
    block_ids: Sequence[str], X: np.ndarray, y: np.ndarray, coefficient_index: int,
    label: str, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, np.ndarray]:
    blocks = sorted(set(map(str, block_ids)))
    block_array = np.asarray(list(map(str, block_ids)))
    xtx = np.stack([X[block_array == block].T @ X[block_array == block] for block in blocks])
    xty = np.stack([X[block_array == block].T @ y[block_array == block] for block in blocks])
    penalty = np.eye(X.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    point = float(np.linalg.solve(X.T @ X + penalty, X.T @ y)[coefficient_index])
    rng = np.random.default_rng(semantic_seed(label))
    counts = rng.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=resamples)
    matrices = np.einsum("rb,bij->rij", counts, xtx) + penalty
    vectors = np.einsum("rb,bi->ri", counts, xty)
    coefficients = np.linalg.solve(matrices, vectors[..., None])[:, coefficient_index, 0]
    return point, coefficients


def adjusted_source_age(paired: pd.DataFrame, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    part = paired[paired.cell_id.eq("W8_L1.00")].copy()
    arrays = [np.ones(len(part))]
    for rho in RHO_LEVELS[1:]:
        arrays.append(np.isclose(part.rho.to_numpy(float), rho).astype(float))
    arrays.extend([
        part.delta_logit_alternative_probability.to_numpy(float),
        part.delta_source_commit_step.to_numpy(float) / 31.0,
        part.delta_source_center_distance.to_numpy(float) / 11.0,
    ])
    X = np.column_stack(arrays)
    point, draws = bootstrap_ridge_coefficient(
        part.block_id, X, part.delta_D_desc_rate.to_numpy(float), 0,
        "adjusted-source-age-intercept", resamples,
    )
    blocks = sorted(part.block_id.unique())
    return {
        "adjusted_source_age_intercept": point,
        "ci95_lower": float(np.quantile(draws, .025, method="higher")),
        "ci95_upper": float(np.quantile(draws, .975, method="higher")),
        "block_count": len(blocks), "semantic_seed": semantic_seed("adjusted-source-age-intercept"),
        "interpretation": "oldest-minus-newest adjusted intercept at rho=0.1 reference and zero standardized covariate deltas",
    }


def source_rho_signs(paired: pd.DataFrame) -> list[dict[str, Any]]:
    part = paired[paired.cell_id.eq("W8_L1.00")]
    return [{"rho": rho, "mean_oldest_minus_newest": float(part[np.isclose(part.rho, rho)].delta_D_desc_rate.mean())} for rho in RHO_LEVELS]


def paired_rho_contrast(frame: pd.DataFrame, value: str, left_rho: float = .5, right_rho: float = .3, label: str = "rho") -> dict[str, Any]:
    means = frame.groupby(["block_id", "rho"])[value].mean().unstack("rho")
    means = means.dropna(subset=[left_rho, right_rho])
    differences = (means[left_rho] - means[right_rho]).to_dict()
    result = bootstrap_mean(differences, label)
    result.update({"left_rho": left_rho, "right_rho": right_rho, "mean_difference": float(np.mean(list(differences.values()))), "paired_blocks": len(differences)})
    return result


def severity_adjusted_rho_contrast(block_source: pd.DataFrame, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    part = block_source[
        block_source.cell_id.eq("W8_L1.00") & block_source.rho.isin([.3, .5])
    ].copy()

    rho = np.isclose(part.rho.to_numpy(float), .5).astype(float)
    source = part.source_type.eq("newest_source").to_numpy(float)
    fallback = part.source_selection_fallback.to_numpy(float)
    severity = part.logit_alternative_probability.to_numpy(float)
    X = np.column_stack([np.ones(len(part)), rho, source, fallback, severity])
    point, draws = bootstrap_ridge_coefficient(
        part.block_id, X, part.D_desc_rate.to_numpy(float), 1,
        "severity-adjusted-rho05-minus-rho03", resamples,
    )
    blocks = sorted(part.block_id.unique())
    return {
        "contrast": "rho_0.5_minus_0.3", "adjusted_coefficient": point,
        "ci95_lower": float(np.quantile(draws, .025, method="higher")),
        "ci95_upper": float(np.quantile(draws, .975, method="higher")),
        "block_count": len(blocks), "semantic_seed": semantic_seed("severity-adjusted-rho05-minus-rho03"),
    }


def profile_label(m1: Mapping[str, float], m2: Mapping[str, float], improvement: Mapping[str, Any], within_interval: Mapping[str, Any], rho_order_reproduced: bool, valid: bool) -> str:
    if not valid:
        return "PROFILE_INCREMENTAL_VALUE_UNRESOLVED"
    adds = (
        m2["MAE"] <= .90 * m1["MAE"] and improvement["ci95_lower"] > 0
        and m2["within_rho_residual_Spearman"] > .20 and within_interval["ci95_lower"] > 0
    )
    if adds:
        return "PROFILE_ADDS_INCREMENTAL_VALUE"
    # A positive within-rho association that falls just short of the fixed
    # >0.20 threshold is not "no association"; it is ordering-only evidence.
    no_better = m2["MAE"] >= m1["MAE"] and m2["within_rho_residual_Spearman"] <= 0.0
    if no_better:
        return "PROFILE_NO_INCREMENTAL_VALUE"
    if rho_order_reproduced:
        return "PROFILE_ORDERING_ONLY"
    return "PROFILE_NO_INCREMENTAL_VALUE"


def source_age_label(full: Mapping[str, Any], no_fallback: Mapping[str, Any], adjusted: Mapping[str, Any], signs: Sequence[Mapping[str, Any]]) -> str:
    robust = (
        full["ci95_lower"] > 0 and no_fallback["ci95_lower"] > 0 and adjusted["ci95_lower"] > 0
        and sum(row["mean_oldest_minus_newest"] > 0 for row in signs) >= 4
    )
    return "SOURCE_AGE_ASSOCIATION_ROBUST" if robust else "SOURCE_AGE_ASSOCIATION_CONFOUNDED_OR_UNSTABLE"


def regime_label(frozen: Mapping[str, Any], source_sensitivities: Sequence[Mapping[str, Any]], no_fallback: Mapping[str, Any], severity: Mapping[str, Any], all_sensitivities: Sequence[Mapping[str, Any]]) -> str:
    source_positive = all(row["mean_difference"] > 0 for row in source_sensitivities)
    no_fallback_positive = no_fallback["mean_difference"] > 0
    severity_positive = severity["adjusted_coefficient"] > 0
    significant_reversal = any(row.get("ci95_upper", 1.0) < 0 for row in all_sensitivities)
    robust = frozen["mean_difference"] > 0 and frozen["ci95_lower"] > 0 and source_positive and no_fallback_positive and severity_positive and not significant_reversal
    return "REGIME_RESULT_ROBUST" if robust else "REGIME_RESULT_SENSITIVE"


def strategy_label(validity: str, profile: str, regime: str) -> str:
    if validity != "PASS_FDCA_VIS_P1M_VALIDITY":
        return "STOP_OR_REFRAME_VISUAL_FDCA"
    if regime != "REGIME_RESULT_ROBUST":
        return "PROCEED_MATCHED_SHOCK_TIMING_CONTROL_FIRST"
    if profile == "PROFILE_ADDS_INCREMENTAL_VALUE":
        return "PROCEED_NATURAL_SHOCK_WITH_PROFILE"
    return "PROCEED_NATURAL_SHOCK_REGIME_PRIMARY_PROFILE_SECONDARY"
