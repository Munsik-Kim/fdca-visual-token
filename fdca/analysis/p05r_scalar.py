"""Scalar-canonical aggregation primitives for FDCA_VIS_P05R_SCALAR.

The functions here are model-free.  They distinguish local measured influence
from the finite-panel prospective envelope and expose independent arithmetic
implementations used by the verification suite.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fdca.analysis.p05_influence import (
    GRID_POSITIONS,
    distance_bin,
    merged_distance_bin,
    merged_step_bin,
    quartile_for_commit_step,
    semantic_digest,
    step_offset_bin,
)


PROFILE_COLUMNS = [
    "cell_id",
    "anchor_target_rho",
    "source_quartile",
    "step_offset_bin",
    "distance_bin",
]


def numpy_categorical_tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Independent FP64 categorical-TV implementation."""

    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    return np.abs(left - right).sum(axis=-1, dtype=np.float64) / 2.0


def _summary(frame: pd.DataFrame, columns: list[str], level: str) -> pd.DataFrame:
    grouped = frame.groupby(columns, dropna=False)["alpha_q"]
    result = grouped.agg(count="size", mean="mean", median="median", maximum="max").reset_index()
    result["q90"] = grouped.apply(
        lambda x: float(np.quantile(x.to_numpy(dtype=float), 0.90, method="higher"))
    ).to_numpy()
    result["q95"] = grouped.apply(
        lambda x: float(np.quantile(x.to_numpy(dtype=float), 0.95, method="higher"))
    ).to_numpy()
    result["fallback_level"] = level
    return result


def fit_influence_profile(frame: pd.DataFrame, minimum_count: int = 20) -> pd.DataFrame:
    """Fit the frozen hierarchy from calibration alpha_q rows only."""

    if frame.empty or set(frame["split"].unique()) != {"calibration"}:
        raise ValueError("profile input must be nonempty calibration-only rows")
    pieces: list[pd.DataFrame] = []
    exact = _summary(frame, PROFILE_COLUMNS, "exact")
    pieces.append(exact[exact["count"] >= minimum_count])

    no_quartile = [column for column in PROFILE_COLUMNS if column != "source_quartile"]
    dropped = _summary(frame, no_quartile, "drop_source_quartile")
    dropped["source_quartile"] = "*"
    pieces.append(dropped[dropped["count"] >= minimum_count])

    merged_distance = frame.copy()
    merged_distance["distance_bin"] = merged_distance["distance_bin"].map(merged_distance_bin)
    md = _summary(merged_distance, no_quartile, "merge_adjacent_distance")
    md["source_quartile"] = "*"
    pieces.append(md[md["count"] >= minimum_count])

    merged_step = merged_distance.copy()
    merged_step["step_offset_bin"] = merged_step["step_offset_bin"].map(merged_step_bin)
    ms = _summary(merged_step, no_quartile, "merge_adjacent_step_offset")
    ms["source_quartile"] = "*"
    pieces.append(ms[ms["count"] >= minimum_count])

    global_profile = _summary(frame, ["cell_id", "anchor_target_rho"], "cell_anchor_global")
    global_profile["source_quartile"] = "*"
    global_profile["step_offset_bin"] = "*"
    global_profile["distance_bin"] = "*"
    pieces.append(global_profile)
    profile = pd.concat(pieces, ignore_index=True)
    return profile[
        [*PROFILE_COLUMNS, "count", "mean", "median", "q90", "q95", "maximum", "fallback_level"]
    ].sort_values(["cell_id", "anchor_target_rho", "fallback_level", "source_quartile", "step_offset_bin", "distance_bin"], ignore_index=True)


def profile_variant(profile: pd.DataFrame, statistic: str) -> pd.DataFrame:
    if statistic not in {"q95", "maximum"}:
        raise ValueError(statistic)
    result = profile.copy()
    result["profile_statistic"] = "q95" if statistic == "q95" else "max"
    result["bound_value"] = result[statistic].astype(float)
    return result


def lookup_profile(
    row: Mapping[str, Any], profile: pd.DataFrame, statistic: str
) -> tuple[float, str, int, bool]:
    """Frozen fallback lookup; returns 1.0 only if all old levels are absent."""

    if statistic not in {"q95", "maximum"}:
        raise ValueError(statistic)
    cache_key = f"_compiled_{statistic}"
    compiled = profile.attrs.get(cache_key)
    if compiled is None:
        compiled = {}
        for item in profile.itertuples(index=False):
            compiled[
                (
                    str(item.cell_id),
                    float(item.anchor_target_rho),
                    str(item.fallback_level),
                    str(item.source_quartile),
                    str(item.step_offset_bin),
                    str(item.distance_bin),
                )
            ] = (float(getattr(item, statistic)), int(item.count))
        profile.attrs[cache_key] = compiled
    tests = [
        ("exact", row["source_quartile"], row["step_offset_bin"], row["distance_bin"]),
        ("drop_source_quartile", "*", row["step_offset_bin"], row["distance_bin"]),
        (
            "merge_adjacent_distance",
            "*",
            row["step_offset_bin"],
            merged_distance_bin(str(row["distance_bin"])),
        ),
        (
            "merge_adjacent_step_offset",
            "*",
            merged_step_bin(str(row["step_offset_bin"])),
            merged_distance_bin(str(row["distance_bin"])),
        ),
        ("cell_anchor_global", "*", "*", "*"),
    ]
    for level, quartile, step_bin, dist_bin in tests:
        key = (
            str(row["cell_id"]),
            float(row["anchor_target_rho"]),
            level,
            str(quartile),
            str(step_bin),
            str(dist_bin),
        )
        if key in compiled:
            value, count = compiled[key]
            return value, level, count, False
    return 1.0, "missing_stratum_fallback_1.0", 0, True


def measured_envelope(
    future_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    direct_b: Mapping[int, float],
    profile: pd.DataFrame,
    cell_id: str,
    anchor_target_rho: float,
    statistic: str,
) -> dict[str, Any]:
    """Time-ordered finite-panel clipped and linear envelope recursion."""

    ordered = sorted(map(int, future_positions), key=lambda position: (commit_steps[position], position))
    clipped: dict[int, float] = {}
    linear: dict[int, float] = {}
    edge_count = 0
    missing_edges = 0
    maximum_edge = 0.0
    for target in ordered:
        target_row, target_col = divmod(target, 16)
        clipped_value = float(direct_b[target])
        linear_value = float(direct_b[target])
        for source in ordered:
            if commit_steps[source] >= commit_steps[target]:
                continue
            source_row, source_col = divmod(source, 16)
            distance = abs(target_row - source_row) + abs(target_col - source_col)
            offset = int(commit_steps[target] - commit_steps[source])
            lookup = {
                "cell_id": cell_id,
                "anchor_target_rho": anchor_target_rho,
                "source_quartile": quartile_for_commit_step(commit_steps[source]),
                "step_offset_bin": step_offset_bin(offset),
                "distance_bin": distance_bin(distance),
            }
            influence, _, _, missing = lookup_profile(lookup, profile, statistic)
            edge_count += 1
            missing_edges += int(missing)
            maximum_edge = max(maximum_edge, influence)
            clipped_value += influence * clipped[source]
            linear_value += influence * linear[source]
        clipped[target] = min(1.0, clipped_value)
        linear[target] = linear_value
    b_stable = len(ordered) / GRID_POSITIONS
    b_clipped = sum(clipped.values()) / GRID_POSITIONS
    b_linear_raw = sum(linear.values()) / GRID_POSITIONS
    b_linear = sum(min(1.0, value) for value in linear.values()) / GRID_POSITIONS
    cap_count = sum(value >= 1.0 - 1e-15 for value in clipped.values())
    return {
        "B_stable": float(b_stable),
        "B_clipped": float(b_clipped),
        "B_linear": float(b_linear),
        "B_linear_raw": float(b_linear_raw),
        "B_combined": float(min(b_stable, b_clipped)),
        "edge_count": edge_count,
        "missing_edge_count": missing_edges,
        "max_profile_edge": maximum_edge,
        "capped_coordinate_count": cap_count,
        "capped_coordinate_rate": cap_count / len(ordered) if ordered else 0.0,
        "clipped_vector": clipped,
        "linear_vector": linear,
    }


def measured_envelope_independent(
    future_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    direct_b: Mapping[int, float],
    profile: pd.DataFrame,
    cell_id: str,
    anchor_target_rho: float,
    statistic: str,
) -> dict[str, float]:
    """Independent list-based recursion used only for arithmetic checking."""

    ordered = sorted((int(x) for x in future_positions), key=lambda x: (int(commit_steps[x]), x))
    clipped_values: list[float] = []
    linear_values: list[float] = []
    for index, target in enumerate(ordered):
        cr = float(direct_b[target])
        lr = float(direct_b[target])
        tr, tc = divmod(target, 16)
        for prior_index in range(index):
            source = ordered[prior_index]
            if int(commit_steps[source]) >= int(commit_steps[target]):
                continue
            sr, sc = divmod(source, 16)
            row = {
                "cell_id": cell_id,
                "anchor_target_rho": anchor_target_rho,
                "source_quartile": quartile_for_commit_step(int(commit_steps[source])),
                "step_offset_bin": step_offset_bin(int(commit_steps[target]) - int(commit_steps[source])),
                "distance_bin": distance_bin(abs(tr - sr) + abs(tc - sc)),
            }
            edge = lookup_profile(row, profile, statistic)[0]
            cr += edge * clipped_values[prior_index]
            lr += edge * linear_values[prior_index]
        clipped_values.append(min(1.0, cr))
        linear_values.append(lr)
    stable = len(ordered) / 256.0
    clipped = sum(clipped_values) / 256.0
    linear = sum(min(1.0, value) for value in linear_values) / 256.0
    return {
        "B_stable": stable,
        "B_clipped": clipped,
        "B_linear": linear,
        "B_combined": min(stable, clipped),
    }


def block_bootstrap(
    envelope_rows: pd.DataFrame,
    cell_id: str,
    ratio_column: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    frame = envelope_rows[
        (envelope_rows["split"] == "holdout") & (envelope_rows["cell_id"] == cell_id)
    ]
    per_block = frame.groupby("block_id", as_index=False)[ratio_column].mean()
    values = per_block[ratio_column].to_numpy(dtype=np.float64)
    if not len(values):
        raise ValueError(f"no holdout blocks for {cell_id}")
    derived_seed = seed ^ int.from_bytes(semantic_digest("P05R", cell_id, ratio_column)[:8], "big")
    rng = np.random.default_rng(derived_seed)
    means = np.empty(resamples, dtype=np.float64)
    medians = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = rng.choice(values, len(values), replace=True)
        means[index] = float(draw.mean())
        medians[index] = float(np.median(draw))
    return {
        "cell_id": cell_id,
        "ratio": ratio_column,
        "block_count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_ci95_lower": float(np.quantile(means, 0.025, method="higher")),
        "mean_ci95_upper": float(np.quantile(means, 0.975, method="higher")),
        "median_ci95_lower": float(np.quantile(medians, 0.025, method="higher")),
        "median_ci95_upper": float(np.quantile(medians, 0.975, method="higher")),
        "mean_u95": float(np.quantile(means, 0.95, method="higher")),
        "median_u95": float(np.quantile(medians, 0.95, method="higher")),
        "strict_tightening_block_fraction": float(np.mean(values < 1.0 - 1e-15)),
        "resamples": resamples,
        "semantic_seed": int(derived_seed),
    }


def dataframe_content_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()
