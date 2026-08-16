"""Model-free P1L propagation, spatial, bootstrap, and gate arithmetic."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fdca.analysis.p05_influence import distance_bin, quartile_for_commit_step, step_offset_bin
from fdca.analysis.p05r_scalar import lookup_profile


N = 256


def semantic_digest(*parts: Any) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()


def mismatch_vector_hex(positions: Sequence[int]) -> str:
    bits = np.zeros(N, dtype=np.uint8)
    bits[list(map(int, positions))] = 1
    return np.packbits(bits, bitorder="little").tobytes().hex()


def positions_from_hex(value: str) -> list[int]:
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8), bitorder="little")[:N]
    return np.flatnonzero(bits).astype(int).tolist()


def connected_components(positions: Sequence[int]) -> int:
    remaining = set(map(int, positions))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            p = stack.pop()
            row, col = divmod(p, 16)
            neighbors = []
            if row: neighbors.append(p - 16)
            if row < 15: neighbors.append(p + 16)
            if col: neighbors.append(p - 1)
            if col < 15: neighbors.append(p + 1)
            for q in neighbors:
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return count


def spatial_metrics(source: int, descendants: Sequence[int]) -> dict[str, Any]:
    sr, sc = divmod(int(source), 16)
    positions = list(map(int, descendants))
    distances = [abs(divmod(p, 16)[0] - sr) + abs(divmod(p, 16)[1] - sc) for p in positions]
    return {
        "descendant_count": len(positions),
        "connected_component_count": connected_components([int(source), *positions]),
        "maximum_manhattan_radius": max(distances, default=0),
        "mean_descendant_distance": float(np.mean(distances)) if distances else np.nan,
        "median_descendant_distance": float(np.median(distances)) if distances else np.nan,
        "descendants_within_distance_5": sum(d <= 5 for d in distances),
        "descendant_distance_sum": int(sum(distances)),
    }


def source_rank_key(block_id: str, anchor_id: str, position: int, source_type: str) -> bytes:
    return semantic_digest("FDCA-P1L-SOURCE", block_id, anchor_id, int(position), source_type)


def choose_source(
    stable_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    block_id: str,
    anchor_id: str,
    source_type: str,
) -> tuple[int, bool, str]:
    if source_type not in {"oldest_source", "newest_source"}:
        raise ValueError(source_type)
    stable = list(map(int, stable_positions))
    wanted = "oldest" if source_type == "oldest_source" else "newest"
    candidates = [p for p in stable if quartile_for_commit_step(int(commit_steps[p])) == wanted]
    fallback = False
    if not candidates:
        fallback = True
        edge = min(commit_steps[p] for p in stable) if wanted == "oldest" else max(commit_steps[p] for p in stable)
        candidates = [p for p in stable if commit_steps[p] == edge]
    chosen = min(candidates, key=lambda p: (source_rank_key(block_id, anchor_id, p, source_type), p))
    return chosen, fallback, wanted


def profile_recursion(
    source_position: int,
    future_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    profile: pd.DataFrame,
    cell_id: str,
    anchor_target_rho: float,
    statistic: str,
) -> dict[str, Any]:
    ordered = sorted(map(int, future_positions), key=lambda p: (int(commit_steps[p]), p))
    values: dict[int, float] = {int(source_position): 1.0}
    caps: dict[int, bool] = {int(source_position): True}
    lookups: list[dict[str, Any]] = []
    for target in ordered:
        tr, tc = divmod(target, 16)
        total = 0.0
        for source, source_value in list(values.items()):
            if source == source_position:
                source_step = int(commit_steps[source])
            else:
                source_step = int(commit_steps[source])
            if source_step >= int(commit_steps[target]):
                continue
            sr, sc = divmod(source, 16)
            lookup = {
                "cell_id": cell_id,
                "anchor_target_rho": float(anchor_target_rho),
                "source_quartile": quartile_for_commit_step(source_step),
                "step_offset_bin": step_offset_bin(int(commit_steps[target]) - source_step),
                "distance_bin": distance_bin(abs(tr - sr) + abs(tc - sc)),
            }
            influence, level, count, missing = lookup_profile(lookup, profile, statistic)
            total += float(influence) * float(source_value)
            lookups.append({
                "source_position": source, "target_position": target,
                "source_commit_step": source_step, "target_commit_step": int(commit_steps[target]),
                **lookup, "profile_value": float(influence), "fallback_level": level,
                "calibration_count": int(count), "missing_stratum": bool(missing),
            })
        values[target] = min(1.0, total)
        caps[target] = total >= 1.0 - 1e-15
    descendant_sum = sum(values[p] for p in ordered)
    return {
        "B_seed": 1.0 / N,
        "B_desc": descendant_sum / N,
        "B_total": (1.0 + descendant_sum) / N,
        "vector": values,
        "caps": caps,
        "lookups": lookups,
        "cap_rate": float(np.mean([caps[p] for p in ordered])) if ordered else 0.0,
        "missing_lookup_count": sum(int(row["missing_stratum"]) for row in lookups),
    }


def profile_recursion_independent(
    source_position: int,
    future_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    profile: pd.DataFrame,
    cell_id: str,
    anchor_target_rho: float,
    statistic: str,
) -> dict[str, Any]:
    """Second implementation used only to audit the load-bearing recursion.

    This deliberately uses dense arrays and an index loop, rather than the
    dictionary/list implementation in :func:`profile_recursion`.
    """

    ordered = np.asarray(
        sorted(map(int, future_positions), key=lambda p: (int(commit_steps[p]), p)),
        dtype=np.int64,
    )
    positions = np.concatenate((np.asarray([int(source_position)]), ordered))
    weights = np.zeros(len(positions), dtype=np.float64)
    weights[0] = 1.0
    for target_index in range(1, len(positions)):
        target = int(positions[target_index])
        target_step = int(commit_steps[target])
        tr, tc = divmod(target, 16)
        accumulated = 0.0
        for source_index in range(target_index):
            source = int(positions[source_index])
            source_step = int(commit_steps[source])
            if source_step >= target_step:
                continue
            sr, sc = divmod(source, 16)
            query = {
                "cell_id": cell_id,
                "anchor_target_rho": float(anchor_target_rho),
                "source_quartile": quartile_for_commit_step(source_step),
                "step_offset_bin": step_offset_bin(target_step - source_step),
                "distance_bin": distance_bin(abs(tr - sr) + abs(tc - sc)),
            }
            influence, _, _, _ = lookup_profile(query, profile, statistic)
            accumulated += float(influence) * float(weights[source_index])
        weights[target_index] = min(1.0, accumulated)
    descendant = float(weights[1:].sum())
    return {
        "B_seed": 1.0 / N,
        "B_desc": descendant / N,
        "B_total": (1.0 + descendant) / N,
    }


def paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    resamples: int,
    seed_key: str,
) -> dict[str, float]:
    ids = sorted(set(left).intersection(right))
    differences = np.array([float(left[i]) - float(right[i]) for i in ids], dtype=np.float64)
    seed = int.from_bytes(semantic_digest("FDCA-P1L-BOOTSTRAP", seed_key)[:8], "big")
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, (resamples, len(differences)), replace=True).mean(axis=1)
    return {
        "block_count": len(ids), "difference_mean": float(differences.mean()),
        "difference_median": float(np.median(differences)),
        "ci95_lower": float(np.quantile(draws, 0.025, method="higher")),
        "ci95_upper": float(np.quantile(draws, 0.975, method="higher")),
        "one_sided_u95": float(np.quantile(draws, 0.95, method="higher")),
        "semantic_seed": int(seed),
    }


def summarize_terminal(terminal: pd.DataFrame) -> pd.DataFrame:
    shocks = terminal[terminal.control_type.eq("single_shock")]
    return shocks.groupby(["cell_id", "anchor_target_rho", "source_type"], as_index=False).agg(
        rows=("D_total", "size"), blocks=("block_id", "nunique"),
        D_total_mean=("D_total", "mean"), D_total_median=("D_total", "median"),
        D_desc_total_mean=("D_desc_total", "mean"), D_desc_rate_mean=("D_desc_rate", "mean"),
        extinction_probability=("propagation_extinction", "mean"),
        first_descendant_step_mean=("first_descendant_step", "mean"),
        max_radius_mean=("maximum_manhattan_radius", "mean"),
        connected_components_mean=("connected_component_count", "mean"),
        descendants_within_5=("descendants_within_distance_5", "sum"),
        descendants_total=("descendant_count", "sum"),
        B_combined_q95_mean=("B_combined_q95", "mean"),
        B_combined_max_mean=("B_combined_max", "mean"),
        max_slack_mean=("max_prediction_slack", "mean"),
        max_cap_rate_mean=("max_cap_rate", "mean"),
    )


def evaluate_science(terminal: pd.DataFrame, resamples: int = 20_000) -> dict[str, Any]:
    shocks = terminal[(terminal.control_type == "single_shock") & (terminal.cell_id == "W8_L1.00")]
    block = shocks.groupby(["block_id", "anchor_target_rho"], as_index=False).agg(
        D_desc_rate=("D_desc_rate", "mean"), D_total=("D_total", "mean"),
        B_combined_max=("B_combined_max", "mean"), max_prediction_slack=("max_prediction_slack", "mean"),
    )
    maps = {rho: dict(zip(g.block_id, g.D_desc_rate)) for rho, g in block.groupby("anchor_target_rho")}
    h1 = paired_bootstrap(maps[0.5], maps[0.3], resamples, "H1-rho0.5-minus-rho0.3")
    h1["ratio_of_block_means"] = float(block[block.anchor_target_rho.eq(0.5)].D_desc_rate.mean() / max(block[block.anchor_target_rho.eq(0.3)].D_desc_rate.mean(), 1e-15))
    h1["pass"] = bool(h1["ci95_lower"] > 0.0 and h1["ratio_of_block_means"] >= 1.5)
    h2 = []
    for rho in [0.1, 0.3, 0.5]:
        values = block[block.anchor_target_rho.eq(rho)].set_index("block_id").max_prediction_slack.to_dict()
        zero = {key: 0.0 for key in values}
        audit = paired_bootstrap(values, zero, resamples, f"H2-rho{rho:.1f}-slack")
        audit.update({"rho": rho, "point_estimate": float(np.mean(list(values.values())))})
        audit["pass"] = bool(audit["point_estimate"] <= 0.0 and audit["one_sided_u95"] <= 0.01)
        h2.append(audit)
    by_rho = block.groupby("anchor_target_rho", as_index=False)[["B_combined_max", "D_total"]].mean().sort_values("anchor_target_rho")
    correlation = float(spearmanr(by_rho.B_combined_max, by_rho.D_total).statistic)
    h3 = {"spearman": correlation, "pass": bool(correlation >= 0.8), "regime_points": by_rho.to_dict("records")}
    source = shocks.groupby(["block_id", "source_type"], as_index=False).D_desc_rate.mean()
    oldest = source[source.source_type.eq("oldest_source")].set_index("block_id").D_desc_rate.to_dict()
    newest = source[source.source_type.eq("newest_source")].set_index("block_id").D_desc_rate.to_dict()
    h4 = paired_bootstrap(oldest, newest, resamples, "H4-oldest-minus-newest")
    h4["direction_pass"] = bool(h4["difference_mean"] > 0.0)
    nonzero_rho05 = float((shocks[shocks.anchor_target_rho.eq(0.5)].descendant_count > 0).mean())
    nonzero_rho03_05 = float((shocks[shocks.anchor_target_rho.isin([0.3, 0.5])].descendant_count > 0).mean())
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "nonzero_fraction_rho0.5": nonzero_rho05, "nonzero_fraction_rho0.3_0.5": nonzero_rho03_05}
