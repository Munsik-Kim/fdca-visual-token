"""Model-free P2N topology, decomposition, profile, and inference helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fdca.analysis.p05_influence import distance_bin, quartile_for_commit_step, step_offset_bin
from fdca.analysis.p05r_scalar import lookup_profile
from fdca.analysis.p1l_single_shock import connected_components, mismatch_vector_hex


N = 256


def semantic_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"FDCA-P2N|{label}".encode()).digest()[:8], "big")


def seed_topology(positions: Sequence[int]) -> dict[str, Any]:
    points = sorted(map(int, positions))
    if not points:
        return {"connected_components": 0, "maximum_pairwise_manhattan": 0, "mean_pairwise_manhattan": 0.0, "bounding_box_area": 0}
    coords = [divmod(p, 16) for p in points]
    distances = [abs(a[0]-b[0]) + abs(a[1]-b[1]) for i, a in enumerate(coords) for b in coords[i+1:]]
    rows, cols = zip(*coords)
    return {
        "connected_components": connected_components(points),
        "maximum_pairwise_manhattan": max(distances, default=0),
        "mean_pairwise_manhattan": float(np.mean(distances)) if distances else 0.0,
        "bounding_box_area": (max(rows)-min(rows)+1) * (max(cols)-min(cols)+1),
    }


def descendant_spatial(seed_positions: Sequence[int], descendants: Sequence[int]) -> dict[str, Any]:
    seeds, desc = list(map(int, seed_positions)), list(map(int, descendants))
    distances = []
    for target in desc:
        tr, tc = divmod(target, 16)
        distances.append(min(abs(tr-divmod(s,16)[0]) + abs(tc-divmod(s,16)[1]) for s in seeds))
    return {
        "descendant_connected_components": connected_components([*seeds, *desc]) if seeds else 0,
        "maximum_seed_to_descendant_radius": max(distances, default=0),
        "mean_seed_to_descendant_distance": float(np.mean(distances)) if distances else np.nan,
        "median_seed_to_descendant_distance": float(np.median(distances)) if distances else np.nan,
    }


def paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float], label: str, resamples: int = 20_000) -> dict[str, Any]:
    ids = sorted(set(left).intersection(right))
    diffs = np.array([float(left[i])-float(right[i]) for i in ids], dtype=np.float64)
    rng = np.random.default_rng(semantic_seed(label))
    draws = rng.choice(diffs, (resamples, len(diffs)), replace=True).mean(axis=1)
    lmean, rmean = float(np.mean(list(left.values()))), float(np.mean(list(right.values())))
    return {
        "block_count": len(ids), "difference_mean": float(diffs.mean()),
        "ci95_lower": float(np.quantile(draws, .025, method="higher")),
        "ci95_upper": float(np.quantile(draws, .975, method="higher")),
        "left_mean": lmean, "right_mean": rmean,
        "ratio_of_aggregate_means": lmean/rmean if rmean else float("inf"),
        "semantic_seed": semantic_seed(label),
    }


def one_sample_bootstrap(values: Mapping[str, float], label: str, resamples: int = 20_000) -> dict[str, Any]:
    array = np.array([float(values[k]) for k in sorted(values)], dtype=np.float64)
    rng = np.random.default_rng(semantic_seed(label))
    draws = rng.choice(array, (resamples, len(array)), replace=True).mean(axis=1)
    return {"block_count": len(array), "mean": float(array.mean()), "ci95_lower": float(np.quantile(draws,.025,method="higher")), "ci95_upper": float(np.quantile(draws,.975,method="higher"))}


def profile_recursion_multi(seed_positions: Sequence[int], future_positions: Sequence[int], commit_steps: Mapping[int,int], profile: pd.DataFrame, cell_id: str, rho: float, statistic: str) -> dict[str, Any]:
    seeds = sorted(map(int, seed_positions))
    ordered = sorted(map(int, future_positions), key=lambda p:(int(commit_steps[p]),p))
    values: dict[int,float] = {p:1.0 for p in seeds}
    caps = 0
    missing = 0
    lookups = 0
    for target in ordered:
        tr,tc = divmod(target,16)
        total = 0.0
        for source, weight in list(values.items()):
            ss,ts=int(commit_steps[source]),int(commit_steps[target])
            if ss >= ts: continue
            sr,sc=divmod(source,16)
            query={"cell_id":cell_id,"anchor_target_rho":float(rho),"source_quartile":quartile_for_commit_step(ss),"step_offset_bin":step_offset_bin(ts-ss),"distance_bin":distance_bin(abs(tr-sr)+abs(tc-sc))}
            influence,_,_,was_missing=lookup_profile(query,profile,statistic)
            total += float(influence)*weight
            lookups += 1
            missing += int(was_missing)
        values[target]=min(1.0,total)
        caps += int(total >= 1.0-1e-15)
    return {"predicted_seed":len(seeds)/N,"predicted_desc":sum(values[p] for p in ordered)/N,"predicted_total":sum(values.values())/N,"cap_rate":caps/max(len(ordered),1),"missing_lookups":missing,"lookup_count":lookups,"vector":values}


def decompose(terminal: pd.DataFrame, seed_law: pd.DataFrame) -> pd.DataFrame:
    conditional = terminal.groupby(["block_id","cell_id","anchor_target_rho"],as_index=False).agg(
        retained_replicates=("replicate_id","nunique"), D_seed_cond=("D_seed","mean"),
        D_desc_cond=("D_desc_total","mean"), D_desc_rate_cond=("D_desc_rate","mean"),
        D_total_cond=("D_total","mean"), A_seed_cond=("A_seed","mean"),
        A_desc_per_seed_cond=("A_desc_per_seed","mean"), extinction_cond=("propagation_extinction","mean"),
    )
    law=seed_law[["block_id","cell_id","anchor_target_rho","split_incidence","expected_seed_count","single_seed_probability","multi_seed_probability"]]
    out=conditional.merge(law,on=["block_id","cell_id","anchor_target_rho"],validate="one_to_one")
    out["E_total_uncond"]=out.split_incidence*out.D_total_cond
    out["E_seed_uncond"]=out.split_incidence*out.D_seed_cond
    out["E_desc_uncond"]=out.split_incidence*out.D_desc_cond
    out["decomposition_error"]=out.E_total_uncond-out.E_seed_uncond-out.E_desc_uncond
    return out


def evaluate_hypotheses(decomposition: pd.DataFrame, terminal: pd.DataFrame, matched: pd.DataFrame) -> dict[str,Any]:
    w8=decomposition[decomposition.cell_id.eq("W8_L1.00")]
    def maps(frame:pd.DataFrame, rho:float, col:str)->dict[str,float]:
        return frame[frame.anchor_target_rho.eq(rho)].set_index("block_id")[col].astype(float).to_dict()
    h1=paired_bootstrap(maps(w8,.5,"D_desc_rate_cond"),maps(w8,.3,"D_desc_rate_cond"),"H1")
    h1["pass"]=h1["ci95_lower"]>0 and h1["ratio_of_aggregate_means"]>=1.5
    h2=paired_bootstrap(maps(w8,.5,"E_total_uncond"),maps(w8,.3,"E_total_uncond"),"H2")
    h2["pass"]=h2["ci95_lower"]>0 and h2["ratio_of_aggregate_means"]>=1.5
    t05=terminal[(terminal.cell_id.eq("W8_L1.00")) & terminal.anchor_target_rho.eq(.5)]
    blockmeans=t05.groupby("block_id").D_desc_rate.mean().to_dict()
    h3=one_sample_bootstrap(blockmeans,"H3")
    h3["descendant_positive_fraction"]=float(t05.descendant_count.gt(0).mean())
    h3["pass"]=h3["descendant_positive_fraction"]>=.2 and h3["mean"]>0 and h3["ci95_lower"]>0
    m=matched.groupby(["block_id","anchor_target_rho"],as_index=False).D_desc_rate.mean()
    h4=paired_bootstrap(maps(m,.5,"D_desc_rate"),maps(m,.3,"D_desc_rate"),"H4")
    h4["pass"]=h4["ci95_lower"]>0 and h4["ratio_of_aggregate_means"]>=1.5
    return {"H1":h1,"H2":h2,"H3":h3,"H4":h4}
