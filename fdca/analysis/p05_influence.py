"""Deterministic primitives for the P05 influence--innovation microaudit.

The functions in this module deliberately distinguish exact local arithmetic
from finite-panel statistical aggregation. No function here implements a
closed-loop reference/approximate coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


MASK_VALUE = 16_384
CODEBOOK_SIZE = 16_384
GRID_SIZE = 16
GRID_POSITIONS = GRID_SIZE * GRID_SIZE


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def semantic_digest(*parts: Any) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()


def generate_class_seed_blocks(count: int = 32) -> list[dict[str, Any]]:
    """Generate unique class--seed pairs without consulting model output."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    counter = 0
    while len(rows) < count:
        digest = semantic_digest("FDCA-P05-BLOCK", f"{counter:04d}")
        class_id = int.from_bytes(digest[:8], "big") % 1000
        seed = int.from_bytes(digest[8:16], "big", signed=False)
        pair = (class_id, seed)
        counter += 1
        if pair in seen:
            continue
        seen.add(pair)
        index = len(rows)
        rows.append(
            {
                "block_index": index,
                "block_id": f"B{index:02d}",
                "class_id": class_id,
                "generation_seed_uint64": seed,
                "generation_seed_torch": seed % (2**63 - 1),
                "full_split": "calibration" if index < 24 else "holdout",
                "core_split": (
                    "calibration" if index < 12 else "holdout" if index < 16 else "unused"
                ),
                "semantic_digest": digest.hex(),
            }
        )
    return rows


def categorical_tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Coordinatewise categorical TV with FP64 accumulation."""

    return 0.5 * torch.sum(torch.abs(p.double() - q.double()), dim=-1)


def stable_product_split_probability(tv: Sequence[float]) -> float:
    values = np.clip(np.asarray(tv, dtype=np.float64), 0.0, 1.0)
    return float(-np.expm1(np.log1p(-values).sum())) if len(values) else 0.0


def rtn_per_output_channel(weight: torch.Tensor, bit: int) -> torch.Tensor:
    """Signed symmetric per-output-channel round-to-nearest reconstruction."""

    if weight.ndim != 2:
        raise ValueError("P05 RTN is defined only for Linear.weight matrices")
    qmax = 2 ** (bit - 1) - 1
    source = weight.detach().float().cpu()
    max_abs = source.abs().amax(dim=1, keepdim=True)
    scale = max_abs / qmax
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    integer = torch.round(source / safe).clamp(-qmax, qmax)
    restored = integer * safe
    restored = torch.where(max_abs == 0, torch.zeros_like(restored), restored)
    return restored


@dataclass
class QuantizedWeight:
    name: str
    parameter: nn.Parameter
    reference_cpu: torch.Tensor
    rtn4_cpu: torch.Tensor
    rtn8_cpu: torch.Tensor


class LinearWeightHomotopy:
    """In-place reversible homotopy over the preregistered Linear set."""

    def __init__(self, model: nn.Module):
        embedding_parameter_ids = {
            id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)
        }
        seen: set[int] = set()
        self.weights: list[QuantizedWeight] = []
        self.excluded_tied_linear: list[str] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            parameter = module.weight
            if id(parameter) in embedding_parameter_ids:
                self.excluded_tied_linear.append(f"{name}.weight")
                continue
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            reference = parameter.detach().float().cpu().clone()
            self.weights.append(
                QuantizedWeight(
                    name=f"{name}.weight",
                    parameter=parameter,
                    reference_cpu=reference,
                    rtn4_cpu=rtn_per_output_channel(reference, 4),
                    rtn8_cpu=rtn_per_output_channel(reference, 8),
                )
            )
        if not self.weights:
            raise ValueError("no eligible Linear weights discovered")

    @torch.no_grad()
    def apply(self, bit: int, lam: float) -> None:
        for item in self.weights:
            quantized = item.rtn8_cpu if bit == 8 else item.rtn4_cpu
            value = item.reference_cpu + float(lam) * (quantized - item.reference_cpu)
            item.parameter.copy_(value.to(item.parameter.device, dtype=item.parameter.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for item in self.weights:
            item.parameter.copy_(
                item.reference_cpu.to(item.parameter.device, dtype=item.parameter.dtype)
            )

    def reference_hashes(self) -> dict[str, str]:
        return {item.name: sha256_tensor(item.reference_cpu) for item in self.weights}

    def cell_hash(self, bit: int, lam: float) -> str:
        digest = hashlib.sha256()
        for item in self.weights:
            quantized = item.rtn8_cpu if bit == 8 else item.rtn4_cpu
            value = item.reference_cpu + float(lam) * (quantized - item.reference_cpu)
            digest.update(item.name.encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def ledger(self) -> pd.DataFrame:
        rows = []
        for item in self.weights:
            for bit, rtn in ((8, item.rtn8_cpu), (4, item.rtn4_cpu)):
                rows.append(
                    {
                        "module_weight": item.name,
                        "bit": bit,
                        "shape": "x".join(map(str, item.reference_cpu.shape)),
                        "numel": item.reference_cpu.numel(),
                        "reference_sha256": sha256_tensor(item.reference_cpu),
                        "rtn_sha256": sha256_tensor(rtn),
                        "max_abs_error": float((rtn - item.reference_cpu).abs().max()),
                        "zero_rows": int((item.reference_cpu.abs().amax(dim=1) == 0).sum()),
                    }
                )
        return pd.DataFrame(rows)


def quartile_for_commit_step(step: int) -> str:
    if step < 8:
        return "oldest"
    if step < 16:
        return "early_middle"
    if step < 24:
        return "late_middle"
    return "newest"


def distance_bin(distance: int) -> str:
    if distance <= 2:
        return "D01_02"
    if distance <= 5:
        return "D03_05"
    if distance <= 8:
        return "D06_08"
    return "D09_30"


def step_offset_bin(offset: int) -> str:
    if offset <= 4:
        return "S01_04"
    if offset <= 8:
        return "S05_08"
    if offset <= 16:
        return "S09_16"
    return "S17_32"


def merged_distance_bin(bin_id: str) -> str:
    return "D01_05" if bin_id in {"D01_02", "D03_05"} else "D06_30"


def merged_step_bin(bin_id: str) -> str:
    return "S01_08" if bin_id in {"S01_04", "S05_08"} else "S09_32"


def choose_sources(
    stable_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    block_id: str,
    anchor_id: str,
    maximum: int,
) -> list[int]:
    """Deterministic quartile-stratified source selection."""

    names = ["oldest", "early_middle", "late_middle", "newest"]
    ranked: dict[str, list[int]] = {q: [] for q in names}
    for position in stable_positions:
        ranked[quartile_for_commit_step(int(commit_steps[position]))].append(int(position))
    for quartile in ranked:
        ranked[quartile].sort(
            key=lambda p: semantic_digest(block_id, anchor_id, p, "FDCA-P05-SOURCE")
        )
    chosen: list[int] = []
    base = 2 if maximum >= 8 else 1
    for quartile in names:
        chosen.extend(ranked[quartile][:base])
    chosen_set = set(chosen)
    remaining = [p for quartile in names for p in ranked[quartile] if p not in chosen_set]
    remaining.sort(key=lambda p: semantic_digest(block_id, anchor_id, p, "FDCA-P05-REDISTRIBUTE"))
    chosen.extend(remaining[: max(0, maximum - len(chosen))])
    return chosen[:maximum]


PROFILE_COLUMNS = [
    "cell_id",
    "anchor_target_rho",
    "source_quartile",
    "step_offset_bin",
    "distance_bin",
]


def _q95(values: pd.Series) -> float:
    return float(np.quantile(values.to_numpy(dtype=float), 0.95, method="higher"))


def _summary_rows(frame: pd.DataFrame, group_cols: list[str], level: str) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)["alpha_hat"]
    result = grouped.agg(median="median", maximum="max", sample_count="size").reset_index()
    result["q90"] = grouped.quantile(0.90, interpolation="higher").to_numpy()
    result["q95"] = grouped.apply(_q95).to_numpy()
    result["fallback_level"] = level
    return result


def fit_calibration_profile(frame: pd.DataFrame, minimum_count: int = 20) -> pd.DataFrame:
    """Fit all preregistered fallback levels using calibration rows only."""

    if set(frame["split"]) != {"calibration"}:
        raise ValueError("profile fit received non-calibration rows")
    parts: list[pd.DataFrame] = []
    exact = _summary_rows(frame, PROFILE_COLUMNS, "exact")
    parts.append(exact[exact.sample_count >= minimum_count])
    drop_q_cols = [c for c in PROFILE_COLUMNS if c != "source_quartile"]
    drop_q = _summary_rows(frame, drop_q_cols, "drop_source_quartile")
    drop_q["source_quartile"] = "*"
    parts.append(drop_q[drop_q.sample_count >= minimum_count])
    merged_d = frame.copy()
    merged_d["distance_bin"] = merged_d.distance_bin.map(merged_distance_bin)
    md = _summary_rows(merged_d, drop_q_cols, "merge_adjacent_distance")
    md["source_quartile"] = "*"
    parts.append(md[md.sample_count >= minimum_count])
    merged_s = merged_d.copy()
    merged_s["step_offset_bin"] = merged_s.step_offset_bin.map(merged_step_bin)
    ms = _summary_rows(merged_s, drop_q_cols, "merge_adjacent_step_offset")
    ms["source_quartile"] = "*"
    parts.append(ms[ms.sample_count >= minimum_count])
    glob = _summary_rows(frame, ["cell_id", "anchor_target_rho"], "cell_anchor_global")
    glob["source_quartile"] = "*"
    glob["step_offset_bin"] = "*"
    glob["distance_bin"] = "*"
    parts.append(glob)
    profile = pd.concat(parts, ignore_index=True)
    return profile[[*PROFILE_COLUMNS, "median", "q90", "q95", "maximum", "sample_count", "fallback_level"]]


def lookup_profile_q95(row: Mapping[str, Any], profile: pd.DataFrame) -> tuple[float, str, int]:
    tests = [
        ("exact", row["source_quartile"], row["step_offset_bin"], row["distance_bin"]),
        ("drop_source_quartile", "*", row["step_offset_bin"], row["distance_bin"]),
        ("merge_adjacent_distance", "*", row["step_offset_bin"], merged_distance_bin(str(row["distance_bin"]))),
        ("merge_adjacent_step_offset", "*", merged_step_bin(str(row["step_offset_bin"])), merged_distance_bin(str(row["distance_bin"]))),
        ("cell_anchor_global", "*", "*", "*"),
    ]
    for level, quartile, step_bin, dist_bin in tests:
        match = profile[
            (profile.cell_id == row["cell_id"])
            & (profile.anchor_target_rho == row["anchor_target_rho"])
            & (profile.source_quartile == quartile)
            & (profile.step_offset_bin == step_bin)
            & (profile.distance_bin == dist_bin)
            & (profile.fallback_level == level)
        ]
        if len(match):
            item = match.iloc[0]
            return float(item.q95), level, int(item.sample_count)
    raise KeyError(f"no profile entry for {dict(row)}")


def compute_measured_envelope(
    future_positions: Sequence[int],
    commit_steps: Mapping[int, int],
    direct_b: Mapping[int, float],
    profile: pd.DataFrame,
    cell_id: str,
    anchor_target_rho: float,
) -> dict[str, Any]:
    """Compute time-ordered measured clipped and linear recursions."""

    ordered = sorted(map(int, future_positions), key=lambda p: (commit_steps[p], p))
    clipped: dict[int, float] = {}
    linear: dict[int, float] = {}
    edge_count = 0
    max_edge = 0.0
    for u in ordered:
        ur, uc = divmod(u, GRID_SIZE)
        csum = float(direct_b[u])
        lsum = float(direct_b[u])
        for v in ordered:
            if commit_steps[v] >= commit_steps[u]:
                continue
            vr, vc = divmod(v, GRID_SIZE)
            distance = abs(ur - vr) + abs(uc - vc)
            offset = int(commit_steps[u] - commit_steps[v])
            lookup = {
                "cell_id": cell_id,
                "anchor_target_rho": anchor_target_rho,
                "source_quartile": quartile_for_commit_step(commit_steps[v]),
                "step_offset_bin": step_offset_bin(offset),
                "distance_bin": distance_bin(distance),
            }
            influence, _, _ = lookup_profile_q95(lookup, profile)
            edge_count += 1
            max_edge = max(max_edge, influence)
            csum += influence * clipped[v]
            lsum += influence * linear[v]
        clipped[u] = min(1.0, csum)
        linear[u] = lsum
    b_stable = len(ordered) / GRID_POSITIONS
    b_clipped = sum(clipped.values()) / GRID_POSITIONS
    b_linear_raw = sum(linear.values()) / GRID_POSITIONS
    b_linear = sum(min(1.0, value) for value in linear.values()) / GRID_POSITIONS
    return {
        "B_stable": float(b_stable),
        "B_clipped": float(b_clipped),
        "B_linear": float(b_linear),
        "B_linear_raw": float(b_linear_raw),
        "B_combined": float(min(b_stable, b_clipped)),
        "edge_count": edge_count,
        "max_profile_edge": max_edge,
        "clipped_vector": clipped,
        "linear_vector": linear,
    }


def bootstrap_ratio_summary(
    envelope_rows: pd.DataFrame, cell_id: str, resamples: int, seed: int
) -> dict[str, Any]:
    frame = envelope_rows[(envelope_rows.split == "holdout") & (envelope_rows.cell_id == cell_id)]
    block = frame.groupby("block_id", as_index=False).agg(
        B_stable=("B_stable", "mean"), B_combined=("B_combined", "mean")
    )
    ratios = (block.B_combined / block.B_stable).to_numpy(dtype=float)
    rng = np.random.default_rng(seed ^ int.from_bytes(semantic_digest(cell_id)[:8], "big"))
    means = np.empty(resamples, dtype=float)
    medians = np.empty(resamples, dtype=float)
    for i in range(resamples):
        sample = rng.choice(ratios, len(ratios), replace=True)
        means[i] = np.mean(sample)
        medians[i] = np.median(sample)
    return {
        "cell_id": cell_id,
        "block_count": int(len(block)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio_bootstrap_u95": float(np.quantile(means, 0.95, method="higher")),
        "median_ratio_bootstrap_u95": float(np.quantile(medians, 0.95, method="higher")),
    }


def assert_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicates = frame.duplicated(list(columns), keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate primary key in {label}: {int(duplicates.sum())}")


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    values = frame[list(columns)].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())
