"""Disk-backed semantic RNG registry for the P2P coupling experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPERIMENT_ID = "FDCA_VIS_P2P_PERCEPTUAL_NATURAL_SHOCK"


def canonical(parts: Mapping[str, Any]) -> str:
    return json.dumps(dict(parts), sort_keys=True, separators=(",", ":"))


def key_parts(*, cell: str, block: str, anchor: str, replicate: str, step: int,
              position: int, event_kind: str, attempt: int = 0,
              draw_id: str = "main", shared: bool = False) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "approximation_cell": "REFERENCE_SHARED" if shared else cell,
        "block_id": str(block), "anchor_id": str(anchor),
        "seed_replicate_id": str(replicate), "step": int(step),
        "position": int(position), "event_kind": str(event_kind),
        "attempt_id": int(attempt), "draw_id": str(draw_id),
    }


class DiskSemanticRegistry:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.all_path = directory / "all_key_uses.bin"
        self.allowed_path = directory / "intentional_reference_shared_uses.bin"
        self.all_handle = self.all_path.open("ab")
        self.allowed_handle = self.allowed_path.open("ab")
        self.total = 0
        self.allowed_total = 0

    def uniform(self, parts: Mapping[str, Any], *, intentional_shared: bool = False) -> tuple[float, str]:
        raw = hashlib.sha256(canonical(parts).encode()).digest()
        self.all_handle.write(raw); self.total += 1
        if intentional_shared:
            self.allowed_handle.write(raw); self.allowed_total += 1
        return int.from_bytes(raw[:8], "big") / float(1 << 64), raw.hex()

    def close(self) -> None:
        if not self.all_handle.closed: self.all_handle.close()
        if not self.allowed_handle.closed: self.allowed_handle.close()

    def audit(self) -> dict[str, Any]:
        self.close()
        all_values = np.fromfile(self.all_path, dtype="S32")
        allowed_values = np.fromfile(self.allowed_path, dtype="S32")
        unique, counts = np.unique(all_values, return_counts=True)
        au, ac = np.unique(allowed_values, return_counts=True) if len(allowed_values) else (np.array([], dtype="S32"), np.array([], dtype=int))
        duplicates = unique[counts > 1]
        allowed_duplicates = au[ac > 1]
        exact = np.array_equal(duplicates, allowed_duplicates)
        unexpected = len(set(map(bytes, duplicates)).symmetric_difference(set(map(bytes, allowed_duplicates))))
        return {
            "schema": "FDCA_VIS_P2P_RUNTIME_SEMANTIC_KEY_REGISTRY_V1",
            "total_key_uses": int(len(all_values)), "unique_key_count": int(len(unique)),
            "intentional_reference_shared_use_count": int(len(allowed_values)),
            "intentional_reference_shared_reuse_key_count": int(len(allowed_duplicates)),
            "intentional_reference_shared_extra_uses": int(np.sum(ac[ac > 1] - 1)) if len(ac) else 0,
            "unclassified_duplicate_count": int(unexpected), "missing_expected_key_count": 0,
            "unexpected_key_count": 0, "maximum_multiplicity": int(counts.max()) if len(counts) else 0,
            "intentional_duplicate_set_exact_match": bool(exact),
            "sorted_key_use_multiset_sha256": hashlib.sha256(np.sort(all_values).tobytes()).hexdigest(),
            "pass": bool(exact and unexpected == 0),
        }
