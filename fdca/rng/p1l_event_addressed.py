"""Semantic event-addressed RNG for P1L conditional coupling.

The baseline proposal key deliberately has no source/shock component. Shock
specific acceptance and residual draws carry a shock identifier in draw_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


FIELDS = (
    "experiment_id", "model_cell", "block_id", "anchor_id", "replicate_id",
    "step", "position", "event_kind", "draw_id",
)


@dataclass(frozen=True)
class EventKey:
    experiment_id: str
    model_cell: str
    block_id: str
    anchor_id: str
    replicate_id: int
    step: int
    position: int
    event_kind: str
    draw_id: str

    def payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FIELDS}

    def canonical(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def uniform(self) -> float:
        integer = int.from_bytes(bytes.fromhex(self.digest())[:8], "big")
        return integer / float(1 << 64)


def baseline_key(
    experiment_id: str,
    model_cell: str,
    block_id: str,
    anchor_id: str,
    replicate_id: int,
    step: int,
    position: int,
) -> EventKey:
    return EventKey(
        experiment_id, model_cell, block_id, anchor_id, replicate_id, step,
        position, "proposal_x", "baseline",
    )


def shock_key(
    experiment_id: str,
    model_cell: str,
    block_id: str,
    anchor_id: str,
    replicate_id: int,
    step: int,
    position: int,
    event_kind: str,
    shock_id: str,
) -> EventKey:
    if event_kind not in {"proposal_accept", "proposal_residual"}:
        raise ValueError(event_kind)
    return EventKey(
        experiment_id, model_cell, block_id, anchor_id, replicate_id, step,
        position, event_kind, shock_id,
    )


@dataclass
class EventRegistry:
    """Register physically generated random events and reject duplicate keys."""

    digests: set[str] = field(default_factory=set)
    duplicate_count: int = 0
    event_count: int = 0

    def uniform(self, key: EventKey) -> float:
        digest = key.digest()
        if digest in self.digests:
            self.duplicate_count += 1
            raise RuntimeError(f"duplicate semantic RNG key: {key.canonical()}")
        self.digests.add(digest)
        self.event_count += 1
        return key.uniform()
