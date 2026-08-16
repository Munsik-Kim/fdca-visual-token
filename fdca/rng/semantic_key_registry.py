"""Model-free semantic-key registry for future event-addressed experiments.

The registry never changes the key-to-uniform mapping. It only normalizes,
classifies, records, and rejects unexpected repeated uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping


AllowReuse = Callable[[Mapping[str, Any], int], str | None]


def normalize_key(parts: Mapping[str, Any]) -> str:
    return json.dumps(dict(parts), sort_keys=True, separators=(",", ":"))


def key_hash(parts: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalize_key(parts).encode("utf-8")).hexdigest()


@dataclass
class SemanticKeyRegistry:
    allow_reuse: AllowReuse | None = None
    counts: dict[str, int] = field(default_factory=dict)
    normalized: dict[str, str] = field(default_factory=dict)
    classifications: dict[str, str] = field(default_factory=dict)

    def register(self, parts: Mapping[str, Any]) -> str:
        normalized = normalize_key(parts)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        previous = self.counts.get(digest, 0)
        if previous and self.normalized[digest] != normalized:
            raise RuntimeError("semantic-key SHA-256 collision")
        new_count = previous + 1
        classification = "UNIQUE"
        if previous:
            classification = self.allow_reuse(parts, new_count) if self.allow_reuse else None
            if not classification:
                raise RuntimeError(f"UNCLASSIFIED_DUPLICATE: {normalized}")
        self.counts[digest] = new_count
        self.normalized[digest] = normalized
        self.classifications[digest] = classification
        return digest

    @staticmethod
    def uniform(parts: Mapping[str, Any]) -> float:
        digest = bytes.fromhex(key_hash(parts))
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "key_hash": digest,
                "normalized_key": self.normalized[digest],
                "use_count": count,
                "classification": self.classifications[digest],
            }
            for digest, count in sorted(self.counts.items())
        ]
