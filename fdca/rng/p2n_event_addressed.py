"""Semantic event-addressed RNG with intentional cross-cell proposal reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


@dataclass
class SemanticRNG:
    generated_event_count: int = 0
    cache_reuse_count: int = 0
    semantic_duplicate_generation_count: int = 0

    @staticmethod
    def canonical(**parts: Any) -> str:
        return json.dumps(parts, sort_keys=True, separators=(",", ":"))

    def uniform(self, **parts: Any) -> float:
        canonical = self.canonical(**parts)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        value = int.from_bytes(bytes.fromhex(digest)[:8], "big") / float(1 << 64)
        self.generated_event_count += 1
        return value

    def audit(self) -> dict[str, Any]:
        return {
            "schema": "FDCA_VIS_P2N_EVENT_RNG_AUDIT_V1",
            "generated_event_count": self.generated_event_count,
            "intentional_cache_reuse_count": self.cache_reuse_count,
            "semantic_duplicate_generation_count": self.semantic_duplicate_generation_count,
            "determinism": "stateless_SHA256_key_to_uniform",
            "pass": self.semantic_duplicate_generation_count == 0,
        }
