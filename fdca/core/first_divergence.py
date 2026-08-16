"""First-divergence summaries for exact coupled paths."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from fdca.synthetic.coupled_paths import CoupledPath


def first_divergence_distribution(paths: Iterable[CoupledPath]) -> dict[str, float]:
    distribution: defaultdict[str, float] = defaultdict(float)
    for path in paths:
        key = "dagger" if path.first_divergence_round is None else str(path.first_divergence_round)
        distribution[key] += path.probability
    return dict(sorted(distribution.items()))


def first_seed_distribution(paths: Iterable[CoupledPath]) -> dict[str, float]:
    distribution: defaultdict[str, float] = defaultdict(float)
    for path in paths:
        if path.first_divergence_round is None:
            key = "dagger"
        else:
            positions = ",".join(str(position) for position in sorted(path.first_seed))
            key = f"round={path.first_divergence_round};seed={{{positions}}}"
        distribution[key] += path.probability
    return dict(sorted(distribution.items()))
