#!/usr/bin/env python3
"""Execute the frozen FDCA_VIS_T0_SYNTHETIC contract and stop at its gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import scipy
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fdca.core.consequence import evaluate_conditional_consequence
from fdca.core.dag_envelope import (
    clipped_envelope,
    finite_resolvent_matrix,
    inverse_resolvent_matrix,
    linear_envelope,
    recursion_rhs,
)
from fdca.core.first_divergence import first_divergence_distribution, first_seed_distribution
from fdca.core.influence import (
    exact_influence,
    hybrid_inequality_violations,
    nilpotence_residual,
    time_order_violations,
)
from fdca.metrics.transport import exact_wasserstein
from fdca.synthetic.coupled_paths import (
    CoupledPath,
    coordinate_mismatch_profile,
    enumerate_coupled_paths,
    joint_marginals,
    marginal_errors,
    normalized_weights,
    paired_cost,
    persistent_plan,
    probability_sum,
    single_shock_plan,
    stable_set_pathwise_violations,
    terminal_joint_law,
    weighted_hamming,
)
from fdca.synthetic.enumerate_laws import (
    enumerate_stage_laws,
    enumerate_terminal_law,
    law_normalization_error,
    transition_table,
)
from fdca.synthetic.models import (
    SyntheticDecoder,
    canonical_families,
    family_b_local_propagation,
    family_d_persistent_fresh_innovation,
    random_table_family,
)
from fdca.synthetic.state_space import enumerate_admissible_states, state_key


PROBABILITY_TOLERANCE = 1e-12
INEQUALITY_TOLERANCE = 1e-10
MATRIX_TOLERANCE = 1e-10
LP_TOLERANCE = 1e-9
RANDOM_SEEDS = (1729, 20260812, 314159)


def json_ready(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, frozenset | set):
        return sorted(value)
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return "UNBORN" if arguments == ("rev-parse", "HEAD") else result.stderr.strip()
    return result.stdout.strip()


def run_property_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Disable pytest's own temporary-file capture because the outer runner
    # already captures stdout. This keeps the nested audit robust in restricted
    # execution environments whose temporary capture file may be reclaimed.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--capture=no",
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"(\d+) passed", result.stdout)
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": int(match.group(1)) if match else 0,
        "output": result.stdout,
    }


def model_definition(model: SyntheticDecoder) -> dict[str, Any]:
    return {
        "name": model.name,
        "description": model.description,
        "n_positions": model.n_positions,
        "vocabulary": model.vocabulary,
        "mask": -1,
        "schedule": model.schedule,
        "context": model.context,
        "metadata": model.metadata,
        "transition_table": transition_table(model),
    }


def law_rows(law: Mapping[tuple[int, ...], float]) -> list[dict[str, Any]]:
    return [
        {"state": state_key(state), "probability": probability}
        for state, probability in sorted(law.items())
    ]


def terminal_joint_rows(paths: Iterable[CoupledPath]) -> list[dict[str, Any]]:
    return [
        {
            "x": state_key(x_state),
            "y": state_key(y_state),
            "probability": probability,
        }
        for (x_state, y_state), probability in sorted(terminal_joint_law(paths).items())
    ]


def path_row(model_name: str, coupling: str, plan_label: str, index: int, path: CoupledPath) -> dict[str, Any]:
    return {
        "model": model_name,
        "coupling": coupling,
        "plan": plan_label,
        "path_index": index,
        "probability": path.probability,
        "x_history": [state_key(state) for state in path.x_history],
        "y_history": [state_key(state) for state in path.y_history],
        "first_divergence_round": path.first_divergence_round,
        "first_seed": sorted(path.first_seed),
    }


def expected_stable_bound(
    model: SyntheticDecoder, paths: Iterable[CoupledPath], weights: Sequence[float]
) -> float:
    weight_vector = np.asarray(weights, dtype=np.float64)
    total = 0.0
    for path in paths:
        if path.first_divergence_round is None:
            continue
        unresolved = [
            position
            for batch in model.schedule[path.first_divergence_round :]
            for position in batch
        ]
        total += path.probability * float(weight_vector[unresolved].sum())
    return float(total)


def indexed_violations(lhs: np.ndarray, rhs: np.ndarray, tolerance: float) -> list[dict[str, float]]:
    return [
        {"position": int(index), "lhs": float(lhs[index]), "rhs": float(rhs[index])}
        for index in range(lhs.size)
        if lhs[index] > rhs[index] + tolerance
    ]


def all_finite(*values: Any) -> bool:
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            return False
    return True


def conditional_event_keys(paths: Iterable[CoupledPath]) -> tuple[tuple[int, frozenset[int]], ...]:
    return tuple(
        sorted(
            {
                (path.first_divergence_round, path.first_seed)
                for path in paths
                if path.first_divergence_round is not None
            },
            key=lambda item: (item[0], tuple(sorted(item[1]))),
        )
    )


def audit_model(model: SyntheticDecoder) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return compact metrics plus full raw enumerations for one finite model."""

    weights = normalized_weights(model.n_positions)
    reference = enumerate_terminal_law(model, "p")
    approximate = enumerate_terminal_law(model, "q")
    influence = exact_influence(model)
    clipped = clipped_envelope(model, influence.direct, influence.matrix)
    linear = linear_envelope(influence.matrix, influence.direct)
    inverse = inverse_resolvent_matrix(influence.matrix)
    finite = finite_resolvent_matrix(influence.matrix)
    inverse_series_error = float(np.max(np.abs(inverse - finite)))
    transport = exact_wasserstein(reference, approximate, weights)

    coupling_metrics: dict[str, Any] = {}
    raw_paths: dict[str, Any] = {}
    persistent_violations: list[dict[str, Any]] = []
    numerical_values: list[Any] = [
        influence.direct,
        influence.matrix,
        clipped,
        linear,
        transport.distance,
        transport.max_marginal_residual,
    ]

    for coupling in ("maximal", "crn"):
        plan = persistent_plan(model)
        paths = enumerate_coupled_paths(model, coupling, plan)
        left_error, right_error = marginal_errors(model, paths, plan)
        profile = coordinate_mismatch_profile(paths, model.n_positions)
        cost = paired_cost(paths, weights)
        stable_violations = stable_set_pathwise_violations(model, paths, weights)
        recursion_violations = (
            indexed_violations(
                profile,
                recursion_rhs(influence.direct, influence.matrix, profile),
                INEQUALITY_TOLERANCE,
            )
            if coupling == "maximal"
            else []
        )
        clipped_violations = (
            indexed_violations(profile, clipped, INEQUALITY_TOLERANCE)
            if coupling == "maximal"
            else []
        )
        linear_resolvent_violations = (
            indexed_violations(profile, linear, INEQUALITY_TOLERANCE)
            if coupling == "maximal"
            else []
        )
        wasserstein_violation = transport.distance > cost + LP_TOLERANCE
        coupling_metrics[coupling] = {
            "trajectory_cases": len(paths),
            "probability_sum": probability_sum(paths),
            "probability_normalization_error": abs(probability_sum(paths) - 1.0),
            "left_terminal_marginal_error": left_error,
            "right_terminal_marginal_error": right_error,
            "mismatch_profile": profile,
            "paired_cost": cost,
            "stable_set_expected_bound": expected_stable_bound(model, paths, weights),
            "stable_set_pathwise_violations": stable_violations,
            "recursion_violations": recursion_violations,
            "clipped_violations": clipped_violations,
            "linear_resolvent_violations": linear_resolvent_violations,
            "wasserstein_violation": wasserstein_violation,
            "first_divergence_distribution": first_divergence_distribution(paths),
            "first_seed_distribution": first_seed_distribution(paths),
        }
        raw_paths[coupling] = {
            "terminal_joint_law": terminal_joint_rows(paths),
            "paths": [path_row(model.name, coupling, plan.label, i, path) for i, path in enumerate(paths)],
        }
        numerical_values.extend([profile, cost, left_error, right_error])

        if coupling == "maximal":
            for split_round, seed in conditional_event_keys(paths):
                consequence = evaluate_conditional_consequence(
                    model,
                    paths,
                    influence.direct,
                    influence.matrix,
                    split_round,
                    seed,
                )
                violations = indexed_violations(
                    consequence.mismatch_profile,
                    consequence.persistent_envelope,
                    INEQUALITY_TOLERANCE,
                )
                if consequence.paired_cost > consequence.persistent_cost_bound + INEQUALITY_TOLERANCE:
                    violations.append(
                        {
                            "position": -1,
                            "lhs": consequence.paired_cost,
                            "rhs": consequence.persistent_cost_bound,
                        }
                    )
                if violations:
                    persistent_violations.append(
                        {
                            "split_round": split_round,
                            "seed": sorted(seed),
                            "violations": violations,
                        }
                    )

    state_space_cases = sum(
        len(enumerate_admissible_states(model, stage))
        for stage in range(len(model.schedule) + 1)
    )
    reference_stage_errors = [
        law_normalization_error(law) for law in enumerate_stage_laws(model, "p")
    ]
    approximate_stage_errors = [
        law_normalization_error(law) for law in enumerate_stage_laws(model, "q")
    ]
    clipped_linear_violations = indexed_violations(clipped, linear, INEQUALITY_TOLERANCE)
    metrics = {
        "name": model.name,
        "family": model.metadata.get("family", "unknown"),
        "n_positions": model.n_positions,
        "vocabulary_size": len(model.vocabulary),
        "rounds": len(model.schedule),
        "state_space_cases": state_space_cases,
        "conditional_state_cases": influence.state_cases,
        "neighbor_pair_cases": influence.neighbor_pair_cases,
        "terminal_support_reference": len(reference),
        "terminal_support_approximate": len(approximate),
        "direct_b_supremum": influence.direct,
        "influence_matrix": influence.matrix,
        "direct_witnesses": influence.direct_witnesses,
        "influence_witnesses": influence.influence_witnesses,
        "clipped_envelope": clipped,
        "linear_resolvent_envelope": linear,
        "clipped_cost_bound": float(weights @ clipped),
        "linear_cost_bound": float(weights @ linear),
        "inverse_series_max_error": inverse_series_error,
        "nilpotence_residual": nilpotence_residual(influence.matrix),
        "time_order_violations": time_order_violations(model, influence.matrix),
        "hybrid_lemma_violations": hybrid_inequality_violations(model, influence),
        "clipped_linear_violations": clipped_linear_violations,
        "persistent_conditional_cases": len(
            conditional_event_keys(enumerate_coupled_paths(model, "maximal"))
        ),
        "persistent_conditional_violations": persistent_violations,
        "terminal_law_normalization_error": {
            "reference_max": max(reference_stage_errors),
            "approximate_max": max(approximate_stage_errors),
        },
        "wasserstein": {
            "distance": transport.distance,
            "lp_success": transport.success,
            "lp_status": transport.status,
            "lp_message": transport.message,
            "max_marginal_residual": transport.max_marginal_residual,
            "variables": transport.variables,
        },
        "couplings": coupling_metrics,
        "all_numeric_finite": all_finite(*numerical_values),
    }
    raw = {
        "model_definition": model_definition(model),
        "terminal_laws": {
            "reference": law_rows(reference),
            "approximate": law_rows(approximate),
        },
        "coupled": raw_paths,
        "direct_innovation_table": influence.direct_table,
    }
    return metrics, raw


def aggregate_violations(metrics: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "time_order": sum(len(item["time_order_violations"]) for item in metrics),
        "hybrid_lemma": sum(len(item["hybrid_lemma_violations"]) for item in metrics),
        "stable_set": sum(
            len(coupling["stable_set_pathwise_violations"])
            for item in metrics
            for coupling in item["couplings"].values()
        ),
        "linear_recursion": sum(
            len(item["couplings"]["maximal"]["recursion_violations"]) for item in metrics
        ),
        "clipped_envelope": sum(
            len(item["couplings"]["maximal"]["clipped_violations"]) for item in metrics
        ),
        "resolvent_envelope": sum(
            len(item["couplings"]["maximal"]["linear_resolvent_violations"])
            for item in metrics
        ),
        "clipped_vs_linear": sum(len(item["clipped_linear_violations"]) for item in metrics),
        "persistent_conditional": sum(
            len(item["persistent_conditional_violations"]) for item in metrics
        ),
        "wasserstein_vs_coupling": sum(
            int(coupling["wasserstein_violation"])
            for item in metrics
            for coupling in item["couplings"].values()
        ),
    }


def single_shock_audit() -> dict[str, Any]:
    model = family_b_local_propagation()
    influence = exact_influence(model)
    plan = single_shock_plan(model, 0)
    paths = enumerate_coupled_paths(model, "maximal", plan)
    consequence = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    coordinate_violations = indexed_violations(
        consequence.mismatch_profile,
        consequence.seed_only_envelope,
        INEQUALITY_TOLERANCE,
    )
    cost_violation = consequence.paired_cost > consequence.seed_only_cost_bound + INEQUALITY_TOLERANCE
    return {
        "model": model.name,
        "plan": plan.label,
        "trajectory_cases": len(paths),
        "conditioning_event_mass": consequence.event_mass,
        "seed": sorted(consequence.seed),
        "conditional_mismatch_profile": consequence.mismatch_profile,
        "seed_only_envelope": consequence.seed_only_envelope,
        "conditional_paired_cost": consequence.paired_cost,
        "seed_only_cost_bound": consequence.seed_only_cost_bound,
        "coordinate_violations": coordinate_violations,
        "cost_violation": cost_violation,
        "passed": not coordinate_violations and not cost_violation,
        "paths": [path_row(model.name, "maximal", plan.label, i, path) for i, path in enumerate(paths)],
    }


def persistent_counterexample_audit() -> dict[str, Any]:
    model = family_d_persistent_fresh_innovation()
    influence = exact_influence(model)
    plan = persistent_plan(model)
    paths = enumerate_coupled_paths(model, "maximal", plan)
    consequence = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    seed_only_coordinate_failures = indexed_violations(
        consequence.mismatch_profile,
        consequence.seed_only_envelope,
        INEQUALITY_TOLERANCE,
    )
    seed_only_cost_gap = consequence.paired_cost - consequence.seed_only_cost_bound
    persistent_coordinate_violations = indexed_violations(
        consequence.mismatch_profile,
        consequence.persistent_envelope,
        INEQUALITY_TOLERANCE,
    )
    persistent_cost_violation = (
        consequence.paired_cost > consequence.persistent_cost_bound + INEQUALITY_TOLERANCE
    )
    found = (
        bool(seed_only_coordinate_failures)
        and seed_only_cost_gap > INEQUALITY_TOLERANCE
        and not persistent_coordinate_violations
        and not persistent_cost_violation
    )
    return {
        "model": model.name,
        "conditioning_event": {"first_divergence_round": 0, "seed": [0]},
        "conditioning_event_mass": consequence.event_mass,
        "direct_b_supremum": influence.direct,
        "influence_matrix": influence.matrix,
        "conditional_mismatch_profile": consequence.mismatch_profile,
        "seed_only_envelope": consequence.seed_only_envelope,
        "persistent_envelope_with_future_innovation": consequence.persistent_envelope,
        "conditional_paired_cost": consequence.paired_cost,
        "seed_only_cost_bound": consequence.seed_only_cost_bound,
        "persistent_cost_bound": consequence.persistent_cost_bound,
        "seed_only_cost_gap": seed_only_cost_gap,
        "seed_only_coordinate_failures": seed_only_coordinate_failures,
        "persistent_coordinate_violations": persistent_coordinate_violations,
        "persistent_cost_violation": persistent_cost_violation,
        "counterexample_found": found,
        "interpretation": (
            "M is zero, so the first-split seed cannot reach later coordinates. "
            "Nevertheless p and q differ at both future rounds, producing fresh "
            "mismatch probabilities 0.5 and 0.4."
        ),
    }


def frozen_input_audit() -> dict[str, Any]:
    expected = {
        "docs/frozen/FDCA_VIS_T0_SYNTHETIC_CODEX_PROMPT.txt": "e7f119d4b8250aca0deabf51bb465fd46df6ff6d86687e83d640a90af8b9db0c",
        "docs/frozen/FDCA_Technical_Note_v1.0.pdf": "6972c27f80f4b5d1ade2286e995b07063e0cf1a306d2b17a4096d246e7d1731c",
        "docs/frozen/FDCA_Research_Plan_v1.0.md": "b90a40d538c99a8d84a68c8cff773531575c9a2028d3900fda47ae738bb34f4e",
    }
    rows = []
    for relative, expected_digest in expected.items():
        path = PROJECT_ROOT / relative
        actual = sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected_digest,
                "actual_sha256": actual,
                "matches": actual == expected_digest,
            }
        )
    return {"files": rows, "all_match": all(row["matches"] for row in rows)}


def environment_audit() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pytest": pytest.__version__,
        "float_dtype": "float64",
        "transport_solver": "scipy.optimize.linprog(method='highs')",
        "gpu_used": False,
        "external_model_or_checkpoint_downloaded": False,
    }


def compact_family_table(metrics: Sequence[dict[str, Any]]) -> str:
    rows = [
        "| Family | W_d | D_max | Stable | Clipped | Linear | D_max-W_d |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        if item["family"] not in {"A", "B", "C", "D"}:
            continue
        wasserstein = item["wasserstein"]["distance"]
        maximal = item["couplings"]["maximal"]
        rows.append(
            "| {family} | {w:.12f} | {d:.12f} | {stable:.12f} | {clip:.12f} | {linear:.12f} | {gap:.12f} |".format(
                family=item["family"],
                w=wasserstein,
                d=maximal["paired_cost"],
                stable=maximal["stable_set_expected_bound"],
                clip=item["clipped_cost_bound"],
                linear=item["linear_cost_bound"],
                gap=maximal["paired_cost"] - wasserstein,
            )
        )
    return "\n".join(rows)


def build_report(
    timestamp: str,
    run_directory: Path,
    metrics: Sequence[dict[str, Any]],
    violations: dict[str, int],
    tests: dict[str, Any],
    single_shock: dict[str, Any],
    counterexample: dict[str, Any],
    frozen: dict[str, Any],
    environment: dict[str, Any],
    gate_label: str,
    gate_passed: bool,
    counts: dict[str, int],
) -> str:
    max_probability_error = max(
        max(
            coupling["probability_normalization_error"]
            for coupling in item["couplings"].values()
        )
        for item in metrics
    )
    max_coupling_marginal_error = max(
        max(
            coupling["left_terminal_marginal_error"],
            coupling["right_terminal_marginal_error"],
        )
        for item in metrics
        for coupling in item["couplings"].values()
    )
    max_inverse_error = max(item["inverse_series_max_error"] for item in metrics)
    max_lp_residual = max(item["wasserstein"]["max_marginal_residual"] for item in metrics)
    failed_tests = "None." if tests["returncode"] == 0 else "See run `audits/pytest_output.txt`."
    recommendation = (
        "STOP. T0 passed and is scientifically eligible for user review, but WP-P0 requires explicit authorization."
        if gate_passed
        else "STOP. Do not proceed to WP-P0; inspect the preserved violations/counterexamples."
    )
    return f"""# FDCA Visual-Token T0 Synthetic Report

Run timestamp (Asia/Seoul): `{timestamp}`
Run directory: `{run_directory}`

## 1. Executive verdict

**{gate_label}**

The frozen T0 finite-state contract was executed without a real visual model,
checkpoint, GPU service, or Monte Carlo primary estimate. Four named families
and three deterministic random-table panels were exhaustively enumerated.

## 2. Git/source provenance

- Project root: `{PROJECT_ROOT}`
- Git branch: `{git_output('branch', '--show-current') or 'main'}`
- Git HEAD during run generation: `{git_output('rev-parse', 'HEAD')}` (the contract's one local commit is created after artifacts are complete)
- Git remotes during run: `{git_output('remote', '-v') or 'none'}`
- Frozen inputs byte-match expected SHA-256: `{frozen['all_match']}`
- Code snapshot: `{run_directory / 'code_snapshot'}`
- Existing dLLM repository: not accessed or modified by T0 execution

## 3. Synthetic families

- Family A: independent ternary coordinates, including a two-coordinate product batch; exact `M=0`.
- Family B: sparse binary local propagation with one/two-parent dependencies.
- Family C: dense strong parity propagation; clipped envelope saturation stress test.
- Family D: independent persistent fresh innovation; explicit seed-only counterexample.
- Additional property panel: random conditional tables with seeds `{', '.join(map(str, RANDOM_SEEDS))}`.

## 4. Exact enumeration verification

- Admissible state cases: `{counts['admissible_state_cases']}`
- Conditional state-law cases: `{counts['conditional_state_cases']}`
- Single-site neighbor pairs used for exact `M`: `{counts['neighbor_pair_cases']}`
- Positive-mass paired trajectory cases: `{counts['trajectory_cases']}`
- Persistent conditional first-split cases: `{counts['persistent_conditional_cases']}`
- Maximum probability normalization error: `{max_probability_error:.3e}`
- Terminal laws were computed directly and independently reconstructed from coupled trajectories.

## 5. Coupling verification

Conditional-form categorical maximal coupling and inverse-CDF CRN both retained
their declared marginals. Maximum terminal marginal error was
`{max_coupling_marginal_error:.3e}`. Maximal-coupling mismatch=TV and categorical
boundary/support cases are covered by the property suite. Influence envelopes
were claimed only for the coordinatewise maximal rollout; CRN was used only for
marginal-valid and transport comparisons.

## 6. Influence matrix verification

Every `b`/`s` supremum and every admissible single-site `M[u,v]` supremum was
enumerated. Time-order violations: `{violations['time_order']}`; hybrid-lemma
violations: `{violations['hybrid_lemma']}`; maximum nilpotence residual:
`{max(item['nilpotence_residual'] for item in metrics):.3e}`.

## 7. Stable-set theorem verification

The pathwise inequality `terminal weighted Hamming <= unresolved weighted mass
before first divergence` was checked on every maximal and CRN trajectory.
Violation count: `{violations['stable_set']}`.

## 8. Clipped vs resolvent envelope

Maximal-rollout recursion violations: `{violations['linear_recursion']}`. Clipped
envelope violations: `{violations['clipped_envelope']}`. Direct linear-resolvent
envelope violations: `{violations['resolvent_envelope']}`. `clipped <= linear`
violations: `{violations['clipped_vs_linear']}`. The inverse and finite
nilpotent series agreed to maximum error `{max_inverse_error:.3e}`.

{compact_family_table(metrics)}

All costs use normalized uniform token Hamming. `Stable` is the expected
first-divergence residual-mass bound under the maximal coupling.

## 9. Single-shock result

Family B was split only at round 0 and both branches then used the same `q`
suffix. Conditional seed `{single_shock['seed']}` had mass
`{single_shock['conditioning_event_mass']:.12f}`. Exact conditional mismatch
profile was `{single_shock['conditional_mismatch_profile']}` and the seed-only
propagation envelope was `{single_shock['seed_only_envelope']}`. Conditional
cost/envelope were `{single_shock['conditional_paired_cost']:.12f}` /
`{single_shock['seed_only_cost_bound']:.12f}`. Result: `{'PASS' if single_shock['passed'] else 'FAIL'}`.

## 10. Persistent-approximation result

The reference branch used `p` and the approximate branch used `q` at every
round. Global direct innovation remained present after a split. Persistent
conditional cases checked: `{counts['persistent_conditional_cases']}`;
violations with future innovation included: `{violations['persistent_conditional']}`.

## 11. Fresh-innovation counterexample

Counterexample found: `{counterexample['counterexample_found']}`. In Family D,
condition on first divergence at round 0 with seed `{{0}}` (event mass
`{counterexample['conditioning_event_mass']:.12f}`). `M=0`, so seed-only gives
`{counterexample['seed_only_envelope']}`, while the exact terminal mismatch
profile is `{counterexample['conditional_mismatch_profile']}` because future
fresh innovations are `0.5` and `0.4`. Normalized Hamming is
`{counterexample['conditional_paired_cost']:.12f}` versus seed-only
`{counterexample['seed_only_cost_bound']:.12f}`, a strict gap of
`{counterexample['seed_only_cost_gap']:.12f}`. Adding future innovation gives
`{counterexample['persistent_envelope_with_future_innovation']}` and restores
the valid bound.

## 12. Exact Wasserstein comparisons

Each finite terminal transportation problem was solved over the complete
support with SciPy HiGHS. `W_d <= D_Gamma` violations across maximal and CRN
couplings: `{violations['wasserstein_vs_coupling']}`. Maximum LP marginal
residual: `{max_lp_residual:.3e}`. Exact family gaps are in the table in Section
8 and `derived/wasserstein_table.json`.

## 13. Numerical tolerance policy

- dtype: float64
- probability/marginal normalization: `{PROBABILITY_TOLERANCE:.1e}`
- theorem inequalities: `{INEQUALITY_TOLERANCE:.1e}`
- matrix inverse/finite-series equality: `{MATRIX_TOLERANCE:.1e}`
- LP/coupling comparison: `{LP_TOLERANCE:.1e}`
- Monte Carlo was not used for any T0 claim.

Observed numerical-instability flag: `{not all(item['all_numeric_finite'] for item in metrics)}`.

## 14. Failed tests/counterexamples

- Property test result: `{tests['passed']} passed`, return code `{tests['returncode']}`.
- Failed tests: {failed_tests}
- Unexplained theorem/code counterexamples: `{'none' if sum(violations.values()) == 0 else 'present; see audits/theorem_violations.json'}`.
- The Family D seed-only failure is an intended scientific counterexample, not a gate violation; it is preserved under `reports/counterexamples/`.

## 15. Gate decision

**{gate_label}**

All gate criteria and machine-readable evidence are recorded in
`{PROJECT_ROOT / 'reports/T0_GATE.json'}`.

## 16. Recommendation for P0

{recommendation}
"""


def create_code_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for directory in ("fdca", "tests", "configs", "experiments/FDCA_VIS_T0_SYNTHETIC"):
        source = PROJECT_ROOT / directory
        shutil.copytree(
            source,
            destination / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in (
        "README.md",
        "DECISION_LOG.md",
        "research_state.yaml",
        "pyproject.toml",
        "requirements-t0.txt",
    ):
        shutil.copy2(PROJECT_ROOT / filename, destination / filename)


def finalize_manifests(run_directory: Path) -> None:
    excluded = {"FILE_MANIFEST.json", "SHA256SUMS.txt"}
    files = sorted(
        path
        for path in run_directory.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    manifest = [
        {
            "path": str(path.relative_to(run_directory)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_json(run_directory / "FILE_MANIFEST.json", {"files": manifest})
    hash_files = sorted(
        path for path in run_directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    write_text(
        run_directory / "SHA256SUMS.txt",
        "".join(
            f"{sha256(path)}  {path.relative_to(run_directory)}\n" for path in hash_files
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        help="Optional deterministic run timestamp in YYYYMMDD_HHMMSS form.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        raise ValueError("timestamp must have YYYYMMDD_HHMMSS form")
    run_directory = PROJECT_ROOT / "runs" / f"FDCA_VIS_T0_SYNTHETIC_{timestamp}"
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    for subdirectory in ("config", "raw", "derived", "audits", "figures", "reports", "code_snapshot"):
        (run_directory / subdirectory).mkdir(parents=True, exist_ok=False)

    tests = run_property_tests()
    frozen = frozen_input_audit()
    environment = environment_audit()
    named_models = canonical_families()
    random_models = tuple(random_table_family(seed) for seed in RANDOM_SEEDS)
    models = named_models + random_models

    metrics: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    jsonl_rows: list[dict[str, Any]] = []
    for model in models:
        model_metrics, model_raw = audit_model(model)
        metrics.append(model_metrics)
        raw[model.name] = model_raw
        for coupling, coupling_raw in model_raw["coupled"].items():
            jsonl_rows.extend(coupling_raw["paths"])

    single_shock = single_shock_audit()
    counterexample = persistent_counterexample_audit()
    violations = aggregate_violations(metrics)

    probability_ok = all(
        coupling["probability_normalization_error"] <= PROBABILITY_TOLERANCE
        for item in metrics
        for coupling in item["couplings"].values()
    ) and all(
        max(item["terminal_law_normalization_error"].values()) <= PROBABILITY_TOLERANCE
        for item in metrics
    )
    coupling_marginals_ok = all(
        max(coupling["left_terminal_marginal_error"], coupling["right_terminal_marginal_error"])
        <= PROBABILITY_TOLERANCE
        for item in metrics
        for coupling in item["couplings"].values()
    )
    nilpotence_ok = all(
        item["nilpotence_residual"] <= MATRIX_TOLERANCE and not item["time_order_violations"]
        for item in metrics
    )
    inverse_series_ok = all(
        item["inverse_series_max_error"] <= MATRIX_TOLERANCE for item in metrics
    )
    numerical_ok = all(item["all_numeric_finite"] for item in metrics) and all(
        item["wasserstein"]["lp_success"]
        and item["wasserstein"]["max_marginal_residual"] <= LP_TOLERANCE
        for item in metrics
    )
    all_violation_count = sum(violations.values())
    criteria = {
        "exhaustive_theorem_code_violation_count_zero": all_violation_count == 0,
        "exact_terminal_marginals_verified": probability_ok,
        "exact_coupling_marginals_verified": coupling_marginals_ok,
        "m_nilpotence_verified": nilpotence_ok,
        "inverse_equals_finite_series": inverse_series_ok,
        "clipped_le_linear_every_claimed_case": violations["clipped_vs_linear"] == 0,
        "exact_pi_le_linear_resolvent": violations["resolvent_envelope"] == 0,
        "stable_set_pathwise_violation_zero": violations["stable_set"] == 0,
        "wasserstein_le_constructed_cost": violations["wasserstein_vs_coupling"] == 0,
        "single_shock_propagation_passes": single_shock["passed"],
        "persistent_fresh_innovation_counterexample_found": counterexample[
            "counterexample_found"
        ],
        "no_unexplained_numerical_instability": numerical_ok,
        "test_suite_all_pass": tests["returncode"] == 0 and tests["passed"] > 0,
        "frozen_inputs_match": frozen["all_match"],
    }
    gate_passed = all(criteria.values())
    gate_label = "PASS_FDCA_VIS_T0_SYNTHETIC" if gate_passed else "FAIL_FDCA_VIS_T0_SYNTHETIC"

    counts = {
        "named_synthetic_families": len(named_models),
        "random_property_instances": len(random_models),
        "admissible_state_cases": sum(item["state_space_cases"] for item in metrics),
        "conditional_state_cases": sum(item["conditional_state_cases"] for item in metrics),
        "neighbor_pair_cases": sum(item["neighbor_pair_cases"] for item in metrics),
        "trajectory_cases": sum(
            coupling["trajectory_cases"]
            for item in metrics
            for coupling in item["couplings"].values()
        )
        + single_shock["trajectory_cases"],
        "persistent_conditional_cases": sum(
            item["persistent_conditional_cases"] for item in metrics
        ),
        "property_tests_passed": tests["passed"],
    }

    named_metrics = [item for item in metrics if item["family"] in {"A", "B", "C", "D"}]
    family_gaps = {
        item["family"]: {
            "wasserstein": item["wasserstein"]["distance"],
            "paired_cost_maximal": item["couplings"]["maximal"]["paired_cost"],
            "stable_set_expected_bound": item["couplings"]["maximal"][
                "stable_set_expected_bound"
            ],
            "clipped_cost_bound": item["clipped_cost_bound"],
            "linear_cost_bound": item["linear_cost_bound"],
            "paired_minus_wasserstein": item["couplings"]["maximal"]["paired_cost"]
            - item["wasserstein"]["distance"],
            "stable_minus_paired": item["couplings"]["maximal"][
                "stable_set_expected_bound"
            ]
            - item["couplings"]["maximal"]["paired_cost"],
            "clipped_minus_paired": item["clipped_cost_bound"]
            - item["couplings"]["maximal"]["paired_cost"],
            "linear_minus_paired": item["linear_cost_bound"]
            - item["couplings"]["maximal"]["paired_cost"],
        }
        for item in named_metrics
    }

    gate = {
        "work_package": "FDCA_VIS_T0_SYNTHETIC",
        "gate": gate_label,
        "passed": gate_passed,
        "timestamp_asia_seoul": timestamp,
        "criteria": criteria,
        "counts": counts,
        "violations": violations,
        "all_theorem_code_violation_count": all_violation_count,
        "fresh_innovation_counterexample": {
            "found": counterexample["counterexample_found"],
            "seed_only_cost_gap": counterexample["seed_only_cost_gap"],
        },
        "key_numerical_gaps": family_gaps,
        "numerical_tolerances": {
            "probability": PROBABILITY_TOLERANCE,
            "inequality": INEQUALITY_TOLERANCE,
            "matrix": MATRIX_TOLERANCE,
            "lp": LP_TOLERANCE,
        },
        "environment": environment,
        "run_directory": str(run_directory),
        "report": str(PROJECT_ROOT / "reports/T0_REPORT.md"),
        "recommendation": (
            "STOP_PENDING_USER_REVIEW; P0 requires explicit authorization"
            if gate_passed
            else "STOP; do not proceed to P0"
        ),
    }

    # Core run artifacts.
    shutil.copy2(PROJECT_ROOT / "configs/t0.yaml", run_directory / "config/t0.yaml")
    write_json(run_directory / "config/resolved_config.json", yaml.safe_load((PROJECT_ROOT / "configs/t0.yaml").read_text()))
    write_json(
        run_directory / "raw/canonical_model_definitions.json",
        {model.name: raw[model.name]["model_definition"] for model in named_models},
    )
    write_json(
        run_directory / "raw/random_model_definitions.json",
        {model.name: raw[model.name]["model_definition"] for model in random_models},
    )
    write_json(
        run_directory / "raw/exact_terminal_laws.json",
        {name: value["terminal_laws"] for name, value in raw.items()},
    )
    write_json(
        run_directory / "raw/exact_terminal_joint_laws.json",
        {
            name: {
                coupling: coupled["terminal_joint_law"]
                for coupling, coupled in value["coupled"].items()
            }
            for name, value in raw.items()
        },
    )
    write_text(
        run_directory / "raw/exact_coupled_trajectories.jsonl",
        "".join(json.dumps(json_ready(row), sort_keys=True) + "\n" for row in jsonl_rows),
    )
    write_json(
        run_directory / "raw/direct_innovation_tables.json",
        {name: value["direct_innovation_table"] for name, value in raw.items()},
    )
    write_json(run_directory / "derived/family_metrics.json", metrics)
    write_json(run_directory / "derived/key_numerical_gaps.json", family_gaps)
    write_json(
        run_directory / "derived/exact_vectors_and_matrices.json",
        {
            item["name"]: {
                "b_supremum": item["direct_b_supremum"],
                "M": item["influence_matrix"],
                "pi_maximal": item["couplings"]["maximal"]["mismatch_profile"],
                "pi_crn": item["couplings"]["crn"]["mismatch_profile"],
                "clipped": item["clipped_envelope"],
                "linear_resolvent": item["linear_resolvent_envelope"],
            }
            for item in metrics
        },
    )
    write_json(
        run_directory / "derived/wasserstein_table.json",
        {
            item["name"]: {
                "W_d": item["wasserstein"]["distance"],
                "D_maximal": item["couplings"]["maximal"]["paired_cost"],
                "D_crn": item["couplings"]["crn"]["paired_cost"],
                "W_le_D_maximal": not item["couplings"]["maximal"]["wasserstein_violation"],
                "W_le_D_crn": not item["couplings"]["crn"]["wasserstein_violation"],
            }
            for item in metrics
        },
    )
    single_shock_for_json = {key: value for key, value in single_shock.items() if key != "paths"}
    write_json(run_directory / "derived/single_shock.json", single_shock_for_json)
    write_json(run_directory / "raw/single_shock_paths.json", single_shock["paths"])
    write_json(run_directory / "derived/persistent_fresh_innovation_counterexample.json", counterexample)
    write_json(run_directory / "audits/theorem_violations.json", violations)
    write_json(
        run_directory / "audits/coupling_marginal_audit.json",
        {
            item["name"]: {
                coupling_name: {
                    "left_error": coupling["left_terminal_marginal_error"],
                    "right_error": coupling["right_terminal_marginal_error"],
                    "probability_sum": coupling["probability_sum"],
                }
                for coupling_name, coupling in item["couplings"].items()
            }
            for item in metrics
        },
    )
    write_json(run_directory / "audits/frozen_input_audit.json", frozen)
    write_json(run_directory / "audits/environment.json", environment)
    write_text(run_directory / "audits/pytest_output.txt", tests["output"])
    write_json(
        run_directory / "audits/pytest_summary.json",
        {key: value for key, value in tests.items() if key != "output"},
    )
    write_text(
        run_directory / "figures/README.md",
        "# Figures\n\nT0 claims are exact finite tables; no stochastic or decorative figure is required.\n",
    )

    report = build_report(
        timestamp,
        run_directory,
        metrics,
        violations,
        tests,
        single_shock,
        counterexample,
        frozen,
        environment,
        gate_label,
        gate_passed,
        counts,
    )
    write_text(PROJECT_ROOT / "reports/T0_REPORT.md", report)
    write_json(PROJECT_ROOT / "reports/T0_GATE.json", gate)
    write_json(
        PROJECT_ROOT / "reports/counterexamples/persistent_first_split_seed_only.json",
        counterexample,
    )
    shutil.copy2(PROJECT_ROOT / "reports/T0_REPORT.md", run_directory / "reports/T0_REPORT.md")
    shutil.copy2(PROJECT_ROOT / "reports/T0_GATE.json", run_directory / "reports/T0_GATE.json")

    research_state = {
        "project": "FDCA visual-token",
        "work_package": "FDCA_VIS_T0_SYNTHETIC",
        "phase": "T0 synthetic exhaustive theorem/code validation",
        "status": "complete" if gate_passed else "failed",
        "gate": gate_label,
        "date": "2026-08-12",
        "run_directory": str(run_directory),
        "next_action": "STOP_PENDING_USER_REVIEW" if gate_passed else "STOP_AND_REVIEW_FAILURE",
        "scope": {
            "real_models": "prohibited_not_used",
            "external_checkpoints": "prohibited_not_used",
            "p0_automatic_transition": "prohibited",
            "existing_dllm_repository": "read_only_out_of_scope_not_touched",
        },
        "frozen_contracts": [row["path"] for row in frozen["files"]],
    }
    write_text(
        PROJECT_ROOT / "research_state.yaml",
        yaml.safe_dump(research_state, sort_keys=False, allow_unicode=True),
    )
    create_code_snapshot(run_directory / "code_snapshot")
    write_json(
        run_directory / "STATUS.json",
        {
            "status": "complete" if gate_passed else "failed",
            "gate": gate_label,
            "timestamp_asia_seoul": timestamp,
            "next_action": "STOP_PENDING_USER_REVIEW",
        },
    )
    finalize_manifests(run_directory)

    print(gate_label)
    print(run_directory)
    print(f"tests={tests['passed']} violations={all_violation_count}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
