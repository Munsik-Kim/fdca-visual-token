"""Read-only audit adapter for the pinned Halton-MaskGIT samplers.

The adapter deliberately calls the upstream sampler.  It observes model inputs,
model-output metadata, and the VQ decoder input through wrappers that do not
change logits, random-number draws, selector decisions, or returned values.
The returned ``l_codes``/``l_mask`` objects are then interpreted according to
the sampler-specific source contract.

This module is an audit aid, not the P05 coupling implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


UPSTREAM_COMMIT = "f61b0a1314717004dc7487531fd16a8bb71e1888"
MASK_VALUE = 16_384
CODEBOOK_SIZE = 16_384
INPUT_SIZE = 16


def seed_all(seed: int) -> None:
    """Reset every RNG used by the pinned samplers."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_tensor(value: torch.Tensor) -> str:
    """Return a byte-level digest without relying on NumPy dtype support."""

    tensor = value.detach().contiguous().cpu()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_record(value: torch.Tensor, *, include_values: bool = False) -> dict[str, Any]:
    tensor = value.detach().contiguous().cpu()
    record: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": sha256_tensor(tensor),
    }
    if include_values:
        record["values"] = tensor.tolist()
    return record


def halton_cumulative_counts(step: int = 32, input_size: int = 16) -> list[int]:
    """Exact integer schedule used in ``Sampler/halton_sampler.py``."""

    result: list[int] = []
    for index in range(step):
        ratio = (index + 1) / step
        # Match the upstream float32 torch expression, not a float64 rewrite.
        r = 1 - (torch.arccos(torch.tensor(ratio)) / (torch.pi * 0.5))
        cumulative = int(r * (input_size**2))
        result.append(max(index + 1, cumulative))
    return result


def incremental_counts(cumulative: Iterable[int]) -> list[int]:
    previous = 0
    increments: list[int] = []
    for current in cumulative:
        increments.append(int(current) - previous)
        previous = int(current)
    return increments


def import_upstream(upstream_root: Path) -> dict[str, Any]:
    root = str(upstream_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return {
        "Transformer": importlib.import_module("Network.transformer").Transformer,
        "VQ_models": importlib.import_module("Network.vq_model").VQ_models,
        "HaltonSampler": importlib.import_module("Sampler.halton_sampler").HaltonSampler,
        "ConfidenceSampler": importlib.import_module("Sampler.confidence_sampler").ConfidenceSampler,
    }


@dataclass
class LoadedRuntime:
    trainer: Any
    transformer_parameters: int
    vq_parameters: int
    checkpoint_iteration: int
    checkpoint_epoch: int


def load_runtime(
    upstream_root: Path,
    generator_checkpoint: Path,
    vq_checkpoint: Path,
    device: str = "cuda",
) -> LoadedRuntime:
    """Load only weights, bypassing the upstream training/data stack."""

    modules = import_upstream(upstream_root)
    Transformer = modules["Transformer"]
    VQ_models = modules["VQ_models"]

    vit = Transformer(
        input_size=INPUT_SIZE,
        nclass=1000,
        hidden_dim=384,
        codebook_size=CODEBOOK_SIZE,
        depth=6,
        heads=6,
        mlp_dim=384 * 4,
        dropout=0.1,
        register=1,
        proj=1,
    )
    generator = torch.load(generator_checkpoint, map_location="cpu", weights_only=True)
    state_dict = {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in generator["model_state_dict"].items()
    }
    vit.load_state_dict(state_dict, strict=True)
    vit = vit.to(device).eval()

    ae = VQ_models["VQ-16"](codebook_size=CODEBOOK_SIZE, codebook_embed_dim=8)
    vq_payload = torch.load(vq_checkpoint, map_location="cpu", weights_only=True)
    ae.load_state_dict(vq_payload["model"], strict=True)
    ae = ae.to(device).eval()

    args = SimpleNamespace(
        device=device,
        mask_value=MASK_VALUE,
        codebook_size=CODEBOOK_SIZE,
        use_ema=False,
    )
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    trainer = SimpleNamespace(
        vit=vit,
        ae=ae,
        input_size=INPUT_SIZE,
        args=args,
        autocast=autocast,
    )
    return LoadedRuntime(
        trainer=trainer,
        transformer_parameters=sum(p.numel() for p in vit.parameters()),
        vq_parameters=sum(p.numel() for p in ae.parameters()),
        checkpoint_iteration=int(generator["iter"]),
        checkpoint_epoch=int(generator["global_epoch"]),
    )


class TracingModel(nn.Module):
    """Transparent model wrapper recording pre-transition states and metadata."""

    def __init__(self, base: nn.Module, batch_size: int):
        super().__init__()
        self.base = base
        self.batch_size = batch_size
        self.pre_states: list[torch.Tensor] = []
        self.output_metadata: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.pre_states.append(x[: self.batch_size].detach().clone().cpu())
        output = self.base(x, *args, **kwargs)
        finite = bool(torch.isfinite(output).all().item())
        self.output_metadata.append(
            {
                "raw_logit_shape": list(output.shape),
                "raw_logit_dtype": str(output.dtype),
                "raw_logit_device": str(output.device),
                "raw_logits_all_finite": finite,
                "proposal_probability_shape_after_cfg": [
                    self.batch_size,
                    INPUT_SIZE * INPUT_SIZE,
                    CODEBOOK_SIZE + 1,
                ],
                "proposal_probability_retained": False,
            }
        )
        return output


class TracingAutoencoder(nn.Module):
    """Transparent VQ wrapper capturing the exact grid passed to decoding."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.decode_inputs: list[torch.Tensor] = []

    def decode_code(self, code: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(code.detach().clone().cpu())
        return self.base.decode_code(code)


@dataclass
class SamplerCall:
    image: torch.Tensor | None
    proposals: list[torch.Tensor]
    returned_masks: list[torch.Tensor]
    wall_seconds: float
    peak_vram_bytes: int
    error_type: str | None
    error_message: str | None
    pre_states: list[torch.Tensor]
    terminal_decode_grid: torch.Tensor | None
    probability_metadata: list[dict[str, Any]]


def call_upstream(
    sampler: Any,
    trainer: Any,
    *,
    seed: int,
    class_id: int,
    batch_size: int,
    instrument: bool,
) -> SamplerCall:
    """Call upstream exactly, optionally through transparent observers."""

    seed_all(seed)
    trainer.vit.eval()
    trainer.ae.eval()
    tracing_model: TracingModel | None = None
    tracing_ae: TracingAutoencoder | None = None
    call_trainer = trainer
    if instrument:
        tracing_model = TracingModel(trainer.vit, batch_size).to(trainer.args.device)
        tracing_model.eval()
        tracing_ae = TracingAutoencoder(trainer.ae).to(trainer.args.device)
        tracing_ae.eval()
        call_trainer = SimpleNamespace(
            vit=tracing_model,
            ae=tracing_ae,
            input_size=trainer.input_size,
            args=trainer.args,
            autocast=trainer.autocast,
        )

    labels = torch.full((batch_size,), class_id, dtype=torch.long, device=trainer.args.device)
    if torch.cuda.is_available() and trainer.args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    image: torch.Tensor | None = None
    proposals: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    error_type: str | None = None
    error_message: str | None = None
    try:
        image, proposals, masks = sampler(
            call_trainer,
            nb_sample=batch_size,
            labels=labels,
            verbose=False,
        )
        image = image.detach().clone().cpu()
        proposals = [value.detach().clone().cpu() for value in proposals]
        masks = [value.detach().clone().cpu() for value in masks]
    except Exception as exc:  # The source-original Confidence B=1 failure is data.
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        if torch.cuda.is_available() and trainer.args.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = 0
    elapsed = time.perf_counter() - started
    terminal = None
    if tracing_ae is not None and tracing_ae.decode_inputs:
        terminal = tracing_ae.decode_inputs[-1]
    return SamplerCall(
        image=image,
        proposals=proposals,
        returned_masks=masks,
        wall_seconds=elapsed,
        peak_vram_bytes=peak,
        error_type=error_type,
        error_message=error_message,
        pre_states=[] if tracing_model is None else tracing_model.pre_states,
        terminal_decode_grid=terminal,
        probability_metadata=[] if tracing_model is None else tracing_model.output_metadata,
    )


def calls_byte_identical(left: SamplerCall, right: SamplerCall) -> tuple[bool, dict[str, Any]]:
    """Compare every object returned by the upstream sampler byte-for-byte."""

    details: dict[str, Any] = {
        "error_type_equal": left.error_type == right.error_type,
        "error_message_equal": left.error_message == right.error_message,
    }
    if left.error_type or right.error_type:
        equal = bool(details["error_type_equal"] and details["error_message_equal"])
        return equal, details
    assert left.image is not None and right.image is not None
    details["image_equal"] = torch.equal(left.image, right.image)
    details["proposal_count_equal"] = len(left.proposals) == len(right.proposals)
    details["mask_count_equal"] = len(left.returned_masks) == len(right.returned_masks)
    details["proposals_equal"] = details["proposal_count_equal"] and all(
        torch.equal(a, b) for a, b in zip(left.proposals, right.proposals)
    )
    details["returned_masks_equal"] = details["mask_count_equal"] and all(
        torch.equal(a, b) for a, b in zip(left.returned_masks, right.returned_masks)
    )
    equal = all(bool(value) for value in details.values())
    return equal, details


def materialize_trace(
    sampler_name: str,
    call: SamplerCall,
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Classify actual transitions using sampler-specific returned semantics."""

    if call.error_type:
        return [], {}
    if len(call.proposals) != len(call.returned_masks):
        raise ValueError("proposal/mask length mismatch")
    steps = len(call.proposals)
    initial = torch.full((batch_size, INPUT_SIZE, INPUT_SIZE), MASK_VALUE, dtype=torch.long)
    state = initial.clone()
    committed = torch.zeros_like(state, dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    newly_masks: list[np.ndarray] = []
    committed_masks: list[np.ndarray] = []
    unresolved_masks: list[np.ndarray] = []
    revised_masks: list[np.ndarray] = []

    for index in range(steps):
        pre = state.clone()
        proposal = call.proposals[index].long()
        returned = call.returned_masks[index].bool()
        if sampler_name == "halton":
            selected = returned
            newly = selected & ~committed
            revised = selected & committed & (proposal != pre)
            state[selected] = proposal[selected]
            committed = committed | selected
            unresolved = ~committed
            returned_semantics = "incremental_newly_selected_positions"
        elif sampler_name == "confidence":
            unresolved = returned
            if index + 1 < len(call.pre_states):
                post = call.pre_states[index + 1].long()
            else:
                if call.terminal_decode_grid is None:
                    raise ValueError("missing terminal decode grid")
                post = call.terminal_decode_grid.long().clone()
                post[unresolved] = MASK_VALUE
            previous_committed = ~((pre == MASK_VALUE))
            newly = (pre == MASK_VALUE) & ~unresolved
            selected = previous_committed | newly
            revised = previous_committed & (post != pre)
            state = post
            committed = ~unresolved
            returned_semantics = "post_transition_unresolved_positions"
        else:
            raise ValueError(f"unknown sampler: {sampler_name}")

        if index < len(call.pre_states):
            observed_pre = call.pre_states[index].long()
            pre_matches_model_input = bool(torch.equal(pre, observed_pre))
        else:
            pre_matches_model_input = False
        actual_mask_tokens = state == MASK_VALUE
        rows.append(
            {
                "step_index": index,
                "pre_state_sha256": sha256_tensor(pre),
                "proposal_sha256": sha256_tensor(proposal),
                "returned_mask_sha256": sha256_tensor(returned),
                "selected_count": int(selected.sum()),
                "newly_committed_count": int(newly.sum()),
                "cumulative_committed_count": int(committed.sum()),
                "revised_previously_committed_count": int(revised.sum()),
                "unresolved_count": int(unresolved.sum()),
                "actual_mask_token_count": int(actual_mask_tokens.sum()),
                "post_state_sha256": sha256_tensor(state),
                "returned_mask_semantics": returned_semantics,
                "returned_code_semantics": "full_sampled_proposal_grid_not_post_state",
                "pre_state_matches_observed_model_input": pre_matches_model_input,
                "rng_sources": rng_sources(sampler_name, index),
                "probability_metadata": (
                    call.probability_metadata[index]
                    if index < len(call.probability_metadata)
                    else None
                ),
            }
        )
        states.append(state.numpy().copy())
        newly_masks.append(newly.numpy().copy())
        committed_masks.append(committed.numpy().copy())
        unresolved_masks.append(unresolved.numpy().copy())
        revised_masks.append(revised.numpy().copy())

    arrays = {
        "initial_state": initial.numpy(),
        "post_states": np.stack(states),
        "newly_committed_masks": np.stack(newly_masks),
        "cumulative_committed_masks": np.stack(committed_masks),
        "unresolved_masks": np.stack(unresolved_masks),
        "revised_committed_masks": np.stack(revised_masks),
        "proposals": np.stack([value.numpy() for value in call.proposals]),
        "returned_masks": np.stack([value.numpy() for value in call.returned_masks]),
    }
    return rows, arrays


def rng_sources(sampler_name: str, index: int, randomize: Any = None) -> list[str]:
    if sampler_name == "halton":
        sources = ["torch.distributions.Categorical.sample"]
        if randomize is True:
            sources.append("torch.randint:spatial_halton_roll_offset")
        return sources
    sources = ["torch.distributions.Categorical.sample"]
    if randomize == "linear":
        sources.append("numpy.random.gumbel")
    elif randomize in {"warm_up", "random"}:
        sources.append("torch.rand_like")
    return sources


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
