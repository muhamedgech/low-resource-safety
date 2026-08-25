"""Benchmark fixed-length generation overhead on stratified PolyRefuse prompts."""

import argparse
import contextlib
import gc
import json
import platform
import random
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import nn

from multilingual_latent_safety.adasteer import (
    AdaSteerBundle,
    install_adasteer_hooks,
    load_adasteer_bundle,
)
from multilingual_latent_safety.cast import CastBundle, install_cast_hooks
from multilingual_latent_safety.conditional_vhrl import (
    LanguageGate,
    conditional_alpha_calibration,
    train_gates,
)
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import load_generation_model
from multilingual_latent_safety.interventions import get_decoder_blocks
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.paths import pooled_direction_file
from multilingual_latent_safety.runtime import stable_seed
from multilingual_latent_safety.vector_ops import unit_normalize

from run_cast_steering import (
    condition_points_for_targets,
    layer_list,
    make_bundle,
    train_or_load_cast_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("base", "ours_lrl_32", "cast_adapted_32", "adasteer_adapted_32")
DEFAULT_LANGUAGES = ("sw", "am", "my", "km", "si", "yo")


@dataclass(frozen=True)
class PromptBatch:
    batch_id: str
    language: str
    subset: str
    prompt_ids: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    inputs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RuntimeGate:
    basis: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    mean: torch.Tensor | None
    scale: torch.Tensor | None
    threshold: float

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        features = hidden.to(torch.float32) @ self.basis
        if self.mean is not None and self.scale is not None:
            features = (features - self.mean) / self.scale
        return features @ self.weight + self.bias


class Lrl32State:
    """Compute the learned gate during prefill, then reuse its decision while decoding."""

    def __init__(self, gate: RuntimeGate, direction: torch.Tensor, alpha: float) -> None:
        self.gate = gate
        self.direction = direction
        self.alpha = alpha
        self.harmful: torch.Tensor | None = None

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[1] > 1:
            logits = self.gate.logits(residual[:, -1, :])
            self.harmful = logits >= self.gate.threshold
        if self.harmful is None:
            return residual

        harmful = self.harmful[: residual.shape[0]]
        harmless = ~harmful
        modified = residual.clone()
        if bool(harmless.any()):
            hidden = residual[harmless]
            coeffs = hidden @ self.direction
            modified[harmless] = hidden - coeffs.unsqueeze(-1) * self.direction
        if bool(harmful.any()):
            token_slice = slice(-1, None) if residual.shape[1] > 1 else slice(None)
            modified[harmful, token_slice] += self.alpha * self.direction
        return modified


@contextlib.contextmanager
def install_lrl32_hook(
    model: nn.Module,
    layer: int,
    gate: RuntimeGate,
    direction: torch.Tensor,
    alpha: float,
) -> Iterator[Lrl32State]:
    state = Lrl32State(gate, direction, alpha)

    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        modified = state.apply(residual)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = get_decoder_blocks(model)[layer].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        choices=("qwen2.5-7b-instruct", "gemma-2-9b-it", "llama-3.1-8b-instruct"),
        default="qwen2.5-7b-instruct",
    )
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--samples-per-cell", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--budget-per-class", type=int, default=32)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/polyrefuse"))
    parser.add_argument(
        "--adasteer-root",
        type=Path,
        default=Path("artifacts/external/adasteer_adapted_polyrefuse_b32"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_configs(args: argparse.Namespace) -> tuple[DictConfig, DictConfig, dict[str, Path]]:
    model = OmegaConf.load(ROOT / "configs/model" / f"{args.model_config}.yaml")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        gate_cfg = compose(
            config_name="routing_refusal/intervene_and_generate",
            overrides=["intervention=conditional_vhrl"],
        )
        cast_cfg = compose(config_name="routing_refusal/run_cast_steering")
    for cfg in (gate_cfg, cast_cfg):
        cfg.model = OmegaConf.create(OmegaConf.to_container(model, resolve=True))
        cfg.dataset.root = str(absolute(args.dataset_root))
        cfg.target_languages = list(args.languages)
    gate_cfg.rank = int(args.rank)
    gate_cfg.budget = int(args.budget_per_class)
    gate_cfg.seed = int(args.seed)
    gate_cfg.probe_device = "cpu"
    if args.layer is not None:
        gate_cfg.layer = int(args.layer)
    cast_cfg.cast.variant = "cast_adapted"
    cast_cfg.cast.training_source = "polyrefuse"
    cast_cfg.cast.budget_per_class = int(args.budget_per_class)
    cast_cfg.cast.artifact_root = (
        f"artifacts/cast/cast_adapted/{cast_cfg.model.name}"
    )
    OmegaConf.resolve(gate_cfg)
    paths = {
        "activations": absolute(Path(gate_cfg.activations_root)),
        "pooled": absolute(Path(gate_cfg.pooled_root)),
        "cast": absolute(Path(cast_cfg.cast.artifact_root)),
        "adasteer": absolute(args.adasteer_root),
    }
    gate_cfg.activations_root = str(paths["activations"])
    gate_cfg.pooled_root = str(paths["pooled"])
    cast_cfg.cast.artifact_root = str(paths["cast"])
    return gate_cfg, cast_cfg, paths


def runtime_gate(gate: LanguageGate, threshold: float, device: torch.device) -> RuntimeGate:
    scaler = gate.probe.scaler
    return RuntimeGate(
        basis=gate.basis.to(device=device, dtype=torch.float32),
        weight=gate.probe.weight.to(device=device, dtype=torch.float32),
        bias=gate.probe.bias.to(device=device, dtype=torch.float32),
        mean=None if scaler is None else scaler.mean.to(device=device, dtype=torch.float32),
        scale=None if scaler is None else scaler.scale.to(device=device, dtype=torch.float32),
        threshold=float(threshold),
    )


def move_cast(bundle: CastBundle, device: torch.device, dtype: torch.dtype) -> CastBundle:
    return CastBundle(
        behavior_directions={
            layer: direction.to(device=device, dtype=dtype)
            for layer, direction in bundle.behavior_directions.items()
        },
        condition_directions={
            layer: direction.to(device=device, dtype=torch.float32)
            for layer, direction in bundle.condition_directions.items()
        },
        condition_layers=bundle.condition_layers,
        behavior_layers=bundle.behavior_layers,
        threshold=bundle.threshold,
        comparator_threshold_is=bundle.comparator_threshold_is,
        behavior_strength=bundle.behavior_strength,
        condition_mode=bundle.condition_mode,
        apply_behavior_on_first_call=bundle.apply_behavior_on_first_call,
    )


def move_adasteer(
    bundle: AdaSteerBundle, device: torch.device, dtype: torch.dtype
) -> AdaSteerBundle:
    return AdaSteerBundle(
        spec=bundle.spec,
        rd_direction=bundle.rd_direction.to(device=device, dtype=dtype),
        hd_direction=bundle.hd_direction.to(device=device, dtype=dtype),
        rd_harmful_anchors=bundle.rd_harmful_anchors.to(device=device),
        rd_harmless_anchors=bundle.rd_harmless_anchors.to(device=device),
        hd_harmful_anchors=bundle.hd_harmful_anchors.to(device=device),
        hd_harmless_anchors=bundle.hd_harmless_anchors.to(device=device),
    )


def prepare_batches(
    cfg: DictConfig,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> list[PromptBatch]:
    batches: list[PromptBatch] = []
    for language in args.languages:
        for subset in ("harmful", "harmless"):
            instructions = load_polyrefuse(cfg.dataset.root, subset, "test", language)
            if len(instructions) < args.samples_per_cell:
                raise ValueError(
                    f"{language}/{subset} has {len(instructions)} test prompts; "
                    f"requested {args.samples_per_cell}"
                )
            rng = random.Random(stable_seed(args.seed, "overhead", language, subset))
            prompt_ids = rng.sample(range(len(instructions)), args.samples_per_cell)
            for batch_index, start in enumerate(range(0, len(prompt_ids), args.batch_size)):
                ids = tuple(prompt_ids[start : start + args.batch_size])
                prompts = [
                    format_prompt(tokenizer, instructions[prompt_id], cfg.model.chat)
                    for prompt_id in ids
                ]
                encoded = tokenizer(
                    prompts, return_tensors="pt", padding=True, add_special_tokens=False
                )
                prompt_tokens = tuple(int(value) for value in encoded["attention_mask"].sum(1))
                inputs = {key: value.to(device) for key, value in encoded.items()}
                batches.append(
                    PromptBatch(
                        batch_id=f"{language}:{subset}:{batch_index:02d}",
                        language=language,
                        subset=subset,
                        prompt_ids=ids,
                        prompt_tokens=prompt_tokens,
                        inputs=inputs,
                    )
                )
    return batches


def method_contexts(
    model: nn.Module,
    batch: PromptBatch,
    layer: int,
    gates: dict[str, RuntimeGate],
    direction: torch.Tensor,
    alpha: float,
    cast_bundles: dict[str, CastBundle],
    adasteer_bundle: AdaSteerBundle,
) -> dict[str, Callable[[], contextlib.AbstractContextManager]]:
    n_layers = len(get_decoder_blocks(model))
    return {
        "base": lambda: contextlib.nullcontext(None),
        "ours_lrl_32": lambda: install_lrl32_hook(
            model, layer, gates[batch.language], direction, alpha
        ),
        "cast_adapted_32": lambda: install_cast_hooks(
            model, cast_bundles[batch.language]
        ),
        "adasteer_adapted_32": lambda: install_adasteer_hooks(
            model, adasteer_bundle, list(range(n_layers))
        ),
    }


def synchronize() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_generate(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    context_factory: Callable[[], contextlib.AbstractContextManager],
    generation_kwargs: dict[str, Any],
) -> tuple[float, int | None]:
    with context_factory() as state:
        synchronize()
        started = time.perf_counter_ns()
        output = model.generate(**inputs, **generation_kwargs)
        synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        expected_length = inputs["input_ids"].shape[1] + generation_kwargs["max_new_tokens"]
        if output.shape[1] != expected_length:
            raise RuntimeError(
                f"fixed-length generation produced {output.shape[1]} tokens; "
                f"expected {expected_length}"
            )
        harmful = None
        if isinstance(state, Lrl32State) and state.harmful is not None:
            harmful = int(state.harmful.sum().item())
        del output
    return elapsed_ms, harmful


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap_mean_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    if not values or samples < 1:
        raise ValueError("bootstrap requires values and a positive sample count")
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(
    batches: list[PromptBatch],
    timings: dict[str, dict[str, list[float]]],
    new_tokens: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    total_tokens = sum(len(batch.prompt_ids) * new_tokens for batch in batches)
    rows = []
    for method in METHODS:
        latencies = [value for batch in batches for value in timings[batch.batch_id][method]]
        paired_overheads = []
        for batch in batches:
            base = statistics.median(timings[batch.batch_id]["base"])
            steered = statistics.median(timings[batch.batch_id][method])
            paired_overheads.append(100 * (steered / base - 1))
        low, high = bootstrap_mean_ci(
            paired_overheads, bootstrap_samples, stable_seed(seed, method, "bootstrap")
        )
        rows.append(
            {
                "method": method,
                "latency_ms_median": statistics.median(latencies),
                "latency_ms_mean": statistics.fmean(latencies),
                "tokens_per_second": total_tokens * len(latencies) / len(batches) / (
                    sum(latencies) / 1000
                ),
                "paired_overhead_percent_mean": statistics.fmean(paired_overheads),
                "paired_overhead_percent_median": statistics.median(paired_overheads),
                "paired_overhead_percent_ci95": [low, high],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this benchmark on a GPU pod")
    if min(args.samples_per_cell, args.batch_size, args.new_tokens, args.repeats) < 1:
        raise ValueError("sample, batch, token, and repeat counts must be positive")
    if args.warmups < 0 or args.bootstrap_samples < 1:
        raise ValueError("warmups must be non-negative and bootstrap-samples must be positive")
    if args.rank < 1 or args.budget_per_class < 1:
        raise ValueError("rank and budget-per-class must be positive")
    if args.samples_per_cell % args.batch_size:
        raise ValueError("samples-per-cell must be divisible by batch-size")

    gate_cfg, cast_cfg, paths = build_configs(args)
    print("[prep] fitting the HRL + 32 LRL gates from cached train activations")
    gate_result = train_gates(gate_cfg)
    direction_path = pooled_direction_file(
        paths["pooled"], "hrl", gate_cfg.token_position, int(gate_cfg.layer)
    )
    if not direction_path.exists():
        raise FileNotFoundError(f"missing pooled HRL direction: {direction_path}")
    direction_cpu = load_file(direction_path)["direction"]
    alpha_calibration = conditional_alpha_calibration(
        gate_cfg, gate_result.source_languages, direction_cpu
    )
    adasteer_cpu = load_adasteer_bundle(paths["adasteer"], str(gate_cfg.model.name))

    model, tokenizer = load_generation_model(gate_cfg.model)
    blocks = get_decoder_blocks(model)
    block_devices = {next(block.parameters()).device for block in blocks}
    if len(block_devices) != 1 or next(iter(block_devices)).type != "cuda":
        raise RuntimeError("benchmark requires all decoder layers on one CUDA device")
    device = next(iter(block_devices))
    dtype = next(blocks[0].parameters()).dtype

    print("[prep] loading or training adapted CAST artifacts outside the timed region")
    condition_layers = layer_list(cast_cfg.cast.condition_layers, len(blocks))
    behavior_layers = layer_list(cast_cfg.cast.behavior_layers, len(blocks))
    cast_artifact = train_or_load_cast_artifact(
        cast_cfg, model, tokenizer, condition_layers, behavior_layers
    )
    condition_points, _ = condition_points_for_targets(
        cast_cfg, model, tokenizer, cast_artifact, condition_layers
    )
    cast_bundles = {
        language: move_cast(
            make_bundle(cast_cfg, cast_artifact, condition_points[language], behavior_layers),
            device,
            dtype,
        )
        for language in args.languages
    }
    gates = {
        language: runtime_gate(
            gate_result.gates[language], gate_result.thresholds[language], device
        )
        for language in args.languages
    }
    direction = unit_normalize(direction_cpu.to(device=device, dtype=dtype), dim=0)
    adasteer_bundle = move_adasteer(adasteer_cpu, device, dtype)
    batches = prepare_batches(gate_cfg, tokenizer, args, device)
    gc.collect()
    torch.cuda.empty_cache()

    generation_kwargs = {
        "do_sample": False,
        "use_cache": True,
        "min_new_tokens": int(args.new_tokens),
        "max_new_tokens": int(args.new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
    }
    first = batches[0]
    warmup_contexts = method_contexts(
        model,
        first,
        int(gate_cfg.layer),
        gates,
        direction,
        alpha_calibration.base,
        cast_bundles,
        adasteer_bundle,
    )
    for method in METHODS:
        print(f"[warm] {method}")
        for _ in range(args.warmups):
            timed_generate(model, first.inputs, warmup_contexts[method], generation_kwargs)

    timings = {
        batch.batch_id: {method: [] for method in METHODS} for batch in batches
    }
    gate_counts: dict[str, int] = {}
    for repeat in range(args.repeats):
        batch_order = list(batches)
        random.Random(stable_seed(args.seed, repeat, "batch-order")).shuffle(batch_order)
        for batch_number, batch in enumerate(batch_order, start=1):
            contexts = method_contexts(
                model,
                batch,
                int(gate_cfg.layer),
                gates,
                direction,
                alpha_calibration.base,
                cast_bundles,
                adasteer_bundle,
            )
            method_order = list(METHODS)
            random.Random(stable_seed(args.seed, repeat, batch.batch_id)).shuffle(method_order)
            measured = {}
            for method in method_order:
                elapsed_ms, harmful = timed_generate(
                    model, batch.inputs, contexts[method], generation_kwargs
                )
                timings[batch.batch_id][method].append(elapsed_ms)
                measured[method] = elapsed_ms
                if harmful is not None:
                    gate_counts[batch.batch_id] = harmful
            print(
                f"[run ] {repeat + 1}/{args.repeats} {batch_number:02d}/{len(batches)} "
                f"{batch.batch_id} "
                + " ".join(f"{method}={measured[method]:.1f}ms" for method in METHODS)
            )

    results = summarize(
        batches, timings, args.new_tokens, args.bootstrap_samples, args.seed
    )
    output = args.output or Path(
        f"artifacts/benchmarks/intervention_overhead/{args.model_config}.json"
    )
    output = absolute(output)
    payload = {
        "protocol": {
            "dataset": "PolyRefuse",
            "split": "test",
            "languages": list(args.languages),
            "subsets": ["harmful", "harmless"],
            "samples_per_cell": args.samples_per_cell,
            "batch_size": args.batch_size,
            "fixed_new_tokens": args.new_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "timed_region": "model.generate only; tokenization and preparation excluded",
            "parameters_preloaded_on_gpu": True,
        },
        "model": OmegaConf.to_container(gate_cfg.model, resolve=True),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "ours": {
            "budget_per_class": args.budget_per_class,
            "rank": args.rank,
            "layer": int(gate_cfg.layer),
            "alpha": alpha_calibration.base,
            "prefill_passes": 1,
            "thresholds": gate_result.thresholds,
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
        "results": results,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "language": batch.language,
                "subset": batch.subset,
                "prompt_ids": list(batch.prompt_ids),
                "prompt_tokens": list(batch.prompt_tokens),
                "ours_predicted_harmful": gate_counts[batch.batch_id],
                "timings_ms": timings[batch.batch_id],
            }
            for batch in batches
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print("\nmethod                    median ms  paired overhead (mean, 95% CI)  tokens/s")
    for row in results:
        low, high = row["paired_overhead_percent_ci95"]
        print(
            f"{row['method']:<25} {row['latency_ms_median']:>9.1f}  "
            f"{row['paired_overhead_percent_mean']:>7.2f}% "
            f"[{low:>7.2f}, {high:>7.2f}]  {row['tokens_per_second']:>8.1f}"
        )
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
