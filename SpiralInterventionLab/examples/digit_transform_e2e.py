from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ..bridge import ProviderControllerClient, ProviderPromptHintController
from ..controllers.factory import create_controller_provider, normalize_provider_name, provider_api_env_var
from ..runtime import (
    BaselineSuiteResult,
    EpisodeResult,
    HookedTransformerAdapter,
    HookedTransformerRuntimeState,
    HookedTransformerWorkerRuntime,
    JSONLStructuredLogger,
    resolve_text_codec,
    run_c1,
    run_minimal_baseline_suite,
)
from ..tasks import SpiralDigitCopyEnv, SpiralDigitTransformEnv

try:
    from transformer_lens import HookedTransformer
except Exception:  # pragma: no cover - optional dependency at import time
    HookedTransformer = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at import time
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


_DECODE_ALLOWLIST_CACHE: dict[tuple[int, str], tuple[int, ...]] = {}

DigitTaskEnv = SpiralDigitTransformEnv | SpiralDigitCopyEnv


def _default_surface_layers(model: Any) -> tuple[int, ...]:
    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 1))
    if n_layers <= 1:
        return (0,)
    middle = max(0, min(n_layers - 1, n_layers // 2))
    if n_layers <= 3:
        return tuple(sorted({middle}))
    return tuple(sorted({max(0, middle - 1), middle}))


def build_default_activation_surface_catalog(
    model: Any,
    *,
    worker_id: str = "os_0",
    layers: Sequence[int] | None = None,
    sites: Sequence[str] = ("resid_pre",),
    max_alpha: float = 0.18,
    max_ttl_steps: int = 2,
    norm_clip: float = 1.5,
    step_size: float = 0.12,
) -> list[dict[str, Any]]:
    selected_layers = tuple(int(layer) for layer in (layers or _default_surface_layers(model)))
    catalog: list[dict[str, Any]] = []
    for layer in selected_layers:
        for site in sites:
            catalog.append(
                {
                    "surface_id": f"s_{site}_l{layer}_last",
                    "target": {
                        "kind": "activation",
                        "worker": worker_id,
                        "site": site,
                        "layer": layer,
                        "token": {"mode": "last"},
                    },
                    "allow_ops": ["resid_add"],
                    "caps": {
                        "max_alpha": float(max_alpha),
                        "max_ttl_steps": int(max_ttl_steps),
                        "norm_clip": float(norm_clip),
                        "step_size": float(step_size),
                        "revertible_only": True,
                    },
                }
            )
    return catalog


def _model_vocab_size(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    d_vocab = getattr(cfg, "d_vocab", None)
    if isinstance(d_vocab, int) and d_vocab > 0:
        return int(d_vocab)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return int(vocab_size)
        try:
            return int(len(tokenizer))
        except Exception:
            pass
    raise ValueError("could not infer model vocabulary size for decode constraint compilation")


def _token_text_matches_constraint(token_text: str, constraint: str) -> bool:
    if constraint == "digits_only":
        text = str(token_text)
        return bool(text) and text.isdigit()
    if constraint == "whitespace_digits":
        stripped = str(token_text).strip()
        return bool(stripped) and stripped.isdigit()
    raise ValueError(f"unknown decode constraint '{constraint}'")


def build_allowed_token_ids_for_constraint(
    model: Any,
    *,
    codec: Any,
    decode_constraint: str | None,
) -> tuple[int, ...]:
    if decode_constraint in (None, "", "none"):
        return ()
    cache_key = (id(model), str(decode_constraint))
    cached = _DECODE_ALLOWLIST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vocab_size = _model_vocab_size(model)
    allowed = [
        token_id
        for token_id in range(vocab_size)
        if _token_text_matches_constraint(codec.decode([token_id]), str(decode_constraint))
    ]
    resolved = tuple(allowed)
    _DECODE_ALLOWLIST_CACHE[cache_key] = resolved
    return resolved


def build_hooked_transformer_worker_runtime(
    model: Any,
    task_env: SpiralDigitTransformEnv,
    *,
    seed: int,
    worker_id: str = "os_0",
    run_id: str = "run_digit_transform",
    episode_id: str = "episode_digit_transform",
    task_view_mode: str = "redacted",
    surface_catalog: Sequence[Mapping[str, Any]] | None = None,
    codec: Any | None = None,
    max_generated_tokens: int | None = None,
    max_edits_per_step: int = 1,
    max_edits_per_run: int = 4,
    max_total_alpha: float = 0.5,
    max_active_patch_slots: int = 1,
) -> HookedTransformerWorkerRuntime:
    runtime_state = HookedTransformerRuntimeState(model, seed=seed)
    adapter = HookedTransformerAdapter(model)
    resolved_codec = resolve_text_codec(model, codec)
    try:
        task_kwargs = dict(task_env.worker_runtime_kwargs())
    except RuntimeError:
        task_env.reset(seed)
        task_kwargs = dict(task_env.worker_runtime_kwargs())
    resolved_max_generated_tokens = int(
        max_generated_tokens if max_generated_tokens is not None else task_kwargs.pop("max_generated_tokens", 32)
    )
    resolved_min_generated_tokens = int(task_kwargs.pop("min_generated_tokens", 0))
    allowed_token_ids = task_kwargs.pop("allowed_token_ids", None)
    decode_constraint = task_kwargs.pop("decode_constraint", None)
    if allowed_token_ids is None and decode_constraint is not None:
        compiled_allowed_token_ids = build_allowed_token_ids_for_constraint(
            model,
            codec=resolved_codec,
            decode_constraint=str(decode_constraint),
        )
        allowed_token_ids = compiled_allowed_token_ids or None
    return HookedTransformerWorkerRuntime(
        runtime_state=runtime_state,
        adapter=adapter,
        model=model,
        codec=resolved_codec,
        surface_catalog=surface_catalog or build_default_activation_surface_catalog(model, worker_id=worker_id),
        run_id=run_id,
        episode_id=episode_id,
        worker_id=worker_id,
        task_view_mode=task_view_mode,
        max_generated_tokens=resolved_max_generated_tokens,
        min_generated_tokens=resolved_min_generated_tokens,
        max_edits_per_step=max_edits_per_step,
        max_edits_per_run=max_edits_per_run,
        max_total_alpha=max_total_alpha,
        max_active_patch_slots=max_active_patch_slots,
        allowed_token_ids=allowed_token_ids,
        trace_metadata={
            "paired_baseline": {
                "origin": "paired_baseline",
                "compatible": True,
                "tags": ["same_seed", "baseline_reference"],
            }
        },
        **task_kwargs,
    )


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype}")
    resolved = getattr(torch, dtype)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype}")
    return resolved


def _infer_tlens_model_ref(model_ref: str, hf_model: Any) -> str:
    if not Path(str(model_ref)).exists():
        return model_ref

    config = getattr(hf_model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type != "llama":
        return model_ref

    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)

    if hidden_size == 2048 and num_layers == 16 and vocab_size == 128256:
        return "meta-llama/Llama-3.2-1B"
    if hidden_size == 3072 and num_layers == 28 and vocab_size == 128256:
        return "meta-llama/Llama-3.2-3B"
    if hidden_size == 4096 and num_layers == 32 and vocab_size == 128256:
        return "meta-llama/Llama-3.1-8B"

    return model_ref


def _load_local_hooked_transformer_from_hf(
    *,
    model_ref: str,
    hf_model: Any,
    tokenizer: Any,
    device: str | None,
    dtype: str | torch.dtype,
    first_n_layers: int | None,
    move_to_device: bool,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    import transformer_lens.loading_from_pretrained as tl_loading

    torch_dtype = _resolve_torch_dtype(dtype)
    hf_cfg = hf_model.config.to_dict()
    tlens_model_ref = _infer_tlens_model_ref(model_ref, hf_model)
    cfg = tl_loading.get_pretrained_model_config(
        tlens_model_ref,
        hf_cfg=hf_cfg,
        device=device,
        dtype=torch_dtype,
        first_n_layers=first_n_layers,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    state_dict = tl_loading.get_pretrained_state_dict(
        tlens_model_ref,
        cfg,
        hf_model=hf_model,
        dtype=torch_dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = HookedTransformer(
        cfg,
        tokenizer,
        move_to_device=False,
    )
    model.load_and_process_state_dict(state_dict)
    if move_to_device:
        model.move_model_modules_to_device()
    return model


def load_worker_model(
    model_name: str,
    *,
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    device: str | None = None,
    dtype: str = "float32",
    first_n_layers: int | None = None,
    move_to_device: bool = True,
    hf_offline: bool = False,
    trust_remote_code: bool = False,
) -> Any:
    if HookedTransformer is None:
        raise ImportError("transformer_lens is not installed; install the 'tlens' extra to run the end-to-end example")
    resolved_model_ref = str(model_path or model_name)
    resolved_tokenizer_ref = str(tokenizer_path or resolved_model_ref)
    local_path = Path(resolved_model_ref).expanduser()
    use_local_hf = model_path is not None or local_path.exists()
    local_files_only = bool(hf_offline or use_local_hf)

    if use_local_hf:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is not installed; install the 'hf' extra to load a local Hugging Face worker model")
        hf_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_ref,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_ref,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if not model_name:
            raise ValueError(
                "A TransformerLens-compatible --worker-model name is required when loading a local HF worker path"
            )
        return _load_local_hooked_transformer_from_hf(
            model_ref=resolved_model_ref,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            first_n_layers=first_n_layers,
            move_to_device=move_to_device,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        first_n_layers=first_n_layers,
        move_to_device=move_to_device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def _logger_factory(log_dir: str | Path | None):
    if log_dir is None:
        return None
    base = Path(log_dir)

    def factory(name: str):
        return JSONLStructuredLogger(base / f"{name}.jsonl")

    return factory


def _write_summary_artifact(log_dir: str | Path | None, filename: str, payload: Mapping[str, Any]) -> None:
    if log_dir is None:
        return
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / filename).write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def create_task_env(task_name: str) -> DigitTaskEnv:
    normalized = str(task_name).strip().lower().replace("-", "_")
    if normalized in {"digit_transform", "transform"}:
        return SpiralDigitTransformEnv()
    if normalized in {"digit_copy", "copy", "echo"}:
        return SpiralDigitCopyEnv()
    raise ValueError(f"unknown task '{task_name}'")


@dataclass(frozen=True)
class DigitTransformExperimentResult:
    seed: int
    task_id: str
    worker_model_name: str
    controller_provider: str
    controller_model_name: str
    surface_ids: tuple[str, ...]
    suite: BaselineSuiteResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_mode": "minimal_baseline_suite",
            "seed": self.seed,
            "task_id": self.task_id,
            "worker_model_name": self.worker_model_name,
            "controller_provider": self.controller_provider,
            "controller_model_name": self.controller_model_name,
            "surface_ids": list(self.surface_ids),
            "paired_trace_id": self.suite.paired_trace_id,
            "b0": asdict(self.suite.b0),
            "b1": None if self.suite.b1 is None else asdict(self.suite.b1),
            "c1": asdict(self.suite.c1),
        }


@dataclass(frozen=True)
class DigitTransformC1OnlyExperimentResult:
    seed: int
    task_id: str
    worker_model_name: str
    controller_provider: str
    controller_model_name: str
    surface_ids: tuple[str, ...]
    c1: EpisodeResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_mode": "c1_only",
            "seed": self.seed,
            "task_id": self.task_id,
            "worker_model_name": self.worker_model_name,
            "controller_provider": self.controller_provider,
            "controller_model_name": self.controller_model_name,
            "surface_ids": list(self.surface_ids),
            "paired_trace_id": None,
            "b0": None,
            "b1": None,
            "c1": asdict(self.c1),
        }


@dataclass(frozen=True)
class DigitTransformSweepResult:
    seeds: tuple[int, ...]
    runs: tuple[DigitTransformExperimentResult, ...]

    def summary(self) -> dict[str, Any]:
        run_count = len(self.runs)
        if run_count == 0:
            return {
                "num_runs": 0,
                "b0_mean_score": 0.0,
                "b1_mean_score": 0.0,
                "c1_mean_score": 0.0,
                "delta_c1_over_b0_mean": 0.0,
                "delta_c1_over_b1_mean": 0.0,
                "b0_successes": 0,
                "b1_successes": 0,
                "c1_successes": 0,
            }

        b0_scores = [float(run.suite.b0.score) for run in self.runs]
        b1_scores = [float(run.suite.b1.score if run.suite.b1 is not None else 0.0) for run in self.runs]
        c1_scores = [float(run.suite.c1.score) for run in self.runs]

        def _mean(values: Iterable[float]) -> float:
            values = tuple(values)
            return sum(values) / len(values)

        return {
            "num_runs": run_count,
            "b0_mean_score": _mean(b0_scores),
            "b1_mean_score": _mean(b1_scores),
            "c1_mean_score": _mean(c1_scores),
            "delta_c1_over_b0_mean": _mean(c1 - b0 for c1, b0 in zip(c1_scores, b0_scores)),
            "delta_c1_over_b1_mean": _mean(c1 - b1 for c1, b1 in zip(c1_scores, b1_scores)),
            "b0_successes": sum(1 for score in b0_scores if score >= 1.0),
            "b1_successes": sum(1 for score in b1_scores if score >= 1.0),
            "c1_successes": sum(1 for score in c1_scores if score >= 1.0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }


def run_digit_transform_experiment(
    *,
    provider_name: str,
    controller_model_name: str,
    worker_model_name: str,
    seed: int = 0,
    controller_api_key: str | None = None,
    worker_model: Any | None = None,
    task_env: DigitTaskEnv | None = None,
    include_prompt_baseline: bool = True,
    controller_prompt_asset: str = "controller_v01.txt",
    hint_prompt_asset: str = "prompt_hint_v01.txt",
    task_view_mode: str = "redacted",
    log_dir: str | Path | None = None,
    codec: Any | None = None,
    surface_catalog: Sequence[Mapping[str, Any]] | None = None,
    worker_model_path: str | Path | None = None,
    worker_tokenizer_path: str | Path | None = None,
    worker_device: str | None = None,
    worker_dtype: str = "float32",
    worker_first_n_layers: int | None = None,
    worker_hf_offline: bool = False,
    worker_trust_remote_code: bool = False,
) -> DigitTransformExperimentResult:
    env = task_env or SpiralDigitTransformEnv()
    model = worker_model or load_worker_model(
        worker_model_name,
        model_path=worker_model_path,
        tokenizer_path=worker_tokenizer_path,
        device=worker_device,
        dtype=worker_dtype,
        first_n_layers=worker_first_n_layers,
        hf_offline=worker_hf_offline,
        trust_remote_code=worker_trust_remote_code,
    )
    provider = create_controller_provider(
        provider_name,
        model=controller_model_name,
        api_key=controller_api_key,
    )
    c1_controller = ProviderControllerClient(provider, prompt_asset=controller_prompt_asset, max_attempts=3)
    b1_controller = (
        ProviderPromptHintController(provider, prompt_asset=hint_prompt_asset)
        if include_prompt_baseline
        else None
    )

    def make_worker_runtime() -> HookedTransformerWorkerRuntime:
        return build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=seed,
            task_view_mode=task_view_mode,
            surface_catalog=surface_catalog,
            codec=codec,
        )

    suite = run_minimal_baseline_suite(
        env,
        make_worker_runtime=make_worker_runtime,
        c1_controller=c1_controller,
        b1_controller=b1_controller,
        logger_factory=_logger_factory(log_dir),
    )

    worker = make_worker_runtime()
    surface_ids = tuple(surface["surface_id"] for surface in worker._surface_catalog_raw)
    result = DigitTransformExperimentResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        controller_provider=normalize_provider_name(provider_name),
        controller_model_name=controller_model_name,
        surface_ids=surface_ids,
        suite=suite,
    )
    _write_summary_artifact(log_dir, "experiment_summary.json", result.to_dict())
    return result


def run_digit_transform_c1_only_experiment(
    *,
    provider_name: str,
    controller_model_name: str,
    worker_model_name: str,
    seed: int = 0,
    controller_api_key: str | None = None,
    worker_model: Any | None = None,
    task_env: DigitTaskEnv | None = None,
    controller_prompt_asset: str = "controller_v01.txt",
    task_view_mode: str = "redacted",
    log_dir: str | Path | None = None,
    codec: Any | None = None,
    surface_catalog: Sequence[Mapping[str, Any]] | None = None,
    worker_model_path: str | Path | None = None,
    worker_tokenizer_path: str | Path | None = None,
    worker_device: str | None = None,
    worker_dtype: str = "float32",
    worker_first_n_layers: int | None = None,
    worker_hf_offline: bool = False,
    worker_trust_remote_code: bool = False,
) -> DigitTransformC1OnlyExperimentResult:
    env = task_env or SpiralDigitTransformEnv()
    model = worker_model or load_worker_model(
        worker_model_name,
        model_path=worker_model_path,
        tokenizer_path=worker_tokenizer_path,
        device=worker_device,
        dtype=worker_dtype,
        first_n_layers=worker_first_n_layers,
        hf_offline=worker_hf_offline,
        trust_remote_code=worker_trust_remote_code,
    )
    provider = create_controller_provider(
        provider_name,
        model=controller_model_name,
        api_key=controller_api_key,
    )
    c1_controller = ProviderControllerClient(provider, prompt_asset=controller_prompt_asset, max_attempts=3)
    worker = build_hooked_transformer_worker_runtime(
        model,
        env,
        seed=seed,
        task_view_mode=task_view_mode,
        surface_catalog=surface_catalog,
        codec=codec,
    )
    logger_factory = _logger_factory(log_dir)
    c1 = run_c1(
        env,
        worker,
        c1_controller,
        logger=None if logger_factory is None else logger_factory("c1"),
    )
    surface_ids = tuple(surface["surface_id"] for surface in worker._surface_catalog_raw)
    result = DigitTransformC1OnlyExperimentResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        controller_provider=normalize_provider_name(provider_name),
        controller_model_name=controller_model_name,
        surface_ids=surface_ids,
        c1=c1,
    )
    _write_summary_artifact(log_dir, "experiment_summary.json", result.to_dict())
    return result


def run_digit_transform_sweep(
    *,
    provider_name: str,
    controller_model_name: str,
    worker_model_name: str,
    seeds: Sequence[int],
    controller_api_key: str | None = None,
    worker_model: Any | None = None,
    task_env_factory: Callable[[], DigitTaskEnv] | None = None,
    include_prompt_baseline: bool = True,
    controller_prompt_asset: str = "controller_v01.txt",
    hint_prompt_asset: str = "prompt_hint_v01.txt",
    task_view_mode: str = "redacted",
    log_dir: str | Path | None = None,
    codec: Any | None = None,
    surface_catalog: Sequence[Mapping[str, Any]] | None = None,
    worker_model_path: str | Path | None = None,
    worker_tokenizer_path: str | Path | None = None,
    worker_device: str | None = None,
    worker_dtype: str = "float32",
    worker_first_n_layers: int | None = None,
    worker_hf_offline: bool = False,
    worker_trust_remote_code: bool = False,
) -> DigitTransformSweepResult:
    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds:
        raise ValueError("run_digit_transform_sweep requires at least one seed")

    shared_model = worker_model or load_worker_model(
        worker_model_name,
        model_path=worker_model_path,
        tokenizer_path=worker_tokenizer_path,
        device=worker_device,
        dtype=worker_dtype,
        first_n_layers=worker_first_n_layers,
        hf_offline=worker_hf_offline,
        trust_remote_code=worker_trust_remote_code,
    )
    env_factory = task_env_factory or SpiralDigitTransformEnv
    runs: list[DigitTransformExperimentResult] = []
    for seed in resolved_seeds:
        seed_log_dir = None
        if log_dir is not None:
            seed_log_dir = Path(log_dir) / f"seed_{seed}"
        runs.append(
            run_digit_transform_experiment(
                provider_name=provider_name,
                controller_model_name=controller_model_name,
                worker_model_name=worker_model_name,
                seed=seed,
                controller_api_key=controller_api_key,
                worker_model=shared_model,
                task_env=env_factory(),
                include_prompt_baseline=include_prompt_baseline,
                controller_prompt_asset=controller_prompt_asset,
                hint_prompt_asset=hint_prompt_asset,
                task_view_mode=task_view_mode,
                log_dir=seed_log_dir,
                codec=codec,
                surface_catalog=surface_catalog,
                worker_model_path=worker_model_path,
                worker_tokenizer_path=worker_tokenizer_path,
                worker_device=worker_device,
                worker_dtype=worker_dtype,
                worker_first_n_layers=worker_first_n_layers,
                worker_hf_offline=worker_hf_offline,
                worker_trust_remote_code=worker_trust_remote_code,
            )
        )
    result = DigitTransformSweepResult(seeds=resolved_seeds, runs=tuple(runs))
    _write_summary_artifact(log_dir, "sweep_summary.json", result.to_dict())
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first end-to-end Spiral digit-transform experiment.")
    parser.add_argument("--provider", required=True, help="Controller provider: openai, anthropic, mistral, or google")
    parser.add_argument("--controller-model", required=True, help="Black-box controller model name")
    parser.add_argument("--worker-model", default="gpt2-small", help="HookedTransformer worker model name or alias")
    parser.add_argument(
        "--task",
        default="digit_transform",
        choices=["digit_transform", "digit_copy"],
        help="Task environment to run",
    )
    parser.add_argument("--worker-model-path", default=None, help="Optional local Hugging Face model directory for offline worker loading")
    parser.add_argument("--worker-tokenizer-path", default=None, help="Optional local tokenizer directory; defaults to --worker-model-path")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--num-seeds", type=int, default=1, help="Run a consecutive multi-seed sweep starting at --seed")
    parser.add_argument("--controller-api-key", default=None, help="Optional API key override for the controller provider")
    parser.add_argument("--no-b1", action="store_true", help="Disable the prompt-hint baseline; B1 runs by default")
    parser.add_argument("--c1-only", action="store_true", help="Run only the C1 controller loop without B0/B1 baselines")
    parser.add_argument("--task-view-mode", default="redacted", choices=["redacted", "full"], help="Controller task view mode")
    parser.add_argument("--log-dir", default=None, help="Optional directory for baseline JSONL logs")
    parser.add_argument("--worker-device", default=None, help="Optional worker device override, e.g. mps or cpu")
    parser.add_argument("--worker-dtype", default="float32", help="HookedTransformer dtype, e.g. float32 or float16")
    parser.add_argument("--worker-first-n-layers", type=int, default=None, help="Optionally truncate the worker model depth")
    parser.add_argument("--worker-hf-offline", action="store_true", help="Force local_files_only when loading the worker model/tokenizer")
    parser.add_argument("--worker-trust-remote-code", action="store_true", help="Pass trust_remote_code through to HF / TransformerLens loading")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be >= 1")
    if args.c1_only and args.num_seeds != 1:
        parser.error("--c1-only currently supports only --num-seeds 1")
    if args.controller_api_key is None:
        env_var = provider_api_env_var(args.provider)
        if os.getenv(env_var) is None:
            parser.error(f"{env_var} is not set and --controller-api-key was not provided")

    if args.c1_only:
        result = run_digit_transform_c1_only_experiment(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seed=args.seed,
            controller_api_key=args.controller_api_key,
            task_env=create_task_env(args.task),
            task_view_mode=args.task_view_mode,
            log_dir=args.log_dir,
            worker_model_path=args.worker_model_path,
            worker_tokenizer_path=args.worker_tokenizer_path,
            worker_device=args.worker_device,
            worker_dtype=args.worker_dtype,
            worker_first_n_layers=args.worker_first_n_layers,
            worker_hf_offline=args.worker_hf_offline,
            worker_trust_remote_code=args.worker_trust_remote_code,
        )
        payload = result.to_dict()
    elif args.num_seeds == 1:
        result = run_digit_transform_experiment(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seed=args.seed,
            controller_api_key=args.controller_api_key,
            task_env=create_task_env(args.task),
            include_prompt_baseline=not args.no_b1,
            task_view_mode=args.task_view_mode,
            log_dir=args.log_dir,
            worker_model_path=args.worker_model_path,
            worker_tokenizer_path=args.worker_tokenizer_path,
            worker_device=args.worker_device,
            worker_dtype=args.worker_dtype,
            worker_first_n_layers=args.worker_first_n_layers,
            worker_hf_offline=args.worker_hf_offline,
            worker_trust_remote_code=args.worker_trust_remote_code,
        )
        payload = result.to_dict()
    else:
        sweep = run_digit_transform_sweep(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seeds=tuple(range(args.seed, args.seed + args.num_seeds)),
            controller_api_key=args.controller_api_key,
            task_env_factory=lambda: create_task_env(args.task),
            include_prompt_baseline=not args.no_b1,
            task_view_mode=args.task_view_mode,
            log_dir=args.log_dir,
            worker_model_path=args.worker_model_path,
            worker_tokenizer_path=args.worker_tokenizer_path,
            worker_device=args.worker_device,
            worker_dtype=args.worker_dtype,
            worker_first_n_layers=args.worker_first_n_layers,
            worker_hf_offline=args.worker_hf_offline,
            worker_trust_remote_code=args.worker_trust_remote_code,
        )
        payload = sweep.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
