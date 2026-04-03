from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..bridge import ProviderControllerClient, ProviderPromptHintController
from ..controllers.factory import create_controller_provider, normalize_provider_name, provider_api_env_var
from ..runtime import (
    BaselineSuiteResult,
    HookedTransformerAdapter,
    HookedTransformerRuntimeState,
    HookedTransformerWorkerRuntime,
    JSONLStructuredLogger,
    run_minimal_baseline_suite,
)
from ..tasks import SpiralDigitTransformEnv

try:
    from transformer_lens import HookedTransformer
except Exception:  # pragma: no cover - optional dependency at import time
    HookedTransformer = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at import time
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


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
                        "revertible_only": True,
                    },
                }
            )
    return catalog


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
    task_kwargs = dict(task_env.worker_runtime_kwargs())
    resolved_max_generated_tokens = int(max_generated_tokens or task_kwargs.pop("max_generated_tokens", 32))
    return HookedTransformerWorkerRuntime(
        runtime_state=runtime_state,
        adapter=adapter,
        model=model,
        codec=codec,
        surface_catalog=surface_catalog or build_default_activation_surface_catalog(model, worker_id=worker_id),
        run_id=run_id,
        episode_id=episode_id,
        worker_id=worker_id,
        task_view_mode=task_view_mode,
        max_generated_tokens=resolved_max_generated_tokens,
        max_edits_per_step=max_edits_per_step,
        max_edits_per_run=max_edits_per_run,
        max_total_alpha=max_total_alpha,
        max_active_patch_slots=max_active_patch_slots,
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
    cfg = tl_loading.get_pretrained_model_config(
        model_ref,
        hf_cfg=hf_cfg,
        device=device,
        dtype=torch_dtype,
        first_n_layers=first_n_layers,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    state_dict = tl_loading.get_pretrained_state_dict(
        model_ref,
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


def run_digit_transform_experiment(
    *,
    provider_name: str,
    controller_model_name: str,
    worker_model_name: str,
    seed: int = 0,
    controller_api_key: str | None = None,
    worker_model: Any | None = None,
    task_env: SpiralDigitTransformEnv | None = None,
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
    return DigitTransformExperimentResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        controller_provider=normalize_provider_name(provider_name),
        controller_model_name=controller_model_name,
        surface_ids=surface_ids,
        suite=suite,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first end-to-end Spiral digit-transform experiment.")
    parser.add_argument("--provider", required=True, help="Controller provider: openai, anthropic, mistral, or google")
    parser.add_argument("--controller-model", required=True, help="Black-box controller model name")
    parser.add_argument("--worker-model", default="gpt2-small", help="HookedTransformer worker model name or alias")
    parser.add_argument("--worker-model-path", default=None, help="Optional local Hugging Face model directory for offline worker loading")
    parser.add_argument("--worker-tokenizer-path", default=None, help="Optional local tokenizer directory; defaults to --worker-model-path")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--controller-api-key", default=None, help="Optional API key override for the controller provider")
    parser.add_argument("--no-b1", action="store_true", help="Disable the prompt-hint baseline; B1 runs by default")
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
    if args.controller_api_key is None:
        env_var = provider_api_env_var(args.provider)
        if os.getenv(env_var) is None:
            parser.error(f"{env_var} is not set and --controller-api-key was not provided")

    result = run_digit_transform_experiment(
        provider_name=args.provider,
        controller_model_name=args.controller_model,
        worker_model_name=args.worker_model,
        seed=args.seed,
        controller_api_key=args.controller_api_key,
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
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
