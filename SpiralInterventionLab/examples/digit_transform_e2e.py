from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
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
    InMemoryStructuredLogger,
    JSONLStructuredLogger,
    ReadoutAnalyzer,
    ReadoutSidecarAnalyzer,
    StepContext,
    build_heuristic_readout_analyzer,
    build_heuristic_readout_sidecar_analyzer,
    resolve_text_codec,
    run_c1,
    run_episode,
    run_minimal_baseline_suite,
)
from ..tasks import (
    MiniLMSemanticCritic,
    SpiralConstrainedRewriteEnv,
    SpiralDigitCopyEnv,
    SpiralDigitTransformEnv,
    SpiralEntailmentReasoningEnv,
    SpiralSentenceOrderingEnv,
    SpiralStructuredSummaryEnv,
)

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
_WORKER_MPS_MODES = ("auto", "conservative")
_CONTROLLER_REFLECTION_MODES = ("off", "structured")
_SEMANTIC_CRITIC_MODES = ("off", "minilm")
_READOUT_ANALYZER_MODES = ("off", "heuristic")
_READOUT_ANALYZER_RERANK_MODES = ("off", "shadow", "apply")
_WORKER_DECODER_CONTROL_MODES = (
    "off",
    "loop_aware",
    "loop_aware_prune",
    "loop_aware_constraint",
    "loop_aware_entity_recall",
    "logit_bias_entity_soft",
)

ExperimentTaskEnv = (
    SpiralDigitTransformEnv
    | SpiralDigitCopyEnv
    | SpiralSentenceOrderingEnv
    | SpiralEntailmentReasoningEnv
    | SpiralConstrainedRewriteEnv
    | SpiralStructuredSummaryEnv
)


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
    if selected_layers:
        shot_layer = min(selected_layers)
        for site in sites:
            if site != "resid_pre":
                continue
            catalog.append(
                {
                    "surface_id": f"s_{site}_l{shot_layer}_prev",
                    "target": {
                        "kind": "activation",
                        "worker": worker_id,
                        "site": site,
                        "layer": shot_layer,
                        "token": {"mode": "index", "value": -2},
                    },
                    "allow_ops": ["resid_add"],
                    "caps": {
                        "max_alpha": float(min(max_alpha, 0.12)),
                        "max_ttl_steps": 1,
                        "norm_clip": float(norm_clip),
                        "step_size": float(min(step_size, 0.08)),
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
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
) -> HookedTransformerWorkerRuntime:
    runtime_state = HookedTransformerRuntimeState(model, seed=seed)
    adapter = HookedTransformerAdapter(model)
    resolved_codec = resolve_text_codec(model, codec)
    try:
        task_kwargs = dict(task_env.worker_runtime_kwargs())
    except RuntimeError:
        task_env.reset(seed)
        task_kwargs = dict(task_env.worker_runtime_kwargs())
    task_max_generated_tokens = task_kwargs.pop("max_generated_tokens", None)
    resolved_max_generated_tokens = int(
        max_generated_tokens if max_generated_tokens is not None else (32 if task_max_generated_tokens is None else task_max_generated_tokens)
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
        controller_reflection_mode=controller_reflection_mode,
        controller_memory_window=controller_memory_window,
        decoder_control_mode=worker_decoder_control_mode,
        max_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        max_loop_rescue_alpha=worker_loop_rescue_total_alpha,
        max_loop_rescue_edit_cost=worker_loop_rescue_total_edit_cost,
        allowed_token_ids=allowed_token_ids,
        readout_sidecar_analyzer=readout_sidecar_analyzer,
        readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
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


def _normalize_worker_mps_mode(mps_mode: str) -> str:
    normalized = str(mps_mode).strip().lower().replace("-", "_")
    if normalized not in _WORKER_MPS_MODES:
        raise ValueError(f"Unsupported worker MPS mode '{mps_mode}'; expected one of {_WORKER_MPS_MODES}")
    return normalized


def _device_targets_mps(device: str | torch.device | None) -> bool:
    if device is None:
        try:
            return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:
            return False
    try:
        return torch.device(device).type == "mps"
    except Exception:
        return str(device).strip().lower() == "mps"


def _resolve_worker_device(
    *,
    device: str | None,
    mps_mode: str,
) -> str | None:
    normalized_mps_mode = _normalize_worker_mps_mode(mps_mode)
    if device is not None:
        return str(torch.device(device))
    if normalized_mps_mode == "conservative" and _device_targets_mps(None):
        return "mps"
    return None


def _configure_torch_default_device_for_worker(
    *,
    target_device: str | torch.device | None,
    mps_mode: str,
) -> bool:
    normalized_mps_mode = _normalize_worker_mps_mode(mps_mode)
    if normalized_mps_mode != "conservative" or not _device_targets_mps(target_device):
        return False
    if not hasattr(torch, "set_default_device") or not hasattr(torch, "get_default_device"):
        return False
    try:
        current_default = torch.device(torch.get_default_device())
    except Exception:
        return False
    if current_default.type != "mps":
        return False
    torch.set_default_device("cpu")
    return True


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
    mps_mode: str = "auto",
) -> Any:
    if HookedTransformer is None:
        raise ImportError("transformer_lens is not installed; install the 'tlens' extra to run the end-to-end example")
    resolved_device = _resolve_worker_device(device=device, mps_mode=mps_mode)
    _configure_torch_default_device_for_worker(target_device=resolved_device, mps_mode=mps_mode)
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
            device=resolved_device,
            dtype=dtype,
            first_n_layers=first_n_layers,
            move_to_device=move_to_device,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    return HookedTransformer.from_pretrained(
        model_name,
        device=resolved_device,
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


@dataclass
class _StructuredLoggerMux:
    loggers: tuple[Any, ...]

    def log(self, event: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.log(dict(event))


def _write_summary_artifact(log_dir: str | Path | None, filename: str, payload: Mapping[str, Any]) -> None:
    if log_dir is None:
        return
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / filename).write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def create_semantic_critic(
    mode: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Any | None:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "off":
        return None
    if normalized == "minilm":
        return MiniLMSemanticCritic(model_name=model_name, device=device)
    raise ValueError(f"unknown semantic critic mode '{mode}'")


def create_readout_analyzer(mode: str) -> ReadoutAnalyzer | None:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "off":
        return None
    if normalized == "heuristic":
        return build_heuristic_readout_analyzer()
    raise ValueError(f"unknown readout analyzer mode '{mode}'")


def create_readout_sidecar_analyzer(mode: str) -> ReadoutSidecarAnalyzer | None:
    return create_readout_analyzer(mode)


class _FrontierReplayControllerClient:
    def __init__(self, *, replay_mode: str = "frontier_apply"):
        self.replay_mode = str(replay_mode)
        self.calls = 0
        self._last_trace: dict[str, Any] | None = None

    def latest_trace(self) -> Mapping[str, Any] | None:
        return self._last_trace

    @staticmethod
    def _candidate_compile_rank(candidate: Mapping[str, Any]) -> tuple[int, int]:
        has_source = int(isinstance(candidate.get("source"), Mapping))
        has_op = int(isinstance(candidate.get("op"), Mapping))
        has_target = int(
            isinstance(candidate.get("target"), Mapping)
            or candidate.get("surface_id") not in (None, "")
        )
        has_budget = int(isinstance(candidate.get("budget"), Mapping))
        return (has_source + has_op + has_target + has_budget, has_source + has_op)

    def invoke(self, packet: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        suggested_bundle_key = strategy_hints.get(
            "readout_analyzer_suggested_bundle_key",
            strategy_hints.get("readout_sidecar_suggested_bundle_key"),
        )
        frontier_bundle_key = strategy_hints.get(
            "gate_report_frontier_bundle_key",
            strategy_hints.get("selected_bundle_key"),
        )
        candidates: list[dict[str, Any]] = []
        for key in ("kv_candidate_edits", "shot_candidate_edits", "kv_retry_candidate_edits"):
            raw_items = strategy_hints.get(key)
            if not isinstance(raw_items, SequenceABC) or isinstance(raw_items, (str, bytes, bytearray)):
                continue
            candidates.extend(dict(item) for item in raw_items if isinstance(item, Mapping))

        command: dict[str, Any] = {"version": "0.1", "decision": "noop"}
        selected_bundle_key: str | None = None
        if self.calls == 1:
            target_bundle_key = frontier_bundle_key or suggested_bundle_key
            matching = [
                dict(item)
                for item in candidates
                if isinstance(item, Mapping) and str(item.get("bundle_key", "") or "") == str(target_bundle_key or "")
            ]
            if matching:
                matching.sort(key=self._candidate_compile_rank, reverse=True)
                selected_bundle_key = str(target_bundle_key)
                selected_edit = {
                    key: matching[0][key]
                    for key in ("id", "target", "source", "op", "budget")
                    if key in matching[0]
                }
                if "id" not in selected_edit:
                    selected_edit["id"] = f"replay_select_{self.calls}"
                if "target" not in selected_edit and matching[0].get("surface_id") not in (None, ""):
                    selected_edit["target"] = {"surface_id": matching[0].get("surface_id")}
                command = {
                    "version": "0.1",
                    "decision": "apply",
                    "edits": [selected_edit],
                    "meta": {
                        "hypothesis": "forced_readout_escape_frontier_replay",
                        "micro_rationale": "inspect readout_escape logging contract on a forced frontier candidate",
                        "focus_term": matching[0].get("focus_feature"),
                        "surface_family_key": matching[0].get("candidate_family"),
                        "evidence_bullets": [
                            f"frontier_bundle={target_bundle_key}",
                            f"suggested_bundle={suggested_bundle_key}",
                        ],
                    },
                }

        self._last_trace = {
            "provider": "replay",
            "model": "frontier-replay-controller",
            "observation": {
                "step": packet.get("step"),
                "control_phase_hint": packet.get("control_phase_hint"),
                "sidecar_suggested_bundle_key": suggested_bundle_key,
                "gate_report_frontier_bundle_key": frontier_bundle_key,
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "replay",
                    "model": "frontier-replay-controller",
                    "latency_ms": 0.0,
                    "parse_ok": True,
                    "response_text": json.dumps(command, ensure_ascii=False, sort_keys=True),
                }
            ],
            "decision": {
                "decision": command.get("decision"),
                "selected_bundle_key": selected_bundle_key,
                "replay_mode": self.replay_mode,
            },
            "success": True,
        }
        return command


def create_task_env(task_name: str, *, semantic_critic: Any | None = None) -> ExperimentTaskEnv:
    normalized = str(task_name).strip().lower().replace("-", "_")
    if normalized in {"digit_transform", "transform"}:
        return SpiralDigitTransformEnv()
    if normalized in {"digit_copy", "copy", "echo"}:
        return SpiralDigitCopyEnv()
    if normalized in {"sentence_ordering", "story_ordering", "ordering"}:
        return SpiralSentenceOrderingEnv()
    if normalized in {"entailment_reasoning", "entailment", "nli"}:
        return SpiralEntailmentReasoningEnv()
    if normalized in {"constrained_rewrite", "rewrite"}:
        return SpiralConstrainedRewriteEnv(semantic_critic=semantic_critic)
    if normalized in {"structured_summary", "summary"}:
        return SpiralStructuredSummaryEnv(semantic_critic=semantic_critic)
    raise ValueError(f"unknown task '{task_name}'")


@dataclass(frozen=True)
class DigitTransformExperimentResult:
    seed: int
    task_id: str
    worker_model_name: str
    controller_provider: str
    controller_model_name: str
    controller_reflection_mode: str
    semantic_critic_mode: str
    semantic_critic_model_name: str | None
    worker_decoder_control_mode: str
    worker_loop_rescue_edits_per_run: int
    worker_loop_rescue_total_alpha: float
    worker_loop_rescue_total_edit_cost: float
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
            "controller_reflection_mode": self.controller_reflection_mode,
            "semantic_critic_mode": self.semantic_critic_mode,
            "semantic_critic_model_name": self.semantic_critic_model_name,
            "worker_decoder_control_mode": self.worker_decoder_control_mode,
            "worker_loop_rescue_edits_per_run": self.worker_loop_rescue_edits_per_run,
            "worker_loop_rescue_total_alpha": self.worker_loop_rescue_total_alpha,
            "worker_loop_rescue_total_edit_cost": self.worker_loop_rescue_total_edit_cost,
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
    controller_reflection_mode: str
    semantic_critic_mode: str
    semantic_critic_model_name: str | None
    worker_decoder_control_mode: str
    worker_loop_rescue_edits_per_run: int
    worker_loop_rescue_total_alpha: float
    worker_loop_rescue_total_edit_cost: float
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
            "controller_reflection_mode": self.controller_reflection_mode,
            "semantic_critic_mode": self.semantic_critic_mode,
            "semantic_critic_model_name": self.semantic_critic_model_name,
            "worker_decoder_control_mode": self.worker_decoder_control_mode,
            "worker_loop_rescue_edits_per_run": self.worker_loop_rescue_edits_per_run,
            "worker_loop_rescue_total_alpha": self.worker_loop_rescue_total_alpha,
            "worker_loop_rescue_total_edit_cost": self.worker_loop_rescue_total_edit_cost,
            "surface_ids": list(self.surface_ids),
            "paired_trace_id": None,
            "b0": None,
            "b1": None,
            "c1": asdict(self.c1),
        }


@dataclass(frozen=True)
class ShotModeProbeHarnessResult:
    seed: int
    task_id: str
    worker_model_name: str
    worker_decoder_control_mode: str
    shot_mode_reached: bool
    shot_mode_step: int | None
    probe_mode: str
    steps_executed: int
    prompt: str
    final_output: str
    final_task_feedback: dict[str, Any]
    observation_summary: dict[str, Any]
    probe_round_count: int
    probe_results: tuple[dict[str, Any], ...]
    surface_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_mode": "shot_mode_probe_harness",
            "seed": self.seed,
            "task_id": self.task_id,
            "worker_model_name": self.worker_model_name,
            "worker_decoder_control_mode": self.worker_decoder_control_mode,
            "shot_mode_reached": self.shot_mode_reached,
            "shot_mode_step": self.shot_mode_step,
            "probe_mode": self.probe_mode,
            "steps_executed": self.steps_executed,
            "prompt": self.prompt,
            "final_output": self.final_output,
            "final_task_feedback": dict(self.final_task_feedback),
            "observation_summary": dict(self.observation_summary),
            "probe_round_count": int(self.probe_round_count),
            "probe_results": [dict(item) for item in self.probe_results],
            "surface_ids": list(self.surface_ids),
        }


@dataclass(frozen=True)
class ReadoutEscapeReplayHarnessResult:
    seed: int
    task_id: str
    worker_model_name: str
    replay_mode: str
    packet_mode: str
    prompt: str
    episode: EpisodeResult
    readout_escape_seen: bool
    controller_selection_event_count: int
    nonnull_gate_frontier_count: int
    nonnull_controller_selected_count: int
    sidecar_suggested_bundle_key: str | None
    gate_report_frontier_bundle_key: str | None
    controller_selected_bundle_key: str | None
    controller_selection_source: str | None
    controller_rejected_signals: tuple[str, ...]
    forced_base_bundle_key: str
    forced_challenger_bundle_key: str
    first_selection_event: dict[str, Any] | None
    surface_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_mode": "readout_escape_replay_harness",
            "seed": self.seed,
            "task_id": self.task_id,
            "worker_model_name": self.worker_model_name,
            "replay_mode": self.replay_mode,
            "packet_mode": self.packet_mode,
            "prompt": self.prompt,
            "episode": asdict(self.episode),
            "readout_escape_seen": self.readout_escape_seen,
            "controller_selection_event_count": self.controller_selection_event_count,
            "nonnull_gate_frontier_count": self.nonnull_gate_frontier_count,
            "nonnull_controller_selected_count": self.nonnull_controller_selected_count,
            "sidecar_suggested_bundle_key": self.sidecar_suggested_bundle_key,
            "gate_report_frontier_bundle_key": self.gate_report_frontier_bundle_key,
            "controller_selected_bundle_key": self.controller_selected_bundle_key,
            "controller_selection_source": self.controller_selection_source,
            "controller_rejected_signals": list(self.controller_rejected_signals),
            "forced_base_bundle_key": self.forced_base_bundle_key,
            "forced_challenger_bundle_key": self.forced_challenger_bundle_key,
            "first_selection_event": None if self.first_selection_event is None else dict(self.first_selection_event),
            "surface_ids": list(self.surface_ids),
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
    task_env: ExperimentTaskEnv | None = None,
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
    worker_mps_mode: str = "auto",
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    semantic_critic_mode: str = "off",
    semantic_critic_model_name: str | None = None,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
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
        mps_mode=worker_mps_mode,
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
            controller_reflection_mode=controller_reflection_mode,
            controller_memory_window=controller_memory_window,
            readout_sidecar_analyzer=readout_sidecar_analyzer,
            readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
            worker_decoder_control_mode=worker_decoder_control_mode,
            worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
            worker_loop_rescue_total_alpha=worker_loop_rescue_total_alpha,
            worker_loop_rescue_total_edit_cost=worker_loop_rescue_total_edit_cost,
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
        controller_reflection_mode=controller_reflection_mode,
        semantic_critic_mode=str(semantic_critic_mode),
        semantic_critic_model_name=semantic_critic_model_name,
        worker_decoder_control_mode=worker_decoder_control_mode,
        worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        worker_loop_rescue_total_alpha=float(worker_loop_rescue_total_alpha),
        worker_loop_rescue_total_edit_cost=float(
            worker_loop_rescue_total_edit_cost
            if worker_loop_rescue_total_edit_cost is not None
            else worker_loop_rescue_total_alpha
        ),
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
    task_env: ExperimentTaskEnv | None = None,
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
    worker_mps_mode: str = "auto",
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    semantic_critic_mode: str = "off",
    semantic_critic_model_name: str | None = None,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
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
        mps_mode=worker_mps_mode,
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
        controller_reflection_mode=controller_reflection_mode,
        controller_memory_window=controller_memory_window,
        readout_sidecar_analyzer=readout_sidecar_analyzer,
        readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
        worker_decoder_control_mode=worker_decoder_control_mode,
        worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        worker_loop_rescue_total_alpha=worker_loop_rescue_total_alpha,
        worker_loop_rescue_total_edit_cost=worker_loop_rescue_total_edit_cost,
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
        controller_reflection_mode=controller_reflection_mode,
        semantic_critic_mode=str(semantic_critic_mode),
        semantic_critic_model_name=semantic_critic_model_name,
        worker_decoder_control_mode=worker_decoder_control_mode,
        worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        worker_loop_rescue_total_alpha=float(worker_loop_rescue_total_alpha),
        worker_loop_rescue_total_edit_cost=float(
            worker_loop_rescue_total_edit_cost
            if worker_loop_rescue_total_edit_cost is not None
            else worker_loop_rescue_total_alpha
        ),
        surface_ids=surface_ids,
        c1=c1,
    )
    _write_summary_artifact(log_dir, "experiment_summary.json", result.to_dict())
    return result


def run_shot_mode_probe_harness(
    *,
    worker_model_name: str,
    seed: int = 0,
    worker_model: Any | None = None,
    task_env: ExperimentTaskEnv | None = None,
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
    worker_mps_mode: str = "auto",
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    semantic_critic_mode: str = "off",
    semantic_critic_model_name: str | None = None,
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
    max_steps: int = 8,
    max_probe_candidates: int = 3,
    bootstrap_after_steps: int = 4,
) -> ShotModeProbeHarnessResult:
    if max_steps <= 0:
        raise ValueError("run_shot_mode_probe_harness requires max_steps >= 1")
    if max_probe_candidates <= 0:
        raise ValueError("run_shot_mode_probe_harness requires max_probe_candidates >= 1")
    if bootstrap_after_steps <= 0:
        raise ValueError("run_shot_mode_probe_harness requires bootstrap_after_steps >= 1")

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
        mps_mode=worker_mps_mode,
    )
    worker = build_hooked_transformer_worker_runtime(
        model,
        env,
        seed=seed,
        task_view_mode=task_view_mode,
        surface_catalog=surface_catalog,
        codec=codec,
        controller_reflection_mode=controller_reflection_mode,
        controller_memory_window=controller_memory_window,
        worker_decoder_control_mode=worker_decoder_control_mode,
        worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        worker_loop_rescue_total_alpha=worker_loop_rescue_total_alpha,
        worker_loop_rescue_total_edit_cost=worker_loop_rescue_total_edit_cost,
        readout_sidecar_analyzer=readout_sidecar_analyzer,
        readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
    )
    prompt = env.reset(seed)
    worker.reset(prompt)
    logger_factory = _logger_factory(log_dir)
    logger = None if logger_factory is None else logger_factory("shot_harness")
    surface_ids = tuple(surface["surface_id"] for surface in worker._surface_catalog_raw)

    shot_mode_packet: dict[str, Any] | None = None
    probe_results: list[dict[str, Any]] = []
    probe_round_count = 0
    probe_mode = "not_reached"

    def _packet_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        task_feedback = packet.get("task_feedback") if isinstance(packet.get("task_feedback"), Mapping) else {}
        return {
            "step": int(packet.get("step", 0) or 0),
            "control_phase_hint": packet.get("control_phase_hint"),
            "shot_mode_ready": bool(strategy_hints.get("shot_mode_ready", False)),
            "shot_probe_needed": bool(strategy_hints.get("shot_probe_needed", False)),
            "kv_probe_needed": bool(strategy_hints.get("kv_probe_needed", False)),
            "preferred_shot_surface_id": strategy_hints.get("preferred_shot_surface_id"),
            "preferred_kv_surface_id": strategy_hints.get("preferred_kv_surface_id"),
            "shot_candidate_count": len(strategy_hints.get("shot_candidate_edits", []) or []),
            "kv_candidate_count": len(strategy_hints.get("kv_candidate_edits", []) or []),
            "required_term_recall": task_feedback.get("required_term_recall"),
            "required_term_span_progress": task_feedback.get("required_term_span_progress"),
            "partial_score": task_feedback.get("partial_score"),
        }

    def _candidate_pool(packet: Mapping[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        candidates: list[dict[str, Any]] = []
        kv_candidates = strategy_hints.get("kv_candidate_edits")
        if isinstance(kv_candidates, SequenceABC) and not isinstance(kv_candidates, (str, bytes, bytearray)):
            candidates.extend(dict(item) for item in kv_candidates if isinstance(item, Mapping))
        if not candidates:
            shot_candidates = strategy_hints.get("shot_candidate_edits")
            if isinstance(shot_candidates, SequenceABC) and not isinstance(shot_candidates, (str, bytes, bytearray)):
                candidates.extend(dict(item) for item in shot_candidates if isinstance(item, Mapping))
        if limit is not None:
            return candidates[: max(0, int(limit))]
        return candidates

    def _select_probe_candidates(
        packet: Mapping[str, Any],
        *,
        stage: int,
        already_probed_surface_ids: set[str],
    ) -> list[dict[str, Any]]:
        if stage <= 1:
            first_round_limit = 1 if max_probe_candidates <= 1 else max(1, int(max_probe_candidates) - 1)
            return _candidate_pool(packet, limit=first_round_limit)

        if max_probe_candidates <= 1:
            return []

        pool = _candidate_pool(packet)
        positive_labels = {"positive", "weak_positive"}
        selected: list[dict[str, Any]] = []
        seen_surface_ids: set[str] = set()

        def _candidate_surface_id(candidate: Mapping[str, Any]) -> str:
            return str(candidate.get("surface_id", "") or "")

        for candidate in pool:
            surface_id = _candidate_surface_id(candidate)
            if not surface_id or surface_id in seen_surface_ids or surface_id not in already_probed_surface_ids:
                continue
            recent_probe = candidate.get("recent_probe") if isinstance(candidate.get("recent_probe"), Mapping) else {}
            label = str(recent_probe.get("label", "") or "")
            if label not in positive_labels:
                continue
            selected.append(dict(candidate))
            seen_surface_ids.add(surface_id)
            if len(selected) >= 1:
                return selected

        for candidate in pool:
            surface_id = _candidate_surface_id(candidate)
            if not surface_id or surface_id in seen_surface_ids or surface_id in already_probed_surface_ids:
                continue
            selected.append(dict(candidate))
            seen_surface_ids.add(surface_id)
            if len(selected) >= 1:
                return selected

        return selected

    for _ in range(max_steps):
        packet = worker.build_controller_packet()
        if (
            packet.get("latest_observer_check") is None
            and int(packet.get("step", 0) or 0) >= max(0, bootstrap_after_steps - 1)
        ):
            observer_result = worker.request_observer_check(
                {"kind": "semantic_progress", "reason": "shot_harness_bootstrap"},
                source="shot_harness",
            )
            if observer_result is not None:
                packet = worker.build_controller_packet()
        summary = _packet_summary(packet)
        if logger is not None:
            logger.log({"event": "shot_harness_observation"} | summary)
        control_phase_hint = str(packet.get("control_phase_hint", "") or "")
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        task_feedback = packet.get("task_feedback") if isinstance(packet.get("task_feedback"), Mapping) else {}
        should_bootstrap_probe = (
            control_phase_hint != "shot_mode"
            and int(packet.get("step", 0) or 0) >= int(bootstrap_after_steps)
            and float(task_feedback.get("required_term_recall", 0.0) or 0.0) <= 0.0
            and str(strategy_hints.get("loop_severity", "low") or "low") != "high"
        )
        if control_phase_hint == "shot_mode" or should_bootstrap_probe:
            shot_mode_packet = dict(packet)
            probe_mode = "shot_mode" if control_phase_hint == "shot_mode" else "bootstrap_probe"
            initial_candidates = _select_probe_candidates(packet, stage=1, already_probed_surface_ids=set())
            if initial_candidates:
                tool_requests = [
                    {
                        "tool": "dry_run_decode",
                        "candidate_edit": candidate,
                        "max_new_tokens": 2,
                        "top_k": 8,
                        "reason": "shot_harness_probe",
                    }
                    for candidate in initial_candidates
                ]
                probe_results = [
                    dict(result, probe_stage=1)
                    for result in worker.request_controller_tools(tool_requests, source="shot_harness")
                ]
                probe_round_count = 1 if probe_results else 0
                if logger is not None:
                    for result in probe_results:
                        logger.log({"event": "shot_harness_probe", **dict(result)})
                if max_probe_candidates > 1 and probe_results:
                    reranked_packet = worker.build_controller_packet()
                    shot_mode_packet = dict(reranked_packet)
                    if logger is not None:
                        logger.log({"event": "shot_harness_observation", "probe_stage": 2, **_packet_summary(reranked_packet)})
                    probed_surface_ids = {
                        str(result.get("candidate_edit", {}).get("surface_id", "") or "")
                        for result in probe_results
                        if isinstance(result.get("candidate_edit"), Mapping)
                    }
                    second_stage_candidates = _select_probe_candidates(
                        reranked_packet,
                        stage=2,
                        already_probed_surface_ids={surface_id for surface_id in probed_surface_ids if surface_id},
                    )
                    if second_stage_candidates:
                        second_stage_requests = [
                            {
                                "tool": "dry_run_decode",
                                "candidate_edit": candidate,
                                "max_new_tokens": 2,
                                "top_k": 8,
                                "reason": "shot_harness_probe_rerank",
                            }
                            for candidate in second_stage_candidates
                        ]
                        second_stage_results = [
                            dict(result, probe_stage=2)
                            for result in worker.request_controller_tools(second_stage_requests, source="shot_harness")
                        ]
                        if second_stage_results:
                            probe_round_count = 2
                            probe_results.extend(second_stage_results)
                            if logger is not None:
                                for result in second_stage_results:
                                    logger.log({"event": "shot_harness_probe", **dict(result)})
            break
        if worker.done():
            break
        worker.step()

    final_output = worker.final_text()
    observation_summary = (
        _packet_summary(shot_mode_packet)
        if shot_mode_packet is not None
        else {
            "step": int(getattr(worker, "_steps", 0) or 0),
            "control_phase_hint": "not_reached",
            "shot_mode_ready": False,
            "shot_probe_needed": False,
            "kv_probe_needed": False,
            "preferred_shot_surface_id": None,
            "preferred_kv_surface_id": None,
            "shot_candidate_count": 0,
            "kv_candidate_count": 0,
            "required_term_recall": worker._last_task_feedback.get("required_term_recall"),
            "required_term_span_progress": worker._last_task_feedback.get("required_term_span_progress"),
            "partial_score": worker._last_task_feedback.get("partial_score"),
        }
    )
    result = ShotModeProbeHarnessResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        worker_decoder_control_mode=worker_decoder_control_mode,
        shot_mode_reached=probe_mode == "shot_mode",
        shot_mode_step=(
            int(shot_mode_packet.get("step", 0) or 0)
            if shot_mode_packet is not None and probe_mode == "shot_mode"
            else None
        ),
        probe_mode=probe_mode,
        steps_executed=int(getattr(worker, "_steps", 0) or 0),
        prompt=prompt,
        final_output=final_output,
        final_task_feedback=dict(worker._last_task_feedback),
        observation_summary=observation_summary,
        probe_round_count=int(probe_round_count),
        probe_results=tuple(dict(item) for item in probe_results),
        surface_ids=surface_ids,
    )
    if logger is not None:
        logger.log(
            {
                "event": "shot_harness_complete",
                "shot_mode_reached": result.shot_mode_reached,
                "shot_mode_step": result.shot_mode_step,
                "probe_mode": result.probe_mode,
                "steps_executed": result.steps_executed,
                "probe_round_count": result.probe_round_count,
                "probe_result_count": len(result.probe_results),
            }
        )
    _write_summary_artifact(log_dir, "shot_harness_summary.json", result.to_dict())
    return result


def run_readout_escape_replay_harness(
    *,
    worker_model_name: str,
    seed: int,
    packet_mode: str = "forced_frontier",
    worker_model: Any | None = None,
    task_env: ExperimentTaskEnv | None = None,
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
    worker_mps_mode: str = "auto",
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
    max_generated_tokens: int = 4,
) -> ReadoutEscapeReplayHarnessResult:
    normalized_packet_mode = str(packet_mode).strip().lower().replace("-", "_")
    if normalized_packet_mode not in {"forced_frontier", "directscan", "fixed_candidate"}:
        raise ValueError("run_readout_escape_replay_harness packet_mode must be one of forced_frontier, directscan, fixed_candidate")
    env = task_env or SpiralConstrainedRewriteEnv()
    model = worker_model or load_worker_model(
        worker_model_name,
        model_path=worker_model_path,
        tokenizer_path=worker_tokenizer_path,
        device=worker_device,
        dtype=worker_dtype,
        first_n_layers=worker_first_n_layers,
        hf_offline=worker_hf_offline,
        trust_remote_code=worker_trust_remote_code,
        mps_mode=worker_mps_mode,
    )
    worker = build_hooked_transformer_worker_runtime(
        model,
        env,
        seed=seed,
        task_view_mode="redacted",
        surface_catalog=surface_catalog,
        codec=codec,
        max_generated_tokens=max_generated_tokens,
        controller_reflection_mode=controller_reflection_mode,
        controller_memory_window=controller_memory_window,
        worker_decoder_control_mode=worker_decoder_control_mode,
        worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
        worker_loop_rescue_total_alpha=worker_loop_rescue_total_alpha,
        worker_loop_rescue_total_edit_cost=worker_loop_rescue_total_edit_cost,
        readout_sidecar_analyzer=readout_sidecar_analyzer,
        readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
    )
    logger_factory = _logger_factory(log_dir)
    memory_logger = InMemoryStructuredLogger()
    file_logger = None if logger_factory is None else logger_factory("readout_escape_replay")
    logger = _StructuredLoggerMux(tuple(item for item in (memory_logger, file_logger) if item is not None))

    forced_state: dict[str, Any] = {
        "base_bundle_key": "",
        "challenger_bundle_key": "",
        "surface_ids": (),
        "directscan_hints": None,
        "cached_strategy_hints": None,
    }
    original_reset = worker.reset
    original_build_controller_packet = worker.build_controller_packet
    original_answer_readout_canary = worker._current_answer_readout_canary

    def _make_forced_candidate(*, surface_id: str, bundle_key: str, focus_term: str, candidate_family: str) -> dict[str, Any]:
        command = worker._build_dry_run_command(
            {
                "surface_id": surface_id,
                "kind": "resid_add",
                "alpha": 0.04,
                "ttl_steps": 1,
                "step_size": 0.04,
            }
        )
        if not isinstance(command, Mapping):
            raise ValueError("failed to synthesize forced replay command")
        edits = command.get("edits")
        if not isinstance(edits, SequenceABC) or isinstance(edits, (str, bytes, bytearray)) or not edits:
            raise ValueError("forced replay command did not include edits")
        candidate = dict(edits[0])
        candidate["surface_id"] = surface_id
        candidate["bundle_key"] = bundle_key
        candidate["phase_objective"] = "readout_escape"
        candidate["provenance_class"] = "source_body"
        candidate["bundle_provenance_tier"] = 3
        candidate["bundle_ready"] = True
        candidate["focus_feature"] = focus_term
        candidate["focus_term"] = focus_term
        candidate["span_kind"] = "forced_exact_prompt_span_mean"
        candidate["source_span"] = {"start": 0, "end": 0}
        candidate["read_source_resolved"] = True
        candidate["write_target_resolved"] = True
        candidate["candidate_family"] = candidate_family
        candidate["is_actionable_candidate"] = focus_term == "budget"
        candidate["first_piece_reachability"] = 0.24 if focus_term == "budget" else 0.0
        candidate["reachability_ratio"] = 1.0 if focus_term == "budget" else 0.0
        candidate["site"] = "resid_pre"
        return candidate

    def _model_layer_count() -> int:
        return max(1, int(getattr(getattr(worker.model, "cfg", None), "n_layers", 1) or 1))

    def _model_head_count() -> int:
        return max(1, int(getattr(getattr(worker.model, "cfg", None), "n_heads", 1) or 1))

    def _best_prompt_span(term: str) -> dict[str, Any] | None:
        spans = [dict(item) for item in worker._prompt_term_spans(term, max_spans=4)]
        if not spans:
            return None
        spans.sort(
            key=lambda item: (
                0 if str(item.get("provenance_class", "")) == "source_body" else 1,
                -int(item.get("length", 0) or 0),
                int(item.get("start", 0) or 0),
            )
        )
        return spans[0]

    def _directscan_support() -> dict[str, Any]:
        top_layer = max(0, _model_layer_count() - 1)
        head_count = _model_head_count()
        lower_layer = max(0, top_layer - 1)
        send_k_head = 0
        send_v_head = 1 if head_count > 1 else 0
        budget_k_head = 0 if top_layer != lower_layer else (2 if head_count > 2 else send_k_head)
        budget_v_head = 1 if top_layer != lower_layer else (3 if head_count > 3 else send_v_head)
        source_terms = ("send", "budget")
        top_feature_hits: list[dict[str, Any]] = []
        selected_spans: dict[str, dict[str, Any]] = {}
        for feature, site, layer, head, alignment in (
            ("send", "k_cache", lower_layer, send_k_head, 0.31),
            ("send", "v_cache", lower_layer, send_v_head, 0.32),
            ("budget", "k_cache", top_layer, budget_k_head, 0.37),
            ("budget", "v_cache", top_layer, budget_v_head, 0.39),
        ):
            span = _best_prompt_span(feature)
            if span is None:
                continue
            selected_spans.setdefault(feature, span)
            start = int(span.get("start", 0) or 0)
            top_feature_hits.append(
                {
                    "group": "required_terms",
                    "feature": feature,
                    "polarity": "promote",
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "token_mode": "last",
                    "alignment": alignment,
                    "argmax_pos": start,
                    "argmax_relative_index": -max(1, start + 1),
                    "argmax_piece": str(span.get("text", feature)),
                    "argmax_segment_kind": "prompt",
                    "source_positions": [
                        {
                            "position": start,
                            "relative_index": -max(1, start + 1),
                            "segment_kind": "prompt",
                            "piece": str(span.get("text", feature)),
                            "alignment": alignment,
                        }
                    ],
                    "coverage_progress": 0.0,
                }
            )
        return {
            "feedback": {
                "done": False,
                "progress_label": "stalled",
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "missing_required_terms": list(source_terms) + ["Mira", "Omar"],
                "entity_recall_terms": list(source_terms) + ["Mira", "Omar"],
                "partial_score": 0.45,
            },
            "observer_check": {
                "check_type": "semantic_progress",
                "trigger": "coverage_progress",
                "score": 0.2,
                "verdict": "flat",
                "kv_feature_scan": {
                    "projection_mode": "attn_weight_head_projection",
                    "surface_count": len(top_feature_hits),
                    "group_count": 1,
                    "top_feature_hits": top_feature_hits,
                },
            },
            "canary": {
                "semantic_focus_term": "Mira",
                "semantic_focus_source": "directscan_replay",
                "reachable_focus_term": "budget",
                "reachable_focus_piece": " budget",
                "reachable_focus_rank": 768,
                "target_mass": 0.000142,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.300192,
                "attractor_family_top_overlap": 4,
                "attractor_family_overlap_tokens": ["EW", " EW", "inou", " shards"],
                "top_tokens": ["EW", " EW", "inou", " shards", "Lawson"],
            },
            "spans": selected_spans,
        }

    def _ensure_compileable_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(candidate)
        if (
            isinstance(normalized.get("source"), Mapping)
            and isinstance(normalized.get("op"), Mapping)
            and isinstance(normalized.get("budget"), Mapping)
            and (
                isinstance(normalized.get("target"), Mapping)
                or normalized.get("surface_id") not in (None, "")
            )
        ):
            normalized.setdefault("id", f"replay_candidate_{normalized.get('surface_id', 'unknown')}")
            if "target" not in normalized and normalized.get("surface_id") not in (None, ""):
                normalized["target"] = {"surface_id": normalized.get("surface_id")}
            return normalized
        command = worker._build_dry_run_command(normalized)
        if isinstance(command, Mapping):
            edits = command.get("edits")
            if isinstance(edits, SequenceABC) and not isinstance(edits, (str, bytes, bytearray)) and edits:
                merged = dict(normalized)
                merged.update(dict(edits[0]))
                return merged
        return normalized

    def _forced_reset(self: HookedTransformerWorkerRuntime, prompt: str) -> None:
        original_reset(prompt)
        surface_ids = tuple(surface["surface_id"] for surface in self._surface_catalog_raw if surface.get("surface_id"))
        if len(surface_ids) < 2:
            raise ValueError("readout escape replay harness requires at least two activation surfaces")
        forced_state["surface_ids"] = surface_ids
        forced_state["cached_strategy_hints"] = None
        if normalized_packet_mode == "forced_frontier":
            base_surface_id, challenger_surface_id = surface_ids[:2]
            forced_state["base_bundle_key"] = f"kv_anchor:send:source_body:{base_surface_id}"
            forced_state["challenger_bundle_key"] = f"kv_anchor:budget:source_body:{challenger_surface_id}"
            self._last_task_feedback = {
                "done": False,
                "progress_label": "stalled",
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "missing_required_terms": ["budget", "Mira", "Omar"],
                "entity_recall_terms": ["budget", "Mira", "Omar"],
                "partial_score": 0.45,
            }
            return

        directscan = _directscan_support()
        forced_state["directscan_hints"] = directscan
        self._last_task_feedback = dict(directscan["feedback"])
        self._latest_observer_check = dict(directscan["observer_check"])
        self._observer_checks = [dict(directscan["observer_check"])]

    def _forced_canary(self: HookedTransformerWorkerRuntime, **_kwargs: Any) -> Mapping[str, Any]:
        if normalized_packet_mode == "forced_frontier":
            return {
                "semantic_focus_term": "Mira",
                "semantic_focus_source": "forced_replay",
                "reachable_focus_term": "budget",
                "reachable_focus_piece": " budget",
                "reachable_focus_rank": 768,
                "target_mass": 0.000142,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.300192,
                "attractor_family_top_overlap": 4,
                "attractor_family_overlap_tokens": ["EW", " EW", "inou", " shards"],
                "top_tokens": ["EW", " EW", "inou", " shards", "Lawson"],
            }
        directscan = forced_state.get("directscan_hints") if isinstance(forced_state.get("directscan_hints"), Mapping) else {}
        return dict(directscan.get("canary", {}))

    def _forced_build_controller_packet(self: HookedTransformerWorkerRuntime) -> dict[str, Any]:
        packet = dict(original_build_controller_packet())
        if normalized_packet_mode in {"directscan", "fixed_candidate"}:
            strategy_hints = dict(packet.get("strategy_hints", {}))
            candidate_fields = ("kv_candidate_edits", "shot_candidate_edits", "kv_retry_candidate_edits")
            live_candidates: list[dict[str, Any]] = []
            for key in candidate_fields:
                raw_items = strategy_hints.get(key)
                if not isinstance(raw_items, SequenceABC) or isinstance(raw_items, (str, bytes, bytearray)):
                    continue
                normalized_items = [_ensure_compileable_candidate(item) for item in raw_items if isinstance(item, Mapping)]
                strategy_hints[key] = normalized_items
                live_candidates.extend(normalized_items)
            if not live_candidates:
                raise ValueError("directscan replay did not produce bundle candidates")
            if normalized_packet_mode == "fixed_candidate":
                cached = forced_state.get("cached_strategy_hints")
                if isinstance(cached, Mapping):
                    strategy_hints.update(dict(cached))
                    packet["strategy_hints"] = strategy_hints
                    packet["control_phase_hint"] = "readout_escape"
                    return packet
            base_bundle_key = str(strategy_hints.get("base_winner_bundle_key", "") or "")
            challenger_bundle_key = str(strategy_hints.get("challenger_bundle_key", "") or "")
            if not challenger_bundle_key:
                unique_bundle_keys = []
                for item in live_candidates:
                    if not isinstance(item, Mapping):
                        continue
                    bundle_key = str(item.get("bundle_key", "") or "")
                    if bundle_key and bundle_key not in unique_bundle_keys:
                        unique_bundle_keys.append(bundle_key)
                if unique_bundle_keys:
                    challenger_bundle_key = unique_bundle_keys[-1]
                    strategy_hints.setdefault("challenger_bundle_key", challenger_bundle_key)
                if len(unique_bundle_keys) > 1:
                    base_bundle_key = unique_bundle_keys[0]
                    strategy_hints.setdefault("base_winner_bundle_key", base_bundle_key)
            forced_state["base_bundle_key"] = base_bundle_key
            forced_state["challenger_bundle_key"] = challenger_bundle_key or base_bundle_key
            if normalized_packet_mode == "fixed_candidate":
                cached_keys = (
                    "kv_candidate_edits",
                    "shot_candidate_edits",
                    "kv_retry_candidate_edits",
                    "base_winner_bundle_key",
                    "challenger_bundle_key",
                    "selected_bundle_key",
                    "selection_source",
                    "gate_report_frontier_bundle_key",
                    "gate_report_selection_source",
                    "gate_report_pairwise_reason_text",
                    "readout_analyzer_suggested_bundle_key",
                    "readout_sidecar_suggested_bundle_key",
                    "bundle_selector_phase",
                )
                forced_state["cached_strategy_hints"] = {
                    key: (
                        [dict(item) for item in strategy_hints.get(key, []) if isinstance(item, Mapping)]
                        if key in {"kv_candidate_edits", "shot_candidate_edits", "kv_retry_candidate_edits"}
                        else strategy_hints.get(key)
                    )
                    for key in cached_keys
                    if strategy_hints.get(key) not in (None, "")
                }
            packet["strategy_hints"] = strategy_hints
            packet["control_phase_hint"] = "readout_escape"
            return packet

        strategy_hints = dict(packet.get("strategy_hints", {}))
        base_bundle_key = str(forced_state.get("base_bundle_key", "") or "")
        challenger_bundle_key = str(forced_state.get("challenger_bundle_key", "") or "")
        surface_ids = tuple(str(item) for item in forced_state.get("surface_ids", ()) if str(item))
        if len(surface_ids) < 2:
            raise ValueError("forced replay surfaces not initialized")
        send_candidate = _make_forced_candidate(
            surface_id=surface_ids[0],
            bundle_key=base_bundle_key,
            focus_term="send",
            candidate_family="forced_readout_escape:send",
        )
        budget_candidate = _make_forced_candidate(
            surface_id=surface_ids[1],
            bundle_key=challenger_bundle_key,
            focus_term="budget",
            candidate_family="forced_readout_escape:budget",
        )
        strategy_hints.update(
            {
                "shot_mode_ready": True,
                "readout_escape_needed": True,
                "readout_escape_reason": "forced_contract_replay",
                "controller_focus_term": "budget",
                "controller_focus_source": "reachable_focus",
                "kv_candidate_edits": [send_candidate, budget_candidate],
                "base_winner_bundle_key": base_bundle_key,
                "challenger_bundle_key": challenger_bundle_key,
                "readout_analyzer_suggested_bundle_key": challenger_bundle_key,
                "gate_report_frontier_bundle_key": challenger_bundle_key,
                "gate_report_selection_source": "sidecar_tiebreak",
                "gate_report_pairwise_reason_text": "forced replay prefers budget challenger at readout_escape",
                "selection_source": "sidecar_tiebreak",
                "bundle_selector_phase": "readout_escape",
            }
        )
        task_feedback = dict(packet.get("task_feedback", {}))
        task_feedback.setdefault("required_term_recall", 0.0)
        task_feedback.setdefault("required_term_span_progress", 0.0)
        task_feedback.setdefault("partial_score", 0.45)
        packet["control_phase_hint"] = "readout_escape"
        packet["strategy_hints"] = strategy_hints
        packet["task_feedback"] = task_feedback
        return packet

    worker.reset = MethodType(_forced_reset, worker)
    worker._current_answer_readout_canary = MethodType(_forced_canary, worker)
    worker.build_controller_packet = MethodType(_forced_build_controller_packet, worker)

    try:
        controller_client = _FrontierReplayControllerClient()
        ctx = StepContext(
            packet={},
            runtime_state=worker.runtime_state,
            traces={},
            stats={},
            adapter=worker.adapter,
        )
        episode = run_episode(env, worker, controller_client, ctx, logger=logger)
    finally:
        worker.reset = original_reset
        worker._current_answer_readout_canary = original_answer_readout_canary
        worker.build_controller_packet = original_build_controller_packet

    selection_events = [event for event in memory_logger.events if event.get("event") == "controller_selection"]
    first_selection_event = dict(selection_events[0]) if selection_events else None
    result = ReadoutEscapeReplayHarnessResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        replay_mode=f"{normalized_packet_mode}_apply",
        packet_mode=normalized_packet_mode,
        prompt=episode.prompt,
        episode=episode,
        readout_escape_seen=any(
            event.get("event") == "controller_observation" and event.get("control_phase_hint") == "readout_escape"
            for event in memory_logger.events
        )
        or any(
            event.get("event") == "controller_command" and event.get("gate_report_frontier_bundle_key") not in (None, "")
            for event in memory_logger.events
        ),
        controller_selection_event_count=len(selection_events),
        nonnull_gate_frontier_count=sum(
            1
            for event in memory_logger.events
            if event.get("event") == "controller_selection" and event.get("gate_report_frontier_bundle_key") not in (None, "")
        ),
        nonnull_controller_selected_count=sum(
            1
            for event in memory_logger.events
            if event.get("event") == "controller_selection" and event.get("controller_selected_bundle_key") not in (None, "")
        ),
        sidecar_suggested_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("sidecar_suggested_bundle_key"),
        gate_report_frontier_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("gate_report_frontier_bundle_key"),
        controller_selected_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("controller_selected_bundle_key"),
        controller_selection_source=None
        if first_selection_event is None
        else first_selection_event.get("controller_selection_source"),
        controller_rejected_signals=tuple(
            str(item)
            for item in (
                [] if first_selection_event is None else first_selection_event.get("controller_rejected_signals", [])
            )
        ),
        forced_base_bundle_key=str(forced_state.get("base_bundle_key", "") or ""),
        forced_challenger_bundle_key=str(forced_state.get("challenger_bundle_key", "") or ""),
        first_selection_event=first_selection_event,
        surface_ids=tuple(str(item) for item in forced_state.get("surface_ids", ()) if str(item)),
    )

    _write_summary_artifact(log_dir, "readout_escape_replay_summary.json", result.to_dict())
    return result


def run_digit_transform_sweep(
    *,
    provider_name: str,
    controller_model_name: str,
    worker_model_name: str,
    seeds: Sequence[int],
    controller_api_key: str | None = None,
    worker_model: Any | None = None,
    task_env_factory: Callable[[], ExperimentTaskEnv] | None = None,
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
    worker_mps_mode: str = "auto",
    controller_reflection_mode: str = "off",
    controller_memory_window: int = 3,
    semantic_critic_mode: str = "off",
    semantic_critic_model_name: str | None = None,
    readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
    readout_analyzer_rerank_mode: str = "apply",
    worker_decoder_control_mode: str = "off",
    worker_loop_rescue_edits_per_run: int = 0,
    worker_loop_rescue_total_alpha: float = 0.0,
    worker_loop_rescue_total_edit_cost: float | None = None,
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
        mps_mode=worker_mps_mode,
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
                worker_mps_mode=worker_mps_mode,
                controller_reflection_mode=controller_reflection_mode,
                controller_memory_window=controller_memory_window,
                semantic_critic_mode=semantic_critic_mode,
                semantic_critic_model_name=semantic_critic_model_name,
                readout_sidecar_analyzer=readout_sidecar_analyzer,
                readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
                worker_decoder_control_mode=worker_decoder_control_mode,
                worker_loop_rescue_edits_per_run=worker_loop_rescue_edits_per_run,
                worker_loop_rescue_total_alpha=worker_loop_rescue_total_alpha,
                worker_loop_rescue_total_edit_cost=worker_loop_rescue_total_edit_cost,
            )
            )
    result = DigitTransformSweepResult(seeds=resolved_seeds, runs=tuple(runs))
    _write_summary_artifact(log_dir, "sweep_summary.json", result.to_dict())
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Spiral intervention-lab task experiment.")
    parser.add_argument("--provider", required=True, help="Controller provider: openai, anthropic, mistral, or google")
    parser.add_argument("--controller-model", required=True, help="Black-box controller model name")
    parser.add_argument("--worker-model", default="gpt2-small", help="HookedTransformer worker model name or alias")
    parser.add_argument(
        "--task",
        default="digit_transform",
        choices=[
            "digit_transform",
            "digit_copy",
            "sentence_ordering",
            "entailment_reasoning",
            "constrained_rewrite",
            "structured_summary",
        ],
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
    parser.add_argument(
        "--worker-mps-mode",
        default="auto",
        choices=list(_WORKER_MPS_MODES),
        help="MPS worker loading mode: auto keeps current defaults, conservative keeps MPS execution but resets torch default-device auto-placement to CPU",
    )
    parser.add_argument(
        "--controller-reflection-mode",
        default="off",
        choices=list(_CONTROLLER_REFLECTION_MODES),
        help="Whether the controller can write structured controller_memory entries that are fed back on later steps",
    )
    parser.add_argument(
        "--controller-memory-window",
        type=int,
        default=3,
        help="How many recent controller_memory entries to retain in the observation packet when reflection mode is enabled",
    )
    parser.add_argument(
        "--semantic-critic-mode",
        default="off",
        choices=list(_SEMANTIC_CRITIC_MODES),
        help="Optional bounded runtime semantic observer. 'minilm' enables MiniLM observer checks triggered by runtime progress signals instead of a per-step heartbeat metric.",
    )
    parser.add_argument(
        "--semantic-critic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name for --semantic-critic-mode=minilm",
    )
    parser.add_argument(
        "--semantic-critic-device",
        default="cpu",
        help="Device for the semantic critic model, e.g. cpu or mps",
    )
    parser.add_argument(
        "--readout-analyzer",
        default=None,
        choices=list(_READOUT_ANALYZER_MODES),
        help="Optional bounded readout analyzer used only to rerank / veto readout-escape candidates without changing runtime actuation APIs.",
    )
    parser.add_argument(
        "--readout-sidecar-analyzer",
        dest="readout_sidecar_analyzer",
        default=None,
        choices=list(_READOUT_ANALYZER_MODES),
        help="Deprecated alias for --readout-analyzer.",
    )
    parser.add_argument(
        "--readout-analyzer-rerank-mode",
        default="apply",
        choices=list(_READOUT_ANALYZER_RERANK_MODES),
        help="How the bounded readout analyzer may affect final bundle selection: off disables rerank, shadow records challenger decisions without applying them, apply allows bounded pairwise promotion.",
    )
    parser.add_argument(
        "--worker-decoder-control-mode",
        default="off",
        choices=list(_WORKER_DECODER_CONTROL_MODES),
        help=(
            "Optional worker-side soft-control mode: loop_aware is task-agnostic anti-loop shaping; "
            "loop_aware_prune also demotes the current overconfident top token when stalled; "
            "loop_aware_constraint adds a small soft bias toward explicitly missing constraint tokens; "
            "loop_aware_entity_recall softly continues or starts explicit missing entity token sequences; "
            "logit_bias_entity_soft softly biases logits toward tokens from explicit missing entities without forcing output."
        ),
    )
    parser.add_argument(
        "--worker-loop-rescue-edits-per-run",
        type=int,
        default=0,
        help="Separate run budget for small loop-rescue controller edits; 0 disables the rescue pool",
    )
    parser.add_argument(
        "--worker-loop-rescue-total-alpha",
        type=float,
        default=0.0,
        help="Separate total alpha budget for loop-rescue edits",
    )
    parser.add_argument(
        "--worker-loop-rescue-total-edit-cost",
        type=float,
        default=None,
        help="Separate total edit-cost budget for loop-rescue edits; defaults to --worker-loop-rescue-total-alpha",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be >= 1")
    if args.controller_memory_window <= 0:
        parser.error("--controller-memory-window must be >= 1")
    if args.worker_loop_rescue_edits_per_run < 0:
        parser.error("--worker-loop-rescue-edits-per-run must be >= 0")
    if args.worker_loop_rescue_total_alpha < 0.0:
        parser.error("--worker-loop-rescue-total-alpha must be >= 0")
    if args.worker_loop_rescue_total_edit_cost is not None and args.worker_loop_rescue_total_edit_cost < 0.0:
        parser.error("--worker-loop-rescue-total-edit-cost must be >= 0")
    if args.c1_only and args.num_seeds != 1:
        parser.error("--c1-only currently supports only --num-seeds 1")
    if args.controller_api_key is None:
        env_var = provider_api_env_var(args.provider)
        if os.getenv(env_var) is None:
            parser.error(f"{env_var} is not set and --controller-api-key was not provided")

    semantic_critic = create_semantic_critic(
        args.semantic_critic_mode,
        model_name=args.semantic_critic_model,
        device=args.semantic_critic_device,
    )
    readout_analyzer_mode = (
        args.readout_analyzer
        if args.readout_analyzer is not None
        else (args.readout_sidecar_analyzer if args.readout_sidecar_analyzer is not None else "off")
    )
    readout_sidecar_analyzer = create_readout_analyzer(readout_analyzer_mode)

    if args.c1_only:
        result = run_digit_transform_c1_only_experiment(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seed=args.seed,
            controller_api_key=args.controller_api_key,
            task_env=create_task_env(args.task, semantic_critic=semantic_critic),
            task_view_mode=args.task_view_mode,
            log_dir=args.log_dir,
            worker_model_path=args.worker_model_path,
            worker_tokenizer_path=args.worker_tokenizer_path,
            worker_device=args.worker_device,
            worker_dtype=args.worker_dtype,
            worker_first_n_layers=args.worker_first_n_layers,
            worker_hf_offline=args.worker_hf_offline,
            worker_trust_remote_code=args.worker_trust_remote_code,
            worker_mps_mode=args.worker_mps_mode,
            controller_reflection_mode=args.controller_reflection_mode,
            controller_memory_window=args.controller_memory_window,
            semantic_critic_mode=args.semantic_critic_mode,
            semantic_critic_model_name=None if semantic_critic is None else args.semantic_critic_model,
            readout_sidecar_analyzer=readout_sidecar_analyzer,
            readout_analyzer_rerank_mode=args.readout_analyzer_rerank_mode,
            worker_decoder_control_mode=args.worker_decoder_control_mode,
            worker_loop_rescue_edits_per_run=args.worker_loop_rescue_edits_per_run,
            worker_loop_rescue_total_alpha=args.worker_loop_rescue_total_alpha,
            worker_loop_rescue_total_edit_cost=args.worker_loop_rescue_total_edit_cost,
        )
        payload = result.to_dict()
    elif args.num_seeds == 1:
        result = run_digit_transform_experiment(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seed=args.seed,
            controller_api_key=args.controller_api_key,
            task_env=create_task_env(args.task, semantic_critic=semantic_critic),
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
            worker_mps_mode=args.worker_mps_mode,
            controller_reflection_mode=args.controller_reflection_mode,
            controller_memory_window=args.controller_memory_window,
            semantic_critic_mode=args.semantic_critic_mode,
            semantic_critic_model_name=None if semantic_critic is None else args.semantic_critic_model,
            readout_sidecar_analyzer=readout_sidecar_analyzer,
            readout_analyzer_rerank_mode=args.readout_analyzer_rerank_mode,
            worker_decoder_control_mode=args.worker_decoder_control_mode,
            worker_loop_rescue_edits_per_run=args.worker_loop_rescue_edits_per_run,
            worker_loop_rescue_total_alpha=args.worker_loop_rescue_total_alpha,
            worker_loop_rescue_total_edit_cost=args.worker_loop_rescue_total_edit_cost,
        )
        payload = result.to_dict()
    else:
        sweep = run_digit_transform_sweep(
            provider_name=args.provider,
            controller_model_name=args.controller_model,
            worker_model_name=args.worker_model,
            seeds=tuple(range(args.seed, args.seed + args.num_seeds)),
            controller_api_key=args.controller_api_key,
            task_env_factory=lambda: create_task_env(args.task, semantic_critic=semantic_critic),
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
            worker_mps_mode=args.worker_mps_mode,
            controller_reflection_mode=args.controller_reflection_mode,
            controller_memory_window=args.controller_memory_window,
            semantic_critic_mode=args.semantic_critic_mode,
            semantic_critic_model_name=None if semantic_critic is None else args.semantic_critic_model,
            readout_sidecar_analyzer=readout_sidecar_analyzer,
            readout_analyzer_rerank_mode=args.readout_analyzer_rerank_mode,
            worker_decoder_control_mode=args.worker_decoder_control_mode,
            worker_loop_rescue_edits_per_run=args.worker_loop_rescue_edits_per_run,
            worker_loop_rescue_total_alpha=args.worker_loop_rescue_total_alpha,
            worker_loop_rescue_total_edit_cost=args.worker_loop_rescue_total_edit_cost,
        )
        payload = sweep.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
