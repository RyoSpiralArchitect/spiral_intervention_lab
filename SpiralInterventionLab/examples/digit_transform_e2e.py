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
    build_sae_feature_emitter_readout_analyzer,
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
_READOUT_ANALYZER_MODES = ("off", "heuristic", "sae_scaffold")
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
    max_production_trial_edits_per_run: int = 1,
    max_production_trial_alpha: float = 0.15,
    max_production_trial_edit_cost: float | None = None,
    max_production_trial_followup_edits_per_run: int = 1,
    max_production_trial_followup_alpha: float = 0.05,
    max_production_trial_followup_edit_cost: float | None = None,
    max_diagnostic_calls_per_run: int = 8,
    diagnostic_result_window: int = 8,
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
        max_production_trial_edits_per_run=max_production_trial_edits_per_run,
        max_production_trial_alpha=max_production_trial_alpha,
        max_production_trial_edit_cost=max_production_trial_edit_cost,
        max_production_trial_followup_edits_per_run=max_production_trial_followup_edits_per_run,
        max_production_trial_followup_alpha=max_production_trial_followup_alpha,
        max_production_trial_followup_edit_cost=max_production_trial_followup_edit_cost,
        max_diagnostic_calls_per_run=max_diagnostic_calls_per_run,
        diagnostic_result_window=diagnostic_result_window,
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


def _controller_step_views(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, dict[str, Any]] = {}
    for event in events:
        try:
            step = int(event.get("step", 0))
        except Exception:
            continue
        bucket = grouped.setdefault(
            step,
            {
                "step": step,
                "provider_attempts": [],
                "controller_observation": None,
                "controller_decision": None,
                "controller_command": None,
                "controller_selection": None,
                "diagnostic_requests": [],
                "diagnostic_results": [],
                "diagnostic_vetoes": [],
                "loop_patience": [],
            },
        )
        event_name = str(event.get("event", "") or "")
        if event_name == "controller_provider_attempt":
            bucket["provider_attempts"].append(
                {
                    "attempt": event.get("attempt"),
                    "parse_ok": event.get("parse_ok"),
                    "response_text": event.get("response_text"),
                    "raw_json_object": event.get("raw_json_object"),
                    "normalized_payload": event.get("normalized_payload"),
                    "normalization_delta": event.get("normalization_delta"),
                }
            )
        elif event_name == "controller_observation":
            bucket["controller_observation"] = dict(event)
        elif event_name == "controller_decision":
            bucket["controller_decision"] = dict(event)
        elif event_name == "controller_command":
            bucket["controller_command"] = dict(event)
        elif event_name == "controller_selection":
            bucket["controller_selection"] = dict(event)
        elif event_name == "controller_diagnostic_request":
            bucket["diagnostic_requests"].append(dict(event))
        elif event_name == "controller_diagnostic_result":
            bucket["diagnostic_results"].append(dict(event))
        elif event_name == "controller_diagnostic_veto":
            bucket["diagnostic_vetoes"].append(dict(event))
        elif event_name == "controller_loop_patience":
            bucket["loop_patience"].append(dict(event))
    views: list[dict[str, Any]] = []
    for step in sorted(grouped):
        bucket = grouped[step]
        selection = bucket.get("controller_selection")
        bridge_visible = bool(selection.get("controller_bridge_plan_visible", False)) if isinstance(selection, Mapping) else False
        views.append(
            {
                "step": step,
                "provider_attempt_count": len(bucket["provider_attempts"]),
                "provider_attempts": list(bucket["provider_attempts"]),
                "controller_observation": bucket.get("controller_observation"),
                "controller_decision": bucket.get("controller_decision"),
                "controller_command": bucket.get("controller_command"),
                "controller_selection": selection,
                "diagnostic_request_count": len(bucket["diagnostic_requests"]),
                "diagnostic_requests": list(bucket["diagnostic_requests"]),
                "diagnostic_result_count": len(bucket["diagnostic_results"]),
                "diagnostic_results": list(bucket["diagnostic_results"]),
                "diagnostic_veto_count": len(bucket["diagnostic_vetoes"]),
                "diagnostic_vetoes": list(bucket["diagnostic_vetoes"]),
                "loop_patience_count": len(bucket["loop_patience"]),
                "loop_patience": list(bucket["loop_patience"]),
                "bridge_visible": bridge_visible,
                "bridge_dual_layer_missing": (
                    bool(selection.get("controller_bridge_dual_layer_missing", False))
                    if isinstance(selection, Mapping)
                    else False
                ),
            }
        )
    return tuple(views)


def _bridge_eval_rows_have_context_drift(rows: Sequence[Mapping[str, Any]]) -> bool:
    usable_rows = [row for row in rows if isinstance(row, Mapping)]
    if not usable_rows:
        return False
    for row in usable_rows:
        fingerprint = row.get("eval_context_fingerprint")
        if not isinstance(fingerprint, Mapping):
            return False
        decode_step = int(fingerprint.get("decode_step", 0) or 0)
        answer_prefix = str(fingerprint.get("answer_prefix", "") or "")
        active_patch_count = int(fingerprint.get("active_patch_count", 0) or 0)
        if decode_step <= 0 and not answer_prefix and active_patch_count <= 0:
            return False
    return True


def _bridge_plan_unavailable_summary(
    bridge_eval_summary: Mapping[str, Any],
    *,
    preferred_objective_keys: Sequence[str] = (),
) -> dict[str, Any]:
    rows = [
        dict(item)
        for item in bridge_eval_summary.get("matrix", ())
        if isinstance(item, Mapping)
    ] if isinstance(bridge_eval_summary.get("matrix"), SequenceABC) and not isinstance(
        bridge_eval_summary.get("matrix"), (str, bytes, bytearray)
    ) else []
    shadow_bundle_keys = [
        str(item)
        for item in bridge_eval_summary.get("shadow_bundle_keys", ())
        if str(item)
    ] if isinstance(bridge_eval_summary.get("shadow_bundle_keys"), SequenceABC) and not isinstance(
        bridge_eval_summary.get("shadow_bundle_keys"), (str, bytes, bytearray)
    ) else []
    bundle_keys = [
        str(item)
        for item in bridge_eval_summary.get("bundle_keys", ())
        if str(item)
    ] if isinstance(bridge_eval_summary.get("bundle_keys"), SequenceABC) and not isinstance(
        bridge_eval_summary.get("bundle_keys"), (str, bytes, bytearray)
    ) else []
    exception_text = str(bridge_eval_summary.get("exception", "") or "")
    rows_by_objective: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        objective_key = str(row.get("objective_bundle_key", "") or "")
        if objective_key:
            rows_by_objective.setdefault(objective_key, []).append(row)

    def _reason_for_rows(objective_key: str, objective_rows: Sequence[Mapping[str, Any]]) -> str:
        usable_rows = [row for row in objective_rows if isinstance(row, Mapping)]
        if not usable_rows:
            return "no_bridge_eval_rows"
        classes = [str(row.get("actuator_class", "") or "") for row in usable_rows if str(row.get("actuator_class", "") or "")]
        if classes and all(cls == "collapse_sharpener" for cls in classes):
            return "all_collapse_sharpener"
        if classes and all(cls == "dead_actuator" for cls in classes):
            return "all_dead_actuator"
        if classes and all(cls in {"dead_actuator", "collapse_sharpener"} for cls in classes):
            return "no_safe_actuator"
        realized_to_objective = any(
            str(row.get("realized_lift_bundle_key", "") or "") == objective_key
            for row in usable_rows
        )
        if not realized_to_objective:
            return "no_objective_lift_to_frontier"
        if _bridge_eval_rows_have_context_drift(usable_rows):
            return "bridge_eval_context_drift"
        return "bridge_plan_not_certified"

    objective_reasons = {
        objective_key: _reason_for_rows(objective_key, objective_rows)
        for objective_key, objective_rows in rows_by_objective.items()
    }
    preferred_objective = next((str(item) for item in preferred_objective_keys if str(item)), "")
    selected_objective_key = ""
    if preferred_objective and preferred_objective in objective_reasons:
        selected_objective_key = preferred_objective
    elif preferred_objective and preferred_objective in bundle_keys:
        selected_objective_key = preferred_objective
    elif objective_reasons:
        selected_objective_key = next(iter(objective_reasons))
    elif bundle_keys:
        selected_objective_key = bundle_keys[0]
    if exception_text:
        reason = "bridge_eval_failed"
    elif not bundle_keys:
        reason = "no_bridge_eval_bundle"
    elif not shadow_bundle_keys:
        reason = "no_shadow_competitor"
    elif selected_objective_key and selected_objective_key in objective_reasons:
        reason = objective_reasons[selected_objective_key]
    elif rows:
        reason = "bridge_plan_not_certified"
    else:
        reason = "no_bridge_eval_rows"
    return {
        "reason": reason,
        "objective_bundle_key": selected_objective_key or None,
        "objective_reasons": objective_reasons,
        "context_drift": bool(_bridge_eval_rows_have_context_drift(rows)),
    }


def _classify_attention_shadow_actuator(
    *,
    response_profile: str,
    focus_rank_delta: int,
    target_mass_delta: float,
    target_top20_hit_delta: int,
    first_logit_delta_l2: float,
) -> tuple[str, bool, str]:
    """Classify attention-scale evidence as a shadow diagnostic, not an apply permit."""
    if response_profile == "unsafe_zero_response":
        return "unsafe_sharpener", False, "unsafe_attention_zero_response"
    if target_top20_hit_delta > 0:
        return "top20_carrier", True, "target_entered_top20_counterfactually"
    if target_mass_delta > 0.00002:
        return "mass_positive_carrier", True, "target_mass_increased_counterfactually"
    if focus_rank_delta >= 20 and target_mass_delta >= -0.00001 and first_logit_delta_l2 > 0.0:
        return "rank_carrier_only", True, "rank_lift_without_mass_or_top20_gain"
    if focus_rank_delta >= 4 and target_mass_delta >= -0.00002 and first_logit_delta_l2 > 0.0:
        return "weak_rank_carrier", False, "rank_lift_below_certification_threshold"
    if focus_rank_delta <= 0 and target_mass_delta <= 0.0 and target_top20_hit_delta <= 0:
        return "flat_shadow", False, "no_rank_or_target_lift"
    return "unresolved_shadow", False, "mixed_signal_requires_more_diagnostics"


def _attention_scale_response_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for item in rows:
        if not isinstance(item, Mapping) or str(item.get("diagnostic_family", "") or "") != "attention_head_scale":
            continue
        fingerprint = item.get("candidate_fingerprint") if isinstance(item.get("candidate_fingerprint"), Mapping) else {}
        objective_key = str(item.get("objective_bundle_key", "") or "")
        site = str(fingerprint.get("site", "") or "")
        layer = fingerprint.get("layer")
        head = fingerprint.get("head")
        if not objective_key or not site:
            continue
        if isinstance(layer, bool) or not isinstance(layer, int):
            continue
        if isinstance(head, bool) or not isinstance(head, int):
            continue
        grouped.setdefault((objective_key, site, int(layer), int(head)), []).append(dict(item))

    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    summaries: list[dict[str, Any]] = []
    for (objective_key, site, layer, head), items in grouped.items():
        normalized: list[dict[str, Any]] = []
        for item in items:
            fingerprint = item.get("candidate_fingerprint") if isinstance(item.get("candidate_fingerprint"), Mapping) else {}
            row = dict(item)
            row["_scale"] = _as_float(fingerprint.get("scale"), default=1.0)
            row["_focus_rank_delta"] = max(
                _as_int(item.get("focus_rank_delta", 0)),
                _as_int(item.get("rank_focus_delta", 0)),
            )
            row["_target_mass_delta"] = _as_float(item.get("target_mass_delta", 0.0))
            row["_target_top20_hit_delta"] = _as_int(item.get("target_top20_hit_delta", 0))
            row["_logit_l2"] = _as_float(item.get("first_logit_delta_l2", 0.0))
            normalized.append(row)
        normalized.sort(key=lambda item: float(item["_scale"]))
        zero_rows = [item for item in normalized if abs(float(item["_scale"])) <= 1e-9]
        partial_rows = [item for item in normalized if float(item["_scale"]) > 1e-9]
        zero = zero_rows[0] if zero_rows else None
        best_partial = max(
            partial_rows,
            key=lambda item: (
                int(item["_target_top20_hit_delta"]),
                float(item["_target_mass_delta"]),
                int(item["_focus_rank_delta"]),
                -float(item["_scale"]),
            ),
            default=None,
        )
        zero_focus_rank_delta = 0 if zero is None else int(zero["_focus_rank_delta"])
        zero_target_mass_delta = 0.0 if zero is None else float(zero["_target_mass_delta"])
        partial_focus_rank_delta = 0 if best_partial is None else int(best_partial["_focus_rank_delta"])
        partial_target_mass_delta = 0.0 if best_partial is None else float(best_partial["_target_mass_delta"])
        zero_logit_l2 = 0.0 if zero is None else float(zero["_logit_l2"])
        partial_logit_l2 = max((float(item["_logit_l2"]) for item in partial_rows), default=0.0)
        partial_to_zero_logit_ratio = (
            round(float(partial_logit_l2 / zero_logit_l2), 6)
            if zero_logit_l2 > 1e-12
            else None
        )
        if zero is not None and str(zero.get("actuator_class", "") or "") in {"harmful", "collapse_sharpener"}:
            response_profile = "unsafe_zero_response"
            next_probe = "avoid_head_scale_apply_until_counterfactual_safe"
        elif zero_focus_rank_delta >= 6 and partial_focus_rank_delta < 4 and partial_target_mass_delta <= 0.00002:
            response_profile = "zero_only_rank_lift"
            next_probe = "try_sparse_head_suppression_or_compensated_head_patch"
        elif partial_focus_rank_delta >= 4 or partial_target_mass_delta > 0.00002:
            response_profile = "partial_scale_signal"
            next_probe = "refine_scale_sweep_near_best_partial"
        elif zero is not None and str(zero.get("actuator_class", "") or "") == "target_lift":
            response_profile = "zero_only_weak_target_lift"
            next_probe = "test_zero_with_repetition_and_semantic_guardrails"
        else:
            response_profile = "attention_scale_flat"
            next_probe = "do_not_promote_attention_scale_yet"
        shadow_source = best_partial if best_partial is not None else zero
        shadow_actuator = None
        shadow_actuator_class = None
        shadow_promotable = False
        shadow_promotion_reason = None
        if shadow_source is not None:
            shadow_scale = float(shadow_source["_scale"])
            shadow_actuator_class, shadow_promotable, shadow_promotion_reason = (
                _classify_attention_shadow_actuator(
                    response_profile=response_profile,
                    focus_rank_delta=int(shadow_source["_focus_rank_delta"]),
                    target_mass_delta=float(shadow_source["_target_mass_delta"]),
                    target_top20_hit_delta=int(shadow_source["_target_top20_hit_delta"]),
                    first_logit_delta_l2=float(shadow_source["_logit_l2"]),
                )
            )
            shadow_actuator = {
                "kind": "attention_shadow_actuator",
                "decision": "shadow",
                "objective_bundle_key": objective_key,
                "actuator_bundle_key": objective_key,
                "objective_term": shadow_source.get("objective_term"),
                "site": site,
                "layer": int(layer),
                "head": int(head),
                "scale": round(float(shadow_scale), 6),
                "recipe_name": shadow_source.get("recipe_name"),
                "response_profile": response_profile,
                "attention_shadow_actuator_class": shadow_actuator_class,
                "promotable_to_certification_replay": bool(shadow_promotable),
                "promotion_reason": shadow_promotion_reason,
                "counterfactual_delta": {
                    "focus_rank_delta": int(shadow_source["_focus_rank_delta"]),
                    "target_mass_delta": round(float(shadow_source["_target_mass_delta"]), 6),
                    "target_top20_hit_delta": int(shadow_source["_target_top20_hit_delta"]),
                    "first_logit_delta_l2": round(float(shadow_source["_logit_l2"]), 6),
                    "head_norm_before": round(_as_float(shadow_source.get("attention_head_norm_before", 0.0)), 6),
                    "head_norm_after": round(_as_float(shadow_source.get("attention_head_norm_after", 0.0)), 6),
                    "head_norm_removed_estimate": round(
                        _as_float(shadow_source.get("attention_head_norm_removed_estimate", 0.0)),
                        6,
                    ),
                },
                "production_apply_allowed": False,
                "certified_for_apply": False,
            }
        summaries.append(
            {
                "objective_bundle_key": objective_key,
                "objective_term": normalized[0].get("objective_term") if normalized else None,
                "site": site,
                "layer": int(layer),
                "head": int(head),
                "response_profile": response_profile,
                "next_probe": next_probe,
                "zero_actuator_class": None if zero is None else zero.get("actuator_class"),
                "zero_focus_rank_delta": int(zero_focus_rank_delta),
                "zero_target_mass_delta": round(float(zero_target_mass_delta), 6),
                "zero_target_top20_hit_delta": 0 if zero is None else int(zero["_target_top20_hit_delta"]),
                "zero_logit_delta_l2": round(float(zero_logit_l2), 6),
                "best_partial_recipe": None if best_partial is None else best_partial.get("recipe_name"),
                "best_partial_scale": None if best_partial is None else round(float(best_partial["_scale"]), 6),
                "best_partial_actuator_class": None if best_partial is None else best_partial.get("actuator_class"),
                "best_partial_focus_rank_delta": int(partial_focus_rank_delta),
                "best_partial_target_mass_delta": round(float(partial_target_mass_delta), 6),
                "best_partial_target_top20_hit_delta": 0
                if best_partial is None
                else int(best_partial["_target_top20_hit_delta"]),
                "partial_to_zero_logit_delta_ratio": partial_to_zero_logit_ratio,
                "attention_shadow_actuator_class": shadow_actuator_class,
                "attention_shadow_promotable_to_certification_replay": bool(shadow_promotable),
                "attention_shadow_promotion_reason": shadow_promotion_reason,
                "shadow_actuator": shadow_actuator,
                "scale_points": [
                    {
                        "scale": round(float(item["_scale"]), 6),
                        "recipe_name": item.get("recipe_name"),
                        "actuator_class": item.get("actuator_class"),
                        "focus_rank_delta": int(item["_focus_rank_delta"]),
                        "target_mass_delta": round(float(item["_target_mass_delta"]), 6),
                        "target_top20_hit_delta": int(item["_target_top20_hit_delta"]),
                        "first_logit_delta_l2": round(float(item["_logit_l2"]), 6),
                        "hook_call_count": _as_int(item.get("attention_hook_call_count", 0)),
                        "head_norm_before": round(_as_float(item.get("attention_head_norm_before", 0.0)), 6),
                        "head_norm_after": round(_as_float(item.get("attention_head_norm_after", 0.0)), 6),
                        "head_norm_removed_estimate": round(
                            _as_float(item.get("attention_head_norm_removed_estimate", 0.0)),
                            6,
                        ),
                    }
                    for item in normalized
                ],
            }
        )
    summaries.sort(
        key=lambda item: (
            str(item.get("objective_bundle_key", "") or ""),
            str(item.get("site", "") or ""),
            int(item.get("layer", 0) or 0),
            int(item.get("head", 0) or 0),
        )
    )
    return summaries


def _diagnostic_evidence_ledger(
    bridge_eval_summary: Mapping[str, Any],
    strategy_hints: Mapping[str, Any],
) -> dict[str, Any]:
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _append_unique(items: list[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)

    def _status_for(bundle_key: str) -> dict[str, Any]:
        return status.setdefault(
            bundle_key,
            {
                "bundle_key": bundle_key,
                "readout_reachable": False,
                "head_sensitive": False,
                "rank_readout_carrier": False,
                "feature_supported": False,
                "operator_certified": False,
                "diagnostic_operator_supported": False,
                "policy_candidate_ready": False,
                "production_operator_certified": False,
                "operator_mode_diagnosis": None,
                "best_single_mode": None,
                "best_pair_mode": None,
                "pair_interaction_delta": None,
                "attention_scale_response_profile": None,
                "attention_scale_next_probe": None,
                "attention_carrier_profile": None,
                "attention_carrier_next_probe": None,
                "attention_shadow_actuator": None,
                "attention_shadow_actuator_class": None,
                "attention_shadow_promotable_to_certification_replay": False,
                "attention_shadow_promotion_reason": None,
                "attention_guided_operator_checked": False,
                "attention_guided_operator_status": None,
                "attention_guided_operator_blocked_count": 0,
                "attention_guided_operator_recipe_count": 0,
                "activation_patch_checked": False,
                "activation_patch_status": None,
                "activation_patch_blocked_count": 0,
                "activation_patch_recipe_count": 0,
                "activation_patch_shadow_actuator": None,
                "activation_patch_actuator_class": None,
                "activation_patch_promotable_to_candidate": False,
                "activation_patch_promotion_reason": None,
                "carrier_to_actuator_status": None,
                "blocked_by": [],
                "evidence_count": 0,
                "evidence_kinds": [],
                "diagnostic_request": None,
                "next_evidence_needed": None,
                "reason_text": None,
            },
        )

    def _record(
        *,
        bundle_key: str,
        evidence_kind: str,
        diagnostic_family: str,
        actuator_class: str | None = None,
        evidence_status: str = "observed",
        recipe_name: str | None = None,
        operator_recipe_id: str | None = None,
        target_mass_delta: Any = None,
        target_top20_hit_delta: Any = None,
        focus_rank_delta: Any = None,
        blocked_by: str | None = None,
    ) -> None:
        if not bundle_key:
            return
        row = {
            "bundle_key": bundle_key,
            "evidence_kind": evidence_kind,
            "diagnostic_family": diagnostic_family,
            "status": evidence_status,
        }
        if actuator_class not in (None, ""):
            row["actuator_class"] = str(actuator_class)
        if recipe_name not in (None, ""):
            row["recipe_name"] = str(recipe_name)
        if operator_recipe_id not in (None, ""):
            row["operator_recipe_id"] = str(operator_recipe_id)
        if target_mass_delta is not None:
            row["target_mass_delta"] = round(_as_float(target_mass_delta), 6)
        if target_top20_hit_delta is not None:
            row["target_top20_hit_delta"] = _as_int(target_top20_hit_delta)
        if focus_rank_delta is not None:
            row["focus_rank_delta"] = _as_int(focus_rank_delta)
        if blocked_by not in (None, ""):
            row["blocked_by"] = str(blocked_by)
        ledger.append(row)
        bundle_status = _status_for(bundle_key)
        bundle_status["evidence_count"] = int(bundle_status["evidence_count"]) + 1
        _append_unique(bundle_status["evidence_kinds"], evidence_kind)
        if blocked_by not in (None, ""):
            _append_unique(bundle_status["blocked_by"], blocked_by)

    status: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []

    for raw_key in bridge_eval_summary.get("bundle_keys", ()):
        bundle_key = str(raw_key or "")
        if bundle_key:
            _status_for(bundle_key)
    for raw_key in (
        strategy_hints.get("selected_bundle_key"),
        strategy_hints.get("challenger_bundle_key"),
        strategy_hints.get("base_winner_bundle_key"),
        strategy_hints.get("gate_report_frontier_bundle_key"),
    ):
        bundle_key = str(raw_key or "")
        if bundle_key:
            _status_for(bundle_key)

    matrix = bridge_eval_summary.get("matrix", ())
    if isinstance(matrix, SequenceABC) and not isinstance(matrix, (str, bytes, bytearray)):
        for item in matrix:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("objective_bundle_key", "") or "")
            if not bundle_key:
                continue
            actuator_class = str(item.get("actuator_class", "") or "")
            bundle_status = _status_for(bundle_key)
            if actuator_class in {"self_actuator", "bridge_actuator"}:
                bundle_status["diagnostic_operator_supported"] = True
                evidence_status = "certified"
            elif actuator_class in {"dead_actuator", "collapse_sharpener"}:
                evidence_status = "blocked"
                _append_unique(bundle_status["blocked_by"], actuator_class)
            else:
                evidence_status = "observed"
            _record(
                bundle_key=bundle_key,
                evidence_kind="operator_replay",
                diagnostic_family="focused_ownership_replay",
                actuator_class=actuator_class or None,
                evidence_status=evidence_status,
                recipe_name=item.get("recipe_name"),
                operator_recipe_id=item.get("operator_recipe_id"),
                target_mass_delta=item.get("target_mass_delta"),
                target_top20_hit_delta=item.get("target_top20_hit_delta"),
                blocked_by=actuator_class if evidence_status == "blocked" else None,
            )

    mode_diagnostics = bridge_eval_summary.get("operator_recipe_mode_diagnostics", ())
    if isinstance(mode_diagnostics, SequenceABC) and not isinstance(mode_diagnostics, (str, bytes, bytearray)):
        for item in mode_diagnostics:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("intended_bundle_key", "") or "")
            if not bundle_key:
                continue
            diagnosis = str(item.get("diagnosis", "") or "")
            bundle_status = _status_for(bundle_key)
            if diagnosis:
                bundle_status["operator_mode_diagnosis"] = diagnosis
            if item.get("best_single_mode") not in (None, ""):
                bundle_status["best_single_mode"] = str(item.get("best_single_mode"))
            if item.get("best_pair_mode") not in (None, ""):
                bundle_status["best_pair_mode"] = str(item.get("best_pair_mode"))
            if item.get("pair_interaction_delta") is not None:
                bundle_status["pair_interaction_delta"] = round(_as_float(item.get("pair_interaction_delta")), 6)
            if diagnosis == "all_dead_or_weak":
                _append_unique(bundle_status["blocked_by"], "all_modes_dead_or_weak")
            best_any = item.get("best_any") if isinstance(item.get("best_any"), Mapping) else {}
            _record(
                bundle_key=bundle_key,
                evidence_kind="operator_mode_decomposition",
                diagnostic_family="kv_mode_decomposition",
                actuator_class=diagnosis or None,
                evidence_status="blocked" if diagnosis == "all_dead_or_weak" else "observed",
                recipe_name=item.get("recipe_name"),
                operator_recipe_id=best_any.get("operator_recipe_id") if isinstance(best_any, Mapping) else None,
                target_mass_delta=best_any.get("target_mass_delta") if isinstance(best_any, Mapping) else None,
                target_top20_hit_delta=best_any.get("target_top20_hit_delta") if isinstance(best_any, Mapping) else None,
                blocked_by="all_modes_dead_or_weak" if diagnosis == "all_dead_or_weak" else None,
            )
            ledger[-1]["best_single_mode"] = item.get("best_single_mode")
            ledger[-1]["best_pair_mode"] = item.get("best_pair_mode")
            ledger[-1]["pair_interaction_delta"] = item.get("pair_interaction_delta")
            ledger[-1]["pair_alignment_delta"] = item.get("pair_alignment_delta")

    attention_response = bridge_eval_summary.get("attention_scale_response_diagnostics", ())
    if isinstance(attention_response, SequenceABC) and not isinstance(attention_response, (str, bytes, bytearray)):
        for item in attention_response:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("objective_bundle_key", "") or "")
            if not bundle_key:
                continue
            profile = str(item.get("response_profile", "") or "")
            bundle_status = _status_for(bundle_key)
            if profile:
                bundle_status["attention_scale_response_profile"] = profile
                bundle_status["attention_carrier_profile"] = profile
            if item.get("next_probe") not in (None, ""):
                bundle_status["attention_scale_next_probe"] = str(item.get("next_probe"))
                bundle_status["attention_carrier_next_probe"] = str(item.get("next_probe"))
            shadow_actuator = item.get("shadow_actuator") if isinstance(item.get("shadow_actuator"), Mapping) else None
            if shadow_actuator is not None:
                bundle_status["attention_shadow_actuator"] = dict(shadow_actuator)
                bundle_status["attention_shadow_actuator_class"] = shadow_actuator.get(
                    "attention_shadow_actuator_class"
                )
                bundle_status["attention_shadow_promotable_to_certification_replay"] = bool(
                    shadow_actuator.get("promotable_to_certification_replay", False)
                )
                bundle_status["attention_shadow_promotion_reason"] = shadow_actuator.get("promotion_reason")
            if profile in {"attention_scale_flat", "unsafe_zero_response"}:
                _append_unique(bundle_status["blocked_by"], profile)
            if profile in {"zero_only_rank_lift", "zero_only_weak_target_lift", "partial_scale_signal"}:
                bundle_status["rank_readout_carrier"] = True
                bundle_status["head_sensitive"] = True
            _record(
                bundle_key=bundle_key,
                evidence_kind="attention_readout_carrier",
                diagnostic_family="attention_head_scale_response",
                actuator_class=profile or None,
                evidence_status="supportive"
                if profile in {"zero_only_rank_lift", "zero_only_weak_target_lift", "partial_scale_signal"}
                else "blocked",
                recipe_name=item.get("best_partial_recipe") or "head_zero_ablation",
                target_mass_delta=item.get("zero_target_mass_delta"),
                target_top20_hit_delta=item.get("zero_target_top20_hit_delta"),
                focus_rank_delta=item.get("zero_focus_rank_delta"),
                blocked_by=profile if profile in {"attention_scale_flat", "unsafe_zero_response"} else None,
            )
            ledger[-1]["response_profile"] = profile
            ledger[-1]["next_probe"] = item.get("next_probe")
            ledger[-1]["zero_focus_rank_delta"] = item.get("zero_focus_rank_delta")
            ledger[-1]["best_partial_focus_rank_delta"] = item.get("best_partial_focus_rank_delta")
            ledger[-1]["partial_to_zero_logit_delta_ratio"] = item.get("partial_to_zero_logit_delta_ratio")
            ledger[-1]["attention_shadow_actuator_class"] = item.get("attention_shadow_actuator_class")
            ledger[-1]["promotable_to_certification_replay"] = item.get(
                "attention_shadow_promotable_to_certification_replay"
            )
            ledger[-1]["promotion_reason"] = item.get("attention_shadow_promotion_reason")
            if shadow_actuator is not None:
                ledger[-1]["shadow_actuator"] = dict(shadow_actuator)
                delta = shadow_actuator.get("counterfactual_delta")
                if isinstance(delta, Mapping):
                    ledger[-1]["counterfactual_delta"] = dict(delta)
                _record(
                    bundle_key=bundle_key,
                    evidence_kind="attention_shadow_actuator",
                    diagnostic_family="attention_head_scale_shadow",
                    actuator_class=profile or None,
                    evidence_status="shadow",
                    recipe_name=shadow_actuator.get("recipe_name"),
                    target_mass_delta=(delta or {}).get("target_mass_delta") if isinstance(delta, Mapping) else None,
                    target_top20_hit_delta=(delta or {}).get("target_top20_hit_delta") if isinstance(delta, Mapping) else None,
                    focus_rank_delta=(delta or {}).get("focus_rank_delta") if isinstance(delta, Mapping) else None,
                )
                ledger[-1]["shadow_actuator"] = dict(shadow_actuator)
                ledger[-1]["attention_shadow_actuator_class"] = shadow_actuator.get(
                    "attention_shadow_actuator_class"
                )
                ledger[-1]["promotable_to_certification_replay"] = shadow_actuator.get(
                    "promotable_to_certification_replay"
                )
                ledger[-1]["promotion_reason"] = shadow_actuator.get("promotion_reason")
                if isinstance(delta, Mapping):
                    ledger[-1]["counterfactual_delta"] = dict(delta)

    extras = bridge_eval_summary.get("extra_operator_diagnostics", ())
    if isinstance(extras, SequenceABC) and not isinstance(extras, (str, bytes, bytearray)):
        for item in extras:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("objective_bundle_key", "") or "")
            if not bundle_key:
                continue
            diagnostic_family = str(item.get("diagnostic_family", "") or "")
            actuator_class = str(item.get("actuator_class", "") or "")
            target_mass_delta = _as_float(item.get("target_mass_delta", 0.0))
            target_top20_hit_delta = _as_int(item.get("target_top20_hit_delta", 0))
            focus_rank_delta = _as_int(item.get("focus_rank_delta", 0))
            bundle_status = _status_for(bundle_key)
            evidence_kind = "operator_probe"
            evidence_status = "observed"
            activation_shadow_actuator: dict[str, Any] | None = None
            if diagnostic_family == "logit_adjacent":
                evidence_kind = "readout_probe"
                if actuator_class == "target_lift" or target_mass_delta > 0.0 or target_top20_hit_delta > 0 or focus_rank_delta > 0:
                    bundle_status["readout_reachable"] = True
                    evidence_status = "supportive"
            elif diagnostic_family == "attention_head_ablation":
                evidence_kind = "attention_ablation"
                if actuator_class not in {"dead_actuator", "collapse_sharpener", ""} or focus_rank_delta > 0:
                    bundle_status["head_sensitive"] = True
                    evidence_status = "supportive"
            elif diagnostic_family == "attention_head_scale":
                evidence_kind = "attention_readout_carrier"
                if actuator_class not in {"dead_actuator", "collapse_sharpener", ""} or focus_rank_delta > 0:
                    bundle_status["rank_readout_carrier"] = True
                    bundle_status["head_sensitive"] = True
                    evidence_status = "supportive"
            elif diagnostic_family == "attention_guided_operator":
                evidence_kind = "attention_guided_operator_certification"
                bundle_status["rank_readout_carrier"] = True
                bundle_status["head_sensitive"] = True
                bundle_status["attention_guided_operator_checked"] = True
                bundle_status["attention_guided_operator_recipe_count"] = int(
                    bundle_status.get("attention_guided_operator_recipe_count", 0) or 0
                ) + 1
                if actuator_class in {"target_lift", "self_actuator", "bridge_actuator"} or target_mass_delta > 0.0 or target_top20_hit_delta > 0:
                    evidence_status = "supportive"
                    bundle_status["attention_guided_operator_status"] = "supportive"
                    bundle_status["diagnostic_operator_supported"] = True
                elif actuator_class in {"dead_actuator", "collapse_sharpener", "harmful"}:
                    evidence_status = "blocked"
                    bundle_status["attention_guided_operator_status"] = "blocked"
                    bundle_status["attention_guided_operator_blocked_count"] = int(
                        bundle_status.get("attention_guided_operator_blocked_count", 0) or 0
                    ) + 1
                    _append_unique(bundle_status["blocked_by"], actuator_class)
                    _append_unique(bundle_status["blocked_by"], "attention_guided_operator_blocked")
            elif diagnostic_family == "activation_patch":
                evidence_kind = "activation_patch_certification"
                bundle_status["activation_patch_checked"] = True
                bundle_status["activation_patch_recipe_count"] = int(
                    bundle_status.get("activation_patch_recipe_count", 0) or 0
                ) + 1
                if actuator_class in {"self_actuator", "bridge_actuator"}:
                    evidence_status = "supportive"
                    bundle_status["activation_patch_status"] = "supportive"
                    bundle_status["diagnostic_operator_supported"] = True
                    delta = {
                        "target_mass_delta": round(float(target_mass_delta), 6),
                        "target_top20_hit_delta": int(target_top20_hit_delta),
                        "focus_rank_delta": int(focus_rank_delta),
                        "self_delta": round(_as_float(item.get("self_delta", 0.0)), 6),
                        "cross_delta": round(_as_float(item.get("cross_delta", 0.0)), 6),
                        "alignment_margin": round(_as_float(item.get("alignment_margin", 0.0)), 6),
                    }
                    promotion_reason = (
                        "owned_activation_patch_self_actuator"
                        if actuator_class == "self_actuator"
                        else "activation_patch_bridge_requires_dual_layer_plan"
                    )
                    activation_shadow_actuator = {
                        "kind": "activation_patch_shadow_actuator",
                        "decision": "shadow",
                        "objective_bundle_key": bundle_key,
                        "actuator_bundle_key": str(item.get("actuator_bundle_key", "") or bundle_key),
                        "objective_term": item.get("objective_term"),
                        "recipe_name": item.get("recipe_name"),
                        "operator_recipe_id": item.get("operator_recipe_id"),
                        "activation_patch_site": item.get("activation_patch_site"),
                        "activation_patch_layer": item.get("activation_patch_layer"),
                        "activation_patch_alpha": item.get("activation_patch_alpha"),
                        "activation_patch_source_localization": item.get("activation_patch_source_localization"),
                        "activation_patch_patch_mode": item.get("activation_patch_patch_mode"),
                        "activation_patch_base_localization": item.get("activation_patch_base_localization"),
                        "activation_patch_contrast_mode": item.get("activation_patch_contrast_mode"),
                        "activation_patch_contrast_scale": item.get("activation_patch_contrast_scale"),
                        "activation_patch_stealer_bundle_key": item.get("activation_patch_stealer_bundle_key"),
                        "activation_patch_stealer_term": item.get("activation_patch_stealer_term"),
                        "activation_patch_actuator_class": actuator_class,
                        "actual_delta_class": item.get("actual_delta_class"),
                        "realized_lift_bundle_key": item.get("realized_lift_bundle_key"),
                        "realized_lift_term": item.get("realized_lift_term"),
                        "counterfactual_delta": delta,
                        "promotable_to_candidate_compiler": bool(actuator_class == "self_actuator"),
                        "promotion_reason": promotion_reason,
                        "production_apply_allowed": False,
                        "certified_for_apply": False,
                    }
                    existing_shadow = bundle_status.get("activation_patch_shadow_actuator")
                    existing_delta = (
                        existing_shadow.get("counterfactual_delta", {})
                        if isinstance(existing_shadow, Mapping)
                        else {}
                    )
                    def _activation_source_specificity(value: Any) -> int:
                        text = str(value or "")
                        if text.startswith("source_term_token"):
                            return 2
                        if text.startswith("source_centered_pm1"):
                            return 1
                        return 0

                    source_specificity = _activation_source_specificity(
                        item.get("activation_patch_source_localization")
                    )
                    existing_source_specificity = (
                        _activation_source_specificity(existing_shadow.get("activation_patch_source_localization"))
                        if isinstance(existing_shadow, Mapping)
                        else 0
                    )
                    shadow_rank = (
                        2 if actuator_class == "self_actuator" else 1,
                        1 if str(item.get("actual_delta_class", "") or "") == "target_lift" else 0,
                        source_specificity,
                        float(delta.get("self_delta", 0.0) or 0.0)
                        if actuator_class == "self_actuator"
                        else float(delta.get("cross_delta", 0.0) or 0.0),
                        float(delta.get("alignment_margin", 0.0) or 0.0),
                    )
                    existing_rank = (
                        2
                        if str(existing_shadow.get("activation_patch_actuator_class", "") or "") == "self_actuator"
                        else 1,
                        1 if str(existing_shadow.get("actual_delta_class", "") or "") == "target_lift" else 0,
                        existing_source_specificity,
                        _as_float(existing_delta.get("self_delta", 0.0))
                        if str(existing_shadow.get("activation_patch_actuator_class", "") or "") == "self_actuator"
                        else _as_float(existing_delta.get("cross_delta", 0.0)),
                        _as_float(existing_delta.get("alignment_margin", 0.0)),
                    ) if isinstance(existing_shadow, Mapping) else (-1, -1, -1, -1.0, -10.0)
                    if (
                        not isinstance(existing_shadow, Mapping)
                        or shadow_rank > existing_rank
                    ):
                        bundle_status["activation_patch_shadow_actuator"] = dict(activation_shadow_actuator)
                        bundle_status["activation_patch_actuator_class"] = actuator_class
                        bundle_status["activation_patch_promotable_to_candidate"] = bool(
                            activation_shadow_actuator["promotable_to_candidate_compiler"]
                        )
                        bundle_status["activation_patch_promotion_reason"] = promotion_reason
                elif actuator_class in {"dead_actuator", "collapse_sharpener", "harmful", "cross_bound"}:
                    evidence_status = "blocked"
                    bundle_status["activation_patch_status"] = "blocked"
                    bundle_status["activation_patch_blocked_count"] = int(
                        bundle_status.get("activation_patch_blocked_count", 0) or 0
                    ) + 1
                    _append_unique(bundle_status["blocked_by"], actuator_class)
                    _append_unique(bundle_status["blocked_by"], "activation_patch_blocked")
            elif actuator_class in {"dead_actuator", "collapse_sharpener"}:
                evidence_status = "blocked"
                _append_unique(bundle_status["blocked_by"], actuator_class)
            _record(
                bundle_key=bundle_key,
                evidence_kind=evidence_kind,
                diagnostic_family=diagnostic_family,
                actuator_class=actuator_class or None,
                evidence_status=evidence_status,
                recipe_name=item.get("recipe_name"),
                operator_recipe_id=item.get("operator_recipe_id"),
                target_mass_delta=item.get("target_mass_delta"),
                target_top20_hit_delta=item.get("target_top20_hit_delta"),
                focus_rank_delta=item.get("focus_rank_delta"),
                blocked_by=actuator_class if evidence_status == "blocked" else None,
            )
            if item.get("operator_axis") not in (None, ""):
                ledger[-1]["operator_axis"] = str(item.get("operator_axis"))
            if activation_shadow_actuator is not None:
                ledger[-1]["activation_patch_shadow_actuator"] = dict(activation_shadow_actuator)
                ledger[-1]["activation_patch_actuator_class"] = actuator_class
                ledger[-1]["activation_patch_promotable_to_candidate"] = bool(
                    activation_shadow_actuator.get("promotable_to_candidate_compiler", False)
                )
                ledger[-1]["activation_patch_promotion_reason"] = activation_shadow_actuator.get(
                    "promotion_reason"
                )
            for activation_key in (
                "activation_hook_call_count",
                "activation_patch_site",
                "activation_patch_layer",
                "activation_patch_alpha",
                "activation_patch_source_localization",
                "activation_patch_patch_mode",
                "activation_patch_base_localization",
                "activation_patch_contrast_mode",
                "activation_patch_contrast_scale",
                "activation_patch_stealer_bundle_key",
                "activation_patch_stealer_term",
                "activation_source_norm",
                "activation_target_norm_before",
                "activation_target_norm_after",
                "actual_delta_class",
                "realized_lift_bundle_key",
                "realized_lift_term",
                "self_delta",
                "cross_delta",
                "alignment_margin",
                "bundle_lift_scores",
            ):
                if item.get(activation_key) not in (None, ""):
                    ledger[-1][activation_key] = item.get(activation_key)

    analyzer_hints = strategy_hints.get("readout_analyzer_hints")
    if not isinstance(analyzer_hints, Mapping):
        analyzer_hints = strategy_hints.get("readout_sidecar_hints")
    if not isinstance(analyzer_hints, Mapping):
        analyzer_hints = {}
    bundle_support_scores = analyzer_hints.get("bundle_support_scores")
    if isinstance(bundle_support_scores, Mapping):
        for raw_key, raw_score in list(bundle_support_scores.items())[:8]:
            bundle_key = str(raw_key or "")
            score = _as_float(raw_score, default=-10.0)
            if not bundle_key:
                continue
            if score > 0.0:
                _status_for(bundle_key)["feature_supported"] = True
            _record(
                bundle_key=bundle_key,
                evidence_kind="feature_emitter",
                diagnostic_family="readout_analyzer_support",
                evidence_status="supportive" if score > 0.0 else "observed",
                target_mass_delta=None,
                recipe_name="bundle_support_score",
            )
            ledger[-1]["support_score"] = round(float(score), 6)
    sae_feature_hints = analyzer_hints.get("sae_feature_hints")
    if isinstance(sae_feature_hints, SequenceABC) and not isinstance(sae_feature_hints, (str, bytes, bytearray)):
        for item in sae_feature_hints[:6]:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("bundle_key", "") or "")
            support = _as_float(item.get("support", 0.0))
            if not bundle_key:
                continue
            if support > 0.0:
                _status_for(bundle_key)["feature_supported"] = True
            _record(
                bundle_key=bundle_key,
                evidence_kind="feature_emitter",
                diagnostic_family="sae_feature_emitter",
                evidence_status="supportive" if support > 0.0 else "observed",
                recipe_name=item.get("feature_family"),
            )
            ledger[-1]["support_score"] = round(float(support), 6)
            if item.get("operator_family_prior") not in (None, ""):
                ledger[-1]["operator_family_prior"] = str(item.get("operator_family_prior"))
            raw_priors = item.get("operator_family_priors")
            if isinstance(raw_priors, SequenceABC) and not isinstance(raw_priors, (str, bytes, bytearray)):
                priors = [str(raw_prior) for raw_prior in raw_priors[:4] if str(raw_prior or "")]
                if priors:
                    ledger[-1]["operator_family_priors"] = priors

    unavailable_reasons = bridge_eval_summary.get("unavailable_objective_reasons")
    if isinstance(unavailable_reasons, Mapping):
        for raw_key, raw_reason in unavailable_reasons.items():
            bundle_key = str(raw_key or "")
            reason = str(raw_reason or "")
            if bundle_key and reason:
                _append_unique(_status_for(bundle_key)["blocked_by"], reason)

    def _finalize(bundle_key: str, bundle_status: dict[str, Any]) -> None:
        diagnostic_operator_supported = bool(bundle_status.get("diagnostic_operator_supported", False))
        guided_checked = bool(bundle_status.get("attention_guided_operator_checked", False))
        guided_blocked = (
            guided_checked
            and not diagnostic_operator_supported
            and str(bundle_status.get("attention_guided_operator_status", "") or "") == "blocked"
        )
        activation_patch_checked = bool(bundle_status.get("activation_patch_checked", False))
        activation_patch_blocked = (
            activation_patch_checked
            and not diagnostic_operator_supported
            and str(bundle_status.get("activation_patch_status", "") or "") == "blocked"
        )
        if guided_blocked and bundle_status["rank_readout_carrier"]:
            bundle_status["carrier_to_actuator_status"] = "missing_after_attention_guided_operator"
        if activation_patch_blocked and not diagnostic_operator_supported:
            bundle_status["carrier_to_actuator_status"] = "missing_after_activation_patch"
        if not bundle_status["readout_reachable"]:
            bundle_status["next_evidence_needed"] = "readout_reachable_signal"
            bundle_status["diagnostic_request"] = "readout_logit_adjacent_probe"
        elif not bundle_status["rank_readout_carrier"]:
            bundle_status["next_evidence_needed"] = "rank_readout_carrier_signal"
            bundle_status["diagnostic_request"] = "attention_readout_carrier_probe"
        elif (
            bool(bundle_status.get("attention_shadow_promotable_to_certification_replay", False))
            and not diagnostic_operator_supported
            and not guided_checked
        ):
            bundle_status["next_evidence_needed"] = "attention_guided_operator_certification"
            bundle_status["diagnostic_request"] = "operator_diagnostic_replay"
        elif isinstance(bundle_status.get("activation_patch_shadow_actuator"), Mapping):
            bundle_status["next_evidence_needed"] = "activation_patch_candidate_compiler_review"
            bundle_status["diagnostic_request"] = "activation_patch_candidate_review"
        elif guided_blocked:
            bundle_status["next_evidence_needed"] = "carrier_to_actuator_bridge_missing"
            bundle_status["diagnostic_request"] = "compare_extra_operator_diagnostics"
        elif activation_patch_blocked:
            bundle_status["next_evidence_needed"] = "non_kv_operator_search"
            bundle_status["diagnostic_request"] = "compare_extra_operator_diagnostics"
        elif not bundle_status["feature_supported"]:
            bundle_status["next_evidence_needed"] = "readout_feature_emitter_support"
            bundle_status["diagnostic_request"] = "sae_feature_emitter_scan"
        elif not diagnostic_operator_supported:
            bundle_status["next_evidence_needed"] = "certified_self_or_bridge_actuator"
            bundle_status["diagnostic_request"] = "operator_diagnostic_replay"
        else:
            bundle_status["next_evidence_needed"] = "production_policy_review"
            bundle_status["diagnostic_request"] = "none"
        blocked = ", ".join(str(item) for item in bundle_status["blocked_by"][:3])
        guided_note = ""
        if guided_checked:
            guided_note = (
                f"; attention_guided={bundle_status.get('attention_guided_operator_status')}"
                f"({bundle_status.get('attention_guided_operator_blocked_count', 0)}/"
                f"{bundle_status.get('attention_guided_operator_recipe_count', 0)} blocked)"
            )
        patch_note = ""
        if activation_patch_checked:
            patch_note = (
                f"; activation_patch={bundle_status.get('activation_patch_status')}"
                f"({bundle_status.get('activation_patch_blocked_count', 0)}/"
                f"{bundle_status.get('activation_patch_recipe_count', 0)} blocked)"
            )
        bundle_status["reason_text"] = (
            f"{bundle_key} needs {bundle_status['next_evidence_needed']}"
            + (f"; blocked_by={blocked}" if blocked else "")
            + guided_note
            + patch_note
        )

    for key, bundle_status in status.items():
        _finalize(key, bundle_status)

    frontier_bundle_key = str(
        strategy_hints.get(
            "gate_report_frontier_bundle_key",
            strategy_hints.get("selected_bundle_key", ""),
        )
        or ""
    )
    if not frontier_bundle_key or frontier_bundle_key not in status:
        frontier_bundle_key = next(iter(status), "")
    frontier_status = dict(status.get(frontier_bundle_key, {})) if frontier_bundle_key else {}
    visible_ledger = sorted(
        ledger,
        key=lambda row: (
            0 if str(row.get("evidence_kind", "") or "") == "operator_mode_decomposition" else 1,
            str(row.get("bundle_key", "") or ""),
            str(row.get("recipe_name", "") or ""),
            str(row.get("evidence_kind", "") or ""),
        ),
    )
    return {
        "diagnostic_evidence_ledger": visible_ledger[:48],
        "diagnostic_evidence_ledger_count": len(ledger),
        "bundle_diagnostic_status": status,
        "diagnostic_frontier_bundle_key": frontier_bundle_key or None,
        "diagnostic_frontier_status": frontier_status,
        "diagnostic_frontier_next_evidence": frontier_status.get("next_evidence_needed"),
        "diagnostic_frontier_request": frontier_status.get("diagnostic_request"),
        "diagnostic_frontier_reason_text": frontier_status.get("reason_text"),
    }


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
    if normalized == "sae_scaffold":
        return build_sae_feature_emitter_readout_analyzer()
    raise ValueError(f"unknown readout analyzer mode '{mode}'")


def create_readout_sidecar_analyzer(mode: str) -> ReadoutSidecarAnalyzer | None:
    return create_readout_analyzer(mode)


class _FrontierReplayControllerClient:
    def __init__(self, *, replay_mode: str = "frontier_apply"):
        self.replay_mode = str(replay_mode).strip().lower().replace("-", "_")
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

    @staticmethod
    def _diagnostic_next_action(diagnostic_name: str) -> str:
        normalized = str(diagnostic_name or "").strip().lower().replace("-", "_")
        if normalized in {"attention_head_ablation_on_frontier", "attention_readout_carrier_probe"}:
            return "request_attention_readout_carrier_probe"
        if normalized == "sae_feature_emitter_scan":
            return "request_sae_feature_scan"
        if normalized == "activation_patch_candidate_review":
            return "request_activation_patch_candidate_review"
        if normalized == "activation_patch_runtime_support_probe":
            return "request_activation_patch_runtime_support_probe"
        if normalized == "activation_patch_promotion_gate_review":
            return "request_activation_patch_promotion_gate_review"
        if normalized == "activation_patch_production_shadow_replay":
            return "request_activation_patch_production_shadow_replay"
        if normalized == "activation_patch_production_trial_gate_review":
            return "request_activation_patch_production_trial_gate_review"
        if normalized == "compare_extra_operator_diagnostics":
            return "request_compare_extra_operator_diagnostics"
        return "request_operator_diagnostic"

    @staticmethod
    def _compact_diagnostic_result(result: Mapping[str, Any]) -> dict[str, Any]:
        evidence_rows = result.get("evidence_rows")
        evidence_row_count = (
            len(evidence_rows)
            if isinstance(evidence_rows, SequenceABC) and not isinstance(evidence_rows, (str, bytes, bytearray))
            else 0
        )
        return {
            "diagnostic": result.get("diagnostic"),
            "bundle_key": result.get("bundle_key"),
            "objective_bundle_key": result.get("objective_bundle_key"),
            "step_actuator_bundle_key": result.get("step_actuator_bundle_key"),
            "next_evidence_needed": result.get("next_evidence_needed"),
            "blocked_by": list(result.get("blocked_by", ()))
            if isinstance(result.get("blocked_by"), SequenceABC)
            and not isinstance(result.get("blocked_by"), (str, bytes, bytearray))
            else [],
            "operator_certified": result.get("operator_certified"),
            "diagnostic_operator_supported": result.get("diagnostic_operator_supported"),
            "policy_candidate_ready": result.get("policy_candidate_ready"),
            "production_operator_certified": result.get("production_operator_certified"),
            "rank_readout_carrier": result.get("rank_readout_carrier"),
            "attention_carrier_profile": result.get("attention_carrier_profile"),
            "attention_shadow_actuator": result.get("attention_shadow_actuator"),
            "attention_shadow_actuator_class": result.get("attention_shadow_actuator_class"),
            "attention_shadow_promotable_to_certification_replay": result.get(
                "attention_shadow_promotable_to_certification_replay"
            ),
            "attention_shadow_promotion_reason": result.get("attention_shadow_promotion_reason"),
            "activation_patch_compile_preview_created": result.get("activation_patch_compile_preview_created"),
            "activation_patch_compile_preview_blocked_reason": result.get(
                "activation_patch_compile_preview_blocked_reason"
            ),
            "activation_patch_compile_preview": result.get("activation_patch_compile_preview"),
            "activation_patch_runtime_support_status": result.get("activation_patch_runtime_support_status"),
            "activation_patch_diagnostic_executable_created": result.get(
                "activation_patch_diagnostic_executable_created"
            ),
            "activation_patch_executable_shadow": result.get("activation_patch_executable_shadow"),
            "activation_patch_runtime_supported_shadow_candidate": result.get(
                "activation_patch_runtime_supported_shadow_candidate"
            ),
            "activation_patch_promotion_gate_passed": result.get("activation_patch_promotion_gate_passed"),
            "activation_patch_production_apply_candidate": result.get(
                "activation_patch_production_apply_candidate"
            ),
            "activation_patch_production_shadow_replay": result.get(
                "activation_patch_production_shadow_replay"
            ),
            "activation_patch_production_shadow_dossier": result.get(
                "activation_patch_production_shadow_dossier"
            ),
            "activation_patch_production_shadow_replay_passed": result.get(
                "activation_patch_production_shadow_replay_passed"
            ),
            "activation_patch_production_trial_gate_review": result.get(
                "activation_patch_production_trial_gate_review"
            ),
            "activation_patch_production_trial_dossier": result.get(
                "activation_patch_production_trial_dossier"
            ),
            "activation_patch_production_trial_candidate": result.get(
                "activation_patch_production_trial_candidate"
            ),
            "activation_patch_production_trial_contract": result.get(
                "activation_patch_production_trial_contract"
            ),
            "production_trial_allowed": result.get("production_trial_allowed"),
            "production_trial_blocked_reasons": result.get("production_trial_blocked_reasons"),
            "activation_patch_production_denial_dossier": result.get(
                "activation_patch_production_denial_dossier"
            ),
            "production_denial_reasons": result.get("production_denial_reasons"),
            "production_denial_reasons_after_shadow": result.get("production_denial_reasons_after_shadow"),
            "production_apply_allowed": result.get("production_apply_allowed"),
            "evidence_row_count": evidence_row_count,
        }

    @staticmethod
    def _activation_patch_compile_preview_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            preview = result.get("activation_patch_compile_preview")
            if isinstance(preview, Mapping):
                return dict(preview)
            review = result.get("activation_patch_candidate_review")
            if isinstance(review, Mapping) and isinstance(review.get("compile_preview"), Mapping):
                return dict(review["compile_preview"])
        return None

    @staticmethod
    def _activation_patch_runtime_support_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            probe = result.get("activation_patch_runtime_support_probe")
            if isinstance(probe, Mapping):
                return dict(probe)
            executable = result.get("activation_patch_executable_shadow")
            if isinstance(executable, Mapping):
                return {
                    "status": result.get("activation_patch_runtime_support_status"),
                    "executable_shadow": dict(executable),
                    "runtime_supported_shadow_candidate": result.get(
                        "activation_patch_runtime_supported_shadow_candidate"
                    ),
                    "next_evidence_needed": result.get("next_evidence_needed"),
                }
        return None

    @staticmethod
    def _activation_patch_promotion_gate_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            review = result.get("activation_patch_promotion_gate_review")
            if isinstance(review, Mapping):
                return dict(review)
            candidate = result.get("activation_patch_production_apply_candidate")
            if isinstance(candidate, Mapping):
                return {
                    "status": "candidate_ready_for_policy_review",
                    "production_apply_candidate": dict(candidate),
                    "gate_passed": result.get("activation_patch_promotion_gate_passed"),
                    "production_denial_dossier": result.get("activation_patch_production_denial_dossier"),
                    "production_denial_reasons": result.get("production_denial_reasons"),
                    "next_evidence_needed": result.get("next_evidence_needed"),
                }
        return None

    @staticmethod
    def _activation_patch_production_shadow_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            replay = result.get("activation_patch_production_shadow_replay")
            if isinstance(replay, Mapping):
                return dict(replay)
            dossier = result.get("activation_patch_production_shadow_dossier")
            if isinstance(dossier, Mapping):
                return {
                    "status": "shadow_passed" if result.get("activation_patch_production_shadow_replay_passed") else "blocked",
                    "shadow_replay_passed": bool(result.get("activation_patch_production_shadow_replay_passed", False)),
                    "production_shadow_dossier": dict(dossier),
                    "production_denial_reasons_after_shadow": result.get(
                        "production_denial_reasons_after_shadow"
                    ),
                    "next_evidence_needed": result.get("next_evidence_needed"),
                }
        return None

    @staticmethod
    def _activation_patch_production_trial_gate_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            review = result.get("activation_patch_production_trial_gate_review")
            if isinstance(review, Mapping):
                return dict(review)
            dossier = result.get("activation_patch_production_trial_dossier")
            if isinstance(dossier, Mapping):
                return {
                    "status": "trial_ready" if result.get("production_trial_allowed") else "blocked",
                    "production_trial_allowed": bool(result.get("production_trial_allowed", False)),
                    "production_trial_dossier": dict(dossier),
                    "production_trial_candidate": result.get("activation_patch_production_trial_candidate"),
                    "production_trial_contract": result.get("activation_patch_production_trial_contract"),
                    "production_trial_budget_class": result.get("production_trial_budget_class"),
                    "production_trial_followup_allowed": result.get("production_trial_followup_allowed"),
                    "production_trial_blocked_reasons": result.get("production_trial_blocked_reasons"),
                    "next_evidence_needed": result.get("next_evidence_needed"),
                    "why_not_apply": result.get("why_not_apply"),
                }
        return None

    def _recent_production_trial_effect_seen(
        self,
        packet: Mapping[str, Any],
        *,
        objective_bundle_key: str,
        operator_recipe_id: Any,
    ) -> bool:
        return self._production_trial_outcome_from_packet(
            packet,
            objective_bundle_key=objective_bundle_key,
            operator_recipe_id=operator_recipe_id,
        ) is not None

    @staticmethod
    def _production_trial_outcome_from_packet(
        packet: Mapping[str, Any],
        *,
        objective_bundle_key: str,
        operator_recipe_id: Any,
    ) -> dict[str, Any] | None:
        summary = packet.get("recent_effect_summary")
        if not isinstance(summary, Mapping):
            return None
        raw_items = summary.get("production_trial_outcome_ledger")
        latest = summary.get("latest_effects")
        items: list[Any] = []
        if isinstance(raw_items, SequenceABC) and not isinstance(raw_items, (str, bytes, bytearray)):
            items.extend(raw_items)
        if isinstance(latest, SequenceABC) and not isinstance(latest, (str, bytes, bytearray)):
            items.extend(latest)
        recipe_text = str(operator_recipe_id or "")
        for effect in reversed(items):
            if not isinstance(effect, Mapping):
                continue
            if str(effect.get("apply_kind", "") or "") != "production_trial":
                continue
            effect_objective = str(effect.get("objective_bundle_key") or effect.get("bundle_key") or "")
            effect_recipe = str(effect.get("operator_recipe_id") or "")
            if objective_bundle_key and effect_objective and effect_objective != objective_bundle_key:
                continue
            if recipe_text and effect_recipe and effect_recipe != recipe_text:
                continue
            return dict(effect)
        return None

    @staticmethod
    def _production_trial_followup_from_outcome(outcome: Mapping[str, Any]) -> dict[str, str]:
        trial_class = str(outcome.get("trial_effect_class") or outcome.get("verdict") or "unknown")
        actuator_class = str(outcome.get("actuator_class", "unknown") or "unknown")
        surface_family = str(outcome.get("surface_family_key", "") or "")
        if trial_class in {"harmful", "regressing", "collapse_sharpener"}:
            return {
                "next_evidence_needed": "alternate_operator_recipe_after_trial",
                "next_action": "request_compare_extra_operator_diagnostics",
                "diagnostic": "compare_extra_operator_diagnostics",
                "blocked_by": f"production_trial_{trial_class}",
                "promotion_ladder_stage": "trial_blocked_needs_alternate_operator",
                "alternate_trial_candidate_plan": "compare_extra_operator_diagnostics",
                "why_not_apply": (
                    f"bounded production_trial returned {trial_class}/{actuator_class}; "
                    f"veto this recipe family"
                    + (f" on {surface_family}" if surface_family else "")
                    + " and request alternate operator evidence"
                ),
            }
        if trial_class == "neutral":
            return {
                "next_evidence_needed": "confirm_or_alternate_trial_candidate",
                "next_action": "request_compare_extra_operator_diagnostics",
                "diagnostic": "compare_extra_operator_diagnostics",
                "blocked_by": "production_trial_neutral",
                "promotion_ladder_stage": "trial_neutral_needs_alternate_candidate",
                "alternate_trial_candidate_plan": "compare_extra_operator_diagnostics",
                "why_not_apply": "bounded production_trial was neutral; keep production closed and compare alternate operator recipes",
            }
        if trial_class == "helpful":
            return {
                "next_evidence_needed": "production_trial_confirmation_replay",
                "next_action": "request_activation_patch_production_shadow_replay",
                "diagnostic": "activation_patch_production_shadow_replay",
                "blocked_by": "production_trial_needs_confirmation",
                "promotion_ladder_stage": "trial_supported_needs_confirmation_replay",
                "alternate_trial_candidate_plan": "confirm_same_trial_before_promotion",
                "why_not_apply": "bounded production_trial was helpful; require confirmation replay before any production policy review",
            }
        return {
            "next_evidence_needed": "production_trial_effect_followup",
            "next_action": "request_compare_extra_operator_diagnostics",
            "diagnostic": "compare_extra_operator_diagnostics",
            "blocked_by": "production_trial_unknown",
            "promotion_ladder_stage": "trial_unknown_needs_followup",
            "alternate_trial_candidate_plan": "compare_extra_operator_diagnostics",
            "why_not_apply": "bounded production_trial outcome was unknown; keep production closed and request alternate evidence",
        }

    @staticmethod
    def _diagnostic_seen_in_results(results: Sequence[Any], diagnostic_name: str) -> bool:
        expected = str(diagnostic_name or "")
        if not expected:
            return False
        for result in results:
            if not isinstance(result, Mapping):
                continue
            if str(result.get("diagnostic", "") or "") == expected:
                return True
        return False

    @staticmethod
    def _production_trial_alternate_evidence_from_results(
        results: Sequence[Any],
        *,
        objective_bundle_key: str,
    ) -> dict[str, Any] | None:
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            if str(result.get("diagnostic", "") or "") != "compare_extra_operator_diagnostics":
                continue
            result_objective = str(result.get("objective_bundle_key") or result.get("bundle_key") or "")
            if objective_bundle_key and result_objective and result_objective != objective_bundle_key:
                continue
            summary = result.get("diagnostic_summary")
            evidence_rows = result.get("evidence_rows")
            return {
                "diagnostic": "compare_extra_operator_diagnostics",
                "objective_bundle_key": result_objective or objective_bundle_key,
                "status": result.get("status"),
                "next_evidence_needed": result.get("next_evidence_needed"),
                "diagnostic_operator_supported": result.get("diagnostic_operator_supported"),
                "policy_candidate_ready": result.get("policy_candidate_ready"),
                "production_apply_allowed": bool(result.get("production_apply_allowed", False)),
                "evidence_row_count": (
                    len(evidence_rows)
                    if isinstance(evidence_rows, SequenceABC)
                    and not isinstance(evidence_rows, (str, bytes, bytearray))
                    else result.get("evidence_row_count")
                ),
                "diagnostic_summary": dict(summary) if isinstance(summary, Mapping) else None,
            }
        return None

    @staticmethod
    def _alternate_trial_candidate_from_compare_results(
        results: Sequence[Any],
        *,
        objective_bundle_key: str,
        blocked_operator_recipe_id: Any,
    ) -> dict[str, Any] | None:
        blocked_recipe = str(blocked_operator_recipe_id or "")
        def _recipe_family(recipe_id: Any) -> str:
            parts = [part for part in str(recipe_id or "").split("|") if part]
            if parts and parts[-1].startswith("a"):
                try:
                    float(parts[-1][1:])
                    parts = parts[:-1]
                except Exception:
                    pass
            return "|".join(parts)

        def _recipe_structural_family(recipe_id: Any) -> str:
            parts = [part for part in str(recipe_id or "").split("|") if part]
            if parts and parts[-1].startswith("a"):
                try:
                    float(parts[-1][1:])
                    parts = parts[:-1]
                except Exception:
                    pass
            if len(parts) >= 6 and parts[1] == "activation_patch":
                return "|".join([parts[0], parts[1], *parts[4:]])
            return "|".join(parts)

        blocked_recipe_family = _recipe_family(blocked_recipe)
        blocked_structural_family = _recipe_structural_family(blocked_recipe)
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return default

        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            if str(result.get("diagnostic", "") or "") != "compare_extra_operator_diagnostics":
                continue
            rows = result.get("activation_patch_candidate_pool")
            if not isinstance(rows, SequenceABC) or isinstance(rows, (str, bytes, bytearray)):
                rows = result.get("evidence_rows")
            if not isinstance(rows, SequenceABC) or isinstance(rows, (str, bytes, bytearray)):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("evidence_kind", "") or "") != "activation_patch_certification":
                    continue
                row_objective = str(row.get("bundle_key") or result.get("objective_bundle_key") or "")
                if objective_bundle_key and row_objective and row_objective != objective_bundle_key:
                    continue
                recipe_id = str(row.get("operator_recipe_id") or "")
                if blocked_recipe and recipe_id == blocked_recipe:
                    continue
                recipe_family = _recipe_family(recipe_id)
                same_failed_family = bool(
                    blocked_recipe_family and recipe_family and recipe_family == blocked_recipe_family
                )
                structural_family = _recipe_structural_family(recipe_id)
                same_failed_structural_family = bool(
                    blocked_structural_family
                    and structural_family
                    and structural_family == blocked_structural_family
                )
                actuator_class = str(row.get("actuator_class") or row.get("activation_patch_actuator_class") or "")
                actual_delta_class = str(row.get("actual_delta_class") or "")
                if actuator_class in {"dead_actuator", "collapse_sharpener", "harmful", "cross_bound"}:
                    continue
                if actuator_class != "self_actuator":
                    continue
                target_mass_delta = _as_float(row.get("target_mass_delta"))
                target_top20_delta = _as_int(row.get("target_top20_hit_delta"))
                focus_rank_delta = _as_int(row.get("focus_rank_delta"))
                self_delta = _as_float(row.get("self_delta"))
                cross_delta = _as_float(row.get("cross_delta"))
                alignment_margin = _as_float(row.get("alignment_margin"))
                if actual_delta_class not in {"target_lift", "self_actuator"} and max(
                    target_mass_delta,
                    float(target_top20_delta),
                    float(focus_rank_delta),
                    self_delta,
                ) <= 0.0:
                    continue
                alpha = _as_float(row.get("activation_patch_alpha"), 0.0)
                candidate = {
                    "kind": "activation_patch_shadow_actuator",
                    "decision": "shadow",
                    "objective_bundle_key": objective_bundle_key or row_objective or None,
                    "actuator_bundle_key": str(row.get("actuator_bundle_key") or objective_bundle_key or row_objective or "") or None,
                    "objective_term": row.get("realized_lift_term") or row.get("objective_term"),
                    "recipe_name": row.get("recipe_name"),
                    "operator_recipe_id": recipe_id or None,
                    "activation_patch_site": row.get("activation_patch_site"),
                    "activation_patch_layer": row.get("activation_patch_layer"),
                    "activation_patch_alpha": alpha,
                    "activation_patch_source_localization": row.get("activation_patch_source_localization"),
                    "activation_patch_patch_mode": row.get("activation_patch_patch_mode"),
                    "activation_patch_base_localization": row.get("activation_patch_base_localization"),
                    "activation_patch_contrast_mode": row.get("activation_patch_contrast_mode"),
                    "activation_patch_contrast_scale": row.get("activation_patch_contrast_scale"),
                    "activation_patch_stealer_bundle_key": row.get("activation_patch_stealer_bundle_key"),
                    "activation_patch_stealer_term": row.get("activation_patch_stealer_term"),
                    "activation_patch_actuator_class": actuator_class,
                    "actual_delta_class": actual_delta_class or None,
                    "realized_lift_bundle_key": row.get("realized_lift_bundle_key"),
                    "realized_lift_term": row.get("realized_lift_term"),
                    "counterfactual_delta": {
                        "target_mass_delta": round(float(target_mass_delta), 6),
                        "target_top20_hit_delta": int(target_top20_delta),
                        "focus_rank_delta": int(focus_rank_delta),
                        "self_delta": round(float(self_delta), 6),
                        "cross_delta": round(float(cross_delta), 6),
                        "alignment_margin": round(float(alignment_margin), 6),
                    },
                    "promotable_to_candidate_compiler": True,
                    "promotion_reason": "alternate_after_failed_production_trial",
                    "production_apply_allowed": False,
                    "certified_for_apply": False,
                    "operator_recipe_family_key": recipe_family or None,
                    "blocked_recipe_family_key": blocked_recipe_family or None,
                    "same_failed_recipe_family": bool(same_failed_family),
                    "operator_recipe_structural_family_key": structural_family or None,
                    "blocked_recipe_structural_family_key": blocked_structural_family or None,
                    "same_failed_recipe_structural_family": bool(same_failed_structural_family),
                }
                score = (
                    0.0 if same_failed_structural_family else 1.0,
                    0.0 if same_failed_family else 1.0,
                    1.0 if actual_delta_class == "target_lift" else 0.0,
                    alignment_margin,
                    self_delta,
                    float(focus_rank_delta) * 0.01,
                    target_mass_delta * 1000.0,
                    float(target_top20_delta) * 0.1,
                    -alpha,
                )
                candidates.append((score, candidate))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _activation_patch_shadow_from_preview_results(
        results: Sequence[Any],
        *,
        preview: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        recipe_id = str(preview.get("operator_recipe_id") or "")
        if not recipe_id:
            return None
        for result in reversed(list(results)):
            if not isinstance(result, Mapping):
                continue
            review = result.get("activation_patch_candidate_review")
            if not isinstance(review, Mapping):
                continue
            review_preview = review.get("compile_preview")
            if not isinstance(review_preview, Mapping):
                continue
            if str(review_preview.get("operator_recipe_id") or "") != recipe_id:
                continue
            return {
                "kind": "activation_patch_shadow_actuator",
                "decision": "shadow",
                "objective_bundle_key": preview.get("objective_bundle_key"),
                "actuator_bundle_key": preview.get("actuator_bundle_key") or preview.get("objective_bundle_key"),
                "objective_term": preview.get("objective_term"),
                "recipe_name": preview.get("recipe_name"),
                "operator_recipe_id": recipe_id,
                "activation_patch_site": preview.get("site"),
                "activation_patch_layer": preview.get("layer"),
                "activation_patch_alpha": preview.get("alpha"),
                "activation_patch_source_localization": preview.get("source_localization"),
                "activation_patch_patch_mode": preview.get("patch_mode"),
                "activation_patch_base_localization": preview.get("base_localization"),
                "activation_patch_contrast_mode": preview.get("contrast_mode"),
                "activation_patch_contrast_scale": preview.get("contrast_scale"),
                "activation_patch_stealer_bundle_key": preview.get("stealer_bundle_key"),
                "activation_patch_stealer_term": preview.get("stealer_term"),
                "activation_patch_actuator_class": review.get("actuator_class") or "self_actuator",
                "actual_delta_class": review.get("actual_delta_class"),
                "counterfactual_delta": dict(review.get("counterfactual_delta", {}))
                if isinstance(review.get("counterfactual_delta"), Mapping)
                else {},
                "promotable_to_candidate_compiler": bool(
                    review.get("promotable_to_candidate_compiler", True)
                ),
                "promotion_reason": review.get("promotion_reason") or "alternate_candidate_review_preview",
                "production_apply_allowed": False,
                "certified_for_apply": False,
            }
        return {
            "kind": "activation_patch_shadow_actuator",
            "decision": "shadow",
            "objective_bundle_key": preview.get("objective_bundle_key"),
            "actuator_bundle_key": preview.get("actuator_bundle_key") or preview.get("objective_bundle_key"),
            "objective_term": preview.get("objective_term"),
            "recipe_name": preview.get("recipe_name"),
            "operator_recipe_id": recipe_id,
            "activation_patch_site": preview.get("site"),
            "activation_patch_layer": preview.get("layer"),
            "activation_patch_alpha": preview.get("alpha"),
            "activation_patch_source_localization": preview.get("source_localization"),
            "activation_patch_patch_mode": preview.get("patch_mode"),
            "activation_patch_base_localization": preview.get("base_localization"),
            "activation_patch_contrast_mode": preview.get("contrast_mode"),
            "activation_patch_contrast_scale": preview.get("contrast_scale"),
            "activation_patch_stealer_bundle_key": preview.get("stealer_bundle_key"),
            "activation_patch_stealer_term": preview.get("stealer_term"),
            "activation_patch_actuator_class": preview.get("actuator_class") or "self_actuator",
            "actual_delta_class": preview.get("actual_delta_class"),
            "counterfactual_delta": {},
            "promotable_to_candidate_compiler": True,
            "promotion_reason": "alternate_preview_fallback",
            "production_apply_allowed": False,
            "certified_for_apply": False,
        }

    def _production_trial_apply_command(
        self,
        *,
        packet: Mapping[str, Any],
        objective_bundle_key: str,
        step_actuator_bundle_key: str,
        trial_gate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(trial_gate.get("production_trial_allowed", False)):
            return None
        trial_candidate = trial_gate.get("production_trial_candidate")
        if not isinstance(trial_candidate, Mapping):
            return None
        trial_edit = trial_candidate.get("trial_edit")
        if not isinstance(trial_edit, Mapping):
            return None
        trial_budget_class = str(
            trial_gate.get("production_trial_budget_class")
            or trial_candidate.get("production_trial_budget_class")
            or "primary"
        )
        trial_budget_class = "alternate_followup" if trial_budget_class == "alternate_followup" else "primary"
        budget = packet.get("budget")
        if isinstance(budget, Mapping):
            try:
                trial_edits_left = int(
                    budget.get(
                        "production_trial_followup_edits_left_this_run"
                        if trial_budget_class == "alternate_followup"
                        else "production_trial_edits_left_this_run",
                        0,
                    )
                    or 0
                )
            except Exception:
                trial_edits_left = 0
            if trial_edits_left <= 0:
                return None
        operator_recipe_id = trial_candidate.get("operator_recipe_id")
        if self._recent_production_trial_effect_seen(
            packet,
            objective_bundle_key=objective_bundle_key,
            operator_recipe_id=operator_recipe_id,
        ):
            return None
        edit = dict(trial_edit)
        edit_meta = dict(edit.get("meta", {})) if isinstance(edit.get("meta"), Mapping) else {}
        edit_meta.update(
            {
                "apply_kind": "production_trial",
                "production_trial_budget_class": trial_budget_class,
                "production_trial_followup_allowed": trial_budget_class == "alternate_followup",
                "production_trial_allowed": True,
                "production_apply_allowed": False,
                "production_policy_would_apply": False,
                "certified_for_apply": False,
                "objective_bundle_key": objective_bundle_key,
                "step_actuator_bundle_key": step_actuator_bundle_key,
                "bundle_key": objective_bundle_key,
            }
        )
        edit["meta"] = edit_meta
        return {
            "version": "0.1",
            "decision": "apply",
            "edits": [edit],
            "meta": {
                "apply_kind": "production_trial",
                "hypothesis": "activation_patch_production_trial_apply",
                "micro_rationale": "Run one bounded production trial; production apply remains closed.",
                "objective_bundle_key": objective_bundle_key,
                "step_actuator_bundle_key": step_actuator_bundle_key,
                "operator_recipe_id": operator_recipe_id,
                "production_trial_budget_class": trial_budget_class,
                "production_trial_followup_allowed": trial_budget_class == "alternate_followup",
                "surface_family_key": edit_meta.get("surface_family_key"),
                "production_trial_allowed": True,
                "production_apply_allowed": False,
                "production_policy_would_apply": False,
                "certified_for_apply": False,
                "why_not_apply": "this is a bounded production_trial apply, not production apply permission",
                "next_evidence_needed": "production_trial_effect",
                "next_action": "noop",
                "activation_patch_production_trial_candidate": dict(trial_candidate),
                "activation_patch_production_trial_contract": trial_candidate.get("production_trial_contract"),
                "controller_memory": {
                    "objective_bundle_key": objective_bundle_key,
                    "step_actuator_bundle_key": step_actuator_bundle_key,
                    "next_evidence_needed": "production_trial_effect",
                    "next_action": "noop",
                    "production_trial_budget_class": trial_budget_class,
                    "production_trial_followup_allowed": trial_budget_class == "alternate_followup",
                },
            },
        }

    @staticmethod
    def _activation_patch_compile_preview_blocked_reason_from_results(results: Sequence[Any]) -> str | None:
        for result in results:
            if not isinstance(result, Mapping):
                continue
            reason = result.get("activation_patch_compile_preview_blocked_reason")
            if reason not in (None, ""):
                return str(reason)
            review = result.get("activation_patch_candidate_review")
            if isinstance(review, Mapping) and review.get("compile_preview_blocked_reason") not in (None, ""):
                return str(review["compile_preview_blocked_reason"])
        return None

    @staticmethod
    def _attention_shadow_from_results(results: Sequence[Any]) -> dict[str, Any] | None:
        for result in results:
            if not isinstance(result, Mapping):
                continue
            shadow = result.get("attention_shadow_actuator")
            if isinstance(shadow, Mapping):
                return dict(shadow)
            rows = result.get("evidence_rows")
            if not isinstance(rows, SequenceABC) or isinstance(rows, (str, bytes, bytearray)):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                shadow = row.get("shadow_actuator")
                if isinstance(shadow, Mapping):
                    return dict(shadow)
        return None

    @staticmethod
    def _should_request_attention_shadow_round(results: Sequence[Any], diagnostic_budget_left: Any) -> bool:
        try:
            budget_left = int(diagnostic_budget_left)
        except Exception:
            budget_left = 0
        if budget_left <= 0:
            return False
        saw_operator_block = False
        saw_attention_round = False
        saw_carrier = False
        for result in results:
            if not isinstance(result, Mapping):
                continue
            diagnostic = str(result.get("diagnostic", "") or "")
            if diagnostic == "attention_readout_carrier_probe":
                saw_attention_round = True
            if diagnostic == "operator_diagnostic_replay" and not bool(
                result.get("diagnostic_operator_supported", result.get("operator_certified", False))
            ):
                saw_operator_block = True
            if bool(result.get("rank_readout_carrier", False)) or isinstance(result.get("attention_shadow_actuator"), Mapping):
                saw_carrier = True
        return saw_operator_block and saw_carrier and not saw_attention_round

    def _diagnostic_request_command(
        self,
        *,
        packet: Mapping[str, Any],
        strategy_hints: Mapping[str, Any],
        frontier_bundle_key: Any,
        suggested_bundle_key: Any,
    ) -> dict[str, Any]:
        objective_bundle_key = str(
            strategy_hints.get(
                "diagnostic_frontier_bundle_key",
                frontier_bundle_key or suggested_bundle_key or strategy_hints.get("selected_bundle_key", ""),
            )
            or ""
        )
        step_actuator_bundle_key = str(
            strategy_hints.get("bridge_plan_actuator_bundle_key", objective_bundle_key) or objective_bundle_key
        )
        diagnostic_name = str(strategy_hints.get("diagnostic_frontier_request", "") or "")
        if diagnostic_name in {"", "none"}:
            diagnostic_name = "operator_diagnostic_replay"
        next_evidence = str(strategy_hints.get("diagnostic_frontier_next_evidence", "") or "")
        if not next_evidence:
            next_evidence = "certified_self_or_bridge_actuator"
        why_not_apply = str(strategy_hints.get("diagnostic_frontier_reason_text", "") or "")
        if not why_not_apply:
            why_not_apply = "frontier visible but no certified actuator; request bounded diagnostic evidence"
        diagnostic_request = {
            "diagnostic": diagnostic_name,
            "bundle_key": objective_bundle_key,
            "objective_bundle_key": objective_bundle_key,
            "step_actuator_bundle_key": step_actuator_bundle_key,
            "next_evidence_needed": next_evidence,
            "reason": why_not_apply,
        }
        latest_results = packet.get("latest_diagnostic_results")
        has_latest_results = (
            isinstance(latest_results, SequenceABC)
            and not isinstance(latest_results, (str, bytes, bytearray))
            and bool(latest_results)
        )
        telemetry = packet.get("telemetry") if isinstance(packet.get("telemetry"), Mapping) else {}
        latest_result_items = list(latest_results) if has_latest_results else []
        activation_patch_preview = self._activation_patch_compile_preview_from_results(latest_result_items)
        activation_patch_runtime_support = self._activation_patch_runtime_support_from_results(latest_result_items)
        activation_patch_promotion_gate = self._activation_patch_promotion_gate_from_results(latest_result_items)
        activation_patch_production_shadow = self._activation_patch_production_shadow_from_results(
            latest_result_items
        )
        activation_patch_production_trial_gate = self._activation_patch_production_trial_gate_from_results(
            latest_result_items
        )
        activation_patch_preview_blocked_reason = self._activation_patch_compile_preview_blocked_reason_from_results(
            latest_result_items
        )
        attention_shadow = self._attention_shadow_from_results(latest_result_items)
        request_attention_shadow_round = self._should_request_attention_shadow_round(
            latest_result_items,
            telemetry.get("diagnostic_call_budget_left", 0),
        )
        request_activation_patch_runtime_round = False
        request_activation_patch_promotion_round = False
        request_activation_patch_production_shadow_round = False
        request_activation_patch_production_trial_round = False
        request_production_trial_followup_round = False
        production_trial_outcome: dict[str, Any] | None = self._production_trial_outcome_from_packet(
            packet,
            objective_bundle_key=objective_bundle_key,
            operator_recipe_id=None,
        )
        production_trial_followup: dict[str, str] | None = (
            self._production_trial_followup_from_outcome(production_trial_outcome)
            if isinstance(production_trial_outcome, Mapping)
            else None
        )
        production_trial_alternate_evidence = self._production_trial_alternate_evidence_from_results(
            latest_result_items,
            objective_bundle_key=objective_bundle_key,
        )
        production_trial_alternate_candidate: dict[str, Any] | None = None
        production_trial_blocked_recipe_id = (
            production_trial_outcome.get("operator_recipe_id")
            if isinstance(production_trial_outcome, Mapping)
            else None
        )
        production_trial_outcome_holds_activation_patch_pipeline = False
        activation_patch_executable_shadow = (
            dict(activation_patch_runtime_support["executable_shadow"])
            if isinstance(activation_patch_runtime_support, Mapping)
            and isinstance(activation_patch_runtime_support.get("executable_shadow"), Mapping)
            else None
        )
        activation_patch_production_candidate = (
            dict(activation_patch_promotion_gate["production_apply_candidate"])
            if isinstance(activation_patch_promotion_gate, Mapping)
            and isinstance(activation_patch_promotion_gate.get("production_apply_candidate"), Mapping)
            else None
        )
        activation_patch_preview_recipe_id = (
            activation_patch_preview.get("operator_recipe_id")
            if isinstance(activation_patch_preview, Mapping)
            else None
        )
        activation_patch_preview_is_alternate_after_trial = bool(
            production_trial_outcome is not None
            and activation_patch_preview_recipe_id not in (None, "")
            and str(activation_patch_preview_recipe_id) != str(production_trial_blocked_recipe_id or "")
        )
        if production_trial_outcome is not None and production_trial_followup is not None:
            followup_diagnostic = production_trial_followup.get("diagnostic", "compare_extra_operator_diagnostics")
            followup_seen = self._diagnostic_seen_in_results(latest_result_items, followup_diagnostic)
            production_trial_alternate_candidate = self._alternate_trial_candidate_from_compare_results(
                latest_result_items,
                objective_bundle_key=objective_bundle_key,
                blocked_operator_recipe_id=production_trial_blocked_recipe_id,
            )
            production_trial_outcome_holds_activation_patch_pipeline = not activation_patch_preview_is_alternate_after_trial
            if not followup_seen:
                request_attention_shadow_round = False
                request_production_trial_followup_round = True
                diagnostic_name = followup_diagnostic
                next_evidence = production_trial_followup["next_evidence_needed"]
                why_not_apply = production_trial_followup["why_not_apply"]
                diagnostic_request = {
                    "diagnostic": diagnostic_name,
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "step_actuator_bundle_key": step_actuator_bundle_key,
                    "operator_recipe_id": production_trial_outcome.get("operator_recipe_id"),
                    "next_evidence_needed": next_evidence,
                    "reason": why_not_apply,
                    "production_trial_outcome": dict(production_trial_outcome),
                }
            elif (
                production_trial_alternate_candidate is not None
                and not activation_patch_preview_is_alternate_after_trial
            ):
                request_attention_shadow_round = False
                request_production_trial_followup_round = True
                diagnostic_name = "activation_patch_candidate_review"
                next_evidence = "alternate_activation_patch_candidate_review"
                why_not_apply = (
                    "production trial followup found an alternate activation-patch recipe; "
                    "review it as shadow-only before any runtime support probe"
                )
                diagnostic_request = {
                    "diagnostic": diagnostic_name,
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "step_actuator_bundle_key": step_actuator_bundle_key,
                    "activation_patch_shadow_actuator": dict(production_trial_alternate_candidate),
                    "alternate_trial_candidate": dict(production_trial_alternate_candidate),
                    "next_evidence_needed": next_evidence,
                    "reason": why_not_apply,
                    "production_trial_outcome": dict(production_trial_outcome),
                }
            else:
                next_evidence = "alternate_trial_candidate_generator"
                why_not_apply = (
                    "production trial followup evidence was received; hold the old activation_patch "
                    "runtime-support loop until an alternate operator candidate consumes it"
                )
        activation_patch_request_shadow: dict[str, Any] | None = None
        if isinstance(activation_patch_preview, Mapping):
            if production_trial_alternate_candidate is not None and activation_patch_preview_is_alternate_after_trial:
                activation_patch_request_shadow = dict(production_trial_alternate_candidate)
            else:
                activation_patch_request_shadow = self._activation_patch_shadow_from_preview_results(
                    latest_result_items,
                    preview=activation_patch_preview,
                )
        if activation_patch_preview is not None and not production_trial_outcome_holds_activation_patch_pipeline:
            request_attention_shadow_round = False
            if activation_patch_production_shadow is not None:
                shadow_passed = bool(activation_patch_production_shadow.get("shadow_replay_passed", False))
                if shadow_passed and activation_patch_production_trial_gate is None:
                    request_activation_patch_production_trial_round = True
                    diagnostic_name = "activation_patch_production_trial_gate_review"
                    next_evidence = "production_trial_gate_review"
                    why_not_apply = (
                        "activation_patch production shadow passed; review bounded trial contract without opening production apply"
                    )
                    diagnostic_request = {
                        "diagnostic": diagnostic_name,
                        "bundle_key": objective_bundle_key,
                        "objective_bundle_key": objective_bundle_key,
                        "step_actuator_bundle_key": step_actuator_bundle_key,
                        "next_evidence_needed": next_evidence,
                        "reason": why_not_apply,
                    }
                    if activation_patch_request_shadow is not None:
                        diagnostic_request["activation_patch_shadow_actuator"] = dict(
                            activation_patch_request_shadow
                        )
                        if activation_patch_preview_is_alternate_after_trial:
                            diagnostic_request["alternate_trial_candidate"] = dict(
                                activation_patch_request_shadow
                            )
                elif activation_patch_production_trial_gate is not None:
                    trial_allowed = bool(activation_patch_production_trial_gate.get("production_trial_allowed", False))
                    trial_candidate = activation_patch_production_trial_gate.get("production_trial_candidate")
                    trial_operator_recipe_id = (
                        trial_candidate.get("operator_recipe_id")
                        if isinstance(trial_candidate, Mapping)
                        else None
                    )
                    trial_outcome = self._production_trial_outcome_from_packet(
                        packet,
                        objective_bundle_key=objective_bundle_key,
                        operator_recipe_id=trial_operator_recipe_id,
                    )
                    production_trial_outcome = dict(trial_outcome) if isinstance(trial_outcome, Mapping) else None
                    trial_apply_command = self._production_trial_apply_command(
                        packet=packet,
                        objective_bundle_key=objective_bundle_key,
                        step_actuator_bundle_key=step_actuator_bundle_key,
                        trial_gate=activation_patch_production_trial_gate,
                    )
                    if trial_apply_command is not None:
                        return trial_apply_command
                    if trial_outcome is not None:
                        followup = self._production_trial_followup_from_outcome(trial_outcome)
                        production_trial_followup = dict(followup)
                        request_production_trial_followup_round = True
                        diagnostic_name = followup["diagnostic"]
                        next_evidence = followup["next_evidence_needed"]
                        why_not_apply = followup["why_not_apply"]
                        diagnostic_request = {
                            "diagnostic": diagnostic_name,
                            "bundle_key": objective_bundle_key,
                            "objective_bundle_key": objective_bundle_key,
                            "step_actuator_bundle_key": step_actuator_bundle_key,
                            "operator_recipe_id": trial_operator_recipe_id,
                            "next_evidence_needed": next_evidence,
                            "reason": why_not_apply,
                            "production_trial_outcome": dict(trial_outcome),
                        }
                    else:
                        next_evidence = str(
                            activation_patch_production_trial_gate.get(
                                "next_evidence_needed",
                                "bounded_production_trial" if trial_allowed else "production_trial_gate_followup",
                            )
                            or ("bounded_production_trial" if trial_allowed else "production_trial_gate_followup")
                        )
                        why_not_apply = (
                            "activation_patch production trial gate is ready only for a bounded trial; production apply remains closed"
                            if trial_allowed
                            else activation_patch_production_trial_gate.get(
                                "why_not_apply",
                                "activation_patch production trial gate remains blocked",
                            )
                        )
                else:
                    next_evidence = str(
                        activation_patch_production_shadow.get(
                            "next_evidence_needed",
                            "production_policy_review" if shadow_passed else "production_shadow_replay_followup",
                        )
                        or ("production_policy_review" if shadow_passed else "production_shadow_replay_followup")
                    )
                    why_not_apply = (
                        "activation_patch production shadow replay passed, but production apply permission remains closed"
                        if shadow_passed
                        else activation_patch_production_shadow.get(
                            "why_not_apply",
                            "activation_patch production shadow replay remains blocked",
                        )
                    )
            elif activation_patch_promotion_gate is not None:
                next_evidence = str(
                    activation_patch_promotion_gate.get("next_evidence_needed", "production_policy_review")
                    or "production_policy_review"
                )
                if activation_patch_production_candidate is not None:
                    request_activation_patch_production_shadow_round = True
                    diagnostic_name = "activation_patch_production_shadow_replay"
                    next_evidence = "production_shadow_replay"
                    why_not_apply = (
                        "activation_patch production candidate needs same-packet shadow ownership and context replay"
                    )
                    diagnostic_request = {
                        "diagnostic": diagnostic_name,
                        "bundle_key": objective_bundle_key,
                        "objective_bundle_key": objective_bundle_key,
                        "step_actuator_bundle_key": step_actuator_bundle_key,
                        "next_evidence_needed": next_evidence,
                        "reason": why_not_apply,
                    }
                    if activation_patch_request_shadow is not None:
                        diagnostic_request["activation_patch_shadow_actuator"] = dict(
                            activation_patch_request_shadow
                        )
                        if activation_patch_preview_is_alternate_after_trial:
                            diagnostic_request["alternate_trial_candidate"] = dict(
                                activation_patch_request_shadow
                            )
                else:
                    why_not_apply = activation_patch_promotion_gate.get(
                        "why_not_apply",
                        "activation patch promotion gate did not produce a production candidate",
                    )
            elif activation_patch_runtime_support is None:
                request_activation_patch_runtime_round = True
                diagnostic_name = "activation_patch_runtime_support_probe"
                next_evidence = "activation_patch_runtime_operator_certification"
                why_not_apply = (
                    "activation_patch_compile_preview needs quarantine executable support before any promotion gate"
                )
                alternate_runtime_shadow = (
                    production_trial_alternate_candidate
                    if production_trial_alternate_candidate is not None
                    else self._activation_patch_shadow_from_preview_results(
                        latest_result_items,
                        preview=activation_patch_preview,
                    )
                    if activation_patch_preview_is_alternate_after_trial
                    and isinstance(activation_patch_preview, Mapping)
                    else None
                )
                diagnostic_request = {
                    "diagnostic": diagnostic_name,
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "step_actuator_bundle_key": step_actuator_bundle_key,
                    "next_evidence_needed": next_evidence,
                    "reason": why_not_apply,
                }
                if activation_patch_request_shadow is not None:
                    diagnostic_request["activation_patch_shadow_actuator"] = dict(
                        activation_patch_request_shadow
                    )
                    if activation_patch_preview_is_alternate_after_trial:
                        diagnostic_request["alternate_trial_candidate"] = dict(
                            activation_patch_request_shadow
                        )
                if (
                    activation_patch_preview_is_alternate_after_trial
                    and alternate_runtime_shadow is not None
                ):
                    diagnostic_request["activation_patch_shadow_actuator"] = dict(alternate_runtime_shadow)
                    diagnostic_request["alternate_trial_candidate"] = dict(alternate_runtime_shadow)
                    production_trial_alternate_candidate = dict(alternate_runtime_shadow)
            elif activation_patch_promotion_gate is None and bool(
                activation_patch_runtime_support.get("runtime_supported_shadow_candidate", False)
            ):
                request_activation_patch_promotion_round = True
                diagnostic_name = "activation_patch_promotion_gate_review"
                next_evidence = "activation_patch_promotion_gate_review"
                why_not_apply = (
                    "activation_patch executable shadow needs promotion gate review before policy can consider it"
                )
                diagnostic_request = {
                    "diagnostic": diagnostic_name,
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "step_actuator_bundle_key": step_actuator_bundle_key,
                    "next_evidence_needed": next_evidence,
                    "reason": why_not_apply,
                }
                if activation_patch_request_shadow is not None:
                    diagnostic_request["activation_patch_shadow_actuator"] = dict(
                        activation_patch_request_shadow
                    )
                    if activation_patch_preview_is_alternate_after_trial:
                        diagnostic_request["alternate_trial_candidate"] = dict(
                            activation_patch_request_shadow
                        )
            else:
                next_evidence = str(
                    (
                        activation_patch_promotion_gate.get("next_evidence_needed")
                        if isinstance(activation_patch_promotion_gate, Mapping)
                        else activation_patch_runtime_support.get(
                            "next_evidence_needed",
                            "activation_patch_promotion_gate_review",
                        )
                    )
                    or "activation_patch_promotion_gate_review"
                )
                why_not_apply = (
                    "activation patch production candidate still requires policy review and production apply permission"
                    if activation_patch_production_candidate is not None
                    else "activation_patch executable shadow is diagnostic-only; production apply remains closed"
                )
        elif activation_patch_preview_blocked_reason is not None:
            next_evidence = "alternate_activation_patch_or_bridge_evidence"
            why_not_apply = activation_patch_preview_blocked_reason
        proposal_attention_shadow = None if activation_patch_preview is not None else attention_shadow
        if request_attention_shadow_round:
            diagnostic_name = "attention_readout_carrier_probe"
            next_evidence = "attention_shadow_actuator_counterfactual"
            diagnostic_request = {
                "diagnostic": diagnostic_name,
                "bundle_key": objective_bundle_key,
                "objective_bundle_key": objective_bundle_key,
                "step_actuator_bundle_key": step_actuator_bundle_key,
                "next_evidence_needed": next_evidence,
                "reason": "operator remains uncertified; inspect attention shadow actuator as carrier evidence only",
            }
        next_action = (
            self._diagnostic_next_action(diagnostic_name)
            if (
                request_activation_patch_runtime_round
                or request_activation_patch_promotion_round
                or request_activation_patch_production_shadow_round
                or request_activation_patch_production_trial_round
                or request_production_trial_followup_round
                or request_attention_shadow_round
                or not has_latest_results
            )
            else "noop"
        )
        meta: dict[str, Any] = {
            "hypothesis": "request_bounded_readout_escape_diagnostic",
            "micro_rationale": "frontier is visible, but apply stays blocked until diagnostic evidence is returned",
            "objective_bundle_key": objective_bundle_key,
            "step_actuator_bundle_key": step_actuator_bundle_key,
            "why_not_apply": why_not_apply,
            "next_evidence_needed": next_evidence,
            "next_action": next_action,
            "production_apply_allowed": False,
            "certified_for_apply": False,
            "shadow_proposals": [
                {
                    "kind": (
                        "activation_patch_production_shadow_replay"
                        if activation_patch_production_shadow is not None and activation_patch_production_trial_gate is None
                        else "activation_patch_production_trial_gate_review"
                        if activation_patch_production_trial_gate is not None
                        else "activation_patch_production_apply_candidate"
                        if activation_patch_production_candidate is not None
                        else "activation_patch_executable_shadow"
                        if activation_patch_executable_shadow is not None
                        else "activation_patch_compile_preview"
                        if activation_patch_preview is not None
                        else "attention_shadow_actuator"
                        if proposal_attention_shadow is not None
                        else "frontier_shadow"
                    ),
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "actuator_bundle_key": (
                        activation_patch_production_candidate.get("actuator_bundle_key")
                        if activation_patch_production_candidate is not None
                        else activation_patch_executable_shadow.get("actuator_bundle_key")
                        if activation_patch_executable_shadow is not None
                        else activation_patch_preview.get("actuator_bundle_key")
                        if activation_patch_preview is not None
                        else
                        proposal_attention_shadow.get("actuator_bundle_key")
                        if proposal_attention_shadow is not None
                        else step_actuator_bundle_key
                    ),
                    "recipe_name": activation_patch_production_candidate.get("recipe_name")
                    if activation_patch_production_candidate is not None
                    else activation_patch_executable_shadow.get("recipe_name")
                    if activation_patch_executable_shadow is not None
                    else activation_patch_preview.get("recipe_name")
                    if activation_patch_preview is not None
                    else None,
                    "operator_recipe_id": activation_patch_production_candidate.get("operator_recipe_id")
                    if activation_patch_production_candidate is not None
                    else activation_patch_executable_shadow.get("operator_recipe_id")
                    if activation_patch_executable_shadow is not None
                    else activation_patch_preview.get("operator_recipe_id")
                    if activation_patch_preview is not None
                    else None,
                    "compile_state": activation_patch_production_candidate.get("compile_state")
                    if activation_patch_production_candidate is not None
                    else activation_patch_executable_shadow.get("compile_state")
                    if activation_patch_executable_shadow is not None
                    else activation_patch_preview.get("compile_state")
                    if activation_patch_preview is not None
                    else None,
                    "site": (
                        activation_patch_production_candidate.get("site")
                        if activation_patch_production_candidate is not None
                        else activation_patch_executable_shadow.get("site")
                        if activation_patch_executable_shadow is not None
                        else activation_patch_preview.get("site")
                        if activation_patch_preview is not None
                        else proposal_attention_shadow.get("site")
                        if proposal_attention_shadow is not None
                        else None
                    ),
                    "layer": (
                        activation_patch_production_candidate.get("layer")
                        if activation_patch_production_candidate is not None
                        else activation_patch_executable_shadow.get("layer")
                        if activation_patch_executable_shadow is not None
                        else activation_patch_preview.get("layer")
                        if activation_patch_preview is not None
                        else proposal_attention_shadow.get("layer")
                        if proposal_attention_shadow is not None
                        else None
                    ),
                    "alpha": activation_patch_production_candidate.get("alpha")
                    if activation_patch_production_candidate is not None
                    else activation_patch_executable_shadow.get("alpha")
                    if activation_patch_executable_shadow is not None
                    else activation_patch_preview.get("alpha")
                    if activation_patch_preview is not None
                    else None,
                    "head": proposal_attention_shadow.get("head") if proposal_attention_shadow is not None else None,
                    "scale": proposal_attention_shadow.get("scale") if proposal_attention_shadow is not None else None,
                    "response_profile": proposal_attention_shadow.get("response_profile")
                    if proposal_attention_shadow is not None
                    else None,
                    "attention_shadow_actuator_class": proposal_attention_shadow.get("attention_shadow_actuator_class")
                    if proposal_attention_shadow is not None
                    else None,
                    "promotable_to_certification_replay": proposal_attention_shadow.get(
                        "promotable_to_certification_replay"
                    )
                    if proposal_attention_shadow is not None
                    else False,
                    "promotion_reason": proposal_attention_shadow.get("promotion_reason")
                    if proposal_attention_shadow is not None
                    else None,
                    "counterfactual_delta": proposal_attention_shadow.get("counterfactual_delta")
                    if proposal_attention_shadow is not None
                    else None,
                    "production_apply_allowed": False,
                    "reason": (
                        "production trial followup evidence is being consumed before restarting activation_patch probes"
                        if production_trial_outcome_holds_activation_patch_pipeline
                        else
                        "activation patch production shadow replay remains non-applying evidence"
                        if activation_patch_production_shadow is not None and activation_patch_production_trial_gate is None
                        else "activation patch production trial gate is bounded-trial evidence, not production apply"
                        if activation_patch_production_trial_gate is not None
                        else
                        "activation patch production candidate remains blocked until policy review opens production apply"
                        if activation_patch_production_candidate is not None
                        else
                        "activation patch executable shadow remains diagnostic-only until promotion gate"
                        if activation_patch_executable_shadow is not None
                        else
                        "activation patch compile preview remains shadow-only until production operator support exists"
                        if activation_patch_preview is not None
                        else "attention carrier remains shadow-only until certified"
                        if proposal_attention_shadow is not None
                        else "diagnostic_request_mode_keeps_frontier_shadow_only"
                    ),
                    "decision": "shadow",
                }
            ]
            if objective_bundle_key
            else [],
            "controller_memory": {
                "objective_bundle_key": objective_bundle_key,
                "step_actuator_bundle_key": step_actuator_bundle_key,
                "next_evidence_needed": next_evidence,
                "next_action": next_action,
            },
        }
        if production_trial_outcome is not None:
            meta["production_trial_outcome"] = dict(production_trial_outcome)
            meta["activation_patch_production_trial_effect_class"] = str(
                production_trial_outcome.get("trial_effect_class", "unknown") or "unknown"
            )
            if production_trial_followup is not None:
                meta["production_trial_followup"] = dict(production_trial_followup)
                meta["blocked_by"] = production_trial_followup.get(
                    "blocked_by",
                    meta.get("blocked_by", "activation_patch_production_trial_outcome"),
                )
                meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
            if production_trial_alternate_evidence is not None:
                meta["production_trial_alternate_evidence"] = dict(production_trial_alternate_evidence)
                meta["production_trial_followup_consumed"] = True
                meta["blocked_by"] = "production_trial_alternate_evidence_pending_candidate_generator"
                meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
                meta["controller_memory"]["production_trial_followup_consumed"] = True
            if production_trial_alternate_candidate is not None:
                meta["production_trial_alternate_candidate"] = dict(production_trial_alternate_candidate)
                meta["controller_memory"]["production_trial_alternate_candidate_recipe_id"] = (
                    production_trial_alternate_candidate.get("operator_recipe_id")
                )
                if request_production_trial_followup_round and diagnostic_name == "activation_patch_candidate_review":
                    meta["blocked_by"] = "production_trial_alternate_candidate_shadow_review"
                    meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
            meta["activation_patch_pipeline_suppressed_by_production_trial_outcome"] = bool(
                production_trial_outcome_holds_activation_patch_pipeline
            )
            meta["hypothesis"] = "activation_patch_production_trial_outcome_reviewed"
            meta["micro_rationale"] = (
                "production trial outcome was folded back into the next diagnostic decision; production apply remains closed"
            )
        if attention_shadow is not None and activation_patch_preview is None:
            meta["attention_shadow_actuator"] = dict(attention_shadow)
        if activation_patch_preview is not None and production_trial_outcome_holds_activation_patch_pipeline:
            meta["activation_patch_compile_preview"] = dict(activation_patch_preview)
            meta["activation_patch_pipeline_suppressed_by_production_trial_outcome"] = True
        elif activation_patch_preview is not None:
            meta["activation_patch_compile_preview"] = dict(activation_patch_preview)
            if activation_patch_executable_shadow is not None:
                meta["activation_patch_executable_shadow"] = dict(activation_patch_executable_shadow)
                meta["activation_patch_runtime_support_status"] = activation_patch_runtime_support.get(
                    "runtime_support_status",
                    activation_patch_runtime_support.get("status"),
                ) if isinstance(activation_patch_runtime_support, Mapping) else None
                if activation_patch_production_shadow is not None:
                    shadow_dossier = activation_patch_production_shadow.get("production_shadow_dossier")
                    if isinstance(shadow_dossier, Mapping):
                        meta["activation_patch_production_shadow_dossier"] = dict(shadow_dossier)
                    meta["activation_patch_production_shadow_replay_passed"] = bool(
                        activation_patch_production_shadow.get("shadow_replay_passed", False)
                    )
                    meta["production_denial_reasons_after_shadow"] = list(
                        activation_patch_production_shadow.get("production_denial_reasons_after_shadow", ())
                    ) if isinstance(
                        activation_patch_production_shadow.get("production_denial_reasons_after_shadow"),
                        SequenceABC,
                    ) and not isinstance(
                        activation_patch_production_shadow.get("production_denial_reasons_after_shadow"),
                        (str, bytes, bytearray),
                    ) else []
                    if activation_patch_production_trial_gate is not None:
                        trial_dossier = activation_patch_production_trial_gate.get("production_trial_dossier")
                        trial_candidate = activation_patch_production_trial_gate.get("production_trial_candidate")
                        trial_contract = activation_patch_production_trial_gate.get("production_trial_contract")
                        if isinstance(trial_dossier, Mapping):
                            meta["activation_patch_production_trial_dossier"] = dict(trial_dossier)
                        if isinstance(trial_candidate, Mapping):
                            meta["activation_patch_production_trial_candidate"] = dict(trial_candidate)
                        if isinstance(trial_contract, Mapping):
                            meta["activation_patch_production_trial_contract"] = dict(trial_contract)
                        meta["production_trial_allowed"] = bool(
                            activation_patch_production_trial_gate.get("production_trial_allowed", False)
                        )
                        meta["production_trial_blocked_reasons"] = list(
                            activation_patch_production_trial_gate.get("production_trial_blocked_reasons", ())
                        ) if isinstance(
                            activation_patch_production_trial_gate.get("production_trial_blocked_reasons"),
                            SequenceABC,
                        ) and not isinstance(
                            activation_patch_production_trial_gate.get("production_trial_blocked_reasons"),
                            (str, bytes, bytearray),
                        ) else []
                        meta["hypothesis"] = "activation_patch_production_trial_gate_reviewed"
                        meta["micro_rationale"] = (
                            "production trial gate reviewed bounded trial contract; production apply remains closed"
                        )
                        meta["blocked_by"] = "activation_patch_production_trial_only"
                        meta["controller_memory"]["blocked_by"] = "activation_patch_production_trial_only"
                    else:
                        meta["hypothesis"] = "activation_patch_production_shadow_replay_reviewed"
                        meta["micro_rationale"] = "production shadow replay updated denial axes; production apply remains closed"
                        meta["blocked_by"] = "activation_patch_production_shadow_policy_closed"
                        meta["controller_memory"]["blocked_by"] = "activation_patch_production_shadow_policy_closed"
                elif activation_patch_production_candidate is not None:
                    meta["activation_patch_production_apply_candidate"] = dict(activation_patch_production_candidate)
                    meta["activation_patch_promotion_gate_passed"] = bool(
                        activation_patch_promotion_gate.get("gate_passed", False)
                    ) if isinstance(activation_patch_promotion_gate, Mapping) else False
                    meta["hypothesis"] = "activation_patch_candidate_ready_for_policy_review"
                    meta["micro_rationale"] = "activation patch passed promotion gate, but production apply remains closed"
                    meta["blocked_by"] = "activation_patch_production_policy_review_required"
                    meta["controller_memory"]["blocked_by"] = "activation_patch_production_policy_review_required"
                else:
                    meta["hypothesis"] = "activation_patch_executable_shadow_diagnostic_only"
                    meta["micro_rationale"] = "activation patch reached quarantine executable support; production apply remains closed"
                    meta["blocked_by"] = "activation_patch_promotion_gate_required"
                    meta["controller_memory"]["blocked_by"] = "activation_patch_promotion_gate_required"
            else:
                meta["hypothesis"] = "activation_patch_compile_preview_shadow_only"
                meta["micro_rationale"] = "activation patch preview exists, but production apply remains closed"
                meta["blocked_by"] = "activation_patch_compile_preview_shadow_only"
                meta["controller_memory"]["blocked_by"] = "activation_patch_compile_preview_shadow_only"
        elif activation_patch_preview_blocked_reason is not None:
            meta["blocked_by"] = activation_patch_preview_blocked_reason
            meta["controller_memory"]["blocked_by"] = activation_patch_preview_blocked_reason
        request_round_open = bool(
            request_activation_patch_runtime_round
            or request_activation_patch_promotion_round
            or request_activation_patch_production_shadow_round
            or request_activation_patch_production_trial_round
            or request_production_trial_followup_round
            or request_attention_shadow_round
        )
        if has_latest_results and not request_round_open:
            meta["observed_outcome"] = "diagnostic_result_received"
            meta["latest_diagnostic_result_count"] = len(latest_results)
        else:
            meta["diagnostic_request"] = diagnostic_request
            meta["controller_memory"]["diagnostic_request"] = diagnostic_name
            if request_attention_shadow_round:
                meta["observed_outcome"] = "operator_diagnostic_blocked_attention_shadow_round_requested"
                meta["latest_diagnostic_result_count"] = len(latest_results)
        if production_trial_outcome is not None:
            meta["production_trial_outcome"] = dict(production_trial_outcome)
            meta["activation_patch_production_trial_effect_class"] = str(
                production_trial_outcome.get("trial_effect_class", "unknown") or "unknown"
            )
            if production_trial_followup is not None:
                meta["production_trial_followup"] = dict(production_trial_followup)
                meta["blocked_by"] = production_trial_followup.get(
                    "blocked_by",
                    meta.get("blocked_by", "activation_patch_production_trial_outcome"),
                )
                meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
            if production_trial_alternate_evidence is not None:
                meta["production_trial_alternate_evidence"] = dict(production_trial_alternate_evidence)
                meta["production_trial_followup_consumed"] = True
                meta["blocked_by"] = "production_trial_alternate_evidence_pending_candidate_generator"
                meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
                meta["controller_memory"]["production_trial_followup_consumed"] = True
            if production_trial_alternate_candidate is not None:
                meta["production_trial_alternate_candidate"] = dict(production_trial_alternate_candidate)
                meta["controller_memory"]["production_trial_alternate_candidate_recipe_id"] = (
                    production_trial_alternate_candidate.get("operator_recipe_id")
                )
                if request_production_trial_followup_round and diagnostic_name == "activation_patch_candidate_review":
                    meta["blocked_by"] = "production_trial_alternate_candidate_shadow_review"
                    meta["controller_memory"]["blocked_by"] = meta["blocked_by"]
            meta["activation_patch_pipeline_suppressed_by_production_trial_outcome"] = bool(
                production_trial_outcome_holds_activation_patch_pipeline
            )
            meta["hypothesis"] = "activation_patch_production_trial_outcome_reviewed"
            meta["micro_rationale"] = (
                "production trial outcome was folded back into the next diagnostic decision; production apply remains closed"
            )
        return {"version": "0.1", "decision": "noop", "meta": meta}

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
        if self.replay_mode == "diagnostic_request":
            command = self._diagnostic_request_command(
                packet=packet,
                strategy_hints=strategy_hints,
                frontier_bundle_key=frontier_bundle_key,
                suggested_bundle_key=suggested_bundle_key,
            )
        elif self.calls == 1:
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
                        "apply_kind": "diagnostic_probe",
                        "production_apply_allowed": False,
                        "production_policy_would_apply": False,
                        "certified_for_apply": False,
                        "hypothesis": "forced_readout_escape_frontier_replay",
                        "micro_rationale": "inspect readout_escape logging contract on a forced frontier candidate",
                        "objective_bundle_key": str(target_bundle_key or ""),
                        "step_actuator_bundle_key": str(target_bundle_key or ""),
                        "focus_term": matching[0].get("focus_feature"),
                        "surface_family_key": matching[0].get("candidate_family"),
                        "operator_recipe_id": matching[0].get("operator_recipe_id"),
                        "shadow_proposals": [
                            {
                                "kind": "frontier_runner_up",
                                "bundle_key": str(candidate.get("bundle_key", "") or ""),
                                "reason": "retain runner-up frontier candidate as shadow-only context",
                                "decision": "shadow",
                            }
                            for candidate in candidates
                            if str(candidate.get("bundle_key", "") or "") not in ("", str(target_bundle_key or ""))
                        ][:1],
                        "why_not_apply": "other plausible bundles remain shadow_only until certified_for_apply",
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
                "diagnostic_call_count": (
                    packet.get("diagnostic_call_count")
                    if packet.get("diagnostic_call_count") is not None
                    else (
                        packet.get("telemetry", {}).get("diagnostic_call_count")
                        if isinstance(packet.get("telemetry"), Mapping)
                        else None
                    )
                ),
                "diagnostic_call_budget_left": (
                    packet.get("diagnostic_call_budget_left")
                    if packet.get("diagnostic_call_budget_left") is not None
                    else (
                        packet.get("telemetry", {}).get("diagnostic_call_budget_left")
                        if isinstance(packet.get("telemetry"), Mapping)
                        else None
                    )
                ),
                "latest_diagnostic_result_count": len(packet.get("latest_diagnostic_results", ()))
                if isinstance(packet.get("latest_diagnostic_results"), SequenceABC)
                and not isinstance(packet.get("latest_diagnostic_results"), (str, bytes, bytearray))
                else 0,
                "latest_diagnostic_results": [
                    self._compact_diagnostic_result(item)
                    for item in packet.get("latest_diagnostic_results", ())
                    if isinstance(item, Mapping)
                ][:2]
                if isinstance(packet.get("latest_diagnostic_results"), SequenceABC)
                and not isinstance(packet.get("latest_diagnostic_results"), (str, bytes, bytearray))
                else [],
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "replay",
                    "model": "frontier-replay-controller",
                    "latency_ms": 0.0,
                    "parse_ok": True,
                    "response_text": json.dumps(command, ensure_ascii=False, sort_keys=True),
                    "raw_json_object": dict(command),
                    "normalized_payload": dict(command),
                    "normalization_delta": {
                        "raw_top_level_keys": sorted(str(key) for key in command.keys()),
                        "normalized_top_level_keys": sorted(str(key) for key in command.keys()),
                        "moved_into_meta": [],
                        "normalization_kind": "replay_passthrough",
                    },
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
    readout_analyzer_name: str | None
    readout_analyzer_feature_backend: str | None
    readout_analyzer_sae_status: str | None
    readout_analyzer_sae_feature_hint_count: int
    gate_report_frontier_bundle_key: str | None
    controller_selected_bundle_key: str | None
    controller_selection_source: str | None
    controller_rejected_signals: tuple[str, ...]
    controller_objective_bundle_key: str | None
    controller_step_actuator_bundle_key: str | None
    controller_plan_mode: str | None
    controller_why_not_apply: str | None
    controller_shadow_proposal_count: int
    bridge_plan_objective_bundle_key: str | None
    bridge_plan_actuator_bundle_key: str | None
    bridge_plan_reason: str | None
    bridge_plan_unavailable_reason: str | None
    bridge_plan_unavailable_objective_bundle_key: str | None
    bridge_plan_unavailable_objective_reasons: dict[str, str]
    bridge_eval_context_drift: bool
    bridge_eval_locked_step: int | None
    bridge_plan_used: bool
    bridge_plan_packet_invariant_ok: bool | None
    bridge_plan_packet_invariant_diff: tuple[str, ...]
    bridge_plan_recommendation_count: int
    bridge_plan_recommendation_objectives: tuple[str, ...]
    bridge_eval_shadow_bundle_keys: tuple[str, ...]
    bridge_eval_recipe_names: tuple[str, ...]
    bridge_eval_matrix: tuple[dict[str, Any], ...]
    bridge_eval_operator_recipe_mode_diagnostics: tuple[dict[str, Any], ...]
    bridge_eval_attention_scale_response_diagnostics: tuple[dict[str, Any], ...]
    bridge_eval_objective_class_counts: dict[str, dict[str, int]]
    bridge_eval_evaluations_count: int
    bridge_eval_ownership_count: int
    bridge_eval_exception: str | None
    diagnostic_evidence_ledger: tuple[dict[str, Any], ...]
    diagnostic_evidence_ledger_count: int
    bundle_diagnostic_status: dict[str, dict[str, Any]]
    diagnostic_frontier_bundle_key: str | None
    diagnostic_frontier_next_evidence: str | None
    diagnostic_frontier_request: str | None
    diagnostic_frontier_reason_text: str | None
    diagnostic_request_event_count: int
    diagnostic_result_event_count: int
    controller_diagnostic_request_names: tuple[str, ...]
    latest_diagnostic_results: tuple[dict[str, Any], ...]
    recent_diagnostic_results: tuple[dict[str, Any], ...]
    bridge_eval_extra_operator_diagnostics: tuple[dict[str, Any], ...]
    bridge_eval_poststep_comparison: dict[str, Any] | None
    bridge_eval_candidate_swap_comparison: dict[str, Any] | None
    controller_step_views: tuple[dict[str, Any], ...]
    bridge_visible_step_views: tuple[dict[str, Any], ...]
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
            "readout_analyzer_name": self.readout_analyzer_name,
            "readout_analyzer_feature_backend": self.readout_analyzer_feature_backend,
            "readout_analyzer_sae_status": self.readout_analyzer_sae_status,
            "readout_analyzer_sae_feature_hint_count": int(self.readout_analyzer_sae_feature_hint_count),
            "gate_report_frontier_bundle_key": self.gate_report_frontier_bundle_key,
            "controller_selected_bundle_key": self.controller_selected_bundle_key,
            "controller_selection_source": self.controller_selection_source,
            "controller_rejected_signals": list(self.controller_rejected_signals),
            "controller_objective_bundle_key": self.controller_objective_bundle_key,
            "controller_step_actuator_bundle_key": self.controller_step_actuator_bundle_key,
            "controller_plan_mode": self.controller_plan_mode,
            "controller_why_not_apply": self.controller_why_not_apply,
            "controller_shadow_proposal_count": int(self.controller_shadow_proposal_count),
            "bridge_plan_objective_bundle_key": self.bridge_plan_objective_bundle_key,
            "bridge_plan_actuator_bundle_key": self.bridge_plan_actuator_bundle_key,
            "bridge_plan_reason": self.bridge_plan_reason,
            "bridge_plan_unavailable_reason": self.bridge_plan_unavailable_reason,
            "bridge_plan_unavailable_objective_bundle_key": self.bridge_plan_unavailable_objective_bundle_key,
            "bridge_plan_unavailable_objective_reasons": {
                str(key): str(value) for key, value in sorted(self.bridge_plan_unavailable_objective_reasons.items())
            },
            "bridge_eval_context_drift": bool(self.bridge_eval_context_drift),
            "bridge_eval_locked_step": self.bridge_eval_locked_step,
            "bridge_plan_used": bool(self.bridge_plan_used),
            "bridge_plan_packet_invariant_ok": self.bridge_plan_packet_invariant_ok,
            "bridge_plan_packet_invariant_diff": list(self.bridge_plan_packet_invariant_diff),
            "bridge_plan_recommendation_count": int(self.bridge_plan_recommendation_count),
            "bridge_plan_recommendation_objectives": list(self.bridge_plan_recommendation_objectives),
            "bridge_eval_shadow_bundle_keys": list(self.bridge_eval_shadow_bundle_keys),
            "bridge_eval_recipe_names": list(self.bridge_eval_recipe_names),
            "bridge_eval_matrix": [dict(item) for item in self.bridge_eval_matrix],
            "bridge_eval_operator_recipe_mode_diagnostics": [
                dict(item) for item in self.bridge_eval_operator_recipe_mode_diagnostics
            ],
            "bridge_eval_attention_scale_response_diagnostics": [
                dict(item) for item in self.bridge_eval_attention_scale_response_diagnostics
            ],
            "bridge_eval_objective_class_counts": {
                str(key): {str(inner_key): int(inner_value) for inner_key, inner_value in sorted(value.items())}
                for key, value in sorted(self.bridge_eval_objective_class_counts.items())
            },
            "bridge_eval_evaluations_count": int(self.bridge_eval_evaluations_count),
            "bridge_eval_ownership_count": int(self.bridge_eval_ownership_count),
            "bridge_eval_exception": self.bridge_eval_exception,
            "diagnostic_evidence_ledger": [dict(item) for item in self.diagnostic_evidence_ledger],
            "diagnostic_evidence_ledger_count": int(self.diagnostic_evidence_ledger_count),
            "bundle_diagnostic_status": {
                str(key): dict(value) for key, value in sorted(self.bundle_diagnostic_status.items())
            },
            "diagnostic_frontier_bundle_key": self.diagnostic_frontier_bundle_key,
            "diagnostic_frontier_next_evidence": self.diagnostic_frontier_next_evidence,
            "diagnostic_frontier_request": self.diagnostic_frontier_request,
            "diagnostic_frontier_reason_text": self.diagnostic_frontier_reason_text,
            "diagnostic_request_event_count": int(self.diagnostic_request_event_count),
            "diagnostic_result_event_count": int(self.diagnostic_result_event_count),
            "controller_diagnostic_request_names": list(self.controller_diagnostic_request_names),
            "latest_diagnostic_results": [dict(item) for item in self.latest_diagnostic_results],
            "recent_diagnostic_results": [dict(item) for item in self.recent_diagnostic_results],
            "bridge_eval_extra_operator_diagnostics": [
                dict(item) for item in self.bridge_eval_extra_operator_diagnostics
            ],
            "bridge_eval_poststep_comparison": (
                None if self.bridge_eval_poststep_comparison is None else dict(self.bridge_eval_poststep_comparison)
            ),
            "bridge_eval_candidate_swap_comparison": (
                None if self.bridge_eval_candidate_swap_comparison is None else dict(self.bridge_eval_candidate_swap_comparison)
            ),
            "controller_step_views": [dict(item) for item in self.controller_step_views],
            "bridge_visible_step_views": [dict(item) for item in self.bridge_visible_step_views],
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


def _focused_bridge_eval_recipe_specs() -> list[dict[str, Any]]:
    return [
        {
            "recipe_name": "baseline_span_mean",
            "localization": "exact_prompt_span_mean",
            "pooling": "mean",
            "contrast_mode": "none",
            "modes": ("kv_pair", "kv_v", "kv_k", "kv_pair_asymmetric"),
        },
        {
            "recipe_name": "term_token",
            "localization": "exact_term_token",
            "pooling": "single",
            "contrast_mode": "none",
            "modes": ("kv_pair", "kv_v", "kv_k"),
        },
        {
            "recipe_name": "term_fused",
            "localization": "exact_term_fused",
            "pooling": "fused",
            "contrast_mode": "none",
            "modes": ("kv_pair", "kv_v", "kv_k"),
        },
        {
            "recipe_name": "term_centered_pm1",
            "localization": "exact_term_centered_pm1",
            "pooling": "centered_mean",
            "contrast_mode": "none",
            "modes": ("kv_pair", "kv_v", "kv_k", "kv_pair_asymmetric"),
        },
        {
            "recipe_name": "term_centered_pm1_v060_k020",
            "localization": "exact_term_centered_pm1",
            "pooling": "centered_mean",
            "contrast_mode": "none",
            "v_alpha": 0.06,
            "k_alpha": 0.02,
            "modes": ("kv_pair_asymmetric",),
        },
        {
            "recipe_name": "term_centered_pm1_v025_k045",
            "localization": "exact_term_centered_pm1",
            "pooling": "centered_mean",
            "contrast_mode": "none",
            "v_alpha": 0.025,
            "k_alpha": 0.045,
            "modes": ("kv_pair_asymmetric",),
        },
        {
            "recipe_name": "term_centered_pm1_minus_stealer_l025",
            "localization": "exact_term_centered_pm1",
            "pooling": "centered_mean",
            "contrast_mode": "minus_stealer",
            "contrast_scale": 0.25,
            "competitor_strategy": "stealer",
            "modes": ("kv_pair",),
        },
        {
            "recipe_name": "term_centered_pm1_orthogonal_stealer",
            "localization": "exact_term_centered_pm1",
            "pooling": "centered_mean",
            "contrast_mode": "orthogonal_stealer",
            "competitor_strategy": "stealer",
            "modes": ("kv_pair",),
        },
    ]


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
    controller_replay_mode: str = "frontier_apply",
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
    max_production_trial_edits_per_run: int = 1,
    max_production_trial_alpha: float = 0.15,
    max_production_trial_edit_cost: float | None = None,
    max_production_trial_followup_edits_per_run: int = 1,
    max_production_trial_followup_alpha: float = 0.05,
    max_production_trial_followup_edit_cost: float | None = None,
    max_diagnostic_calls_per_run: int = 12,
    diagnostic_result_window: int = 12,
    max_controller_diagnostic_rethink_rounds: int = 4,
) -> ReadoutEscapeReplayHarnessResult:
    normalized_packet_mode = str(packet_mode).strip().lower().replace("-", "_")
    if normalized_packet_mode not in {"forced_frontier", "directscan", "fixed_candidate"}:
        raise ValueError("run_readout_escape_replay_harness packet_mode must be one of forced_frontier, directscan, fixed_candidate")
    normalized_controller_replay_mode = str(controller_replay_mode).strip().lower().replace("-", "_")
    if normalized_controller_replay_mode not in {"frontier_apply", "diagnostic_request"}:
        raise ValueError(
            "run_readout_escape_replay_harness controller_replay_mode must be one of frontier_apply, diagnostic_request"
        )
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
        max_production_trial_edits_per_run=max_production_trial_edits_per_run,
        max_production_trial_alpha=max_production_trial_alpha,
        max_production_trial_edit_cost=max_production_trial_edit_cost,
        max_production_trial_followup_edits_per_run=max_production_trial_followup_edits_per_run,
        max_production_trial_followup_alpha=max_production_trial_followup_alpha,
        max_production_trial_followup_edit_cost=max_production_trial_followup_edit_cost,
        max_diagnostic_calls_per_run=max_diagnostic_calls_per_run,
        diagnostic_result_window=diagnostic_result_window,
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
        "bridge_plan_report": None,
        "bridge_plan_packet_invariant_ok": None,
        "bridge_plan_packet_invariant_diff": (),
        "bridge_eval_summary": None,
        "bridge_eval_shadow_candidates": (),
        "bridge_eval_shadow_bundle_keys": (),
        "bridge_eval_active": False,
        "bridge_eval_packet_snapshot": None,
        "bridge_eval_locked_step": None,
        "bridge_eval_poststep_comparison": None,
        "bridge_eval_candidate_swap_comparison": None,
        "bridge_eval_pre_replay_summary": None,
        "bridge_eval_last_replay_summary": None,
        "bridge_eval_extra_operator_diagnostics": (),
        "pre_step_segments": (),
        "pre_step_steps": 0,
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
        forced_state["bridge_plan_report"] = None
        forced_state["bridge_plan_packet_invariant_ok"] = None
        forced_state["bridge_plan_packet_invariant_diff"] = ()
        forced_state["bridge_eval_summary"] = None
        forced_state["bridge_eval_shadow_candidates"] = ()
        forced_state["bridge_eval_shadow_bundle_keys"] = ()
        forced_state["bridge_eval_active"] = False
        forced_state["bridge_eval_packet_snapshot"] = None
        forced_state["bridge_eval_locked_step"] = None
        forced_state["bridge_eval_poststep_comparison"] = None
        forced_state["bridge_eval_candidate_swap_comparison"] = None
        forced_state["bridge_eval_pre_replay_summary"] = None
        forced_state["bridge_eval_last_replay_summary"] = None
        forced_state["bridge_eval_extra_operator_diagnostics"] = ()
        forced_state["pre_step_segments"] = tuple((segment.kind, tuple(segment.token_ids)) for segment in self._segments)
        forced_state["pre_step_steps"] = int(self._steps)
        self._operator_bridge_plan_table = {}
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
        if normalized_packet_mode in {"directscan", "fixed_candidate"}:
            try:
                self.build_controller_packet()
            except Exception:
                forced_state["bridge_plan_report"] = None
                forced_state["bridge_plan_packet_invariant_ok"] = None
                forced_state["bridge_plan_packet_invariant_diff"] = ()
                forced_state["bridge_eval_summary"] = None
                forced_state["bridge_eval_shadow_candidates"] = ()
                forced_state["bridge_eval_shadow_bundle_keys"] = ()
                forced_state["bridge_eval_active"] = False
                forced_state["bridge_eval_packet_snapshot"] = None
                forced_state["bridge_eval_locked_step"] = None

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

    def _bridge_frontier_snapshot(strategy_hints: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "base_winner_bundle_key": str(strategy_hints.get("base_winner_bundle_key", "") or ""),
            "challenger_bundle_key": str(strategy_hints.get("challenger_bundle_key", "") or ""),
            "selected_bundle_key": str(strategy_hints.get("selected_bundle_key", "") or ""),
            "gate_report_frontier_bundle_key": str(strategy_hints.get("gate_report_frontier_bundle_key", "") or ""),
            "selection_source": str(strategy_hints.get("selection_source", "") or ""),
        }

    def _candidate_bundle_keys(raw_items: Any) -> list[str]:
        if not isinstance(raw_items, SequenceABC) or isinstance(raw_items, (str, bytes, bytearray)):
            return []
        seen: list[str] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("bundle_key", "") or "")
            if bundle_key and bundle_key not in seen:
                seen.append(bundle_key)
        return seen

    def _bundle_term(bundle_key: str) -> str:
        parts = str(bundle_key or "").split(":")
        if len(parts) >= 2 and parts[0] == "kv_pair":
            return str(parts[1])
        return ""

    def _build_bridge_eval_shadow_candidates(
        self: HookedTransformerWorkerRuntime,
        strategy_hints: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        objective_term = ""
        for raw_key in (
            strategy_hints.get("selected_bundle_key"),
            strategy_hints.get("gate_report_frontier_bundle_key"),
            strategy_hints.get("challenger_bundle_key"),
            strategy_hints.get("base_winner_bundle_key"),
        ):
            objective_term = _bundle_term(str(raw_key or ""))
            if objective_term:
                break
        if not objective_term:
            objective_term = str(strategy_hints.get("controller_focus_term", "") or "")
        if not objective_term:
            return [], []

        promoted_cache_surfaces = self._promoted_cache_surfaces()
        surface_lookup = self._cache_surface_lookup(promoted_cache_surfaces=promoted_cache_surfaces)
        shadow_candidates: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, int, int, str]] = set()
        for hit in self._actionable_kv_hits(limit=6, promoted_cache_surfaces=promoted_cache_surfaces):
            feature = str(hit.get("feature", "") or "")
            if not feature or feature == objective_term:
                continue
            site = str(hit.get("site", "") or "")
            token_mode = str(hit.get("token_mode", "last") or "last")
            layer = hit.get("layer")
            head = hit.get("head")
            if site not in {"k_cache", "v_cache"} or token_mode != "last":
                continue
            if isinstance(layer, bool) or not isinstance(layer, int):
                continue
            if isinstance(head, bool) or not isinstance(head, int):
                continue
            record = surface_lookup.get((site, int(layer), int(head), token_mode))
            if record is None:
                continue
            source_variants = self._kv_source_variants_for_hit(hit, control_phase_hint="readout_escape")
            if not source_variants:
                continue
            preferred_variant = None
            for variant in source_variants:
                if str(variant.get("provenance_class", "") or "") == "source_body" and str(variant.get("span_kind", "")).startswith("exact_prompt_span"):
                    preferred_variant = variant
                    break
            if preferred_variant is None:
                preferred_variant = source_variants[0]
            source_span = preferred_variant.get("source_span") if isinstance(preferred_variant.get("source_span"), Mapping) else None
            source_position = int(preferred_variant.get("source_position", 0) or 0)
            source_end = source_position + 1 if source_span is None else int(source_span.get("end", source_position + 1))
            surface_id = str(record.get("surface_id", "") or "")
            if not surface_id:
                continue
            dedupe_key = (
                surface_id,
                feature,
                source_position,
                source_end,
                str(preferred_variant.get("span_kind", "")),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            max_alpha = float(record.get("max_alpha", 0.06) or 0.06)
            step_cap = record.get("step_size")
            norm_clip = record.get("norm_clip")
            if norm_clip is None:
                norm_clip = 1.0
            base_alpha = 0.03 if site == "k_cache" else 0.04
            if str(preferred_variant.get("span_kind", "")).startswith("exact_prompt_span"):
                base_alpha = 0.035 if site == "k_cache" else 0.045
            alpha = min(max_alpha, base_alpha)
            step_size = float(alpha if step_cap is None else min(float(step_cap), alpha))
            which = "v" if site == "v_cache" else "k"
            candidate = {
                "surface_id": surface_id,
                "kind": "kv_mix",
                "role": f"kv_shadow_{which}_source_anchor",
                "site": site,
                "layer": int(layer),
                "head": int(head),
                "token_mode": token_mode,
                "focus_feature": feature,
                "alignment": round(float(hit.get("alignment", 0.0) or 0.0), 6),
                "candidate_family": str(preferred_variant.get("candidate_family", "")),
                "phase_objective": "readout_escape",
                "span_kind": str(preferred_variant.get("span_kind", "source_position_single")),
                "recipe_localization": str(preferred_variant.get("span_kind", "source_position_single")),
                "recipe_pooling": "mean"
                if str(preferred_variant.get("span_kind", "")).startswith("exact_prompt_span")
                else "single",
                "contrast_mode": "none",
                "provenance_class": str(preferred_variant.get("provenance_class", "misc_prompt") or "misc_prompt"),
                "read_source_resolved": True,
                "write_target_resolved": True,
                "source_position": int(source_position),
                "source_piece": preferred_variant.get("source_piece"),
                "source_segment_kind": preferred_variant.get("source_segment_kind"),
                "source": {
                    "dtype": "cache_pair",
                    which: {
                        "ref": {
                            "scope": "runtime",
                            "worker": self.worker_id,
                            "tensor": site,
                            "layer": int(layer),
                            "head": int(head),
                            "token": dict(preferred_variant["token_selector"]),
                        }
                    },
                },
                "op": {"kind": "kv_mix", "alpha": float(alpha), "which": which},
                "budget": {
                    "ttl_steps": 1,
                    "norm_clip": float(norm_clip),
                    "step_size": float(step_size),
                    "revertible": True,
                },
                "meta": {
                    "hypothesis": f"bridge_eval_shadow_{which}",
                    "expected_effect": "shadow_competitor_bridge_eval",
                },
            }
            if source_span is not None:
                candidate["source_span"] = {"start": int(source_span["start"]), "end": int(source_span["end"])}
            candidate["operator_family_key"] = self._operator_family_key(candidate)
            candidate["operator_recipe_id"] = self._operator_recipe_id(candidate)
            shadow_candidates.append(candidate)

        bundle_inputs = self._bundle_inputs_for_candidates(shadow_candidates)
        if not bundle_inputs:
            return [], []
        selected_bundle = bundle_inputs[0]
        bundle_key = str(selected_bundle.get("bundle_key", "") or "")
        if not bundle_key:
            return [], []
        focus_term = str(selected_bundle.get("term", "") or "")
        provenance_class = str(selected_bundle.get("provenance_class", "misc_prompt") or "misc_prompt")
        source_span = selected_bundle.get("source_span") if isinstance(selected_bundle.get("source_span"), Mapping) else None
        span_id = (
            f"{int(source_span.get('start', 0) or 0)}:{int(source_span.get('end', 0) or 0)}"
            if source_span is not None
            else "unknown"
        )
        selected: list[dict[str, Any]] = []
        for candidate in shadow_candidates:
            if (
                str(candidate.get("focus_feature", "") or "") == focus_term
                and str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt") == provenance_class
                and self._candidate_span_id(candidate) == span_id
            ):
                candidate["bundle_key"] = bundle_key
                candidate["bundle_ready"] = True
                candidate["bundle_family"] = "kv_pair_source_anchor"
                selected.append(candidate)
        return selected, [bundle_key] if selected else []

    def _bridge_eval_extra_operator_diagnostics(
        self: HookedTransformerWorkerRuntime,
        *,
        replay_bundle_keys: Sequence[str],
        strategy_hints: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        bundle_terms: dict[str, str] = {}
        for raw_key in replay_bundle_keys:
            key = str(raw_key or "")
            term = _bundle_term(key)
            if key and term and key not in bundle_terms:
                bundle_terms[key] = term
        if not bundle_terms:
            return []
        ownership_terms = list(dict.fromkeys(str(term) for term in bundle_terms.values() if str(term)))

        activation_surfaces: list[Any] = []
        for surface in self.surface_catalog:
            target = getattr(surface, "target", None)
            if getattr(target, "kind", None) != "activation" or getattr(target, "site", None) != "resid_pre":
                continue
            token = getattr(target, "token", None)
            token_mode = str(getattr(token, "mode", "") or "")
            token_value = getattr(token, "value", None)
            if token_mode == "last" or (token_mode == "index" and token_value == -2):
                activation_surfaces.append(surface)
        activation_surfaces.sort(
            key=lambda surface: (
                int(getattr(getattr(surface, "target", None), "layer", 0) or 0),
                0 if str(getattr(getattr(getattr(surface, "target", None), "token", None), "mode", "") or "") == "last" else -1,
                str(getattr(surface, "surface_id", "") or ""),
            ),
            reverse=True,
        )
        readout_surface = activation_surfaces[0] if activation_surfaces else None
        if readout_surface is None:
            return []

        def _best_source_span(term: str) -> dict[str, Any] | None:
            spans = [dict(item) for item in self._prompt_term_spans(term, max_spans=4)]
            if not spans:
                return None
            spans.sort(
                key=lambda item: (
                    0 if str(item.get("provenance_class", "") or "") == "source_body" else 1,
                    int(item.get("start", 0) or 0),
                    -int(item.get("length", 0) or 0),
                )
            )
            return spans[0]

        def _surface_caps(surface: Any) -> tuple[float, float, float]:
            caps = getattr(surface, "caps", None)
            norm_clip = 1.0 if getattr(caps, "norm_clip", None) is None else float(caps.norm_clip)
            max_alpha = 0.05 if getattr(caps, "max_alpha", None) is None else float(caps.max_alpha)
            step_cap = getattr(caps, "step_size", None)
            step_size = max_alpha if step_cap is None else min(float(step_cap), max_alpha)
            return norm_clip, max_alpha, step_size

        def _source_selector_for_span(span: Mapping[str, Any] | None) -> dict[str, Any]:
            if not isinstance(span, Mapping):
                return {"mode": "last"}
            start = int(span.get("start", 0) or 0)
            end = int(span.get("end", start + 1) or (start + 1))
            if end - start <= 1:
                return {"mode": "index", "value": start}
            return {"mode": "span", "start": start, "end": end, "pool": "mean"}

        def _make_resid_edit(
            *,
            surface: Any,
            bundle_key: str,
            term: str,
            diagnostic_family: str,
            recipe_name: str,
            source_selector: Mapping[str, Any],
            source_span: Mapping[str, Any] | None,
            alpha: float,
        ) -> dict[str, Any]:
            target = getattr(surface, "target", None)
            layer = int(getattr(target, "layer", 0) or 0)
            surface_id = str(getattr(surface, "surface_id", "") or "")
            norm_clip, max_alpha, step_cap = _surface_caps(surface)
            alpha_value = min(float(alpha), float(max_alpha))
            step_size = min(float(step_cap), alpha_value)
            provenance_class = (
                str(source_span.get("provenance_class", "misc_prompt") or "misc_prompt")
                if isinstance(source_span, Mapping)
                else "answer_prefix"
            )
            span_kind = (
                "exact_prompt_span_mean"
                if isinstance(source_span, Mapping) and int(source_span.get("end", 0) or 0) - int(source_span.get("start", 0) or 0) > 1
                else ("exact_prompt_piece" if isinstance(source_span, Mapping) else "answer_boundary_last")
            )
            source_position = int(source_span.get("start", 0) or 0) if isinstance(source_span, Mapping) else 0
            candidate = {
                "id": f"bridge_diag_{diagnostic_family}_{term}_{surface_id}",
                "surface_id": surface_id,
                "target": {"surface_id": surface_id},
                "kind": "resid_add",
                "role": diagnostic_family,
                "site": "resid_pre",
                "layer": layer,
                "bundle_key": bundle_key,
                "bundle_family": diagnostic_family,
                "focus_feature": term,
                "focus_term": term,
                "candidate_family": f"{diagnostic_family}:{term}:{span_kind}",
                "phase_objective": "readout_escape",
                "span_kind": span_kind,
                "recipe_localization": span_kind,
                "recipe_pooling": "mean" if span_kind.endswith("_mean") else "single",
                "contrast_mode": "none",
                "recipe_alpha": float(alpha_value),
                "provenance_class": provenance_class,
                "read_source_resolved": True,
                "write_target_resolved": True,
                "source_position": source_position,
                "source_piece": source_span.get("text") if isinstance(source_span, Mapping) else None,
                "source_segment_kind": source_span.get("segment_kind") if isinstance(source_span, Mapping) else "answer",
                "source": {
                    "dtype": "vector",
                    "expr": {
                        "fn": "clip_norm",
                        "max_norm": float(norm_clip),
                        "arg": {
                            "fn": "scale",
                            "by": float(step_size),
                            "arg": {
                                "fn": "normalize",
                                "arg": {
                                    "ref": {
                                        "scope": "runtime",
                                        "worker": self.worker_id,
                                        "tensor": "hidden",
                                        "layer": layer,
                                        "token": dict(source_selector),
                                    }
                                },
                            },
                        },
                    },
                },
                "op": {"kind": "resid_add", "alpha": float(alpha_value)},
                "budget": {
                    "ttl_steps": 1,
                    "norm_clip": float(norm_clip),
                    "step_size": float(step_size),
                    "revertible": True,
                },
                "meta": {
                    "hypothesis": "bridge_eval_extra_operator_diagnostic",
                    "expected_effect": diagnostic_family,
                },
            }
            if isinstance(source_span, Mapping):
                candidate["source_span"] = {
                    "start": int(source_span.get("start", 0) or 0),
                    "end": int(source_span.get("end", 0) or 0),
                }
            candidate["operator_recipe_seed_key"] = f"{diagnostic_family}|{span_kind}|{candidate['recipe_pooling']}"
            candidate["operator_recipe_id"] = self._operator_recipe_id(candidate)
            candidate["diagnostic_family"] = diagnostic_family
            candidate["diagnostic_recipe_name"] = recipe_name
            return candidate

        def _compact_actual_delta(
            result: Mapping[str, Any],
            *,
            diagnostic_family: str,
            recipe_name: str,
            objective_bundle_key: str,
            objective_term: str,
        ) -> dict[str, Any]:
            return {
                "diagnostic_only": True,
                "diagnostic_family": diagnostic_family,
                "recipe_name": recipe_name,
                "objective_bundle_key": objective_bundle_key,
                "objective_term": objective_term,
                "actuator_bundle_key": objective_bundle_key,
                "status": str(result.get("status", "") or ""),
                "actuator_class": str(result.get("actual_delta_class", "") or "") or None,
                "operator_recipe_id": result.get("operator_recipe_id"),
                "target_mass_delta": result.get("target_mass_delta"),
                "target_top20_hit_delta": result.get("target_top20_hit_delta"),
                "focus_rank_delta": result.get("focus_rank_delta"),
                "rank_focus_delta": result.get("rank_focus_delta"),
                "required_term_recall_delta": result.get("required_term_recall_delta"),
                "required_term_span_progress_delta": result.get("required_term_span_progress_delta"),
                "entropy_delta": result.get("entropy_delta"),
                "top1_margin_delta": result.get("top1_margin_delta"),
                "repeat_delta": result.get("repeat_flag_delta"),
                "candidate_fingerprint": dict(result.get("candidate_fingerprint", {}))
                if isinstance(result.get("candidate_fingerprint"), Mapping)
                else {},
                "eval_context_fingerprint": dict(result.get("eval_context_fingerprint", {}))
                if isinstance(result.get("eval_context_fingerprint"), Mapping)
                else {},
            }

        def _kv_candidates_by_bundle() -> dict[str, list[dict[str, Any]]]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            raw_candidates: list[Any] = []
            for key in ("kv_candidate_edits", "kv_retry_candidate_edits"):
                items = strategy_hints.get(key)
                if isinstance(items, SequenceABC) and not isinstance(items, (str, bytes, bytearray)):
                    raw_candidates.extend(items)
            raw_candidates.extend(
                item
                for item in forced_state.get("bridge_eval_shadow_candidates", ())
                if isinstance(item, Mapping)
            )
            for item in raw_candidates:
                if not isinstance(item, Mapping):
                    continue
                bundle_key = str(item.get("bundle_key", "") or "")
                if not bundle_key:
                    continue
                grouped.setdefault(bundle_key, []).append(dict(item))
            return grouped

        def _run_attention_head_scale(
            candidate: Mapping[str, Any],
            *,
            objective_bundle_key: str,
            objective_term: str,
            scale: float,
            recipe_name: str,
        ) -> dict[str, Any] | None:
            site = str(candidate.get("site", "") or "")
            if site not in {"k_cache", "v_cache"}:
                return None
            layer = candidate.get("layer")
            head = candidate.get("head")
            if isinstance(layer, bool) or not isinstance(layer, int):
                return None
            if isinstance(head, bool) or not isinstance(head, int):
                return None
            hook_name = f"blocks.{int(layer)}.attn.hook_{'v' if site == 'v_cache' else 'k'}"
            hook_dict = getattr(self.model, "hook_dict", None)
            if hook_dict is None or hook_name not in hook_dict:
                return None
            baseline = self._simulate_decode(max_new_tokens=3, top_k=8)
            if baseline is None:
                return None

            hook_point = hook_dict[hook_name]
            original_fwd_hooks = list(getattr(hook_point, "fwd_hooks", []))
            scale_value = max(0.0, min(1.0, float(scale)))
            hook_stats: dict[str, Any] = {
                "call_count": 0,
                "max_head_norm_before": 0.0,
                "max_head_norm_after": 0.0,
                "first_head_norm_before": None,
                "first_head_norm_after": None,
            }

            def _scale_head(act: torch.Tensor, hook: Any | None = None) -> torch.Tensor:
                if not isinstance(act, torch.Tensor) or act.ndim < 4 or int(head) < 0 or act.shape[-2] <= int(head):
                    return act
                out = act.clone()
                head_slice = act[..., :, int(head), :]
                scaled = head_slice * float(scale_value)
                before_norm = float(torch.linalg.vector_norm(head_slice.detach().float()).item())
                after_norm = float(torch.linalg.vector_norm(scaled.detach().float()).item())
                hook_stats["call_count"] = int(hook_stats.get("call_count", 0) or 0) + 1
                hook_stats["max_head_norm_before"] = max(float(hook_stats["max_head_norm_before"]), before_norm)
                hook_stats["max_head_norm_after"] = max(float(hook_stats["max_head_norm_after"]), after_norm)
                if hook_stats["first_head_norm_before"] is None:
                    hook_stats["first_head_norm_before"] = before_norm
                    hook_stats["first_head_norm_after"] = after_norm
                out[..., :, int(head), :] = scaled
                return out

            try:
                before_len = len(getattr(hook_point, "fwd_hooks", []))
                hook_point.add_hook(_scale_head, dir="fwd")
                fwd_hooks = getattr(hook_point, "fwd_hooks", [])
                if len(fwd_hooks) <= before_len:
                    return None
                edited = self._simulate_decode(max_new_tokens=3, top_k=8)
            finally:
                if hasattr(hook_point, "fwd_hooks"):
                    original_ids = {id(item) for item in original_fwd_hooks}
                    for item in list(getattr(hook_point, "fwd_hooks", [])):
                        if id(item) in original_ids:
                            continue
                        hook_handle = getattr(item, "hook", None)
                        if hook_handle is not None and hasattr(hook_handle, "remove"):
                            try:
                                hook_handle.remove()
                            except Exception:
                                pass
                    hook_point.fwd_hooks = list(original_fwd_hooks)
            if edited is None:
                return None

            baseline_score = baseline.get("scoring", {}) if isinstance(baseline.get("scoring"), Mapping) else {}
            edited_score = edited.get("scoring", {}) if isinstance(edited.get("scoring"), Mapping) else {}
            result: dict[str, Any] = {
                "status": "ok",
                "label": f"attention_head_scale:{recipe_name}:{site}:L{int(layer)}H{int(head)}:{objective_bundle_key}",
                "intended_bundle_key": objective_bundle_key,
                "intended_term": objective_term,
                "operator_recipe_id": (
                    f"readout_escape|attention_head_scale|{site}|L{int(layer)}H{int(head)}|scale{scale_value:.2f}"
                ),
                "continuation_baseline": baseline["continuation"],
                "continuation_candidate": edited["continuation"],
                "entropy_delta": round(float(edited["entropy"]) - float(baseline["entropy"]), 6),
                "repeat_flag_delta": int(bool(edited["repeat_flag"])) - int(bool(baseline["repeat_flag"])),
                "top1_margin_delta": round(float(edited.get("top1_margin", 0.0) or 0.0) - float(baseline.get("top1_margin", 0.0) or 0.0), 6),
                "repetition_score_delta": round(
                    float(edited.get("repetition_score", 0.0) or 0.0)
                    - float(baseline.get("repetition_score", 0.0) or 0.0),
                    6,
                ),
                "required_term_recall_delta": round(
                    float(edited_score.get("required_term_recall") or 0.0)
                    - float(baseline_score.get("required_term_recall") or 0.0),
                    6,
                ),
                "required_term_span_progress_delta": round(
                    float(edited_score.get("required_term_span_progress") or 0.0)
                    - float(baseline_score.get("required_term_span_progress") or 0.0),
                    6,
                ),
                "semantic_progress_delta": round(
                    float(edited_score.get("semantic_progress_score") or 0.0)
                    - float(baseline_score.get("semantic_progress_score") or 0.0),
                    6,
                ),
            }
            logit_delta = edited["first_logits"].detach().cpu().float() - baseline["first_logits"].detach().cpu().float()
            result["first_logit_delta_l2"] = round(float(torch.linalg.vector_norm(logit_delta).item()), 6)
            result["first_logit_delta_max_abs"] = round(float(logit_delta.abs().max().item()), 6)
            result["first_logit_delta_mean_abs"] = round(float(logit_delta.abs().mean().item()), 6)
            result["attention_hook_call_count"] = int(hook_stats.get("call_count", 0) or 0)
            result["attention_head_norm_before"] = round(float(hook_stats.get("max_head_norm_before", 0.0) or 0.0), 6)
            result["attention_head_norm_after"] = round(float(hook_stats.get("max_head_norm_after", 0.0) or 0.0), 6)
            result["attention_head_norm_removed_estimate"] = round(
                max(
                    0.0,
                    float(hook_stats.get("max_head_norm_before", 0.0) or 0.0)
                    - float(hook_stats.get("max_head_norm_after", 0.0) or 0.0),
                ),
                6,
            )
            result["topk_token_diff"] = self._topk_token_diff(
                baseline["first_logits"],
                edited["first_logits"],
                top_k=5,
            )
            readout_metrics = self._first_token_target_readout_metrics(
                baseline["first_logits"],
                edited["first_logits"],
                focus_terms=ownership_terms,
            )
            if readout_metrics:
                result.update(readout_metrics)
            result["actual_delta_class"] = self._classify_actual_delta_result(result)
            return {
                "diagnostic_only": True,
                "diagnostic_family": "attention_head_scale",
                "recipe_name": recipe_name,
                "objective_bundle_key": objective_bundle_key,
                "objective_term": objective_term,
                "actuator_bundle_key": objective_bundle_key,
                "status": "ok",
                "actuator_class": result.get("actual_delta_class"),
                "operator_recipe_id": result.get("operator_recipe_id"),
                "target_mass_delta": result.get("target_mass_delta"),
                "target_top20_hit_delta": result.get("target_top20_hit_delta"),
                "focus_rank_delta": result.get("focus_rank_delta"),
                "rank_focus_delta": result.get("rank_focus_delta"),
                "required_term_recall_delta": result.get("required_term_recall_delta"),
                "required_term_span_progress_delta": result.get("required_term_span_progress_delta"),
                "entropy_delta": result.get("entropy_delta"),
                "top1_margin_delta": result.get("top1_margin_delta"),
                "repeat_delta": result.get("repeat_flag_delta"),
                "first_logit_delta_l2": result.get("first_logit_delta_l2"),
                "first_logit_delta_max_abs": result.get("first_logit_delta_max_abs"),
                "first_logit_delta_mean_abs": result.get("first_logit_delta_mean_abs"),
                "attention_hook_call_count": result.get("attention_hook_call_count"),
                "attention_head_norm_before": result.get("attention_head_norm_before"),
                "attention_head_norm_after": result.get("attention_head_norm_after"),
                "attention_head_norm_removed_estimate": result.get("attention_head_norm_removed_estimate"),
                "topk_token_diff": result.get("topk_token_diff"),
                "candidate_fingerprint": {
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "actuator_bundle_key": objective_bundle_key,
                    "term": objective_term,
                    "recipe_name": recipe_name,
                    "site": site,
                    "layer": int(layer),
                    "head": int(head),
                    "op_kind": "attention_head_scale",
                    "scale": round(float(scale_value), 6),
                    "edit_count": 0,
                },
                "eval_context_fingerprint": self._eval_context_fingerprint(
                    max_new_tokens=3,
                    top_k=8,
                    max_edits_per_step=0,
                    focus_terms=ownership_terms,
                ),
            }

        def _simulate_decode_with_attention_scale(
            candidate: Mapping[str, Any],
            *,
            scale: float,
            command: Mapping[str, Any] | None = None,
            max_edits_per_step_override: int | None = None,
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            site = str(candidate.get("site", "") or "")
            if site not in {"k_cache", "v_cache"}:
                return None, {"error": "unsupported_attention_site"}
            layer = candidate.get("layer")
            head = candidate.get("head")
            if isinstance(layer, bool) or not isinstance(layer, int):
                return None, {"error": "invalid_attention_layer"}
            if isinstance(head, bool) or not isinstance(head, int):
                return None, {"error": "invalid_attention_head"}
            hook_name = f"blocks.{int(layer)}.attn.hook_{'v' if site == 'v_cache' else 'k'}"
            hook_dict = getattr(self.model, "hook_dict", None)
            if hook_dict is None or hook_name not in hook_dict:
                return None, {"error": "missing_attention_hook"}
            hook_point = hook_dict[hook_name]
            original_fwd_hooks = list(getattr(hook_point, "fwd_hooks", []))
            scale_value = max(0.0, min(1.0, float(scale)))
            hook_stats: dict[str, Any] = {
                "call_count": 0,
                "max_head_norm_before": 0.0,
                "max_head_norm_after": 0.0,
            }

            def _scale_head(act: torch.Tensor, hook: Any | None = None) -> torch.Tensor:
                if not isinstance(act, torch.Tensor) or act.ndim < 4 or int(head) < 0 or act.shape[-2] <= int(head):
                    return act
                out = act.clone()
                head_slice = act[..., :, int(head), :]
                scaled = head_slice * float(scale_value)
                before_norm = float(torch.linalg.vector_norm(head_slice.detach().float()).item())
                after_norm = float(torch.linalg.vector_norm(scaled.detach().float()).item())
                hook_stats["call_count"] = int(hook_stats.get("call_count", 0) or 0) + 1
                hook_stats["max_head_norm_before"] = max(float(hook_stats["max_head_norm_before"]), before_norm)
                hook_stats["max_head_norm_after"] = max(float(hook_stats["max_head_norm_after"]), after_norm)
                out[..., :, int(head), :] = scaled
                return out

            try:
                before_len = len(getattr(hook_point, "fwd_hooks", []))
                hook_point.add_hook(_scale_head, dir="fwd")
                fwd_hooks = getattr(hook_point, "fwd_hooks", [])
                if len(fwd_hooks) <= before_len:
                    return None, {"error": "attention_hook_not_registered"}
                policy_override = (
                    None
                    if command is None
                    else self._replay_policy(max_edits_per_step_override=max_edits_per_step_override)
                )
                return (
                    self._simulate_decode(
                        max_new_tokens=3,
                        top_k=8,
                        command=command,
                        policy_override=policy_override,
                    ),
                    hook_stats,
                )
            finally:
                if hasattr(hook_point, "fwd_hooks"):
                    original_ids = {id(item) for item in original_fwd_hooks}
                    for item in list(getattr(hook_point, "fwd_hooks", [])):
                        if id(item) in original_ids:
                            continue
                        hook_handle = getattr(item, "hook", None)
                        if hook_handle is not None and hasattr(hook_handle, "remove"):
                            try:
                                hook_handle.remove()
                            except Exception:
                                pass
                    hook_point.fwd_hooks = list(original_fwd_hooks)

        def _run_attention_guided_operator(
            *,
            attention_candidate: Mapping[str, Any],
            operator_candidates: Sequence[Mapping[str, Any]],
            objective_bundle_key: str,
            objective_term: str,
            recipe_name: str,
            scale: float = 0.5,
        ) -> dict[str, Any] | None:
            prepared_edits = [_ensure_compileable_candidate(item) for item in operator_candidates if isinstance(item, Mapping)]
            prepared_edits = [dict(item) for item in prepared_edits if isinstance(item, Mapping)]
            if not prepared_edits:
                return None
            carrier, carrier_stats = _simulate_decode_with_attention_scale(
                attention_candidate,
                scale=scale,
            )
            if carrier is None:
                return None
            command = {"version": "0.1", "decision": "apply", "edits": prepared_edits}
            edited, edited_stats = _simulate_decode_with_attention_scale(
                attention_candidate,
                scale=scale,
                command=command,
                max_edits_per_step_override=max(1, len(prepared_edits)),
            )
            if edited is None:
                return None
            carrier_score = carrier.get("scoring", {}) if isinstance(carrier.get("scoring"), Mapping) else {}
            edited_score = edited.get("scoring", {}) if isinstance(edited.get("scoring"), Mapping) else {}
            result: dict[str, Any] = {
                "status": "ok",
                "label": f"{recipe_name}:{objective_bundle_key}",
                "edit_count": len(prepared_edits),
                "intended_bundle_key": objective_bundle_key,
                "intended_term": objective_term,
                "operator_recipe_id": f"attention_guided|scale{float(scale):.2f}|{recipe_name}",
                "operator_recipe_ids": sorted(
                    {
                        str(item.get("operator_recipe_id", self._operator_recipe_id(item)) or "")
                        for item in prepared_edits
                        if str(item.get("operator_recipe_id", self._operator_recipe_id(item)) or "")
                    }
                ),
                "operator_recipe_seed_key": f"attention_guided_operator|scale{float(scale):.2f}|{recipe_name}",
                "continuation_baseline": carrier["continuation"],
                "continuation_candidate": edited["continuation"],
                "entropy_delta": round(float(edited["entropy"]) - float(carrier["entropy"]), 6),
                "repeat_flag_delta": int(bool(edited["repeat_flag"])) - int(bool(carrier["repeat_flag"])),
                "top1_margin_delta": round(
                    float(edited.get("top1_margin", 0.0) or 0.0) - float(carrier.get("top1_margin", 0.0) or 0.0),
                    6,
                ),
                "repetition_score_delta": round(
                    float(edited.get("repetition_score", 0.0) or 0.0)
                    - float(carrier.get("repetition_score", 0.0) or 0.0),
                    6,
                ),
                "required_term_recall_delta": round(
                    float(edited_score.get("required_term_recall") or 0.0)
                    - float(carrier_score.get("required_term_recall") or 0.0),
                    6,
                ),
                "required_term_span_progress_delta": round(
                    float(edited_score.get("required_term_span_progress") or 0.0)
                    - float(carrier_score.get("required_term_span_progress") or 0.0),
                    6,
                ),
                "semantic_progress_delta": round(
                    float(edited_score.get("semantic_progress_score") or 0.0)
                    - float(carrier_score.get("semantic_progress_score") or 0.0),
                    6,
                ),
            }
            readout_metrics = self._first_token_target_readout_metrics(
                carrier["first_logits"],
                edited["first_logits"],
                focus_terms=ownership_terms,
            )
            if readout_metrics:
                result.update(readout_metrics)
            term_readout_deltas = self._term_readout_deltas(
                carrier["first_logits"],
                edited["first_logits"],
                focus_terms=ownership_terms,
            )
            if term_readout_deltas:
                result["term_readout_deltas"] = dict(term_readout_deltas)
            result["topk_token_diff"] = self._topk_token_diff(
                carrier["first_logits"],
                edited["first_logits"],
                top_k=5,
            )
            result["candidate_fingerprint"] = self._candidate_fingerprint(
                prepared_edits,
                intended_bundle_key=objective_bundle_key,
                label=f"{recipe_name}:{objective_bundle_key}",
            )
            result["eval_context_fingerprint"] = self._eval_context_fingerprint(
                max_new_tokens=3,
                top_k=8,
                max_edits_per_step=len(prepared_edits),
                focus_terms=ownership_terms,
            )
            result["actual_delta_class"] = self._classify_actual_delta_result(result)
            site = str(attention_candidate.get("site", "") or "")
            layer = attention_candidate.get("layer")
            head = attention_candidate.get("head")
            return {
                "diagnostic_only": True,
                "diagnostic_family": "attention_guided_operator",
                "recipe_name": recipe_name,
                "objective_bundle_key": objective_bundle_key,
                "objective_term": objective_term,
                "actuator_bundle_key": objective_bundle_key,
                "status": "ok",
                "actuator_class": result.get("actual_delta_class"),
                "operator_recipe_id": result.get("operator_recipe_id"),
                "operator_recipe_ids": result.get("operator_recipe_ids"),
                "operator_recipe_seed_key": result.get("operator_recipe_seed_key"),
                "target_mass_delta": result.get("target_mass_delta"),
                "target_top20_hit_delta": result.get("target_top20_hit_delta"),
                "focus_rank_delta": result.get("focus_rank_delta"),
                "rank_focus_delta": result.get("rank_focus_delta"),
                "required_term_recall_delta": result.get("required_term_recall_delta"),
                "required_term_span_progress_delta": result.get("required_term_span_progress_delta"),
                "entropy_delta": result.get("entropy_delta"),
                "top1_margin_delta": result.get("top1_margin_delta"),
                "repeat_delta": result.get("repeat_flag_delta"),
                "first_logit_delta_l2": round(
                    float(torch.linalg.vector_norm(edited["first_logits"].detach().cpu().float() - carrier["first_logits"].detach().cpu().float()).item()),
                    6,
                ),
                "attention_carrier_site": site,
                "attention_carrier_layer": int(layer) if isinstance(layer, int) and not isinstance(layer, bool) else None,
                "attention_carrier_head": int(head) if isinstance(head, int) and not isinstance(head, bool) else None,
                "attention_carrier_scale": round(float(scale), 6),
                "attention_hook_call_count": int(edited_stats.get("call_count", carrier_stats.get("call_count", 0)) or 0),
                "attention_head_norm_before": round(
                    float(edited_stats.get("max_head_norm_before", carrier_stats.get("max_head_norm_before", 0.0)) or 0.0),
                    6,
                ),
                "attention_head_norm_after": round(
                    float(edited_stats.get("max_head_norm_after", carrier_stats.get("max_head_norm_after", 0.0)) or 0.0),
                    6,
                ),
                "topk_token_diff": result.get("topk_token_diff"),
                "candidate_fingerprint": result.get("candidate_fingerprint"),
                "eval_context_fingerprint": result.get("eval_context_fingerprint"),
            }

        def _activation_hook_name(site: str, layer: int) -> str:
            templates = getattr(self.adapter, "HOOK_SITE_TEMPLATES", None)
            if isinstance(templates, Mapping) and str(site) in templates:
                return str(templates[str(site)]).format(layer=int(layer))
            return f"blocks.{int(layer)}.hook_{str(site)}"

        def _activation_source_vector(
            *,
            site: str,
            layer: int,
            source_span: Mapping[str, Any] | None,
            stealer_span: Mapping[str, Any] | None = None,
            stealer_bundle_key: str | None = None,
            stealer_term: str | None = None,
            source_localization: str = "source_span_mean",
        ) -> tuple[torch.Tensor, dict[str, Any]] | None:
            if not isinstance(source_span, Mapping):
                return None
            if getattr(self.runtime_state, "last_cache", None) is None:
                primer_tokens = self._current_token_tensor()
                self.runtime_state.run_with_cache(primer_tokens, return_type="logits")
            cache = getattr(self.runtime_state, "last_cache", None) or {}
            hook_name = _activation_hook_name(site, int(layer))
            tensor = cache.get(hook_name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
                return None
            batch0 = tensor[0].detach()
            seq_len = int(batch0.shape[0])
            if seq_len <= 0:
                return None

            def _parse_source_localization(raw_localization: str) -> tuple[str, str, float]:
                normalized = str(raw_localization or "source_span_mean")
                if normalized.endswith("_minus_stealer_l025"):
                    return normalized[: -len("_minus_stealer_l025")], "minus_stealer", 0.25
                if normalized.endswith("_minus_stealer_l050"):
                    return normalized[: -len("_minus_stealer_l050")], "minus_stealer", 0.50
                if normalized.endswith("_orthogonal_stealer"):
                    return normalized[: -len("_orthogonal_stealer")], "orthogonal_stealer", 1.0
                return normalized, "none", 0.0

            def _base_source_vector_for_span(
                span: Mapping[str, Any],
                *,
                base_localization: str,
            ) -> tuple[torch.Tensor, dict[str, Any]] | None:
                raw_start = int(span.get("start", 0) or 0)
                raw_end = int(span.get("end", raw_start + 1) or (raw_start + 1))
                start = max(0, min(seq_len - 1, raw_start))
                end = max(start + 1, min(seq_len, raw_end))
                localization = str(base_localization or "source_span_mean")
                span_records = self._prompt_records_for_span(start, end)
                if localization == "source_term_token" and span_records:
                    best_record = max(
                        span_records,
                        key=lambda record: self._content_piece_score(str(record.get("piece", ""))),
                    )
                    token_position = max(0, min(seq_len - 1, int(best_record.get("position", start) or start)))
                    vector = batch0[token_position].float().reshape(-1)
                    positions = [int(token_position)]
                elif localization == "source_centered_pm1":
                    window_start = max(0, start - 1)
                    window_end = min(seq_len, end + 1)
                    positive = batch0[start:end].float().mean(dim=0)
                    neighbor_chunks = []
                    if window_start < start:
                        neighbor_chunks.append(batch0[window_start:start].float())
                    if end < window_end:
                        neighbor_chunks.append(batch0[end:window_end].float())
                    if neighbor_chunks:
                        neighbor = torch.cat(neighbor_chunks, dim=0).mean(dim=0)
                        vector = (positive - neighbor).reshape(-1)
                    else:
                        vector = positive.reshape(-1)
                    positions = list(range(window_start, window_end))
                else:
                    localization = "source_span_mean"
                    vector = batch0[start:end].float().mean(dim=0).reshape(-1)
                    positions = list(range(start, end))
                return vector, {
                    "source_token_index": int(start),
                    "source_span_start": int(start),
                    "source_span_end": int(end),
                    "source_piece": span.get("text"),
                    "source_segment_kind": span.get("segment_kind"),
                    "source_provenance_class": span.get("provenance_class"),
                    "source_localization": localization,
                    "source_positions": positions,
                }

            requested_localization = str(source_localization or "source_span_mean")
            base_localization, contrast_mode, contrast_scale = _parse_source_localization(requested_localization)
            base_source = _base_source_vector_for_span(source_span, base_localization=base_localization)
            if base_source is None:
                return None
            vector, source_meta = base_source
            source_positions = list(source_meta.get("source_positions", []))
            if contrast_mode != "none":
                if not isinstance(stealer_span, Mapping):
                    return None
                stealer_source = _base_source_vector_for_span(stealer_span, base_localization=base_localization)
                if stealer_source is None:
                    return None
                stealer_vector, stealer_meta = stealer_source
                source_positions = list(
                    dict.fromkeys(source_positions + list(stealer_meta.get("source_positions", [])))
                )
                if contrast_mode == "orthogonal_stealer":
                    basis = stealer_vector.float().reshape(-1)
                    denom = float(torch.dot(basis, basis).item())
                    if denom > 1e-8:
                        vector = vector.float().reshape(-1) - (torch.dot(vector.float().reshape(-1), basis) / denom) * basis
                    else:
                        vector = vector.float().reshape(-1)
                else:
                    vector = vector.float().reshape(-1) - (float(contrast_scale) * stealer_vector.float().reshape(-1))
                source_meta["stealer_source_token_index"] = stealer_meta.get("source_token_index")
                source_meta["stealer_source_span_start"] = stealer_meta.get("source_span_start")
                source_meta["stealer_source_span_end"] = stealer_meta.get("source_span_end")
                source_meta["stealer_source_piece"] = stealer_meta.get("source_piece")
                source_meta["stealer_source_segment_kind"] = stealer_meta.get("source_segment_kind")
                source_meta["stealer_source_provenance_class"] = stealer_meta.get("source_provenance_class")
            else:
                vector = vector.float().reshape(-1)
            localization = requested_localization
            if vector.numel() <= 0:
                return None
            source_meta["source_localization"] = localization
            source_meta["source_base_localization"] = base_localization
            source_meta["source_contrast_mode"] = contrast_mode
            source_meta["source_contrast_scale"] = round(float(contrast_scale), 6)
            source_meta["source_positions"] = source_positions
            source_meta["source_norm"] = round(float(torch.linalg.vector_norm(vector).item()), 6)
            source_meta["source_hook_name"] = hook_name
            source_meta["stealer_bundle_key"] = stealer_bundle_key
            source_meta["stealer_term"] = stealer_term
            return vector.detach().cpu().float(), source_meta

        def _run_activation_patch_operator(
            *,
            site: str,
            layer: int,
            source_span: Mapping[str, Any] | None,
            objective_bundle_key: str,
            objective_term: str,
            recipe_name: str,
            stealer_span: Mapping[str, Any] | None = None,
            stealer_bundle_key: str | None = None,
            stealer_term: str | None = None,
            source_localization: str = "source_span_mean",
            patch_mode: str = "blend",
            alpha: float = 0.15,
        ) -> dict[str, Any] | None:
            source = _activation_source_vector(
                site=site,
                layer=int(layer),
                source_span=source_span,
                stealer_span=stealer_span,
                stealer_bundle_key=stealer_bundle_key,
                stealer_term=stealer_term,
                source_localization=source_localization,
            )
            if source is None:
                return None
            source_vector, source_meta = source
            hook_name = _activation_hook_name(site, int(layer))
            hook_dict = getattr(self.model, "hook_dict", None)
            if hook_dict is None or hook_name not in hook_dict:
                return None
            baseline = self._simulate_decode(max_new_tokens=3, top_k=8)
            if baseline is None:
                return None

            hook_point = hook_dict[hook_name]
            original_fwd_hooks = list(getattr(hook_point, "fwd_hooks", []))
            alpha_value = max(0.0, min(1.0, float(alpha)))
            hook_stats: dict[str, Any] = {
                "call_count": 0,
                "max_target_norm_before": 0.0,
                "max_target_norm_after": 0.0,
                "source_norm": float(source_meta.get("source_norm", 0.0) or 0.0),
            }

            def _patch_activation(act: torch.Tensor, hook: Any | None = None) -> torch.Tensor:
                if not isinstance(act, torch.Tensor) or act.ndim < 3:
                    return act
                if act.shape[-1] != source_vector.numel() or act.shape[-2] <= 0:
                    return act
                out = act.clone()
                target_slice = act[:, -1, :]
                src = source_vector.to(device=act.device, dtype=act.dtype).reshape(1, -1)
                if patch_mode == "add_norm":
                    src_norm = torch.linalg.vector_norm(src.detach().float(), dim=-1, keepdim=True).to(
                        device=act.device,
                        dtype=act.dtype,
                    )
                    src_unit = src / torch.clamp(src_norm, min=1e-8)
                    patched = target_slice + (alpha_value * src_unit)
                else:
                    patched = ((1.0 - alpha_value) * target_slice) + (alpha_value * src)
                before_norm = float(torch.linalg.vector_norm(target_slice.detach().float()).item())
                after_norm = float(torch.linalg.vector_norm(patched.detach().float()).item())
                hook_stats["call_count"] = int(hook_stats.get("call_count", 0) or 0) + 1
                hook_stats["max_target_norm_before"] = max(float(hook_stats["max_target_norm_before"]), before_norm)
                hook_stats["max_target_norm_after"] = max(float(hook_stats["max_target_norm_after"]), after_norm)
                out[:, -1, :] = patched
                return out

            try:
                before_len = len(getattr(hook_point, "fwd_hooks", []))
                hook_point.add_hook(_patch_activation, dir="fwd")
                fwd_hooks = getattr(hook_point, "fwd_hooks", [])
                if len(fwd_hooks) <= before_len:
                    return None
                edited = self._simulate_decode(max_new_tokens=3, top_k=8)
            finally:
                if hasattr(hook_point, "fwd_hooks"):
                    original_ids = {id(item) for item in original_fwd_hooks}
                    for item in list(getattr(hook_point, "fwd_hooks", [])):
                        if id(item) in original_ids:
                            continue
                        hook_handle = getattr(item, "hook", None)
                        if hook_handle is not None and hasattr(hook_handle, "remove"):
                            try:
                                hook_handle.remove()
                            except Exception:
                                pass
                    hook_point.fwd_hooks = list(original_fwd_hooks)
            if edited is None:
                return None

            baseline_score = baseline.get("scoring", {}) if isinstance(baseline.get("scoring"), Mapping) else {}
            edited_score = edited.get("scoring", {}) if isinstance(edited.get("scoring"), Mapping) else {}
            recipe_segment = (
                "source_span_to_last"
                if source_localization == "source_span_mean"
                else f"{source_localization}_to_last"
            )
            result: dict[str, Any] = {
                "status": "ok",
                "label": f"{recipe_name}:{objective_bundle_key}",
                "edit_count": 0,
                "intended_bundle_key": objective_bundle_key,
                "intended_term": objective_term,
                "operator_recipe_id": (
                    f"readout_escape|activation_patch|{site}|L{int(layer)}|"
                    f"{recipe_segment}|{patch_mode}|a{alpha_value:.3f}"
                ),
                "operator_recipe_seed_key": f"activation_patch|{site}|{recipe_segment}|{patch_mode}",
                "continuation_baseline": baseline["continuation"],
                "continuation_candidate": edited["continuation"],
                "entropy_delta": round(float(edited["entropy"]) - float(baseline["entropy"]), 6),
                "repeat_flag_delta": int(bool(edited["repeat_flag"])) - int(bool(baseline["repeat_flag"])),
                "top1_margin_delta": round(
                    float(edited.get("top1_margin", 0.0) or 0.0) - float(baseline.get("top1_margin", 0.0) or 0.0),
                    6,
                ),
                "repetition_score_delta": round(
                    float(edited.get("repetition_score", 0.0) or 0.0)
                    - float(baseline.get("repetition_score", 0.0) or 0.0),
                    6,
                ),
                "required_term_recall_delta": round(
                    float(edited_score.get("required_term_recall") or 0.0)
                    - float(baseline_score.get("required_term_recall") or 0.0),
                    6,
                ),
                "required_term_span_progress_delta": round(
                    float(edited_score.get("required_term_span_progress") or 0.0)
                    - float(baseline_score.get("required_term_span_progress") or 0.0),
                    6,
                ),
                "semantic_progress_delta": round(
                    float(edited_score.get("semantic_progress_score") or 0.0)
                    - float(baseline_score.get("semantic_progress_score") or 0.0),
                    6,
                ),
            }
            readout_metrics = self._first_token_target_readout_metrics(
                baseline["first_logits"],
                edited["first_logits"],
                focus_terms=ownership_terms,
            )
            if readout_metrics:
                result.update(readout_metrics)
            term_readout_deltas = self._term_readout_deltas(
                baseline["first_logits"],
                edited["first_logits"],
                focus_terms=ownership_terms,
            )
            if term_readout_deltas:
                result["term_readout_deltas"] = dict(term_readout_deltas)
            result["topk_token_diff"] = self._topk_token_diff(
                baseline["first_logits"],
                edited["first_logits"],
                top_k=5,
            )
            logit_delta = edited["first_logits"].detach().cpu().float() - baseline["first_logits"].detach().cpu().float()
            result["first_logit_delta_l2"] = round(float(torch.linalg.vector_norm(logit_delta).item()), 6)
            result["actual_delta_class"] = self._classify_actual_delta_result(result)
            bundle_scores: dict[str, float] = {}
            term_deltas = result.get("term_readout_deltas") if isinstance(result.get("term_readout_deltas"), Mapping) else {}
            for lift_bundle_key, lift_term in bundle_terms.items():
                metrics = term_deltas.get(str(lift_term)) if isinstance(term_deltas, Mapping) else None
                score = float(metrics.get("lift_score", 0.0) or 0.0) if isinstance(metrics, Mapping) else 0.0
                bundle_scores[str(lift_bundle_key)] = float(score)
            realized_lift_bundle_key = (
                max(bundle_scores.items(), key=lambda entry: (float(entry[1]), str(entry[0])))[0]
                if bundle_scores
                else objective_bundle_key
            )
            self_delta = float(bundle_scores.get(objective_bundle_key, 0.0) or 0.0)
            cross_delta = max(
                (
                    float(score)
                    for lift_bundle_key, score in bundle_scores.items()
                    if str(lift_bundle_key) != objective_bundle_key
                ),
                default=0.0,
            )
            alignment_margin = round(float(self_delta - cross_delta), 6)
            actual_delta_class = str(result.get("actual_delta_class", "") or "")
            if actual_delta_class in {"collapse_sharpener", "harmful"}:
                actuator_class = actual_delta_class
            elif self_delta > 0.03 and alignment_margin > 0.01:
                actuator_class = "self_actuator"
            elif self_delta > 0.005 and alignment_margin < -0.01 and str(realized_lift_bundle_key) != objective_bundle_key:
                actuator_class = "cross_bound"
            elif cross_delta > 0.03 and str(realized_lift_bundle_key) != objective_bundle_key:
                actuator_class = "bridge_actuator"
            elif max(self_delta, cross_delta) <= 0.01:
                actuator_class = "dead_actuator"
            else:
                actuator_class = "noisy_or_harmful"
            return {
                "diagnostic_only": True,
                "diagnostic_family": "activation_patch",
                "operator_axis": "activation_patch",
                "recipe_name": recipe_name,
                "objective_bundle_key": objective_bundle_key,
                "objective_term": objective_term,
                "actuator_bundle_key": objective_bundle_key,
                "status": "ok",
                "actuator_class": actuator_class,
                "actual_delta_class": actual_delta_class,
                "realized_lift_bundle_key": str(realized_lift_bundle_key) if bundle_scores else None,
                "realized_lift_term": str(bundle_terms.get(str(realized_lift_bundle_key), "") or "") if bundle_scores else None,
                "self_delta": round(float(self_delta), 6),
                "cross_delta": round(float(cross_delta), 6),
                "alignment_margin": alignment_margin,
                "bundle_lift_scores": {
                    str(lift_bundle_key): round(float(score), 6)
                    for lift_bundle_key, score in sorted(bundle_scores.items(), key=lambda entry: str(entry[0]))
                },
                "operator_recipe_id": result.get("operator_recipe_id"),
                "operator_recipe_seed_key": result.get("operator_recipe_seed_key"),
                "target_mass_delta": result.get("target_mass_delta"),
                "target_top20_hit_delta": result.get("target_top20_hit_delta"),
                "focus_rank_delta": result.get("focus_rank_delta"),
                "rank_focus_delta": result.get("rank_focus_delta"),
                "required_term_recall_delta": result.get("required_term_recall_delta"),
                "required_term_span_progress_delta": result.get("required_term_span_progress_delta"),
                "entropy_delta": result.get("entropy_delta"),
                "top1_margin_delta": result.get("top1_margin_delta"),
                "repeat_delta": result.get("repeat_flag_delta"),
                "first_logit_delta_l2": result.get("first_logit_delta_l2"),
                "activation_patch_site": site,
                "activation_patch_layer": int(layer),
                "activation_patch_alpha": round(float(alpha_value), 6),
                "activation_patch_source_localization": source_localization,
                "activation_patch_patch_mode": patch_mode,
                "activation_patch_base_localization": source_meta.get("source_base_localization"),
                "activation_patch_contrast_mode": source_meta.get("source_contrast_mode"),
                "activation_patch_contrast_scale": source_meta.get("source_contrast_scale"),
                "activation_patch_stealer_bundle_key": stealer_bundle_key,
                "activation_patch_stealer_term": stealer_term,
                "activation_hook_call_count": int(hook_stats.get("call_count", 0) or 0),
                "activation_source_norm": round(float(hook_stats.get("source_norm", 0.0) or 0.0), 6),
                "activation_target_norm_before": round(float(hook_stats.get("max_target_norm_before", 0.0) or 0.0), 6),
                "activation_target_norm_after": round(float(hook_stats.get("max_target_norm_after", 0.0) or 0.0), 6),
                "topk_token_diff": result.get("topk_token_diff"),
                "candidate_fingerprint": {
                    "bundle_key": objective_bundle_key,
                    "objective_bundle_key": objective_bundle_key,
                    "actuator_bundle_key": objective_bundle_key,
                    "term": objective_term,
                    "recipe_name": recipe_name,
                    "site": site,
                    "layer": int(layer),
                    "source_token_index": source_meta.get("source_token_index"),
                    "source_piece": source_meta.get("source_piece"),
                    "source_segment_kind": source_meta.get("source_segment_kind"),
                    "source_localization": source_localization,
                    "base_localization": source_meta.get("source_base_localization"),
                    "contrast_mode": source_meta.get("source_contrast_mode"),
                    "contrast_scale": source_meta.get("source_contrast_scale"),
                    "stealer_bundle_key": stealer_bundle_key,
                    "stealer_term": stealer_term,
                    "stealer_source_token_index": source_meta.get("stealer_source_token_index"),
                    "stealer_source_piece": source_meta.get("stealer_source_piece"),
                    "op_kind": "activation_patch",
                    "which": f"{source_localization}_to_last_{patch_mode}",
                    "alpha": round(float(alpha_value), 6),
                    "edit_count": 0,
                },
                "eval_context_fingerprint": self._eval_context_fingerprint(
                    max_new_tokens=3,
                    top_k=8,
                    max_edits_per_step=0,
                    focus_terms=ownership_terms,
                ),
            }

        rows: list[dict[str, Any]] = []
        grouped_candidates = _kv_candidates_by_bundle()
        for bundle_key, term in list(bundle_terms.items())[:2]:
            source_span = _best_source_span(term)
            source_selector = _source_selector_for_span(source_span)
            stealer_bundle_key: str | None = None
            stealer_term: str | None = None
            stealer_span: dict[str, Any] | None = None
            for other_bundle_key, other_term in bundle_terms.items():
                if str(other_bundle_key) == str(bundle_key):
                    continue
                other_span = _best_source_span(str(other_term))
                if isinstance(other_span, Mapping):
                    stealer_bundle_key = str(other_bundle_key)
                    stealer_term = str(other_term)
                    stealer_span = dict(other_span)
                    break
            non_kv_operator_candidates: list[dict[str, Any]] = []
            for diagnostic_family, recipe_name, selector, span, alpha in (
                (
                    "resid_source_span",
                    "resid_source_span_exact",
                    source_selector,
                    source_span,
                    0.035,
                ),
                (
                    "readout_local_boundary",
                    "readout_boundary_self",
                    {"mode": "last"},
                    None,
                    0.02,
                ),
            ):
                edit = _make_resid_edit(
                    surface=readout_surface,
                    bundle_key=bundle_key,
                    term=term,
                    diagnostic_family=diagnostic_family,
                    recipe_name=recipe_name,
                    source_selector=selector,
                    source_span=span,
                    alpha=alpha,
                )
                non_kv_operator_candidates.append(dict(edit))
                result = self.replay_candidate_edits_actual_delta(
                    [edit],
                    max_new_tokens=3,
                    top_k=8,
                    label=f"{recipe_name}:{bundle_key}",
                    ownership_terms=ownership_terms,
                    intended_bundle_key=bundle_key,
                    intended_term=term,
                )
                result["operator_recipe_id"] = str(edit.get("operator_recipe_id", "") or result.get("operator_recipe_id", ""))
                result["operator_recipe_seed_key"] = str(edit.get("operator_recipe_seed_key", "") or "")
                rows.append(
                    _compact_actual_delta(
                        result,
                        diagnostic_family=diagnostic_family,
                        recipe_name=recipe_name,
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                    )
                )
            readout_target = getattr(readout_surface, "target", None)
            readout_layer = int(getattr(readout_target, "layer", 0) or 0)
            if isinstance(source_span, Mapping):
                for patch_site, patch_alpha, source_localization, patch_mode in (
                    ("resid_pre", 0.05, "source_term_token", "blend"),
                    ("resid_pre", 0.05, "source_term_token_minus_stealer_l025", "blend"),
                    ("resid_pre", 0.05, "source_centered_pm1", "blend"),
                    ("resid_pre", 0.05, "source_centered_pm1_minus_stealer_l025", "blend"),
                    ("resid_pre", 0.05, "source_term_token_orthogonal_stealer", "blend"),
                    ("resid_pre", 0.05, "source_span_mean", "add_norm"),
                    ("resid_pre", 0.05, "source_span_mean", "blend"),
                ):
                    if "stealer" in source_localization and not isinstance(stealer_span, Mapping):
                        continue
                    patch_row = _run_activation_patch_operator(
                        site=patch_site,
                        layer=readout_layer,
                        source_span=source_span,
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                        recipe_name=(
                            f"{patch_site}_{source_localization}_to_last_{patch_mode}"
                            f"_a{int(round(patch_alpha * 1000)):03d}"
                        ),
                        stealer_span=stealer_span,
                        stealer_bundle_key=stealer_bundle_key,
                        stealer_term=stealer_term,
                        source_localization=source_localization,
                        patch_mode=patch_mode,
                        alpha=patch_alpha,
                    )
                    if patch_row is not None:
                        rows.append(patch_row)
            head_candidate = next(
                (
                    item
                    for item in grouped_candidates.get(bundle_key, [])
                    if str(item.get("site", "") or "") == "v_cache"
                ),
                None,
            ) or next(iter(grouped_candidates.get(bundle_key, [])), None)
            if isinstance(head_candidate, Mapping):
                for scale, recipe_name in (
                    (0.0, "head_zero_ablation"),
                    (0.25, "head_scale_025"),
                    (0.5, "head_scale_050"),
                    (0.75, "head_scale_075"),
                ):
                    scale_row = _run_attention_head_scale(
                        head_candidate,
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                        scale=scale,
                        recipe_name=recipe_name,
                    )
                    if scale_row is not None:
                        rows.append(scale_row)
                v_member = next(
                    (
                        item
                        for item in grouped_candidates.get(bundle_key, [])
                        if str(item.get("site", "") or "") == "v_cache"
                    ),
                    None,
                )
                k_member = next(
                    (
                        item
                        for item in grouped_candidates.get(bundle_key, [])
                        if str(item.get("site", "") or "") == "k_cache"
                    ),
                    None,
                )
                if isinstance(v_member, Mapping):
                    guided_v = _run_attention_guided_operator(
                        attention_candidate=head_candidate,
                        operator_candidates=[v_member],
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                        recipe_name="head_scale_050_plus_kv_v",
                        scale=0.5,
                    )
                    if guided_v is not None:
                        rows.append(guided_v)
                if isinstance(v_member, Mapping) and isinstance(k_member, Mapping):
                    guided_pair = _run_attention_guided_operator(
                        attention_candidate=head_candidate,
                        operator_candidates=[v_member, k_member],
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                        recipe_name="head_scale_050_plus_kv_pair",
                        scale=0.5,
                    )
                    if guided_pair is not None:
                        rows.append(guided_pair)
                for non_kv_edit in non_kv_operator_candidates[:2]:
                    non_kv_recipe = str(non_kv_edit.get("recipe_name", "") or non_kv_edit.get("role", "") or "")
                    guided_non_kv = _run_attention_guided_operator(
                        attention_candidate=head_candidate,
                        operator_candidates=[non_kv_edit],
                        objective_bundle_key=bundle_key,
                        objective_term=term,
                        recipe_name=f"head_scale_050_plus_{non_kv_recipe or 'non_kv_operator'}",
                        scale=0.5,
                    )
                    if guided_non_kv is not None:
                        guided_non_kv["operator_axis"] = "attention_guided_non_kv"
                        rows.append(guided_non_kv)

        baseline = self._simulate_decode(max_new_tokens=1, top_k=8)
        if isinstance(baseline, Mapping) and isinstance(baseline.get("first_logits"), torch.Tensor):
            baseline_logits = baseline["first_logits"].detach().cpu().float()
            vocab_size = int(baseline_logits.shape[-1])
            for bundle_key, term in list(bundle_terms.items())[:2]:
                edited_logits = baseline_logits.clone()
                token_sequences = self._target_token_sequences([term], vocab_size=vocab_size)
                token_ids = sorted({int(sequence.token_ids[0]) for sequence in token_sequences if sequence.token_ids})
                for token_id in token_ids:
                    if 0 <= token_id < vocab_size:
                        edited_logits[token_id] += 0.25
                readout_metrics = self._first_token_target_readout_metrics(
                    baseline_logits,
                    edited_logits,
                    focus_terms=[term],
                )
                target_mass_delta = float(readout_metrics.get("target_mass_delta", 0.0) or 0.0) if readout_metrics else 0.0
                target_top20_hit_delta = int(readout_metrics.get("target_top20_hit_delta", 0) or 0) if readout_metrics else 0
                focus_rank_delta = int(readout_metrics.get("focus_rank_delta", 0) or 0) if readout_metrics else 0
                rows.append(
                    {
                        "diagnostic_only": True,
                        "diagnostic_family": "logit_adjacent",
                        "recipe_name": "first_piece_soft_bias_probe",
                        "objective_bundle_key": bundle_key,
                        "objective_term": term,
                        "actuator_bundle_key": None,
                        "status": "ok",
                        "actuator_class": "target_lift"
                        if target_mass_delta > 0.00002 or target_top20_hit_delta > 0 or focus_rank_delta >= 6
                        else "neutral",
                        "operator_recipe_id": "diagnostic|logit_adjacent|first_piece_soft_bias_probe|bias0.2500",
                        "target_mass_delta": round(float(target_mass_delta), 6),
                        "target_top20_hit_delta": int(target_top20_hit_delta),
                        "focus_rank_delta": int(focus_rank_delta),
                        "rank_focus_delta": int(readout_metrics.get("rank_focus_delta", 0) or 0) if readout_metrics else 0,
                        "required_term_recall_delta": 0.0,
                        "required_term_span_progress_delta": 0.0,
                        "entropy_delta": None,
                        "top1_margin_delta": None,
                        "repeat_delta": 0,
                        "candidate_fingerprint": {
                            "bundle_key": bundle_key,
                            "objective_bundle_key": bundle_key,
                            "actuator_bundle_key": None,
                            "term": term,
                            "recipe_name": "first_piece_soft_bias_probe",
                            "site": "logits",
                            "op_kind": "diagnostic_logit_bias",
                            "alpha": 0.25,
                            "edit_count": 0,
                        },
                        "eval_context_fingerprint": self._eval_context_fingerprint(
                            max_new_tokens=1,
                            top_k=8,
                            max_edits_per_step=0,
                            focus_terms=[term],
                        ),
                    }
                )
        return rows

    def _run_bridge_eval_from_packet(
        self: HookedTransformerWorkerRuntime,
        *,
        packet: Mapping[str, Any],
        strategy_hints: Mapping[str, Any],
    ) -> None:
        forced_state["bridge_eval_packet_snapshot"] = None
        shadow_candidates, shadow_bundle_keys = _build_bridge_eval_shadow_candidates(self, strategy_hints)
        forced_state["bridge_eval_shadow_candidates"] = tuple(dict(item) for item in shadow_candidates)
        forced_state["bridge_eval_shadow_bundle_keys"] = tuple(str(item) for item in shadow_bundle_keys if str(item))
        forced_state["bridge_eval_active"] = True
        forced_state["bridge_eval_locked_step"] = int(packet.get("step", 0) or 0)
        frontier_snapshot = _bridge_frontier_snapshot(strategy_hints)
        try:
            bridge_eval_recipe_specs = _focused_bridge_eval_recipe_specs()
            replay_bundle_keys = [
                str(item)
                for item in _candidate_bundle_keys(strategy_hints.get("kv_candidate_edits"))
                if str(item)
            ]
            replay_bundle_keys.extend(
                [
                    str(item)
                    for item in forced_state.get("bridge_eval_shadow_bundle_keys", ())
                    if str(item) and str(item) not in replay_bundle_keys
                ]
            )
            if not replay_bundle_keys:
                replay_bundle_keys = [
                    str(item)
                    for item in (
                        strategy_hints.get("selected_bundle_key"),
                        strategy_hints.get("challenger_bundle_key"),
                        strategy_hints.get("base_winner_bundle_key"),
                    )
                    if item not in (None, "") and str(item)
                ]
            replay_summary = self.replay_operator_recipe_matrix(
                bundle_keys=replay_bundle_keys,
                recipe_specs=bridge_eval_recipe_specs,
            )
            forced_state["bridge_eval_last_replay_summary"] = replay_summary
            recommendations = replay_summary.get("bridge_plan_recommendations") if isinstance(replay_summary, Mapping) else None
            ownership_items = (
                replay_summary.get("operator_recipe_bundle_ownership")
                if isinstance(replay_summary, Mapping)
                else None
            )
            mode_diagnostic_items = (
                replay_summary.get("operator_recipe_mode_diagnostics")
                if isinstance(replay_summary, Mapping)
                else None
            )
            objective_class_counts: dict[str, dict[str, int]] = {}
            bridge_eval_matrix: list[dict[str, Any]] = []
            if isinstance(ownership_items, SequenceABC) and not isinstance(ownership_items, (str, bytes, bytearray)):
                for item in ownership_items:
                    if not isinstance(item, Mapping):
                        continue
                    objective_key = str(item.get("intended_bundle_key", "") or "")
                    actuator_class = str(item.get("actuator_class", "") or "")
                    if not objective_key or not actuator_class:
                        continue
                    bucket = objective_class_counts.setdefault(objective_key, {})
                    bucket[actuator_class] = int(bucket.get(actuator_class, 0)) + 1
                    candidate_fingerprint = (
                        dict(item.get("best_eval_candidate_fingerprint"))
                        if isinstance(item.get("best_eval_candidate_fingerprint"), Mapping)
                        else {}
                    )
                    eval_context_fingerprint = (
                        dict(item.get("best_eval_context_fingerprint"))
                    if isinstance(item.get("best_eval_context_fingerprint"), Mapping)
                        else {}
                    )
                    bridge_eval_matrix.append(
                        {
                            "objective_bundle_key": objective_key,
                            "objective_term": str(item.get("intended_term", "") or "") or None,
                            "actuator_bundle_key": str(
                                candidate_fingerprint.get("actuator_bundle_key", "") or item.get("intended_bundle_key", "") or ""
                            )
                            or None,
                            "recipe_name": str(candidate_fingerprint.get("recipe_name", "") or "") or None,
                            "operator_recipe_id": str(item.get("operator_recipe_id", "") or "") or None,
                            "actuator_class": actuator_class,
                            "realized_lift_bundle_key": str(item.get("realized_lift_bundle_key", "") or "") or None,
                            "self_delta": round(float(item.get("self_delta", 0.0) or 0.0), 6),
                            "cross_delta": round(float(item.get("cross_delta", 0.0) or 0.0), 6),
                            "alignment_margin": round(float(item.get("alignment_margin", 0.0) or 0.0), 6),
                            "entropy_delta": round(float(item.get("best_eval_entropy_delta", 0.0) or 0.0), 6),
                            "top1_margin_delta": round(float(item.get("best_eval_top1_margin_delta", 0.0) or 0.0), 6),
                            "repeat_delta": int(item.get("best_eval_repeat_flag_delta", 0) or 0),
                            "repetition_score_delta": round(float(item.get("best_eval_repetition_score_delta", 0.0) or 0.0), 6),
                            "required_term_recall_delta": round(float(item.get("best_eval_required_term_recall_delta", 0.0) or 0.0), 6),
                            "required_term_span_progress_delta": round(float(item.get("best_eval_required_term_span_progress_delta", 0.0) or 0.0), 6),
                            "target_mass_delta": round(float(item.get("best_eval_target_mass_delta", 0.0) or 0.0), 6),
                            "target_top20_hit_delta": int(item.get("best_eval_target_top20_hit_delta", 0) or 0),
                            "candidate_fingerprint": candidate_fingerprint,
                            "eval_context_fingerprint": eval_context_fingerprint,
                        }
                    )
            extra_operator_diagnostics = _bridge_eval_extra_operator_diagnostics(
                self,
                replay_bundle_keys=replay_bundle_keys,
                strategy_hints=strategy_hints,
            )
            forced_state["bridge_eval_extra_operator_diagnostics"] = tuple(
                dict(item) for item in extra_operator_diagnostics
            )
            preferred_objective_keys = [
                str(item)
                for item in (
                    strategy_hints.get("selected_bundle_key"),
                    strategy_hints.get("challenger_bundle_key"),
                    strategy_hints.get("base_winner_bundle_key"),
                )
                if item not in (None, "") and str(item)
            ]
            unavailable_summary = _bridge_plan_unavailable_summary(
                {
                    "bundle_keys": replay_bundle_keys,
                    "shadow_bundle_keys": [
                        str(item)
                        for item in forced_state.get("bridge_eval_shadow_bundle_keys", ())
                        if str(item)
                    ],
                    "matrix": bridge_eval_matrix,
                    "exception": None,
                },
                preferred_objective_keys=preferred_objective_keys,
            )
            forced_state["bridge_eval_summary"] = {
                "bundle_keys": replay_bundle_keys,
                "shadow_bundle_keys": [
                    str(item)
                    for item in forced_state.get("bridge_eval_shadow_bundle_keys", ())
                    if str(item)
                ],
                "recipe_names": [
                    str(item.get("recipe_name", "") or "")
                    for item in bridge_eval_recipe_specs
                    if isinstance(item, Mapping) and str(item.get("recipe_name", "") or "")
                ],
                "evaluations_count": (
                    len(replay_summary.get("evaluations", ()))
                    if isinstance(replay_summary, Mapping)
                    and isinstance(replay_summary.get("evaluations"), SequenceABC)
                    and not isinstance(replay_summary.get("evaluations"), (str, bytes, bytearray))
                    else 0
                ),
                "ownership_count": (
                    len(ownership_items)
                    if isinstance(ownership_items, SequenceABC) and not isinstance(ownership_items, (str, bytes, bytearray))
                    else 0
                ),
                "recommendation_count": (
                    len(recommendations)
                    if isinstance(recommendations, SequenceABC) and not isinstance(recommendations, (str, bytes, bytearray))
                    else 0
                ),
                "recommendation_objectives": [
                    str(item.get("objective_bundle_key", "") or "")
                    for item in (recommendations or [])
                    if isinstance(item, Mapping) and str(item.get("objective_bundle_key", "") or "")
                ]
                if isinstance(recommendations, SequenceABC) and not isinstance(recommendations, (str, bytes, bytearray))
                else [],
                "matrix": bridge_eval_matrix,
                "operator_recipe_mode_diagnostics": [
                    dict(item)
                    for item in (mode_diagnostic_items or [])
                    if isinstance(item, Mapping)
                ][:24]
                if isinstance(mode_diagnostic_items, SequenceABC)
                and not isinstance(mode_diagnostic_items, (str, bytes, bytearray))
                else [],
                "objective_class_counts": objective_class_counts,
                "extra_operator_diagnostics": [dict(item) for item in extra_operator_diagnostics],
                "extra_operator_diagnostic_count": len(extra_operator_diagnostics),
                "attention_scale_response_diagnostics": _attention_scale_response_diagnostics(
                    extra_operator_diagnostics
                ),
                "unavailable_reason": unavailable_summary.get("reason"),
                "unavailable_objective_bundle_key": unavailable_summary.get("objective_bundle_key"),
                "unavailable_objective_reasons": dict(unavailable_summary.get("objective_reasons", {})),
                "context_drift": bool(unavailable_summary.get("context_drift", False)),
                "exception": None,
            }
            forced_state["bridge_eval_summary"].update(
                _diagnostic_evidence_ledger(forced_state["bridge_eval_summary"], strategy_hints)
            )
            if isinstance(recommendations, SequenceABC) and not isinstance(recommendations, (str, bytes, bytearray)):
                objective_keys = [
                    str(strategy_hints.get("selected_bundle_key", "") or ""),
                    str(strategy_hints.get("challenger_bundle_key", "") or ""),
                    str(strategy_hints.get("base_winner_bundle_key", "") or ""),
                ]
                chosen_report = None
                for objective_key in objective_keys:
                    if not objective_key:
                        continue
                    for item in recommendations:
                        if not isinstance(item, Mapping):
                            continue
                        if str(item.get("objective_bundle_key", "") or "") == objective_key:
                            chosen_report = dict(item)
                            break
                    if chosen_report is not None:
                        break
                if chosen_report is None and len(recommendations) == 1 and isinstance(recommendations[0], Mapping):
                    chosen_report = dict(recommendations[0])
                forced_state["bridge_plan_report"] = chosen_report
        except Exception:
            unavailable_summary = _bridge_plan_unavailable_summary(
                {
                    "bundle_keys": [],
                    "shadow_bundle_keys": [],
                    "matrix": [],
                    "exception": "replay_operator_recipe_matrix_failed",
                }
            )
            forced_state["bridge_plan_report"] = None
            forced_state["bridge_eval_summary"] = {
                "bundle_keys": [],
                "shadow_bundle_keys": [],
                "recipe_names": [
                    str(item.get("recipe_name", "") or "")
                    for item in _focused_bridge_eval_recipe_specs()
                    if isinstance(item, Mapping) and str(item.get("recipe_name", "") or "")
                ],
                "matrix": [],
                "operator_recipe_mode_diagnostics": [],
                "attention_scale_response_diagnostics": [],
                "evaluations_count": 0,
                "ownership_count": 0,
                "recommendation_count": 0,
                "recommendation_objectives": [],
                "objective_class_counts": {},
                "extra_operator_diagnostics": [],
                "extra_operator_diagnostic_count": 0,
                "unavailable_reason": unavailable_summary.get("reason"),
                "unavailable_objective_bundle_key": unavailable_summary.get("objective_bundle_key"),
                "unavailable_objective_reasons": dict(unavailable_summary.get("objective_reasons", {})),
                "context_drift": bool(unavailable_summary.get("context_drift", False)),
                "exception": "replay_operator_recipe_matrix_failed",
            }
            forced_state["bridge_eval_summary"].update(
                _diagnostic_evidence_ledger(forced_state["bridge_eval_summary"], strategy_hints)
            )
            forced_state["bridge_eval_extra_operator_diagnostics"] = ()
            self._operator_bridge_plan_table = {}
            forced_state["bridge_eval_last_replay_summary"] = None
        finally:
            forced_state["bridge_eval_active"] = False
        replay_snapshot = forced_state.get("bridge_eval_packet_snapshot")
        if isinstance(replay_snapshot, Mapping):
            diff_keys = [
                key
                for key in sorted(set(frontier_snapshot) | set(replay_snapshot))
                if frontier_snapshot.get(key) != replay_snapshot.get(key)
            ]
            forced_state["bridge_plan_packet_invariant_ok"] = not diff_keys
            forced_state["bridge_plan_packet_invariant_diff"] = tuple(diff_keys)
        else:
            forced_state["bridge_plan_packet_invariant_ok"] = None
            forced_state["bridge_plan_packet_invariant_diff"] = ("replay_snapshot_missing",)

    def _compact_swap_result(result: Mapping[str, Any], *, source_label: str) -> dict[str, Any]:
        return {
            "source_label": source_label,
            "status": str(result.get("status", "") or ""),
            "label": str(result.get("label", "") or ""),
            "intended_bundle_key": result.get("intended_bundle_key"),
            "intended_term": result.get("intended_term"),
            "actual_delta_class": result.get("actual_delta_class"),
            "realized_lift_bundle_key": result.get("realized_lift_bundle_key"),
            "target_mass_delta": result.get("target_mass_delta"),
            "target_top20_hit_delta": result.get("target_top20_hit_delta"),
            "required_term_recall_delta": result.get("required_term_recall_delta"),
            "required_term_span_progress_delta": result.get("required_term_span_progress_delta"),
            "candidate_fingerprint": dict(result.get("candidate_fingerprint", {}))
            if isinstance(result.get("candidate_fingerprint"), Mapping)
            else {},
            "eval_context_fingerprint": dict(result.get("eval_context_fingerprint", {}))
            if isinstance(result.get("eval_context_fingerprint"), Mapping)
            else {},
        }

    def _candidate_swap_comparison(
        self: HookedTransformerWorkerRuntime,
        *,
        pre_summary: Mapping[str, Any],
        post_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        def _evaluations_by_label(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
            raw_evaluations = summary.get("evaluations")
            if not isinstance(raw_evaluations, SequenceABC) or isinstance(raw_evaluations, (str, bytes, bytearray)):
                return {}
            items: dict[str, dict[str, Any]] = {}
            for item in raw_evaluations:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("label", "") or "")
                edits = item.get("candidate_edits")
                if not label or not isinstance(edits, SequenceABC) or isinstance(edits, (str, bytes, bytearray)) or not edits:
                    continue
                if str(item.get("status", "") or "") != "ok":
                    continue
                items.setdefault(label, dict(item))
            return items

        pre_items = _evaluations_by_label(pre_summary)
        post_items = _evaluations_by_label(post_summary)
        preferred_labels = [
            label
            for label in sorted(set(pre_items) & set(post_items))
            if any(
                marker in label
                for marker in (
                    "baseline_span_mean:pair:",
                    "term_token:pair:",
                    "term_fused:pair:",
                    "term_centered_pm1:pair:",
                    "term_centered_pm1_minus_stealer_l025:pair:",
                    "term_centered_pm1_orthogonal_stealer:pair:",
                )
            )
        ][:6]
        if not preferred_labels:
            preferred_labels = sorted(set(pre_items) & set(post_items))[:6]

        saved_segments = [(segment.kind, tuple(segment.token_ids)) for segment in self._segments]
        saved_steps = int(self._steps)
        saved_last_packet = self._last_packet
        segment_cls = type(self._segments[0]) if self._segments else None

        def _restore(snapshot: Sequence[tuple[str, Sequence[int]]], steps: int) -> None:
            if segment_cls is None:
                return
            self._segments = [segment_cls(kind=str(kind), token_ids=[int(token) for token in token_ids]) for kind, token_ids in snapshot]
            self._steps = int(steps)
            self._last_packet = None

        def _run_item(item: Mapping[str, Any], *, target_context: str, source_label: str) -> dict[str, Any]:
            edits = [dict(edit) for edit in item.get("candidate_edits", ()) if isinstance(edit, Mapping)]
            result = self.replay_candidate_edits_actual_delta(
                edits,
                max_new_tokens=3,
                top_k=8,
                max_edits_per_step_override=2,
                label=f"candidate_swap:{source_label}:to_{target_context}:{item.get('label')}",
                ownership_terms=item.get("focus_terms") if isinstance(item.get("focus_terms"), SequenceABC) else None,
                intended_bundle_key=str(item.get("intended_bundle_key", "") or "") or None,
                intended_term=str(item.get("intended_term", "") or "") or None,
                contrast_partner_bundle_key=str(item.get("contrast_partner_bundle_key", "") or "") or None,
                contrast_partner_term=str(item.get("contrast_partner_term", "") or "") or None,
            )
            return _compact_swap_result(result, source_label=source_label)

        rows: list[dict[str, Any]] = []
        try:
            pre_snapshot = [
                (str(kind), tuple(int(token) for token in token_ids))
                for kind, token_ids in forced_state.get("pre_step_segments", ())
                if str(kind)
            ]
            pre_steps = int(forced_state.get("pre_step_steps", 0) or 0)
            for label in preferred_labels:
                pre_item = pre_items[label]
                post_item = post_items[label]
                _restore(pre_snapshot, pre_steps)
                post_in_pre = _run_item(post_item, target_context="pre", source_label="post")
                _restore(saved_segments, saved_steps)
                pre_in_post = _run_item(pre_item, target_context="post", source_label="pre")
                rows.append(
                    {
                        "label": label,
                        "pre_original_class": pre_item.get("actual_delta_class"),
                        "post_original_class": post_item.get("actual_delta_class"),
                        "post_candidate_in_pre_context": post_in_pre,
                        "pre_candidate_in_post_context": pre_in_post,
                    }
                )
        finally:
            _restore(saved_segments, saved_steps)
            self._last_packet = saved_last_packet

        return {
            "matched_label_count": len(preferred_labels),
            "labels": preferred_labels,
            "rows": rows,
        }

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
            frontier_snapshot = _bridge_frontier_snapshot(strategy_hints)
            if bool(forced_state.get("bridge_eval_active", False)):
                shadow_candidates = [
                    dict(item)
                    for item in forced_state.get("bridge_eval_shadow_candidates", ())
                    if isinstance(item, Mapping)
                ]
                if shadow_candidates:
                    merged_candidates = list(strategy_hints.get("kv_candidate_edits", []))
                    seen_bundle_keys = set(_candidate_bundle_keys(merged_candidates))
                    for candidate in shadow_candidates:
                        merged_candidates.append(dict(candidate))
                        bundle_key = str(candidate.get("bundle_key", "") or "")
                        if bundle_key:
                            seen_bundle_keys.add(bundle_key)
                    strategy_hints["kv_candidate_edits"] = merged_candidates
                    strategy_hints["bridge_eval_shadow_bundle_keys"] = [
                        str(item) for item in forced_state.get("bridge_eval_shadow_bundle_keys", ()) if str(item)
                    ]
                forced_state["bridge_eval_packet_snapshot"] = dict(frontier_snapshot)
                packet["strategy_hints"] = strategy_hints
                packet["control_phase_hint"] = "readout_escape"
                return packet
            if normalized_packet_mode == "fixed_candidate":
                cached = forced_state.get("cached_strategy_hints")
                if isinstance(cached, Mapping):
                    strategy_hints.update(dict(cached))
                    packet["strategy_hints"] = strategy_hints
                    packet["control_phase_hint"] = "readout_escape"
                    return packet
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
            if forced_state.get("bridge_eval_summary") is None:
                _run_bridge_eval_from_packet(self, packet=packet, strategy_hints=strategy_hints)
                if isinstance(forced_state.get("bridge_eval_last_replay_summary"), Mapping):
                    forced_state["bridge_eval_pre_replay_summary"] = dict(forced_state["bridge_eval_last_replay_summary"])
            elif (
                forced_state.get("bridge_eval_poststep_comparison") is None
                and int(packet.get("step", 0) or 0) > int(forced_state.get("bridge_eval_locked_step", 0) or 0)
            ):
                saved_bridge_state = {
                    "bridge_plan_report": forced_state.get("bridge_plan_report"),
                    "bridge_plan_packet_invariant_ok": forced_state.get("bridge_plan_packet_invariant_ok"),
                    "bridge_plan_packet_invariant_diff": forced_state.get("bridge_plan_packet_invariant_diff"),
                    "bridge_eval_summary": forced_state.get("bridge_eval_summary"),
                    "bridge_eval_shadow_candidates": forced_state.get("bridge_eval_shadow_candidates"),
                    "bridge_eval_shadow_bundle_keys": forced_state.get("bridge_eval_shadow_bundle_keys"),
                    "bridge_eval_active": forced_state.get("bridge_eval_active"),
                    "bridge_eval_packet_snapshot": forced_state.get("bridge_eval_packet_snapshot"),
                    "bridge_eval_locked_step": forced_state.get("bridge_eval_locked_step"),
                }
                _run_bridge_eval_from_packet(self, packet=packet, strategy_hints=strategy_hints)
                comparison_summary = forced_state.get("bridge_eval_summary")
                if isinstance(comparison_summary, Mapping):
                    forced_state["bridge_eval_poststep_comparison"] = {
                        "locked_step": int(forced_state.get("bridge_eval_locked_step", 0) or 0),
                        "context_drift": bool(comparison_summary.get("context_drift", False)),
                        "objective_class_counts": dict(comparison_summary.get("objective_class_counts", {}))
                        if isinstance(comparison_summary.get("objective_class_counts"), Mapping)
                        else {},
                        "matrix": [
                            dict(item)
                            for item in comparison_summary.get("matrix", ())
                            if isinstance(item, Mapping)
                        ][:8]
                        if isinstance(comparison_summary.get("matrix"), SequenceABC)
                        and not isinstance(comparison_summary.get("matrix"), (str, bytes, bytearray))
                        else [],
                    }
                    pre_replay_summary = forced_state.get("bridge_eval_pre_replay_summary")
                    post_replay_summary = forced_state.get("bridge_eval_last_replay_summary")
                    if isinstance(pre_replay_summary, Mapping) and isinstance(post_replay_summary, Mapping):
                        forced_state["bridge_eval_candidate_swap_comparison"] = _candidate_swap_comparison(
                            self,
                            pre_summary=pre_replay_summary,
                            post_summary=post_replay_summary,
                        )
                forced_state.update(saved_bridge_state)
            bridge_plan_report = forced_state.get("bridge_plan_report")
            if isinstance(bridge_plan_report, Mapping):
                strategy_hints["bridge_plan_available"] = True
                strategy_hints["bridge_plan_required"] = bool(bridge_plan_report.get("bridge_required", True))
                strategy_hints["bridge_plan_report"] = dict(bridge_plan_report)
                for source_key, target_key in (
                    ("objective_bundle_key", "bridge_plan_objective_bundle_key"),
                    ("objective_term", "bridge_plan_objective_term"),
                    ("actuator_bundle_key", "bridge_plan_actuator_bundle_key"),
                    ("actuator_term", "bridge_plan_actuator_term"),
                    ("operator_recipe_id", "bridge_plan_recipe_id"),
                    ("actuator_class", "bridge_plan_actuator_class"),
                    ("bridge_plan_reason", "bridge_plan_reason"),
                ):
                    if bridge_plan_report.get(source_key) not in (None, ""):
                        strategy_hints[target_key] = bridge_plan_report.get(source_key)
            if forced_state.get("bridge_plan_packet_invariant_ok") is not None:
                strategy_hints["bridge_plan_packet_invariant_ok"] = bool(forced_state.get("bridge_plan_packet_invariant_ok"))
            diff_keys = forced_state.get("bridge_plan_packet_invariant_diff")
            if isinstance(diff_keys, SequenceABC) and not isinstance(diff_keys, (str, bytes, bytearray)) and diff_keys:
                strategy_hints["bridge_plan_packet_invariant_diff"] = [str(item) for item in diff_keys if str(item)]
            bridge_eval_summary = forced_state.get("bridge_eval_summary")
            if isinstance(bridge_eval_summary, Mapping):
                strategy_hints["bridge_plan_recommendation_count"] = int(bridge_eval_summary.get("recommendation_count", 0) or 0)
                strategy_hints["bridge_eval_evaluations_count"] = int(bridge_eval_summary.get("evaluations_count", 0) or 0)
                strategy_hints["bridge_eval_ownership_count"] = int(bridge_eval_summary.get("ownership_count", 0) or 0)
                replay_bundle_keys = bridge_eval_summary.get("bundle_keys")
                if isinstance(replay_bundle_keys, SequenceABC) and not isinstance(replay_bundle_keys, (str, bytes, bytearray)):
                    strategy_hints["bridge_eval_bundle_keys"] = [str(item) for item in replay_bundle_keys if str(item)]
                recommendation_objectives = bridge_eval_summary.get("recommendation_objectives")
                shadow_bundle_keys = bridge_eval_summary.get("shadow_bundle_keys")
                if isinstance(shadow_bundle_keys, SequenceABC) and not isinstance(shadow_bundle_keys, (str, bytes, bytearray)):
                    strategy_hints["bridge_eval_shadow_bundle_keys"] = [str(item) for item in shadow_bundle_keys if str(item)]
                recipe_names = bridge_eval_summary.get("recipe_names")
                if isinstance(recipe_names, SequenceABC) and not isinstance(recipe_names, (str, bytes, bytearray)):
                    strategy_hints["bridge_eval_recipe_names"] = [str(item) for item in recipe_names if str(item)]
                mode_diagnostics = bridge_eval_summary.get("operator_recipe_mode_diagnostics")
                if isinstance(mode_diagnostics, SequenceABC) and not isinstance(mode_diagnostics, (str, bytes, bytearray)):
                    strategy_hints["bridge_eval_operator_recipe_mode_diagnostics"] = [
                        dict(item) for item in mode_diagnostics[:24] if isinstance(item, Mapping)
                    ]
                attention_response = bridge_eval_summary.get("attention_scale_response_diagnostics")
                if isinstance(attention_response, SequenceABC) and not isinstance(attention_response, (str, bytes, bytearray)):
                    strategy_hints["bridge_eval_attention_scale_response_diagnostics"] = [
                        dict(item) for item in attention_response[:12] if isinstance(item, Mapping)
                    ]
                if isinstance(recommendation_objectives, SequenceABC) and not isinstance(recommendation_objectives, (str, bytes, bytearray)):
                    strategy_hints["bridge_plan_recommendation_objectives"] = [
                        str(item) for item in recommendation_objectives if str(item)
                    ]
                objective_class_counts = bridge_eval_summary.get("objective_class_counts")
                if isinstance(objective_class_counts, Mapping):
                    strategy_hints["bridge_eval_objective_class_counts"] = {
                        str(key): {
                            str(inner_key): int(inner_value)
                            for inner_key, inner_value in dict(value).items()
                            if str(inner_key)
                        }
                        for key, value in objective_class_counts.items()
                        if isinstance(value, Mapping) and str(key)
                    }
                strategy_hints["bridge_eval_extra_operator_diagnostic_count"] = int(
                    bridge_eval_summary.get("extra_operator_diagnostic_count", 0) or 0
                )
                strategy_hints["diagnostic_evidence_ledger_count"] = int(
                    bridge_eval_summary.get("diagnostic_evidence_ledger_count", 0) or 0
                )
                diagnostic_ledger = bridge_eval_summary.get("diagnostic_evidence_ledger")
                if isinstance(diagnostic_ledger, SequenceABC) and not isinstance(diagnostic_ledger, (str, bytes, bytearray)):
                    strategy_hints["diagnostic_evidence_ledger"] = [
                        dict(item) for item in diagnostic_ledger[:48] if isinstance(item, Mapping)
                    ]
                bundle_diagnostic_status = bridge_eval_summary.get("bundle_diagnostic_status")
                if isinstance(bundle_diagnostic_status, Mapping):
                    strategy_hints["bundle_diagnostic_status"] = {
                        str(key): dict(value)
                        for key, value in bundle_diagnostic_status.items()
                        if str(key) and isinstance(value, Mapping)
                    }
                for source_key in (
                    "diagnostic_frontier_bundle_key",
                    "diagnostic_frontier_next_evidence",
                    "diagnostic_frontier_request",
                    "diagnostic_frontier_reason_text",
                ):
                    if bridge_eval_summary.get(source_key) not in (None, ""):
                        strategy_hints[source_key] = bridge_eval_summary.get(source_key)
                diagnostic_frontier_status = bridge_eval_summary.get("diagnostic_frontier_status")
                if isinstance(diagnostic_frontier_status, Mapping):
                    strategy_hints["diagnostic_frontier_status"] = dict(diagnostic_frontier_status)
                unavailable_reason = bridge_eval_summary.get("unavailable_reason")
                if unavailable_reason not in (None, ""):
                    strategy_hints["bridge_plan_unavailable_reason"] = str(unavailable_reason)
                unavailable_objective_bundle_key = bridge_eval_summary.get("unavailable_objective_bundle_key")
                if unavailable_objective_bundle_key not in (None, ""):
                    strategy_hints["bridge_plan_unavailable_objective_bundle_key"] = str(unavailable_objective_bundle_key)
                unavailable_objective_reasons = bridge_eval_summary.get("unavailable_objective_reasons")
                if isinstance(unavailable_objective_reasons, Mapping):
                    strategy_hints["bridge_plan_unavailable_objective_reasons"] = {
                        str(key): str(value)
                        for key, value in unavailable_objective_reasons.items()
                        if str(key) and str(value)
                    }
                if bridge_eval_summary.get("context_drift") is not None:
                    strategy_hints["bridge_eval_context_drift"] = bool(bridge_eval_summary.get("context_drift"))
                if forced_state.get("bridge_eval_locked_step") is not None:
                    strategy_hints["bridge_eval_locked_step"] = int(forced_state.get("bridge_eval_locked_step", 0) or 0)
                exception_text = str(bridge_eval_summary.get("exception", "") or "")
                if exception_text:
                    strategy_hints["bridge_eval_exception"] = exception_text
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
        controller_client = _FrontierReplayControllerClient(replay_mode=normalized_controller_replay_mode)
        ctx = StepContext(
            packet={},
            runtime_state=worker.runtime_state,
            traces={},
            stats={},
            adapter=worker.adapter,
        )
        episode = run_episode(
            env,
            worker,
            controller_client,
            ctx,
            logger=logger,
            max_controller_diagnostic_rethink_rounds=max_controller_diagnostic_rethink_rounds
            if normalized_controller_replay_mode == "diagnostic_request"
            else 0,
        )
    finally:
        worker.reset = original_reset
        worker._current_answer_readout_canary = original_answer_readout_canary
        worker.build_controller_packet = original_build_controller_packet

    selection_events = [event for event in memory_logger.events if event.get("event") == "controller_selection"]
    first_selection_event = dict(selection_events[0]) if selection_events else None
    diagnostic_request_events = [
        dict(event) for event in memory_logger.events if event.get("event") == "controller_diagnostic_request"
    ]
    diagnostic_result_events = [
        dict(event) for event in memory_logger.events if event.get("event") == "controller_diagnostic_result"
    ]
    controller_diagnostic_request_names = tuple(
        str(event.get("diagnostic", event.get("diagnostic_request", "")) or "")
        for event in diagnostic_request_events
        if str(event.get("diagnostic", event.get("diagnostic_request", "")) or "")
    )
    latest_diagnostic_results = tuple(
        {
            key: value
            for key, value in event.items()
            if key not in {"event", "step"}
        }
        for event in diagnostic_result_events[-2:]
    )
    recent_diagnostic_results = tuple(
        {
            key: value
            for key, value in event.items()
            if key not in {"event", "step"}
        }
        for event in diagnostic_result_events[-6:]
    )
    controller_step_views = _controller_step_views(memory_logger.events)
    bridge_visible_step_views = tuple(
        dict(item)
        for item in controller_step_views
        if bool(item.get("bridge_visible"))
    )
    result = ReadoutEscapeReplayHarnessResult(
        seed=seed,
        task_id=env.task_id,
        worker_model_name=worker_model_name,
        replay_mode=f"{normalized_packet_mode}_{normalized_controller_replay_mode}",
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
        readout_analyzer_name=None
        if first_selection_event is None
        else first_selection_event.get("readout_analyzer_name"),
        readout_analyzer_feature_backend=None
        if first_selection_event is None
        else first_selection_event.get("readout_analyzer_feature_backend"),
        readout_analyzer_sae_status=None
        if first_selection_event is None
        else first_selection_event.get("readout_analyzer_sae_status"),
        readout_analyzer_sae_feature_hint_count=int(
            0
            if first_selection_event is None
            else first_selection_event.get("readout_analyzer_sae_feature_hint_count", 0)
        ),
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
        controller_objective_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("controller_objective_bundle_key"),
        controller_step_actuator_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("controller_step_actuator_bundle_key"),
        controller_plan_mode=None
        if first_selection_event is None
        else first_selection_event.get("controller_plan_mode"),
        controller_why_not_apply=None
        if first_selection_event is None
        else first_selection_event.get("controller_why_not_apply"),
        controller_shadow_proposal_count=int(
            0 if first_selection_event is None else first_selection_event.get("controller_shadow_proposal_count", 0)
        ),
        bridge_plan_objective_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("bridge_plan_objective_bundle_key"),
        bridge_plan_actuator_bundle_key=None
        if first_selection_event is None
        else first_selection_event.get("bridge_plan_actuator_bundle_key"),
        bridge_plan_reason=None
        if first_selection_event is None
        else first_selection_event.get("bridge_plan_reason"),
        bridge_plan_unavailable_reason=(
            None
            if first_selection_event is None
            else first_selection_event.get("bridge_plan_unavailable_reason")
        ),
        bridge_plan_unavailable_objective_bundle_key=(
            None
            if first_selection_event is None
            else first_selection_event.get("bridge_plan_unavailable_objective_bundle_key")
        ),
        bridge_plan_unavailable_objective_reasons={
            str(key): str(value)
            for key, value in (
                (
                    first_selection_event.get("bridge_plan_unavailable_objective_reasons", {}).items()
                    if first_selection_event is not None
                    and isinstance(first_selection_event.get("bridge_plan_unavailable_objective_reasons"), Mapping)
                    else ()
                )
            )
            if str(key) and str(value)
        },
        bridge_eval_context_drift=bool(
            (
                forced_state.get("bridge_eval_summary", {}).get("context_drift", False)
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else False
            )
        ),
        bridge_eval_locked_step=(
            None
            if forced_state.get("bridge_eval_locked_step") is None
            else int(forced_state.get("bridge_eval_locked_step", 0) or 0)
        ),
        bridge_plan_used=bool(
            False if first_selection_event is None else first_selection_event.get("bridge_plan_used", False)
        ),
        bridge_plan_packet_invariant_ok=(
            None
            if forced_state.get("bridge_plan_packet_invariant_ok") is None
            else bool(forced_state.get("bridge_plan_packet_invariant_ok"))
        ),
        bridge_plan_packet_invariant_diff=tuple(
            str(item)
            for item in forced_state.get("bridge_plan_packet_invariant_diff", ())
            if str(item)
        ),
        bridge_plan_recommendation_count=int(
            (
                forced_state.get("bridge_eval_summary", {}).get("recommendation_count", 0)
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else 0
            )
            or 0
        ),
        bridge_plan_recommendation_objectives=tuple(
            str(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("recommendation_objectives", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if str(item)
        ),
        bridge_eval_shadow_bundle_keys=tuple(
            str(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("shadow_bundle_keys", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if str(item)
        ),
        bridge_eval_recipe_names=tuple(
            str(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("recipe_names", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if str(item)
        ),
        bridge_eval_matrix=tuple(
            dict(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("matrix", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        ),
        bridge_eval_operator_recipe_mode_diagnostics=tuple(
            dict(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("operator_recipe_mode_diagnostics", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        ),
        bridge_eval_attention_scale_response_diagnostics=tuple(
            dict(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("attention_scale_response_diagnostics", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        ),
        bridge_eval_objective_class_counts={
            str(key): {
                str(inner_key): int(inner_value)
                for inner_key, inner_value in dict(value).items()
                if str(inner_key)
            }
            for key, value in (
                forced_state.get("bridge_eval_summary", {}).get("objective_class_counts", {}).items()
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                and isinstance(forced_state.get("bridge_eval_summary", {}).get("objective_class_counts"), Mapping)
                else ()
            )
            if isinstance(value, Mapping) and str(key)
        },
        bridge_eval_evaluations_count=int(
            (
                forced_state.get("bridge_eval_summary", {}).get("evaluations_count", 0)
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else 0
            )
            or 0
        ),
        bridge_eval_ownership_count=int(
            (
                forced_state.get("bridge_eval_summary", {}).get("ownership_count", 0)
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else 0
            )
            or 0
        ),
        bridge_eval_exception=(
            None
            if not isinstance(forced_state.get("bridge_eval_summary"), Mapping)
            else (
                None
                if not str(forced_state.get("bridge_eval_summary", {}).get("exception", "") or "")
                else str(forced_state.get("bridge_eval_summary", {}).get("exception", "") or "")
            )
        ),
        diagnostic_evidence_ledger=tuple(
            dict(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("diagnostic_evidence_ledger", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        ),
        diagnostic_evidence_ledger_count=int(
            (
                forced_state.get("bridge_eval_summary", {}).get("diagnostic_evidence_ledger_count", 0)
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else 0
            )
            or 0
        ),
        bundle_diagnostic_status={
            str(key): dict(value)
            for key, value in (
                forced_state.get("bridge_eval_summary", {}).get("bundle_diagnostic_status", {}).items()
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                and isinstance(forced_state.get("bridge_eval_summary", {}).get("bundle_diagnostic_status"), Mapping)
                else ()
            )
            if str(key) and isinstance(value, Mapping)
        },
        diagnostic_frontier_bundle_key=(
            None
            if not isinstance(forced_state.get("bridge_eval_summary"), Mapping)
            or forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_bundle_key") in (None, "")
            else str(forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_bundle_key"))
        ),
        diagnostic_frontier_next_evidence=(
            None
            if not isinstance(forced_state.get("bridge_eval_summary"), Mapping)
            or forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_next_evidence") in (None, "")
            else str(forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_next_evidence"))
        ),
        diagnostic_frontier_request=(
            None
            if not isinstance(forced_state.get("bridge_eval_summary"), Mapping)
            or forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_request") in (None, "")
            else str(forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_request"))
        ),
        diagnostic_frontier_reason_text=(
            None
            if not isinstance(forced_state.get("bridge_eval_summary"), Mapping)
            or forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_reason_text") in (None, "")
            else str(forced_state.get("bridge_eval_summary", {}).get("diagnostic_frontier_reason_text"))
        ),
        diagnostic_request_event_count=len(diagnostic_request_events),
        diagnostic_result_event_count=len(diagnostic_result_events),
        controller_diagnostic_request_names=controller_diagnostic_request_names,
        latest_diagnostic_results=latest_diagnostic_results,
        recent_diagnostic_results=recent_diagnostic_results,
        bridge_eval_extra_operator_diagnostics=tuple(
            dict(item)
            for item in (
                forced_state.get("bridge_eval_summary", {}).get("extra_operator_diagnostics", ())
                if isinstance(forced_state.get("bridge_eval_summary"), Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        ),
        bridge_eval_poststep_comparison=(
            dict(forced_state.get("bridge_eval_poststep_comparison"))
            if isinstance(forced_state.get("bridge_eval_poststep_comparison"), Mapping)
            else None
        ),
        bridge_eval_candidate_swap_comparison=(
            dict(forced_state.get("bridge_eval_candidate_swap_comparison"))
            if isinstance(forced_state.get("bridge_eval_candidate_swap_comparison"), Mapping)
            else None
        ),
        controller_step_views=controller_step_views,
        bridge_visible_step_views=bridge_visible_step_views,
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
