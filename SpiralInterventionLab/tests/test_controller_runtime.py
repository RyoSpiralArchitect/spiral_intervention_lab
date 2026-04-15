import unittest
from importlib.util import find_spec
from unittest.mock import patch

import torch

from SpiralInterventionLab.runtime.adapter import BoundSurface, HookedTransformerAdapter, ModelAdapter
from SpiralInterventionLab.runtime.baselines import activation_only_policy, run_b0, run_b1, run_c1, run_minimal_baseline_suite
from SpiralInterventionLab.runtime.codecs import CharacterCodec
from SpiralInterventionLab.runtime.compiler import StepContext, compile_command
from SpiralInterventionLab.runtime.edit_budget import prepare_direction
from SpiralInterventionLab.runtime.effects import build_edit_effect, summarize_effects
from SpiralInterventionLab.runtime.loop import InMemoryStructuredLogger, run_episode
from SpiralInterventionLab.runtime.overlays import OverlayHandle
from SpiralInterventionLab.runtime.policy import GlobalBudget, HarnessPolicy, PolicyViolation, validate_command_against_packet
from SpiralInterventionLab.runtime.rank1_bridge import HybridRank1VectorBridge, Rank1Geometry
from SpiralInterventionLab.runtime.sidecar import (
    ReadoutSidecarCapture,
    ReadoutSidecarSiteCapture,
    build_heuristic_readout_sidecar_analyzer,
)
from SpiralInterventionLab.runtime.tlens_runtime import HookedTransformerRuntimeState
from SpiralInterventionLab.runtime.trace_recorder import StepAlignedTrace
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime, _TargetTokenSequence
from SpiralInterventionLab.runtime.schema import (
    ControllerCommand,
    ControllerObservationPacket,
    SurfaceTargetRef,
    parse_controller_command,
    parse_observation_packet,
)

HAS_TRANSFORMER_LENS = bool(find_spec("transformer_lens"))
if HAS_TRANSFORMER_LENS:
    from transformer_lens import HookedTransformer, HookedTransformerConfig


def _make_packet(*, rollbackable_ids=None):
    rollbackable_ids = rollbackable_ids or []
    return {
        "version": "0.1",
        "run_id": "run_0001",
        "episode_id": "ep_0001",
        "worker_id": "os_0",
        "step": 0,
        "horizon": {
            "generated_tokens": 1,
            "max_generated_tokens": 16,
            "done": False,
        },
        "task_view": {
            "mode": "redacted",
            "task_id": "toy_task",
            "prompt_hash": "sha256:test",
            "goal_hint": "toy task",
            "constraints": ["stay concise"],
        },
        "worker_view": {
            "generated_tail": "looping...",
            "status": "looping",
        },
        "telemetry": {
            "entropy": 2.1,
            "top1_margin": 0.05,
            "repetition_score": 0.4,
            "repeat_flag": True,
            "no_progress_steps": 2,
        },
        "surface_catalog": [
            {
                "surface_id": "s_resid_l11_last",
                "target": {
                    "kind": "activation",
                    "worker": "os_0",
                    "site": "resid_pre",
                    "layer": 11,
                    "token": {"mode": "last"},
                },
                "allow_ops": ["resid_add"],
                "caps": {
                    "max_alpha": 0.2,
                    "max_ttl_steps": 3,
                    "norm_clip": 1.5,
                    "revertible_only": True,
                },
            },
            {
                "surface_id": "s_weight_l7_mlp",
                "target": {
                    "kind": "weight",
                    "worker": "os_0",
                    "module": "mlp_out",
                    "layer": 7,
                },
                "allow_ops": ["rank1_patch"],
                "caps": {
                    "max_alpha": 0.08,
                    "max_ttl_steps": 4,
                    "rank_cap": 1,
                    "revertible_only": True,
                },
            },
            {
                "surface_id": "s_weight_l10_mlp",
                "target": {
                    "kind": "weight",
                    "worker": "os_0",
                    "module": "mlp_out",
                    "layer": 10,
                },
                "allow_ops": ["rank1_patch"],
                "caps": {
                    "max_alpha": 0.08,
                    "max_ttl_steps": 4,
                    "rank_cap": 1,
                    "revertible_only": True,
                },
            },
            {
                "surface_id": "s_weight_l11_mlp",
                "target": {
                    "kind": "weight",
                    "worker": "os_0",
                    "module": "mlp_out",
                    "layer": 11,
                },
                "allow_ops": ["rank1_patch"],
                "caps": {
                    "max_alpha": 0.08,
                    "max_ttl_steps": 4,
                    "rank_cap": 1,
                    "revertible_only": True,
                },
            },
        ],
        "probe_frames": [
            {
                "surface_id": "s_resid_l11_last",
                "stats": {
                    "norm": 18.3,
                    "delta_prev": 1.9,
                    "anomaly_score": 0.61,
                    "cosine_to_best_success": 0.34,
                },
            }
        ],
        "trace_bank": [
            {
                "trace_id": "best_success",
                "origin": "best_success",
                "compatible": True,
                "similarity_hint": 0.71,
                "tags": ["successful"],
            },
            {
                "trace_id": "paired_baseline",
                "origin": "paired_baseline",
                "compatible": True,
                "similarity_hint": 0.88,
                "tags": ["same_seed"],
            },
        ],
        "active_edits": [],
        "recent_effects": [],
        "budget": {
            "edits_left_this_step": 1,
            "edits_left_this_run": 4,
            "alpha_left_total": 0.5,
            "edit_cost_left_total": 0.5,
            "loop_rescue_edits_left_this_run": 0,
            "loop_rescue_alpha_left_total": 0.0,
            "loop_rescue_edit_cost_left_total": 0.0,
            "active_patch_slots_left": 2,
            "rollbackable_ids": rollbackable_ids,
        },
        "task_feedback": {
            "done": False,
            "progress_label": "stalled",
        },
    }


def _resid_command():
    return {
        "version": "0.1",
        "decision": "apply",
        "meta": {"hypothesis": "rescue", "confidence": 0.6},
        "edits": [
            {
                "id": "e_rescue",
                "target": {"surface_id": "s_resid_l11_last"},
                "source": {
                    "dtype": "vector",
                    "expr": {
                        "ref": {
                            "scope": "runtime",
                            "worker": "os_0",
                            "tensor": "hidden",
                            "layer": 11,
                            "token": {"mode": "last"},
                        }
                    },
                },
                "op": {"kind": "resid_add", "alpha": 0.2},
                "budget": {"ttl_steps": 2, "norm_clip": 1.5, "revertible": True},
                "meta": {"expected_effect": "break_loop"},
            }
        ],
    }


def _loop_rescue_command(*, alpha: float = 0.06, step_size: float = 0.06, edit_id: str = "e_loop_rescue"):
    return {
        "version": "0.1",
        "decision": "apply",
        "meta": {"hypothesis": "small_loop_rescue", "confidence": 0.6},
        "edits": [
            {
                "id": edit_id,
                "target": {"surface_id": "s_resid_l11_last"},
                "source": {
                    "dtype": "vector",
                    "expr": {
                        "ref": {
                            "scope": "runtime",
                            "worker": "os_0",
                            "tensor": "hidden",
                            "layer": 11,
                            "token": {"mode": "last"},
                        }
                    },
                },
                "op": {"kind": "resid_add", "alpha": alpha},
                "budget": {"ttl_steps": 1, "norm_clip": 1.5, "step_size": step_size, "revertible": True},
                "meta": {"expected_effect": "break_loop"},
            }
        ],
    }


def _resid_command_with_memory():
    command = _resid_command()
    command["meta"] = dict(command["meta"]) | {
        "controller_memory": {
            "hypothesis": "small_rescue",
            "micro_rationale": "test a small reversible rescue first",
            "focus_term": "budget",
            "expected_effect": "break_loop",
            "observed_outcome": "unknown",
            "why_failed_or_helped": "first attempt; no effect observed yet",
            "surface_family_key": "resid_add_s_resid_l11_last",
            "transfer_confidence": 0.35,
            "same_family_escalation_risk": 0.4,
            "finish_budget_reserved": 1,
            "evidence_bullets": [
                "helpful_on=budget@l11",
                "reserve last edit until transfer is proven",
            ],
            "next_change": "wait for effect before stacking",
            "next_trigger": "loop_relief_without_coverage",
            "next_action": "request_observer_check",
            "stop_condition": "if no progress improves, stop",
            "confidence": 0.55,
        }
    }
    return command


def _noop_command_with_observer_request():
    return {
        "version": "0.1",
        "decision": "noop",
        "meta": {
            "hypothesis": "loop_relief_but_coverage_zero",
            "micro_rationale": "Loop eased, but coverage is still zero.",
            "next_trigger": "observer_flat_after_loop_relief",
            "next_action": "apply",
            "observer_check_request": {
                "kind": "semantic_progress",
                "reason": "fresh semantic read after loop relief",
            },
        },
    }


def _noop_command_with_tool_requests():
    return {
        "version": "0.1",
        "decision": "noop",
        "meta": {
            "hypothesis": "need_local_tools_before_next_apply",
            "micro_rationale": "Gather local evidence before the next edit.",
            "tool_requests": [
                {"tool": "tokenize_terms", "terms": ["Mira", "budget"]},
                {
                    "tool": "dry_run_decode",
                    "candidate_edit": {
                        "surface_id": "s_resid_l11_last",
                        "kind": "resid_add",
                        "alpha": 0.04,
                        "ttl_steps": 1,
                        "step_size": 0.04,
                    },
                    "max_new_tokens": 3,
                },
            ],
        },
    }


def _rank1_command():
    return {
        "version": "0.1",
        "decision": "apply",
        "edits": [
            {
                "id": "e_rank1",
                "target": {"surface_id": "s_weight_l7_mlp"},
                "source": {
                    "dtype": "rank1",
                    "u": {
                        "ref": {
                            "scope": "runtime",
                            "worker": "os_0",
                            "tensor": "hidden",
                            "layer": 7,
                            "token": {"mode": "last"},
                        }
                    },
                    "v": {
                        "ref": {
                            "scope": "trace",
                            "trace_id": "paired_baseline",
                            "worker": "os_0",
                            "tensor": "hidden",
                            "layer": 7,
                            "token": {"mode": "last"},
                        }
                    },
                },
                "op": {"kind": "rank1_patch", "alpha": 0.05},
                "budget": {"ttl_steps": 4, "rank_cap": 1, "revertible": True},
            }
        ],
    }


def _attn_rank1_command():
    return {
        "version": "0.1",
        "decision": "apply",
        "edits": [
            {
                "id": "e_attn_rank1",
                "target": {"surface_id": "s_weight_l0_attn"},
                "source": {
                    "dtype": "rank1",
                    "u": {
                        "ref": {
                            "scope": "runtime",
                            "worker": "os_0",
                            "tensor": "resid_post",
                            "layer": 0,
                            "token": {"mode": "last"},
                        }
                    },
                    "v": {
                        "ref": {
                            "scope": "trace",
                            "trace_id": "paired_baseline",
                            "worker": "os_0",
                            "tensor": "resid_post",
                            "layer": 0,
                            "token": {"mode": "last"},
                        }
                    },
                },
                "op": {"kind": "rank1_patch", "alpha": 0.05},
                "budget": {"ttl_steps": 3, "rank_cap": 1, "revertible": True},
            }
        ],
    }


def _mlp_rank1_command():
    return {
        "version": "0.1",
        "decision": "apply",
        "edits": [
            {
                "id": "e_mlp_rank1",
                "target": {"surface_id": "s_weight_l0_mlp"},
                "source": {
                    "dtype": "rank1",
                    "u": {
                        "ref": {
                            "scope": "runtime",
                            "worker": "os_0",
                            "tensor": "resid_post",
                            "layer": 0,
                            "token": {"mode": "last"},
                        }
                    },
                    "v": {
                        "ref": {
                            "scope": "trace",
                            "trace_id": "paired_baseline",
                            "worker": "os_0",
                            "tensor": "resid_post",
                            "layer": 0,
                            "token": {"mode": "last"},
                        }
                    },
                },
                "op": {"kind": "rank1_patch", "alpha": 0.05},
                "budget": {"ttl_steps": 3, "rank_cap": 1, "revertible": True},
            }
        ],
    }


class DummyOverlayHandle(OverlayHandle):
    def __init__(self, u, v, alpha):
        self.u = u
        self.v = v
        self.alpha = alpha
        self.attached = False
        self.detached = False

    def attach(self):
        self.attached = True

    def detach(self):
        self.detached = True

    def tick(self):
        return None


class FakeRuntimeState:
    def __init__(self):
        self.seed = 17
        self.hooks = {}
        self.overlays = {}
        self.tensors = {
            ("runtime", None, "hidden", 11): torch.tensor([1.0, 2.0, 3.0]),
            ("runtime", None, "hidden", 7): torch.tensor([0.3, 0.4, 0.5]),
            ("trace", "paired_baseline", "hidden", 7): torch.tensor([0.6, 0.2, 0.1]),
        }

    def read_tensor(self, ref, _ctx):
        return self.tensors[(ref["scope"], ref.get("trace_id"), ref["tensor"], ref["layer"])]

    def register_hook(self, *, hook_name, hook_fn, edit_id, ttl_steps, revertible, metadata=None):
        self.hooks[edit_id] = {
            "hook_name": hook_name,
            "hook_fn": hook_fn,
            "ttl_steps": ttl_steps,
            "revertible": revertible,
            "metadata": dict(metadata or {}),
        }

    def register_overlay(self, *, edit_id, handle, ttl_steps, revertible, metadata=None):
        handle.attach()
        self.overlays[edit_id] = {
            "handle": handle,
            "ttl_steps": ttl_steps,
            "revertible": revertible,
            "metadata": dict(metadata or {}),
        }

    def remove_edit(self, edit_id):
        hook = self.hooks.pop(edit_id, None)
        if hook is not None:
            return
        overlay = self.overlays.pop(edit_id, None)
        if overlay is not None:
            overlay["handle"].detach()

    def tick_ttl(self):
        expired = []
        for edit_id, record in list(self.hooks.items()):
            record["ttl_steps"] -= 1
            if record["ttl_steps"] <= 0:
                expired.append(edit_id)
        for edit_id, record in list(self.overlays.items()):
            record["ttl_steps"] -= 1
            record["handle"].tick()
            if record["ttl_steps"] <= 0:
                expired.append(edit_id)
        for edit_id in expired:
            self.remove_edit(edit_id)

    def cleanup_expired(self):
        return None

    def has_edit(self, edit_id):
        return edit_id in self.hooks or edit_id in self.overlays

    def clear_edits(self):
        for edit_id in list(self.hooks):
            self.remove_edit(edit_id)
        for edit_id in list(self.overlays):
            self.remove_edit(edit_id)


class FakeAdapter(ModelAdapter):
    def bind_surface(self, surface):
        target = surface.target
        hook_name = None
        module_ref = None
        if target.kind == "activation":
            hook_name = f"{target.site}:{target.layer}"
        elif target.kind == "cache":
            hook_name = f"{target.site}:{target.layer}"
        else:
            module_ref = object()
        site = getattr(target, "site", getattr(target, "module", "unknown"))
        token_selector = getattr(target, "token", None)
        return BoundSurface(
            surface_id=surface.surface_id,
            kind=target.kind,
            hook_name=hook_name,
            module_ref=module_ref,
            layer=target.layer,
            site=site,
            token_selector=token_selector,
            head=getattr(target, "head", None),
            caps=surface.caps,
            allow_ops=list(surface.allow_ops),
            target=target,
        )

    def read_ref(self, ref, ctx):
        return ctx.runtime_state.read_tensor(ref, ctx)

    def make_activation_hook(self, surface, op_kind, tensor_fn, alpha, budget):
        self._last_activation_args = (surface, op_kind, alpha, budget)

        def hook_fn(act, _hook):
            vec = tensor_fn(self._current_step_ctx)
            vec = prepare_direction(vec, alpha=alpha, norm_clip=budget.get("norm_clip"), step_size=budget.get("step_size"))
            return self._add_to_selected_tokens(act, vec, surface.token_selector, alpha)

        return surface.hook_name, hook_fn

    def make_kv_hook(self, surface, tensor_fn, alpha, which, budget):
        self._last_kv_args = (surface, which, alpha, budget)

        def hook_fn(act, _hook):
            vec = tensor_fn(self._current_step_ctx)
            vec = prepare_direction(vec, alpha=alpha, norm_clip=budget.get("norm_clip"), step_size=budget.get("step_size"))
            return self._mix_selected_tokens(act, vec, surface.token_selector, alpha)

        return surface.hook_name, hook_fn

    def make_rank1_overlay(self, surface, u_fn, v_fn, alpha, budget):
        self._last_rank1_args = (surface, alpha, budget)
        return DummyOverlayHandle(u_fn(self._current_step_ctx), v_fn(self._current_step_ctx), alpha)


class TestSchemaAndPolicy(unittest.TestCase):
    def test_parse_controller_command_uses_surface_alias(self):
        command = parse_controller_command(_resid_command())
        self.assertIsInstance(command, ControllerCommand)
        self.assertEqual(command.decision, "apply")
        self.assertIsInstance(command.edits[0].target, SurfaceTargetRef)
        self.assertEqual(command.edits[0].budget.ttl_steps, 2)

    def test_parse_observation_packet(self):
        packet = parse_observation_packet(_make_packet())
        self.assertIsInstance(packet, ControllerObservationPacket)
        self.assertEqual(packet.surface_catalog[0].surface_id, "s_resid_l11_last")
        self.assertEqual(packet.trace_bank[0].trace_id, "best_success")

    def test_policy_rejects_surface_alpha_violation(self):
        packet = _make_packet()
        command = _resid_command()
        command["edits"][0]["op"]["alpha"] = 0.25
        with self.assertRaises(PolicyViolation):
            validate_command_against_packet(command, packet)

    def test_policy_rejects_unknown_rollback_id(self):
        packet = _make_packet(rollbackable_ids=["known"])
        command = {"version": "0.1", "decision": "rollback", "rollback_ids": ["missing"]}
        with self.assertRaises(PolicyViolation):
            validate_command_against_packet(command, packet)

    def test_policy_rejects_step_size_above_surface_cap(self):
        packet = _make_packet()
        packet["surface_catalog"][0]["caps"]["step_size"] = 0.05
        command = _resid_command()
        command["edits"][0]["budget"]["step_size"] = 0.08
        with self.assertRaises(PolicyViolation):
            validate_command_against_packet(command, packet)

    def test_policy_allows_loop_rescue_edit_when_main_run_budget_is_exhausted(self):
        packet = _make_packet()
        packet["budget"]["edits_left_this_run"] = 0
        packet["budget"]["alpha_left_total"] = 0.0
        packet["budget"]["edit_cost_left_total"] = 0.0
        packet["budget"]["loop_rescue_edits_left_this_run"] = 2
        packet["budget"]["loop_rescue_alpha_left_total"] = 0.12
        packet["budget"]["loop_rescue_edit_cost_left_total"] = 0.12

        validate_command_against_packet(_loop_rescue_command(), packet)


class TestCompiler(unittest.TestCase):
    def test_compile_resid_add_registers_hook(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)

        compiled = compile_command(_resid_command(), packet, ctx)
        self.assertEqual(len(compiled), 1)
        compiled[0].apply(ctx)
        self.assertTrue(runtime_state.has_edit("e_rescue"))

        act = torch.zeros(1, 4, 3)
        hook_fn = runtime_state.hooks["e_rescue"]["hook_fn"]
        out = hook_fn(act, None)
        expected_vec = prepare_direction(
            torch.tensor([1.0, 2.0, 3.0]),
            alpha=0.2,
            norm_clip=1.5,
            step_size=None,
        )

        self.assertTrue(torch.allclose(out[0, -1], 0.2 * expected_vec))
        self.assertTrue(torch.allclose(out[0, 0], torch.zeros(3)))

    def test_compile_rank1_patch_registers_overlay_and_rolls_back(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        permissive_policy = HarnessPolicy(deny_targets=())

        compiled = compile_command(_rank1_command(), packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)
        self.assertIn("e_rank1", runtime_state.overlays)
        overlay = runtime_state.overlays["e_rank1"]["handle"]
        self.assertTrue(overlay.attached)

        compiled[0].rollback(ctx)
        self.assertNotIn("e_rank1", runtime_state.overlays)
        self.assertTrue(overlay.detached)

    def test_compile_resid_add_enforces_step_size_and_records_cost(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet_payload = _make_packet()
        packet_payload["surface_catalog"][0]["caps"]["step_size"] = 0.1
        packet_payload["budget"]["edit_cost_left_total"] = 0.5
        packet = parse_observation_packet(packet_payload)
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        command = _resid_command()
        command["edits"][0]["budget"]["step_size"] = 0.1

        permissive_policy = HarnessPolicy(global_budget=GlobalBudget(max_total_alpha=1.0, max_total_edit_cost=1.0))
        compiled = compile_command(command, packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)

        hook_record = runtime_state.hooks["e_rescue"]
        self.assertAlmostEqual(hook_record["metadata"]["step_size"], 0.1, places=6)
        self.assertGreater(hook_record["metadata"]["edit_cost"], 0.0)

        act = torch.zeros(1, 4, 3)
        out = hook_record["hook_fn"](act, None)
        self.assertLessEqual(float(out[0, -1].norm().item()), 0.1001)

    def test_compile_loop_rescue_resid_add_records_loop_rescue_budget_pool(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet_payload = _make_packet()
        packet_payload["budget"]["loop_rescue_edits_left_this_run"] = 2
        packet_payload["budget"]["loop_rescue_alpha_left_total"] = 0.12
        packet_payload["budget"]["loop_rescue_edit_cost_left_total"] = 0.12
        packet_payload["surface_catalog"][0]["caps"]["step_size"] = 0.06
        packet = parse_observation_packet(packet_payload)
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)

        compiled = compile_command(_loop_rescue_command(), packet, ctx)
        compiled[0].apply(ctx)

        hook_record = runtime_state.hooks["e_loop_rescue"]
        self.assertEqual(hook_record["metadata"]["budget_pool"], "loop_rescue")


class _ToyTaskEnv:
    def reset(self, seed: int) -> str:
        self.seed = seed
        return f"seed={seed}"

    def score(self, output: str) -> float:
        return 1.0 if "patched" in output else 0.0

    def done(self, output: str) -> bool:
        return "patched" in output


class _ToyController:
    def __init__(self):
        self.calls = 0
        self._last_trace = None

    def latest_trace(self):
        return self._last_trace

    def invoke(self, packet):
        self.calls += 1
        self._last_trace = {
            "provider": "fake",
            "model": "toy-controller",
            "observation": {
                "step": packet["step"],
                "worker_status": packet["worker_view"]["status"],
                "surface_ids": [surface["surface_id"] for surface in packet["surface_catalog"]],
                "controller_memory": list(packet.get("controller_memory", [])),
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "fake",
                    "model": "toy-controller",
                    "latency_ms": 1.0,
                    "parse_ok": True,
                    "response_text": "{\"version\":\"0.1\",\"decision\":\"apply\"}",
                }
            ],
            "decision": {
                "decision": "apply" if self.calls == 1 else "noop",
                "edit_ids": ["e_rescue"] if self.calls == 1 else [],
                "ops": ["resid_add"] if self.calls == 1 else [],
                "step_sizes": [0.1] if self.calls == 1 else [],
                "total_edit_cost": 0.1 if self.calls == 1 else 0.0,
            },
            "success": True,
        }
        if self.calls == 1:
            return _resid_command()
        return {"version": "0.1", "decision": "noop"}


class _LoopRescueToyController(_ToyController):
    def invoke(self, packet):
        self.calls += 1
        self._last_trace = {
            "provider": "fake",
            "model": "toy-controller",
            "observation": {
                "step": packet["step"],
                "worker_status": packet["worker_view"]["status"],
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "fake",
                    "model": "toy-controller",
                    "latency_ms": 1.0,
                    "parse_ok": True,
                    "response_text": "{\"version\":\"0.1\",\"decision\":\"apply\"}",
                }
            ],
            "decision": {
                "decision": "apply" if self.calls == 1 else "noop",
                "edit_ids": ["e_loop_rescue"] if self.calls == 1 else [],
                "ops": ["resid_add"] if self.calls == 1 else [],
            },
            "success": True,
        }
        if self.calls == 1:
            return _loop_rescue_command()
        return {"version": "0.1", "decision": "noop"}


class _OverBudgetLoopRescueToyController(_ToyController):
    def invoke(self, packet):
        self.calls += 1
        self._last_trace = {
            "provider": "fake",
            "model": "toy-controller",
            "observation": {
                "step": packet["step"],
                "worker_status": packet["worker_view"]["status"],
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "fake",
                    "model": "toy-controller",
                    "latency_ms": 1.0,
                    "parse_ok": True,
                    "response_text": "{\"version\":\"0.1\",\"decision\":\"apply\"}",
                }
            ],
            "decision": {
                "decision": "apply" if self.calls == 1 else "noop",
                "edit_ids": ["e_loop_rescue_overbudget"] if self.calls == 1 else [],
                "ops": ["resid_add"] if self.calls == 1 else [],
            },
            "success": True,
        }
        if self.calls == 1:
            return _loop_rescue_command(alpha=0.08, step_size=0.08, edit_id="e_loop_rescue_overbudget")
        return {"version": "0.1", "decision": "noop"}


class _ParsedCommandController:
    def __init__(self, inner):
        self.inner = inner

    def latest_trace(self):
        return self.inner.latest_trace()

    def invoke(self, packet):
        command = self.inner.invoke(packet)
        if isinstance(command, ControllerCommand):
            return command
        return parse_controller_command(command)


class _ToyWorkerRuntime:
    def __init__(self, runtime_state):
        self.runtime_state = runtime_state
        self.output = ""
        self.steps = 0
        self._latest_effect_trace = {"completed_effects": [], "summary": {"window_size": 0, "verdict_counts": {}, "hypothesis_stats": [], "latest_effects": []}}
        self._controller_memory = []
        self._observer_checks = []
        self._pending_observer_check_events = []
        self._tool_results = []
        self._pending_tool_events = []

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self.output = "base"
        self.steps = 0
        self._controller_memory = []
        self._observer_checks = []
        self._pending_observer_check_events = []
        self._tool_results = []
        self._pending_tool_events = []

    def step(self) -> None:
        self.steps += 1
        if self.steps >= 2 and self.runtime_state.has_edit("e_rescue"):
            self.output = "patched"
        elif self.steps >= 2:
            self.output = "base"

    def done(self) -> bool:
        return self.steps >= 2

    def build_controller_packet(self):
        rollbackable_ids = list(self.runtime_state.hooks) + list(self.runtime_state.overlays)
        packet = _make_packet(rollbackable_ids=rollbackable_ids)
        packet["controller_memory"] = [dict(entry) for entry in self._controller_memory]
        if self._observer_checks:
            packet["latest_observer_check"] = dict(self._observer_checks[-1])
            packet["recent_observer_checks"] = [dict(entry) for entry in self._observer_checks[-4:]]
        packet["tool_catalog"] = [
            {"tool": "tokenize_terms", "available": True, "cost_hint": "cheap", "budget_left": 4},
            {"tool": "dry_run_decode", "available": True, "cost_hint": "expensive", "budget_left": 4},
        ]
        if self._tool_results:
            packet["latest_tool_results"] = [dict(entry) for entry in self._tool_results[-2:]]
            packet["recent_tool_results"] = [dict(entry) for entry in self._tool_results[-4:]]
        return packet

    def observe_recent_effects(self) -> None:
        if self.steps >= 2 and self.runtime_state.has_edit("e_rescue"):
            effect = {
                "edit_id": "e_rescue",
                "surface_id": "s_resid_l11_last",
                "observed_window_steps": 1,
                "before": {"entropy": 2.5, "top1_margin": 0.05, "repetition_score": 0.4},
                "after": {"entropy": 2.0, "top1_margin": 0.08, "repetition_score": 0.2},
                "delta": {"entropy": -0.5, "top1_margin": 0.03, "repetition_score": -0.2},
                "verdict": "helpful",
                "hypothesis": "rescue",
                "expected_effect": "break_loop",
            }
            self._latest_effect_trace = {
                "completed_effects": [effect],
                "summary": summarize_effects([effect]),
            }
            return None
        self._latest_effect_trace = {"completed_effects": [], "summary": summarize_effects(())}
        return None

    def tick_ttl(self) -> None:
        return None

    def cleanup_expired(self) -> None:
        return None

    def final_text(self) -> str:
        return self.output

    def latest_effect_trace(self):
        return self._latest_effect_trace

    def record_controller_memory(self, entry, *, decision=None):
        stored = dict(entry)
        if decision is not None:
            stored["decision"] = decision
        self._controller_memory.append(stored)
        self._controller_memory = self._controller_memory[-3:]
        return stored

    def request_observer_check(self, request, *, source="controller"):
        stored = {
            "check_type": "semantic_progress",
            "trigger": f"{source}_request",
            "verdict": "baseline" if not self._observer_checks else "flat",
            "score": 0.4,
            "requested_by": source,
            "request_kind": request.get("kind", "semantic_progress"),
            "request_reason": request.get("reason"),
            "recorded_step": self.steps,
        }
        self._observer_checks.append(stored)
        self._pending_observer_check_events.append(stored)
        return dict(stored)

    def pop_observer_check_events(self):
        events = [dict(entry) for entry in self._pending_observer_check_events]
        self._pending_observer_check_events = []
        return events

    def request_controller_tools(self, requests, *, source="controller"):
        results = []
        for request in requests:
            tool_name = request.get("tool")
            if tool_name == "tokenize_terms":
                result = {
                    "tool": "tokenize_terms",
                    "status": "ok",
                    "requested_by": source,
                    "recorded_step": self.steps,
                    "term_count": len(request.get("terms", [])),
                }
            elif tool_name == "dry_run_decode":
                result = {
                    "tool": "dry_run_decode",
                    "status": "ok",
                    "requested_by": source,
                    "recorded_step": self.steps,
                    "entropy_delta": -0.1,
                    "repeat_flag_delta": -1,
                    "required_term_recall_delta": 0.25,
                }
            else:
                continue
            self._tool_results.append(result)
            self._pending_tool_events.append(result)
            results.append(dict(result))
        self._tool_results = self._tool_results[-4:]
        return results

    def pop_tool_events(self):
        events = [dict(entry) for entry in self._pending_tool_events]
        self._pending_tool_events = []
        return events


class _BudgetExhaustedToyWorkerRuntime(_ToyWorkerRuntime):
    def build_controller_packet(self):
        packet = super().build_controller_packet()
        packet["budget"]["edits_left_this_run"] = 0
        return packet


class _LoopRescueBudgetToyWorkerRuntime(_ToyWorkerRuntime):
    def build_controller_packet(self):
        packet = super().build_controller_packet()
        packet["budget"]["edits_left_this_run"] = 0
        packet["budget"]["alpha_left_total"] = 0.0
        packet["budget"]["edit_cost_left_total"] = 0.0
        packet["budget"]["loop_rescue_edits_left_this_run"] = 2
        packet["budget"]["loop_rescue_alpha_left_total"] = 0.12
        packet["budget"]["loop_rescue_edit_cost_left_total"] = 0.12
        return packet


class _LowAlphaLoopRescueBudgetToyWorkerRuntime(_ToyWorkerRuntime):
    def build_controller_packet(self):
        packet = super().build_controller_packet()
        packet["budget"]["edits_left_this_run"] = 0
        packet["budget"]["alpha_left_total"] = 0.0
        packet["budget"]["edit_cost_left_total"] = 0.0
        packet["budget"]["loop_rescue_edits_left_this_run"] = 1
        packet["budget"]["loop_rescue_alpha_left_total"] = 0.04
        packet["budget"]["loop_rescue_edit_cost_left_total"] = 0.04
        return packet


class _MemoryToyController(_ToyController):
    def invoke(self, packet):
        response = super().invoke(packet)
        if self.calls == 1:
            return _resid_command_with_memory()
        return response


class _ObserverRequestToyController(_ToyController):
    def invoke(self, packet):
        response = super().invoke(packet)
        if self.calls == 1:
            return _noop_command_with_observer_request()
        return response


class _ToolRequestToyController(_ToyController):
    def invoke(self, packet):
        response = super().invoke(packet)
        if self.calls == 1:
            return _noop_command_with_tool_requests()
        return response


class TestLoopAndEffects(unittest.TestCase):
    def test_build_edit_effect_marks_helpful(self):
        effect = build_edit_effect(
            edit_id="e1",
            surface_id="s1",
            observed_window_steps=1,
            before={
                "entropy": 2.5,
                "top1_margin": 0.05,
                "repetition_score": 0.4,
                "required_term_recall": 0.0,
                "forbidden_term_clean": 1.0,
                "word_budget_score": 1.0,
                "budget_ok": 1.0,
            },
            after={
                "entropy": 2.1,
                "top1_margin": 0.09,
                "repetition_score": 0.2,
                "required_term_recall": 0.25,
                "forbidden_term_clean": 1.0,
                "word_budget_score": 1.0,
                "budget_ok": 1.0,
            },
            hypothesis="rescue",
            expected_effect="break_loop",
        )
        self.assertEqual(effect["verdict"], "helpful")
        self.assertEqual(effect["hypothesis"], "rescue")
        summary = summarize_effects([effect])
        self.assertEqual(summary["hypothesis_stats"][0]["hypothesis"], "rescue")
        self.assertEqual(summary["verdict_counts"]["helpful"], 1)
        self.assertIn("mean_progress_delta", summary["hypothesis_stats"][0])
        self.assertIn("progress_delta", summary["latest_effects"][0])
        self.assertIn("mean_required_term_recall_delta", summary["hypothesis_stats"][0])
        self.assertIn("required_term_recall_delta", summary["latest_effects"][0])
        self.assertIn("mean_required_term_span_progress_delta", summary["hypothesis_stats"][0])
        self.assertIn("required_term_span_progress_delta", summary["latest_effects"][0])

    def test_build_edit_effect_marks_looping_without_progress_as_harmful(self):
        effect = build_edit_effect(
            edit_id="e_loop",
            surface_id="s1",
            observed_window_steps=1,
            before={
                "entropy": 10.1,
                "top1_margin": 0.01,
                "repetition_score": 0.0,
                "partial_score": 0.45,
                "repeat_flag": 0.0,
                "no_progress_steps": 0.0,
                "progress_score": 0.25,
                "task_violation_count": 4.0,
                "done": 0.0,
            },
            after={
                "entropy": 7.8,
                "top1_margin": 0.29,
                "repetition_score": 1.0,
                "partial_score": 0.45,
                "repeat_flag": 1.0,
                "no_progress_steps": 3.0,
                "progress_score": 0.25,
                "task_violation_count": 4.0,
                "done": 0.0,
            },
            hypothesis="bad_reward",
        )

        self.assertEqual(effect["verdict"], "harmful")
        summary = summarize_effects([effect])
        self.assertEqual(summary["verdict_counts"]["harmful"], 1)
        self.assertTrue(summary["latest_effects"][0]["after_is_looping"])

    def test_build_edit_effect_marks_loop_relief_without_coverage_as_stabilizing_only(self):
        effect = build_edit_effect(
            edit_id="e_stabilize",
            surface_id="s1",
            observed_window_steps=1,
            before={
                "entropy": 5.0,
                "top1_margin": 0.02,
                "repetition_score": 1.0,
                "partial_score": 0.45,
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "repeat_flag": 1.0,
                "no_progress_steps": 2.0,
                "progress_score": 0.0,
            },
            after={
                "entropy": 3.2,
                "top1_margin": 0.1,
                "repetition_score": 0.2,
                "partial_score": 0.45,
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "repeat_flag": 0.0,
                "no_progress_steps": 0.0,
                "progress_score": 0.0,
            },
            hypothesis="loop_break_only",
        )

        self.assertEqual(effect["verdict"], "neutral")
        self.assertEqual(effect["signal_profile"], "stabilizing_only")
        summary = summarize_effects([effect])
        self.assertEqual(summary["verdict_counts"]["neutral"], 1)
        self.assertEqual(summary["hypothesis_stats"][0]["stabilizing_only_count"], 1)
        self.assertEqual(summary["latest_effects"][0]["signal_profile"], "stabilizing_only")

    def test_run_episode_smoke(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _ToyWorkerRuntime(runtime_state),
            _ToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.output, "patched")
        self.assertTrue(any(event["event"] == "controller_observation" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_provider_attempt" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_decision" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_effect" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_effect_summary" for event in logger.events))
        self.assertTrue(any(event["event"] == "compiled_edit" for event in logger.events))
        self.assertEqual(logger.events[-1]["event"], "episode_end")

    def test_run_episode_guards_apply_when_run_budget_is_exhausted(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _BudgetExhaustedToyWorkerRuntime(runtime_state),
            _ToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.output, "base")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any(event["event"] == "controller_guardrail" for event in logger.events))
        self.assertFalse(any(event["event"] == "compiled_edit" for event in logger.events))
        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["command"]["decision"], "noop")

    def test_run_episode_guards_apply_when_run_budget_is_exhausted_for_parsed_command(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _BudgetExhaustedToyWorkerRuntime(runtime_state),
            _ParsedCommandController(_ToyController()),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.output, "base")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any(event["event"] == "controller_guardrail" for event in logger.events))
        self.assertFalse(any(event["event"] == "compiled_edit" for event in logger.events))
        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["command"]["decision"], "noop")

    def test_run_episode_allows_loop_rescue_when_loop_rescue_budget_remains(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _LoopRescueBudgetToyWorkerRuntime(runtime_state),
            _LoopRescueToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.output, "base")
        self.assertFalse(any(event["event"] == "controller_guardrail" for event in logger.events))
        self.assertTrue(any(event["event"] == "compiled_edit" for event in logger.events))
        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["command"]["decision"], "apply")

    def test_run_episode_guards_over_budget_loop_rescue_apply(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _LowAlphaLoopRescueBudgetToyWorkerRuntime(runtime_state),
            _OverBudgetLoopRescueToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.output, "base")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any(event["event"] == "controller_guardrail" for event in logger.events))
        self.assertFalse(any(event["event"] == "controller_error" for event in logger.events))
        self.assertFalse(any(event["event"] == "compiled_edit" for event in logger.events))
        first_guard = next(event for event in logger.events if event["event"] == "controller_guardrail")
        self.assertEqual(first_guard["reason"], "command exceeds packet loop_rescue alpha budget")
        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["command"]["decision"], "noop")
        self.assertEqual(logger.events[-1]["event"], "episode_end")

    def test_run_episode_guards_over_budget_loop_rescue_apply_for_parsed_command(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        result = run_episode(
            _ToyTaskEnv(),
            _LowAlphaLoopRescueBudgetToyWorkerRuntime(runtime_state),
            _ParsedCommandController(_OverBudgetLoopRescueToyController()),
            ctx,
            logger=logger,
        )

        self.assertEqual(result.output, "base")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any(event["event"] == "controller_guardrail" for event in logger.events))
        self.assertFalse(any(event["event"] == "controller_error" for event in logger.events))
        self.assertFalse(any(event["event"] == "compiled_edit" for event in logger.events))
        first_guard = next(event for event in logger.events if event["event"] == "controller_guardrail")
        self.assertEqual(first_guard["reason"], "command exceeds packet loop_rescue alpha budget")
        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["command"]["decision"], "noop")
        self.assertEqual(logger.events[-1]["event"], "episode_end")

    def test_run_episode_records_controller_memory(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        worker = _ToyWorkerRuntime(runtime_state)
        logger = InMemoryStructuredLogger()

        run_episode(
            _ToyTaskEnv(),
            worker,
            _MemoryToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(worker._controller_memory[0]["hypothesis"], "small_rescue")
        self.assertEqual(worker._controller_memory[0]["micro_rationale"], "test a small reversible rescue first")
        self.assertEqual(worker._controller_memory[0]["focus_term"], "budget")
        self.assertEqual(worker._controller_memory[0]["surface_family_key"], "resid_add_s_resid_l11_last")
        self.assertAlmostEqual(worker._controller_memory[0]["transfer_confidence"], 0.35)
        self.assertAlmostEqual(worker._controller_memory[0]["same_family_escalation_risk"], 0.4)
        self.assertEqual(worker._controller_memory[0]["finish_budget_reserved"], 1)
        self.assertEqual(worker._controller_memory[0]["evidence_bullets"][0], "helpful_on=budget@l11")
        self.assertEqual(worker._controller_memory[0]["next_trigger"], "loop_relief_without_coverage")
        self.assertEqual(worker._controller_memory[0]["next_action"], "request_observer_check")
        self.assertEqual(worker._controller_memory[0]["decision"], "apply")
        self.assertTrue(any(event["event"] == "controller_memory" for event in logger.events))
        observations = [event for event in logger.events if event["event"] == "controller_observation"]
        self.assertEqual(observations[-1]["controller_memory"][0]["hypothesis"], "small_rescue")

    def test_run_episode_logs_controller_selection_report(self):
        class _BundleAwareToyWorkerRuntime(_ToyWorkerRuntime):
            def build_controller_packet(self):
                packet = super().build_controller_packet()
                packet["strategy_hints"] = {
                    "readout_analyzer_suggested_bundle_key": "kv_pair:budget:source_body:10:12",
                    "gate_report_frontier_bundle_key": "kv_pair:budget:source_body:10:12",
                    "gate_report_selection_source": "sidecar_tiebreak",
                    "shot_candidate_edits": [
                        {
                            "bundle_key": "kv_pair:budget:source_body:10:12",
                            "surface_id": "s_resid_l11_last",
                            "op": {"kind": "resid_add"},
                        }
                    ],
                }
                return packet

        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        logger = InMemoryStructuredLogger()

        run_episode(
            _ToyTaskEnv(),
            _BundleAwareToyWorkerRuntime(runtime_state),
            _ToyController(),
            ctx,
            logger=logger,
        )

        first_command = next(event for event in logger.events if event["event"] == "controller_command")
        self.assertEqual(first_command["sidecar_suggested_bundle_key"], "kv_pair:budget:source_body:10:12")
        self.assertEqual(first_command["gate_report_frontier_bundle_key"], "kv_pair:budget:source_body:10:12")
        self.assertEqual(first_command["controller_selected_bundle_key"], "kv_pair:budget:source_body:10:12")
        self.assertEqual(first_command["controller_rejected_signals"], [])
        selection_event = next(event for event in logger.events if event["event"] == "controller_selection")
        self.assertEqual(selection_event["controller_selection_source"], "sidecar_tiebreak")

    def test_run_episode_records_guarded_noop_reason_in_controller_memory(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        worker = _BudgetExhaustedToyWorkerRuntime(runtime_state)
        logger = InMemoryStructuredLogger()

        run_episode(
            _ToyTaskEnv(),
            worker,
            _ToyController(),
            ctx,
            logger=logger,
        )

        self.assertEqual(worker._controller_memory[0]["decision"], "noop")
        self.assertEqual(worker._controller_memory[0]["noop_reason"], "budget_exhausted")
        self.assertEqual(worker._controller_memory[0]["apply_block_reason"], "edits_left_this_run_exhausted")
        self.assertEqual(worker._controller_memory[0]["surface_family_key"], "resid_add_s_resid_l11_last")
        self.assertEqual(worker._controller_memory[0]["evidence_bullets"][0], "main_edit_budget exhausted")
        memory_events = [event for event in logger.events if event["event"] == "controller_memory"]
        self.assertEqual(memory_events[0]["noop_reason"], "budget_exhausted")
        self.assertEqual(memory_events[0]["apply_block_reason"], "edits_left_this_run_exhausted")

    def test_run_episode_executes_controller_requested_observer_check(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        worker = _ToyWorkerRuntime(runtime_state)
        logger = InMemoryStructuredLogger()

        run_episode(
            _ToyTaskEnv(),
            worker,
            _ObserverRequestToyController(),
            ctx,
            logger=logger,
        )

        self.assertTrue(any(event["event"] == "controller_observer_check_request" for event in logger.events))
        self.assertTrue(any(event["event"] == "observer_check" for event in logger.events))
        self.assertEqual(worker._controller_memory[0]["micro_rationale"], "Loop eased, but coverage is still zero.")
        self.assertEqual(worker._controller_memory[0]["next_trigger"], "observer_flat_after_loop_relief")
        self.assertEqual(worker._controller_memory[0]["next_action"], "apply")
        self.assertEqual(worker._observer_checks[-1]["trigger"], "controller_request")

    def test_run_episode_executes_controller_requested_tools(self):
        adapter = FakeAdapter()
        runtime_state = FakeRuntimeState()
        packet = parse_observation_packet(_make_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        worker = _ToyWorkerRuntime(runtime_state)
        logger = InMemoryStructuredLogger()

        run_episode(
            _ToyTaskEnv(),
            worker,
            _ToolRequestToyController(),
            ctx,
            logger=logger,
        )

        self.assertTrue(any(event["event"] == "controller_tool_request" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_tool_result" for event in logger.events))
        self.assertEqual(worker._tool_results[0]["tool"], "tokenize_terms")
        self.assertEqual(worker._tool_results[1]["tool"], "dry_run_decode")
        packet_after = worker.build_controller_packet()
        self.assertEqual(packet_after["tool_catalog"][0]["tool"], "tokenize_terms")
        self.assertEqual(packet_after["latest_tool_results"][0]["tool"], "tokenize_terms")


class TestRank1Bridge(unittest.TestCase):
    def test_parameter_aware_lift_maps_hidden_to_mlp_row_space(self):
        bridge = HybridRank1VectorBridge()
        matrix = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        geometry = Rank1Geometry(target_shape=(4, 3), rows=4, cols=3, matrix=matrix)
        raw = torch.tensor([1.0, 2.0, 3.0])

        adapted = bridge.adapt(raw, side="row", geometry=geometry)
        expected = matrix @ raw
        expected = expected / expected.norm()

        self.assertEqual(tuple(adapted.shape), (4,))
        self.assertTrue(torch.allclose(adapted, expected, atol=1e-6, rtol=1e-5))

    def test_resample_fallback_hits_target_length(self):
        bridge = HybridRank1VectorBridge()
        matrix = torch.eye(5, dtype=torch.float32)
        geometry = Rank1Geometry(target_shape=(5, 5), rows=5, cols=5, matrix=matrix)
        raw = torch.tensor([1.0, -1.0])

        adapted = bridge.adapt(raw, side="row", geometry=geometry)

        self.assertEqual(tuple(adapted.shape), (5,))


class DeterministicRuntimeState(FakeRuntimeState):
    HOOK_NAMES = {
        "resid_pre": "resid_pre:11",
        "resid_post": "resid_post:11",
        "hidden": "resid_post:11",
    }

    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.seed = 23
        self.vocab_size = len(codec.alphabet)
        self.base_token = int(codec.encode("a")[0].item())
        self.hint_token = int(codec.encode("b")[0].item())
        self.edit_token = int(codec.encode("c")[0].item())
        self.last_tokens = None
        self.last_logits = None
        self.last_cache = None
        self.trace_caches = {}
        self.trace_sequences = {}
        self.trace_alignment_step = None
        self.running_stats = {}

    def _freeze_cache(self, cache):
        return {str(name): tensor.detach().clone() for name, tensor in cache.items()}

    def run_with_cache(self, model_input, *, return_type="logits", **_kwargs):
        tokens = model_input.detach().clone().to(dtype=torch.long)
        seq_len = tokens.shape[1]
        base = tokens[0].float().unsqueeze(-1)
        resid_pre = torch.cat([base + 0.1, base + 0.2, base + 0.3], dim=-1).unsqueeze(0)
        resid_post = resid_pre + 0.5
        mlp_out = resid_post + 0.25

        for record in self.hooks.values():
            if record["hook_name"] == "resid_pre:11":
                resid_pre = record["hook_fn"](resid_pre, None)

        cache = {
            "resid_pre:11": resid_pre,
            "resid_post:11": resid_post,
            "mlp_out:11": mlp_out,
            "blocks.11.hook_resid_pre": resid_pre,
            "blocks.11.hook_resid_post": resid_post,
            "blocks.11.hook_mlp_out": mlp_out,
            "blocks.7.hook_resid_post": resid_post,
        }
        self.last_tokens = tokens
        self.last_cache = self._freeze_cache(cache)

        next_token = self.base_token
        if any(int(token_id) == self.hint_token for token_id in tokens[0].tolist()):
            next_token = self.hint_token
        if self.hooks or self.overlays:
            next_token = self.edit_token

        logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32)
        logits[0, -1, next_token] = 8.0
        logits[0, -1, (next_token + 1) % self.vocab_size] = 2.0
        self.last_logits = logits.detach().clone()
        return logits, cache

    def snapshot_last_cache(self, trace_id):
        self.trace_caches[trace_id] = self._freeze_cache(self.last_cache)

    def put_trace_cache(self, trace_id, cache):
        self.trace_caches[trace_id] = self._freeze_cache(cache)

    def put_step_trace(self, trace_id, trace):
        self.trace_sequences[trace_id] = trace

    def set_trace_alignment(self, step):
        self.trace_alignment_step = step

    def get_cache(self, scope, trace_id=None, *, step=None):
        if scope == "runtime":
            return self.last_cache
        if scope == "trace":
            if trace_id in self.trace_sequences:
                aligned_step = self.trace_alignment_step if step is None else step
                return self.trace_sequences[trace_id].aligned_cache(aligned_step)
            return self.trace_caches[trace_id]
        raise KeyError(scope)

    def read_tensor(self, ref, ctx):
        packet = getattr(ctx, "packet", {}) or {}
        step = packet.get("step") if isinstance(packet, dict) else getattr(packet, "step", None)
        cache = self.get_cache(ref["scope"], ref.get("trace_id"), step=step)
        hook_name = self.HOOK_NAMES[ref["tensor"]]
        return cache[hook_name][0, -1].detach().clone()


class _ThreeStepTaskEnv:
    def __init__(self, target_token):
        self.target_token = target_token

    def reset(self, seed: int) -> str:
        self.seed = seed
        return "p"

    def score(self, output: str) -> float:
        return 1.0 if self.target_token in output else 0.0

    def done(self, output: str) -> bool:
        return len(output) >= 3


class _PromptHintController:
    def __init__(self, hint):
        self.hint = hint
        self.calls = 0
        self._last_trace = None

    def latest_trace(self):
        return self._last_trace

    def invoke(self, packet):
        self.calls += 1
        self._last_trace = {
            "provider": "fake",
            "model": "prompt-hint",
            "observation": {
                "step": packet["step"],
                "worker_status": packet["worker_view"]["status"],
            },
            "attempts": [
                {
                    "attempt": 1,
                    "provider": "fake",
                    "model": "prompt-hint",
                    "latency_ms": 1.0,
                    "parse_ok": True,
                    "response_text": self.hint,
                }
            ],
            "decision": {"advice": self.hint if self.calls == 1 else None, "advice_length": len(self.hint) if self.calls == 1 else 0},
            "success": True,
        }
        if self.calls == 1:
            return self.hint
        return None


class _ResidEditController:
    def __init__(self):
        self.calls = 0

    def invoke(self, _packet):
        self.calls += 1
        if self.calls == 1:
            return _resid_command()
        return {"version": "0.1", "decision": "noop"}


class TestWorkerRuntimeAndBaselines(unittest.TestCase):
    def _surface_catalog(self):
        base_surface = _make_packet()["surface_catalog"][0]
        return [
            base_surface,
            {
                "surface_id": "s_resid_pre_l3_last",
                "target": {
                    "kind": "activation",
                    "worker": "os_0",
                    "site": "resid_pre",
                    "layer": 3,
                    "token": {"mode": "last"},
                },
                "allow_ops": ["resid_add"],
                "caps": {
                    "max_alpha": 0.12,
                    "max_ttl_steps": 1,
                    "norm_clip": 1.5,
                    "step_size": 0.08,
                    "revertible_only": True,
                },
            },
            {
                "surface_id": "s_resid_pre_l4_last",
                "target": {
                    "kind": "activation",
                    "worker": "os_0",
                    "site": "resid_pre",
                    "layer": 4,
                    "token": {"mode": "last"},
                },
                "allow_ops": ["resid_add"],
                "caps": {
                    "max_alpha": 0.12,
                    "max_ttl_steps": 1,
                    "norm_clip": 1.5,
                    "step_size": 0.08,
                    "revertible_only": True,
                },
            },
            {
                "surface_id": "s_resid_pre_l3_prev",
                "target": {
                    "kind": "activation",
                    "worker": "os_0",
                    "site": "resid_pre",
                    "layer": 3,
                    "token": {"mode": "index", "value": -2},
                },
                "allow_ops": ["resid_add"],
                "caps": {
                    "max_alpha": 0.12,
                    "max_ttl_steps": 1,
                    "norm_clip": 1.5,
                    "step_size": 0.08,
                    "revertible_only": True,
                },
            },
        ]

    def _make_worker_runtime(
        self,
        *,
        decoder_control_mode: str = "off",
        task_feedback_fn=None,
        observer_check_fn=None,
        codec=None,
        readout_sidecar_analyzer=None,
        readout_analyzer_rerank_mode: str = "apply",
    ):
        codec = codec or CharacterCodec("pabc!? ")
        runtime_state = DeterministicRuntimeState(codec)
        adapter = FakeAdapter()
        return HookedTransformerWorkerRuntime(
            runtime_state=runtime_state,
            adapter=adapter,
            surface_catalog=self._surface_catalog(),
            codec=codec,
            task_id="toy_baseline",
            goal_hint="emit the useful token",
            constraints=("keep going",),
            max_generated_tokens=3,
            max_edits_per_run=4,
            max_total_alpha=0.5,
            max_active_patch_slots=1,
            stop_checker=lambda output: len(output) >= 3,
            task_feedback_fn=task_feedback_fn,
            observer_check_fn=observer_check_fn,
            decoder_control_mode=decoder_control_mode,
            readout_sidecar_analyzer=readout_sidecar_analyzer,
            readout_analyzer_rerank_mode=readout_analyzer_rerank_mode,
        )

    def test_worker_runtime_packet_is_schema_shaped(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime.step()

        packet = worker_runtime.build_controller_packet()
        parsed = parse_observation_packet(packet)

        self.assertIsInstance(parsed, ControllerObservationPacket)
        self.assertEqual(packet["worker_view"]["generated_tail"], "a")
        self.assertEqual(packet["budget"]["edits_left_this_run"], 4)
        self.assertEqual(packet["budget"]["edit_cost_left_total"], 0.5)
        self.assertEqual(packet["surface_catalog"][0]["surface_id"], "s_resid_l11_last")
        self.assertEqual(packet["recent_effect_summary"]["window_size"], 0)

    def test_worker_runtime_tokenize_terms_tool_returns_token_units(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")

        results = worker_runtime.request_controller_tools(
            [{"tool": "tokenize_terms", "terms": ["ab", "c"]}],
            source="controller",
        )

        self.assertEqual(results[0]["tool"], "tokenize_terms")
        self.assertEqual(results[0]["term_count"], 2)
        self.assertEqual(results[0]["terms"][0]["term"], "ab")
        self.assertEqual(results[0]["terms"][0]["token_ids"], [1, 2])
        self.assertEqual(results[0]["terms"][0]["piece_count"], 2)
        self.assertEqual(results[0]["terms"][0]["control_profile"], "sequence_bias_plus_patience")
        self.assertEqual(results[0]["terms"][0]["biasable_units"][0]["piece"], "a")
        self.assertEqual(results[0]["terms"][1]["control_profile"], "single_token_bias_ok")
        self.assertEqual(results[0]["soft_logit_bias_ok_terms"], ["c"])
        self.assertEqual(results[0]["needs_sequence_support_terms"], ["ab"])

    def test_worker_runtime_constraint_scorer_tool_uses_task_feedback_and_observer(self):
        def task_feedback_fn(output):
            has_c = "c" in output
            return {
                "done": has_c,
                "partial_score": 1.0 if has_c else 0.0,
                "progress_label": "progressing" if has_c else "stalled",
                "required_term_recall": 1.0 if has_c else 0.0,
                "required_term_span_progress_by_term": {"c": 1.0 if has_c else 0.0},
                "constraint_violations": [] if has_c else ["missing_required_terms"],
                "forbidden_term_clean": 1.0,
                "word_budget_score": 1.0,
                "missing_required_terms": [] if has_c else ["c"],
            }

        def observer_check_fn(output, *, task_feedback=None, trigger="runtime", worker_runtime=None):
            return {
                "check_type": "semantic_progress",
                "trigger": trigger,
                "score": 0.9 if "c" in output else 0.2,
                "raw_score": 0.9 if "c" in output else 0.2,
                "coverage_weight": 1.0,
            }

        worker_runtime = self._make_worker_runtime(
            task_feedback_fn=task_feedback_fn,
            observer_check_fn=observer_check_fn,
        )
        worker_runtime.reset("p")

        results = worker_runtime.request_controller_tools(
            [{"tool": "constraint_scorer", "candidate": "c"}],
            source="controller",
        )

        self.assertEqual(results[0]["tool"], "constraint_scorer")
        self.assertAlmostEqual(results[0]["required_term_recall"], 1.0)
        self.assertAlmostEqual(results[0]["semantic_progress_score"], 0.9)
        self.assertIn("semantic_checked", results[0]["explanation_tags"])

    def test_worker_runtime_kv_feature_scan_reports_cache_head_candidates(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("pp")

        class _FakeAttn:
            def __init__(self):
                self.W_K = torch.tensor(
                    [
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )
                self.W_V = torch.tensor(
                    [
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )

        fake_model = type(
            "FakeModel",
            (),
            {"blocks": [type("FakeBlock", (), {"attn": _FakeAttn()})() for _ in range(12)]},
        )()
        worker_runtime.model = fake_model
        worker_runtime.runtime_state.model = fake_model
        worker_runtime.runtime_state.last_cache = {
            "blocks.3.attn.hook_k": torch.tensor(
                [[[[0.1, 0.0], [0.0, 0.1]], [[0.2, 0.0], [0.1, 0.2]]]],
                dtype=torch.float32,
            ),
            "blocks.3.attn.hook_v": torch.tensor(
                [[[[0.1, 0.0], [0.3, 0.2]], [[0.1, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
        }
        worker_runtime._feature_prototype_cache["term_x"] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        scan = worker_runtime.kv_feature_scan(
            feature_groups={
                "required_terms": {
                    "terms": ("term_x",),
                    "polarity": "promote",
                    "feature_kind": "required_term",
                }
            }
        )

        self.assertIsNotNone(scan)
        assert scan is not None
        self.assertEqual(scan["projection_mode"], "attn_weight_head_projection")
        self.assertEqual(scan["groups"][0]["group"], "required_terms")
        self.assertIn(scan["top_feature_hits"][0]["site"], {"k_cache", "v_cache"})
        self.assertEqual(scan["top_feature_hits"][0]["layer"], 3)
        self.assertIn(scan["top_feature_hits"][0]["head"], {0, 1})
        self.assertEqual(scan["top_feature_hits"][0]["token_mode"], "last")
        self.assertIn("argmax_pos", scan["top_feature_hits"][0])
        self.assertTrue(scan["top_feature_hits"][0]["source_positions"])
        self.assertEqual(
            scan["top_feature_hits"][0]["source_positions"][0]["position"],
            scan["top_feature_hits"][0]["argmax_pos"],
        )

    def test_worker_runtime_kv_feature_scan_handles_gqa_expanded_weights(self):
        worker_runtime = self._make_worker_runtime()

        class _FakeAttn:
            def __init__(self):
                self.W_K = torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )
                self.W_V = torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )

        fake_model = type(
            "FakeGQAModel",
            (),
            {
                "blocks": [type("FakeBlock", (), {"attn": _FakeAttn()})() for _ in range(12)],
                "cfg": type("FakeCfg", (), {"n_heads": 6, "n_key_value_heads": 2, "d_head": 2, "d_model": 3})(),
            },
        )()
        worker_runtime.model = fake_model
        worker_runtime.runtime_state.model = fake_model
        worker_runtime.runtime_state.last_cache = {
            "blocks.3.attn.hook_k": torch.tensor(
                [[[[1.0, 0.0], [0.0, 0.1]], [[0.2, 0.0], [0.0, 0.2]]]],
                dtype=torch.float32,
            ),
            "blocks.3.attn.hook_v": torch.tensor(
                [[[[0.5, 0.0], [0.1, 0.1]], [[0.0, 0.2], [0.0, 0.6]]]],
                dtype=torch.float32,
            ),
        }
        worker_runtime._feature_prototype_cache["term_x"] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        scan = worker_runtime.kv_feature_scan(
            feature_groups={
                "required_terms": {
                    "terms": ("term_x",),
                    "polarity": "promote",
                    "feature_kind": "required_term",
                }
            }
        )

        self.assertIsNotNone(scan)
        assert scan is not None
        self.assertEqual(scan["top_feature_hits"][0]["site"], "k_cache")
        self.assertEqual(scan["top_feature_hits"][0]["layer"], 3)
        self.assertEqual(scan["top_feature_hits"][0]["head"], 0)

    def test_worker_runtime_kv_feature_scan_prioritizes_required_promote_hits(self):
        worker_runtime = self._make_worker_runtime()

        class _FakeAttn:
            def __init__(self):
                self.W_K = torch.tensor(
                    [
                        [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )
                self.W_V = torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )

        fake_model = type(
            "FakeModel",
            (),
            {"blocks": [type("FakeBlock", (), {"attn": _FakeAttn()})() for _ in range(12)]},
        )()
        worker_runtime.model = fake_model
        worker_runtime.runtime_state.model = fake_model
        worker_runtime.runtime_state.last_cache = {
            "blocks.3.attn.hook_k": torch.tensor(
                [[[[1.0, 0.0], [0.0, 1.0]], [[0.1, 0.0], [0.0, 0.2]]]],
                dtype=torch.float32,
            ),
            "blocks.3.attn.hook_v": torch.tensor(
                [[[[1.0, 0.0], [0.0, 1.0]], [[0.1, 0.0], [0.0, 0.2]]]],
                dtype=torch.float32,
            ),
        }
        worker_runtime._feature_prototype_cache["term_req"] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        worker_runtime._feature_prototype_cache["term_forbid"] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        scan = worker_runtime.kv_feature_scan(
            feature_groups={
                "required_terms": {
                    "terms": ("term_req",),
                    "polarity": "promote",
                    "feature_kind": "required_term",
                },
                "forbidden_phrases": {
                    "terms": ("term_forbid",),
                    "polarity": "suppress",
                    "feature_kind": "forbidden_phrase",
                },
            }
        )

        self.assertIsNotNone(scan)
        assert scan is not None
        self.assertEqual(scan["top_feature_hits"][0]["group"], "required_terms")
        self.assertEqual(scan["top_feature_hits"][0]["polarity"], "promote")
        self.assertEqual(scan["top_feature_hits"][0]["site"], "v_cache")
        self.assertEqual(scan["top_feature_hits"][1]["group"], "forbidden_phrases")
        self.assertEqual(scan["top_feature_hits"][1]["site"], "k_cache")

    def test_worker_runtime_packet_promotes_kv_scan_hits_into_cache_surfaces_and_hints(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x"],
            "entity_recall_terms": ["term_x"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.41,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.31,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {
                                "position": 0,
                                "relative_index": -1,
                                "segment_kind": "prompt",
                                "piece": "p",
                                "alignment": 0.31,
                            }
                        ],
                    }
                ],
                "groups": [
                    {
                        "group": "required_terms",
                        "polarity": "promote",
                        "top_features": [
                            {
                                "feature": "term_x",
                                "site": "v_cache",
                                "layer": 3,
                                "head": 1,
                                "token_mode": "last",
                                "alignment": 0.31,
                                "argmax_pos": 0,
                                "argmax_relative_index": -1,
                                "argmax_piece": "p",
                                "argmax_segment_kind": "prompt",
                                "source_positions": [
                                    {
                                        "position": 0,
                                        "relative_index": -1,
                                        "segment_kind": "prompt",
                                        "piece": "p",
                                        "alignment": 0.31,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "hypothesis": "small_loop_rescue",
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "observed_window_steps": 1,
                "before": {
                    "entropy": 1.2,
                    "top1_margin": 0.1,
                    "repetition_score": 0.8,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "after": {
                    "entropy": 1.1,
                    "top1_margin": 0.2,
                    "repetition_score": 0.5,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "delta": {
                    "repetition_score": -0.3,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
            }
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_annotate_kv_candidates_with_canary",
            side_effect=lambda items: [dict(item, canary_checked=True, canary_pass=True, canary_focus_logit_delta=0.01) for item in items],
        ):
            packet = worker_runtime.build_controller_packet()
        parsed = parse_observation_packet(packet)

        promoted = next(surface for surface in packet["surface_catalog"] if surface["target"]["kind"] == "cache")
        self.assertEqual(promoted["target"]["site"], "v_cache")
        self.assertEqual(promoted["target"]["layer"], 3)
        self.assertEqual(promoted["target"]["head"], 1)
        self.assertEqual(promoted["allow_ops"], ["kv_mix"])
        self.assertEqual(promoted["caps"]["max_ttl_steps"], 1)
        self.assertEqual(
            packet["latest_observer_check"]["kv_feature_scan"]["top_feature_hits"][0]["surface_id"],
            promoted["surface_id"],
        )
        self.assertEqual(parsed.surface_map()[promoted["surface_id"]].target.kind, "cache")
        self.assertEqual(packet["strategy_hints"]["preferred_kv_surface_id"], promoted["surface_id"])
        self.assertFalse(packet["strategy_hints"]["kv_probe_needed"])
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["surface_id"], promoted["surface_id"])
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["op"]["kind"], "kv_mix")
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["op"]["which"], "v")
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["source"]["v"]["ref"]["tensor"], "v_cache")
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["source"]["v"]["ref"]["head"], 1)
        self.assertEqual(packet["strategy_hints"]["kv_candidate_edits"][0]["source"]["v"]["ref"]["token"]["value"], 0)
        self.assertTrue(packet["strategy_hints"]["kv_candidate_edits"][0]["canary_pass"])

    def test_worker_runtime_packet_keeps_actionable_k_side_kv_candidate(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x", "term_y"],
            "entity_recall_terms": ["term_x", "term_y"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.33,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 3,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.31,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.31}],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "term_y",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 4,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.29,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.29}],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "alignment": 0.34,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.34}],
                        "coverage_progress": 0.0,
                    },
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "hypothesis": "loop_relief_only",
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "expected_effect": "break_loop",
                "signal_profile": "stabilizing_only",
                "verdict": "neutral",
            }
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_annotate_kv_candidates_with_canary",
            side_effect=lambda items: [dict(item, canary_checked=True, canary_pass=False, canary_reason="focus_token_flat") for item in items],
        ):
            packet = worker_runtime.build_controller_packet()

        kv_candidates = packet["strategy_hints"]["kv_candidate_edits"]
        self.assertGreaterEqual(len(kv_candidates), 2)
        self.assertIn("k_cache", {item["site"] for item in kv_candidates})
        self.assertIn("v_cache", {item["site"] for item in kv_candidates})
        k_candidate = next(item for item in kv_candidates if item["site"] == "k_cache")
        self.assertEqual(k_candidate["op"]["which"], "k")
        self.assertEqual(k_candidate["site_preference"], "k_cache")
        promoted_sites = {
            surface["target"]["site"]
            for surface in packet["surface_catalog"]
            if isinstance(surface, dict) and isinstance(surface.get("target"), dict) and surface["target"].get("kind") == "cache"
        }
        self.assertIn("k_cache", promoted_sites)
        self.assertIn("v_cache", promoted_sites)

    def test_worker_runtime_packet_can_expand_k_candidate_from_surface_hits(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x"],
            "entity_recall_terms": ["term_x"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.3,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 2,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.32,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.32}],
                        "coverage_progress": 0.0,
                    }
                ],
                "groups": [
                    {
                        "group": "required_terms",
                        "polarity": "promote",
                        "top_features": [
                            {
                                "feature": "term_x",
                                "site": "v_cache",
                                "layer": 3,
                                "head": 1,
                                "token_mode": "last",
                                "alignment": 0.32,
                                "argmax_pos": 0,
                                "argmax_relative_index": -1,
                                "argmax_piece": "p",
                                "argmax_segment_kind": "prompt",
                                "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.32}],
                                "surface_hits": [
                                    {
                                        "site": "v_cache",
                                        "layer": 3,
                                        "head": 1,
                                        "token_mode": "last",
                                        "alignment": 0.32,
                                        "argmax_pos": 0,
                                        "argmax_relative_index": -1,
                                        "argmax_piece": "p",
                                        "argmax_segment_kind": "prompt",
                                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.32}],
                                    },
                                    {
                                        "site": "k_cache",
                                        "layer": 3,
                                        "head": 0,
                                        "token_mode": "last",
                                        "alignment": 0.31,
                                        "argmax_pos": 0,
                                        "argmax_relative_index": -1,
                                        "argmax_piece": "p",
                                        "argmax_segment_kind": "prompt",
                                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.31}],
                                    },
                                ],
                                "coverage_progress": 0.0,
                            }
                        ],
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "hypothesis": "loop_relief_only",
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "expected_effect": "break_loop",
                "signal_profile": "stabilizing_only",
                "verdict": "neutral",
            }
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_annotate_kv_candidates_with_canary",
            side_effect=lambda items: [dict(item, canary_checked=True, canary_pass=False, canary_reason="focus_token_flat") for item in items],
        ):
            packet = worker_runtime.build_controller_packet()

        self.assertIn("k_cache", {item["site"] for item in packet["strategy_hints"]["kv_candidate_edits"]})

    def test_worker_runtime_build_dry_run_command_accepts_kv_candidate_edit(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x"],
            "entity_recall_terms": ["term_x"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.41,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.31,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {
                                "position": 0,
                                "relative_index": -1,
                                "segment_kind": "prompt",
                                "piece": "p",
                                "alignment": 0.31,
                            }
                        ],
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "hypothesis": "small_loop_rescue",
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "observed_window_steps": 1,
                "before": {
                    "entropy": 1.2,
                    "top1_margin": 0.1,
                    "repetition_score": 0.8,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "after": {
                    "entropy": 1.1,
                    "top1_margin": 0.2,
                    "repetition_score": 0.5,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "delta": {
                    "repetition_score": -0.3,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
            }
        ]
        packet = worker_runtime.build_controller_packet()
        candidate_edit = packet["strategy_hints"]["kv_candidate_edits"][0]

        command = worker_runtime._build_dry_run_command(candidate_edit)

        self.assertIsNotNone(command)
        assert command is not None
        parsed = parse_controller_command(command)
        self.assertEqual(parsed.edits[0].op.kind, "kv_mix")
        self.assertEqual(parsed.edits[0].target.surface_id, candidate_edit["surface_id"])
        validate_command_against_packet(command, packet)

    def test_worker_runtime_kv_canary_accepts_two_token_prefix_progress(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["budget"],
            "entity_recall_terms": ["budget"],
        }
        vocab_size = 6
        baseline_logits = torch.zeros(vocab_size, dtype=torch.float32)
        edited_logits = torch.zeros(vocab_size, dtype=torch.float32)
        edited_logits[1] = 0.0006
        candidate = {
            "surface_id": "s_k_cache_l3_h0_last_promoted",
            "kind": "kv_mix",
            "site": "k_cache",
            "source": {
                "dtype": "cache_pair",
                "k": {
                    "ref": {
                        "scope": "runtime",
                        "worker": "os_0",
                        "tensor": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token": {"mode": "index", "value": 4},
                    }
                },
            },
            "op": {"kind": "kv_mix", "alpha": 0.03, "which": "k"},
            "budget": {"ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03, "revertible": True},
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_target_token_sequences",
            return_value=[_TargetTokenSequence(term="budget", token_ids=(1, 2), variant=" budget")],
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_simulate_decode",
            side_effect=[
                {
                    "continuation": "zz",
                    "continuation_token_ids": [5, 5],
                    "first_logits": baseline_logits,
                    "entropy": 1.0,
                    "repeat_flag": False,
                    "scoring": {
                        "required_term_recall": 0.0,
                        "required_term_span_progress": 0.0,
                        "semantic_progress_score": 0.0,
                    },
                },
                {
                    "continuation": " ab",
                    "continuation_token_ids": [1, 2],
                    "first_logits": edited_logits,
                    "entropy": 0.98,
                    "repeat_flag": False,
                    "scoring": {
                        "required_term_recall": 0.0,
                        "required_term_span_progress": 0.4,
                        "semantic_progress_score": 0.02,
                    },
                },
            ],
        ):
            annotated = worker_runtime._annotate_kv_candidates_with_canary([candidate])

        self.assertTrue(annotated[0]["canary_checked"])
        self.assertTrue(annotated[0]["canary_pass"])
        self.assertEqual(annotated[0]["canary_reason"], "target_prefix_improved")
        self.assertEqual(annotated[0]["canary_prefix_depth_delta"], 2)
        self.assertGreater(annotated[0]["canary_space_focus_logit_delta"], 0.0)

    def test_worker_runtime_kv_canary_accepts_target_rank_improvement(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["budget"],
            "entity_recall_terms": ["budget"],
        }
        vocab_size = 8
        baseline_logits = torch.tensor([0.7, 0.12, 0.55, 0.4, 0.35, 0.3, 0.2, 0.1], dtype=torch.float32)
        edited_logits = torch.tensor([0.7, 0.48, 0.55, 0.4, 0.35, 0.3, 0.2, 0.1], dtype=torch.float32)
        candidate = {
            "surface_id": "s_k_cache_l3_h0_last_promoted",
            "kind": "kv_mix",
            "site": "k_cache",
            "source": {
                "dtype": "cache_pair",
                "k": {
                    "ref": {
                        "scope": "runtime",
                        "worker": "os_0",
                        "tensor": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token": {"mode": "index", "value": 4},
                    }
                },
            },
            "op": {"kind": "kv_mix", "alpha": 0.03, "which": "k"},
            "budget": {"ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03, "revertible": True},
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_target_token_sequences",
            return_value=[_TargetTokenSequence(term="budget", token_ids=(1, 2), variant=" budget")],
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_simulate_decode",
            side_effect=[
                {
                    "continuation": "zz",
                    "continuation_token_ids": [5, 5],
                    "first_logits": baseline_logits,
                    "entropy": 1.0,
                    "repeat_flag": False,
                    "scoring": {
                        "required_term_recall": 0.0,
                        "required_term_span_progress": 0.0,
                        "semantic_progress_score": 0.0,
                    },
                },
                {
                    "continuation": "zz",
                    "continuation_token_ids": [5, 5],
                    "first_logits": edited_logits,
                    "entropy": 0.99,
                    "repeat_flag": False,
                    "scoring": {
                        "required_term_recall": 0.0,
                        "required_term_span_progress": 0.0,
                        "semantic_progress_score": 0.0,
                    },
                },
            ],
        ):
            annotated = worker_runtime._annotate_kv_candidates_with_canary([candidate])

        self.assertTrue(annotated[0]["canary_checked"])
        self.assertTrue(annotated[0]["canary_pass"])
        self.assertEqual(annotated[0]["canary_reason"], "target_rank_improved")
        self.assertGreaterEqual(annotated[0]["canary_focus_rank_delta"], 4)
        self.assertGreater(annotated[0]["canary_target_mass_delta"], 0.0)

    def test_worker_runtime_actionable_kv_hits_demotes_recent_flat_probe(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x", "term_y"],
            "entity_recall_terms": ["term_x", "term_y"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.3,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 2,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "alignment": 0.36,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.36}],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "term_y",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "alignment": 0.32,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [{"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.32}],
                        "coverage_progress": 0.0,
                    },
                ],
            },
        }
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {"surface_id": "s_v_cache_l3_h1_last_promoted", "kind": "kv_mix"},
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "topk_token_diff": [{"piece": "EW", "prob_delta": 0.0, "logit_delta": 0.0}],
            },
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {"surface_id": "s_k_cache_l3_h0_last_promoted", "kind": "kv_mix"},
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.2,
                "semantic_progress_delta": 0.01,
                "repeat_flag_delta": 0,
                "topk_token_diff": [{"piece": " budget", "prob_delta": 0.0001, "logit_delta": 0.0008}],
            },
        ]

        hits = worker_runtime._actionable_kv_hits(limit=2)

        self.assertEqual(hits[0]["surface_id"], "s_k_cache_l3_h0_last_promoted")
        self.assertEqual(hits[0]["recent_probe"]["label"], "positive")
        self.assertEqual(hits[1]["surface_id"], "s_v_cache_l3_h1_last_promoted")
        self.assertEqual(hits[1]["recent_probe"]["label"], "dead_actuator")

    def test_worker_runtime_recent_probe_history_classifies_shot_bridge_in_shared_score_space(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_resid_pre_l3_prev",
                    "kind": "resid_add",
                    "role": "shot_source_bridge_prev",
                    "focus_feature": "budget",
                    "phase_objective": "readout_escape",
                    "span_kind": "exact_prompt_piece",
                    "provenance_class": "source_body",
                    "source_position": 4,
                },
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.00004,
                "target_mass_edited": 0.00035,
                "target_top20_hit_delta": 0,
                "target_top20_hits_edited": 0,
                "focus_rank_delta": 8,
                "rank_focus_delta": 10,
                "focus_logit_delta": 0.0009,
                "focus_prob_delta": 0.00002,
                "topk_token_diff": [{"piece": " budget", "prob_delta": 0.00002, "logit_delta": 0.0009}],
            }
        ]

        history = worker_runtime._recent_probe_history()

        self.assertEqual(history[0]["probe_family"], "shot_bridge")
        self.assertEqual(history[0]["probe_phase_profile"], "readout_escape")
        self.assertIn("readout", history[0]["positive_axes"])
        self.assertGreater(history[0]["readout_score"], history[0]["constraint_score"])

    def test_worker_runtime_recent_probe_history_uses_composition_criteria_for_resid_family(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_resid_l11_last",
                    "kind": "resid_add",
                    "role": "shot_last_anchor",
                    "focus_feature": "Omar",
                    "phase_objective": "entity_insertion",
                    "span_kind": "exact_prompt_span_mean",
                    "provenance_class": "source_body",
                    "source_position": 8,
                    "source_span": {"start": 8, "end": 10},
                },
                "required_term_recall_delta": 0.25,
                "required_term_span_progress_delta": 0.5,
                "semantic_progress_delta": 0.03,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.0,
                "target_top20_hit_delta": 0,
                "focus_rank_delta": 0,
                "rank_focus_delta": 0,
                "focus_logit_delta": 0.0,
                "focus_prob_delta": 0.0,
                "topk_token_diff": [{"piece": " Omar", "prob_delta": 0.0, "logit_delta": 0.0}],
            }
        ]

        history = worker_runtime._recent_probe_history(phase_profiles={"composition"})

        self.assertEqual(history[0]["probe_family"], "resid_add")
        self.assertEqual(history[0]["probe_phase_profile"], "composition")
        self.assertIn("constraint", history[0]["positive_axes"])
        self.assertGreater(history[0]["constraint_score"], history[0]["readout_score"])
        self.assertEqual(history[0]["label"], "actionable_positive")

    def test_worker_runtime_k_source_positions_prefer_term_like_prompt_piece(self):
        worker_runtime = self._make_worker_runtime()
        prototype = torch.tensor([1.0, 0.0], dtype=torch.float32)
        cache_tensor = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
        position_records = [
            {"position": 0, "relative_index": -2, "segment_kind": "prompt", "piece": ".\n"},
            {"position": 1, "relative_index": -1, "segment_kind": "prompt", "piece": " budget"},
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_project_feature_into_kv_head",
            return_value=torch.tensor([1.0, 0.0], dtype=torch.float32, device="cpu"),
        ):
            rows = worker_runtime._kv_source_positions_for_feature(
                prototype,
                feature="budget",
                cache_tensor=cache_tensor,
                layer=3,
                site="k_cache",
                head=0,
                width=2,
                head_count=1,
                position_records=position_records,
                max_positions=2,
            )

        self.assertEqual(rows[0]["piece"], " budget")
        self.assertGreater(rows[0]["anchor_quality"], rows[1]["anchor_quality"])

    def test_worker_runtime_shot_mode_only_prefers_canary_passing_kv_candidates(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["term_x"],
            "entity_recall_terms": ["term_x"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.41,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "term_x",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "alignment": 0.31,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {
                                "position": 0,
                                "relative_index": -1,
                                "segment_kind": "prompt",
                                "piece": "p",
                                "alignment": 0.31,
                            }
                        ],
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "hypothesis": "small_loop_rescue",
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "expected_effect": "break_loop",
                "signal_profile": "stabilizing_only",
                "verdict": "neutral",
            }
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_annotate_kv_candidates_with_canary",
            side_effect=lambda items: [dict(item, canary_checked=True, canary_pass=False, canary_focus_logit_delta=0.0) for item in items],
        ):
            packet = worker_runtime.build_controller_packet()

        self.assertFalse(packet["strategy_hints"]["kv_probe_needed"])
        self.assertNotIn("preferred_kv_surface_id", packet["strategy_hints"])
        self.assertFalse(packet["strategy_hints"]["kv_candidate_edits"][0]["canary_pass"])

    def test_worker_runtime_logit_bias_prefers_space_prefixed_variant(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._no_progress_steps = 1
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
            "required_term_span_progress_by_term": {"ab": 0.0},
        }

        space_token = int(worker_runtime.codec.encode(" ")[0].item())
        a_token = int(worker_runtime.codec.encode("a")[0].item())
        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        logits[space_token] = 1.0
        logits[a_token] = 1.0

        adjusted, state = worker_runtime._apply_soft_entity_logit_bias(logits)

        self.assertGreater(float(adjusted[space_token].item()), float(adjusted[a_token].item()))
        self.assertIn("ab", state["logit_bias_prefer_space_terms"])

    def test_worker_runtime_packet_exposes_entity_phase_and_l4_term_cooldown(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "progressing",
            "required_term_recall": 0.0,
            "missing_required_terms": ["ab", "c"],
            "entity_recall_terms": ["ab", "c"],
        }
        worker_runtime._tool_results = [
            {
                "tool": "tokenize_terms",
                "soft_logit_bias_ok_terms": ["c"],
                "needs_sequence_support_terms": ["ab"],
                "span_progress_watch_terms": ["ab"],
            }
        ]
        worker_runtime._recent_effects = [
            {
                "edit_id": "e_required_term_nudge_l4",
                "surface_id": "s_resid_pre_l4_last",
                "op": "resid_add",
                "hypothesis": "required_term_nudge_l4",
                "verdict": "harmful",
                "delta": {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                },
            }
        ]

        packet = worker_runtime.build_controller_packet()

        self.assertEqual(packet["control_phase_hint"], "shot_mode")
        self.assertEqual(packet["strategy_hints"]["loop_severity"], "none")
        self.assertEqual(packet["strategy_hints"]["easy_entity_terms"], ["c"])
        self.assertEqual(packet["strategy_hints"]["hard_entity_terms"], ["ab"])
        self.assertTrue(packet["strategy_hints"]["prefer_space_prefixed_logit_bias"])
        self.assertTrue(packet["strategy_hints"]["prefer_auxiliary_entity_bias"])
        self.assertTrue(packet["strategy_hints"]["shot_mode_ready"])
        self.assertEqual(packet["strategy_hints"]["direct_entity_edit_gate"], "shot_mode_first")
        self.assertEqual(packet["strategy_hints"]["prefer_shot_tools"], ["constraint_scorer", "dry_run_decode"])
        self.assertTrue(packet["strategy_hints"]["shot_probe_needed"])
        self.assertEqual(packet["strategy_hints"]["preferred_shot_surface_id"], "s_resid_pre_l3_prev")
        self.assertEqual(packet["strategy_hints"]["shot_candidate_edits"][0]["surface_id"], "s_resid_pre_l3_prev")
        self.assertNotIn(
            "s_resid_pre_l4_last",
            [candidate["surface_id"] for candidate in packet["strategy_hints"]["shot_candidate_edits"]],
        )
        self.assertTrue(packet["strategy_hints"]["l4_term_nudge_cooldown"])
        self.assertEqual(packet["strategy_hints"]["avoid_recall_surfaces"], ["s_resid_pre_l4_last"])

    def test_worker_runtime_packet_switches_to_shot_mode_after_stabilizing_only_loop_relief(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget", "send", "Mira"],
            "entity_recall_terms": ["budget", "send", "Mira"],
        }
        worker_runtime._tool_results = [
            {
                "tool": "tokenize_terms",
                "soft_logit_bias_ok_terms": ["budget", "send"],
                "needs_sequence_support_terms": ["Mira"],
                "span_progress_watch_terms": ["Mira"],
            }
        ]
        worker_runtime._recent_effects = [
            {
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "hypothesis": "loop_break_small_rescue",
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
                "delta": {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                    "repeat_flag": -1.0,
                    "no_progress_steps": -1.0,
                },
            }
        ]

        packet = worker_runtime.build_controller_packet()

        self.assertEqual(packet["control_phase_hint"], "shot_mode")
        self.assertTrue(packet["strategy_hints"]["shot_mode_ready"])
        self.assertEqual(packet["strategy_hints"]["direct_entity_edit_gate"], "shot_mode_first")
        self.assertEqual(packet["strategy_hints"]["loop_break_attempt_count"], 1)
        self.assertEqual(packet["strategy_hints"]["stabilizing_only_count"], 1)
        self.assertEqual(packet["strategy_hints"]["prefer_shot_tools"], ["constraint_scorer", "dry_run_decode"])
        self.assertTrue(packet["strategy_hints"]["shot_probe_needed"])
        self.assertEqual(packet["strategy_hints"]["preferred_shot_surface_id"], "s_resid_pre_l3_prev")
        self.assertEqual(packet["strategy_hints"]["shot_candidate_edits"][0]["surface_id"], "s_resid_pre_l3_prev")
        self.assertIn(
            "s_resid_pre_l4_last",
            [candidate["surface_id"] for candidate in packet["strategy_hints"]["shot_candidate_edits"]],
        )

    def test_worker_runtime_packet_exposes_answer_readout_canary_for_missing_terms(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
        }

        packet = worker_runtime.build_controller_packet()

        canary = packet["worker_view"]["answer_readout_canary"]
        self.assertIn("top_tokens", canary)
        self.assertTrue(canary["top_tokens"])
        self.assertIn("target_mass", canary)
        self.assertIn("focus_rank", canary)
        self.assertEqual(
            packet["strategy_hints"]["answer_readout_summary"]["semantic_focus_term"],
            canary["semantic_focus_term"],
        )
        self.assertEqual(
            packet["strategy_hints"]["answer_readout_summary"]["reachable_focus_term"],
            canary["reachable_focus_term"],
        )
        self.assertEqual(
            packet["strategy_hints"]["answer_readout_summary"]["reachable_focus_rank"],
            canary["reachable_focus_rank"],
        )
        self.assertEqual(
            packet["strategy_hints"]["answer_readout_summary"]["attractor_family_overlap_tokens"],
            canary["attractor_family_overlap_tokens"],
        )

    def test_worker_runtime_strategy_hints_arm_preprobe_readout_escape_from_canary(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget", "Mira"],
            "entity_recall_terms": ["budget", "Mira"],
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "Mira",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "budget",
                "reachable_focus_piece": " budget",
                "reachable_focus_rank": 901,
                "target_mass": 0.000142,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.300192,
                "attractor_family_top_overlap": 4,
                "attractor_family_overlap_tokens": ["EW", " EW", "inou", " shards"],
                "top_tokens": ["EW", " EW", "inou", " shards", "Lawson"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        hints = packet["strategy_hints"]
        self.assertEqual(packet["control_phase_hint"], "readout_escape")
        self.assertTrue(hints["shot_mode_ready"])
        self.assertTrue(hints["readout_escape_needed"])
        self.assertEqual(hints["readout_escape_reason"], "preprobe_first_token_collapse")
        self.assertEqual(hints["controller_focus_term"], "budget")
        self.assertEqual(hints["controller_focus_source"], "reachable_focus")
        self.assertEqual(hints["attractor_family_overlap_tokens"], ["EW", " EW", "inou", " shards"])
        self.assertTrue(hints["readout_escape_block_reason"]["preprobe_collapse"])
        self.assertTrue(hints["readout_escape_block_reason"]["mass_below_threshold"])
        self.assertTrue(hints["readout_escape_block_reason"]["rank_bad"])
        self.assertTrue(hints["readout_escape_block_reason"]["top20_hits_zero"])
        self.assertTrue(hints["readout_escape_block_reason"]["attractor_mass_high"])
        self.assertFalse(hints["readout_escape_block_reason"]["shot_mode_not_ready"])
        self.assertTrue(hints["readout_escape_block_reason"]["no_candidates"])
        self.assertTrue(hints["kv_candidate_builder_called"])
        self.assertEqual(hints["kv_candidate_builder_source"], "runtime")
        self.assertEqual(hints["kv_candidate_builder_stage"], "kv_candidate_builder")
        self.assertEqual(hints["kv_candidate_builder_output_count"], 0)
        self.assertIn("no_feature_hits", hints["kv_candidate_builder_prune_reasons"])

    def test_worker_runtime_readout_escape_builder_prefers_reachable_term_and_exact_prompt_span(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p ab c")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab", "c"],
            "entity_recall_terms": ["ab", "c"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.2,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 2,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "alignment": 0.34,
                        "argmax_pos": 2,
                        "argmax_relative_index": -4,
                        "argmax_piece": "a",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 2, "relative_index": -4, "segment_kind": "prompt", "piece": "a", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "c",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "alignment": 0.36,
                        "argmax_pos": 5,
                        "argmax_relative_index": -1,
                        "argmax_piece": "c",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 5, "relative_index": -1, "segment_kind": "prompt", "piece": "c", "alignment": 0.36}
                        ],
                        "coverage_progress": 0.0,
                    },
                ],
            },
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "c",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "ab",
                "reachable_focus_piece": "ab",
                "reachable_focus_rank": 901,
                "target_mass": 0.00012,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.22,
                "attractor_family_top_overlap": 2,
                "attractor_family_overlap_tokens": ["EW", " EW"],
                "top_tokens": ["EW", " EW", "inou"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        hints = packet["strategy_hints"]
        candidate = hints["kv_candidate_edits"][0]
        self.assertEqual(packet["control_phase_hint"], "readout_escape")
        self.assertEqual(hints["escape_builder_target_terms"][0], "ab")
        self.assertGreaterEqual(hints["escape_builder_candidates_before_prune"], 2)
        self.assertEqual(candidate["focus_feature"], "ab")
        self.assertEqual(candidate["span_kind"], "exact_prompt_span_mean")
        self.assertEqual(candidate["phase_objective"], "readout_escape")
        self.assertTrue(candidate["read_source_resolved"])
        self.assertTrue(candidate["write_target_resolved"])
        self.assertEqual(candidate["source_span"], {"start": 1, "end": 4})
        self.assertEqual(candidate["candidate_family"], "kv_anchor:ab:exact_prompt_span")
        self.assertEqual(candidate["provenance_class"], "misc_prompt")

    def test_worker_runtime_readout_escape_source_bridge_uses_exact_prompt_span_mean(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p ab c")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab", "c"],
            "entity_recall_terms": ["ab", "c"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.2,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "alignment": 0.34,
                        "argmax_pos": 2,
                        "argmax_relative_index": -4,
                        "argmax_piece": "a",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 2, "relative_index": -4, "segment_kind": "prompt", "piece": "a", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    }
                ],
            },
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "c",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "ab",
                "reachable_focus_piece": "ab",
                "reachable_focus_rank": 901,
                "target_mass": 0.00012,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.22,
                "attractor_family_top_overlap": 2,
                "attractor_family_overlap_tokens": ["EW", " EW"],
                "top_tokens": ["EW", " EW", "inou"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        matching = [
            candidate
            for candidate in packet["strategy_hints"]["shot_candidate_edits"]
            if candidate.get("role") == "shot_source_bridge_span_mean"
        ]
        self.assertTrue(matching)
        self.assertEqual(matching[0]["focus_feature"], "ab")
        self.assertEqual(matching[0]["span_kind"], "exact_prompt_span_mean")
        self.assertEqual(matching[0]["phase_objective"], "readout_escape")
        self.assertEqual(matching[0]["source_span"], {"start": 1, "end": 4})
        self.assertEqual(matching[0]["provenance_class"], "misc_prompt")

    def test_worker_runtime_readout_escape_builder_prefers_source_body_and_emits_clean_bundle_inputs(self):
        prompt = (
            "Keep these terms: budget\n"
            "SOURCE: budget update\n"
            "ANSWER:"
        )
        codec = CharacterCodec("Keep these terms: budget\nSOURCE: update\nANSWER:")
        worker_runtime = self._make_worker_runtime(codec=codec)
        worker_runtime.reset(prompt)
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget"],
            "entity_recall_terms": ["budget"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.2,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 2,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "budget",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "alignment": 0.34,
                        "argmax_pos": 18,
                        "argmax_relative_index": -18,
                        "argmax_piece": "b",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 18, "relative_index": -18, "segment_kind": "prompt", "piece": "b", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "budget",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "alignment": 0.36,
                        "argmax_pos": 18,
                        "argmax_relative_index": -18,
                        "argmax_piece": "b",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 18, "relative_index": -18, "segment_kind": "prompt", "piece": "b", "alignment": 0.36}
                        ],
                        "coverage_progress": 0.0,
                    },
                ],
            },
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "budget",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "budget",
                "reachable_focus_piece": "budget",
                "reachable_focus_rank": 901,
                "target_mass": 0.00012,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.22,
                "attractor_family_top_overlap": 2,
                "attractor_family_overlap_tokens": ["EW", " EW"],
                "top_tokens": ["EW", " EW", "inou"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        hints = packet["strategy_hints"]
        self.assertEqual(packet["control_phase_hint"], "readout_escape")
        self.assertEqual(hints["candidate_provenance_counts_before_prune"]["constraint_header"], 2)
        self.assertEqual(hints["candidate_provenance_counts_before_prune"]["source_body"], 2)
        self.assertEqual(hints["candidate_provenance_counts_before_prune"]["misc_prompt"], 2)
        self.assertEqual(hints["candidate_provenance_counts_after_prune"], {"source_body": 2})
        self.assertEqual(hints["dominance_prune_drops"], 4)
        self.assertEqual(hints["same_term_family_drops"], 0)
        bundle_inputs = hints["bundle_inputs"]
        self.assertEqual(len(bundle_inputs), 1)
        self.assertEqual(bundle_inputs[0]["term"], "budget")
        self.assertEqual(bundle_inputs[0]["provenance_class"], "source_body")
        kv_candidates = hints["kv_candidate_edits"]
        self.assertEqual(len(kv_candidates), 2)
        self.assertTrue(all(candidate["provenance_class"] == "source_body" for candidate in kv_candidates))
        self.assertTrue(all(candidate.get("bundle_ready", False) for candidate in kv_candidates))
        self.assertEqual({candidate["site"] for candidate in kv_candidates}, {"k_cache", "v_cache"})

    def test_heuristic_readout_sidecar_analyzer_prefers_more_reachable_term(self):
        analyzer = build_heuristic_readout_sidecar_analyzer()
        capture = ReadoutSidecarCapture(
            run_id="run",
            episode_id="ep",
            worker_id="os_0",
            step=1,
            control_phase_hint="readout_escape",
            answer_readout_canary={
                "reachable_focus_term": "budget",
                "reachable_focus_rank": 901,
                "target_mass": 0.00012,
                "attractor_family_mass": 0.24,
                "attractor_family_overlap_tokens": ["EW", " EW"],
            },
            answer_sites=(
                ReadoutSidecarSiteCapture(
                    role="answer_boundary_prev",
                    layer=3,
                    token_selector={"mode": "index", "value": -2},
                    vector=torch.tensor([1.0, 0.0, 0.0]),
                ),
                ReadoutSidecarSiteCapture(
                    role="answer_boundary_last",
                    layer=3,
                    token_selector={"mode": "last"},
                    vector=torch.tensor([0.9, 0.1, 0.0]),
                ),
            ),
            source_sites=(
                ReadoutSidecarSiteCapture(
                    role="source_body_exact_span_mean",
                    layer=3,
                    token_selector={"mode": "span", "start": 10, "end": 12, "pool": "mean"},
                    vector=torch.tensor([0.98, 0.02, 0.0]),
                    term="Mira",
                    provenance_class="source_body",
                    span=(10, 12),
                    piece=" Mira",
                ),
                ReadoutSidecarSiteCapture(
                    role="source_body_exact_span_mean",
                    layer=3,
                    token_selector={"mode": "span", "start": 18, "end": 19, "pool": "mean"},
                    vector=torch.tensor([0.25, 0.75, 0.0]),
                    term="budget",
                    provenance_class="source_body",
                    span=(18, 19),
                    piece=" budget",
                ),
            ),
        )

        hints = analyzer(capture)

        self.assertEqual(hints["focus_term_override"], "Mira")
        self.assertEqual(hints["suggested_focus_term"], "Mira")
        self.assertEqual(hints["suggested_bundle_key"], "kv_pair:Mira:source_body:10:12")
        self.assertGreater(hints["candidate_support_terms"]["Mira"], hints["candidate_support_terms"]["budget"])
        self.assertIn("kv_pair:Mira:source_body:10:12", hints["candidate_support_scores"])
        self.assertTrue(hints["attractor_family_present"])

    def test_worker_runtime_readout_sidecar_hints_shift_escape_target_terms(self):
        prompt = (
            "Keep these terms: budget Mira\n"
            "SOURCE: Mira budget\n"
            "ANSWER:"
        )
        codec = CharacterCodec("Keep these terms: budget Mira\nSOURCE: Mira budget\nANSWER:")
        captured: dict[str, Any] = {}

        def fake_sidecar(capture):
            captured["capture"] = capture
            return {
                "analyzer_name": "fake_sidecar",
                "focus_term_override": "Mira",
                "candidate_support_terms": {"Mira": 1.4},
                "term_anchor_strength_by_term": {"Mira": 0.9},
            }

        worker_runtime = self._make_worker_runtime(codec=codec, readout_sidecar_analyzer=fake_sidecar)
        worker_runtime.reset(prompt)
        worker_runtime.step()
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget", "Mira"],
            "entity_recall_terms": ["budget", "Mira"],
        }

        actionable_hits = [
            {
                "group": "required_terms",
                "feature": "budget",
                "polarity": "promote",
                "site": "v_cache",
                "layer": 3,
                "head": 1,
                "token_mode": "last",
                "surface_id": "s_v_cache_l3_h1_last_promoted",
                "alignment": 0.33,
                "argmax_pos": 17,
                "argmax_relative_index": -17,
                "argmax_piece": " budget",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 17, "relative_index": -17, "segment_kind": "prompt", "piece": " budget", "alignment": 0.33}
                ],
                "coverage_progress": 0.0,
            },
            {
                "group": "required_terms",
                "feature": "Mira",
                "polarity": "promote",
                "site": "k_cache",
                "layer": 3,
                "head": 0,
                "token_mode": "last",
                "surface_id": "s_k_cache_l3_h0_last_promoted",
                "alignment": 0.31,
                "argmax_pos": 12,
                "argmax_relative_index": -22,
                "argmax_piece": " Mira",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 12, "relative_index": -22, "segment_kind": "prompt", "piece": " Mira", "alignment": 0.31}
                ],
                "coverage_progress": 0.0,
            },
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_actionable_kv_hits",
            return_value=actionable_hits,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "budget",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "budget",
                "reachable_focus_piece": " budget",
                "reachable_focus_rank": 901,
                "target_mass": 0.00012,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.22,
                "attractor_family_top_overlap": 2,
                "attractor_family_overlap_tokens": ["EW", " EW"],
                "top_tokens": ["EW", " EW", "inou"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        self.assertIn("capture", captured)
        capture_summary = packet["worker_view"]["readout_sidecar_capture_summary"]
        self.assertEqual(capture_summary["control_phase_hint"], "readout_escape")
        self.assertGreaterEqual(capture_summary["source_provenance_counts"]["source_body"], 1)
        self.assertEqual(packet["worker_view"]["readout_analyzer_capture_summary"]["control_phase_hint"], "readout_escape")
        hints = packet["strategy_hints"]
        self.assertEqual(hints["readout_sidecar_hints"]["focus_term_override"], "Mira")
        self.assertEqual(hints["readout_sidecar_report"]["suggested_focus_term"], "Mira")
        self.assertEqual(hints["readout_analyzer_report"]["suggested_focus_term"], "Mira")
        self.assertEqual(hints["readout_sidecar_suggested_focus_term"], "Mira")
        self.assertEqual(hints["readout_sidecar_focus_term_override"], "Mira")
        self.assertEqual(hints["readout_analyzer_hints"]["focus_term_override"], "Mira")
        self.assertEqual(hints["readout_analyzer_focus_term_override"], "Mira")
        self.assertEqual(hints["readout_analyzer_suggested_focus_term"], "Mira")
        self.assertEqual(hints["controller_focus_term"], "budget")
        self.assertEqual(hints["controller_focus_source"], "reachable_focus")
        self.assertEqual(hints["escape_builder_target_terms"][0], "budget")
        self.assertEqual(hints["kv_candidate_edits"][0]["focus_feature"], "budget")

    def test_worker_runtime_non_escape_term_order_deprioritizes_sidecar_focus(self):
        prompt = (
            "Keep these terms: budget send Mira\n"
            "SOURCE: budget send Mira\n"
            "ANSWER:"
        )
        codec = CharacterCodec(prompt)
        worker_runtime = self._make_worker_runtime(codec=codec)
        worker_runtime.reset(prompt)
        worker_runtime.step()
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget", "send", "Mira"],
            "entity_recall_terms": ["budget", "send", "Mira"],
        }

        ordered_terms = worker_runtime._ordered_missing_terms_for_phase(
            control_phase_hint="entity_insertion",
            answer_readout_canary={
                "semantic_focus_term": "budget",
                "reachable_focus_term": "send",
            },
            readout_sidecar_hints={
                "focus_term_override": "Mira",
                "candidate_support_terms": {"Mira": 1.9, "budget": 0.4, "send": 0.3},
                "term_anchor_strength_by_term": {"Mira": 0.95},
            },
            max_terms=3,
        )

        self.assertEqual(ordered_terms[0], "budget")
        self.assertEqual(ordered_terms[1], "send")
        self.assertEqual(ordered_terms[2], "Mira")

    def test_worker_runtime_post_bundle_rerank_uses_sidecar_support_for_final_selection(self):
        prompt = (
            "Keep these terms: budget send\n"
            "SOURCE: send budget should stay explicit.\n"
            "ANSWER:"
        )
        codec = CharacterCodec(prompt)

        def fake_sidecar(capture):
            support_scores: dict[str, float] = {}
            for site in capture.source_sites:
                if site.term not in {"budget", "send"} or site.provenance_class != "source_body":
                    continue
                if site.span is not None:
                    span_id = f"{int(site.span[0])}:{int(site.span[1])}"
                    span_kind = "exact_prompt_span_mean" if (int(site.span[1]) - int(site.span[0])) > 1 else "exact_prompt_piece"
                else:
                    position = 0 if site.position is None else int(site.position)
                    span_id = f"{position}:{position + 1}"
                    span_kind = "exact_prompt_piece"
                base_score = 1.8 if site.term == "budget" else 0.35
                support_scores[f"kv_pair:{site.term}:source_body:{span_id}"] = base_score
                support_scores[f"{site.term}|source_body|{span_id}|kv_v|{span_kind}"] = base_score - 0.1
                support_scores[f"{site.term}|source_body|{span_id}|kv_k|{span_kind}"] = base_score - 0.2
            return {
                "analyzer_name": "fake_sidecar",
                "focus_term_override": "budget",
                "candidate_support_terms": {"budget": 1.8, "send": 0.35},
                "term_anchor_strength_by_term": {"budget": 0.9, "send": 0.15},
                "bundle_support_scores": {key: value for key, value in support_scores.items() if key.startswith("kv_pair:")},
                "candidate_support_scores": support_scores,
            }

        def _prepare_runtime(readout_sidecar_analyzer):
            worker_runtime = self._make_worker_runtime(codec=codec, readout_sidecar_analyzer=readout_sidecar_analyzer)
            worker_runtime.reset(prompt)
            worker_runtime.step()
            worker_runtime._last_task_feedback = {
                "done": False,
                "progress_label": "stalled",
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "missing_required_terms": ["budget", "send"],
                "entity_recall_terms": ["budget", "send"],
            }
            return worker_runtime

        actionable_hits = [
            {
                "group": "required_terms",
                "feature": "budget",
                "polarity": "promote",
                "site": "v_cache",
                "layer": 3,
                "head": 1,
                "token_mode": "last",
                "surface_id": "s_v_cache_l3_h1_last_promoted",
                "alignment": 0.05,
                "argmax_pos": 24,
                "argmax_relative_index": -24,
                "argmax_piece": " budget",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 24, "relative_index": -24, "segment_kind": "prompt", "piece": " budget", "alignment": 0.31}
                ],
                "coverage_progress": 0.0,
            },
            {
                "group": "required_terms",
                "feature": "budget",
                "polarity": "promote",
                "site": "k_cache",
                "layer": 3,
                "head": 0,
                "token_mode": "last",
                "surface_id": "s_k_cache_l3_h0_last_promoted",
                "alignment": 0.04,
                "argmax_pos": 24,
                "argmax_relative_index": -24,
                "argmax_piece": " budget",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 24, "relative_index": -24, "segment_kind": "prompt", "piece": " budget", "alignment": 0.28}
                ],
                "coverage_progress": 0.0,
            },
            {
                "group": "required_terms",
                "feature": "send",
                "polarity": "promote",
                "site": "v_cache",
                "layer": 3,
                "head": 3,
                "token_mode": "last",
                "surface_id": "s_v_cache_l3_h3_last_promoted",
                "alignment": 1.05,
                "argmax_pos": 19,
                "argmax_relative_index": -29,
                "argmax_piece": " send",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 19, "relative_index": -29, "segment_kind": "prompt", "piece": " send", "alignment": 0.44}
                ],
                "coverage_progress": 0.0,
            },
            {
                "group": "required_terms",
                "feature": "send",
                "polarity": "promote",
                "site": "k_cache",
                "layer": 3,
                "head": 2,
                "token_mode": "last",
                "surface_id": "s_k_cache_l3_h2_last_promoted",
                "alignment": 0.95,
                "argmax_pos": 19,
                "argmax_relative_index": -29,
                "argmax_piece": " send",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 19, "relative_index": -29, "segment_kind": "prompt", "piece": " send", "alignment": 0.41}
                ],
                "coverage_progress": 0.0,
            },
        ]
        answer_readout_canary = {
            "semantic_focus_term": "budget",
            "semantic_focus_source": "kv_feature_scan",
            "reachable_focus_term": "budget",
            "reachable_focus_piece": " budget",
            "reachable_focus_rank": 901,
            "target_mass": 0.00012,
            "target_top20_hits": 0,
            "attractor_family_mass": 0.22,
            "attractor_family_top_overlap": 2,
            "attractor_family_overlap_tokens": ["EW", " EW"],
            "top_tokens": ["EW", " EW", "inou"],
        }
        promoted_cache_surfaces = [
            {
                "surface_id": "s_v_cache_l3_h1_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "v_cache", "layer": 3, "head": 1, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.08, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.04},
            },
            {
                "surface_id": "s_k_cache_l3_h0_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "k_cache", "layer": 3, "head": 0, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.06, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03},
            },
            {
                "surface_id": "s_v_cache_l3_h3_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "v_cache", "layer": 3, "head": 3, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.08, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.04},
            },
            {
                "surface_id": "s_k_cache_l3_h2_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "k_cache", "layer": 3, "head": 2, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.06, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03},
            },
        ]

        worker_runtime_off = _prepare_runtime(None)
        worker_runtime_on = _prepare_runtime(fake_sidecar)

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_actionable_kv_hits",
            return_value=actionable_hits,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_promoted_cache_surfaces",
            return_value=promoted_cache_surfaces,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value=answer_readout_canary,
        ):
            packet_off = worker_runtime_off.build_controller_packet()
            packet_on = worker_runtime_on.build_controller_packet()

        hints_off = packet_off["strategy_hints"]
        hints_on = packet_on["strategy_hints"]
        self.assertEqual(hints_off["controller_focus_term"], "budget")
        self.assertEqual(hints_off["kv_candidate_edits"][0]["focus_feature"], "send")
        self.assertFalse(hints_off["bundle_reorder_applied"])
        self.assertEqual(hints_on["controller_focus_term"], "budget")
        self.assertEqual(hints_on["controller_focus_source"], "reachable_focus")
        self.assertEqual(hints_on["readout_analyzer_suggested_focus_term"], "budget")
        self.assertEqual(hints_on["selected_bundle_key"].split(":")[1], "budget")
        self.assertEqual(hints_on["kv_candidate_edits"][0]["focus_feature"], "budget")
        self.assertEqual(hints_on["bundle_selector_phase"], "readout_escape")
        self.assertTrue(hints_on["bundle_rerank_gate_open"])
        self.assertEqual(hints_on["bundle_rerank_gate_reasons"], [])
        self.assertTrue(hints_on["gate_report_open"])
        self.assertEqual(hints_on["gate_report_reasons"], [])
        self.assertEqual(hints_on["gate_report_frontier_bundle_key"].split(":")[1], "budget")
        self.assertEqual(hints_on["gate_report_selection_source"], "sidecar_tiebreak")
        self.assertGreater(hints_on["bundle_base_gap"], 0.0)
        self.assertGreater(hints_on["bundle_rerank_gap"], 0.0)
        self.assertEqual(hints_on["bundle_score_debug"][0]["focus_term"], "budget")
        self.assertTrue(hints_on["bundle_score_debug"][0]["bundle_support_confident"])
        self.assertTrue(hints_on["bundle_score_debug"][0]["bundle_evidence_agreement"])
        self.assertTrue(hints_on["bundle_score_debug"][0]["bundle_is_actionable_candidate"])
        self.assertGreater(hints_on["hard_margin"], 0.0)
        self.assertGreater(hints_on["pairwise_delta"], 0.0)
        self.assertEqual(
            hints_on["bundle_score_debug"][0]["pairwise_margin_breakdown_vs_base"]["actionable_adv"],
            1.0,
        )

    def test_worker_runtime_post_bundle_rerank_blocks_sidecar_when_bundle_is_not_actionable(self):
        prompt = (
            "Keep these terms: budget send\n"
            "SOURCE: send budget should stay explicit.\n"
            "ANSWER:"
        )
        codec = CharacterCodec(prompt)

        def fake_sidecar(capture):
            support_scores: dict[str, float] = {}
            for site in capture.source_sites:
                if site.term not in {"budget", "send"} or site.provenance_class != "source_body":
                    continue
                if site.span is not None:
                    span_id = f"{int(site.span[0])}:{int(site.span[1])}"
                    span_kind = "exact_prompt_span_mean" if (int(site.span[1]) - int(site.span[0])) > 1 else "exact_prompt_piece"
                else:
                    position = 0 if site.position is None else int(site.position)
                    span_id = f"{position}:{position + 1}"
                    span_kind = "exact_prompt_piece"
                base_score = 1.6 if site.term == "budget" else 0.35
                support_scores[f"kv_pair:{site.term}:source_body:{span_id}"] = base_score
                support_scores[f"{site.term}|source_body|{span_id}|kv_v|{span_kind}"] = base_score - 0.1
                support_scores[f"{site.term}|source_body|{span_id}|kv_k|{span_kind}"] = base_score - 0.2
            return {
                "analyzer_name": "fake_sidecar",
                "candidate_support_terms": {"budget": 1.6, "send": 0.35},
                "bundle_support_scores": {key: value for key, value in support_scores.items() if key.startswith("kv_pair:")},
                "candidate_support_scores": support_scores,
            }

        worker_runtime = self._make_worker_runtime(codec=codec, readout_sidecar_analyzer=fake_sidecar)
        worker_runtime.reset(prompt)
        worker_runtime.step()
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["budget", "send"],
            "entity_recall_terms": ["budget", "send"],
        }

        actionable_hits = [
            {
                "group": "required_terms",
                "feature": "budget",
                "polarity": "promote",
                "site": "v_cache",
                "layer": 3,
                "head": 1,
                "token_mode": "last",
                "surface_id": "s_v_cache_l3_h1_last_promoted",
                "alignment": 0.31,
                "argmax_pos": 24,
                "argmax_relative_index": -24,
                "argmax_piece": " budget",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 24, "relative_index": -24, "segment_kind": "prompt", "piece": " budget", "alignment": 0.31}
                ],
                "coverage_progress": 0.0,
                "recent_probe": {"label": "harmful"},
            },
            {
                "group": "required_terms",
                "feature": "budget",
                "polarity": "promote",
                "site": "k_cache",
                "layer": 3,
                "head": 0,
                "token_mode": "last",
                "surface_id": "s_k_cache_l3_h0_last_promoted",
                "alignment": 0.28,
                "argmax_pos": 24,
                "argmax_relative_index": -24,
                "argmax_piece": " budget",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 24, "relative_index": -24, "segment_kind": "prompt", "piece": " budget", "alignment": 0.28}
                ],
                "coverage_progress": 0.0,
                "recent_probe": {"label": "harmful"},
            },
            {
                "group": "required_terms",
                "feature": "send",
                "polarity": "promote",
                "site": "v_cache",
                "layer": 3,
                "head": 3,
                "token_mode": "last",
                "surface_id": "s_v_cache_l3_h3_last_promoted",
                "alignment": 0.44,
                "argmax_pos": 19,
                "argmax_relative_index": -29,
                "argmax_piece": " send",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 19, "relative_index": -29, "segment_kind": "prompt", "piece": " send", "alignment": 0.44}
                ],
                "coverage_progress": 0.0,
            },
            {
                "group": "required_terms",
                "feature": "send",
                "polarity": "promote",
                "site": "k_cache",
                "layer": 3,
                "head": 2,
                "token_mode": "last",
                "surface_id": "s_k_cache_l3_h2_last_promoted",
                "alignment": 0.41,
                "argmax_pos": 19,
                "argmax_relative_index": -29,
                "argmax_piece": " send",
                "argmax_segment_kind": "prompt",
                "source_positions": [
                    {"position": 19, "relative_index": -29, "segment_kind": "prompt", "piece": " send", "alignment": 0.41}
                ],
                "coverage_progress": 0.0,
            },
        ]
        answer_readout_canary = {
            "semantic_focus_term": "budget",
            "semantic_focus_source": "kv_feature_scan",
            "reachable_focus_term": "send",
            "reachable_focus_piece": " send",
            "reachable_focus_rank": 901,
            "target_mass": 0.00012,
            "target_top20_hits": 0,
            "attractor_family_mass": 0.22,
            "attractor_family_top_overlap": 2,
            "attractor_family_overlap_tokens": ["EW", " EW"],
            "top_tokens": ["EW", " EW", "inou"],
        }
        promoted_cache_surfaces = [
            {
                "surface_id": "s_v_cache_l3_h1_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "v_cache", "layer": 3, "head": 1, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.08, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.04},
            },
            {
                "surface_id": "s_k_cache_l3_h0_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "k_cache", "layer": 3, "head": 0, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.06, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03},
            },
            {
                "surface_id": "s_v_cache_l3_h3_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "v_cache", "layer": 3, "head": 3, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.08, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.04},
            },
            {
                "surface_id": "s_k_cache_l3_h2_last_promoted",
                "target": {"kind": "cache", "worker": "os_0", "site": "k_cache", "layer": 3, "head": 2, "token": {"mode": "last"}},
                "caps": {"max_alpha": 0.06, "max_ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.03},
            },
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_actionable_kv_hits",
            return_value=actionable_hits,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_promoted_cache_surfaces",
            return_value=promoted_cache_surfaces,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value=answer_readout_canary,
        ):
            packet = worker_runtime.build_controller_packet()

        hints = packet["strategy_hints"]
        self.assertFalse(hints["bundle_rerank_gate_open"])
        self.assertIn("no_eligible_challenger", hints["bundle_rerank_gate_reasons"])
        self.assertEqual(hints["selected_bundle_key"].split(":")[1], "send")
        self.assertFalse(hints["gate_report_open"])
        self.assertIn("no_eligible_challenger", hints["gate_report_reasons"])
        self.assertEqual(hints["gate_report_frontier_bundle_key"].split(":")[1], "send")
        self.assertEqual(hints["kv_candidate_edits"][0]["focus_feature"], "send")
        budget_debug = next(item for item in hints["bundle_score_debug"] if item["focus_term"] == "budget")
        self.assertFalse(budget_debug["bundle_is_actionable_candidate"])
        self.assertIn("recent_harmful_family", budget_debug["rerank_vetoes"])

    def test_worker_runtime_shot_candidate_edits_readout_escape_prefers_anchored_candidates(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")

        anchored_candidate = {
            "surface_id": "s_resid_pre_l3_last",
            "kind": "resid_add",
            "role": "shot_source_bridge_last",
            "focus_feature": "budget",
            "candidate_family": "resid_bridge:budget:exact_prompt_span",
            "phase_objective": "readout_escape",
            "span_kind": "exact_prompt_span_mean",
            "provenance_class": "source_body",
        }

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_source_bridge_shot_candidate_edits",
            return_value=[anchored_candidate],
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_recent_probe_outcomes_by_candidate_key",
            return_value={},
        ):
            selected = worker_runtime._shot_candidate_edits(control_phase_hint="readout_escape")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["role"], "shot_source_bridge_last")

    def test_worker_runtime_shot_mode_prefers_source_bridge_candidate(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.3,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "alignment": 0.34,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "hypothesis": "loop_break_small_rescue",
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
                "delta": {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                    "repeat_flag": -1.0,
                    "no_progress_steps": -1.0,
                },
            }
        ]

        packet = worker_runtime.build_controller_packet()

        candidate = packet["strategy_hints"]["shot_candidate_edits"][0]
        self.assertEqual(candidate["surface_id"], "s_resid_pre_l3_prev")
        self.assertEqual(candidate["role"], "shot_source_bridge_prev")
        self.assertEqual(candidate["source_position"], 0)
        self.assertEqual(candidate["source"]["expr"]["arg"]["arg"]["arg"]["ref"]["token"]["value"], 0)

    def test_worker_runtime_packet_exposes_retryable_weak_positive_kv_candidate(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.3,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "alignment": 0.34,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "a",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "a", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    }
                ],
            },
        }
        worker_runtime._recent_effects = [
            {
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "hypothesis": "loop_break_small_rescue",
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
                "delta": {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                    "repeat_flag": -1.0,
                    "no_progress_steps": -1.0,
                },
            }
        ]
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_k_cache_l3_h0_last_promoted",
                    "kind": "kv_mix",
                    "site": "k_cache",
                },
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.00002,
                "target_top20_hit_delta": 0,
                "focus_rank_delta": 5,
                "rank_focus_delta": 5,
                "topk_token_diff": [{"piece": "a", "prob_delta": 0.00002, "logit_delta": 0.0003}],
            }
        ]

        def _annotate(items):
            annotated = []
            for item in items:
                candidate = dict(item)
                if str(candidate.get("retry_stage", "") or "") == "weak_positive_retry":
                    candidate["canary_checked"] = True
                    candidate["canary_pass"] = True
                    candidate["canary_reason"] = "target_rank_improved"
                else:
                    candidate["canary_checked"] = True
                    candidate["canary_pass"] = False
                    candidate["canary_reason"] = "focus_token_flat"
                annotated.append(candidate)
            return annotated

        with patch.object(HookedTransformerWorkerRuntime, "_annotate_kv_candidates_with_canary", side_effect=_annotate):
            packet = worker_runtime.build_controller_packet()

        self.assertEqual(packet["control_phase_hint"], "shot_mode")
        self.assertTrue(packet["strategy_hints"]["kv_retry_needed"])
        self.assertEqual(packet["strategy_hints"]["preferred_kv_retry_surface_id"], "s_k_cache_l3_h0_last_promoted")
        self.assertEqual(packet["strategy_hints"]["preferred_kv_surface_id"], "s_k_cache_l3_h0_last_promoted")
        self.assertEqual(packet["strategy_hints"]["kv_retry_candidate_edits"][0]["retry_stage"], "weak_positive_retry")
        self.assertGreater(packet["strategy_hints"]["kv_retry_candidate_edits"][0]["op"]["alpha"], 0.03)
        self.assertEqual(packet["strategy_hints"]["kv_retry_candidate_edits"][0]["canary_reason"], "target_rank_improved")

    def test_worker_runtime_kv_candidates_pick_up_recent_probe_from_raw_scan(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
        }
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.3,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 1,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "alignment": 0.34,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    }
                ],
            },
        }
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_k_cache_l3_h0_last_promoted",
                    "kind": "kv_mix",
                    "site": "k_cache",
                },
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.00002,
                "target_top20_hit_delta": 0,
                "focus_rank_delta": 5,
                "rank_focus_delta": 5,
                "focus_logit_delta": 0.0008,
                "focus_prob_delta": 0.00002,
                "topk_token_diff": [{"piece": "a", "prob_delta": 0.00002, "logit_delta": 0.0003}],
            }
        ]

        packet = worker_runtime.build_controller_packet()

        matching = [
            candidate
            for candidate in packet["strategy_hints"]["kv_candidate_edits"]
            if candidate["surface_id"] == "s_k_cache_l3_h0_last_promoted"
        ]
        self.assertEqual(matching[0]["recent_probe"]["label"], "weak_positive_subthreshold")

    def test_worker_runtime_strategy_hints_flag_readout_escape_for_collapsed_probe_family(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["ab"],
            "entity_recall_terms": ["ab"],
        }
        worker_runtime._recent_effects = [
            {
                "edit_id": "e_loop_rescue",
                "surface_id": "s_resid_pre_l3_last",
                "op": "resid_add",
                "hypothesis": "loop_break_small_rescue",
                "verdict": "neutral",
                "signal_profile": "stabilizing_only",
                "delta": {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.0,
                    "semantic_progress_score": 0.0,
                    "repeat_flag": -1.0,
                    "no_progress_steps": -1.0,
                },
            }
        ]
        worker_runtime._latest_observer_check = {
            "check_type": "semantic_progress",
            "trigger": "coverage_progress",
            "score": 0.2,
            "verdict": "flat",
            "kv_feature_scan": {
                "projection_mode": "attn_weight_head_projection",
                "surface_count": 2,
                "group_count": 1,
                "top_feature_hits": [
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "token_mode": "last",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "alignment": 0.34,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.34}
                        ],
                        "coverage_progress": 0.0,
                    },
                    {
                        "group": "required_terms",
                        "feature": "ab",
                        "polarity": "promote",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "token_mode": "last",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "alignment": 0.33,
                        "argmax_pos": 0,
                        "argmax_relative_index": -1,
                        "argmax_piece": "p",
                        "argmax_segment_kind": "prompt",
                        "source_positions": [
                            {"position": 0, "relative_index": -1, "segment_kind": "prompt", "piece": "p", "alignment": 0.33}
                        ],
                        "coverage_progress": 0.0,
                    },
                ],
            },
        }
        worker_runtime._tool_results = [
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_v_cache_l3_h1_last_promoted",
                    "kind": "kv_mix",
                    "site": "v_cache",
                    "focus_feature": "ab",
                },
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.00002,
                "target_mass_edited": 0.00021,
                "target_top20_hit_delta": 0,
                "target_top20_hits_edited": 0,
                "focus_rank_delta": 8,
                "focus_rank_edited": 620,
                "rank_focus_delta": 10,
                "rank_focus_rank_edited": 512,
                "focus_logit_delta": 0.0007,
                "focus_prob_delta": 0.00002,
                "sampled_continuations": [{"variant": "candidate", "text": " shards shard"}],
                "topk_token_diff": [{"piece": "shards", "prob_delta": 0.00002, "logit_delta": 0.0003}],
            },
            {
                "tool": "dry_run_decode",
                "status": "ok",
                "candidate_edit": {
                    "surface_id": "s_k_cache_l3_h0_last_promoted",
                    "kind": "kv_mix",
                    "site": "k_cache",
                    "focus_feature": "ab",
                },
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": 0.00003,
                "target_mass_edited": 0.00024,
                "target_top20_hit_delta": 0,
                "target_top20_hits_edited": 0,
                "focus_rank_delta": 9,
                "focus_rank_edited": 604,
                "rank_focus_delta": 11,
                "rank_focus_rank_edited": 501,
                "focus_logit_delta": 0.0008,
                "focus_prob_delta": 0.00003,
                "sampled_continuations": [{"variant": "candidate", "text": " shards shard"}],
                "topk_token_diff": [{"piece": "shards", "prob_delta": 0.00003, "logit_delta": 0.00035}],
            },
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "_current_answer_readout_canary",
            return_value={
                "semantic_focus_term": "ab",
                "semantic_focus_source": "kv_feature_scan",
                "reachable_focus_term": "ab",
                "reachable_focus_piece": " ab",
                "reachable_focus_rank": 804,
                "target_mass": 0.0003,
                "target_top20_hits": 0,
                "attractor_family_mass": 0.0042,
                "attractor_family_top_overlap": 3,
                "top_tokens": ["EW", "inou", "shards"],
            },
        ):
            packet = worker_runtime.build_controller_packet()

        hints = packet["strategy_hints"]
        self.assertEqual(packet["control_phase_hint"], "shot_mode")
        self.assertTrue(hints["readout_escape_needed"])
        self.assertEqual(hints["readout_escape_reason"], "collapsed_decode_basin")
        self.assertTrue(hints["kv_effect_family_collapsed"])
        self.assertEqual(hints["kv_effect_family_count"], 1)
        self.assertEqual(hints["semantic_focus_term"], "ab")
        self.assertEqual(hints["reachable_focus_term"], "ab")
        self.assertEqual(hints["reachable_focus_rank"], 804)
        self.assertEqual(hints["controller_focus_term"], "ab")
        self.assertEqual(hints["controller_focus_source"], "reachable_focus")
        self.assertIn("candidate_family_count", hints["readout_escape_block_reason"])

    def test_worker_runtime_dry_run_decode_tool_compares_small_candidate_edit(self):
        def task_feedback_fn(output):
            has_c = "c" in output
            span = 1.0 if has_c else 0.0
            return {
                "done": has_c,
                "partial_score": 1.0 if has_c else 0.0,
                "progress_label": "progressing" if has_c else "stalled",
                "required_term_recall": 1.0 if has_c else 0.0,
                "required_term_span_progress_by_term": {"c": span},
                "required_term_span_progress": span,
                "constraint_violations": [] if has_c else ["missing_required_terms"],
                "forbidden_term_clean": 1.0,
                "word_budget_score": 1.0,
                "missing_required_terms": [] if has_c else ["c"],
            }

        def observer_check_fn(output, *, task_feedback=None, trigger="runtime", worker_runtime=None):
            return {
                "check_type": "semantic_progress",
                "trigger": trigger,
                "score": 0.8 if "c" in output else 0.1,
                "raw_score": 0.8 if "c" in output else 0.1,
                "coverage_weight": 1.0,
            }

        worker_runtime = self._make_worker_runtime(
            task_feedback_fn=task_feedback_fn,
            observer_check_fn=observer_check_fn,
        )
        worker_runtime.reset("p")

        before = worker_runtime.final_text()
        results = worker_runtime.request_controller_tools(
            [
                {
                    "tool": "dry_run_decode",
                    "candidate_edit": {
                        "surface_id": "s_resid_l11_last",
                        "kind": "resid_add",
                        "alpha": 0.04,
                        "ttl_steps": 1,
                        "step_size": 0.04,
                    },
                    "max_new_tokens": 2,
                }
            ],
            source="controller",
        )

        self.assertEqual(results[0]["tool"], "dry_run_decode")
        self.assertEqual(results[0]["status"], "ok")
        self.assertGreater(results[0]["required_term_recall_delta"], 0.0)
        self.assertGreater(results[0]["semantic_progress_delta"], 0.0)
        self.assertEqual(results[0]["probe_family"], "resid_add")
        self.assertEqual(results[0]["probe_phase_profile"], "composition")
        self.assertIn("constraint", results[0]["positive_axes"])
        self.assertIsInstance(results[0]["probe_summary"], dict)
        self.assertEqual(worker_runtime.final_text(), before)

    def test_worker_runtime_replay_candidate_edits_actual_delta_uses_pair_override(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime._last_task_feedback = {
            "done": False,
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.0,
            "missing_required_terms": ["c"],
            "entity_recall_terms": ["c"],
        }
        baseline = {
            "continuation": "bb",
            "first_logits": torch.tensor([0.0, 8.0, 0.0], dtype=torch.float32),
            "entropy": 0.1,
            "repeat_flag": False,
            "scoring": {
                "required_term_recall": 0.0,
                "required_term_span_progress": 0.0,
                "semantic_progress_score": 0.0,
            },
        }
        edited = {
            "continuation": "cc",
            "first_logits": torch.tensor([0.0, 0.0, 8.0], dtype=torch.float32),
            "entropy": 0.2,
            "repeat_flag": False,
            "scoring": {
                "required_term_recall": 1.0,
                "required_term_span_progress": 1.0,
                "semantic_progress_score": 0.5,
            },
        }
        candidate_a = {
            "surface_id": "s_resid_l11_last",
            "kind": "resid_add",
            "phase_objective": "composition",
            "provenance_class": "source_body",
            "span_kind": "exact_prompt_span_mean",
            "focus_feature": "c",
        }
        candidate_b = {
            "surface_id": "s_resid_pre_l3_prev",
            "kind": "resid_add",
            "phase_objective": "composition",
            "provenance_class": "source_body",
            "span_kind": "exact_prompt_span_mean",
            "focus_feature": "c",
        }
        with patch.object(
            HookedTransformerWorkerRuntime,
            "_simulate_decode",
            side_effect=[baseline, edited],
        ) as simulate_mock, patch.object(
            HookedTransformerWorkerRuntime,
            "_classify_probe_result",
            return_value=None,
        ):
            result = worker_runtime.replay_candidate_edits_actual_delta(
                [candidate_a, candidate_b],
                max_new_tokens=2,
                top_k=3,
                max_edits_per_step_override=2,
                label="pair:test",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["actual_delta_class"], "target_lift")
        policy_override = simulate_mock.call_args_list[1].kwargs["policy_override"]
        self.assertEqual(policy_override.global_budget.max_edits_per_step, 2)
        self.assertEqual(result["operator_family_key"], "composition|resid_add|source_body|exact_prompt_span_mean")

    def test_worker_runtime_classify_actual_delta_result_marks_collapse_isomorphic(self):
        worker_runtime = self._make_worker_runtime()

        outcome = worker_runtime._classify_actual_delta_result(
            {
                "continuation_baseline": "bb",
                "continuation_candidate": "cc",
                "required_term_recall_delta": 0.0,
                "required_term_span_progress_delta": 0.0,
                "semantic_progress_delta": 0.0,
                "repeat_flag_delta": 0,
                "target_mass_delta": -0.972144,
                "target_top20_hit_delta": -1,
                "focus_rank_delta": 0,
                "rank_focus_delta": 0,
            }
        )

        self.assertEqual(outcome, "collapse_isomorphic")

    def test_worker_runtime_replay_operator_certification_populates_packet_hints(self):
        worker_runtime = self._make_worker_runtime()
        packet = {
            "strategy_hints": {
                "kv_candidate_edits": [
                    {
                        "bundle_key": "kv_pair:send:source_body:10:12",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "site": "v_cache",
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                    },
                    {
                        "bundle_key": "kv_pair:send:source_body:10:12",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "site": "k_cache",
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                    },
                    {
                        "bundle_key": "kv_pair:budget:source_body:12:14",
                        "surface_id": "s_v_cache_l3_h3_last_promoted",
                        "site": "v_cache",
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                    },
                    {
                        "bundle_key": "kv_pair:budget:source_body:12:14",
                        "surface_id": "s_k_cache_l3_h2_last_promoted",
                        "site": "k_cache",
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                    },
                ],
                "base_winner_bundle_key": "kv_pair:send:source_body:10:12",
                "challenger_bundle_key": "kv_pair:budget:source_body:12:14",
                "selected_bundle_key": "kv_pair:send:source_body:10:12",
                "selection_source": "base_shadow",
            }
        }
        family_key = "readout_escape|kv_pair_source_anchor|source_body|exact_prompt_span_mean"
        replay_results = [
            {"status": "ok", "label": "single:send:v", "operator_family_key": family_key, "actual_delta_class": "dead_actuator"},
            {"status": "ok", "label": "single:send:k", "operator_family_key": family_key, "actual_delta_class": "collapse_isomorphic"},
            {"status": "ok", "label": "pair:send", "operator_family_key": family_key, "actual_delta_class": "collapse_isomorphic"},
            {"status": "ok", "label": "single:budget:v", "operator_family_key": family_key, "actual_delta_class": "target_lift"},
            {"status": "ok", "label": "single:budget:k", "operator_family_key": family_key, "actual_delta_class": "target_lift"},
            {"status": "ok", "label": "pair:budget", "operator_family_key": family_key, "actual_delta_class": "neutral"},
        ]

        with patch.object(
            HookedTransformerWorkerRuntime,
            "build_controller_packet",
            return_value=packet,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "replay_candidate_edits_actual_delta",
            side_effect=replay_results,
        ):
            summary = worker_runtime.replay_operator_certification()

        self.assertEqual(summary["operator_certifications"][0]["certification_status"], "apply_eligible")
        self.assertTrue(summary["operator_certifications"][0]["certified_for_apply"])

        worker_runtime.reset("p")
        worker_runtime.step()
        live_packet = worker_runtime.build_controller_packet()
        hints = live_packet["strategy_hints"]
        self.assertEqual(hints["operator_certification_count"], 1)
        self.assertEqual(hints["operator_certification_families"][0]["operator_family_key"], family_key)

    def test_worker_runtime_materialize_recipe_candidate_builds_contrastive_window_recipe(self):
        worker_runtime = self._make_worker_runtime()
        token_records = [
            {"position": 0, "segment_kind": "prompt", "piece": " send", "provenance_class": "source_body"},
            {"position": 1, "segment_kind": "prompt", "piece": " budget", "provenance_class": "source_body"},
            {"position": 2, "segment_kind": "prompt", "piece": " Omar", "provenance_class": "source_body"},
        ]
        candidate = {
            "surface_id": "s_v_cache_l3_h1_last_promoted",
            "kind": "kv_mix",
            "site": "v_cache",
            "layer": 3,
            "head": 1,
            "phase_objective": "readout_escape",
            "bundle_family": "kv_pair_source_anchor",
            "provenance_class": "source_body",
            "span_kind": "exact_prompt_span_mean",
            "source_span": {"start": 1, "end": 2},
            "source_position": 1,
            "op": {"kind": "kv_mix", "alpha": 0.04, "which": "v"},
            "source": {"dtype": "cache_pair", "v": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "v_cache", "layer": 3, "head": 1, "token": {"mode": "span", "start": 1, "end": 2, "pool": "mean"}}}},
        }
        competitor = {
            **candidate,
            "source_span": {"start": 0, "end": 1},
            "source_position": 0,
        }

        with patch.object(HookedTransformerWorkerRuntime, "_token_position_records", return_value=token_records):
            recipe = worker_runtime._materialize_recipe_candidate(
                candidate,
                localization="exact_term_window_pm1_weighted",
                pooling="weighted_mean",
                contrast_mode="minus_base",
                competitor_candidate=competitor,
            )

        self.assertIsNotNone(recipe)
        assert recipe is not None
        self.assertEqual(recipe["recipe_localization"], "exact_term_window_pm1_weighted")
        self.assertEqual(recipe["recipe_pooling"], "weighted_mean")
        self.assertEqual(recipe["contrast_mode"], "minus_base")
        self.assertIn("exact_term_window_pm1_weighted|weighted_mean|minus_base", recipe["operator_recipe_id"])
        self.assertEqual(recipe["source"]["v"]["fn"], "sub")

    def test_worker_runtime_replay_operator_recipe_matrix_summarizes_recipe_certifications(self):
        worker_runtime = self._make_worker_runtime()
        token_records = [
            {"position": 10, "segment_kind": "prompt", "piece": " send", "provenance_class": "source_body"},
            {"position": 11, "segment_kind": "prompt", "piece": " now", "provenance_class": "source_body"},
            {"position": 12, "segment_kind": "prompt", "piece": " budget", "provenance_class": "source_body"},
            {"position": 13, "segment_kind": "prompt", "piece": " draft", "provenance_class": "source_body"},
            {"position": 14, "segment_kind": "prompt", "piece": " Omar", "provenance_class": "source_body"},
        ]
        packet = {
            "strategy_hints": {
                "kv_candidate_edits": [
                    {
                        "bundle_key": "kv_pair:send:source_body:10:12",
                        "surface_id": "s_v_cache_l3_h1_last_promoted",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 1,
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                        "source_span": {"start": 10, "end": 12},
                        "source_position": 10,
                        "op": {"kind": "kv_mix", "alpha": 0.04, "which": "v"},
                        "source": {"dtype": "cache_pair", "v": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "v_cache", "layer": 3, "head": 1, "token": {"mode": "span", "start": 10, "end": 12, "pool": "mean"}}}},
                    },
                    {
                        "bundle_key": "kv_pair:send:source_body:10:12",
                        "surface_id": "s_k_cache_l3_h0_last_promoted",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 0,
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                        "source_span": {"start": 10, "end": 12},
                        "source_position": 10,
                        "op": {"kind": "kv_mix", "alpha": 0.03, "which": "k"},
                        "source": {"dtype": "cache_pair", "k": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "k_cache", "layer": 3, "head": 0, "token": {"mode": "span", "start": 10, "end": 12, "pool": "mean"}}}},
                    },
                    {
                        "bundle_key": "kv_pair:budget:source_body:12:14",
                        "surface_id": "s_v_cache_l3_h3_last_promoted",
                        "site": "v_cache",
                        "layer": 3,
                        "head": 3,
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                        "source_span": {"start": 12, "end": 14},
                        "source_position": 12,
                        "op": {"kind": "kv_mix", "alpha": 0.04, "which": "v"},
                        "source": {"dtype": "cache_pair", "v": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "v_cache", "layer": 3, "head": 3, "token": {"mode": "span", "start": 12, "end": 14, "pool": "mean"}}}},
                    },
                    {
                        "bundle_key": "kv_pair:budget:source_body:12:14",
                        "surface_id": "s_k_cache_l3_h2_last_promoted",
                        "site": "k_cache",
                        "layer": 3,
                        "head": 2,
                        "phase_objective": "readout_escape",
                        "bundle_family": "kv_pair_source_anchor",
                        "provenance_class": "source_body",
                        "span_kind": "exact_prompt_span_mean",
                        "source_span": {"start": 12, "end": 14},
                        "source_position": 12,
                        "op": {"kind": "kv_mix", "alpha": 0.03, "which": "k"},
                        "source": {"dtype": "cache_pair", "k": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "k_cache", "layer": 3, "head": 2, "token": {"mode": "span", "start": 12, "end": 14, "pool": "mean"}}}},
                    },
                ],
                "base_winner_bundle_key": "kv_pair:send:source_body:10:12",
                "challenger_bundle_key": "kv_pair:budget:source_body:12:14",
                "selected_bundle_key": "kv_pair:send:source_body:10:12",
                "selection_source": "base_shadow",
            }
        }

        def _fake_actual_delta(candidate_edits, *, label=None, **_kwargs):
            assert label is not None
            if "term_window_pm1_weighted_minus_base:pair:kv_pair:budget:source_body:12:14" in label:
                return {"status": "ok", "label": label, "actual_delta_class": "target_lift"}
            return {"status": "ok", "label": label, "actual_delta_class": "collapse_isomorphic"}

        with patch.object(
            HookedTransformerWorkerRuntime,
            "build_controller_packet",
            return_value=packet,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "_token_position_records",
            return_value=token_records,
        ), patch.object(
            HookedTransformerWorkerRuntime,
            "replay_candidate_edits_actual_delta",
            side_effect=_fake_actual_delta,
        ):
            summary = worker_runtime.replay_operator_recipe_matrix()

        matching = [
            item for item in summary["operator_recipe_certifications"]
            if "exact_term_window_pm1_weighted|weighted_mean|minus_base" in str(item.get("operator_recipe_id", ""))
            and "kv_pair_source_anchor" in str(item.get("operator_recipe_id", ""))
        ]
        self.assertTrue(matching)
        self.assertEqual(matching[0]["certification_status"], "apply_eligible")

    def test_worker_runtime_current_tokens_use_model_device_when_available(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.model = type("FakeModel", (), {"cfg": type("Cfg", (), {"device": "cpu"})()})()
        worker_runtime.reset("p")

        with patch("SpiralInterventionLab.runtime.worker.torch.tensor", return_value=torch.zeros((1, 1), dtype=torch.long)) as tensor_mock:
            worker_runtime._current_token_tensor()

        self.assertEqual(tensor_mock.call_args.kwargs["dtype"], torch.long)
        self.assertEqual(tensor_mock.call_args.kwargs["device"], torch.device("cpu"))

    def test_worker_runtime_snapshot_trace_records_step_aligned_frames(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        for _ in range(3):
            worker_runtime.step()

        trace = worker_runtime.snapshot_trace("paired_baseline")

        self.assertIsInstance(trace, StepAlignedTrace)
        self.assertEqual(trace.step_count, 3)
        self.assertIn("paired_baseline", worker_runtime.runtime_state.trace_sequences)
        self.assertEqual(trace.aligned_frame(1).output_text, "a")
        self.assertEqual(trace.aligned_frame(3).output_text, "aaa")
        self.assertEqual(trace.aligned_cache(1)["resid_pre:11"].shape[1], 1)
        self.assertEqual(trace.aligned_cache(3)["resid_pre:11"].shape[1], 3)

    def test_worker_runtime_recent_effect_summary_tracks_hypothesis(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime.step()

        packet = worker_runtime.build_controller_packet()
        ctx = StepContext(
            packet=parse_observation_packet(packet),
            runtime_state=worker_runtime.runtime_state,
            traces={},
            stats={},
            adapter=worker_runtime.adapter,
        )
        compiled = compile_command(_resid_command(), packet, ctx)
        compiled[0].apply(ctx)

        worker_runtime.observe_recent_effects()
        worker_runtime.step()
        worker_runtime.observe_recent_effects()

        effect_trace = worker_runtime.latest_effect_trace()
        self.assertEqual(effect_trace["completed_effects"][0]["hypothesis"], "rescue")
        self.assertEqual(effect_trace["completed_effects"][0]["expected_effect"], "break_loop")

        packet_after = worker_runtime.build_controller_packet()
        summary = packet_after["recent_effect_summary"]
        self.assertEqual(summary["window_size"], 1)
        self.assertEqual(summary["hypothesis_stats"][0]["hypothesis"], "rescue")
        self.assertEqual(summary["hypothesis_stats"][0]["last_expected_effect"], "break_loop")

    def test_worker_runtime_loop_cycle_detection_marks_alternating_suffix(self):
        worker_runtime = self._make_worker_runtime()
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._append_output_token(worker_runtime.runtime_state.hint_token)
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._append_output_token(worker_runtime.runtime_state.hint_token)

        self.assertEqual(worker_runtime._loop_cycle_length(), 2)
        self.assertTrue(worker_runtime._repeat_flag())

    def test_worker_runtime_loop_aware_decoder_control_breaks_repeated_argmax_loop(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="loop_aware")
        worker_runtime.reset("p")

        for _ in range(3):
            worker_runtime.step()

        self.assertEqual(worker_runtime.final_text(), "aab")
        packet = worker_runtime.build_controller_packet()
        self.assertEqual(packet["telemetry"]["decoder_control_mode"], "loop_aware")
        self.assertTrue(packet["telemetry"]["decoder_rescue_active"])

    def test_worker_runtime_loop_aware_prune_demotes_stalled_top_token(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="loop_aware_prune")
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._no_progress_steps = 2

        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        logits[worker_runtime.runtime_state.base_token] = 1.0
        logits[worker_runtime.runtime_state.hint_token] = 2.0
        logits[worker_runtime.runtime_state.edit_token] = 1.8

        adjusted, state = worker_runtime._apply_decoder_control(logits)

        self.assertEqual(int(torch.argmax(adjusted).item()), worker_runtime.runtime_state.edit_token)
        self.assertTrue(state["candidate_prune_active"])
        self.assertEqual(state["track"], "auxiliary")

    def test_worker_runtime_loop_aware_constraint_biases_missing_term_token(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="loop_aware_constraint")
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._no_progress_steps = 1
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "missing_required_terms": ["b"],
        }

        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        logits[worker_runtime.runtime_state.base_token] = 1.0
        logits[worker_runtime.runtime_state.hint_token] = 0.2
        logits[worker_runtime.runtime_state.edit_token] = 0.7

        adjusted, state = worker_runtime._apply_decoder_control(logits)

        self.assertEqual(int(torch.argmax(adjusted).item()), worker_runtime.runtime_state.hint_token)
        self.assertEqual(state["constraint_target_terms"], ["b"])
        self.assertGreaterEqual(state["constraint_target_count"], 1)
        self.assertEqual(state["track"], "auxiliary")

    def test_worker_runtime_loop_aware_entity_recall_continues_missing_term_sequence(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="loop_aware_entity_recall")
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.hint_token)
        worker_runtime._no_progress_steps = 2
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "entity_recall_terms": ["bc"],
            "missing_required_terms": ["bc"],
        }

        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        logits[worker_runtime.runtime_state.base_token] = 0.8
        logits[worker_runtime.runtime_state.hint_token] = 1.7
        logits[worker_runtime.runtime_state.edit_token] = 1.2

        adjusted, state = worker_runtime._apply_decoder_control(logits)

        self.assertEqual(int(torch.argmax(adjusted).item()), worker_runtime.runtime_state.edit_token)
        self.assertTrue(state["entity_prior_active"])
        self.assertEqual(state["entity_prefix_depth"], 1)
        self.assertIn("bc", state["entity_continued_terms"])
        self.assertEqual(state["track"], "auxiliary")

    def test_worker_runtime_logit_bias_entity_soft_biases_missing_entity_tokens(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._append_output_token(worker_runtime.runtime_state.base_token)
        worker_runtime._no_progress_steps = 2
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "required_term_span_progress": 0.2,
            "entity_recall_terms": ["b", "c"],
            "missing_required_terms": ["b", "c"],
            "entity_recall_progress_by_term": {"b": 0.7, "c": 0.0},
            "required_term_span_progress_by_term": {"b": 0.7, "c": 0.0},
        }

        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        logits[worker_runtime.runtime_state.base_token] = 1.0
        logits[worker_runtime.runtime_state.hint_token] = 0.45
        logits[worker_runtime.runtime_state.edit_token] = 0.45

        adjusted, state = worker_runtime._apply_decoder_control(logits)

        self.assertTrue(state["logit_bias_active"])
        self.assertEqual(state["logit_bias_term_count"], 2)
        self.assertEqual(state["logit_bias_focus_term_count"], 2)
        self.assertGreaterEqual(state["logit_bias_token_count"], 2)
        self.assertGreater(float(adjusted[worker_runtime.runtime_state.hint_token].item()), 0.45)
        self.assertGreater(float(adjusted[worker_runtime.runtime_state.edit_token].item()), 0.45)
        self.assertGreater(
            float(adjusted[worker_runtime.runtime_state.edit_token].item()),
            float(adjusted[worker_runtime.runtime_state.hint_token].item()),
        )
        self.assertEqual(state["track"], "auxiliary")

    def test_worker_runtime_logit_bias_entity_soft_prefers_easy_terms_first(self):
        worker_runtime = self._make_worker_runtime(decoder_control_mode="logit_bias_entity_soft")
        worker_runtime.reset("p")
        worker_runtime._last_task_feedback = {
            "done": False,
            "progress_label": "stalled",
            "required_term_recall": 0.0,
            "entity_recall_terms": ["ab", "c"],
            "missing_required_terms": ["ab", "c"],
            "entity_recall_progress_by_term": {"ab": 0.0, "c": 0.0},
            "required_term_span_progress_by_term": {"ab": 0.0, "c": 0.0},
        }
        worker_runtime._tool_results = [
            {
                "tool": "tokenize_terms",
                "soft_logit_bias_ok_terms": ["c"],
                "needs_sequence_support_terms": ["ab"],
            }
        ]

        logits = torch.full((worker_runtime.runtime_state.vocab_size,), -5.0, dtype=torch.float32)
        adjusted, state = worker_runtime._apply_soft_entity_logit_bias(logits)

        self.assertTrue(state["logit_bias_active"])
        self.assertEqual(state["logit_bias_focus_mode"], "easy_terms_first")
        self.assertEqual(state["logit_bias_focus_terms"][0], "c")
        self.assertEqual(state["logit_bias_easy_terms"], ["c"])
        self.assertEqual(state["logit_bias_hard_terms"], ["ab"])
        self.assertGreaterEqual(float(adjusted.max().item()), -5.0)

    def test_worker_runtime_runs_bounded_observer_check_on_coverage_progress(self):
        seen = []

        def task_feedback_fn(output):
            partial = 1.0 if output.endswith("a") else 0.0
            return {
                "done": False,
                "partial_score": partial,
                "progress_label": "progressing" if partial > 0.0 else "stalled",
                "required_term_recall": partial,
                "required_term_span_progress": partial,
            }

        def observer_check_fn(output, *, task_feedback=None, trigger="runtime"):
            seen.append((output, dict(task_feedback or {}), trigger))
            return {
                "check_type": "semantic_progress",
                "trigger": trigger,
                "raw_score": 0.8,
                "coverage_weight": 1.0,
                "score": 0.8,
            }

        worker_runtime = self._make_worker_runtime(
            task_feedback_fn=task_feedback_fn,
            observer_check_fn=observer_check_fn,
        )
        worker_runtime.reset("p")
        worker_runtime.step()

        packet = worker_runtime.build_controller_packet()

        self.assertEqual(len(seen), 1)
        self.assertIn("latest_observer_check", packet)
        self.assertEqual(packet["latest_observer_check"]["trigger"], "coverage_progress")
        self.assertEqual(packet["telemetry"]["observer_check_count"], 1)
        self.assertEqual(packet["telemetry"]["observer_check_budget_left"], 3)
        self.assertEqual(packet["latest_observer_check"]["verdict"], "baseline")
        self.assertEqual(worker_runtime._effect_metrics()["semantic_progress_score"], 0.8)

    def test_worker_runtime_reports_flat_observer_check_delta_on_repeat_trigger(self):
        def observer_check_fn(output, *, task_feedback=None, trigger="runtime"):
            score = 0.62 if output.endswith("b") else 0.60
            return {
                "check_type": "semantic_progress",
                "trigger": trigger,
                "raw_score": score,
                "coverage_weight": 1.0,
                "score": score,
            }

        def task_feedback_fn(output):
            partial = 0.22 if output.endswith("b") else 0.20 if output.endswith("a") else 0.0
            return {
                "done": False,
                "partial_score": partial,
                "progress_label": "progressing" if partial > 0.0 else "stalled",
                "required_term_recall": partial,
            }

        worker_runtime = self._make_worker_runtime(
            task_feedback_fn=task_feedback_fn,
            observer_check_fn=observer_check_fn,
        )
        worker_runtime.reset("p")
        worker_runtime.step()
        worker_runtime._append_output_token(worker_runtime.runtime_state.hint_token)
        worker_runtime.step()

        packet = worker_runtime.build_controller_packet()

        self.assertEqual(packet["recent_observer_checks"][-1]["verdict"], "flat")
        self.assertAlmostEqual(packet["recent_observer_checks"][-1]["delta_vs_last_check"], 0.02)

    def test_probe_frames_use_step_aligned_baseline_trace(self):
        baseline_worker = self._make_worker_runtime()
        baseline_worker.reset("p")
        for _ in range(3):
            baseline_worker.step()
        baseline_trace = baseline_worker.snapshot_trace("paired_baseline")

        worker_runtime = self._make_worker_runtime()
        worker_runtime.runtime_state.put_step_trace("paired_baseline", baseline_trace)
        worker_runtime.reset("p")

        worker_runtime.step()
        packet_step1 = worker_runtime.build_controller_packet()
        probe_step1 = packet_step1["probe_frames"][0]["stats"]["cosine_to_paired_baseline"]

        worker_runtime.step()
        packet_step2 = worker_runtime.build_controller_packet()
        probe_step2 = packet_step2["probe_frames"][0]["stats"]["cosine_to_paired_baseline"]

        self.assertAlmostEqual(probe_step1, 1.0, places=6)
        self.assertAlmostEqual(probe_step2, 1.0, places=6)

    def test_b1_prompt_hints_affect_context_without_polluting_output(self):
        worker_runtime = self._make_worker_runtime()
        controller = _PromptHintController("b")
        logger = InMemoryStructuredLogger()

        result = run_b1(
            _ThreeStepTaskEnv("b"),
            worker_runtime,
            controller,
            logger=logger,
            max_hints_per_run=1,
        )

        self.assertEqual(result.output, "abb")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(any(event["event"] == "prompt_advice" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_provider_attempt" for event in logger.events))
        self.assertTrue(any(event["event"] == "controller_decision" for event in logger.events))

    def test_c1_runner_uses_activation_only_edit_path(self):
        worker_runtime = self._make_worker_runtime()
        controller = _ResidEditController()

        result = run_c1(_ThreeStepTaskEnv("c"), worker_runtime, controller)

        self.assertEqual(result.output, "aca")
        self.assertEqual(result.score, 1.0)

    def test_activation_only_policy_keeps_cache_edits_but_denies_weight_patches(self):
        policy = activation_only_policy()

        self.assertEqual(policy.allow_ops, ("resid_add", "kv_mix"))
        self.assertIn("kv_mix", policy.allow_ops)
        self.assertNotIn("rank1_patch", policy.allow_ops)

    def test_minimal_baseline_suite_runs_b0_b1_c1_and_seeds_paired_trace(self):
        controller = _ResidEditController()
        hint_controller = _PromptHintController("b")
        created_workers = []

        def make_worker():
            worker = self._make_worker_runtime()
            created_workers.append(worker)
            return worker

        suite = run_minimal_baseline_suite(
            _ThreeStepTaskEnv("c"),
            make_worker_runtime=make_worker,
            c1_controller=controller,
            b1_controller=hint_controller,
        )

        self.assertEqual(suite.b0.output, "aaa")
        self.assertEqual(suite.b1.output, "abb")
        self.assertEqual(suite.c1.output, "aca")
        self.assertEqual(suite.b0.score, 0.0)
        self.assertEqual(suite.c1.score, 1.0)
        self.assertEqual(suite.paired_trace_id, "paired_baseline")
        self.assertIn("paired_baseline", created_workers[1].runtime_state.trace_sequences)
        self.assertIn("paired_baseline", created_workers[2].runtime_state.trace_sequences)
        self.assertEqual(created_workers[1].runtime_state.trace_sequences["paired_baseline"].step_count, 3)


@unittest.skipUnless(HAS_TRANSFORMER_LENS, "transformer_lens is not installed")
class TestHookedTransformerIntegration(unittest.TestCase):
    def _make_model(self):
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=32,
            n_heads=4,
            d_head=8,
            d_mlp=64,
            n_ctx=16,
            d_vocab=97,
            act_fn="relu",
            device="cpu",
            seed=0,
        )
        model = HookedTransformer(cfg)
        model.eval()
        return model

    def _runtime_packet(self):
        return {
            "version": "0.1",
            "run_id": "run_tlens",
            "episode_id": "ep_tlens",
            "worker_id": "os_0",
            "step": 1,
            "horizon": {"generated_tokens": 4, "max_generated_tokens": 8, "done": False},
            "task_view": {
                "mode": "redacted",
                "task_id": "toy_tlens",
                "prompt_hash": "sha256:tlens",
                "goal_hint": "toy tlens",
                "constraints": [],
            },
            "worker_view": {"generated_tail": "x", "status": "thinking"},
            "telemetry": {
                "entropy": 1.5,
                "top1_margin": 0.15,
                "repetition_score": 0.0,
                "repeat_flag": False,
                "no_progress_steps": 0,
            },
            "surface_catalog": [
                {
                    "surface_id": "s_resid_l0_last",
                    "target": {
                        "kind": "activation",
                        "worker": "os_0",
                        "site": "resid_pre",
                        "layer": 0,
                        "token": {"mode": "last"},
                    },
                    "allow_ops": ["resid_add"],
                    "caps": {
                        "max_alpha": 0.5,
                        "max_ttl_steps": 3,
                        "norm_clip": 5.0,
                        "revertible_only": True,
                    },
                },
                {
                    "surface_id": "s_weight_l0_attn",
                    "target": {
                        "kind": "weight",
                        "worker": "os_0",
                        "module": "attn_out",
                        "layer": 0,
                    },
                    "allow_ops": ["rank1_patch"],
                    "caps": {
                        "max_alpha": 0.2,
                        "max_ttl_steps": 3,
                        "rank_cap": 1,
                        "revertible_only": True,
                    },
                },
                {
                    "surface_id": "s_weight_l0_mlp",
                    "target": {
                        "kind": "weight",
                        "worker": "os_0",
                        "module": "mlp_out",
                        "layer": 0,
                    },
                    "allow_ops": ["rank1_patch"],
                    "caps": {
                        "max_alpha": 0.2,
                        "max_ttl_steps": 3,
                        "rank_cap": 1,
                        "revertible_only": True,
                    },
                },
            ],
            "probe_frames": [],
            "trace_bank": [
                {
                    "trace_id": "paired_baseline",
                    "origin": "paired_baseline",
                    "compatible": True,
                    "similarity_hint": 1.0,
                    "tags": ["same_seed"],
                }
            ],
            "active_edits": [],
            "recent_effects": [],
            "budget": {
                "edits_left_this_step": 1,
                "edits_left_this_run": 3,
                "alpha_left_total": 0.5,
                "active_patch_slots_left": 1,
                "rollbackable_ids": [],
            },
            "task_feedback": {"done": False, "progress_label": "progressing"},
        }

    def test_hooked_transformer_runtime_resid_add_changes_cached_activation(self):
        model = self._make_model()
        runtime_state = HookedTransformerRuntimeState(model, seed=11)
        adapter = HookedTransformerAdapter(model)
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        _logits, cache = runtime_state.run_with_cache(tokens, return_type="logits")
        baseline_last = cache["blocks.0.hook_resid_pre"][0, -1].detach().clone()
        runtime_state.snapshot_last_cache("paired_baseline")

        packet = parse_observation_packet(self._runtime_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)

        command = {
            "version": "0.1",
            "decision": "apply",
            "edits": [
                {
                    "id": "e_resid_tlens",
                    "target": {"surface_id": "s_resid_l0_last"},
                    "source": {
                        "dtype": "vector",
                        "expr": {
                            "ref": {
                                "scope": "runtime",
                                "worker": "os_0",
                                "tensor": "resid_pre",
                                "layer": 0,
                                "token": {"mode": "last"},
                            }
                        },
                    },
                    "op": {"kind": "resid_add", "alpha": 0.25},
                    "budget": {"ttl_steps": 2, "norm_clip": 5.0, "revertible": True},
                }
            ],
        }

        permissive_policy = HarnessPolicy(global_budget=GlobalBudget(max_total_alpha=1.0, max_total_edit_cost=1.0))
        compiled = compile_command(command, packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)
        self.assertIn("e_resid_tlens", runtime_state.hooks)

        _logits2, edited_cache = runtime_state.run_with_cache(tokens, return_type="logits")
        edited_last = edited_cache["blocks.0.hook_resid_pre"][0, -1].detach()
        expected = baseline_last + (0.25 * baseline_last)
        self.assertTrue(torch.allclose(edited_last, expected, atol=1e-5, rtol=1e-4))

        runtime_state.tick_ttl()
        runtime_state.tick_ttl()
        self.assertNotIn("e_resid_tlens", runtime_state.hooks)

    def test_hooked_transformer_adapter_reads_cache_refs_from_position_axis(self):
        model = self._make_model()
        runtime_state = HookedTransformerRuntimeState(model, seed=19)
        adapter = HookedTransformerAdapter(model)
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        _logits, cache = runtime_state.run_with_cache(tokens, return_type="logits")
        packet = parse_observation_packet(self._runtime_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)

        ref = {
            "scope": "runtime",
            "worker": "os_0",
            "tensor": "v_cache",
            "layer": 0,
            "head": 0,
            "token": {"mode": "index", "value": 1},
        }
        selected = adapter.read_ref(ref, ctx)
        expected = cache["blocks.0.attn.hook_v"][0, 1, 0].detach()
        self.assertTrue(torch.allclose(selected, expected, atol=1e-6, rtol=1e-5))

    def test_hooked_transformer_runtime_kv_mix_uses_cache_token_positions_not_head_axis(self):
        model = self._make_model()
        runtime_state = HookedTransformerRuntimeState(model, seed=23)
        adapter = HookedTransformerAdapter(model)
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        _logits, cache = runtime_state.run_with_cache(tokens, return_type="logits")
        baseline_last = cache["blocks.0.attn.hook_v"][0, -1, 0].detach().clone()
        source_token = cache["blocks.0.attn.hook_v"][0, 1, 0].detach().clone()
        runtime_state.snapshot_last_cache("paired_baseline")

        packet_dict = self._runtime_packet()
        packet_dict["surface_catalog"].append(
            {
                "surface_id": "s_v_l0_h0_last",
                "target": {
                    "kind": "cache",
                    "worker": "os_0",
                    "site": "v_cache",
                    "layer": 0,
                    "head": 0,
                    "token": {"mode": "last"},
                },
                "allow_ops": ["kv_mix"],
                "caps": {
                    "max_alpha": 0.5,
                    "max_ttl_steps": 2,
                    "norm_clip": 5.0,
                    "step_size": 0.5,
                    "revertible_only": True,
                },
            }
        )
        packet = parse_observation_packet(packet_dict)
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)

        command = {
            "version": "0.1",
            "decision": "apply",
            "edits": [
                {
                    "id": "e_kv_mix_tlens",
                    "target": {"surface_id": "s_v_l0_h0_last"},
                    "source": {
                        "dtype": "cache_pair",
                        "v": {
                            "ref": {
                                "scope": "runtime",
                                "worker": "os_0",
                                "tensor": "v_cache",
                                "layer": 0,
                                "head": 0,
                                "token": {"mode": "index", "value": 1},
                            }
                        },
                    },
                    "op": {"kind": "kv_mix", "which": "v", "alpha": 0.5},
                    "budget": {"ttl_steps": 1, "norm_clip": 5.0, "step_size": 0.5, "revertible": True},
                }
            ],
        }

        permissive_policy = HarnessPolicy(global_budget=GlobalBudget(max_total_alpha=1.0, max_total_edit_cost=1.0))
        compiled = compile_command(command, packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)
        self.assertIn("e_kv_mix_tlens", runtime_state.hooks)

        _logits2, edited_cache = runtime_state.run_with_cache(tokens, return_type="logits")
        edited_last = edited_cache["blocks.0.attn.hook_v"][0, -1, 0].detach()
        prepared_source = prepare_direction(source_token, alpha=0.5, norm_clip=5.0, step_size=0.5)
        expected = ((1.0 - 0.5) * baseline_last) + (0.5 * prepared_source)
        self.assertTrue(torch.allclose(edited_last, expected, atol=1e-5, rtol=1e-4))

        compiled[0].rollback(ctx)
        self.assertNotIn("e_kv_mix_tlens", runtime_state.hooks)

    def test_hooked_transformer_runtime_attn_rank1_overlay_applies_and_rolls_back(self):
        model = self._make_model()
        runtime_state = HookedTransformerRuntimeState(model, seed=13)
        adapter = HookedTransformerAdapter(model)
        tokens = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)

        _logits, _cache = runtime_state.run_with_cache(tokens, return_type="logits")
        runtime_state.snapshot_last_cache("paired_baseline")
        packet = parse_observation_packet(self._runtime_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        permissive_policy = HarnessPolicy(deny_targets=())

        param = model.blocks[0].attn.W_O
        before = param.detach().clone()
        u = adapter.read_ref(
            {
                "scope": "runtime",
                "worker": "os_0",
                "tensor": "resid_post",
                "layer": 0,
                "token": {"mode": "last"},
            },
            ctx,
        )
        v = adapter.read_ref(
            {
                "scope": "trace",
                "trace_id": "paired_baseline",
                "worker": "os_0",
                "tensor": "resid_post",
                "layer": 0,
                "token": {"mode": "last"},
            },
            ctx,
        )

        compiled = compile_command(_attn_rank1_command(), packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)
        self.assertIn("e_attn_rank1", runtime_state.overlays)

        expected_delta = (0.05 * torch.outer(u, v)).reshape_as(param)
        self.assertTrue(torch.allclose(param.detach(), before + expected_delta, atol=1e-5, rtol=1e-4))

        compiled[0].rollback(ctx)
        self.assertTrue(torch.allclose(param.detach(), before, atol=1e-6, rtol=1e-5))

    def test_hooked_transformer_runtime_mlp_rank1_overlay_bridges_hidden_dim_mismatch(self):
        model = self._make_model()
        runtime_state = HookedTransformerRuntimeState(model, seed=17)
        adapter = HookedTransformerAdapter(model)
        tokens = torch.tensor([[9, 10, 11, 12]], dtype=torch.long)

        _logits, _cache = runtime_state.run_with_cache(tokens, return_type="logits")
        runtime_state.snapshot_last_cache("paired_baseline")
        packet = parse_observation_packet(self._runtime_packet())
        ctx = StepContext(packet=packet, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter)
        permissive_policy = HarnessPolicy(deny_targets=())

        param = model.blocks[0].mlp.W_out
        before = param.detach().clone()
        geometry = adapter._parameter_geometry(adapter.resolve_surface(packet, parse_controller_command(_mlp_rank1_command()).edits[0].target))
        raw_u = adapter.read_ref(
            {
                "scope": "runtime",
                "worker": "os_0",
                "tensor": "resid_post",
                "layer": 0,
                "token": {"mode": "last"},
            },
            ctx,
        )
        raw_v = adapter.read_ref(
            {
                "scope": "trace",
                "trace_id": "paired_baseline",
                "worker": "os_0",
                "tensor": "resid_post",
                "layer": 0,
                "token": {"mode": "last"},
            },
            ctx,
        )
        adapted_u = adapter.rank1_bridge.adapt(raw_u, side="row", geometry=geometry)
        adapted_v = adapter.rank1_bridge.adapt(raw_v, side="col", geometry=geometry)

        compiled = compile_command(_mlp_rank1_command(), packet, ctx, policy=permissive_policy)
        compiled[0].apply(ctx)
        self.assertIn("e_mlp_rank1", runtime_state.overlays)

        expected_delta = (0.05 * torch.outer(adapted_u, adapted_v)).reshape_as(param)
        self.assertTrue(torch.allclose(param.detach(), before + expected_delta, atol=1e-5, rtol=1e-4))

        compiled[0].rollback(ctx)
        self.assertTrue(torch.allclose(param.detach(), before, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
