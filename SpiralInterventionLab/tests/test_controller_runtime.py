import unittest
from importlib.util import find_spec

import torch

from SpiralInterventionLab.runtime.adapter import BoundSurface, HookedTransformerAdapter, ModelAdapter
from SpiralInterventionLab.runtime.baselines import run_b0, run_b1, run_c1, run_minimal_baseline_suite
from SpiralInterventionLab.runtime.codecs import CharacterCodec
from SpiralInterventionLab.runtime.compiler import StepContext, compile_command
from SpiralInterventionLab.runtime.effects import build_edit_effect
from SpiralInterventionLab.runtime.loop import InMemoryStructuredLogger, run_episode
from SpiralInterventionLab.runtime.overlays import OverlayHandle
from SpiralInterventionLab.runtime.policy import HarnessPolicy, PolicyViolation, validate_command_against_packet
from SpiralInterventionLab.runtime.rank1_bridge import HybridRank1VectorBridge, Rank1Geometry
from SpiralInterventionLab.runtime.tlens_runtime import HookedTransformerRuntimeState
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime
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
            }
        ],
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
            return self._add_to_selected_tokens(act, vec, surface.token_selector, alpha)

        return surface.hook_name, hook_fn

    def make_kv_hook(self, surface, tensor_fn, alpha, which, budget):
        self._last_kv_args = (surface, which, alpha, budget)

        def hook_fn(act, _hook):
            vec = tensor_fn(self._current_step_ctx)
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

        self.assertTrue(torch.allclose(out[0, -1], torch.tensor([0.2, 0.4, 0.6])))
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

    def invoke(self, packet):
        self.calls += 1
        if self.calls == 1:
            return _resid_command()
        return {"version": "0.1", "decision": "noop"}


class _ToyWorkerRuntime:
    def __init__(self, runtime_state):
        self.runtime_state = runtime_state
        self.output = ""
        self.steps = 0

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self.output = "base"
        self.steps = 0

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
        return _make_packet(rollbackable_ids=rollbackable_ids)

    def observe_recent_effects(self) -> None:
        return None

    def tick_ttl(self) -> None:
        return None

    def cleanup_expired(self) -> None:
        return None

    def final_text(self) -> str:
        return self.output


class TestLoopAndEffects(unittest.TestCase):
    def test_build_edit_effect_marks_helpful(self):
        effect = build_edit_effect(
            edit_id="e1",
            surface_id="s1",
            observed_window_steps=1,
            before={"entropy": 2.5, "top1_margin": 0.05, "repetition_score": 0.4},
            after={"entropy": 2.1, "top1_margin": 0.09, "repetition_score": 0.2},
        )
        self.assertEqual(effect["verdict"], "helpful")

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
        self.assertTrue(any(event["event"] == "compiled_edit" for event in logger.events))
        self.assertEqual(logger.events[-1]["event"], "episode_end")


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
        "resid_pre": "blocks.11.hook_resid_pre",
        "resid_post": "blocks.11.hook_resid_post",
        "hidden": "blocks.11.hook_resid_post",
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

    def get_cache(self, scope, trace_id=None):
        if scope == "runtime":
            return self.last_cache
        if scope == "trace":
            return self.trace_caches[trace_id]
        raise KeyError(scope)

    def read_tensor(self, ref, _ctx):
        cache = self.get_cache(ref["scope"], ref.get("trace_id"))
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

    def invoke(self, _packet):
        self.calls += 1
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
        return [_make_packet()["surface_catalog"][0]]

    def _make_worker_runtime(self):
        codec = CharacterCodec("pabc!? ")
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
        self.assertEqual(packet["surface_catalog"][0]["surface_id"], "s_resid_l11_last")

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

    def test_c1_runner_uses_activation_only_edit_path(self):
        worker_runtime = self._make_worker_runtime()
        controller = _ResidEditController()

        result = run_c1(_ThreeStepTaskEnv("c"), worker_runtime, controller)

        self.assertEqual(result.output, "aca")
        self.assertEqual(result.score, 1.0)

    def test_minimal_baseline_suite_runs_b0_b1_c1_and_seeds_paired_trace(self):
        controller = _ResidEditController()
        hint_controller = _PromptHintController("b")

        suite = run_minimal_baseline_suite(
            _ThreeStepTaskEnv("c"),
            make_worker_runtime=self._make_worker_runtime,
            c1_controller=controller,
            b1_controller=hint_controller,
        )

        self.assertEqual(suite.b0.output, "aaa")
        self.assertEqual(suite.b1.output, "abb")
        self.assertEqual(suite.c1.output, "aca")
        self.assertEqual(suite.b0.score, 0.0)
        self.assertEqual(suite.c1.score, 1.0)
        self.assertEqual(suite.paired_trace_id, "paired_baseline")


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

        compiled = compile_command(command, packet, ctx)
        compiled[0].apply(ctx)
        self.assertIn("e_resid_tlens", runtime_state.hooks)

        _logits2, edited_cache = runtime_state.run_with_cache(tokens, return_type="logits")
        edited_last = edited_cache["blocks.0.hook_resid_pre"][0, -1].detach()
        expected = baseline_last + (0.25 * baseline_last)
        self.assertTrue(torch.allclose(edited_last, expected, atol=1e-5, rtol=1e-4))

        runtime_state.tick_ttl()
        runtime_state.tick_ttl()
        self.assertNotIn("e_resid_tlens", runtime_state.hooks)

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
