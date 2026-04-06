import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import torch

from SpiralInterventionLab.backends.base import (
    AutoregressiveBackend,
    BackendCapabilities,
    BackendStepResult,
    LocalBackendWorkerRuntime,
)
from SpiralInterventionLab.backends.hf_transformers import HFTransformersBackend
from SpiralInterventionLab.backends.mlx_lm import MLXLMBackend
from SpiralInterventionLab.bridge.controller_clients import (
    ProviderControllerClient,
    ProviderPromptHintController,
    _normalize_controller_payload,
    load_prompt_asset,
)
from SpiralInterventionLab.controllers.base import ControllerProvider, ControllerProviderRequest, ControllerProviderResponse
from SpiralInterventionLab.controllers.providers import OpenAIControllerProvider
from SpiralInterventionLab.runtime.baselines import run_b1
from SpiralInterventionLab.runtime.codecs import CharacterCodec
from SpiralInterventionLab.runtime.schema import parse_observation_packet

HAS_TRANSFORMERS = bool(find_spec("transformers"))
if HAS_TRANSFORMERS:
    from transformers import GPT2Config, GPT2LMHeadModel


class _FakeProvider(ControllerProvider):
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.requests: list[ControllerProviderRequest] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        self.requests.append(request)
        text = self.responses.pop(0) if self.responses else ""
        return ControllerProviderResponse(text=text, provider=self.provider_name, model=self.model_name)


class _FakeLocalBackend(AutoregressiveBackend):
    def __init__(self) -> None:
        self.codec = CharacterCodec(" pabc")
        self.capabilities = BackendCapabilities(backend_name="fake_local", device="cpu", supports_logits=True)
        self.prompt = ""
        self.context_ids: list[int] = []
        self.output_ids: list[int] = []
        self.last_logits: torch.Tensor | None = None

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self.context_ids = self.codec.encode(prompt).tolist()
        self.output_ids = []
        self.last_logits = None

    def step(self) -> BackendStepResult:
        emit_b = "b" in self.codec.decode(self.context_ids)
        token_id = 3 if emit_b else 2
        logits = torch.tensor([0.1, 0.2, 0.8 if not emit_b else 0.1, 0.8 if emit_b else 0.1, 0.0], dtype=torch.float32)
        self.context_ids.append(token_id)
        self.output_ids.append(token_id)
        self.last_logits = logits
        return BackendStepResult(token_id=token_id, token_text=self.decode_tokens([token_id]), logits=logits)

    def append_prompt_hint(self, hint: str) -> bool:
        token_ids = self.codec.encode(hint).tolist()
        if not token_ids:
            return False
        self.context_ids.extend(token_ids)
        return True

    def current_tokens(self) -> torch.Tensor:
        return torch.tensor([self.context_ids], dtype=torch.long)

    def output_token_ids(self) -> list[int]:
        return list(self.output_ids)

    def decode_tokens(self, token_ids):
        return self.codec.decode(token_ids)

    def final_text(self) -> str:
        return self.codec.decode(self.output_ids)

    def last_logits_tensor(self) -> torch.Tensor | None:
        return None if self.last_logits is None else self.last_logits.detach().clone()


class _SingleHintTaskEnv:
    def reset(self, seed: int) -> str:
        self.seed = seed
        return "p"

    def score(self, output: str) -> float:
        return 1.0 if "b" in output else 0.0

    def done(self, output: str) -> bool:
        return len(output) >= 2


class _FakeTokenizer:
    def encode(self, text: str):
        return [ord(char) % 11 for char in text]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().reshape(-1).tolist()
        return "".join(chr(97 + (int(token_id) % 26)) for token_id in token_ids)


class _FakeMLXResponse:
    def __init__(self, token: int, text: str, *, finish_reason=None):
        self.token = token
        self.text = text
        self.logprobs = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        self.finish_reason = finish_reason


class _FakeOpenAIResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": "{\"version\":\"0.1\",\"decision\":\"noop\"}", "usage": {}})()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeOpenAIResponsesAPI()


class TestBackendsAndBridge(unittest.TestCase):
    def test_provider_controller_client_retries_until_valid_json(self):
        provider = _FakeProvider("not json", "```json\n{\"version\":\"0.1\",\"decision\":\"noop\"}\n```")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=2)

        command = client.invoke({"step": 1})
        trace = client.latest_trace()

        self.assertEqual(command.decision, "noop")
        self.assertEqual(len(provider.requests), 2)
        self.assertIn("Previous reply was invalid", provider.requests[-1].effective_system_prompt())
        self.assertIsNotNone(trace)
        self.assertEqual(trace["observation"]["step"], 1)
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertFalse(trace["attempts"][0]["parse_ok"])
        self.assertTrue(trace["attempts"][1]["parse_ok"])
        self.assertEqual(trace["decision"]["decision"], "noop")

    def test_provider_controller_client_backfills_missing_version(self):
        provider = _FakeProvider("{\"decision\":\"noop\"}")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})

        self.assertEqual(command.version, "0.1")
        self.assertEqual(command.decision, "noop")

    def test_provider_controller_client_observation_includes_task_feedback(self):
        provider = _FakeProvider("{\"version\":\"0.1\",\"decision\":\"noop\"}")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        client.invoke(
            {
                "step": 1,
                "task_view": {"task_id": "toy_task", "mode": "redacted", "prompt_hash": "sha256:test"},
                "worker_view": {"generated_tail": "12", "status": "acting"},
                "task_feedback": {"done": False, "partial_score": 0.5, "progress_label": "progressing"},
                "surface_catalog": [],
                "trace_bank": [],
                "active_edits": [],
                "recent_effects": [],
                "recent_effect_summary": {"window_size": 0},
                "telemetry": {},
                "budget": {},
            }
        )
        trace = client.latest_trace()

        self.assertIsNotNone(trace)
        self.assertEqual(trace["observation"]["task_feedback"]["partial_score"], 0.5)
        self.assertEqual(trace["observation"]["task_feedback"]["progress_label"], "progressing")

    def test_provider_controller_client_observation_includes_latest_observer_check(self):
        provider = _FakeProvider("{\"version\":\"0.1\",\"decision\":\"noop\"}")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        client.invoke(
            {
                "step": 1,
                "task_view": {"task_id": "toy_task", "mode": "redacted", "prompt_hash": "sha256:test"},
                "worker_view": {"generated_tail": "12", "status": "acting"},
                "task_feedback": {"done": False, "partial_score": 0.5, "progress_label": "progressing"},
                "latest_observer_check": {
                    "check_type": "semantic_progress",
                    "trigger": "coverage_progress",
                    "verdict": "baseline",
                    "score": 0.77,
                    "coverage_weight": 1.0,
                    "latent_feature_scan": {
                        "prototype_mode": "token_embedding_mean",
                        "surface_count": 2,
                        "group_count": 1,
                        "groups": [
                            {
                                "group": "required_terms",
                                "polarity": "promote",
                                "feature_count": 2,
                                "mean_alignment": 0.18,
                                "top_features": [
                                    {"feature": "Mira", "surface_id": "s_resid_pre_l4_last", "alignment": 0.24}
                                ],
                            }
                        ],
                    },
                },
                "recent_observer_checks": [
                    {
                        "check_type": "semantic_progress",
                        "trigger": "coverage_progress",
                        "verdict": "baseline",
                        "score": 0.77,
                    }
                ],
                "surface_catalog": [],
                "trace_bank": [],
                "active_edits": [],
                "recent_effects": [],
                "recent_effect_summary": {"window_size": 0},
                "telemetry": {"observer_check_count": 1},
                "budget": {},
            }
        )
        trace = client.latest_trace()

        self.assertIsNotNone(trace)
        self.assertEqual(trace["observation"]["latest_observer_check"]["trigger"], "coverage_progress")
        self.assertEqual(trace["observation"]["recent_observer_checks"][0]["score"], 0.77)
        self.assertEqual(
            trace["observation"]["latest_observer_check"]["latent_feature_scan"]["groups"][0]["group"],
            "required_terms",
        )

    def test_provider_controller_client_observation_includes_tool_catalog_and_results(self):
        provider = _FakeProvider("{\"version\":\"0.1\",\"decision\":\"noop\"}")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        client.invoke(
            {
                "step": 1,
                "task_view": {"task_id": "toy_task", "mode": "redacted", "prompt_hash": "sha256:test"},
                "worker_view": {"generated_tail": "12", "status": "acting"},
                "surface_catalog": [],
                "trace_bank": [],
                "active_edits": [],
                "recent_effects": [],
                "recent_effect_summary": {"window_size": 0},
                "telemetry": {"tool_call_count": 2},
                "budget": {},
                "tool_catalog": [
                    {"tool": "tokenize_terms", "available": True, "cost_hint": "cheap", "budget_left": 4},
                    {"tool": "dry_run_decode", "available": True, "cost_hint": "expensive", "budget_left": 4},
                ],
                "latest_tool_results": [
                    {
                        "tool": "tokenize_terms",
                        "status": "ok",
                        "term_count": 2,
                        "terms": [{"term": "Mira"}],
                    },
                    {
                        "tool": "dry_run_decode",
                        "status": "ok",
                        "entropy_delta": -0.1,
                        "required_term_recall_delta": 0.25,
                        "topk_token_diff": [{"piece": "M", "prob_delta": 0.2}],
                    },
                ],
            }
        )
        trace = client.latest_trace()

        self.assertEqual(trace["observation"]["tool_catalog"][0]["tool"], "tokenize_terms")
        self.assertEqual(trace["observation"]["latest_tool_results"][0]["tool"], "tokenize_terms")
        self.assertEqual(trace["observation"]["latest_tool_results"][1]["tool"], "dry_run_decode")
        self.assertEqual(trace["observation"]["latest_tool_results"][1]["topk_token_diff"][0]["piece"], "M")

    def test_provider_controller_client_observation_includes_controller_memory(self):
        provider = _FakeProvider("{\"version\":\"0.1\",\"decision\":\"noop\"}")
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        client.invoke(
            {
                "step": 2,
                "task_view": {"task_id": "toy_task", "mode": "redacted", "prompt_hash": "sha256:test"},
                "worker_view": {"generated_tail": "12", "status": "acting"},
                "task_feedback": {"done": False, "partial_score": 0.5, "progress_label": "progressing"},
                "surface_catalog": [],
                "trace_bank": [],
                "active_edits": [],
                "recent_effects": [],
                "recent_effect_summary": {"window_size": 0},
                "controller_memory": [
                    {
                        "hypothesis": "small_resid_loop_rescue",
                        "micro_rationale": "avoid another blind nudge",
                        "observed_outcome": "harmful",
                        "next_change": "prefer noop",
                        "next_trigger": "loop_relief_without_coverage",
                        "next_action": "request_observer_check",
                        "confidence": 0.74,
                    }
                ],
                "telemetry": {},
                "budget": {},
            }
        )
        trace = client.latest_trace()

        self.assertIsNotNone(trace)
        self.assertEqual(trace["observation"]["controller_memory"][0]["hypothesis"], "small_resid_loop_rescue")
        self.assertEqual(trace["observation"]["controller_memory"][0]["micro_rationale"], "avoid another blind nudge")
        self.assertEqual(trace["observation"]["controller_memory"][0]["observed_outcome"], "harmful")
        self.assertEqual(trace["observation"]["controller_memory"][0]["next_trigger"], "loop_relief_without_coverage")
        self.assertEqual(trace["observation"]["controller_memory"][0]["next_action"], "request_observer_check")

    def test_provider_controller_client_normalizes_common_edit_shorthand(self):
        provider = _FakeProvider(
            "{\"decision\":\"apply\",\"edits\":[{\"edit_id\":\"e1\",\"surface_id\":\"s_resid_l1_last\",\"source\":{\"dtype\":\"vector\",\"expr\":{\"fn\":\"sub\",\"a\":{\"ref\":{\"scope\":\"runtime\",\"worker\":\"os_0\",\"tensor\":\"hidden\",\"layer\":1,\"token\":{\"mode\":\"last\"}}},\"b\":{\"ref\":{\"scope\":\"runtime\",\"worker\":\"os_0\",\"tensor\":\"hidden\",\"layer\":1,\"token\":{\"mode\":\"last\"}}}}},\"op\":{\"kind\":\"resid_add\",\"alpha\":0.1},\"ttl_steps\":1}]}"
        )
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})

        self.assertEqual(command.version, "0.1")
        self.assertEqual(command.decision, "apply")
        self.assertEqual(command.edits[0].id, "e1")
        self.assertEqual(command.edits[0].budget.ttl_steps, 1)

    def test_provider_controller_client_moves_controller_memory_into_meta(self):
        provider = _FakeProvider(
            "{\"version\":\"0.1\",\"decision\":\"noop\",\"controller_memory\":{\"hypothesis\":\"rescue\",\"observed_outcome\":\"harmful\",\"next_change\":\"noop\",\"confidence\":0.5}}"
        )
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})
        trace = client.latest_trace()

        self.assertEqual(command.meta["controller_memory"]["hypothesis"], "rescue")
        self.assertEqual(trace["decision"]["controller_memory"]["observed_outcome"], "harmful")

    def test_provider_controller_client_normalizes_observer_request_and_micro_rationale_into_meta(self):
        provider = _FakeProvider(
            "{\"version\":\"0.1\",\"decision\":\"noop\",\"request_observer_check\":{\"kind\":\"semantic_progress\",\"reason\":\"fresh semantic read after loop relief\"},\"micro_rationale\":\"Loop eased, but coverage is still zero.\"}"
        )
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})
        trace = client.latest_trace()

        self.assertEqual(command.meta["observer_check_request"]["kind"], "semantic_progress")
        self.assertEqual(command.meta["micro_rationale"], "Loop eased, but coverage is still zero.")
        self.assertEqual(trace["decision"]["observer_check_request"]["kind"], "semantic_progress")
        self.assertEqual(trace["decision"]["micro_rationale"], "Loop eased, but coverage is still zero.")

    def test_provider_controller_client_normalizes_tool_requests_into_meta(self):
        provider = _FakeProvider(
            "{\"version\":\"0.1\",\"decision\":\"noop\",\"request_tools\":[{\"tool\":\"tokenize_terms\",\"terms\":[\"Mira\",\"Omar\"]},{\"tool\":\"dry_run_decode\",\"candidate_edit\":{\"surface_id\":\"s_resid_pre_l4_last\",\"kind\":\"resid_add\",\"alpha\":0.04,\"ttl_steps\":1},\"max_new_tokens\":4}]}"
        )
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})
        trace = client.latest_trace()

        self.assertEqual(command.meta["tool_requests"][0]["tool"], "tokenize_terms")
        self.assertEqual(command.meta["tool_requests"][1]["tool"], "dry_run_decode")
        self.assertEqual(trace["decision"]["tool_requests"][0]["term_count"], 2)
        self.assertEqual(trace["decision"]["tool_requests"][1]["surface_id"], "s_resid_pre_l4_last")

    def test_provider_controller_client_normalizes_next_trigger_and_action_into_meta(self):
        provider = _FakeProvider(
            "{\"version\":\"0.1\",\"decision\":\"noop\",\"micro_rationale\":\"wait for a clearer shift\",\"next_trigger\":\"loop relief without coverage\",\"next_action\":\"request observer check\"}"
        )
        client = ProviderControllerClient(provider, system_prompt="sys", max_attempts=1)

        command = client.invoke({"step": 1})
        trace = client.latest_trace()

        self.assertEqual(command.meta["next_trigger"], "loop relief without coverage")
        self.assertEqual(command.meta["next_action"], "request observer check")
        self.assertEqual(trace["decision"]["next_trigger"], "loop relief without coverage")
        self.assertEqual(trace["decision"]["next_action"], "request observer check")

    def test_normalize_controller_payload_keeps_target_surface_id_flat(self):
        normalized = _normalize_controller_payload(
            {
                "version": "0.1",
                "decision": "apply",
                "edits": [
                    {
                        "id": "e1",
                        "target": {"surface_id": "s_resid_l1_last"},
                        "source": {
                            "dtype": "vector",
                            "expr": {
                                "fn": "normalize",
                                "arg": {
                                    "ref": {
                                        "scope": "runtime",
                                        "worker": "os_0",
                                        "tensor": "hidden",
                                        "layer": 1,
                                        "token": {"mode": "last"},
                                    }
                                },
                            },
                        },
                        "op": {"kind": "resid_add", "alpha": 0.1},
                        "budget": {"ttl_steps": 1, "revertible": True},
                    }
                ],
            }
        )

        self.assertEqual(
            normalized["edits"][0]["target"],
            {"surface_id": "s_resid_l1_last"},
        )

    def test_normalize_controller_payload_accepts_project_orthogonal_aliases(self):
        normalized = _normalize_controller_payload(
            {
                "fn": "project_orthogonal",
                "input": {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "hidden", "layer": 1, "token": {"mode": "last"}}},
                "against": {"ref": {"scope": "trace", "trace_id": "paired_baseline", "worker": "os_0", "tensor": "hidden", "layer": 1, "token": {"mode": "last"}}},
            }
        )

        self.assertEqual(normalized["fn"], "project_orthogonal")
        self.assertIn("arg", normalized)
        self.assertIn("basis", normalized)

    def test_normalize_controller_payload_rewrites_trace_scope_aliases(self):
        normalized = _normalize_controller_payload(
            {
                "ref": {
                    "scope": "paired_baseline",
                    "worker": "os_0",
                    "tensor": "hidden",
                    "layer": 1,
                    "token": {"mode": "last"},
                }
            }
        )

        self.assertEqual(normalized["ref"]["scope"], "trace")
        self.assertEqual(normalized["ref"]["trace_id"], "paired_baseline")

    def test_normalize_controller_payload_accepts_scale_alpha_alias(self):
        normalized = _normalize_controller_payload(
            {
                "fn": "scale",
                "alpha": -0.25,
                "arg": {
                    "ref": {
                        "scope": "runtime",
                        "worker": "os_0",
                        "tensor": "hidden",
                        "layer": 1,
                        "token": {"mode": "last"},
                    }
                },
            }
        )

        self.assertEqual(normalized["fn"], "scale")
        self.assertEqual(normalized["by"], -0.25)
        self.assertNotIn("alpha", normalized)

    def test_normalize_controller_payload_rewrites_prev_token_alias(self):
        normalized = _normalize_controller_payload(
            {
                "ref": {
                    "scope": "runtime",
                    "worker": "os_0",
                    "tensor": "hidden",
                    "layer": 1,
                    "token": {"mode": "prev"},
                }
            }
        )

        self.assertEqual(normalized["ref"]["token"]["mode"], "index")
        self.assertEqual(normalized["ref"]["token"]["value"], -2)

    def test_provider_prompt_hint_controller_trims_plain_text(self):
        provider = _FakeProvider("  try the digit near the end  \n")
        controller = ProviderPromptHintController(provider, system_prompt="sys")

        hint = controller.invoke({"step": 2})
        trace = controller.latest_trace()

        self.assertEqual(hint, "try the digit near the end")
        self.assertIsNotNone(trace)
        self.assertEqual(trace["observation"]["step"], 2)
        self.assertEqual(trace["decision"]["advice"], "try the digit near the end")

    def test_controller_prompt_asset_mentions_effect_summary_and_partial_score(self):
        prompt = load_prompt_asset("controller_v01.txt")

        self.assertIn("recent_effect_summary", prompt)
        self.assertIn("hypothesis_stats", prompt)
        self.assertIn("latest_effects", prompt)
        self.assertIn("partial_score", prompt)
        self.assertIn("semantic_progress_score", prompt)
        self.assertIn("latest_observer_check", prompt)
        self.assertIn("recent_observer_checks", prompt)
        self.assertIn("tool_catalog", prompt)
        self.assertIn("latest_tool_results", prompt)
        self.assertIn("recent_tool_results", prompt)
        self.assertIn("latent_feature_scan", prompt)
        self.assertIn("top_feature_hits", prompt)
        self.assertIn("forbidden_phrases", prompt)
        self.assertIn("choose between provided surfaces", prompt)
        self.assertIn("s_resid_pre_l4_last", prompt)
        self.assertIn("latent_scan_prefers_l4_for_required_terms", prompt)
        self.assertIn("observer_check_budget_left", prompt)
        self.assertIn("tool_call_budget_left", prompt)
        self.assertIn("meta.observer_check_request", prompt)
        self.assertIn("meta.tool_requests", prompt)
        self.assertIn("meta.micro_rationale", prompt)
        self.assertIn("required_term_recall", prompt)
        self.assertIn("required_term_span_progress", prompt)
        self.assertIn("forbidden_term_clean", prompt)
        self.assertIn("budget_ok", prompt)
        self.assertIn("mean_progress_delta", prompt)
        self.assertIn("mean_required_term_recall_delta", prompt)
        self.assertIn("mean_required_term_span_progress_delta", prompt)
        self.assertIn("after_is_looping", prompt)
        self.assertIn("after_task_violation_count", prompt)
        self.assertIn("edits_left_this_run", prompt)
        self.assertIn("progress beats stability", prompt.lower())
        self.assertIn("Safe v0 apply subset", prompt)
        self.assertIn('the field name must be "by"', prompt)
        self.assertIn('Do not use "ref.stat" in v0.', prompt)
        self.assertIn('If none of the safe templates fit, choose noop', prompt)
        self.assertIn("controller_memory", prompt)
        self.assertIn("structured reflection", prompt.lower())
        self.assertIn("decoder_rescue_active", prompt)
        self.assertIn("decoder_control_track", prompt)
        self.assertIn("decoder_constraint_target_count", prompt)
        self.assertIn("decoder_entity_prior_active", prompt)
        self.assertIn("decoder_entity_target_count", prompt)
        self.assertIn("decoder_entity_prefix_depth", prompt)
        self.assertIn("decoder_logit_bias_active", prompt)
        self.assertIn("decoder_logit_bias_term_count", prompt)
        self.assertIn("decoder_logit_bias_token_count", prompt)
        self.assertIn("decoder_logit_bias_focus_term_count", prompt)
        self.assertIn("loop_rescue_edits_left_this_run", prompt)
        self.assertIn("loop_rescue_alpha_left_total", prompt)
        self.assertIn("do not treat this as a hard noop gate", prompt)
        self.assertIn("budget_only_loop_rescue", prompt)
        self.assertIn("micro_rationale", prompt)
        self.assertIn("measure whether the local shift improved semantic grounding", prompt)
        self.assertIn("tokenize_terms", prompt)
        self.assertIn("constraint_scorer", prompt)
        self.assertIn("dry_run_decode", prompt)
        self.assertNotIn("include meta.hypothesis and meta.confidence when useful", prompt)

    def test_openai_controller_provider_requests_json_mode_for_json_expected(self):
        client = _FakeOpenAIClient()
        provider = OpenAIControllerProvider(model="gpt-5.2", client=client)

        provider.complete(
            ControllerProviderRequest(
                system_prompt="sys",
                payload={"step": 1},
                expect_json=True,
            )
        )

        self.assertEqual(client.responses.calls[0]["text"]["format"]["type"], "json_object")
        self.assertEqual(client.responses.calls[0]["text"]["verbosity"], "low")
        self.assertTrue(client.responses.calls[0]["input"].startswith("JSON packet:\n"))

    def test_local_backend_worker_runtime_packet_is_schema_shaped(self):
        runtime = LocalBackendWorkerRuntime(
            backend=_FakeLocalBackend(),
            task_id="local_task",
            goal_hint="emit b when useful",
            constraints=("be short",),
            max_generated_tokens=2,
        )
        runtime.reset("p")
        runtime.step()

        packet = runtime.build_controller_packet()
        parsed = parse_observation_packet(packet)

        self.assertEqual(parsed.task_view["task_id"], "local_task")
        self.assertEqual(packet["worker_view"]["generated_tail"], "a")
        self.assertEqual(packet["surface_catalog"], [])
        self.assertEqual(packet["budget"]["edits_left_this_run"], 0)

    def test_run_b1_works_with_local_backend_runtime(self):
        runtime = LocalBackendWorkerRuntime(
            backend=_FakeLocalBackend(),
            task_id="hint_task",
            goal_hint="emit b when useful",
            max_generated_tokens=2,
        )
        controller = ProviderPromptHintController(_FakeProvider("b"), system_prompt="sys")

        result = run_b1(_SingleHintTaskEnv(), runtime, controller, max_hints_per_run=1)

        self.assertEqual(result.output, "ab")
        self.assertEqual(result.score, 1.0)

    def test_mlx_backend_step_can_be_smoked_with_patched_stream_generate(self):
        backend = MLXLMBackend(model=object(), tokenizer=_FakeTokenizer())

        def fake_stream_generate(_model, _tokenizer, prompt, max_tokens, temp, top_p):
            self.assertEqual(prompt, "hello")
            self.assertEqual(max_tokens, 1)
            yield _FakeMLXResponse(4, "x")

        with patch("SpiralInterventionLab.backends.mlx_lm.stream_generate", new=fake_stream_generate):
            backend.reset("hello")
            result = backend.step()

        self.assertEqual(result.token_id, 4)
        self.assertEqual(result.token_text, "x")
        self.assertEqual(backend.output_token_ids(), [4])
        self.assertIsInstance(backend.last_logits_tensor(), torch.Tensor)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class TestHFTransformersBackend(unittest.TestCase):
    def _make_backend(self) -> HFTransformersBackend:
        codec = CharacterCodec(" pabc!?")
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(codec.alphabet),
                n_positions=16,
                n_embd=16,
                n_layer=2,
                n_head=2,
                eos_token_id=1,
            )
        )
        model.eval()
        return HFTransformersBackend(model=model, codec=codec, device="cpu", eos_token_id=1)

    def test_hf_backend_steps_and_accepts_prompt_hints(self):
        backend = self._make_backend()
        backend.reset("p")

        first = backend.step()
        appended = backend.append_prompt_hint(" b")
        second = backend.step()

        self.assertIsInstance(first.logits, torch.Tensor)
        self.assertTrue(appended)
        self.assertEqual(tuple(backend.current_tokens().shape[:1]), (1,))
        self.assertEqual(len(backend.output_token_ids()), 2)
        self.assertIsInstance(second.token_text, str)


if __name__ == "__main__":
    unittest.main()
