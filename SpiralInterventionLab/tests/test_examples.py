import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch

from SpiralInterventionLab.controllers.base import ControllerProvider, ControllerProviderRequest, ControllerProviderResponse
from SpiralInterventionLab.examples import (
    create_task_env,
    create_readout_analyzer,
    create_readout_sidecar_analyzer,
    build_default_activation_surface_catalog,
    build_allowed_token_ids_for_constraint,
    build_hooked_transformer_worker_runtime,
    load_worker_model,
    run_readout_escape_replay_harness,
    run_shot_mode_probe_harness,
    run_digit_transform_c1_only_experiment,
    run_digit_transform_experiment,
    run_digit_transform_sweep,
)
from SpiralInterventionLab.examples.digit_transform_e2e import (
    _FrontierReplayControllerClient,
    _build_parser,
    _configure_torch_default_device_for_worker,
    _controller_step_views,
    _diagnostic_evidence_ledger,
    _effective_diagnostic_frontier_summary,
    _focused_bridge_eval_recipe_specs,
    _infer_tlens_model_ref,
    _resolve_worker_device,
)
from SpiralInterventionLab.runtime.codecs import CharacterCodec, ModelTokenizerCodec
from SpiralInterventionLab.runtime.sidecar import (
    ReadoutSidecarCapture,
    ReadoutSidecarSiteCapture,
    normalize_readout_sidecar_hints,
)
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime
from SpiralInterventionLab.tasks import (
    SpiralConstrainedRewriteEnv,
    SpiralDigitCopyEnv,
    SpiralDigitTransformEnv,
    SpiralEntailmentReasoningEnv,
    SpiralSentenceOrderingEnv,
    SpiralStructuredSummaryEnv,
)

HAS_TRANSFORMER_LENS = bool(find_spec("transformer_lens"))
if HAS_TRANSFORMER_LENS:
    from transformer_lens import HookedTransformer, HookedTransformerConfig


class _NoopProvider(ControllerProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-controller"

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        text = '{"version":"0.1","decision":"noop"}' if request.expect_json else ""
        return ControllerProviderResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            raw={"seen_payload": request.payload_text()},
        )


class _FakeTokenizer:
    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        mapping = {0: " ", 1: "1", 2: " 2", 3: "a"}
        return "".join(mapping[int(token_id)] for token_id in token_ids)


class _FakeCodecModel:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def to_tokens(self, text: str, prepend_bos: bool = False):
        return __import__("torch").tensor([[1 if char == "1" else 0 for char in text]], dtype=__import__("torch").long)

    def to_string(self, tensor):
        return ["should-not-leak-list-format"]


class _FakeHFConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", "llama")
        self.hidden_size = kwargs.get("hidden_size", 3072)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 28)
        self.vocab_size = kwargs.get("vocab_size", 128256)


class _StubSemanticCritic:
    mode = "minilm"
    model_name = "stub-minilm"

    def score(self, *, reference_text: str, candidate_text: str) -> float:
        return 0.75


@unittest.skipUnless(HAS_TRANSFORMER_LENS, "transformer_lens is not installed")
class TestExamples(unittest.TestCase):
    def test_model_tokenizer_codec_decodes_without_list_wrapper(self):
        codec = ModelTokenizerCodec(model=_FakeCodecModel())

        self.assertEqual(codec.decode([1, 2]), "1 2")

    def _make_model_and_codec(self):
        alphabet = "".join(chr(i) for i in range(32, 127)) + "\n"
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=32,
            n_heads=4,
            d_head=8,
            d_mlp=64,
            n_ctx=256,
            d_vocab=len(alphabet),
            act_fn="relu",
            device="cpu",
            seed=0,
        )
        model = HookedTransformer(cfg)
        model.eval()
        return model, CharacterCodec(alphabet)

    def test_default_surface_catalog_targets_middle_layer(self):
        model, _codec = self._make_model_and_codec()

        catalog = build_default_activation_surface_catalog(model, worker_id="os_0")

        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0]["target"]["kind"], "activation")
        self.assertEqual(catalog[0]["target"]["site"], "resid_pre")
        self.assertEqual(catalog[0]["target"]["layer"], 1)
        self.assertIn("step_size", catalog[0]["caps"])
        self.assertEqual(catalog[1]["surface_id"], "s_resid_pre_l1_prev")
        self.assertEqual(catalog[1]["target"]["token"], {"mode": "index", "value": -2})

    def test_build_hooked_transformer_worker_runtime_smoke(self):
        model, codec = self._make_model_and_codec()
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        runtime = build_hooked_transformer_worker_runtime(model, env, seed=5, codec=codec)

        prompt = env.reset(5)
        runtime.reset(prompt)
        runtime.step()
        packet = runtime.build_controller_packet()

        self.assertEqual(packet["task_view"]["task_id"], env.task_id)
        self.assertTrue(packet["surface_catalog"])
        self.assertEqual(packet["surface_catalog"][0]["target"]["site"], "resid_pre")
        self.assertIn("step_size", packet["surface_catalog"][0]["caps"])
        self.assertTrue(runtime.allowed_token_ids)

    def test_build_hooked_transformer_worker_runtime_with_structured_reflection(self):
        model, codec = self._make_model_and_codec()
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=5,
            codec=codec,
            controller_reflection_mode="structured",
            controller_memory_window=2,
        )

        runtime.record_controller_memory(
            {
                "hypothesis": "small_rescue",
                "observed_outcome": "harmful",
                "next_change": "prefer noop",
                "confidence": 0.4,
            },
            decision="apply",
        )
        prompt = env.reset(5)
        runtime.reset(prompt)
        runtime.record_controller_memory(
            {
                "hypothesis": "small_rescue",
                "observed_outcome": "harmful",
                "next_change": "prefer noop",
                "confidence": 0.4,
            },
            decision="apply",
        )
        runtime.step()
        packet = runtime.build_controller_packet()

        self.assertIn("controller_memory", packet)
        self.assertEqual(packet["controller_memory"][0]["hypothesis"], "small_rescue")
        self.assertEqual(packet["controller_memory"][0]["decision"], "apply")

    def test_build_hooked_transformer_worker_runtime_with_loop_aware_decoder_control(self):
        model, codec = self._make_model_and_codec()
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=5,
            codec=codec,
            worker_decoder_control_mode="loop_aware",
            worker_loop_rescue_edits_per_run=2,
            worker_loop_rescue_total_alpha=0.12,
        )

        prompt = env.reset(5)
        runtime.reset(prompt)
        runtime.step()
        packet = runtime.build_controller_packet()

        self.assertEqual(runtime.decoder_control_mode, "loop_aware")
        self.assertEqual(packet["telemetry"]["decoder_control_mode"], "loop_aware")
        self.assertIn("decoder_rescue_active", packet["telemetry"])
        self.assertEqual(packet["budget"]["loop_rescue_edits_left_this_run"], 2)

    def test_build_hooked_transformer_worker_runtime_with_loop_aware_constraint_decoder_control(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv()
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
            worker_decoder_control_mode="loop_aware_constraint",
            worker_loop_rescue_edits_per_run=2,
            worker_loop_rescue_total_alpha=0.12,
        )

        prompt = env.reset(7)
        runtime.reset(prompt)
        packet = runtime.build_controller_packet()

        self.assertEqual(runtime.decoder_control_mode, "loop_aware_constraint")
        self.assertEqual(packet["telemetry"]["decoder_control_mode"], "loop_aware_constraint")
        self.assertEqual(packet["telemetry"]["decoder_control_track"], "auxiliary")
        self.assertIn("decoder_constraint_target_count", packet["telemetry"])

    def test_build_hooked_transformer_worker_runtime_with_loop_aware_entity_recall_decoder_control(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv()
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
            worker_decoder_control_mode="loop_aware_entity_recall",
            worker_loop_rescue_edits_per_run=2,
            worker_loop_rescue_total_alpha=0.12,
        )

        prompt = env.reset(7)
        runtime.reset(prompt)
        packet = runtime.build_controller_packet()

        self.assertEqual(runtime.decoder_control_mode, "loop_aware_entity_recall")
        self.assertEqual(packet["telemetry"]["decoder_control_mode"], "loop_aware_entity_recall")
        self.assertEqual(packet["telemetry"]["decoder_control_track"], "auxiliary")
        self.assertIn("decoder_entity_target_count", packet["telemetry"])
        self.assertIn("decoder_entity_prefix_depth", packet["telemetry"])

    def test_build_hooked_transformer_worker_runtime_with_logit_bias_entity_soft_decoder_control(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv()
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
            worker_decoder_control_mode="logit_bias_entity_soft",
            worker_loop_rescue_edits_per_run=2,
            worker_loop_rescue_total_alpha=0.12,
        )

        prompt = env.reset(7)
        runtime.reset(prompt)
        packet = runtime.build_controller_packet()

        self.assertEqual(runtime.decoder_control_mode, "logit_bias_entity_soft")
        self.assertEqual(packet["telemetry"]["decoder_control_mode"], "logit_bias_entity_soft")
        self.assertEqual(packet["telemetry"]["decoder_control_track"], "auxiliary")

    def test_build_hooked_transformer_worker_runtime_wires_readout_sidecar_analyzer(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv()

        def fake_sidecar(_capture):
            return {"analyzer_name": "fake_sidecar"}

        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
            readout_sidecar_analyzer=fake_sidecar,
        )

        self.assertIs(runtime.readout_sidecar_analyzer, fake_sidecar)

    def test_build_hooked_transformer_worker_runtime_wires_readout_analyzer_rerank_mode(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv()

        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
            readout_analyzer_rerank_mode="shadow",
        )

        self.assertEqual(runtime.readout_analyzer_rerank_mode, "shadow")

    def test_create_readout_sidecar_analyzer_supports_heuristic_mode(self):
        analyzer = create_readout_sidecar_analyzer("heuristic")

        self.assertTrue(callable(analyzer))

    def test_create_readout_analyzer_supports_heuristic_mode(self):
        analyzer = create_readout_analyzer("heuristic")

        self.assertTrue(callable(analyzer))

    def test_create_readout_analyzer_supports_sae_scaffold_mode(self):
        analyzer = create_readout_analyzer("sae_scaffold")

        self.assertTrue(callable(analyzer))
        capture = ReadoutSidecarCapture(
            run_id="run",
            episode_id="episode",
            worker_id="os_0",
            step=0,
            control_phase_hint="readout_escape",
            answer_readout_canary={
                "reachable_focus_term": "budget",
                "reachable_focus_rank": 900,
                "target_mass": 0.0,
                "attractor_family_mass": 0.2,
            },
            answer_sites=(
                ReadoutSidecarSiteCapture(
                    role="answer_readout",
                    layer=0,
                    token_selector={"mode": "last"},
                    vector=torch.tensor([1.0, 0.0]),
                ),
            ),
            source_sites=(
                ReadoutSidecarSiteCapture(
                    role="source_span",
                    layer=0,
                    token_selector={"mode": "span", "start": 3, "end": 4},
                    vector=torch.tensor([1.0, 0.0]),
                    span=(3, 4),
                    term="budget",
                    provenance_class="source_body",
                ),
            ),
        )
        hints = normalize_readout_sidecar_hints(analyzer(capture))

        self.assertEqual(hints["feature_backend"], "sae_sidecar")
        self.assertEqual(hints["sae_status"], "scaffold_feature_emitter_no_saelens_runtime")
        self.assertGreaterEqual(len(hints["sae_feature_hints"]), 1)
        self.assertEqual(hints["sae_feature_hints"][0]["operator_family_prior"], "resid_or_readout_boundary")
        self.assertIn(
            "activation_patch_source_to_boundary",
            hints["sae_feature_hints"][0]["operator_family_priors"],
        )

    def test_create_readout_sidecar_analyzer_supports_off_mode(self):
        self.assertIsNone(create_readout_sidecar_analyzer("off"))

    def test_create_readout_analyzer_supports_off_mode(self):
        self.assertIsNone(create_readout_analyzer("off"))

    def test_parser_accepts_readout_analyzer_rerank_mode(self):
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--provider",
                "openai",
                "--controller-model",
                "gpt-5.2",
                "--readout-analyzer-rerank-mode",
                "shadow",
            ]
        )

        self.assertEqual(args.readout_analyzer_rerank_mode, "shadow")

    def test_build_hooked_transformer_worker_runtime_wires_semantic_observer(self):
        model, codec = self._make_model_and_codec()
        env = SpiralConstrainedRewriteEnv(semantic_critic=_StubSemanticCritic())
        runtime = build_hooked_transformer_worker_runtime(
            model,
            env,
            seed=7,
            codec=codec,
        )

        prompt = env.reset(7)
        runtime.reset(prompt)

        self.assertIsNotNone(runtime.observer_check_fn)

    def test_build_allowed_token_ids_for_digit_constraint(self):
        codec = CharacterCodec(" 12a")
        model = type("FakeModel", (), {"cfg": type("Cfg", (), {"d_vocab": 4})()})()

        allowed = build_allowed_token_ids_for_constraint(model, codec=codec, decode_constraint="digits_only")

        self.assertEqual(allowed, (1, 2))

    def test_infer_tlens_model_ref_maps_local_llama32_3b_path(self):
        fake_model = type("FakeHFModel", (), {"config": _FakeHFConfig()})()

        inferred = _infer_tlens_model_ref("/definitely/not/local", fake_model)
        self.assertEqual(inferred, "/definitely/not/local")

        with tempfile.TemporaryDirectory() as tmpdir:
            inferred = _infer_tlens_model_ref(tmpdir, fake_model)

        self.assertEqual(inferred, "meta-llama/Llama-3.2-3B")

    def test_create_task_env_supports_copy_and_transform(self):
        self.assertIsInstance(create_task_env("digit_transform"), SpiralDigitTransformEnv)
        self.assertIsInstance(create_task_env("digit_copy"), SpiralDigitCopyEnv)
        self.assertIsInstance(create_task_env("sentence_ordering"), SpiralSentenceOrderingEnv)
        self.assertIsInstance(create_task_env("entailment_reasoning"), SpiralEntailmentReasoningEnv)
        self.assertIsInstance(create_task_env("constrained_rewrite"), SpiralConstrainedRewriteEnv)
        self.assertIsInstance(create_task_env("structured_summary"), SpiralStructuredSummaryEnv)

    def test_run_digit_transform_experiment_smoke(self):
        model, codec = self._make_model_and_codec()
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "SpiralInterventionLab.examples.digit_transform_e2e.create_controller_provider",
                return_value=_NoopProvider(),
            ):
                result = run_digit_transform_experiment(
                    provider_name="openai",
                    controller_model_name="fake-controller",
                    worker_model_name="tiny-random",
                    seed=7,
                    worker_model=model,
                    task_env=env,
                    codec=codec,
                    log_dir=tmpdir,
                )

            payload = result.to_dict()
            self.assertEqual(payload["controller_provider"], "openai")
            self.assertEqual(payload["task_id"], env.task_id)
            self.assertEqual(payload["controller_reflection_mode"], "off")
            self.assertEqual(payload["semantic_critic_mode"], "off")
            self.assertIsNone(payload["semantic_critic_model_name"])
            self.assertEqual(payload["worker_decoder_control_mode"], "off")

    def test_run_shot_mode_probe_harness_smoke(self):
        class _StubHarnessRuntime:
            def __init__(self):
                self._steps = 0
                self._tool_call_count = 0
                self._surface_catalog_raw = [{"surface_id": "s_resid_pre_l0_last"}]
                self._last_task_feedback = {
                    "required_term_recall": 0.0,
                    "required_term_span_progress": 0.0,
                    "partial_score": 0.45,
                }

            def reset(self, prompt: str) -> None:
                self.prompt = prompt

            def build_controller_packet(self) -> dict[str, object]:
                if self._steps == 0:
                    return {
                        "step": 0,
                        "control_phase_hint": "entity_insertion",
                        "strategy_hints": {
                            "shot_mode_ready": False,
                            "shot_probe_needed": False,
                            "kv_probe_needed": False,
                            "shot_candidate_edits": [],
                            "kv_candidate_edits": [],
                        },
                        "task_feedback": dict(self._last_task_feedback),
                    }
                if self._tool_call_count >= 1:
                    return {
                        "step": 1,
                        "control_phase_hint": "shot_mode",
                        "strategy_hints": {
                            "shot_mode_ready": True,
                            "shot_probe_needed": True,
                            "kv_probe_needed": True,
                            "preferred_kv_surface_id": "s_k_cache_l3_h0_last_promoted",
                            "shot_candidate_edits": [],
                            "kv_candidate_edits": [
                                {
                                    "surface_id": "s_k_cache_l3_h0_last_promoted",
                                    "kind": "kv_mix",
                                    "site": "k_cache",
                                    "recent_probe": {"label": "weak_positive", "score": 1.2},
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
                                    "canary_checked": True,
                                    "canary_pass": True,
                                }
                            ],
                        },
                        "task_feedback": dict(self._last_task_feedback),
                    }
                return {
                    "step": 1,
                    "control_phase_hint": "shot_mode",
                    "strategy_hints": {
                        "shot_mode_ready": True,
                        "shot_probe_needed": True,
                        "kv_probe_needed": True,
                        "preferred_kv_surface_id": "s_k_cache_l3_h0_last_promoted",
                        "shot_candidate_edits": [],
                        "kv_candidate_edits": [
                            {
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
                                "canary_checked": True,
                                "canary_pass": True,
                            },
                            {
                                "surface_id": "s_v_cache_l3_h1_last_promoted",
                                "kind": "kv_mix",
                                "site": "v_cache",
                                "source": {
                                    "dtype": "cache_pair",
                                    "v": {
                                        "ref": {
                                            "scope": "runtime",
                                            "worker": "os_0",
                                            "tensor": "v_cache",
                                            "layer": 3,
                                            "head": 1,
                                            "token": {"mode": "index", "value": 5},
                                        }
                                    },
                                },
                                "op": {"kind": "kv_mix", "alpha": 0.04, "which": "v"},
                                "budget": {"ttl_steps": 1, "norm_clip": 1.0, "step_size": 0.04, "revertible": True},
                                "canary_checked": True,
                                "canary_pass": False,
                            }
                        ],
                    },
                    "task_feedback": dict(self._last_task_feedback),
                }

            def step(self) -> None:
                self._steps += 1

            def done(self) -> bool:
                return False

            def final_text(self) -> str:
                return "EW"

            def request_controller_tools(self, requests, *, source: str = "controller"):
                self._tool_call_count += 1
                return [
                    {
                        "tool": "dry_run_decode",
                        "requested_by": source,
                        "status": "ok",
                        "candidate_edit": dict(requests[0]["candidate_edit"]),
                        "required_term_recall_delta": 0.0,
                        "required_term_span_progress_delta": 0.25,
                        "semantic_progress_delta": 0.02,
                    }
                ]

        env = SpiralConstrainedRewriteEnv()
        stub_runtime = _StubHarnessRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "SpiralInterventionLab.examples.digit_transform_e2e.build_hooked_transformer_worker_runtime",
                return_value=stub_runtime,
            ):
                result = run_shot_mode_probe_harness(
                    worker_model_name="tiny-random",
                    seed=7,
                    worker_model=object(),
                    task_env=env,
                    log_dir=tmpdir,
                    worker_decoder_control_mode="logit_bias_entity_soft",
                    max_steps=4,
                    max_probe_candidates=2,
                )

            payload = result.to_dict()
            self.assertTrue(payload["shot_mode_reached"])
            self.assertEqual(payload["shot_mode_step"], 1)
            self.assertEqual(payload["probe_round_count"], 2)
            self.assertEqual(payload["observation_summary"]["preferred_kv_surface_id"], "s_k_cache_l3_h0_last_promoted")
            self.assertEqual(payload["probe_results"][0]["candidate_edit"]["site"], "k_cache")
            self.assertEqual(payload["probe_results"][0]["required_term_span_progress_delta"], 0.25)
            self.assertEqual(payload["probe_results"][0]["probe_stage"], 1)
            self.assertEqual(payload["probe_results"][1]["probe_stage"], 2)
            self.assertTrue((Path(tmpdir) / "shot_harness_summary.json").exists())

    def test_run_digit_transform_c1_only_experiment_smoke(self):
        model, codec = self._make_model_and_codec()
        env = SpiralDigitCopyEnv(min_digits=4, max_digits=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "SpiralInterventionLab.examples.digit_transform_e2e.create_controller_provider",
                return_value=_NoopProvider(),
            ):
                result = run_digit_transform_c1_only_experiment(
                    provider_name="openai",
                    controller_model_name="fake-controller",
                    worker_model_name="tiny-random",
                    seed=7,
                    worker_model=model,
                    task_env=env,
                    codec=codec,
                    log_dir=tmpdir,
                    controller_reflection_mode="structured",
                    semantic_critic_mode="off",
                    worker_decoder_control_mode="loop_aware",
                    worker_loop_rescue_edits_per_run=2,
                    worker_loop_rescue_total_alpha=0.12,
                )

            payload = result.to_dict()
            self.assertEqual(payload["suite_mode"], "c1_only")
            self.assertEqual(payload["task_id"], env.task_id)
            self.assertEqual(payload["controller_reflection_mode"], "structured")
            self.assertEqual(payload["semantic_critic_mode"], "off")
            self.assertEqual(payload["worker_decoder_control_mode"], "loop_aware")
            self.assertEqual(payload["worker_loop_rescue_edits_per_run"], 2)
            self.assertAlmostEqual(payload["worker_loop_rescue_total_alpha"], 0.12)
            self.assertIsNone(payload["b0"])
            self.assertIsNone(payload["b1"])
            self.assertIn("c1", payload)
            self.assertFalse(Path(tmpdir, "b0.jsonl").exists())
            self.assertFalse(Path(tmpdir, "b1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "c1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "experiment_summary.json").exists())

    def test_run_readout_escape_replay_harness_logs_controller_selection(self):
        model, codec = self._make_model_and_codec()
        class _ShortReplayEnv:
            task_id = "readout_escape_replay"

            def reset(self, seed: int) -> str:
                return "SOURCE: send budget\nANSWER:"

            def score(self, output: str) -> float:
                return 0.0

            def done(self, output: str) -> bool:
                return len(output) >= 2

            def worker_runtime_kwargs(self) -> dict[str, int]:
                return {"max_generated_tokens": 2}

        env = _ShortReplayEnv()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_readout_escape_replay_harness(
                worker_model_name="tiny-random",
                seed=7,
                worker_model=model,
                task_env=env,
                codec=codec,
                log_dir=tmpdir,
            )

            payload = result.to_dict()
            self.assertEqual(payload["suite_mode"], "readout_escape_replay_harness")
            self.assertTrue(payload["readout_escape_seen"])
            self.assertGreaterEqual(payload["controller_selection_event_count"], 1)
            self.assertGreaterEqual(payload["nonnull_gate_frontier_count"], 1)
            self.assertGreaterEqual(payload["nonnull_controller_selected_count"], 1)
            self.assertEqual(payload["gate_report_frontier_bundle_key"], payload["forced_challenger_bundle_key"])
            self.assertEqual(payload["controller_selected_bundle_key"], payload["forced_challenger_bundle_key"])
            self.assertEqual(payload["controller_selection_source"], "forced_diagnostic_apply")
            self.assertEqual(payload["controller_objective_bundle_key"], payload["forced_challenger_bundle_key"])
            self.assertEqual(payload["controller_step_actuator_bundle_key"], payload["forced_challenger_bundle_key"])
            self.assertEqual(payload["controller_plan_mode"], "single_layer")
            self.assertGreaterEqual(payload["controller_shadow_proposal_count"], 1)
            self.assertIn("controller_step_views", payload)
            self.assertTrue(payload["controller_step_views"])
            first_step_view = payload["controller_step_views"][0]
            self.assertIn("provider_attempts", first_step_view)
            self.assertIn("controller_command", first_step_view)
            self.assertIn("controller_selection", first_step_view)
            self.assertGreaterEqual(first_step_view["provider_attempt_count"], 1)
            first_attempt = first_step_view["provider_attempts"][0]
            self.assertIsNotNone(first_attempt["raw_json_object"])
            self.assertIsNotNone(first_attempt["normalized_payload"])
            self.assertEqual(first_attempt["normalization_delta"]["normalization_kind"], "replay_passthrough")
            self.assertTrue(Path(tmpdir, "readout_escape_replay.jsonl").exists())
            self.assertTrue(Path(tmpdir, "readout_escape_replay_summary.json").exists())

    def test_controller_step_views_hoist_positive_operator_deepening_plan(self):
        plan = {
            "kind": "positive_operator_deepening_plan",
            "recipe_family": "readout_steering",
            "next_action": "deepen_local_gap_closer",
            "deepening_axis": "target_top20_gap_closing",
            "reason_code": "positive_memory_local_gap_closer",
            "curiosity_signal": "positive_operator_memory_deepen_local_gap_closer",
        }
        views = _controller_step_views(
            [
                {
                    "event": "controller_observation",
                    "step": 2,
                    "latest_diagnostic_results": [
                        {
                            "diagnostic": "compare_extra_operator_diagnostics",
                            "operator_recipe_expansion_summary": {
                                "positive_operator_deepening_plan": plan,
                            },
                        }
                    ],
                }
            ]
        )

        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["positive_operator_deepening_plan"]["reason_code"], "positive_memory_local_gap_closer")
        self.assertEqual(
            views[0]["controller_observation"]["positive_operator_deepening_plan"]["deepening_axis"],
            "target_top20_gap_closing",
        )
        self.assertEqual(
            views[0]["controller_observation"]["controller_curiosity_signal"],
            "positive_operator_memory_deepen_local_gap_closer",
        )

    def test_diagnostic_evidence_ledger_keeps_readout_gap_closer_rows_visible(self):
        extras = [
            {
                "objective_bundle_key": "kv_pair:budget:source_body:72:73",
                "diagnostic_family": "readout_steering",
                "actuator_class": "self_actuator",
                "recipe_name": f"plain_readout_{idx:02d}",
                "target_mass_delta": 0.0,
                "target_top20_hit_delta": 0,
                "focus_rank_delta": 1,
            }
            for idx in range(60)
        ]
        extras.extend(
            [
                {
                    "objective_bundle_key": "kv_pair:budget:source_body:72:73",
                    "diagnostic_family": "readout_steering",
                    "actuator_class": "self_actuator",
                    "recipe_name": "target_readout_patch_pure_a060_gap",
                    "target_mass_delta": 0.0,
                    "target_top20_hit_delta": 0,
                    "target_top20_threshold_gap": 6.1,
                    "focus_rank_delta": 1,
                    "readout_gap_closer_recipe": True,
                    "readout_gap_closer_axis": "target_top20_gap",
                },
                {
                    "objective_bundle_key": "kv_pair:budget:source_body:72:73",
                    "diagnostic_family": "readout_steering",
                    "actuator_class": "self_actuator",
                    "recipe_name": "target_readout_patch_l005_a060_gap",
                    "target_mass_delta": 0.0,
                    "target_top20_hit_delta": 0,
                    "target_top20_threshold_gap": 6.0,
                    "focus_rank_delta": 1,
                    "readout_gap_closer_recipe": True,
                    "readout_gap_closer_axis": "target_top20_gap",
                },
            ]
        )

        ledger = _diagnostic_evidence_ledger(
            {"extra_operator_diagnostics": extras},
            {"gate_report_frontier_bundle_key": "kv_pair:budget:source_body:72:73"},
        )["diagnostic_evidence_ledger"]

        visible_names = [item.get("recipe_name") for item in ledger[:4]]
        self.assertIn("target_readout_patch_pure_a060_gap", visible_names)
        self.assertIn("target_readout_patch_l005_a060_gap", visible_names)

    def test_run_readout_escape_replay_harness_directscan_logs_controller_selection(self):
        model, codec = self._make_model_and_codec()

        class _ShortReplayEnv:
            task_id = "readout_escape_replay"

            def reset(self, seed: int) -> str:
                return "Keep these terms: send, budget\nSOURCE: send budget now\nANSWER:"

            def score(self, output: str) -> float:
                return 0.0

            def done(self, output: str) -> bool:
                return len(output) >= 2

            def worker_runtime_kwargs(self) -> dict[str, int]:
                return {"max_generated_tokens": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_readout_escape_replay_harness(
                worker_model_name="tiny-random",
                seed=7,
                packet_mode="directscan",
                worker_model=model,
                task_env=_ShortReplayEnv(),
                codec=codec,
                log_dir=tmpdir,
                readout_sidecar_analyzer=create_readout_sidecar_analyzer("heuristic"),
                readout_analyzer_rerank_mode="apply",
            )

            payload = result.to_dict()
            self.assertEqual(payload["suite_mode"], "readout_escape_replay_harness")
            self.assertEqual(payload["packet_mode"], "directscan")
            self.assertTrue(payload["readout_escape_seen"])
            self.assertIn("baseline_span_mean", payload["bridge_eval_recipe_names"])
            self.assertIn("term_centered_pm1_minus_stealer_l025", payload["bridge_eval_recipe_names"])
            self.assertIn("bridge_eval_matrix", payload)
            if payload["bridge_eval_matrix"]:
                first_row = payload["bridge_eval_matrix"][0]
                self.assertIn("candidate_fingerprint", first_row)
                self.assertIn("eval_context_fingerprint", first_row)
                self.assertEqual(payload["bridge_eval_locked_step"], 0)
                self.assertFalse(payload["bridge_eval_context_drift"])
                self.assertEqual(first_row["eval_context_fingerprint"].get("decode_step"), 0)
                self.assertEqual(first_row["eval_context_fingerprint"].get("answer_prefix"), "")
            self.assertIn("bridge_plan_unavailable_reason", payload)
            self.assertIn("bridge_plan_unavailable_objective_reasons", payload)
            self.assertIn("bridge_eval_context_drift", payload)
            self.assertIn("bridge_eval_locked_step", payload)
            self.assertIn("bridge_eval_operator_recipe_mode_diagnostics", payload)
            self.assertTrue(payload["bridge_eval_operator_recipe_mode_diagnostics"])
            first_mode_row = payload["bridge_eval_operator_recipe_mode_diagnostics"][0]
            self.assertIn("diagnosis", first_mode_row)
            self.assertIn("pair_interaction_delta", first_mode_row)
            self.assertIn("modes", first_mode_row)
            self.assertIn("bridge_eval_attention_scale_response_diagnostics", payload)
            self.assertIn("bridge_eval_extra_operator_diagnostics", payload)
            self.assertIn("bridge_eval_poststep_comparison", payload)
            self.assertIn("bridge_eval_candidate_swap_comparison", payload)
            self.assertIn("diagnostic_evidence_ledger", payload)
            self.assertIn("bundle_diagnostic_status", payload)
            self.assertGreaterEqual(payload["diagnostic_evidence_ledger_count"], 0)
            for status in payload["bundle_diagnostic_status"].values():
                self.assertFalse(status.get("operator_certified", False))
                self.assertFalse(status.get("production_operator_certified", False))
                self.assertIn("diagnostic_operator_supported", status)
                self.assertIn("policy_candidate_ready", status)
            if payload["diagnostic_frontier_bundle_key"]:
                self.assertIn(payload["diagnostic_frontier_bundle_key"], payload["bundle_diagnostic_status"])
                self.assertTrue(payload["diagnostic_frontier_next_evidence"])
                self.assertTrue(payload["diagnostic_frontier_request"])
                self.assertIn(
                    payload["diagnostic_frontier_request"],
                    {
                        "operator_diagnostic_replay",
                        "attention_readout_carrier_probe",
                        "attention_head_ablation_on_frontier",
                        "readout_logit_adjacent_probe",
                        "sae_feature_emitter_scan",
                        "compare_extra_operator_diagnostics",
                        "cross_bundle_bridge_search",
                        "activation_patch_candidate_review",
                        "activation_patch_runtime_support_probe",
                        "activation_patch_promotion_gate_review",
                        "activation_patch_production_shadow_replay",
                        "activation_patch_production_trial_gate_review",
                        "none",
                    },
                )
            if payload["bridge_eval_extra_operator_diagnostics"]:
                extra_row = payload["bridge_eval_extra_operator_diagnostics"][0]
                self.assertTrue(extra_row["diagnostic_only"])
                self.assertIn(
                    extra_row["diagnostic_family"],
                    {
                        "resid_source_span",
                        "readout_local_boundary",
                        "attention_head_ablation",
                        "attention_head_scale",
                        "attention_guided_operator",
                        "activation_patch",
                        "logit_adjacent",
                    },
                )
                attention_guided_rows = [
                    row
                    for row in payload["bridge_eval_extra_operator_diagnostics"]
                    if row.get("diagnostic_family") == "attention_guided_operator"
                ]
                for row in attention_guided_rows:
                    self.assertIn("attention_carrier_scale", row)
                    self.assertIn("candidate_fingerprint", row)
                    self.assertIn("eval_context_fingerprint", row)
                activation_patch_rows = [
                    row
                    for row in payload["bridge_eval_extra_operator_diagnostics"]
                    if row.get("diagnostic_family") == "activation_patch"
                ]
                for row in activation_patch_rows:
                    self.assertIn("activation_patch_site", row)
                    self.assertIn("activation_hook_call_count", row)
                    self.assertIn("actual_delta_class", row)
                    self.assertIn("self_delta", row)
                    self.assertIn("cross_delta", row)
                    self.assertIn("alignment_margin", row)
                    self.assertIn("realized_lift_bundle_key", row)
                    self.assertIn("candidate_fingerprint", row)
                    self.assertIn("eval_context_fingerprint", row)
            if payload["bridge_eval_attention_scale_response_diagnostics"]:
                response_row = payload["bridge_eval_attention_scale_response_diagnostics"][0]
                self.assertIn("response_profile", response_row)
                self.assertIn("scale_points", response_row)
                self.assertIn("shadow_actuator", response_row)
                if response_row["shadow_actuator"] is not None:
                    self.assertEqual(response_row["shadow_actuator"]["kind"], "attention_shadow_actuator")
                    self.assertFalse(response_row["shadow_actuator"]["production_apply_allowed"])
                    self.assertIn("counterfactual_delta", response_row["shadow_actuator"])
                    self.assertIn("attention_shadow_actuator_class", response_row["shadow_actuator"])
                    self.assertIn("promotable_to_certification_replay", response_row["shadow_actuator"])
                    self.assertIn("promotion_reason", response_row["shadow_actuator"])
                    self.assertIn("attention_shadow_actuator_class", response_row)
                    self.assertIn("attention_shadow_promotable_to_certification_replay", response_row)
                zero_points = [
                    item
                    for item in response_row["scale_points"]
                    if abs(float(item.get("scale", 1.0) or 0.0)) <= 1e-9
                ]
                if zero_points and float(zero_points[0].get("head_norm_before", 0.0) or 0.0) > 0.0:
                    partial_points = [
                        item
                        for item in response_row["scale_points"]
                        if float(item.get("scale", 0.0) or 0.0) > 0.0
                        and int(item.get("hook_call_count", 0) or 0) > 0
                    ]
                    for item in partial_points:
                        self.assertGreater(float(item.get("head_norm_before", 0.0) or 0.0), 0.0)
            self.assertIsInstance(payload["bridge_eval_locked_step"], (int, type(None)))
            if payload["bridge_plan_recommendation_count"] == 0:
                self.assertTrue(payload["bridge_plan_unavailable_reason"])
            self.assertGreaterEqual(payload["controller_selection_event_count"], 1)
            self.assertIn("controller_step_views", payload)
            self.assertTrue(payload["controller_step_views"])
            if payload["bridge_visible_step_views"]:
                first_bridge_view = payload["bridge_visible_step_views"][0]
                self.assertTrue(first_bridge_view["bridge_visible"])
                self.assertIn("bridge_dual_layer_missing", first_bridge_view)
            if payload["nonnull_gate_frontier_count"] > 0:
                self.assertGreaterEqual(payload["nonnull_controller_selected_count"], 1)
                self.assertEqual(payload["controller_selected_bundle_key"], payload["gate_report_frontier_bundle_key"])

    def test_run_readout_escape_replay_harness_directscan_executes_diagnostic_request_mode(self):
        model, codec = self._make_model_and_codec()

        class _ShortReplayEnv:
            task_id = "readout_escape_replay"

            def reset(self, seed: int) -> str:
                return "Keep these terms: send, budget\nSOURCE: send budget now\nANSWER:"

            def score(self, output: str) -> float:
                return 0.0

            def done(self, output: str) -> bool:
                return len(output) >= 3

            def worker_runtime_kwargs(self) -> dict[str, int]:
                return {"max_generated_tokens": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_readout_escape_replay_harness(
                worker_model_name="tiny-random",
                seed=7,
                packet_mode="directscan",
                controller_replay_mode="diagnostic_request",
                worker_model=model,
                task_env=_ShortReplayEnv(),
                codec=codec,
                log_dir=tmpdir,
                readout_sidecar_analyzer=create_readout_sidecar_analyzer("heuristic"),
                readout_analyzer_rerank_mode="apply",
            )

            payload = result.to_dict()
            self.assertEqual(payload["replay_mode"], "directscan_diagnostic_request")
            self.assertGreaterEqual(payload["diagnostic_request_event_count"], 1)
            self.assertGreaterEqual(payload["diagnostic_result_event_count"], 1)
            self.assertTrue(payload["controller_diagnostic_request_names"])
            self.assertTrue(payload["latest_diagnostic_results"])
            latest = payload["latest_diagnostic_results"][-1]
            self.assertFalse(latest["production_apply_allowed"])
            self.assertFalse(latest.get("operator_certified", False))
            self.assertFalse(latest.get("production_operator_certified", False))
            self.assertIn("diagnostic_operator_supported", latest)
            self.assertIn("policy_candidate_ready", latest)
            if latest["diagnostic"] == "operator_diagnostic_replay":
                self.assertTrue(
                    any(
                        row.get("evidence_kind")
                        in {
                            "operator_mode_decomposition",
                            "attention_guided_operator_certification",
                            "activation_patch_certification",
                        }
                        for row in latest.get("evidence_rows", [])
                    )
                )
            self.assertIn(
                latest["diagnostic"],
                {
                    "operator_diagnostic_replay",
                    "attention_readout_carrier_probe",
                    "attention_head_ablation_on_frontier",
                    "readout_logit_adjacent_probe",
                    "sae_feature_emitter_scan",
                    "compare_extra_operator_diagnostics",
                    "cross_bundle_bridge_search",
                    "activation_patch_candidate_review",
                    "activation_patch_runtime_support_probe",
                    "activation_patch_promotion_gate_review",
                    "activation_patch_production_shadow_replay",
                    "activation_patch_production_trial_gate_review",
                },
            )
            if latest["diagnostic"] == "activation_patch_candidate_review":
                self.assertEqual(latest.get("diagnostic_role"), "activation_patch_candidate_review")
                self.assertFalse(latest["production_apply_allowed"])
                if latest.get("activation_patch_candidate_review") is not None:
                    self.assertIn("blueprint", latest["activation_patch_candidate_review"])
                    self.assertFalse(latest["activation_patch_candidate_review"]["blueprint"]["production_apply_allowed"])
                    self.assertIn("compile_preview_created", latest["activation_patch_candidate_review"])
                    self.assertIn("compile_preview_blocked_reason", latest["activation_patch_candidate_review"])
                    if latest["activation_patch_candidate_review"].get("compile_preview") is not None:
                        self.assertTrue(latest["activation_patch_candidate_review"]["compile_preview_created"])
                        self.assertFalse(latest["activation_patch_candidate_review"]["compile_preview"]["production_apply_allowed"])
                        self.assertEqual(
                            latest["activation_patch_candidate_review"]["compile_preview"]["compile_state"],
                            "preview_only",
                        )
                    else:
                        self.assertFalse(latest["activation_patch_candidate_review"]["compile_preview_created"])
                        self.assertIsNotNone(latest["activation_patch_candidate_review"]["compile_preview_blocked_reason"])
                self.assertIn("activation_patch_compile_preview_created", latest)
            if latest.get("activation_patch_compile_preview_created"):
                self.assertIn(
                    latest["next_evidence_needed"],
                    {
                        "production_activation_patch_operator_support",
                        "activation_patch_promotion_gate_review",
                        "activation_patch_production_shadow_replay",
                        "activation_patch_production_trial_gate_review",
                        "alternate_activation_patch_or_bridge_evidence",
                        "production_policy_review",
                        "production_trial_gate_review",
                        "bounded_production_trial",
                        "production_trial_gate_followup",
                        "post_bridge_exhaustion_recipe_expansion",
                        "readout_steering_deepening",
                    },
                )
                self.assertFalse(latest["production_apply_allowed"])
            if latest["diagnostic"] == "activation_patch_runtime_support_probe":
                self.assertEqual(latest.get("diagnostic_role"), "activation_patch_runtime_support_probe")
                self.assertFalse(latest["production_apply_allowed"])
                self.assertIn("activation_patch_runtime_support_probe", latest)
                self.assertIn("activation_patch_diagnostic_executable_created", latest)
                if latest.get("activation_patch_executable_shadow") is not None:
                    self.assertEqual(
                        latest["activation_patch_executable_shadow"]["compile_state"],
                        "executable_shadow",
                    )
                    self.assertFalse(latest["activation_patch_executable_shadow"]["production_apply_allowed"])
                    self.assertFalse(latest["activation_patch_executable_shadow"]["certified_for_apply"])
            if latest["diagnostic"] == "activation_patch_promotion_gate_review":
                self.assertEqual(latest.get("diagnostic_role"), "activation_patch_promotion_gate_review")
                self.assertFalse(latest["production_apply_allowed"])
                self.assertIn("activation_patch_promotion_gate_review", latest)
                self.assertIn("activation_patch_promotion_gate_passed", latest)
                self.assertIn("activation_patch_production_denial_dossier", latest)
                dossier = latest.get("activation_patch_production_denial_dossier")
                self.assertIsInstance(dossier, dict)
                self.assertFalse(dossier["production_apply_allowed"])
                self.assertFalse(dossier["production_operator_certified"])
                self.assertIn("active_reasons", dossier)
                self.assertIn("axes", dossier)
                self.assertEqual(
                    set(dossier["axes"]),
                    {
                        "ownership_not_live_certified",
                        "safety_not_certified",
                        "context_equivalence_missing",
                        "rollback_contract_missing",
                        "effect_size_too_small",
                    },
                )
                self.assertIn("production_denial_reasons", latest)
                if latest.get("activation_patch_production_apply_candidate") is not None:
                    self.assertTrue(latest["policy_candidate_ready"])
                    self.assertTrue(latest["diagnostic_operator_supported"])
                    self.assertEqual(
                        latest["activation_patch_production_apply_candidate"]["compile_state"],
                        "production_apply_candidate",
                    )
                    self.assertFalse(latest["activation_patch_production_apply_candidate"]["production_apply_allowed"])
                    self.assertFalse(latest["activation_patch_production_apply_candidate"]["certified_for_apply"])
            if latest["diagnostic"] == "activation_patch_production_shadow_replay":
                self.assertEqual(latest.get("diagnostic_role"), "activation_patch_production_shadow_replay")
                self.assertFalse(latest["production_apply_allowed"])
                self.assertFalse(latest.get("production_operator_certified", False))
                self.assertIn("activation_patch_production_shadow_replay", latest)
                self.assertIn("activation_patch_production_shadow_dossier", latest)
                shadow_dossier = latest.get("activation_patch_production_shadow_dossier")
                self.assertIsInstance(shadow_dossier, dict)
                self.assertFalse(shadow_dossier["production_apply_allowed"])
                self.assertFalse(shadow_dossier["production_operator_certified"])
                self.assertIn("active_reasons_after_shadow", shadow_dossier)
                self.assertIn("axes", shadow_dossier)
                self.assertEqual(
                    set(shadow_dossier["axes"]),
                    {
                        "ownership_not_live_certified",
                        "safety_not_certified",
                        "context_equivalence_missing",
                        "rollback_contract_missing",
                        "effect_size_too_small",
                    },
                )
                self.assertIn("production_denial_reasons_after_shadow", latest)
            if latest["diagnostic"] == "activation_patch_production_trial_gate_review":
                self.assertEqual(latest.get("diagnostic_role"), "activation_patch_production_trial_gate_review")
                self.assertFalse(latest["production_apply_allowed"])
                self.assertFalse(latest.get("production_operator_certified", False))
                self.assertFalse(latest.get("operator_certified", False))
                self.assertIn("activation_patch_production_trial_gate_review", latest)
                self.assertIn("activation_patch_production_trial_dossier", latest)
                self.assertIn("production_trial_allowed", latest)
                self.assertIn("production_trial_blocked_reasons", latest)
                trial_dossier = latest.get("activation_patch_production_trial_dossier")
                self.assertIsInstance(trial_dossier, dict)
                self.assertFalse(trial_dossier["production_apply_allowed"])
                self.assertFalse(trial_dossier["production_operator_certified"])
                self.assertIn("trial_blocked_reasons", trial_dossier)
                self.assertIn("contract", trial_dossier)
                trial_contract = latest.get("activation_patch_production_trial_contract")
                if trial_contract is not None:
                    self.assertEqual(trial_contract["ttl_steps"], 1)
                    self.assertLessEqual(float(trial_contract["norm_clip"]), 1.0)
                    self.assertLessEqual(float(trial_contract["max_alpha"]), 0.15)
                    self.assertTrue(trial_contract["revertible"])
                    self.assertTrue(trial_contract["separate_trial_budget"])
                    self.assertFalse(trial_contract["production_apply_allowed"])
                    self.assertFalse(trial_contract["certified_for_apply"])
                if latest.get("production_trial_allowed"):
                    self.assertIn("activation_patch_production_trial_candidate", latest)
                    trial_candidate = latest.get("activation_patch_production_trial_candidate")
                    self.assertIsInstance(trial_candidate, dict)
                    self.assertEqual(trial_candidate["compile_state"], "production_trial_candidate")
                    self.assertEqual(trial_candidate["apply_kind"], "production_trial")
                    self.assertFalse(trial_candidate["production_apply_allowed"])
                    self.assertFalse(trial_candidate["certified_for_apply"])
                    self.assertIn("trial_edit", trial_candidate)
                    self.assertEqual(trial_candidate["trial_edit"]["meta"]["apply_kind"], "production_trial")
                    self.assertFalse(trial_candidate["trial_edit"]["meta"]["production_apply_allowed"])
                    self.assertFalse(trial_candidate["trial_edit"]["meta"]["certified_for_apply"])
            if latest["diagnostic"] == "attention_readout_carrier_probe":
                self.assertEqual(latest.get("diagnostic_role"), "rank_readout_carrier")
                self.assertFalse(latest["production_apply_allowed"])
                if latest.get("attention_shadow_actuator") is not None:
                    self.assertFalse(latest["attention_shadow_actuator"]["production_apply_allowed"])
                    self.assertIn("counterfactual_delta", latest["attention_shadow_actuator"])
                    self.assertIn("attention_shadow_actuator_class", latest["attention_shadow_actuator"])
                    self.assertIn("promotable_to_certification_replay", latest["attention_shadow_actuator"])
                    self.assertIn("attention_shadow_actuator_class", latest)
                    self.assertIn("attention_shadow_promotable_to_certification_replay", latest)
            if latest.get("activation_patch_shadow_actuator") is not None:
                self.assertFalse(latest["activation_patch_shadow_actuator"]["production_apply_allowed"])
                self.assertIn("counterfactual_delta", latest["activation_patch_shadow_actuator"])
                self.assertIn("activation_patch_actuator_class", latest["activation_patch_shadow_actuator"])
                self.assertIn("activation_patch_actuator_class", latest)
                self.assertIn("activation_patch_promotable_to_candidate", latest)
            self.assertTrue(
                any(
                    view["diagnostic_request_count"] >= 1 and view["diagnostic_result_count"] >= 1
                    for view in payload["controller_step_views"]
                )
            )
            self.assertTrue(
                any(
                    view.get("loop_patience_count", 0) >= 1
                    for view in payload["controller_step_views"]
                )
            )
            self.assertTrue(
                any(
                    "loop_patience_budget_left" in row
                    and "next_evidence_transition" in row
                    for view in payload["controller_step_views"]
                    for row in view.get("loop_patience", [])
                )
            )
            if latest.get("activation_patch_compile_preview_created"):
                self.assertTrue(
                    any(
                        row.get("next_evidence_transition") is True
                        and row.get("next_evidence_after")
                        in {
                            "production_activation_patch_operator_support",
                            "activation_patch_runtime_operator_certification",
                            "activation_patch_promotion_gate_review",
                            "production_shadow_replay",
                            "production_policy_review",
                            "production_trial_gate_review",
                            "bounded_production_trial",
                            "production_trial_gate_followup",
                        }
                        for view in payload["controller_step_views"]
                        for row in view.get("loop_patience", [])
                    )
                )
            self.assertTrue(
                any(
                    isinstance(view.get("controller_observation"), dict)
                    and view["controller_observation"].get("latest_diagnostic_result_count", 0) >= 1
                    for view in payload["controller_step_views"]
                )
            )
            self.assertTrue(Path(tmpdir, "readout_escape_replay.jsonl").exists())
            self.assertTrue(Path(tmpdir, "readout_escape_replay_summary.json").exists())
            with Path(tmpdir, "readout_escape_replay.jsonl").open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            trial_apply_rows = [
                row
                for row in rows
                if row.get("event") == "controller_selection"
                and row.get("controller_selection_source") == "production_trial_apply"
            ]
            if latest["diagnostic"] == "activation_patch_production_trial_gate_review" and latest.get(
                "production_trial_allowed"
            ):
                self.assertTrue(trial_apply_rows)
                trial_apply_row = trial_apply_rows[0]
                self.assertEqual(trial_apply_row["controller_apply_kind"], "production_trial")
                self.assertFalse(trial_apply_row["production_apply_allowed"])
                self.assertFalse(trial_apply_row["certified_for_apply"])
                self.assertTrue(
                    any(row.get("event") == "production_trial_ttl_deferred" for row in rows)
                )
                trial_effect_rows = [
                    row
                    for row in rows
                    if row.get("event") == "controller_effect"
                    and row.get("apply_kind") == "production_trial"
                ]
                for row in trial_effect_rows:
                    self.assertFalse(row["production_apply_allowed"])
                    self.assertFalse(row["certified_for_apply"])

    def test_frontier_replay_consumes_production_trial_followup_before_runtime_probe(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        recipe = "readout_escape|activation_patch|resid_post|L6|source_span_to_last|blend|a0.150"
        compile_preview = {
            "bundle_key": objective,
            "objective_bundle_key": objective,
            "actuator_bundle_key": objective,
            "recipe_name": "activation_patch_source_span_to_last",
            "operator_recipe_id": recipe,
            "compile_state": "preview_only",
            "production_apply_allowed": False,
            "certified_for_apply": False,
        }
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": recipe,
            "surface_family_key": "activation_patch_trial:s_resid_pre_l6_last",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "unknown",
        }
        base_packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
        }

        pre_followup_packet = {
            **base_packet,
            "latest_diagnostic_results": [
                {
                    "diagnostic": "activation_patch_production_trial_gate_review",
                    "objective_bundle_key": objective,
                    "activation_patch_compile_preview": compile_preview,
                    "production_apply_allowed": False,
                }
            ],
        }
        pre_command = controller._diagnostic_request_command(
            packet=pre_followup_packet,
            strategy_hints=pre_followup_packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )
        self.assertEqual(pre_command["decision"], "noop")
        self.assertEqual(pre_command["meta"]["next_action"], "request_compare_extra_operator_diagnostics")
        self.assertEqual(pre_command["meta"]["diagnostic_request"]["diagnostic"], "compare_extra_operator_diagnostics")
        self.assertEqual(
            pre_command["meta"]["next_evidence_needed"],
            "alternate_operator_recipe_after_trial",
        )

        post_followup_packet = {
            **base_packet,
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "activation_patch_compile_preview": compile_preview,
                    "evidence_rows": [
                        {
                            "evidence_kind": "activation_patch_certification",
                            "status": "blocked",
                            "operator_recipe_id": recipe,
                        }
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": False,
                    "policy_candidate_ready": False,
                }
            ],
        }
        post_command = controller._diagnostic_request_command(
            packet=post_followup_packet,
            strategy_hints=post_followup_packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )
        meta = post_command["meta"]
        self.assertEqual(post_command["decision"], "noop")
        self.assertEqual(meta["next_action"], "noop")
        self.assertEqual(meta["next_evidence_needed"], "alternate_trial_candidate_generator")
        self.assertNotIn("diagnostic_request", meta)
        self.assertTrue(meta["production_trial_followup_consumed"])
        self.assertTrue(meta["activation_patch_pipeline_suppressed_by_production_trial_outcome"])
        self.assertEqual(
            meta["blocked_by"],
            "production_trial_alternate_evidence_pending_candidate_generator",
        )
        self.assertIn("production_trial_alternate_evidence", meta)

    def test_frontier_replay_neutral_trial_requests_confirmation_neighborhood(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        recipe = "readout_escape|activation_patch|resid_pre|L6|source_centered_pm1_to_last|blend|a0.050"
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": recipe,
            "surface_family_key": "activation_patch_trial:s_resid_pre_l6_last",
            "trial_effect_class": "neutral",
            "verdict": "neutral",
            "actuator_class": "unknown",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "activation_patch_production_trial_gate_review",
                    "objective_bundle_key": objective,
                    "production_apply_allowed": False,
                }
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_compare_extra_operator_diagnostics")
        self.assertEqual(meta["next_evidence_needed"], "neutral_followup_confirmation_replay")
        self.assertEqual(meta["blocked_by"], "production_trial_neutral_followup_confirmation")
        self.assertEqual(meta["neutral_followup_neighborhood"], "contrastive_source_local")
        self.assertTrue(meta["production_trial_neutral_followup_confirmation_requested"])
        self.assertEqual(request["diagnostic"], "compare_extra_operator_diagnostics")
        self.assertEqual(request["trial_followup_mode"], "confirmation_neighborhood")
        self.assertEqual(request["neutral_followup_neighborhood"], "contrastive_source_local")
        self.assertEqual(
            meta["controller_memory"]["neutral_followup_neighborhood"],
            "contrastive_source_local",
        )

    def test_frontier_replay_generates_alternate_trial_candidate_from_compare_evidence(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        stealer = "kv_pair:send:source_body:70:71"
        failed_recipe = "readout_escape|activation_patch|resid_post|L6|source_span_to_last|blend|a0.150"
        site_shift_recipe = "readout_escape|activation_patch|resid_pre|L6|source_span_to_last|blend|a0.050"
        source_local_recipe = "readout_escape|activation_patch|resid_pre|L6|source_centered_pm1_to_last|blend|a0.050"
        alternate_recipe = (
            "readout_escape|activation_patch|resid_pre|L6|"
            "source_term_token_minus_stealer_l025_to_last|blend|a0.050"
        )
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": failed_recipe,
            "surface_family_key": "activation_patch_trial:s_resid_post_l6_last",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "unknown",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "evidence_rows": [
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_pre_source_term_token_minus_stealer_l025_to_last_blend_a050",
                            "operator_recipe_id": alternate_recipe,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_term_token_minus_stealer_l025",
                            "activation_patch_patch_mode": "blend",
                            "activation_patch_base_localization": "source_term_token",
                            "activation_patch_contrast_mode": "minus_stealer",
                            "activation_patch_contrast_scale": 0.25,
                            "activation_patch_stealer_bundle_key": stealer,
                            "activation_patch_stealer_term": "send",
                            "target_mass_delta": 0.0003,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 10,
                            "self_delta": 0.8,
                            "cross_delta": 0.1,
                            "alignment_margin": 0.7,
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                        },
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_pre_source_centered_pm1_to_last_blend_a050",
                            "operator_recipe_id": source_local_recipe,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_centered_pm1",
                            "activation_patch_patch_mode": "blend",
                            "target_mass_delta": 0.0002,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 8,
                            "self_delta": 0.6,
                            "cross_delta": 0.2,
                            "alignment_margin": 0.4,
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                        },
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_pre_source_span_to_last_blend_a050",
                            "operator_recipe_id": site_shift_recipe,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_span_mean",
                            "activation_patch_patch_mode": "blend",
                            "target_mass_delta": 0.003,
                            "focus_rank_delta": 45,
                            "self_delta": 1.4,
                            "alignment_margin": 1.0,
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                        },
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_post_source_span_to_last_blend_a050",
                            "operator_recipe_id": failed_recipe.replace("a0.150", "a0.050"),
                            "activation_patch_site": "resid_post",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_span_mean",
                            "activation_patch_patch_mode": "blend",
                            "target_mass_delta": 0.002,
                            "focus_rank_delta": 40,
                            "self_delta": 1.2,
                            "alignment_margin": 0.9,
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                        },
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_post_source_span_to_last_blend_a150",
                            "operator_recipe_id": failed_recipe,
                            "activation_patch_site": "resid_post",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.15,
                            "target_mass_delta": 0.001,
                            "focus_rank_delta": 20,
                            "self_delta": 1.0,
                            "alignment_margin": 0.8,
                        },
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": True,
                    "policy_candidate_ready": False,
                }
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )
        meta = command["meta"]
        request = meta["diagnostic_request"]
        alternate = request["activation_patch_shadow_actuator"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_activation_patch_candidate_review")
        self.assertEqual(meta["next_evidence_needed"], "alternate_activation_patch_candidate_review")
        self.assertEqual(request["diagnostic"], "activation_patch_candidate_review")
        self.assertEqual(alternate["operator_recipe_id"], alternate_recipe)
        self.assertNotEqual(alternate["operator_recipe_id"], failed_recipe)
        self.assertEqual(meta["blocked_by"], "production_trial_alternate_candidate_shadow_review")
        self.assertEqual(meta["production_trial_alternate_candidate"]["operator_recipe_id"], alternate_recipe)
        self.assertFalse(meta["production_trial_alternate_candidate"]["same_failed_recipe_family"])
        self.assertFalse(meta["production_trial_alternate_candidate"]["same_failed_recipe_structural_family"])
        self.assertEqual(
            alternate["activation_patch_source_localization"],
            "source_term_token_minus_stealer_l025",
        )
        self.assertEqual(alternate["activation_patch_stealer_bundle_key"], stealer)
        self.assertEqual(alternate["activation_patch_stealer_term"], "send")

        alternate_review_packet = {
            **packet,
            "latest_diagnostic_results": [
                packet["latest_diagnostic_results"][0],
                {
                    "diagnostic": "activation_patch_candidate_review",
                    "objective_bundle_key": objective,
                    "activation_patch_compile_preview": {
                        "kind": "activation_patch_compile_preview",
                        "objective_bundle_key": objective,
                        "actuator_bundle_key": objective,
                        "objective_term": "budget",
                        "recipe_name": "resid_pre_source_term_token_minus_stealer_l025_to_last_blend_a050",
                        "operator_recipe_id": alternate_recipe,
                        "site": "resid_pre",
                        "layer": 6,
                        "alpha": 0.05,
                        "source_localization": "source_term_token_minus_stealer_l025",
                        "base_localization": "source_term_token",
                        "contrast_mode": "minus_stealer",
                        "contrast_scale": 0.25,
                        "stealer_bundle_key": stealer,
                        "stealer_term": "send",
                        "source": "source_body_span",
                        "target": "answer_boundary_last",
                        "patch_mode": "blend",
                        "compile_state": "preview_only",
                        "actual_delta_class": "target_lift",
                        "production_apply_allowed": False,
                        "certified_for_apply": False,
                    },
                    "production_apply_allowed": False,
                },
            ],
        }
        runtime_command = controller._diagnostic_request_command(
            packet=alternate_review_packet,
            strategy_hints=alternate_review_packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )
        runtime_meta = runtime_command["meta"]
        runtime_request = runtime_meta["diagnostic_request"]
        self.assertEqual(runtime_meta["next_action"], "request_activation_patch_runtime_support_probe")
        self.assertEqual(runtime_request["diagnostic"], "activation_patch_runtime_support_probe")
        self.assertEqual(
            runtime_request["activation_patch_shadow_actuator"]["operator_recipe_id"],
            alternate_recipe,
        )
        self.assertEqual(
            runtime_request["activation_patch_shadow_actuator"]["activation_patch_stealer_bundle_key"],
            stealer,
        )

        latest_promotion = controller._activation_patch_promotion_gate_from_results(
            [
                {
                    "diagnostic": "activation_patch_promotion_gate_review",
                    "activation_patch_promotion_gate_review": {
                        "production_apply_candidate": {
                            "operator_recipe_id": failed_recipe,
                        }
                    },
                },
                {
                    "diagnostic": "activation_patch_promotion_gate_review",
                    "activation_patch_promotion_gate_review": {
                        "production_apply_candidate": {
                            "operator_recipe_id": alternate_recipe,
                        }
                    },
                },
            ]
        )
        self.assertEqual(
            latest_promotion["production_apply_candidate"]["operator_recipe_id"],
            alternate_recipe,
        )

        alternate_runtime_packet = {
            **packet,
            "latest_diagnostic_results": [
                alternate_review_packet["latest_diagnostic_results"][1],
                {
                    "diagnostic": "activation_patch_runtime_support_probe",
                    "objective_bundle_key": objective,
                    "activation_patch_runtime_support_probe": {
                        "runtime_supported_shadow_candidate": True,
                        "executable_shadow": {
                            "operator_recipe_id": alternate_recipe,
                            "objective_bundle_key": objective,
                            "actuator_bundle_key": objective,
                            "objective_term": "budget",
                            "recipe_name": "resid_pre_source_term_token_minus_stealer_l025_to_last_blend_a050",
                            "site": "resid_pre",
                            "layer": 6,
                            "alpha": 0.05,
                            "source_localization": "source_term_token_minus_stealer_l025",
                            "base_localization": "source_term_token",
                            "contrast_mode": "minus_stealer",
                            "contrast_scale": 0.25,
                            "stealer_bundle_key": stealer,
                            "stealer_term": "send",
                            "compile_state": "executable_shadow",
                            "production_apply_allowed": False,
                            "certified_for_apply": False,
                        },
                    },
                    "production_apply_allowed": False,
                },
            ],
        }
        promotion_command = controller._diagnostic_request_command(
            packet=alternate_runtime_packet,
            strategy_hints=alternate_runtime_packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )
        promotion_request = promotion_command["meta"]["diagnostic_request"]
        self.assertEqual(promotion_request["diagnostic"], "activation_patch_promotion_gate_review")
        self.assertEqual(
            promotion_request["activation_patch_shadow_actuator"]["operator_recipe_id"],
            alternate_recipe,
        )

    def test_frontier_replay_surfaces_bridge_alternate_after_failed_trial(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        stealer = "kv_pair:send:source_body:70:71"
        failed_recipe = (
            "readout_escape|activation_patch|resid_pre|L2|"
            "source_term_token_minus_stealer_l050_to_last|blend|a0.050"
        )
        bridge_recipe = (
            "readout_escape|activation_patch|resid_pre|L2|"
            "source_term_token_minus_stealer_l025_to_last|blend|a0.050"
        )
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": failed_recipe,
            "surface_family_key": "activation_patch_trial:s_resid_pre_l2_last",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "activation_patch_candidate_pool": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "actuator_bundle_key": stealer,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "bridge_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_pre_source_term_token_minus_stealer_l025_to_last_blend_a050",
                            "operator_recipe_id": bridge_recipe,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_term_token_minus_stealer_l025",
                            "activation_patch_patch_mode": "blend",
                            "activation_patch_base_localization": "source_term_token",
                            "activation_patch_contrast_mode": "minus_stealer",
                            "activation_patch_contrast_scale": 0.25,
                            "activation_patch_stealer_bundle_key": stealer,
                            "activation_patch_stealer_term": "send",
                            "target_mass_delta": 0.0002,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 3,
                            "self_delta": 2.73,
                            "cross_delta": 2.74,
                            "alignment_margin": -0.01,
                            "objective_term": "budget",
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                        }
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": True,
                    "policy_candidate_ready": False,
                }
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        alternate = request["alternate_trial_candidate"]
        self.assertEqual(request["diagnostic"], "activation_patch_candidate_review")
        self.assertEqual(request["step_actuator_bundle_key"], stealer)
        self.assertEqual(meta["step_actuator_bundle_key"], stealer)
        self.assertEqual(meta["next_evidence_needed"], "bridge_plan_dual_layer_shadow_review")
        self.assertEqual(meta["blocked_by"], "production_trial_alternate_candidate_shadow_review")
        self.assertTrue(meta["bridge_plan_available"])
        self.assertEqual(meta["bridge_plan_objective_bundle_key"], objective)
        self.assertEqual(meta["bridge_plan_actuator_bundle_key"], stealer)
        self.assertEqual(alternate["activation_patch_actuator_class"], "bridge_actuator")
        self.assertEqual(alternate["objective_bundle_key"], objective)
        self.assertEqual(alternate["actuator_bundle_key"], stealer)
        self.assertFalse(alternate["promotable_to_candidate_compiler"])
        self.assertTrue(alternate["requires_bridge_plan"])
        self.assertEqual(alternate["alternate_candidate_mode"], "bridge_actuator_shadow_review")
        self.assertFalse(alternate["production_apply_allowed"])

    def test_frontier_replay_rejects_wrong_direction_bridge_alternate_after_failed_trial(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        stealer = "kv_pair:send:source_body:70:71"
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": "failed_recipe",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "activation_patch_candidate_pool": [
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "bridge_actuator",
                            "actual_delta_class": "target_lift",
                            "operator_recipe_id": "budget_recipe_that_lifts_send",
                            "realized_lift_bundle_key": stealer,
                            "realized_lift_term": "send",
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.03,
                        }
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": True,
                    "policy_candidate_ready": False,
                }
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        self.assertEqual(command["decision"], "noop")
        request = meta["diagnostic_request"]
        self.assertEqual(meta["next_action"], "request_cross_bundle_bridge_search")
        self.assertEqual(meta["next_evidence_needed"], "direction_correct_bridge_plan_search")
        self.assertEqual(request["diagnostic"], "cross_bundle_bridge_search")
        self.assertEqual(request["production_trial_alternate_evidence"]["alternate_candidate_absent_reason"], "wrong_direction_bridge_only")
        self.assertNotIn("production_trial_alternate_candidate", meta)
        self.assertNotIn("bridge_plan_available", meta)

    def test_frontier_replay_consumes_cross_bundle_bridge_search_shadow(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        actuator = "kv_pair:send:source_body:70:71"
        bridge_recipe = (
            "readout_escape|activation_patch|resid_pre|L2|"
            "send_term_token_to_last|blend|a0.050"
        )
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": "failed_recipe",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        shadow = {
            "kind": "activation_patch_shadow_actuator",
            "decision": "shadow",
            "objective_bundle_key": objective,
            "actuator_bundle_key": actuator,
            "objective_term": "budget",
            "recipe_name": "resid_pre_send_term_token_to_last_blend_a050",
            "operator_recipe_id": bridge_recipe,
            "activation_patch_site": "resid_pre",
            "activation_patch_layer": 2,
            "activation_patch_alpha": 0.05,
            "activation_patch_source_localization": "source_term_token",
            "activation_patch_patch_mode": "blend",
            "activation_patch_actuator_class": "bridge_actuator",
            "actual_delta_class": "target_lift",
            "realized_lift_bundle_key": objective,
            "realized_lift_term": "budget",
            "counterfactual_delta": {
                "target_mass_delta": 0.0002,
                "focus_rank_delta": 4,
                "self_delta": 0.01,
                "cross_delta": 0.04,
                "alignment_margin": -0.03,
            },
            "promotable_to_candidate_compiler": False,
            "promotion_reason": "cross_bundle_bridge_search_direction_correct",
            "requires_bridge_plan": True,
            "production_apply_allowed": False,
            "certified_for_apply": False,
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "activation_patch_candidate_pool": [
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "bridge_actuator",
                            "actual_delta_class": "target_lift",
                            "operator_recipe_id": "budget_recipe_that_lifts_send",
                            "realized_lift_bundle_key": actuator,
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.03,
                        }
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": True,
                    "policy_candidate_ready": False,
                },
                {
                    "diagnostic": "cross_bundle_bridge_search",
                    "objective_bundle_key": objective,
                    "bridge_plan_shadow_actuator": shadow,
                    "cross_bundle_bridge_summary": {
                        "objective_bundle_key": objective,
                        "eligible_bridge_candidate_count": 1,
                        "status": "bridge_found",
                    },
                    "production_apply_allowed": False,
                },
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        alternate = request["alternate_trial_candidate"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_activation_patch_candidate_review")
        self.assertEqual(meta["next_evidence_needed"], "bridge_plan_dual_layer_shadow_review")
        self.assertEqual(request["diagnostic"], "activation_patch_candidate_review")
        self.assertEqual(request["step_actuator_bundle_key"], actuator)
        self.assertEqual(alternate["operator_recipe_id"], bridge_recipe)
        self.assertEqual(alternate["objective_bundle_key"], objective)
        self.assertEqual(alternate["actuator_bundle_key"], actuator)
        self.assertTrue(meta["bridge_plan_available"])
        self.assertEqual(meta["bridge_plan_actuator_bundle_key"], actuator)
        self.assertFalse(alternate["production_apply_allowed"])

    def test_frontier_replay_requests_recipe_expansion_after_bridge_search_exhausted(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        actuator = "kv_pair:send:source_body:70:71"
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": "failed_recipe",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "activation_patch_candidate_pool": [
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "bridge_actuator",
                            "operator_recipe_id": "budget_recipe_that_lifts_send",
                            "realized_lift_bundle_key": actuator,
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                        }
                    ],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": True,
                    "policy_candidate_ready": False,
                },
                {
                    "diagnostic": "cross_bundle_bridge_search",
                    "objective_bundle_key": objective,
                    "status": "no_direction_correct_bridge",
                    "cross_bundle_bridge_summary": {
                        "objective_bundle_key": objective,
                        "matrix_row_count": 9,
                        "eligible_bridge_candidate_count": 0,
                        "direction_correct_bridge_count": 0,
                        "wrong_direction_bridge_count": 8,
                        "status": "no_direction_correct_bridge",
                    },
                    "production_apply_allowed": False,
                },
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_compare_extra_operator_diagnostics")
        self.assertEqual(meta["next_evidence_needed"], "post_bridge_exhaustion_recipe_expansion")
        self.assertTrue(meta["cross_bundle_bridge_search_exhausted"])
        self.assertEqual(meta["operator_recipe_expansion_mode"], "post_bridge_exhaustion")
        self.assertEqual(meta["blocked_by"], "cross_bundle_bridge_search_exhausted")
        self.assertEqual(request["diagnostic"], "compare_extra_operator_diagnostics")
        self.assertEqual(request["operator_recipe_expansion_mode"], "post_bridge_exhaustion")
        self.assertTrue(request["post_bridge_exhaustion_recipe_expansion_requested"])
        self.assertEqual(
            request["cross_bundle_bridge_search_state"]["status"],
            "no_direction_correct_bridge",
        )

    def test_frontier_replay_does_not_repeat_recipe_expansion_after_bridge_search_exhausted(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": "failed_recipe",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "cross_bundle_bridge_search",
                    "objective_bundle_key": objective,
                    "status": "no_direction_correct_bridge",
                    "cross_bundle_bridge_summary": {
                        "objective_bundle_key": objective,
                        "matrix_row_count": 9,
                        "eligible_bridge_candidate_count": 0,
                        "direction_correct_bridge_count": 0,
                        "wrong_direction_bridge_count": 8,
                        "status": "no_direction_correct_bridge",
                    },
                    "production_apply_allowed": False,
                },
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                    "post_bridge_exhaustion_recipe_expansion_requested": True,
                    "activation_patch_candidate_pool": [],
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": False,
                    "policy_candidate_ready": False,
                },
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "noop")
        self.assertNotIn("diagnostic_request", meta)
        self.assertTrue(meta["cross_bundle_bridge_search_exhausted"])
        self.assertTrue(meta["operator_recipe_expansion_consumed"])
        self.assertEqual(meta["blocked_by"], "operator_recipe_expansion_exhausted")
        self.assertEqual(meta["next_evidence_needed"], "operator_recipe_expansion_exhausted")

    def test_frontier_replay_requests_readout_steering_deepening_after_rank_carrier_expansion(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        trial_outcome = {
            "apply_kind": "production_trial",
            "objective_bundle_key": objective,
            "bundle_key": objective,
            "operator_recipe_id": "failed_recipe",
            "trial_effect_class": "regressing",
            "verdict": "harmful",
            "actuator_class": "dead_actuator",
        }
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "recent_effect_summary": {
                "production_trial_outcome_ledger": [trial_outcome],
                "latest_effects": [trial_outcome],
            },
            "latest_diagnostic_results": [
                {
                    "diagnostic": "cross_bundle_bridge_search",
                    "objective_bundle_key": objective,
                    "status": "no_direction_correct_bridge",
                    "cross_bundle_bridge_summary": {
                        "objective_bundle_key": objective,
                        "matrix_row_count": 9,
                        "eligible_bridge_candidate_count": 0,
                        "direction_correct_bridge_count": 0,
                        "wrong_direction_bridge_count": 8,
                        "status": "no_direction_correct_bridge",
                    },
                    "production_apply_allowed": False,
                },
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                    "post_bridge_exhaustion_recipe_expansion_requested": True,
                    "next_evidence_needed": "readout_steering_deepening",
                    "operator_recipe_expansion_summary": {
                        "status": "rank_carrier_family_found",
                        "recommended_next_family": "readout_steering_deepening:readout_steering",
                        "operator_positive_memory": {
                            "readout_steering": {
                                "recipe_family": "readout_steering",
                                "traits": ["ownership_preserving", "target_reachable", "top20_gap_closer_candidate"],
                                "recommended_next_action": "deepen_local_gap_closer",
                                "best_recipe_name": "post_bridge_target_readout_patch_a040",
                                "best_target_top20_threshold_gap": 0.37,
                            }
                        },
                        "positive_memory_family_count": 1,
                        "best_positive_operator_family": "readout_steering",
                        "best_positive_operator_traits": [
                            "ownership_preserving",
                            "target_reachable",
                            "top20_gap_closer_candidate",
                        ],
                        "best_positive_operator_next_action": "deepen_local_gap_closer",
                        "positive_operator_deepening_plan": {
                            "kind": "positive_operator_deepening_plan",
                            "permission": "diagnostic_only",
                            "production_apply_allowed": False,
                            "policy_candidate_ready": False,
                            "recipe_family": "readout_steering",
                            "recipe_name": "post_bridge_target_readout_patch_a040",
                            "next_action": "deepen_local_gap_closer",
                            "suggested_next_evidence": "readout_steering_deepening",
                            "suggested_operator_recipe_expansion_mode": "readout_steering_deepening",
                            "deepening_axis": "target_top20_gap_closing",
                            "reason_code": "positive_memory_local_gap_closer",
                            "curiosity_signal": "positive_operator_memory_deepen_local_gap_closer",
                            "traits": [
                                "ownership_preserving",
                                "target_reachable",
                                "top20_gap_closer_candidate",
                            ],
                            "best_target_top20_threshold_gap": 0.37,
                        },
                        "best_readout_steering_rank_carrier_recipe_family": "readout_steering",
                        "best_readout_steering_rank_carrier_recipe_name": "post_bridge_target_readout_patch_a040",
                        "best_readout_steering_target_top20_threshold_gap": 0.37,
                        "production_apply_allowed": False,
                    },
                    "production_apply_allowed": False,
                    "diagnostic_operator_supported": False,
                    "policy_candidate_ready": False,
                },
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_compare_extra_operator_diagnostics")
        self.assertEqual(meta["next_evidence_needed"], "readout_steering_deepening")
        self.assertEqual(meta["operator_recipe_expansion_mode"], "readout_steering_deepening")
        self.assertTrue(meta["readout_steering_deepening_requested"])
        self.assertEqual(meta["blocked_by"], "readout_steering_rank_carrier_needs_target_lift")
        self.assertEqual(
            meta["readout_steering_deepening_state"]["best_readout_steering_target_top20_threshold_gap"],
            0.37,
        )
        self.assertEqual(meta["best_positive_operator_family"], "readout_steering")
        self.assertIn("top20_gap_closer_candidate", meta["best_positive_operator_traits"])
        self.assertEqual(meta["best_positive_operator_next_action"], "deepen_local_gap_closer")
        self.assertEqual(meta["controller_curiosity_signal"], "positive_operator_memory_deepen_local_gap_closer")
        self.assertEqual(
            meta["positive_operator_deepening_plan"]["reason_code"],
            "positive_memory_local_gap_closer",
        )
        self.assertEqual(
            meta["positive_operator_deepening_plan"]["suggested_operator_recipe_expansion_mode"],
            "readout_steering_deepening",
        )
        self.assertIn("readout_steering", meta["operator_positive_memory"])
        self.assertEqual(request["diagnostic"], "compare_extra_operator_diagnostics")
        self.assertEqual(request["next_evidence_needed"], "readout_steering_deepening")
        self.assertEqual(request["operator_recipe_expansion_mode"], "readout_steering_deepening")
        self.assertTrue(request["readout_steering_deepening_requested"])
        self.assertIn("readout_steering", request["operator_positive_memory"])
        self.assertEqual(
            request["positive_operator_deepening_plan"]["deepening_axis"],
            "target_top20_gap_closing",
        )

    def test_effective_diagnostic_frontier_prefers_post_bridge_exhaustion_loop_state(self):
        objective = "kv_pair:budget:source_body:72:73"
        effective = _effective_diagnostic_frontier_summary(
            bridge_eval_summary={
                "diagnostic_frontier_bundle_key": objective,
                "diagnostic_frontier_request": "activation_patch_candidate_review",
                "diagnostic_frontier_next_evidence": "activation_patch_candidate_compiler_review",
                "diagnostic_frontier_reason_text": "old static frontier request",
            },
            recent_diagnostic_results=[
                {
                    "diagnostic": "cross_bundle_bridge_search",
                    "objective_bundle_key": objective,
                    "status": "no_direction_correct_bridge",
                    "cross_bundle_bridge_summary": {
                        "objective_bundle_key": objective,
                        "matrix_row_count": 9,
                        "eligible_bridge_candidate_count": 0,
                        "direction_correct_bridge_count": 0,
                        "status": "no_direction_correct_bridge",
                    },
                },
                {
                    "diagnostic": "compare_extra_operator_diagnostics",
                    "objective_bundle_key": objective,
                    "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                    "post_bridge_exhaustion_recipe_expansion_requested": True,
                },
            ],
            diagnostic_request_events=[],
        )

        self.assertEqual(effective["bundle_key"], objective)
        self.assertEqual(effective["request"], "compare_extra_operator_diagnostics")
        self.assertEqual(effective["next_evidence"], "post_bridge_exhaustion_recipe_expansion_observed")
        self.assertEqual(effective["state_source"], "diagnostic_loop_result")
        self.assertEqual(effective["loop_state"], "post_bridge_exhaustion_recipe_expansion_observed")
        self.assertTrue(effective["cross_bundle_bridge_search_exhausted"])
        self.assertTrue(effective["post_bridge_exhaustion_recipe_expansion_observed"])

    def test_worker_cross_bundle_bridge_search_builds_direction_correct_shadow(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        actuator = "kv_pair:send:source_body:70:71"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "cross_bundle_bridge_search",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "next_evidence_needed": "direction_correct_bridge_plan_search",
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "bridge_actuator",
                            "actual_delta_class": "readout_gap_movement",
                            "operator_recipe_id": "budget_recipe_that_lifts_send",
                            "realized_lift_bundle_key": actuator,
                            "realized_lift_term": "send",
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.03,
                        },
                        {
                            "bundle_key": actuator,
                            "intended_bundle_key": actuator,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "cross_bound",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "resid_pre_send_term_token_to_last_blend_a050",
                            "operator_recipe_id": "send_recipe_that_lifts_budget",
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.05,
                            "activation_patch_source_localization": "source_term_token",
                            "activation_patch_patch_mode": "blend",
                            "realized_lift_bundle_key": objective,
                            "realized_lift_term": "budget",
                            "target_mass_delta": 0.0002,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 4,
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.03,
                        },
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["diagnostic"], "cross_bundle_bridge_search")
        self.assertEqual(result["diagnostic_role"], "cross_bundle_bridge_search")
        self.assertEqual(result["status"], "bridge_found")
        self.assertFalse(result["production_apply_allowed"])
        summary = result["cross_bundle_bridge_summary"]
        self.assertEqual(summary["eligible_bridge_candidate_count"], 1)
        self.assertEqual(summary["wrong_direction_bridge_count"], 1)
        shadow = result["bridge_plan_shadow_actuator"]
        self.assertEqual(shadow["objective_bundle_key"], objective)
        self.assertEqual(shadow["actuator_bundle_key"], actuator)
        self.assertTrue(shadow["requires_bridge_plan"])
        self.assertFalse(shadow["production_apply_allowed"])

    def test_worker_cross_bundle_bridge_search_uses_post_bridge_exhaustion_next_evidence(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        stealer = "kv_pair:send:source_body:70:71"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "cross_bundle_bridge_search",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "next_evidence_needed": "direction_correct_bridge_plan_search",
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "intended_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "blocked",
                            "actuator_class": "cross_bound",
                            "actual_delta_class": "target_lift",
                            "operator_recipe_id": "budget_recipe_that_lifts_send",
                            "realized_lift_bundle_key": stealer,
                            "realized_lift_term": "send",
                            "self_delta": 0.01,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.03,
                        }
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["diagnostic"], "cross_bundle_bridge_search")
        self.assertEqual(result["status"], "no_direction_correct_bridge")
        self.assertEqual(result["next_evidence_needed"], "post_bridge_exhaustion_recipe_expansion")
        self.assertFalse(result["production_apply_allowed"])
        summary = result["cross_bundle_bridge_summary"]
        self.assertEqual(summary["eligible_bridge_candidate_count"], 0)
        self.assertEqual(summary["wrong_direction_bridge_count"], 1)

    def test_worker_post_bridge_recipe_expansion_summarizes_failure_modes(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        stealer = "kv_pair:send:source_body:70:71"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "compare_extra_operator_diagnostics",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                "post_bridge_exhaustion_recipe_expansion_requested": True,
                "next_evidence_needed": "post_bridge_exhaustion_recipe_expansion",
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "blocked",
                            "actuator_class": "cross_bound",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_resid_pre_term_to_last_blend_a030",
                            "operator_recipe_id": "recipe_wrong_direction",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.03,
                            "activation_patch_source_localization": "source_term_token",
                            "activation_patch_patch_mode": "blend",
                            "realized_lift_bundle_key": stealer,
                            "self_delta": 0.002,
                            "cross_delta": 0.04,
                            "alignment_margin": -0.038,
                            "target_mass_delta": 0.00001,
                            "focus_rank_delta": 0,
                        },
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "observed",
                            "actuator_class": "noisy_or_harmful",
                            "actual_delta_class": "neutral",
                            "recipe_name": "post_bridge_mlp_out_centered_to_last_blend_a030",
                            "operator_recipe_id": "recipe_weak_nonharmful",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "mlp_out",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.03,
                            "activation_patch_source_localization": "source_centered_pm1",
                            "activation_patch_patch_mode": "blend",
                            "realized_lift_bundle_key": objective,
                            "self_delta": 0.012,
                            "cross_delta": 0.004,
                            "alignment_margin": 0.008,
                            "target_mass_delta": 0.0,
                            "focus_rank_delta": 0,
                            "repeat_delta": 0,
                        },
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "blocked",
                            "actuator_class": "collapse_sharpener",
                            "actual_delta_class": "collapse_sharpener",
                            "recipe_name": "post_bridge_resid_post_span_to_last_blend_a030",
                            "operator_recipe_id": "recipe_collapse",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "resid_post",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.03,
                            "activation_patch_source_localization": "source_span_mean",
                            "activation_patch_patch_mode": "blend",
                            "self_delta": 0.0,
                            "cross_delta": 0.0,
                            "alignment_margin": 0.0,
                            "target_mass_delta": -0.0001,
                            "focus_rank_delta": -2,
                            "repeat_delta": 1,
                            "entropy_delta": -0.04,
                            "top1_margin_delta": 0.01,
                        },
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["diagnostic_role"], "operator_recipe_expansion_view")
        self.assertEqual(result["operator_recipe_expansion_mode"], "post_bridge_exhaustion")
        summary = result["operator_recipe_expansion_summary"]
        self.assertEqual(summary["matrix_row_count"], 3)
        self.assertEqual(summary["dedicated_recipe_row_count"], 3)
        self.assertEqual(summary["best_nonharmful_operator_recipe_id"], "recipe_weak_nonharmful")
        self.assertEqual(summary["best_nonharmful_recipe_family"], "mlp_out|source_centered_pm1|blend")
        self.assertEqual(summary["recommended_next_family"], "deepen:mlp_out|source_centered_pm1|blend")
        self.assertFalse(summary["production_apply_allowed"])
        failure_counts = summary["failure_mode_counts"]
        self.assertEqual(failure_counts["wrong_direction"], 1)
        self.assertEqual(failure_counts["weak_nonharmful"], 1)
        self.assertEqual(failure_counts["collapse_sharpener"], 1)
        matrix = result["operator_recipe_expansion_matrix"]
        self.assertEqual(matrix[0]["operator_recipe_id"], "recipe_weak_nonharmful")
        self.assertEqual(matrix[0]["failure_mode"], "weak_nonharmful")

    def test_worker_post_bridge_recipe_expansion_splits_rank_carrier_from_target_actuator(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "compare_extra_operator_diagnostics",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                "post_bridge_exhaustion_recipe_expansion_requested": True,
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_resid_pre_term_to_last_blend_a030",
                            "operator_recipe_id": "recipe_rank_carrier",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "resid_pre",
                            "activation_patch_layer": 2,
                            "activation_patch_alpha": 0.03,
                            "activation_patch_source_localization": "source_term_token",
                            "activation_patch_patch_mode": "blend",
                            "realized_lift_bundle_key": objective,
                            "self_delta": 1.46,
                            "cross_delta": 1.44,
                            "alignment_margin": 0.02,
                            "target_mass_delta": 0.0,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 6,
                        }
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        summary = result["operator_recipe_expansion_summary"]
        self.assertEqual(summary["status"], "rank_carrier_family_found")
        self.assertEqual(summary["failure_mode_counts"]["self_rank_carrier"], 1)
        self.assertEqual(summary["ownership_role_counts"]["self"], 1)
        self.assertEqual(summary["effect_role_counts"]["rank_carrier"], 1)
        self.assertEqual(summary["safety_role_counts"]["neutral"], 1)
        self.assertEqual(summary["best_rank_carrier_recipe_family"], "resid_pre|source_term_token|blend")
        self.assertEqual(summary["recommended_next_family"], "convert_rank_carrier_to_target:resid_pre|source_term_token|blend")
        self.assertFalse(summary["production_apply_allowed"])
        matrix = result["operator_recipe_expansion_matrix"]
        self.assertEqual(matrix[0]["failure_mode"], "self_rank_carrier")
        self.assertEqual(matrix[0]["ownership_role"], "self")
        self.assertEqual(matrix[0]["effect_role"], "rank_carrier")
        self.assertEqual(matrix[0]["safety_role"], "neutral")

    def test_worker_post_bridge_recipe_expansion_reports_readout_steering_rank_gap(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "compare_extra_operator_diagnostics",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                "post_bridge_exhaustion_recipe_expansion_requested": True,
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "operator_probe",
                            "diagnostic_family": "readout_steering",
                            "operator_axis": "readout_steering",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_target_readout_patch_a040",
                            "operator_recipe_id": "readout_recipe",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "recipe_family": "readout_steering",
                            "readout_steering_kind": "target_readout",
                            "realized_lift_bundle_key": objective,
                            "self_delta": 0.15,
                            "cross_delta": 0.0,
                            "alignment_margin": 0.15,
                            "target_mass_delta": 0.0,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 4,
                            "target_piece": " budget",
                            "target_piece_logit_delta": 0.12,
                            "target_piece_prob_delta": 0.00001,
                            "target_rank_after": 42,
                            "target_top20_threshold_gap_baseline": 0.49,
                            "target_top20_threshold_gap": 0.37,
                            "target_top20_threshold_gap_after": 0.37,
                            "target_top20_threshold_gap_delta": -0.12,
                            "target_top20_margin": -0.37,
                            "readout_gap_closer_recipe": True,
                            "readout_gap_closer_axis": "target_top20_gap",
                        }
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["next_evidence_needed"], "readout_steering_deepening")
        summary = result["operator_recipe_expansion_summary"]
        self.assertEqual(summary["status"], "rank_carrier_family_found")
        self.assertEqual(summary["failure_mode_counts"]["self_rank_carrier"], 1)
        self.assertEqual(summary["best_readout_steering_rank_carrier_recipe_family"], "readout_steering")
        self.assertEqual(summary["best_readout_steering_rank_carrier_recipe_name"], "post_bridge_target_readout_patch_a040")
        self.assertEqual(summary["best_readout_steering_target_top20_threshold_gap"], 0.37)
        self.assertEqual(summary["readout_gap_closer_recipe_count"], 1)
        self.assertEqual(summary["readout_gap_probe_recipe_count"], 1)
        self.assertEqual(summary["readout_gap_closer_candidate_count"], 1)
        self.assertEqual(summary["readout_gap_closer_certified_count"], 1)
        self.assertEqual(summary["best_readout_gap_closer_recipe_name"], "post_bridge_target_readout_patch_a040")
        self.assertEqual(summary["best_readout_gap_closer_target_top20_threshold_gap"], 0.37)
        self.assertEqual(summary["best_readout_gap_closer_target_top20_threshold_gap_delta"], -0.12)
        self.assertEqual(summary["recommended_next_family"], "readout_steering_deepening:readout_steering")
        self.assertEqual(summary["best_positive_operator_family"], "readout_steering")
        self.assertIn("ownership_preserving", summary["best_positive_operator_traits"])
        self.assertIn("top20_gap_measured", summary["best_positive_operator_traits"])
        self.assertIn("top20_gap_closer_candidate", summary["best_positive_operator_traits"])
        self.assertIn("top20_gap_closer_certified", summary["best_positive_operator_traits"])
        self.assertEqual(summary["best_positive_operator_next_action"], "deepen_local_gap_closer")
        plan = summary["positive_operator_deepening_plan"]
        self.assertEqual(plan["kind"], "positive_operator_deepening_plan")
        self.assertEqual(plan["permission"], "diagnostic_only")
        self.assertFalse(plan["production_apply_allowed"])
        self.assertEqual(plan["recipe_family"], "readout_steering")
        self.assertEqual(plan["next_action"], "deepen_local_gap_closer")
        self.assertEqual(plan["suggested_next_evidence"], "readout_steering_deepening")
        self.assertEqual(plan["deepening_axis"], "target_top20_gap_closing")
        self.assertEqual(plan["reason_code"], "positive_memory_local_gap_closer")
        self.assertEqual(plan["gap_closer_recipe_name"], "post_bridge_target_readout_patch_a040")
        self.assertEqual(plan["gap_closer_target_top20_threshold_gap"], 0.37)
        self.assertEqual(plan["gap_closer_target_top20_threshold_gap_delta"], -0.12)
        self.assertEqual(plan["ttl_steps"], 2)
        self.assertTrue(plan["stale_after_context_change"])
        memory = summary["operator_positive_memory"]["readout_steering"]
        self.assertEqual(memory["recommended_next_action"], "deepen_local_gap_closer")
        self.assertIn("target_reachable", memory["traits"])
        self.assertIn("top20_gap_closer_certified", memory["traits"])
        self.assertEqual(memory["best_target_top20_threshold_gap"], 0.37)
        self.assertEqual(memory["best_target_top20_threshold_gap_delta"], -0.12)
        self.assertEqual(memory["scope"]["objective_bundle_key"], objective)
        self.assertEqual(memory["scope"]["target_piece"], " budget")
        self.assertEqual(memory["ttl_steps"], 2)
        self.assertFalse(summary["production_apply_allowed"])
        matrix = result["operator_recipe_expansion_matrix"]
        self.assertEqual(matrix[0]["recipe_family"], "readout_steering")
        self.assertEqual(matrix[0]["failure_mode"], "self_rank_carrier")
        self.assertIn("positive_traits", matrix[0])
        self.assertIn("top20_gap_measured", matrix[0]["positive_traits"])
        self.assertIn("top20_gap_closer_candidate", matrix[0]["positive_traits"])
        self.assertIn("top20_gap_closer_certified", matrix[0]["positive_traits"])
        self.assertEqual(matrix[0]["target_piece"], " budget")
        self.assertEqual(matrix[0]["target_rank_after"], 42)
        self.assertEqual(matrix[0]["target_top20_threshold_gap_baseline"], 0.49)
        self.assertEqual(matrix[0]["target_top20_threshold_gap"], 0.37)
        self.assertEqual(matrix[0]["target_top20_threshold_gap_after"], 0.37)
        self.assertEqual(matrix[0]["target_top20_threshold_gap_delta"], -0.12)
        self.assertEqual(matrix[0]["target_top20_margin"], -0.37)
        self.assertTrue(matrix[0]["readout_gap_closer_recipe"])
        self.assertEqual(matrix[0]["readout_gap_closer_axis"], "target_top20_gap")
        self.assertIn("readout_steering", result["operator_positive_memory"])

    def test_default_operator_recipe_specs_include_readout_gap_closer_sweep(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        specs = runtime._default_operator_recipe_specs()
        gap_specs = [
            spec
            for spec in specs
            if isinstance(spec, dict) and spec.get("readout_gap_closer_recipe")
        ]
        names = {str(spec.get("recipe_name")) for spec in gap_specs}
        self.assertIn("target_readout_patch_pure_a060_gap", names)
        self.assertIn("target_readout_patch_l005_a060_gap", names)
        self.assertIn("contrastive_readout_patch_l025_a060_gap", names)
        self.assertTrue(all(spec.get("readout_gap_closer_axis") == "target_top20_gap" for spec in gap_specs))

    def test_worker_post_bridge_recipe_expansion_prioritizes_readout_steering_deepening_when_gap_exists(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "compare_extra_operator_diagnostics",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                "post_bridge_exhaustion_recipe_expansion_requested": True,
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_resid_post_term_to_last_blend_a060",
                            "operator_recipe_id": "activation_rank_recipe",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "resid_post",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.06,
                            "activation_patch_source_localization": "source_term_token",
                            "activation_patch_patch_mode": "blend",
                            "realized_lift_bundle_key": objective,
                            "self_delta": 1.2,
                            "cross_delta": 0.6,
                            "alignment_margin": 0.6,
                            "target_mass_delta": 0.0,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 20,
                        },
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "operator_probe",
                            "diagnostic_family": "readout_steering",
                            "operator_axis": "readout_steering",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_target_readout_patch_a060",
                            "operator_recipe_id": "readout_rank_recipe",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "recipe_family": "readout_steering",
                            "readout_steering_kind": "target_readout",
                            "realized_lift_bundle_key": objective,
                            "self_delta": 0.15,
                            "cross_delta": 0.0,
                            "alignment_margin": 0.15,
                            "target_mass_delta": 0.0,
                            "target_top20_hit_delta": 0,
                            "focus_rank_delta": 4,
                            "target_top20_threshold_gap": 6.1,
                        },
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["next_evidence_needed"], "readout_steering_deepening")
        summary = result["operator_recipe_expansion_summary"]
        self.assertEqual(summary["best_rank_carrier_recipe_family"], "resid_post|source_term_token|blend")
        self.assertEqual(summary["best_readout_steering_rank_carrier_recipe_family"], "readout_steering")
        self.assertEqual(summary["best_readout_steering_target_top20_threshold_gap"], 6.1)
        self.assertEqual(summary["recommended_next_family"], "readout_steering_deepening:readout_steering")
        self.assertIn("readout_steering", summary["operator_positive_memory"])
        self.assertEqual(summary["best_positive_operator_next_action"], "deepen_local_gap_closer")
        self.assertEqual(
            summary["positive_operator_deepening_plan"]["suggested_operator_recipe_expansion_mode"],
            "readout_steering_deepening",
        )

    def test_readout_steering_deepening_does_not_fallback_to_activation_patch_rows(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        runtime._steps = 0
        objective = "kv_pair:budget:source_body:72:73"
        result = runtime._execute_controller_diagnostic_request(
            {
                "diagnostic": "compare_extra_operator_diagnostics",
                "bundle_key": objective,
                "objective_bundle_key": objective,
                "step_actuator_bundle_key": objective,
                "operator_recipe_expansion_mode": "readout_steering_deepening",
                "next_evidence_needed": "readout_steering_deepening",
            },
            source="unit_test",
            packet={
                "strategy_hints": {
                    "diagnostic_frontier_bundle_key": objective,
                    "diagnostic_evidence_ledger": [
                        {
                            "bundle_key": objective,
                            "objective_bundle_key": objective,
                            "evidence_kind": "activation_patch_certification",
                            "status": "supportive",
                            "actuator_class": "self_actuator",
                            "actual_delta_class": "target_lift",
                            "recipe_name": "post_bridge_resid_post_term_to_last_blend_a060",
                            "operator_recipe_id": "activation_rank_recipe",
                            "operator_recipe_expansion_mode": "post_bridge_exhaustion",
                            "post_bridge_exhaustion_recipe": True,
                            "activation_patch_site": "resid_post",
                            "activation_patch_layer": 6,
                            "activation_patch_alpha": 0.06,
                            "target_mass_delta": 0.0002,
                            "target_top20_hit_delta": 0,
                        }
                    ],
                }
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        summary = result["operator_recipe_expansion_summary"]
        self.assertEqual(summary["status"], "no_readout_steering_rows")
        self.assertEqual(summary["matrix_row_count"], 0)
        self.assertEqual(summary["positive_memory_family_count"], 0)
        self.assertEqual(result["operator_recipe_expansion_matrix"], [])

    def test_activation_patch_candidate_review_blocks_rank_carrier_only_shadow(self):
        runtime = object.__new__(HookedTransformerWorkerRuntime)
        bundle_status = {
            "activation_patch_shadow_actuator": {
                "activation_patch_actuator_class": "self_actuator",
                "promotable_to_candidate_compiler": True,
                "objective_bundle_key": "kv_pair:budget:source_body:72:73",
                "actuator_bundle_key": "kv_pair:budget:source_body:72:73",
                "recipe_name": "resid_pre_source_term_token_minus_stealer_l050_to_last_blend_a050",
                "operator_recipe_id": "rank_carrier_recipe",
                "activation_patch_site": "resid_pre",
                "activation_patch_layer": 2,
                "activation_patch_alpha": 0.05,
                "activation_patch_source_localization": "source_term_token_minus_stealer_l050",
                "activation_patch_patch_mode": "blend",
                "actual_delta_class": "target_lift",
                "counterfactual_delta": {
                    "target_mass_delta": -0.000002,
                    "target_top20_hit_delta": 0,
                    "focus_rank_delta": 304,
                    "self_delta": 3.21,
                    "cross_delta": 3.04,
                    "alignment_margin": 0.17,
                },
            }
        }

        result = runtime._activation_patch_candidate_review(bundle_status)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "bridge_plan_or_more_evidence_required")
        self.assertFalse(result["compile_preview_created"])
        self.assertEqual(result["compile_preview_blocked_reason"], "rank_carrier_not_target_actuator")
        self.assertEqual(result["ownership_role"], "self")
        self.assertEqual(result["effect_role"], "rank_carrier")
        self.assertEqual(result["safety_role"], "neutral")
        self.assertEqual(result["blueprint"]["effect_role"], "rank_carrier")
        self.assertEqual(result["ownership_gate"]["effect_role"], "rank_carrier")
        self.assertTrue(result["rank_carrier_only"])
        self.assertFalse(result["target_readout_effect_certified"])
        self.assertFalse(result["target_promotable_to_candidate_compiler"])

    def test_frontier_replay_requests_compare_after_rank_carrier_block(self):
        controller = _FrontierReplayControllerClient(replay_mode="diagnostic_request")
        objective = "kv_pair:budget:source_body:72:73"
        packet = {
            "strategy_hints": {
                "diagnostic_frontier_bundle_key": objective,
                "selected_bundle_key": objective,
            },
            "telemetry": {"diagnostic_call_budget_left": 2},
            "latest_diagnostic_results": [
                {
                    "diagnostic": "activation_patch_candidate_review",
                    "objective_bundle_key": objective,
                    "bundle_key": objective,
                    "activation_patch_compile_preview_created": False,
                    "activation_patch_compile_preview_blocked_reason": "rank_carrier_not_target_actuator",
                    "activation_patch_effect_role": "self_rank_carrier",
                    "activation_patch_promotable_to_candidate": False,
                    "diagnostic_operator_supported": False,
                    "policy_candidate_ready": False,
                    "production_apply_allowed": False,
                }
            ],
        }

        command = controller._diagnostic_request_command(
            packet=packet,
            strategy_hints=packet["strategy_hints"],
            frontier_bundle_key=objective,
            suggested_bundle_key=objective,
        )

        meta = command["meta"]
        request = meta["diagnostic_request"]
        self.assertEqual(command["decision"], "noop")
        self.assertEqual(meta["next_action"], "request_compare_extra_operator_diagnostics")
        self.assertEqual(meta["next_evidence_needed"], "rank_carrier_to_target_conversion")
        self.assertEqual(request["diagnostic"], "compare_extra_operator_diagnostics")
        self.assertEqual(request["activation_patch_compile_preview_blocked_reason"], "rank_carrier_not_target_actuator")

    def test_focused_bridge_eval_recipe_specs_stay_small_and_ownership_oriented(self):
        recipe_names = [item["recipe_name"] for item in _focused_bridge_eval_recipe_specs()]

        self.assertEqual(
            recipe_names,
            [
                "baseline_span_mean",
                "term_token",
                "term_fused",
                "term_centered_pm1",
                "term_centered_pm1_v060_k020",
                "term_centered_pm1_v025_k045",
                "term_centered_pm1_minus_stealer_l025",
                "term_token_minus_stealer_l050",
                "term_centered_pm1_minus_stealer_l050",
                "term_centered_pm1_orthogonal_stealer",
            ],
        )
        centered = next(item for item in _focused_bridge_eval_recipe_specs() if item["recipe_name"] == "term_centered_pm1")
        self.assertIn("kv_v", centered["modes"])
        self.assertIn("kv_k", centered["modes"])
        self.assertIn("kv_pair_asymmetric", centered["modes"])

    def test_run_digit_transform_sweep_smoke(self):
        model, codec = self._make_model_and_codec()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "SpiralInterventionLab.examples.digit_transform_e2e.create_controller_provider",
                return_value=_NoopProvider(),
            ):
                result = run_digit_transform_sweep(
                    provider_name="openai",
                    controller_model_name="fake-controller",
                    worker_model_name="tiny-random",
                    seeds=(3, 4),
                    worker_model=model,
                    codec=codec,
                    log_dir=tmpdir,
                )
            payload = result.to_dict()
            self.assertEqual(payload["summary"]["num_runs"], 2)
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(payload["seeds"], [3, 4])
            self.assertTrue(Path(tmpdir, "seed_3", "b0.jsonl").exists())
            self.assertTrue(Path(tmpdir, "seed_4", "c1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "seed_3", "experiment_summary.json").exists())
            self.assertTrue(Path(tmpdir, "seed_4", "experiment_summary.json").exists())
            self.assertTrue(Path(tmpdir, "sweep_summary.json").exists())

    @patch("SpiralInterventionLab.examples.digit_transform_e2e._load_local_hooked_transformer_from_hf")
    @patch("SpiralInterventionLab.examples.digit_transform_e2e.AutoTokenizer")
    @patch("SpiralInterventionLab.examples.digit_transform_e2e.AutoModelForCausalLM")
    def test_load_worker_model_uses_local_hf_path_offline(
        self,
        auto_model_cls,
        auto_tokenizer_cls,
        load_local_hooked_transformer,
    ):
        local_dir = "/tmp/local-hf-worker"
        auto_model = object()
        tokenizer = object()
        sentinel = object()
        auto_model_cls.from_pretrained.return_value = auto_model
        auto_tokenizer_cls.from_pretrained.return_value = tokenizer
        load_local_hooked_transformer.return_value = sentinel

        model = load_worker_model(
            "gpt2",
            model_path=local_dir,
            tokenizer_path=local_dir,
            hf_offline=True,
            trust_remote_code=True,
            device="cpu",
        )

        self.assertIs(model, sentinel)
        auto_model_cls.from_pretrained.assert_called_once_with(
            local_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        auto_tokenizer_cls.from_pretrained.assert_called_once_with(
            local_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        load_local_hooked_transformer.assert_called_once_with(
            model_ref=local_dir,
            hf_model=auto_model,
            tokenizer=tokenizer,
            device="cpu",
            dtype="float32",
            first_n_layers=None,
            move_to_device=True,
            local_files_only=True,
            trust_remote_code=True,
        )

    def test_resolve_worker_device_promotes_explicit_mps_in_conservative_mode(self):
        with patch("SpiralInterventionLab.examples.digit_transform_e2e.torch.backends.mps.is_available", return_value=True):
            self.assertIsNone(_resolve_worker_device(device=None, mps_mode="auto"))
            self.assertEqual(_resolve_worker_device(device=None, mps_mode="conservative"), "mps")

    @patch("SpiralInterventionLab.examples.digit_transform_e2e.torch.set_default_device")
    @patch("SpiralInterventionLab.examples.digit_transform_e2e.torch.get_default_device", return_value="mps:0")
    def test_configure_torch_default_device_for_worker_relaxes_auto_mps(self, get_default_device, set_default_device):
        changed = _configure_torch_default_device_for_worker(target_device="mps", mps_mode="conservative")

        self.assertTrue(changed)
        get_default_device.assert_called_once_with()
        set_default_device.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
