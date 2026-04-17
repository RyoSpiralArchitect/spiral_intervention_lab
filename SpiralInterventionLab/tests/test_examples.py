import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

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
    _build_parser,
    _configure_torch_default_device_for_worker,
    _infer_tlens_model_ref,
    _resolve_worker_device,
)
from SpiralInterventionLab.runtime.codecs import CharacterCodec, ModelTokenizerCodec
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
            self.assertEqual(payload["controller_selection_source"], "sidecar_tiebreak")
            self.assertTrue(Path(tmpdir, "readout_escape_replay.jsonl").exists())
            self.assertTrue(Path(tmpdir, "readout_escape_replay_summary.json").exists())

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
            self.assertGreaterEqual(payload["controller_selection_event_count"], 1)
            if payload["nonnull_gate_frontier_count"] > 0:
                self.assertGreaterEqual(payload["nonnull_controller_selected_count"], 1)
                self.assertEqual(payload["controller_selected_bundle_key"], payload["gate_report_frontier_bundle_key"])

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
