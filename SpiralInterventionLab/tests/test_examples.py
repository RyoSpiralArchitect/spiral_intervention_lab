import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from SpiralInterventionLab.controllers.base import ControllerProvider, ControllerProviderRequest, ControllerProviderResponse
from SpiralInterventionLab.examples import (
    create_task_env,
    build_default_activation_surface_catalog,
    build_allowed_token_ids_for_constraint,
    build_hooked_transformer_worker_runtime,
    load_worker_model,
    run_digit_transform_c1_only_experiment,
    run_digit_transform_experiment,
    run_digit_transform_sweep,
)
from SpiralInterventionLab.examples.digit_transform_e2e import (
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
            self.assertEqual(payload["worker_loop_rescue_edits_per_run"], 0)
            self.assertIn("b0", payload)
            self.assertIn("b1", payload)
            self.assertIn("c1", payload)
            self.assertEqual(payload["paired_trace_id"], "paired_baseline")
            self.assertTrue(Path(tmpdir, "b0.jsonl").exists())
            self.assertTrue(Path(tmpdir, "b1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "c1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "experiment_summary.json").exists())

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
