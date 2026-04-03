import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from SpiralInterventionLab.controllers.base import ControllerProvider, ControllerProviderRequest, ControllerProviderResponse
from SpiralInterventionLab.examples import (
    build_default_activation_surface_catalog,
    build_hooked_transformer_worker_runtime,
    load_worker_model,
    run_digit_transform_experiment,
)
from SpiralInterventionLab.runtime.codecs import CharacterCodec
from SpiralInterventionLab.tasks import SpiralDigitTransformEnv

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


@unittest.skipUnless(HAS_TRANSFORMER_LENS, "transformer_lens is not installed")
class TestExamples(unittest.TestCase):
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

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["target"]["kind"], "activation")
        self.assertEqual(catalog[0]["target"]["site"], "resid_pre")
        self.assertEqual(catalog[0]["target"]["layer"], 1)

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
            self.assertIn("b0", payload)
            self.assertIn("b1", payload)
            self.assertIn("c1", payload)
            self.assertEqual(payload["paired_trace_id"], "paired_baseline")
            self.assertTrue(Path(tmpdir, "b0.jsonl").exists())
            self.assertTrue(Path(tmpdir, "b1.jsonl").exists())
            self.assertTrue(Path(tmpdir, "c1.jsonl").exists())

    @patch("SpiralInterventionLab.examples.digit_transform_e2e.AutoTokenizer")
    @patch("SpiralInterventionLab.examples.digit_transform_e2e.AutoModelForCausalLM")
    @patch("SpiralInterventionLab.examples.digit_transform_e2e.HookedTransformer")
    def test_load_worker_model_uses_local_hf_path_offline(
        self,
        hooked_transformer_cls,
        auto_model_cls,
        auto_tokenizer_cls,
    ):
        local_dir = "/tmp/local-hf-worker"
        auto_model = object()
        tokenizer = object()
        sentinel = object()
        auto_model_cls.from_pretrained.return_value = auto_model
        auto_tokenizer_cls.from_pretrained.return_value = tokenizer
        hooked_transformer_cls.from_pretrained.return_value = sentinel

        model = load_worker_model(
            "unused-alias",
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
        hooked_transformer_cls.from_pretrained.assert_called_once()
        args, kwargs = hooked_transformer_cls.from_pretrained.call_args
        self.assertEqual(args[0], local_dir)
        self.assertIs(kwargs["hf_model"], auto_model)
        self.assertIs(kwargs["tokenizer"], tokenizer)
        self.assertTrue(kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
