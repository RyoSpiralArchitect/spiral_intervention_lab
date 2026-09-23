import sys
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from SpiralInterventionLab.examples.worker_fidelity_preflight import checkpoint_file_hashes, compare_logits, main
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


def test_fidelity_accepts_only_distribution_invariant_offset():
    logits = torch.arange(32, dtype=torch.float32, device="cpu") / 8
    assert compare_logits(logits, logits + 2)["passed"]
    assert not compare_logits(logits, logits.flip(0))["passed"]
    assert not compare_logits(logits, logits * float("nan"))["passed"]
    assert not compare_logits(logits, logits[:2])["passed"]


def test_fidelity_rejects_same_reference_and_output_before_writing(tmp_path, monkeypatch):
    output = tmp_path / "reference"
    monkeypatch.setattr(sys, "argv", [
        "worker_fidelity_preflight", "--backend", "tlens", "--worker-model-path", str(tmp_path),
        "--reference-dir", str(output), "--output-dir", str(output / ".." / "reference"),
    ])
    with patch("SpiralInterventionLab.examples.worker_fidelity_preflight.AutoConfig.from_pretrained") as load:
        with pytest.raises(SystemExit):
            main()
    load.assert_not_called()
    assert not output.exists()


def test_checkpoint_identity_changes_when_shard_bytes_change(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights-a")
    first = checkpoint_file_hashes(tmp_path)
    shard.write_bytes(b"weights-b")
    second = checkpoint_file_hashes(tmp_path)
    assert first["model.safetensors.index.json"] == second["model.safetensors.index.json"]
    assert first[shard.name] != second[shard.name]
    (tmp_path / "pytorch_model.bin").write_bytes(b"other-weights")
    assert "pytorch_model.bin" in checkpoint_file_hashes(tmp_path)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_k_projection_matches_pre_or_post_rotary_cache_space(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    worker = object.__new__(HookedTransformerWorkerRuntime)
    vectors = torch.arange(24, dtype=torch.float32, device="cpu").reshape(1, 3, 1, 8)
    calls = []

    def rotate(x, *, past_kv_pos_offset):
        calls.append(tuple(x.shape))
        assert x.device.type == device
        return x + past_kv_pos_offset + 1

    attn = SimpleNamespace(cfg=SimpleNamespace(positional_embedding_type="rotary"),
                           rotary_cos=torch.ones(16, 8, device=device), apply_rotary=rotate)
    worker.model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    with patch.object(worker, "_cache_hook_name", return_value="blocks.0.attn.hook_k"):
        actual = worker._apply_k_rotary_vectors(vectors, layer=0, token_index=2)
        torch.testing.assert_close(actual, vectors)
        assert calls == []
    with patch.object(worker, "_cache_hook_name", return_value="blocks.0.attn.hook_rot_k"):
        actual = worker._apply_k_rotary_vectors(vectors, layer=0, token_index=2)
        torch.testing.assert_close(actual, vectors + 3)
        assert calls == [(1, 3, 1, 8)]


def test_local_llama_preserves_rmsnorm_and_matches_hf_logits():
    transformers = pytest.importorskip("transformers")
    tlens = pytest.importorskip("transformer_lens")
    from SpiralInterventionLab.examples.digit_transform_e2e import _load_local_hooked_transformer_from_hf

    with torch.device("cpu"):
        torch.manual_seed(9)
        hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
            max_position_embeddings=64, rms_norm_eps=1e-5, rope_theta=10000,
        )).eval()
        cfg = tlens.HookedTransformerConfig(
            n_layers=2, d_model=32, d_head=8, n_heads=4, n_key_value_heads=2,
            d_mlp=64, d_vocab=128, n_ctx=64, act_fn="silu", gated_mlp=True,
            normalization_type="RMS", final_rms=True, eps=1e-5,
            original_architecture="LlamaForCausalLM",
            positional_embedding_type="rotary", rotary_dim=8,
            rotary_base=10000, rotary_adjacent_pairs=False, device="cpu",
        )
        with patch("transformer_lens.loading_from_pretrained.get_pretrained_model_config", return_value=cfg):
            hooked = _load_local_hooked_transformer_from_hf(
                model_ref="meta-llama/Llama-3.2-3B", hf_model=hf, tokenizer=None,
                device="cpu", dtype="float32", first_n_layers=None, move_to_device=True,
                local_files_only=True, trust_remote_code=False,
            ).eval()
        tokens = torch.tensor([[3, 8, 16, 27]])
        with torch.no_grad():
            actual = hooked(tokens)
            expected = hf(tokens).logits
        assert hooked.cfg.normalization_type == "RMS"
        assert hooked.cfg.n_layers == 2
        torch.testing.assert_close(hooked.blocks[0].ln1.w, hf.model.layers[0].input_layernorm.weight)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
