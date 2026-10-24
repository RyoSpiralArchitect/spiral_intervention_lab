from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.examples import replay_source_clause_trajectory as cli
from SpiralInterventionLab.examples.replay_source_clause_trajectory import main, next_boundary_plan, verify_reference
from SpiralInterventionLab.runtime import clause_actuation, clause_trajectory, diagnostic_budget
from SpiralInterventionLab.runtime.response_probe import state_identity, tensor_identity
from SpiralInterventionLab.tests.test_clause_actuation import make_worker


STAGES = ["resid_pre:11", "resid_post:11"]


def worker_and_records():
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    return worker, clause_trajectory.measure(worker, [1, 3], stages=STAGES)


def test_existing_probes_capture_without_extra_forwards_or_authority():
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    before, budget = state_identity(worker), diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        records = clause_trajectory.measure(worker, [1, 3], stages=STAGES)
    assert simulate.call_count == 16 == sum(len(r["frames"]) for r in records)
    assert all("policy_override" not in call.kwargs for call in simulate.call_args_list)
    assert diagnostic_budget.left(worker) == budget - 4
    assert before == state_identity(worker) and worker._steps == 1
    assert "run_with_cache" not in vars(worker.runtime_state)
    assert not worker._collect_active_edits() and not worker._frozen_diagnostic_candidates
    summary = clause_trajectory.summarize(records)
    assert summary["status"] == "ok" and not summary["production_apply_allowed"]
    surface = summary["surfaces"][0]
    assert max(max(r["expected_vs_captured_residual_norms"]) for r in surface["capture_checks"]) < 1e-6
    assert len(surface["stages"]) == 3
    assert surface["stages"][0]["self_copy_delta_norm_max"] == 0
    # The old fake model changes logits merely because a hook is registered.
    # This must be visible as a nonzero downstream null, not a proof of causality.
    assert surface["stages"][-1]["self_copy_delta_norm_max"] > 0
    compact = clause_trajectory.compact_record(records[1])
    assert "sha256" in compact["frames"][0]["vectors"][STAGES[0]]


@pytest.mark.parametrize("failure", ["count", "context", "site", "pre_patch_cache", "missing_stage", "not_restored"])
def test_incomparable_or_pre_patch_capture_fails_closed(failure):
    _, records = worker_and_records()
    bad = deepcopy(records)
    if failure == "count":
        bad[0]["frames"].pop()
    elif failure == "context":
        bad[0]["result"]["operator_rows"][0]["context_fingerprint"] = "other"
    elif failure == "site":
        bad[0]["result"]["operator_rows"][0]["surface"]["alpha"] = 0.2
    elif failure == "pre_patch_cache":
        for i in range(2):
            bad[1]["frames"][i + 2]["vectors"][STAGES[0]] = bad[1]["frames"][i]["vectors"][STAGES[0]].clone()
    elif failure == "missing_stage":
        del bad[1]["frames"][2]["vectors"][STAGES[0]]
    else:
        bad[0]["result"]["state_restored"] = False
    with pytest.raises(ValueError):
        clause_trajectory.summarize(bad)


def test_geometry_is_displacement_not_large_shared_background_or_probability_sign():
    assert clause_trajectory.geometry(torch.tensor([1., 0.]), torch.tensor([0., 1.]))["cosine"] == 0
    assert clause_trajectory.geometry(torch.tensor([1., 0.]), torch.tensor([2., 0.]))["cosine"] == 1
    assert clause_trajectory.geometry(torch.zeros(2), torch.ones(2))["cosine"] is None
    with pytest.raises(ValueError):
        clause_trajectory.geometry(torch.tensor([float("nan")]), torch.ones(1))


def test_capture_restores_instance_override_after_missing_stage_or_prefix_failure():
    tokens = torch.tensor([[1, 2]])
    original = lambda x, **kwargs: (torch.zeros(1, 2, 3), {})
    runtime = SimpleNamespace(run_with_cache=original)
    with pytest.raises(ValueError, match="stage_missing"):
        with clause_trajectory.capture_forwards(runtime, tokens, ["missing"]):
            runtime.run_with_cache(tokens)
    assert runtime.run_with_cache is original
    with pytest.raises(ValueError, match="prefix_changed"):
        with clause_trajectory.capture_forwards(runtime, tokens, ["missing"]):
            runtime.run_with_cache(torch.tensor([[1, 3]]))
    assert runtime.run_with_cache is original


def test_packet_canaries_are_separate_from_replays_and_total_counted():
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    original_packet = worker.build_controller_packet

    def packet_with_canary():
        worker.runtime_state.run_with_cache(worker._current_token_tensor())
        return original_packet()

    with patch.object(worker, "build_controller_packet", side_effect=packet_with_canary):
        with clause_trajectory.count_forwards(worker.runtime_state) as counts:
            records = clause_trajectory.measure(worker, [1, 3], stages=STAGES)
    assert counts == {"attempted": 28, "completed": 28}
    assert all(len(r["frames"]) == 7 and len(clause_trajectory.replay_frames(r)) == 4 for r in records)
    summary = clause_trajectory.summarize(records)
    assert summary["captured_replay_forward_count"] == 16
    assert summary["captured_preparation_forward_count"] == 12
    assert all(r["preparation_vs_baseline_norm_max"] == 0 for r in summary["surfaces"][0]["stages"])
    assert "run_with_cache" not in vars(worker.runtime_state)
    assert "_simulate_decode" not in vars(worker)


def test_counting_retains_failed_attempts_and_restores_method():
    runtime = SimpleNamespace(run_with_cache=lambda *a, **kw: (_ for _ in ()).throw(ValueError("failed")))
    original = runtime.run_with_cache
    with pytest.raises(ValueError, match="failed"):
        with clause_trajectory.count_forwards(runtime) as counts:
            runtime.run_with_cache(None)
    assert counts == {"attempted": 1, "completed": 0}
    assert runtime.run_with_cache is original


def test_transformerlens_cache_is_post_hook_and_instrumentation_adds_no_hook():
    tl = pytest.importorskip("transformer_lens")
    from SpiralInterventionLab.runtime.tlens_runtime import HookedTransformerRuntimeState

    cfg = tl.HookedTransformerConfig(n_layers=2, d_model=12, n_ctx=8, d_head=6,
                                    n_heads=2, d_mlp=24, d_vocab=32, act_fn="relu", seed=0)
    model = tl.HookedTransformer(cfg)
    runtime = HookedTransformerRuntimeState(model)
    tokens = torch.tensor([[1, 2, 3]])
    stages = clause_trajectory.stage_names(2)
    _, before = runtime.run_with_cache(tokens)
    hook = "blocks.0.hook_resid_pre"

    def edit(act, hook):
        changed = act.clone()
        changed[0, -1, 0] += 0.08
        return changed

    runtime.register_hook(hook_name=hook, hook_fn=edit, edit_id="test", ttl_steps=1, revertible=True)
    hook_count = sum(len(h.fwd_hooks) for h in model.hook_dict.values())
    with clause_trajectory.capture_forwards(runtime, tokens, stages) as frames:
        runtime.run_with_cache(tokens)
    assert len(frames) == 1
    assert sum(len(h.fwd_hooks) for h in model.hook_dict.values()) == hook_count
    delta = frames[0]["vectors"][hook] - before[hook][0, -1]
    assert float(delta[0]) == pytest.approx(0.08)
    assert torch.count_nonzero(delta).item() == 1
    runtime.remove_edit("test")


@pytest.mark.parametrize("cause", ["budget", "unknown_token", "duplicate", "stale_cache", "missing_stage"])
def test_measurement_preflights_before_any_forward(cause):
    worker = make_worker()
    ids, stages = [1, 3], STAGES
    if cause == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run - 3
    elif cause == "unknown_token":
        ids = [1, 400]
    elif cause == "duplicate":
        ids = [1, 1]
    elif cause == "stale_cache":
        worker.runtime_state.last_tokens = torch.tensor([[1]])
    else:
        stages = ["not_a_stage"]
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("preflight only")):
        with pytest.raises(ValueError):
            clause_trajectory.measure(worker, ids, stages=stages)


def test_reference_matches_tokens_and_source_bindings_not_just_prefix_text():
    worker = make_worker()
    candidates = clause_actuation.offer(worker)["candidates"]
    reference = {"status": "complete", "device": "mps", "dtype": "float16",
                 "cache_alignment": {"tokens_identity": tensor_identity(worker._current_token_tensor())},
                 "measurements": [{"prefix": worker.final_text(), "operator_measurements": [
                     {"result": {"status": "ok", "candidate_binding": c}} for c in candidates]}]}
    assert verify_reference(worker, reference, [1, 3])["status"] == "matched"
    reference["cache_alignment"]["tokens_identity"]["sha256"] = "different"
    with pytest.raises(ValueError, match="prefix_or_tokens"):
        verify_reference(worker, reference, [1, 3])


def test_cli_rejects_unspecified_pair_without_loading_a_model():
    args = ["--source-jsonl", "unused", "--source-manifest", "unused", "--reference-receipt", "unused",
            "--worker-model-path", "unused", "--position", "8", "--source-token-ids", "11,11", "--output-dir", "unused"]
    with patch("SpiralInterventionLab.examples.replay_source_clause_trajectory.load_worker_model",
               side_effect=AssertionError("invalid arguments")):
        with pytest.raises(SystemExit) as exc:
            main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("change", [None, "source", "manifest", "checkpoint", "missing_checkpoint"])
def test_reference_provenance_requires_same_inputs_and_checkpoint(change):
    reference = {"source_jsonl_sha256": "source", "source_manifest_sha256": "manifest",
                 "checkpoint_sha256_verified": {"model": "original"}}
    manifest = {"checkpoint_sha256": {"model": "original"}}
    if change in {"source", "manifest"}:
        reference["source_jsonl_sha256" if change == "source" else "source_manifest_sha256"] = "other"
    elif change == "checkpoint":
        manifest["checkpoint_sha256"]["model"] = "other"
    elif change == "missing_checkpoint":
        reference.pop("checkpoint_sha256_verified")
    if change:
        with pytest.raises(ValueError):
            cli.verify_reference_provenance(reference, manifest, "source", "manifest")
    else:
        cli.verify_reference_provenance(reference, manifest, "source", "manifest")


@pytest.mark.parametrize("change", [None, "position", "prefix", "source", "manifest", "incomplete"])
def test_adjacent_boundary_plan_is_explicit_and_source_bound(change):
    reference = {"status": "complete", "source_jsonl_sha256": "source", "source_manifest_sha256": "manifest",
                 "measurements": [{"generated_tokens": 8, "prefix": " old"}]}
    prefixes, position = {8: " old", 9: " old next"}, 9
    if change == "position":
        position = 10
    elif change == "prefix":
        prefixes[8] = "different"
    elif change in {"source", "manifest"}:
        reference["source_jsonl_sha256" if change == "source" else "source_manifest_sha256"] = "different"
    elif change == "incomplete":
        reference["status"] = "error"
    if change:
        with pytest.raises(ValueError):
            next_boundary_plan(reference, prefixes, position, "source", "manifest")
    else:
        plan = next_boundary_plan(reference, prefixes, position, "source", "manifest")
        assert plan["anchor_position"] == 8 and plan["measurement_position"] == 9
        assert not plan["fallback_position_search_allowed"]
        assert "not_same_prefix" in plan["mode"]


def test_multiple_surfaces_have_distinct_replay_offsets_and_verified_hook_deltas():
    worker = make_worker()
    first = worker.surface_catalog[0]
    second = replace(first, surface_id="test_mlp", target=replace(first.target, site="mlp_out"),
                     caps=replace(first.caps, step_size=0.04))
    worker.surface_catalog.append(second)
    raw = deepcopy(worker._surface_catalog_raw[0])
    raw.update(surface_id="test_mlp", target={**raw["target"], "site": "mlp_out"})
    raw["caps"]["step_size"] = 0.04
    worker._surface_catalog_raw.append(raw)
    runtime = worker.runtime_state
    original = runtime.run_with_cache

    def with_mlp_hook(tokens, **kwargs):
        logits, cache = original(tokens, **kwargs)
        for hook in runtime.hooks.values():
            if hook["hook_name"] == "mlp_out:11":
                cache["mlp_out:11"] = hook["hook_fn"](cache["mlp_out:11"], None)
        runtime.last_cache = runtime._freeze_cache(cache)
        return logits, cache

    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    before = state_identity(worker)
    with patch.object(runtime, "run_with_cache", side_effect=with_mlp_hook):
        with clause_trajectory.count_forwards(runtime) as counts:
            records = clause_trajectory.measure(worker, [1, 3], stages=[*STAGES, "mlp_out:11"])
    assert counts == {"attempted": 24, "completed": 24}
    assert state_identity(worker) == before
    summary = clause_trajectory.summarize(records)
    assert len(summary["surfaces"]) == 2
    for index, entry in enumerate(summary["surfaces"]):
        assert all(max(row["expected_vs_captured_residual_norms"]) < 1e-6 for row in entry["capture_checks"])
        for record in records:
            row = record["result"]["operator_rows"][index]
            frames = clause_trajectory.replay_frames(record)
            assert all(f["origin"]["edit_ids"] == [row["edit"]["id"]]
                       for f in frames[2 + 2 * index:4 + 2 * index])
    # Swapping the second site's replay with the first must not pass validation.
    bad = deepcopy(records)
    bad[1]["frames"][2], bad[1]["frames"][4] = bad[1]["frames"][4], bad[1]["frames"][2]
    with pytest.raises(ValueError):
        clause_trajectory.summarize(bad)


@pytest.fixture
def cpu_cli_state():
    device, grad, threads = torch.get_default_device(), torch.is_grad_enabled(), torch.get_num_threads()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(device)
        torch.set_grad_enabled(grad)
        torch.set_num_threads(threads)


def test_cli_partial_failure_saves_failed_receipt_counts_and_restores_wrappers(tmp_path, monkeypatch, cpu_cli_state):
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    before, used = state_identity(worker), diagnostic_budget.used(worker)
    candidates = clause_actuation.offer(worker)["candidates"]
    reference = {"status": "complete", "device": "cpu", "dtype": "float32",
                 "cache_alignment": {"tokens_identity": tensor_identity(worker._current_token_tensor())},
                 "measurements": [{"prefix": worker.final_text(), "operator_measurements": [
                     {"result": {"status": "ok", "candidate_binding": c}} for c in candidates]}]}
    manifest = {"profile": {"task": "test", "seed": 0, "worker_model": "test"},
                "checkpoint_sha256": {"model": "test"}}
    import hashlib
    reference.update(source_jsonl_sha256=hashlib.sha256(json.dumps({}).encode()).hexdigest(),
                     source_manifest_sha256=hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
                     checkpoint_sha256_verified=manifest["checkpoint_sha256"])
    for name, value in [("reference.json", reference), ("manifest.json", manifest), ("source.jsonl", {})]:
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(cli, "source_prefixes", lambda _: ("pac", {1: worker.final_text()}))
    monkeypatch.setattr(cli, "create_task_env", lambda _: SimpleNamespace(reset=lambda _: "pac"))
    monkeypatch.setattr(cli, "verify_checkpoint", lambda *_: {})
    monkeypatch.setattr(cli, "load_worker_model", lambda *a, **kw: SimpleNamespace(
        parameters=lambda: [torch.zeros(1)], cfg=SimpleNamespace(n_layers=12)))
    monkeypatch.setattr(cli, "build_hooked_transformer_worker_runtime", lambda *a, **kw: worker)
    monkeypatch.setattr(worker, "reset", lambda _: None)
    monkeypatch.setattr(cli.clause_specificity, "align_cache", lambda _: {})
    monkeypatch.setattr(cli, "measure_current_prefix", lambda _: {"result": {"status": "ok"}})
    measure = clause_trajectory.measure
    monkeypatch.setattr(clause_trajectory, "measure", lambda w, ids, **kw: measure(w, ids, stages=STAGES, **kw))
    original = worker.runtime_state.run_with_cache
    attempts = 0

    def fail_second_condition(*a, **kw):
        nonlocal attempts
        attempts += 1
        if attempts == 5:
            raise ValueError("injected second-condition failure")
        return original(*a, **kw)

    monkeypatch.setattr(worker.runtime_state, "run_with_cache", fail_second_condition)
    output = tmp_path / "measurement"
    args = ["--source-jsonl", str(tmp_path / "source.jsonl"), "--source-manifest", str(tmp_path / "manifest.json"),
            "--reference-receipt", str(tmp_path / "reference.json"), "--worker-model-path", str(tmp_path),
            "--position", "1", "--source-token-ids", "1,3", "--output-dir", str(output)]
    with pytest.raises(ValueError, match="trajectory_measurement_failed"):
        main(args)
    receipt = json.loads((output / "trajectory.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["recorded_condition_count"] == 2
    assert receipt["recorded_condition_replay_count"] == 5
    assert receipt["failed_condition_labels"] == ["source:3"]
    assert receipt["forward_counts_total"] == {"attempted": 5, "completed": 4}
    assert receipt["diagnostic_budget_used"] == used + 2
    assert "comparison" not in receipt and "choice_comparison" not in receipt
    assert (output / "vectors_01.pt").is_file()
    assert state_identity(worker) == before
    assert worker.runtime_state.run_with_cache is fail_second_condition
    assert "_simulate_decode" not in vars(worker) and "build_controller_packet" not in vars(worker)
    assert not worker._collect_active_edits()
