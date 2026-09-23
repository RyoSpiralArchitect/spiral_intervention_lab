from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from SpiralInterventionLab.runtime import candidate_trial, diagnostic_budget
from SpiralInterventionLab.runtime.response_promotion import response_confirmation_readiness, review_response_evidence
from SpiralInterventionLab.runtime.schema import parse_controller_command
from SpiralInterventionLab.runtime.policy import PolicyViolation
from SpiralInterventionLab.runtime.loop import _extract_diagnostic_requests, run_episode, InMemoryStructuredLogger
from SpiralInterventionLab.runtime.compiler import StepContext
from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload
from SpiralInterventionLab.tests.test_diagnostic_tooling import fixture_promotable_response
from SpiralInterventionLab.tests import test_controller_runtime as fixtures
from SpiralInterventionLab.tests.test_controller_runtime import (
    _ToyWorkerRuntime, _ToyTaskEnv, FakeAdapter, FakeRuntimeState,
)


@pytest.fixture
def positive(monkeypatch):
    monkeypatch.setattr(candidate_trial, "state_identity", lambda w: "context:fixed")
    monkeypatch.setattr("SpiralInterventionLab.runtime.response_promotion.state_identity", lambda w: "context:fixed")
    worker, row, req, packet, calls = fixture_promotable_response()
    row["evidence_id"] = row["observable_id"]
    row.update(evidence_kind="current_prefix_candidate_measurement", candidate_id="frozen:1",
               actual_delta_class="unassessed", actuator_class="unknown")
    for key in ("self_delta", "alignment_margin", "repeat_flag_delta", "repetition_score_delta", "entropy_delta",
                "top1_margin_delta", "required_term_recall_delta", "required_term_span_progress_delta"):
        row.pop(key, None)
    worker.max_diagnostic_calls_per_run = 12
    worker._diagnostic_calls_used = 1
    return worker, row, req, packet, calls


def test_readout_only_evidence_can_request_but_not_bypass_confirmation(positive):
    worker, row, req, packet, calls = positive
    ready = response_confirmation_readiness(row, "context:fixed")
    assert ready["review_eligible"] and ready["safety_status"] == "not_yet_certified"
    report = candidate_trial.handoff(worker, row, context="context:fixed", budget_left=1)
    assert report["same_prefix_followup_available"] and not report["production_trial_allowed"]
    before = len(calls)
    result = review_response_evidence(worker, req, packet)
    assert result["production_trial_allowed"], result
    assert len(calls) - before == 2 and result["physical_replay_count"] == 4
    assert result["same_prefix_followup_available"] and not result["production_apply_allowed"]
    assert candidate_trial.followup_available(worker, [result])
    hints = candidate_trial.packet_hints(worker)
    compact = _compact_controller_payload({"strategy_hints": hints})["strategy_hints"]
    assert compact["candidate_trial_offer"] == hints["candidate_trial_offer"]


@pytest.mark.parametrize("change", [
    {"target_piece_prob_delta": 0.000001, "bound_token_top20_hit_delta": 0},
    {"measurement_context_id": "old"}, {"target_piece_binding_variant": "alternate"},
    {"state_restored": False}, {"no_edit_max_abs_logit_delta": float("nan")},
    {"repeat_max_abs_logit_delta": -1}, {"activation_patch_hook_call_count": 0},
])
def test_readout_handoff_rejects_weak_stale_or_uncontrolled_evidence(positive, change):
    worker, row, req, packet, calls = positive
    row.update(change)
    assert not candidate_trial.handoff(worker, row, context="context:fixed", budget_left=1)["same_prefix_followup_available"]


def test_handoff_checks_budget_and_exact_normal_cap(positive):
    worker, row, req, packet, calls = positive
    assert "diagnostic_budget_exhausted" in candidate_trial.handoff(worker, row, context="context:fixed", budget_left=0)["blocked_reasons"]
    materialize = worker._activation_patch_trial_edit_from_candidate
    def clipped(*args, **kwargs):
        edit = materialize(*args, **kwargs)
        edit["budget"]["step_size"] = 0.03
        return edit
    worker._activation_patch_trial_edit_from_candidate = clipped
    assert "normal_trial_edit_differs_from_diagnostic" in candidate_trial.handoff(worker, row, context="context:fixed", budget_left=1)["blocked_reasons"]


def test_confirmation_does_not_inherit_safety_or_repeat_control(positive):
    worker, row, req, packet, calls = positive
    prior = worker.replay_candidate_edits_actual_delta
    def missing_safety(*args, **kwargs):
        result = prior(*args, **kwargs)
        result.pop("required_term_recall_delta")
        return result
    worker.replay_candidate_edits_actual_delta = missing_safety
    result = review_response_evidence(worker, req, packet)
    assert not result["production_trial_allowed"]
    assert "finite_controls_and_effects" in result["confirmation"]["blocked_reasons"]
    assert not getattr(worker, "_response_trial_grants", {})


@pytest.mark.parametrize("bad_budget", [None, True, "oops", float("nan"), float("inf"), -1, 0])
def test_trial_review_does_not_invent_missing_budget(positive, bad_budget):
    worker, row, req, packet, calls = positive
    packet["budget"]["production_trial_alpha_left_total"] = bad_budget
    result = review_response_evidence(worker, req, packet)
    assert not result["production_trial_allowed"] and not getattr(worker, "_response_trial_grants", {})


def test_final_trial_cannot_change_confirmed_edit(positive):
    worker, row, req, packet, calls = positive
    edit = worker._activation_patch_trial_edit_from_candidate(row["candidate_descriptor"], trial_contract={})
    wrong = deepcopy(edit)
    wrong["op"]["alpha"] = 0.1
    result = {"production_trial_allowed": True, "production_trial_candidate": {"trial_edit": wrong},
              "activation_patch_production_trial_gate_review": {"production_trial_allowed": True}}
    candidate_trial.authorize(worker, result, row, edit)
    assert not result["production_trial_allowed"]
    assert not result["activation_patch_production_trial_gate_review"]["production_trial_allowed"]
    assert not getattr(worker, "_response_trial_grants", {})


def test_trial_offer_is_exact_current_single_use_and_accepts_parsed_command(positive, monkeypatch):
    worker, row, req, packet, calls = positive
    result = review_response_evidence(worker, req, packet)
    assert result["production_trial_allowed"], result
    edit = result["production_trial_candidate"]["trial_edit"]
    command = {"version": "0.1", "decision": "apply", "edits": [edit]}
    parsed = parse_controller_command(command)
    accepted = candidate_trial.validate_apply(worker, parsed, packet)
    assert accepted["edits"] == [edit] and accepted["meta"]["apply_kind"] == "production_trial"
    changed = deepcopy(command)
    changed["edits"][0]["op"]["alpha"] = 0.06
    with pytest.raises(PolicyViolation, match="differs_from_authorized"):
        candidate_trial.validate_apply(worker, changed, packet)
    monkeypatch.setattr(candidate_trial, "state_identity", lambda w: "context:next")
    with pytest.raises(PolicyViolation, match="stale_trial_authorization"):
        candidate_trial.validate_apply(worker, parsed, packet)
    monkeypatch.setattr(candidate_trial, "state_identity", lambda w: "context:fixed")
    candidate_trial.consume(worker, accepted)
    with pytest.raises(PolicyViolation, match="stale_trial_authorization"):
        candidate_trial.validate_apply(worker, parsed, packet)


def test_simulated_one_token_edit_does_not_persist_through_two_token_lookahead():
    worker = fixtures.TestWorkerRuntimeAndBaselines()._make_worker_runtime()
    worker.reset("pa")
    candidate = {"objective_bundle_key": "entity_insert:a:source_body:1:2", "objective_term": "a",
                 "site": "resid_pre", "layer": 3, "alpha": 0.04, "source_localization": "source_term_token"}
    edit = worker._activation_patch_trial_edit_from_candidate(candidate,
        trial_contract={"max_alpha": 0.15, "norm_clip": 1, "trial_budget_class": "diagnostic_only"})
    assert edit is not None
    calls = []
    run = worker.runtime_state.run_with_cache
    def observed(*args, **kwargs):
        calls.append(list(worker.runtime_state.hooks))
        return run(*args, **kwargs)
    worker.runtime_state.run_with_cache = observed
    result = worker._simulate_decode(max_new_tokens=2, top_k=3,
        command={"version": "0.1", "decision": "apply", "edits": [edit]}, score_candidate_text=False)
    assert result is not None, worker._last_simulate_decode_error
    assert calls[-2:] == [[edit["id"]], []]
    assert not worker.runtime_state.hooks and worker.final_text() == ""


def test_evidence_review_is_exclusive_not_stale_prose():
    req = {"diagnostic": "activation_patch_production_trial_gate_review", "evidence_id": "obs:1", "objective_bundle_key": "o"}
    command = {"decision": "noop", "meta": {"diagnostic_request": req,
        "next_action": "request_operator_diagnostic", "controller_memory": {"diagnostic_request": "operator_diagnostic_replay"}}}
    assert _extract_diagnostic_requests(command, {}) == [req]


def test_missing_requested_site_does_not_fall_back_to_another_surface():
    worker = fixtures.TestWorkerRuntimeAndBaselines()._make_worker_runtime()
    candidate = {"objective_bundle_key": "entity_insert:a:source_body:0:1", "objective_term": "a",
                 "site": "mlp_out", "layer": 3, "alpha": 0.04, "source_localization": "source_term_token"}
    assert worker._activation_patch_trial_edit_from_candidate(candidate, trial_contract={}) is None
    candidate["site"] = "unknown"
    assert worker._activation_patch_trial_edit_from_candidate(candidate, trial_contract={}) is None


@pytest.mark.parametrize("rounds,expected", [(0, 2), (1, 3), (2, 4), (99, 4)])
def test_same_prefix_followups_are_phase_independent_bounded_and_optional(rounds, expected):
    state, adapter = FakeRuntimeState(), FakeAdapter()
    class Worker(_ToyWorkerRuntime):
        def build_controller_packet(self):
            packet = super().build_controller_packet()
            packet["control_phase_hint"] = "entity_insertion"
            return packet
        def candidate_trial_followup_available(self, results):
            return self.steps == 1 and bool(results)
    worker = Worker(state)
    class Controller:
        def __init__(self):
            self.prefixes = []
        def invoke(self, packet):
            self.prefixes.append(worker.steps)
            return {"version": "0.1", "decision": "noop", "meta": {"diagnostic_request": {
                "diagnostic": "activation_patch_production_trial_gate_review",
                "evidence_id": f"obs:{len(self.prefixes)}", "objective_bundle_key": "o"}}}
    controller, logger = Controller(), InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, controller,
        StepContext(packet={}, runtime_state=state, adapter=adapter, traces={}, stats={}),
        logger=logger, max_candidate_handoff_rounds=rounds)
    assert len(controller.prefixes) == expected
    assert controller.prefixes == [1] * (expected - 1) + [2]
    assert not any(e["event"] == "compiled_edit" for e in logger.events)


def test_controller_chosen_trial_runs_on_one_token_then_expires_before_next_review():
    state, adapter = FakeRuntimeState(), FakeAdapter()
    class Worker(_ToyWorkerRuntime):
        def __init__(self):
            super().__init__(state)
            self.token_edits = []
        def step(self):
            self.token_edits.append(list(state.hooks))
            super().step()
        def done(self):
            return self.steps >= 3
        def candidate_trial_followup_available(self, results):
            return self.steps == 1 and bool(results)
        def build_controller_packet(self):
            packet = super().build_controller_packet()
            packet["control_phase_hint"] = "entity_insertion"
            packet["budget"].update(production_trial_edits_left_this_run=1,
                production_trial_alpha_left_total=0.15, production_trial_edit_cost_left_total=0.15)
            return packet
    worker = Worker()
    class Controller:
        def __init__(self):
            self.calls = 0
        def invoke(self, packet):
            self.calls += 1
            if self.calls <= 2:
                return {"version": "0.1", "decision": "noop", "meta": {"diagnostic_request": {
                    "diagnostic": "activation_patch_production_trial_gate_review",
                    "evidence_id": f"obs:{self.calls}", "objective_bundle_key": "o"}}}
            if self.calls == 3:
                command = fixtures._resid_command()
                command["meta"]["apply_kind"] = "production_trial"
                edit = command["edits"][0]
                edit["op"] = {"kind": "activation_patch", "mode": "blend", "alpha": 0.04}
                edit["budget"].update(ttl_steps=1, norm_clip=1.0, step_size=0.04)
                edit["meta"].update(apply_kind="production_trial", production_trial_allowed=True)
                return parse_controller_command(command)
            assert not state.hooks
            return {"version": "0.1", "decision": "noop"}
    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=state, adapter=adapter, traces={}, stats={}),
        logger=logger, max_candidate_handoff_rounds=2)
    assert worker.token_edits == [[], ["e_rescue"], []]
    assert len([e for e in logger.events if e["event"] == "compiled_edit"]) == 1
    assert len([e for e in logger.events if e["event"] == "production_trial_expired"]) == 1


def test_legacy_review_checks_do_not_advertise_an_unconfirmed_permission():
    result = candidate_trial.require_physical_confirmation({"production_trial_allowed": True,
        "candidate": {"trial_edit": {"meta": {"production_trial_allowed": True}}}})
    assert result["trial_review_checks_passed"] and not result["production_trial_allowed"]
    assert not result["candidate"]["trial_edit"]["meta"]["production_trial_allowed"]


@pytest.mark.parametrize("ttl,kind", [(1, "resid_add"), (2, "resid_add"), (1, "activation_patch")])
def test_normal_edit_ttl_counts_generated_tokens_not_controller_rounds(ttl, kind):
    state = FakeRuntimeState()

    class Worker(_ToyWorkerRuntime):
        def __init__(self):
            super().__init__(state)
            self.token_hooks = []
            self.packet_hooks = []

        def step(self):
            self.token_hooks.append(list(state.hooks))
            super().step()

        def done(self):
            return self.steps >= 4

        def tick_ttl(self):
            state.tick_ttl()

        def cleanup_expired(self):
            state.cleanup_expired()

        def build_controller_packet(self):
            self.packet_hooks.append(list(state.hooks))
            return super().build_controller_packet()

    worker = Worker()

    class Controller:
        def invoke(self, packet):
            if worker.steps == 1:
                command = fixtures._resid_command()
                command["edits"][0]["budget"]["ttl_steps"] = ttl
                if kind == "activation_patch":
                    command["edits"][0]["op"] = {"kind": kind, "mode": "blend", "alpha": 0.04}
                    command["edits"][0]["budget"].update(norm_clip=1.0, step_size=0.04)
                return command
            return {"version": "0.1", "decision": "noop"}

    ctx = StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={})
    run_episode(_ToyTaskEnv(), worker, Controller(), ctx)
    assert worker.token_hooks == [[]] + [["e_rescue"]] * ttl + [[]] * (3 - ttl)
    assert not worker.packet_hooks[ttl]
    assert not state.hooks and not ctx.active_edits


def test_rejected_trial_memory_records_noop_not_unexecuted_apply():
    state = FakeRuntimeState()

    class Worker(_ToyWorkerRuntime):
        def build_controller_packet(self):
            packet = super().build_controller_packet()
            packet["budget"].update(production_trial_edits_left_this_run=1,
                production_trial_alpha_left_total=0.5, production_trial_edit_cost_left_total=0.5)
            return packet

        def validate_controller_trial(self, command, packet):
            raise PolicyViolation("missing_or_stale_trial_authorization")

    worker = Worker(state)

    class Controller:
        def invoke(self, packet):
            if worker.steps == 1:
                command = fixtures._resid_command_with_memory()
                command["meta"]["apply_kind"] = "production_trial"
                command["edits"][0]["op"]["alpha"] = 0.04
                return command
            assert packet["controller_memory"][-1]["decision"] == "noop"
            assert packet["controller_memory"][-1]["apply_block_reason"] == "missing_or_stale_trial_authorization"
            return {"version": "0.1", "decision": "noop"}

    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
                StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}), logger=logger)
    assert not any(e["event"] == "compiled_edit" for e in logger.events)


def test_frontier_preference_does_not_override_exact_physical_confirmation(positive):
    worker, row, req, packet, calls = positive
    packet["strategy_hints"]["diagnostic_frontier_bundle_key"] = "another_objective"
    result = review_response_evidence(worker, req, packet)
    assert result["confirmation"]["review_eligible"]
    assert result["repeat_confirmation"]["review_eligible"]
    assert result["production_trial_allowed"]
    shadow = result["activation_patch_production_shadow_replay"]
    assert not shadow["production_denial_reasons_after_shadow"]
    evidence = shadow["production_shadow_dossier"]["axes"]["context_equivalence_missing"]["evidence"]
    assert evidence["physical_context_verified"] and not evidence["frontier_matches_objective"]
    assert not result["production_apply_allowed"]
    assert len(worker._response_trial_grants) == 1


@pytest.mark.parametrize("change", ["objective", "actuator", "state", "binding", "source"])
def test_frontier_independence_does_not_allow_evidence_transfer(positive, change):
    worker, row, req, packet, calls = positive
    packet["strategy_hints"]["diagnostic_frontier_bundle_key"] = "another_objective"
    if change == "objective":
        req["objective_bundle_key"] = "another_objective"
    elif change == "actuator":
        req["step_actuator_bundle_key"] = "unmeasured_actuator"
    elif change == "state":
        row["measurement_context_id"] = "old"
    elif change == "binding":
        row["target_piece_binding_variant"] = "alternate"
    else:
        worker.adapter.read_ref = lambda *args, **kwargs: torch.zeros(4)
    result = review_response_evidence(worker, req, packet)
    assert not result["production_trial_allowed"], result
    assert not getattr(worker, "_response_trial_grants", {})


def test_controller_cannot_supply_physical_confirmation_in_request(positive):
    worker, row, req, packet, calls = positive
    packet["strategy_hints"]["diagnostic_frontier_bundle_key"] = "another_objective"
    result = review_response_evidence(worker, req, packet)
    req["physical_confirmation"] = deepcopy(row)
    shadow = worker._activation_patch_production_shadow_replay({}, request=req, packet_context=packet,
        promotion_gate_review=result["activation_patch_promotion_gate_review"])
    assert not shadow["context_equivalence_certified"] and not shadow["production_apply_allowed"]


def test_terminal_prefix_cannot_install_an_edit_with_no_future_generated_token():
    state = FakeRuntimeState()
    worker = _ToyWorkerRuntime(state)

    class Controller:
        def invoke(self, packet):
            if worker.done():
                return fixtures._resid_command()
            return {"version": "0.1", "decision": "noop"}

    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
                StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}), logger=logger)
    assert not state.hooks
    assert not any(e["event"] == "compiled_edit" for e in logger.events)
    assert any(e["event"] == "controller_guardrail" and e["reason"] == "no_next_token_for_edit" for e in logger.events)


def test_word_limit_mask_is_distinct_from_model_nonfinite_readout():
    import torch
    from SpiralInterventionLab.runtime.response_probe import bound_metrics, ReadoutUnavailable
    worker = fixtures.TestWorkerRuntimeAndBaselines()._make_worker_runtime()
    worker.final_text = lambda: " In the case of a rewrite, the budget draft"
    worker._single_token_text = lambda i: "." if i == 1 else " Mir"
    logits, state = worker._apply_word_budget_guardrail(torch.zeros(40, device="cpu"), max_words=9)
    assert state["word_budget_guardrail_active"] and torch.isneginf(logits[30])
    with pytest.raises(ReadoutUnavailable, match="bound_target_masked") as raised:
        bound_metrics(logits, logits, 30)
    assert raised.value.details["before_finite_count"] < 20
    broken = logits.clone()
    broken[1] = float("nan")
    with pytest.raises(ReadoutUnavailable, match="nonfinite_bound_token"):
        bound_metrics(broken, broken, 30)
