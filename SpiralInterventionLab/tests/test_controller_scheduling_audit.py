"""Scheduling contracts, independent of operator efficacy.

The toy confirmation deliberately succeeds at the current prefix; physical
certification and exact trial authorization have separate worker tests.
"""
from copy import deepcopy

import pytest

from SpiralInterventionLab.runtime.compiler import StepContext
from SpiralInterventionLab.runtime.loop import InMemoryStructuredLogger, run_episode
from SpiralInterventionLab.runtime.policy import PolicyViolation
from SpiralInterventionLab.tests.test_controller_runtime import (
    FakeAdapter, FakeRuntimeState, _ToyTaskEnv, _ToyWorkerRuntime, _resid_command,
)


@pytest.mark.parametrize("repair_diagnostic_cap", [False, True])
def test_measure_confirm_decide_fits_one_prefix(repair_diagnostic_cap):
    state = FakeRuntimeState()

    class Worker(_ToyWorkerRuntime):
        offer_prefix = None

        def build_controller_packet(self):
            packet = super().build_controller_packet()
            packet["control_phase_hint"] = "entity_insertion"
            packet["budget"].update(production_trial_edits_left_this_run=1,
                production_trial_alpha_left_total=0.15, production_trial_edit_cost_left_total=0.15)
            return packet

        def request_controller_diagnostics(self, requests, **kwargs):
            results = super().request_controller_diagnostics(requests, **kwargs)
            if requests[0]["diagnostic"] == "activation_patch_production_trial_gate_review" or (
                    requests[0]["diagnostic"] == "candidate_action"
                    and requests[0].get("action_id") == "action:normal_cap_investigation"):
                self.offer_prefix = self.steps
            return results

        def candidate_trial_followup_available(self, results):
            return bool(results) and not self.done()

        def validate_controller_trial(self, command, packet):
            if self.offer_prefix != self.steps or self.done():
                raise PolicyViolation("missing_or_stale_toy_confirmation")
            return command

    worker = Worker(state)
    plan = (["diagnostic_cap_measurement", "normal_cap_investigation", "apply"]
            if repair_diagnostic_cap else ["normal_cap_measurement", "confirmation", "apply"])
    observed_plan = []

    class Controller:
        def invoke(self, packet):
            if worker.done():
                return {"version": "0.1", "decision": "noop"}
            action = plan[len(observed_plan)]
            observed_plan.append((worker.steps, action))
            if action == "apply":
                command = deepcopy(_resid_command())
                command["meta"]["apply_kind"] = "production_trial"
                edit = command["edits"][0]
                edit["op"] = {"kind": "activation_patch", "mode": "blend", "alpha": 0.04}
                edit["budget"].update(ttl_steps=1, norm_clip=1.0, step_size=0.04)
                edit["meta"].update(apply_kind="production_trial", production_trial_allowed=True)
                return command
            request = ({"diagnostic": "activation_patch_production_trial_gate_review",
                        "evidence_id": "obs:normal_cap", "objective_bundle_key": "o"}
                       if action == "confirmation" else
                       {"diagnostic": "candidate_action", "action_id": "action:" + action})
            return {"version": "0.1", "decision": "noop", "meta": {
                "generation_action": "hold_prefix", "diagnostic_request": request}}

    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2, logger=logger)
    assert worker.offer_prefix == 1, "toy confirmation must succeed before testing scheduling"
    compiled = [r for r in logger.events if r["event"] == "compiled_edit"]
    assert len(compiled) == 1, {"observed_plan": observed_plan, "offer_prefix": worker.offer_prefix}
    assert compiled[0]["step"] == 0


@pytest.mark.parametrize("request_round", [0, 1])
def test_tool_and_observer_requests_are_dispatched_in_each_round(request_round):
    worker = _ToyWorkerRuntime(FakeRuntimeState())

    class Controller:
        calls = 0

        def invoke(self, packet):
            if worker.done():
                return {"version": "0.1", "decision": "noop"}
            current = self.calls
            self.calls += 1
            meta = {}
            if current == 0:
                meta.update(generation_action="hold_prefix", diagnostic_request={
                    "diagnostic": "candidate_action", "action_id": "action:inspect"})
            if current == request_round:
                meta.update(tool_requests=[{"tool": "tokenize_terms", "terms": ["x"]}],
                            observer_check_request={"kind": "semantic_progress"})
            return {"version": "0.1", "decision": "noop", "meta": meta}

    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=worker.runtime_state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2, logger=logger)
    commands = [r for r in logger.events if r["event"] == "controller_command"]
    assert any(r["command"].get("meta", {}).get("tool_requests") for r in commands)
    assert len(worker._tool_results) == len(worker._observer_checks) == 1, {
        "request_round": request_round, "tool_results": worker._tool_results,
        "observer_results": worker._observer_checks}
