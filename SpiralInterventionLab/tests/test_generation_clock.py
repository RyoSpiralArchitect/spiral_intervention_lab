from types import SimpleNamespace
import json

import pytest

from SpiralInterventionLab.bridge.controller_clients import ProviderControllerClient
from SpiralInterventionLab.controllers.base import ControllerProviderResponse
from SpiralInterventionLab.runtime import operation_cards, prefix_control
from SpiralInterventionLab.runtime.compiler import StepContext
from SpiralInterventionLab.runtime.loop import InMemoryStructuredLogger, run_episode
from SpiralInterventionLab.runtime.policy import PolicyViolation
from SpiralInterventionLab.tests.test_controller_runtime import (
    _ToyTaskEnv, _ToyWorkerRuntime, FakeRuntimeState, FakeAdapter, _resid_command,
)


def command(action, *, diagnostic=None):
    meta = {"generation_action": action}
    if diagnostic:
        meta["diagnostic_request"] = {"diagnostic": diagnostic}
    return {"version": "0.1", "decision": "noop", "meta": meta}


def run(controller, *, rounds=2, worker_type=_ToyWorkerRuntime, **kwargs):
    state = FakeRuntimeState()
    worker = worker_type(state)
    logger = InMemoryStructuredLogger()
    result = run_episode(_ToyTaskEnv(), worker, SimpleNamespace(invoke=lambda p: controller(worker, p)),
        StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}),
        logger=logger, generation_clock_mode="explicit", max_candidate_handoff_rounds=rounds, **kwargs)
    return worker, logger.events, result


def test_inspect_before_first_token_then_explicit_commit_and_token_only_ttl():
    positions, ticks = [], []
    class Worker(_ToyWorkerRuntime):
        def tick_ttl(self):
            ticks.append(self.steps)
    def decide(worker, packet):
        control = packet["strategy_hints"]["generation_control"]
        positions.append((worker.steps, control["controller_round"], control["generated_token_count"]))
        assert control["default_noop_action"] == "requires_explicit_generation_action"
        if control["same_prefix_rounds_left"]:
            return command("inspect_prefix", diagnostic=f"diagnostic_{control['controller_round']}")
        return command("commit_token")
    worker, events, result = run(decide, worker_type=Worker, max_controller_diagnostic_rethink_rounds=10)
    assert positions == [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1)]
    assert ticks == [1, 2] and result.steps == 2
    assert len([e for e in events if e["event"] == "generation_commit"]) == 2
    assert not any(e["event"] == "compiled_edit" for e in events)


@pytest.mark.parametrize("result", [[], [{"status": "blocked"}], [{"status": "no_cached_evidence"}]])
def test_empty_or_failed_diagnosis_returns_to_controller_before_token(result):
    positions = []
    class Worker(_ToyWorkerRuntime):
        def request_controller_diagnostics(self, *args, **kwargs):
            return result
    def decide(worker, packet):
        positions.append(worker.steps)
        return command("inspect_prefix", diagnostic="probe") if len(positions) == 1 else command("commit_token")
    run(decide, worker_type=Worker)
    assert positions == [0, 0, 1]


def test_duplicate_diagnostic_does_not_repeat_work_and_cannot_hold_forever():
    positions = []
    def decide(worker, packet):
        positions.append(worker.steps)
        return command("inspect_prefix", diagnostic="same_probe")
    _, events, _ = run(decide, rounds=99)
    assert positions == [0, 0, 0, 1, 1, 1]
    assert len([e for e in events if e["event"] == "controller_diagnostic_result"]) == 2
    commits = [e for e in events if e["event"] == "generation_commit"]
    assert all(e["commit_source"] == "research_budget_fallback" for e in commits)


def test_noop_without_time_action_is_not_a_silent_commit():
    positions = []
    def decide(worker, packet):
        positions.append(worker.steps)
        return {"version": "0.1", "decision": "noop"}
    _, events, _ = run(decide)
    assert positions == [0, 0, 0, 1, 1, 1]
    assert all(e["blocked_reason"] == "explicit_generation_action_required"
               for e in events if e["event"] == "controller_clock_decision")


def test_inspection_cannot_apply_and_commit_does_not_run_diagnostics():
    def decide(worker, packet):
        if packet["strategy_hints"]["generation_control"]["same_prefix_rounds_left"]:
            cmd = _resid_command()
            cmd["meta"] = {"generation_action": "inspect_prefix"}
            return cmd
        return command("commit_token", diagnostic="must_not_execute")
    _, events, _ = run(decide)
    assert not any(e["event"] in {"compiled_edit", "controller_diagnostic_request"} for e in events)
    assert any(e.get("blocked_reason") == "inspection_cannot_apply_or_rollback" for e in events)


def test_one_token_edit_is_present_at_commit_and_expires_only_after_token():
    edit_seen = []
    class Worker(_ToyWorkerRuntime):
        def step(self):
            edit_seen.append(self.runtime_state.has_edit("e_rescue"))
            super().step()
    def decide(worker, packet):
        if worker.steps == 0:
            cmd = _resid_command()
            cmd["edits"][0]["budget"]["ttl_steps"] = 1
            cmd["meta"] = {"generation_action": "commit_token"}
            return cmd
        return command("commit_token")
    _, events, _ = run(decide, worker_type=Worker)
    assert edit_seen == [True, False]
    names = [e["event"] for e in events]
    assert names.index("compiled_edit") < names.index("generation_commit") < names.index("runtime_edit_expired")


def test_restoration_failure_aborts_without_committing_any_token():
    positions = []
    class Worker(_ToyWorkerRuntime):
        def step(self):
            positions.append(self.steps)
            super().step()
        def request_controller_diagnostics(self, *args, **kwargs):
            return [{"status": "ok", "evidence": {"state_restored": False}}]
    with pytest.raises(PolicyViolation, match="restoration_failed"):
        run(lambda w, p: command("inspect_prefix", diagnostic="probe"), worker_type=Worker)
    assert positions == []


def test_diagnostic_cannot_silently_advance_generated_prefix():
    class Worker(_ToyWorkerRuntime):
        def request_controller_diagnostics(self, *args, **kwargs):
            self.output += " leaked"
            return [{"status": "ok"}]
    with pytest.raises(PolicyViolation, match="inspection_changed_generated_prefix"):
        run(lambda w, p: command("inspect_prefix", diagnostic="probe"), worker_type=Worker)


def test_provider_repairs_missing_clock_action_and_blocks_exhausted_operation_cards():
    calls = []
    packet = prefix_control.annotate({"strategy_hints": {"available_next_diagnostics": [
        {"request": {"diagnostic": "probe"}}]}}, rounds_left=0, terminal=False,
        clock_mode="explicit", token_count=0)
    assert not operation_cards.build_menu(packet)[0]["cards"][0]["available"]
    def complete(request):
        calls.append(request)
        assert "Explicit generation clock" in request.system_prompt
        reply = command("commit_token") if len(calls) > 1 else {"version": "0.1", "decision": "noop"}
        return ControllerProviderResponse(json.dumps(reply), "fake", "fake")
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=complete)
    client = ProviderControllerClient(provider, packet_view="compact", action_view="cards")
    assert client.invoke(packet).meta["generation_action"] == "commit_token"
    assert client.latest_trace()["attempt_count"] == 2
