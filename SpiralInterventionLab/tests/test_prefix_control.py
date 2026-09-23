from copy import deepcopy

import pytest

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload
from SpiralInterventionLab.runtime import prefix_control
from SpiralInterventionLab.runtime.compiler import StepContext
from SpiralInterventionLab.runtime.loop import run_episode, InMemoryStructuredLogger
from SpiralInterventionLab.runtime.schema import parse_controller_command
from SpiralInterventionLab.tests.test_controller_runtime import (
    _ToyWorkerRuntime, _ToyTaskEnv, FakeRuntimeState, FakeAdapter, _resid_command,
)


def command(index, *, hold=True):
    return {"version": "0.1", "decision": "noop", "meta": {
        "generation_action": "hold_prefix" if hold else "advance",
        "diagnostic_request": {"diagnostic": "candidate_action", "action_id": f"action:{index}"}}}


def test_generation_control_is_compact_and_does_not_mutate_packet():
    packet = {"strategy_hints": {"existing": True}}
    annotated = prefix_control.annotate(packet, rounds_left=2, terminal=False)
    assert packet == {"strategy_hints": {"existing": True}}
    assert _compact_controller_payload(annotated)["strategy_hints"]["generation_control"] == annotated["strategy_hints"]["generation_control"]
    assert not annotated["strategy_hints"]["generation_control"]["production_apply_allowed"]


@pytest.mark.parametrize("rounds,hold,expected", [(0, True, [1, 2]), (2, False, [1, 2]),
    (1, True, [1, 1, 2]), (2, True, [1, 1, 1, 2]), (99, True, [1, 1, 1, 2])])
def test_explicit_hold_preserves_token_clock_with_shared_bounded_rounds(rounds, hold, expected):
    state = FakeRuntimeState()
    class Worker(_ToyWorkerRuntime):
        ticks = 0
        effects = 0
        def tick_ttl(self):
            self.ticks += 1
        def observe_recent_effects(self):
            self.effects += 1
    worker = Worker(state)
    prefixes, budgets = [], []
    class Controller:
        def invoke(self, packet):
            prefixes.append(worker.steps)
            budgets.append(deepcopy(packet["budget"]))
            return parse_controller_command(command(len(prefixes), hold=hold))
    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=rounds, logger=logger)
    assert prefixes == expected and worker.ticks == 2
    assert all(b == budgets[0] for b in budgets)  # no edit budget is spent or replenished
    assert not any(e["event"] == "compiled_edit" for e in logger.events)
    holds = [e for e in logger.events if e["event"] == "controller_prefix_hold"]
    assert sum(e["accepted"] for e in holds) == len(expected) - 2
    if hold:
        assert holds[-1]["blocked_reason"] == "terminal_prefix"


def test_hold_and_positive_handoff_do_not_stack_an_extra_round_budget():
    state = FakeRuntimeState()
    class Worker(_ToyWorkerRuntime):
        def candidate_trial_followup_available(self, results):
            return bool(results)
    worker = Worker(state)
    prefixes = []
    class Controller:
        def invoke(self, packet):
            prefixes.append(worker.steps)
            return command(len(prefixes), hold=len(prefixes) == 1)
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2)
    assert prefixes == [1, 1, 1, 2]


def test_duplicate_hold_request_cannot_spend_replays_or_loop_forever():
    worker = _ToyWorkerRuntime(FakeRuntimeState())
    prefixes = []
    class Controller:
        def invoke(self, packet):
            prefixes.append(worker.steps)
            return command(1)
    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=worker.runtime_state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2, logger=logger)
    assert prefixes == [1, 1, 2]
    assert any(e["event"] == "controller_diagnostic_veto" for e in logger.events)
    assert any(e["event"] == "controller_prefix_hold" and e["blocked_reason"] == "no_fresh_requested_result" for e in logger.events)


@pytest.mark.parametrize("request_kind", ["tool", "observer"])
def test_explicit_hold_can_read_fresh_tool_or_observer_result_at_same_prefix(request_kind):
    worker = _ToyWorkerRuntime(FakeRuntimeState())
    prefixes, observed = [], []

    class Controller:
        def invoke(self, packet):
            prefixes.append(worker.steps)
            if len(prefixes) == 1:
                meta = {"generation_action": "hold_prefix"}
                if request_kind == "tool":
                    meta["tool_requests"] = [{"tool": "tokenize_terms", "terms": ["x"]}]
                else:
                    meta["observer_check_request"] = {"kind": "semantic_progress"}
                return {"version": "0.1", "decision": "noop", "meta": meta}
            observed.append(packet.get("latest_tool_results") if request_kind == "tool"
                            else packet.get("latest_observer_check"))
            return {"version": "0.1", "decision": "noop"}

    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=worker.runtime_state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2, logger=logger)
    assert prefixes == [1, 1, 2]
    assert observed[0] and not any(e["event"] == "compiled_edit" for e in logger.events)
    assert any(e["event"] == "controller_prefix_hold" and e["accepted"] for e in logger.events)


@pytest.mark.parametrize("result", [{"status": "blocked"}, {"status": "held"},
    {"status": "incomplete"}, {"status": "no_rows"}, {"status": "ok", "state_restored": False},
    {"status": "ok", "evidence": {"state_restored": False}}])
def test_failed_or_empty_diagnostic_cannot_hold(result):
    assert not prefix_control.evaluate(command(1), [{}], [result], rounds_left=2, terminal=False)["accepted"]
    assert not prefix_control.evaluate(command(1), [], [], rounds_left=2, terminal=False)["accepted"]


def test_hold_result_does_not_authorize_an_edit():
    unsafe = _resid_command()
    unsafe["meta"]["generation_action"] = "hold_prefix"
    report = prefix_control.evaluate(unsafe, [{}], [{"status": "ok"}], rounds_left=2, terminal=False)
    assert report["blocked_reason"] == "hold_requires_noop" and not report["production_apply_allowed"]


def test_hold_does_not_falsely_decrement_legacy_patience_in_logs():
    state = FakeRuntimeState()
    worker = _ToyWorkerRuntime(state)
    class Controller:
        calls = 0
        def invoke(self, packet):
            self.calls += 1
            return command(self.calls)
    logger = InMemoryStructuredLogger()
    run_episode(_ToyTaskEnv(), worker, Controller(),
        StepContext(packet={}, runtime_state=state, adapter=FakeAdapter(), traces={}, stats={}),
        max_candidate_handoff_rounds=2, max_controller_diagnostic_rethink_rounds=2, logger=logger)
    rounds = [e for e in logger.events if e["event"] == "controller_loop_patience" and e["step"] == 0]
    assert [e["candidate_handoff_budget_left"] for e in rounds] == [2, 1, 0]
    assert [e["loop_patience_budget_left"] for e in rounds] == [2, 2, 2]
