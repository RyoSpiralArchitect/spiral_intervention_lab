import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from SpiralInterventionLab.examples import structural_controller_sampling as sampling
from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.runtime import reach_datasheet
from SpiralInterventionLab.runtime.response_probe import identity

RECEIPTS = Path(__file__).resolve().parents[2] / "results" / "structural_rescue_contract_20261103"


def test_the_plain_arm_is_the_pr39_prompt_and_packet():
    assert sampling.prompt("controller", 0.08) == v2.SYSTEM_PROMPT
    assert "0.12 of the" in sampling.prompt("controller", 0.12)
    assert sampling.prompt("controller_datasheet", 0.08).startswith(sampling.prompt("controller_mechanics", 0.08))
    stored = json.loads((RECEIPTS / "rescue" / "gen_058_observation.json").read_text())
    trace = next(c for c in json.loads((RECEIPTS / "rescue" / "results.json").read_text())["cases"]
                 if c["case_id"] == "gen_058")["policies"]["controller"]["selection"]
    case = {"case_id": "gen_058", "structure_readout": stored["structure_readout"]}
    assert identity("packet:", sampling.arm_packet("controller", stored, case, None, opaque=False)) \
        == trace["packet_sha256"]


def test_datasheet_arm_adds_entries_and_opaque_ids_hide_the_split():
    stored = json.loads((RECEIPTS / "rescue" / "gen_058_observation.json").read_text())
    case = {"case_id": "gen_058", "structure_readout": stored["structure_readout"], "b0_winner_id": 1603,
            "eligible_sources": [{"token_id": r["sources"][0]["token_id"]} for r in stored["recipes"]
                                 if r["kind"] == "readout"]}
    sheet = reach_datasheet.build([], cap=0.08, provenance={})
    shown = sampling.arm_packet("controller_datasheet", stored, case, sheet, opaque=True)
    assert shown["case_id"] == sampling.opaque_id("gen_058") and "gen" not in shown["case_id"]
    readouts = [r for r in shown["recipes"] if r["kind"] == "readout"]
    assert [r["datasheet"]["reach"] for r in readouts] == ["uncalibrated"] * 3
    assert [r["datasheet"]["ends_output"] for r in readouts] == [False, False, True]
    assert all("datasheet" not in r for r in stored["recipes"])


class Provider:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def complete(self, request):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, usage={}, metadata={}, raw=SimpleNamespace(model="snap-1"))


class Timeout(Exception):
    pass


def test_one_retry_only_without_a_response_and_invalid_choices_are_kept():
    packet = {"recipes": [{"recipe_id": "r1"}]}
    Timeout.__name__ = "APITimeoutError"
    retried = sampling.ask(Provider(Timeout(), '{"decision":"probe","recipe_id":"r1"}'), "s", packet)
    assert retried["status"] == "selected" and retried["recipe_id"] == "r1" and len(retried["attempts"]) == 1
    assert retried["resolved_model"] == "snap-1"
    failed = sampling.ask(Provider(ValueError("bad request")), "s", packet)
    assert failed["status"] == "error" and failed["recipe_id"] is None
    invalid = sampling.ask(Provider('{"decision":"probe","recipe_id":"other"}'), "s", packet)
    assert invalid["status"] == "invalid" and invalid["response"]
    assert sampling.ask(Provider('{"decision":"noop"}'), "s", packet)["status"] == "noop"
