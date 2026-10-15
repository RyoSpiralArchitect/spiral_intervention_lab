from __future__ import annotations

import json
from pathlib import Path

import pytest

from SpiralInterventionLab.examples.replay_orphan_blueprint_ceiling import _edited_top1, load_source


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _blueprint(term: str) -> dict:
    return {"objective_term": term, "first_piece": f" {term}", "first_token_id": 7,
            "candidate_key": f"entity_insert:{term.lower()}:source_body:near_reachable"}


def _episode(*, replayed_term: str | None = None) -> list[dict]:
    events = [
        {"event": "episode_start", "prompt": "P"},
        {"event": "controller_observation", "step": 1, "generated_tail": " A"},
        {"event": "controller_diagnostic_result", "diagnostic": "target_entity_insertion_probe", "recorded_step": 1},
        {"event": "controller_diagnostic_result", "diagnostic": "entity_insertion_operator_candidate_review",
         "recorded_step": 1, "candidate_blueprints": [_blueprint("Selim"), _blueprint("send")]},
        {"event": "controller_observation", "step": 2, "generated_tail": " A b"},
    ]
    if replayed_term is not None:
        events.append({"event": "controller_diagnostic_result", "diagnostic": "operator_diagnostic_replay",
                       "evidence_rows": [{"intended_term": replayed_term}]})
    events.append({"event": "episode_end", "output": " A b."})
    return events


def test_load_source_finds_orphan_blueprints_and_prefixes(tmp_path: Path) -> None:
    source = load_source(_write(tmp_path / "c1.jsonl", _episode()))

    assert source["prompt"] == "P"
    assert source["prefixes"] == {1: " A", 2: " A b"}
    assert source["final_output"] == " A b."
    assert source["review_line"] == 4 and source["probe_line"] == 3
    assert source["orphan_terms"] == ["Selim", "send"]


def test_load_source_excludes_blueprints_replayed_after_review(tmp_path: Path) -> None:
    source = load_source(_write(tmp_path / "c1.jsonl", _episode(replayed_term="send")))

    assert source["orphan_terms"] == ["Selim"]


def test_load_source_rejects_review_without_preceding_probe(tmp_path: Path) -> None:
    events = [event for event in _episode() if event.get("diagnostic") != "target_entity_insertion_probe"]

    with pytest.raises(ValueError, match="no preceding entity probe"):
        load_source(_write(tmp_path / "c1.jsonl", events))


def test_load_source_rejects_multiple_episodes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one episode"):
        load_source(_write(tmp_path / "c1.jsonl", _episode() + _episode()))


def test_edited_top1_uses_edited_logit_not_listing_order() -> None:
    replay = {"topk_token_diff": [
        {"piece": " sending", "token_id": 1, "edited_logit": 3.0, "edited_prob": 0.4},
        {"piece": " Sel", "token_id": 2, "edited_logit": 4.5, "edited_prob": 0.5},
    ]}

    assert _edited_top1(replay) == {"piece": " Sel", "token_id": 2, "edited_prob": 0.5}
    assert _edited_top1({}) is None
