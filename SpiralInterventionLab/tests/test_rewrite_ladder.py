import pytest

from SpiralInterventionLab.examples.digit_transform_e2e import create_task_env
from SpiralInterventionLab.examples.rewrite_ladder_baseline import build_fixtures, summarize
from SpiralInterventionLab.tasks import SpiralConstrainedRewriteEnv, SpiralRewriteLadderEnv
from SpiralInterventionLab.tasks.rewrite_ladder import REWRITE_LADDER_SEEDS


# Scoring-only witnesses, not runtime seeds or controller few-shot examples.
WITNESSES = (
    "Before noon, Mira can send Omar budget; Leena can bring Priya contract.",
    "Before dusk, Nora can take Ivo sample; Selim can send Yara report.",
    "Before lunch, Ada can bring Niko cable; Rami can send Vera diagram.",
)


def test_legacy_seed7_prompt_and_generation_budget_unchanged():
    env = SpiralConstrainedRewriteEnv()
    assert env.reset(7) == (
        "You are solving a constrained rewrite task.\n"
        "Rewrite the source into one shorter sentence.\n"
        "Keep these terms: Mira, send, budget, Omar\n"
        "Do not use these terms: in order to, should\n"
        "Use at most 9 words.\nReturn only the rewritten sentence.\n"
        "SOURCE: In order to stay on schedule, Mira should send the budget draft to Omar before lunch.\n"
        "ANSWER:"
    )
    assert env.worker_runtime_kwargs()["max_generated_tokens"] == 18
    assert env.done("Mira will send Omar the budget draft before lunch.")


@pytest.mark.parametrize("level", (1, 2, 3))
def test_ladder_is_feasible_and_runtime_receives_no_answer(level):
    for seed, witness in zip(REWRITE_LADDER_SEEDS, WITNESSES, strict=True):
        env = create_task_env(f"constrained_rewrite_l{level}")
        prompt = env.reset(seed)
        assert isinstance(env, SpiralRewriteLadderEnv)
        assert witness not in prompt
        assert witness not in str(env.task_feedback(""))
        assert env.done(witness)
        assert env.score(witness) == 1.0
        feedback = env.task_feedback(witness)
        assert feedback["constraint_violations"] == []
        assert feedback["rewrite_ladder"]["semantic_preservation_certified"] is False
        assert env.worker_runtime_kwargs()["max_generated_tokens"] == 64
        assert not env.stop_checker(" ".join(["word"] * 25))
        assert env.stop_checker(witness)


def test_ladder_changes_one_axis_per_increment_and_repeats_deterministically():
    for seed in REWRITE_LADDER_SEEDS:
        envs = [SpiralRewriteLadderEnv(level=level) for level in (1, 2, 3)]
        for env in envs:
            assert env.reset(seed) == env.reset(seed)
        episodes = [env.current_episode for env in envs]
        assert len({ep.source_text for ep in episodes}) == 1
        assert len({ep.required_terms for ep in episodes}) == 1
        assert episodes[0].max_words > episodes[1].max_words == episodes[2].max_words
        assert episodes[0].forbidden_terms == episodes[1].forbidden_terms
        assert set(episodes[1].forbidden_terms) < set(episodes[2].forbidden_terms)


def test_each_failure_axis_is_observable_without_semantic_success_claim():
    env = SpiralRewriteLadderEnv(level=3)
    env.reset(1)
    assert env.done(WITNESSES[0])
    missing = env.task_feedback(WITNESSES[0].replace("Mira", "someone"))
    assert missing["missing_required_terms"] == ["Mira"]
    forbidden = env.task_feedback(WITNESSES[0].replace("can", "must"))
    assert forbidden["forbidden_terms_present"] == ["must"]
    overflow = env.task_feedback(WITNESSES[0] + " right now")
    assert overflow["constraint_violations"] == ["over_word_budget"]
    assert not env.done("")


@pytest.mark.parametrize("level", (True, 0, 4, "1", 1.0))
def test_invalid_level_rejected(level):
    with pytest.raises(ValueError):
        SpiralRewriteLadderEnv(level=level)


def test_prespecified_manifest_and_failure_selection():
    fixtures = build_fixtures()
    assert len(fixtures) == 10
    assert len({fixture["condition_id"] for fixture in fixtures}) == 10
    assert [f["benchmark"]["case_id"] for f in fixtures[1:4]] == ["delivery", "review", "workshop"]
    assert [f["max_new_tokens"] for f in fixtures] == [18] + [64] * 9
    rows = [{"condition_id": f["condition_id"], "difficulty_level": f["benchmark"]["difficulty_level"],
             "task_done": True, "truncated": False, "output": "example text.",
             "task_feedback": {"constraint_violations": []}} for f in fixtures]
    assert summarize(rows, fixtures)["selected_condition_id"] is None
    rows[1].update(task_done=False, truncated=True)
    rows[2].update(task_done=False)
    summary = summarize(rows, fixtures)
    assert summary["selected_condition_id"] == rows[2]["condition_id"]
    assert summary["production_apply_allowed"] is False
    rows[0]["task_done"] = False
    assert summarize(rows, fixtures)["selected_condition_id"] is None
    with pytest.raises(ValueError):
        summarize(rows[:-1], fixtures)
