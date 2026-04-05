import os
import unittest
from unittest.mock import patch

from SpiralInterventionLab.tasks import (
    MiniLMSemanticCritic,
    SpiralConstrainedRewriteEnv,
    SpiralDigitCopyEnv,
    SpiralDigitTransformEnv,
    SpiralEntailmentReasoningEnv,
    SpiralSentenceOrderingEnv,
    SpiralStructuredSummaryEnv,
)


class _StubSemanticCritic:
    mode = "stub"
    model_name = "stub-minilm"

    def score(self, *, reference_text: str, candidate_text: str) -> float:
        if not candidate_text:
            return 0.0
        if "Mira" in candidate_text or "museum" in candidate_text.lower():
            return 0.81
        return 0.44


class _StubFeatureScanWorker:
    def latent_feature_scan(self, *, feature_groups, max_features_per_group=4, max_surface_hits=2):
        return {
            "prototype_mode": "token_embedding_mean",
            "surface_count": 2,
            "group_count": 1,
            "mean_alignment": 0.18,
            "max_alignment": 0.24,
            "groups": [
                {
                    "group": "required_terms",
                    "polarity": "promote",
                    "feature_kind": "required_term",
                    "feature_count": 2,
                    "mean_alignment": 0.18,
                    "top_features": [
                        {
                            "feature": "Mira",
                            "alignment": 0.24,
                            "surface_id": "s_resid_pre_l4_last",
                            "coverage_progress": 0.0,
                        }
                    ],
                }
            ],
            "top_feature_hits": [
                {
                    "group": "required_terms",
                    "feature": "Mira",
                    "polarity": "promote",
                    "surface_id": "s_resid_pre_l4_last",
                    "alignment": 0.24,
                    "coverage_progress": 0.0,
                }
            ],
        }


class _FakeMiniLMTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        import torch

        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }


class _FakeMiniLMModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        import torch

        batch, seq_len = encoded["input_ids"].shape
        hidden = torch.ones((batch, seq_len, 4), dtype=torch.float32, device=encoded["input_ids"].device)
        return type("Outputs", (), {"last_hidden_state": hidden})()


class TestSpiralDigitTransformEnv(unittest.TestCase):
    def test_reset_is_seed_reproducible(self):
        env = SpiralDigitTransformEnv(min_digits=5, max_digits=5)

        prompt_a = env.reset(17)
        episode_a = env.current_episode
        prompt_b = env.reset(17)
        episode_b = env.current_episode

        self.assertIsNotNone(episode_a)
        self.assertIsNotNone(episode_b)
        self.assertEqual(prompt_a, prompt_b)
        self.assertEqual(episode_a.source_text, episode_b.source_text)
        self.assertEqual(episode_a.target_text, episode_b.target_text)

    def test_transform_rule_matches_reverse_then_increment(self):
        self.assertEqual(SpiralDigitTransformEnv.transform_digits("7041"), "2518")

    def test_score_and_done_require_exact_match(self):
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        env.reset(3)
        self.assertIsNotNone(env.current_episode)
        target = env.current_episode.target_text

        self.assertEqual(env.score(target), 1.0)
        self.assertEqual(env.score(target[:-1]), 0.0)
        self.assertTrue(env.done(target))
        self.assertFalse(env.done(target + "0"))

    def test_task_feedback_reports_partial_progress_and_violations(self):
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        env.reset(11)
        self.assertIsNotNone(env.current_episode)
        target = env.current_episode.target_text
        partial = target[:2]
        wrong = f"{target[:2]}x"

        partial_feedback = env.task_feedback(partial)
        wrong_feedback = env.task_feedback(wrong)

        self.assertAlmostEqual(partial_feedback["partial_score"], 2 / len(target))
        self.assertEqual(partial_feedback["progress_label"], "progressing")
        self.assertAlmostEqual(wrong_feedback["partial_score"], 2 / len(target))
        self.assertEqual(wrong_feedback["progress_label"], "regressing")
        self.assertIn("non_digit_output", wrong_feedback["constraint_violations"])

    def test_stop_checker_and_worker_runtime_kwargs_are_episode_ready(self):
        env = SpiralDigitTransformEnv(min_digits=4, max_digits=4)
        env.reset(29)
        self.assertIsNotNone(env.current_episode)
        target = env.current_episode.target_text

        self.assertFalse(env.stop_checker(target[:-1]))
        self.assertTrue(env.stop_checker(target))

        kwargs = env.worker_runtime_kwargs()
        self.assertEqual(kwargs["task_id"], env.task_id)
        self.assertEqual(kwargs["max_generated_tokens"], len(target))
        self.assertEqual(kwargs["decode_constraint"], "digits_only")
        self.assertTrue(kwargs["stop_checker"](target))
        self.assertEqual(kwargs["task_feedback_fn"](target)["partial_score"], 1.0)


class TestSpiralDigitCopyEnv(unittest.TestCase):
    def test_reset_is_seed_reproducible(self):
        env = SpiralDigitCopyEnv(min_digits=5, max_digits=5)

        prompt_a = env.reset(17)
        episode_a = env.current_episode
        prompt_b = env.reset(17)
        episode_b = env.current_episode

        self.assertIsNotNone(episode_a)
        self.assertIsNotNone(episode_b)
        self.assertEqual(prompt_a, prompt_b)
        self.assertEqual(episode_a.source_text, episode_b.source_text)
        self.assertEqual(episode_a.target_text, episode_b.target_text)

    def test_score_feedback_and_worker_runtime_kwargs(self):
        env = SpiralDigitCopyEnv(min_digits=4, max_digits=4)
        env.reset(29)
        self.assertIsNotNone(env.current_episode)
        target = env.current_episode.target_text
        partial = target[:2]
        wrong = f"{target[:2]}x"

        self.assertEqual(env.score(target), 1.0)
        self.assertFalse(env.done(target[:-1]))
        self.assertTrue(env.done(target))
        self.assertAlmostEqual(env.task_feedback(partial)["partial_score"], 2 / len(target))
        self.assertEqual(env.task_feedback(partial)["progress_label"], "progressing")
        self.assertEqual(env.task_feedback(wrong)["progress_label"], "regressing")
        self.assertIn("non_digit_output", env.task_feedback(wrong)["constraint_violations"])
        self.assertFalse(env.stop_checker(target[:-1]))
        self.assertTrue(env.stop_checker(target))

        kwargs = env.worker_runtime_kwargs()
        self.assertEqual(kwargs["task_id"], env.task_id)
        self.assertEqual(kwargs["max_generated_tokens"], len(target))
        self.assertEqual(kwargs["decode_constraint"], "digits_only")
        self.assertEqual(kwargs["task_feedback_fn"](target)["partial_score"], 1.0)


class TestSpiralSentenceOrderingEnv(unittest.TestCase):
    def test_ordering_feedback_and_runtime_kwargs(self):
        env = SpiralSentenceOrderingEnv()
        env.reset(7)
        self.assertIsNotNone(env.current_episode)
        target = env.current_episode.target_text
        partial = target[:2]

        self.assertEqual(env.score(target), 1.0)
        self.assertAlmostEqual(env.score(partial), 2 / 3)
        self.assertTrue(env.done(target))
        self.assertFalse(env.done(partial))
        self.assertAlmostEqual(env.task_feedback(partial)["partial_score"], 2 / 3)
        self.assertEqual(env.task_feedback(partial)["progress_label"], "progressing")
        self.assertTrue(env.stop_checker(target))

        kwargs = env.worker_runtime_kwargs()
        self.assertEqual(kwargs["task_id"], env.task_id)
        self.assertEqual(kwargs["decode_constraint"], "digits_only")
        self.assertEqual(kwargs["max_generated_tokens"], 3)


class TestSpiralEntailmentReasoningEnv(unittest.TestCase):
    def test_entailment_feedback_and_runtime_kwargs(self):
        env = SpiralEntailmentReasoningEnv()
        env.reset(5)
        self.assertIsNotNone(env.current_episode)
        target = f"{env.current_episode.label_digit}{env.current_episode.reason_digit}"
        partial = target[:1]
        wrong_reason = f"{target[0]}9"

        self.assertEqual(env.score(target), 1.0)
        self.assertAlmostEqual(env.score(partial), 0.6)
        self.assertAlmostEqual(env.score(wrong_reason), 0.6)
        self.assertTrue(env.done(target))
        self.assertFalse(env.done(partial))
        self.assertEqual(env.task_feedback(partial)["progress_label"], "progressing")
        self.assertEqual(env.worker_runtime_kwargs()["decode_constraint"], "digits_only")
        self.assertEqual(env.worker_runtime_kwargs()["max_generated_tokens"], 2)


class TestSpiralConstrainedRewriteEnv(unittest.TestCase):
    def test_rewrite_feedback_and_runtime_kwargs(self):
        env = SpiralConstrainedRewriteEnv(semantic_critic=_StubSemanticCritic())
        env.reset(3)
        self.assertIsNotNone(env.current_episode)
        episode = env.current_episode
        candidate = " ".join(episode.required_terms) + "."
        bad_candidate = f"{candidate} {episode.forbidden_terms[0]}"
        partial_candidate = " ".join(episode.required_terms[:2])

        self.assertEqual(env.score(candidate), 1.0)
        self.assertEqual(env.done(candidate), True)
        self.assertLess(env.score(bad_candidate), 1.0)
        self.assertIn("forbidden_terms_present", env.task_feedback(bad_candidate)["constraint_violations"])
        feedback = env.task_feedback(partial_candidate)
        self.assertEqual(feedback["required_term_recall"], 2 / len(episode.required_terms))
        self.assertIn("required_term_span_progress", feedback)
        self.assertIn("required_term_span_progress_by_term", feedback)
        self.assertEqual(feedback["forbidden_term_clean"], 1.0)
        self.assertTrue(feedback["budget_ok"])
        self.assertEqual(feedback["word_budget_score"], 1.0)
        self.assertEqual(feedback["required_terms_present"], list(episode.required_terms[:2]))
        self.assertEqual(feedback["entity_recall_terms"], list(episode.required_terms[2:]))
        self.assertEqual(set(feedback["entity_recall_progress_by_term"]), set(episode.required_terms[2:]))
        self.assertNotIn("semantic_progress_score", feedback)
        kwargs = env.worker_runtime_kwargs()
        self.assertGreater(kwargs["max_generated_tokens"], episode.max_words)
        self.assertIn("observer_check_fn", kwargs)
        observer = kwargs["observer_check_fn"](
            partial_candidate,
            task_feedback=feedback,
            trigger="coverage_progress",
            worker_runtime=_StubFeatureScanWorker(),
        )
        self.assertIsNotNone(observer)
        self.assertIn("score", observer)
        self.assertIn("coverage_weight", observer)
        self.assertIn("latent_feature_scan", observer)
        self.assertEqual(observer["latent_feature_scan"]["groups"][0]["group"], "required_terms")
        full_feedback = kwargs["task_feedback_fn"](candidate)
        self.assertEqual(full_feedback["partial_score"], 1.0)
        self.assertEqual(full_feedback["required_term_span_progress"], 1.0)


class TestSpiralStructuredSummaryEnv(unittest.TestCase):
    def test_summary_feedback_and_runtime_kwargs(self):
        env = SpiralStructuredSummaryEnv(semantic_critic=_StubSemanticCritic())
        env.reset(9)
        self.assertIsNotNone(env.current_episode)
        episode = env.current_episode
        summary_line = f"summary: {' '.join(episode.required_summary_terms)}."
        keywords_line = f"keywords: {episode.required_keywords[0]}, {episode.required_keywords[1]}"
        candidate = f"{summary_line}\n{keywords_line}"
        partial = summary_line

        self.assertEqual(env.score(candidate), 1.0)
        self.assertTrue(env.done(candidate))
        self.assertLess(env.score(partial), 1.0)
        partial_feedback = env.task_feedback(partial)
        self.assertIn("missing_keywords_line", partial_feedback["constraint_violations"])
        self.assertIn("summary_term_span_progress", partial_feedback)
        self.assertIn("keyword_recall", partial_feedback)
        full_feedback = env.task_feedback(candidate)
        self.assertNotIn("semantic_progress_score", full_feedback)
        kwargs = env.worker_runtime_kwargs()
        self.assertGreater(kwargs["max_generated_tokens"], episode.max_summary_words)
        self.assertIn("observer_check_fn", kwargs)
        observer = kwargs["observer_check_fn"](
            candidate,
            task_feedback=full_feedback,
            trigger="coverage_progress",
            worker_runtime=_StubFeatureScanWorker(),
        )
        self.assertIsNotNone(observer)
        self.assertIn("score", observer)
        self.assertIn("latent_feature_scan", observer)
        self.assertEqual(kwargs["task_feedback_fn"](candidate)["partial_score"], 1.0)


class TestMiniLMSemanticCritic(unittest.TestCase):
    def test_load_components_overrides_offline_env_flags(self):
        seen = {}

        def fake_tokenizer_loader(model_name, *args, **kwargs):
            seen["tokenizer_env"] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            }
            seen["tokenizer_kwargs"] = dict(kwargs)
            return _FakeMiniLMTokenizer()

        def fake_model_loader(model_name, *args, **kwargs):
            seen["model_env"] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            }
            seen["model_kwargs"] = dict(kwargs)
            return _FakeMiniLMModel()

        critic = MiniLMSemanticCritic()
        with patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
            clear=False,
        ):
            with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tokenizer_loader):
                with patch("transformers.AutoModel.from_pretrained", side_effect=fake_model_loader):
                    score = critic.score(reference_text="Mira sends budget", candidate_text="Mira sends budget")
            restored_env = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            }

        self.assertGreater(score, 0.9)
        self.assertEqual(seen["tokenizer_env"]["HF_HUB_OFFLINE"], "0")
        self.assertEqual(seen["tokenizer_env"]["TRANSFORMERS_OFFLINE"], "0")
        self.assertEqual(seen["model_env"]["HF_DATASETS_OFFLINE"], "0")
        self.assertIs(seen["tokenizer_kwargs"]["local_files_only"], False)
        self.assertIs(seen["model_kwargs"]["local_files_only"], False)
        self.assertEqual(restored_env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(restored_env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(restored_env["HF_DATASETS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
