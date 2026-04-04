import unittest

from SpiralInterventionLab.tasks import (
    SpiralConstrainedRewriteEnv,
    SpiralDigitCopyEnv,
    SpiralDigitTransformEnv,
    SpiralEntailmentReasoningEnv,
    SpiralSentenceOrderingEnv,
    SpiralStructuredSummaryEnv,
)


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
        env = SpiralConstrainedRewriteEnv()
        env.reset(3)
        self.assertIsNotNone(env.current_episode)
        episode = env.current_episode
        candidate = " ".join(episode.required_terms) + "."
        bad_candidate = f"{candidate} {episode.forbidden_terms[0]}"

        self.assertEqual(env.score(candidate), 1.0)
        self.assertEqual(env.done(candidate), True)
        self.assertLess(env.score(bad_candidate), 1.0)
        self.assertIn("forbidden_terms_present", env.task_feedback(bad_candidate)["constraint_violations"])
        kwargs = env.worker_runtime_kwargs()
        self.assertGreater(kwargs["max_generated_tokens"], episode.max_words)
        self.assertEqual(kwargs["task_feedback_fn"](candidate)["partial_score"], 1.0)


class TestSpiralStructuredSummaryEnv(unittest.TestCase):
    def test_summary_feedback_and_runtime_kwargs(self):
        env = SpiralStructuredSummaryEnv()
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
        self.assertIn("missing_keywords_line", env.task_feedback(partial)["constraint_violations"])
        kwargs = env.worker_runtime_kwargs()
        self.assertGreater(kwargs["max_generated_tokens"], episode.max_summary_words)
        self.assertEqual(kwargs["task_feedback_fn"](candidate)["partial_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
