import unittest

from SpiralInterventionLab.tasks import SpiralDigitTransformEnv


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
        self.assertEqual(kwargs["max_generated_tokens"], 4)
        self.assertTrue(kwargs["stop_checker"](target))
        self.assertEqual(kwargs["task_feedback_fn"](target)["partial_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
