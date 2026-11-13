"""Offline behavioral checks for model/policy-specific planner calibration."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from planner import MODELS, TaskEstimator, TaskPlan


class PlannerCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.task = {"task_id": "short", "task_type": "math",
                     "instruction": "Compute 2 + 3.", "reward": 1}

    def test_cost_adjustment_preserves_task_size_signal(self):
        estimator = TaskEstimator()
        short = estimator.estimate(self.task, self.workspace)
        long_task = dict(self.task, task_id="long", instruction="x" * 4000)
        long = estimator.estimate(long_task, self.workspace)
        estimator.observe_plan(short, short.raw_predicted_tokens * 3)
        new_short = estimator.estimate(self.task, self.workspace)
        new_long = estimator.estimate(long_task, self.workspace)
        self.assertAlmostEqual(new_short.predicted_tokens / short.predicted_tokens, 1.5)
        self.assertAlmostEqual(new_long.predicted_tokens / long.predicted_tokens, 1.5)

    def test_model_policy_and_kind_do_not_share_cost_observations(self):
        estimator = TaskEstimator()
        original = estimator.estimate(self.task, self.workspace)
        variants = [
            replace(original, model=MODELS["large_reason"]),
            replace(original, policy_id="other-caps-and-tools"),
            replace(original, kind="search"),
        ]
        for variant in variants:
            estimator.observe_plan(variant, original.raw_predicted_tokens * 10)
        self.assertEqual(estimator.estimate(self.task, self.workspace).predicted_tokens,
                         original.predicted_tokens)

    def test_repeated_updates_use_raw_snapshot_not_adapted_prediction(self):
        estimator = TaskEstimator()
        original = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(original, original.raw_predicted_tokens * 2)
        second = estimator.estimate(self.task, self.workspace)
        self.assertEqual(second.raw_predicted_tokens, original.raw_predicted_tokens)
        estimator.observe_plan(second, original.raw_predicted_tokens * 2)
        third = estimator.estimate(self.task, self.workspace)
        self.assertAlmostEqual(third.predicted_tokens / original.predicted_tokens, 1.4)

    def test_public_and_unknown_verdicts_do_not_train_hidden_correctness(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        for source, verdict in [("public_tests", True), ("public_tests", False),
                                ("unknown", False), ("unknown", True)]:
            estimator.observe_plan(plan, None, usage_status="unknown",
                                   verdict=verdict, verifier_kind=source,
                                   produced_answer=True)
        after = estimator.estimate(self.task, self.workspace)
        self.assertEqual(after.success_probability, plan.success_probability)
        self.assertFalse(estimator.outcome_observations)

    def test_trusted_verdict_updates_correctness_even_with_zero_cost(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(plan, 0, verdict=False, verifier_kind="oracle",
                               produced_answer=True)
        after = estimator.estimate(self.task, self.workspace)
        self.assertLess(after.success_probability, plan.success_probability)
        self.assertLess(after.predicted_tokens, plan.predicted_tokens)

    def test_availability_counts_both_answers_and_missing_answers(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(plan, None, produced_answer=False)
        missing = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(plan, None, produced_answer=True,
                               verdict=True, verifier_kind="public_tests")
        recovered = estimator.estimate(self.task, self.workspace)
        self.assertAlmostEqual(missing.success_probability / plan.success_probability, 4 / 5)
        self.assertAlmostEqual(recovered.success_probability / plan.success_probability, 5 / 6)
        self.assertFalse(estimator.outcome_observations)

    def test_missing_answer_is_not_double_counted_as_wrong_answer(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(plan, None, produced_answer=False,
                               verdict=False, verifier_kind="official")
        after = estimator.estimate(self.task, self.workspace)
        self.assertAlmostEqual(after.success_probability / plan.success_probability, 4 / 5)
        self.assertFalse(estimator.outcome_observations)

    def test_unreconciled_and_invalid_costs_do_not_change_cost_scale(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        estimator.observe_plan(plan, 100000, usage_status="reservation")
        for invalid in [float("nan"), float("inf"), -1, None]:
            estimator.observe_plan(plan, invalid)
        after = estimator.estimate(self.task, self.workspace)
        self.assertEqual(after.predicted_tokens, plan.predicted_tokens)
        self.assertFalse(estimator.cost_observations)

    def test_hidden_and_availability_observations_are_scoped(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        other = replace(plan, policy_id="different-policy")
        estimator.observe_plan(other, None, produced_answer=False)
        estimator.observe_plan(other, None, produced_answer=True,
                               verdict=False, verifier_kind="hidden_tests")
        self.assertEqual(estimator.estimate(self.task, self.workspace).success_probability,
                         plan.success_probability)

    def test_reward_changes_cost_only_through_retry_policy(self):
        estimator = TaskEstimator()
        low = estimator.estimate(self.task, self.workspace)
        same_route = estimator.estimate(dict(self.task, reward=3), self.workspace)
        self.assertEqual(low.raw_predicted_tokens, same_route.raw_predicted_tokens)
        retry = estimator.estimate(dict(self.task, reward=4), self.workspace)
        self.assertNotEqual(low.policy_id, retry.policy_id)
        estimator.observe_plan(retry, retry.raw_predicted_tokens * 10)
        self.assertEqual(estimator.estimate(self.task, self.workspace).predicted_tokens,
                         low.predicted_tokens)
        no_retry = TaskEstimator(policy_id="tools-v2", allow_reasoning_retry=False)
        a = no_retry.estimate(self.task, self.workspace)
        b = no_retry.estimate(dict(self.task, reward=5), self.workspace)
        self.assertEqual(a.policy_id, b.policy_id)
        self.assertEqual(a.raw_predicted_tokens, b.raw_predicted_tokens)

    def test_legacy_adapter_preserves_availability_and_cost_updates(self):
        estimator = TaskEstimator()
        plan = estimator.estimate(self.task, self.workspace)
        estimator.observe("math", 1000, False)
        after = estimator.estimate(self.task, self.workspace)
        self.assertGreater(after.predicted_tokens, plan.predicted_tokens)
        self.assertLess(after.success_probability, plan.success_probability)
        estimator.observe("math", 10, True, verified=True)
        self.assertFalse(estimator.outcome_observations)

    def test_old_task_plan_construction_remains_supported(self):
        plan = TaskPlan("old", "math", 1, 20, .8, MODELS["small_plain"])
        self.assertIsNone(plan.raw_predicted_tokens)
        estimator = TaskEstimator()
        estimator.observe_plan(plan, 40)
        self.assertEqual(list(estimator.cost_observations.values()), [[2.0]])


if __name__ == "__main__":
    unittest.main()

