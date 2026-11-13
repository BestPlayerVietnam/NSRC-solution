"""Offline tests; synthetic labels check contracts, not predictive superiority."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from cost_model import (CostModel, assert_unseen_groups, evaluate_cost_model,
    fit_cost_artifact, grouped_fold_ids, load_cost_model, retry_marginal_utility,
    save_cost_artifact)


POLICY = "unit-test-policy"
NAMES = ["size", "kind_math"]


def row(i, *, group=None, model="plain", first=None, total=None, domain="math"):
    first = 10.0 + i if first is None else first
    total = first * 2 if total is None else total
    return {"task_group": group or f"group-{i}", "model": model, "policy_id": POLICY,
            "domain": domain, "features": {"size": float(i), "kind_math": float(domain == "math")},
            "first_call_cost": first, "total_policy_cost": total,
            "first_usage_known": True, "total_usage_known": True, "execution_complete": True}


def fit(rows=None, calibration=(), **kwargs):
    return fit_cost_artifact(rows or [row(i) for i in range(16)], calibration,
                             feature_names=NAMES, policy_id=POLICY, **kwargs)


class CostModelTests(unittest.TestCase):
    def forecast(self, estimator, size=8, **kwargs):
        return estimator.predict({"size": size, "kind_math": 1},
                             model=kwargs.get("model", "plain"),
                             policy_id=kwargs.get("policy_id", POLICY),
                             domain=kwargs.get("domain", "math"))

    def test_separate_first_and_total_and_nonnegative_consistency(self):
        model = CostModel(fit())
        forecast = self.forecast(model)
        self.assertIsNotNone(forecast.first_call_mean)
        self.assertGreaterEqual(forecast.total_policy_mean, forecast.first_call_mean)
        self.assertEqual(forecast.support["first_call_groups"], 16)
        self.assertFalse(forecast.calibrated)
        self.assertIsNone(forecast.first_call_upper)

    def test_model_and_policy_scopes_are_not_silently_mixed(self):
        rows = [row(i) for i in range(12)] + [
            row(i, model="reasoning", first=500 + i, total=1000 + i) for i in range(12)]
        model = CostModel(fit(rows))
        self.assertGreater(self.forecast(model, model="reasoning").first_call_mean,
                           self.forecast(model).first_call_mean * 5)
        self.assertEqual(self.forecast(model, model="missing").fallback_reason, "model_not_trained")
        self.assertEqual(self.forecast(model, policy_id="other").fallback_reason, "policy_mismatch")

    def test_unknown_first_and_total_are_excluded_independently(self):
        rows = [row(i) for i in range(12)]
        rows[0]["first_usage_known"] = False
        rows[0]["first_call_cost"] = None
        rows[1]["total_usage_known"] = False
        rows[1]["total_policy_cost"] = None
        rows[2]["execution_complete"] = False
        rows[2]["total_policy_cost"] = 1000000
        forecast = self.forecast(CostModel(fit(rows)))
        self.assertEqual(forecast.support["first_call_rows"], 11)
        self.assertEqual(forecast.support["total_policy_rows"], 10)

    def test_partial_policy_has_no_total_head_but_keeps_first(self):
        rows = [dict(row(i), execution_complete=False) for i in range(12)]
        forecast = self.forecast(CostModel(fit(rows)))
        self.assertIsNotNone(forecast.first_call_mean)
        self.assertIsNone(forecast.total_policy_mean)
        self.assertIsNone(forecast.extra_cost_if_continued)

    def test_group_split_rejects_overlap_across_models(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit([row(i) for i in range(12)], [row(100, group="group-3", model="plain")])
        rows = [row(i) for i in range(12)] + [row(1, group="group-1")]
        folds = grouped_fold_ids(rows)
        self.assertEqual(len(folds), 12)

    def test_calibration_does_not_select_coefficients(self):
        train = [row(i) for i in range(12)]
        calibration = [row(i % 12, group=f"cal-{i}", first=100 + i, total=200 + i)
                       for i in range(10)]
        uncalibrated = fit(train)
        calibrated = fit(train, calibration)
        for target in ("first_call", "total_policy"):
            self.assertEqual(uncalibrated["models"]["plain"]["heads"][target]["fitted"],
                             calibrated["models"]["plain"]["heads"][target]["fitted"])
        forecast = self.forecast(CostModel(calibrated))
        self.assertTrue(forecast.calibrated)
        self.assertGreater(forecast.first_call_upper, forecast.first_call_mean)
        self.assertGreaterEqual(forecast.total_policy_upper, forecast.first_call_upper)

    def test_repeated_calibration_runs_do_not_inflate_group_count(self):
        calibration = [row(5, group="one-cal", first=100, total=200) for _ in range(20)]
        forecast = self.forecast(CostModel(fit(calibration=calibration)))
        self.assertEqual(forecast.support["first_call_calibration_groups"], 1)
        self.assertFalse(forecast.calibrated)

    def test_repeated_train_group_does_not_reweight_mean(self):
        rows = [row(0, first=10, total=20), row(1, first=90, total=180)]
        artifact = fit(rows, candidates=(("domain_mean", 2),))
        duplicated = fit(rows + [row(0, first=10, total=20)] * 10,
                         candidates=(("domain_mean", 2),))
        self.assertAlmostEqual(self.forecast(CostModel(artifact)).first_call_mean,
                               self.forecast(CostModel(duplicated)).first_call_mean)

    def test_log_ridge_smearing_does_not_floor_log_prediction_to_one(self):
        rows = [row(i, first=.2, total=.4) for i in range(12)]
        artifact = fit(rows, candidates=(("log_ridge", 1),))
        forecast = self.forecast(CostModel(artifact))
        self.assertAlmostEqual(forecast.first_call_mean, .2, places=6)
        self.assertAlmostEqual(forecast.total_policy_mean, .4, places=6)

    def test_domain_shrinkage_and_small_sample_fallback(self):
        rows = [row(0, first=10, total=20, domain="math"),
                row(1, first=100, total=200, domain="coding")]
        artifact = fit(rows)
        self.assertEqual(artifact["models"]["plain"]["heads"]["first_call"]["selection_note"],
                         "small_sample_domain_shrinkage")
        model = CostModel(artifact)
        math_forecast = model.predict({"size": 0, "kind_math": 1}, model="plain",
                                      policy_id=POLICY, domain="math")
        coding_forecast = model.predict({"size": 1, "kind_math": 0}, model="plain",
                                        policy_id=POLICY, domain="coding")
        self.assertLess(math_forecast.first_call_mean, coding_forecast.first_call_mean)

    def test_hurdle_distinguishes_conditional_extra_from_unconditional_cost(self):
        rows = [row(i, first=10, total=30 if i < 4 else 10) for i in range(8)]
        forecast = self.forecast(CostModel(fit(rows)), 3)
        self.assertAlmostEqual(forecast.continuation_probability, .5)
        self.assertAlmostEqual(forecast.extra_cost_if_continued, 20)
        self.assertAlmostEqual(forecast.hurdle_total_mean - forecast.first_call_mean, 10)

    def test_no_continuation_does_not_invent_a_conditional_retry_cost(self):
        rows = [row(i, first=10, total=10) for i in range(8)]
        forecast = self.forecast(CostModel(fit(rows)), 3)
        self.assertEqual(forecast.continuation_probability, 0)
        self.assertIsNone(forecast.extra_cost_if_continued)
        self.assertEqual(forecast.hurdle_total_mean, forecast.first_call_mean)

    def test_nonfinite_or_missing_features_explicitly_fall_back(self):
        model = CostModel(fit())
        for features in ({"size": math.nan, "kind_math": 1}, {"size": 2}):
            result = model.predict(features, model="plain", policy_id=POLICY, domain="math")
            self.assertIsNone(result.first_call_mean)
            self.assertTrue(result.fallback_reason.startswith("invalid_prediction:"))
        self.assertTrue(self.forecast(model, 1000).warnings)

    def test_invalid_known_labels_and_policy_are_rejected(self):
        for changes in ({"first_call_cost": -1}, {"first_call_cost": math.nan},
                        {"total_policy_cost": 0}, {"policy_id": "wrong"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fit([dict(row(0), **changes)])

    def test_load_save_and_artifact_validation(self):
        artifact = fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cost.json"
            save_cost_artifact(artifact, path)
            model = load_cost_model(path, expected_policy_id=POLICY)
            self.assertEqual(model.artifact, artifact)
            self.assertEqual(self.forecast(model).as_dict()["model"], "plain")
            with self.assertRaises(ValueError):
                load_cost_model(path, expected_policy_id="wrong")
            with self.assertRaises(ValueError):
                load_cost_model(path, expected_feature_schema_id="wrong")
        broken = copy.deepcopy(artifact)
        broken["schema_version"] = 999
        with self.assertRaises(ValueError):
            CostModel(broken)
        broken = copy.deepcopy(artifact)
        broken["feature_names"] = ["size", "size"]
        with self.assertRaises(ValueError):
            CostModel(broken)

    def test_evaluation_refuses_old_train_or_calibration_groups(self):
        artifact = fit()
        model = CostModel(artifact)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_cost_model(model, [row(0)])
        report = evaluate_cost_model(model, [row(100, group="fresh")])
        self.assertEqual(report["plain"]["first_call"]["n"], 1)
        self.assertEqual(report["plain"]["first_call"]["upper_coverage_n"], 0)
        assert_unseen_groups([row(1, group="new")], artifact)

    def test_mae_selection_is_explicit_and_uses_training_cv(self):
        artifact = fit(criterion="mae")
        for head in artifact["models"]["plain"]["heads"].values():
            self.assertEqual(head["cv_criterion"], "mae")
            selected = min(report["mae"] for report in head["cv"])
            winner = [report for report in head["cv"]
                      if report["method"] == head["fitted"]["method"]
                      and report["regularization"] == head["fitted"].get(
                          "regularization", head["fitted"].get("strength"))]
            self.assertAlmostEqual(winner[0]["mae"], selected)

    def test_retry_utility_requires_gain_and_has_no_hidden_task_cap(self):
        self.assertIsNone(retry_marginal_utility(
            reward=5, success_gain=None, extra_cost=100, alternative_reward_per_token=.01))
        self.assertGreater(retry_marginal_utility(
            reward=5, success_gain=.5, extra_cost=100, alternative_reward_per_token=.01), 0)
        self.assertLess(retry_marginal_utility(
            reward=5, success_gain=.1, extra_cost=100, alternative_reward_per_token=.01), 0)
        with self.assertRaises(ValueError):
            retry_marginal_utility(
                reward=5, success_gain=1.1, extra_cost=100, alternative_reward_per_token=.01)


if __name__ == "__main__":
    unittest.main()

