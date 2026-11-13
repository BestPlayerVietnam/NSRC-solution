"""Trust-boundary tests; no API or execution policy changes."""
import copy
import math
import unittest
from dataclasses import asdict, replace
from cost_model import CostForecast
from forecast_guard import choose_forecast

VALID = CostForecast(model="qwen-test", policy_id="frozen-test",
                     first_call_mean=40, total_policy_mean=100)


class ForecastGuardTests(unittest.TestCase):
    def test_missing_forecast_uses_prior(self):
        selected = choose_forecast(None, 20, 60)
        self.assertEqual((selected.first_call, selected.total_policy), (20, 60))
        self.assertEqual((selected.source, selected.reason), ("prior", "forecast_missing"))

    def test_valid_uncalibrated_points_are_accepted(self):
        self.assertFalse(VALID.calibrated)
        selected = choose_forecast(VALID, 20, 60)
        self.assertEqual((selected.first_call, selected.total_policy), (40, 100))
        self.assertEqual(selected.source, "learned")
        self.assertIsNone(selected.reason)

    def test_unknown_model_and_wrong_policy_keep_explicit_reason(self):
        for reason in ("model_not_trained", "policy_mismatch"):
            forecast = replace(VALID, fallback_reason=reason)
            selected = choose_forecast(forecast, 20, 60)
            self.assertEqual((selected.source, selected.reason), ("prior", reason))
            self.assertEqual(selected.total_policy, 60)

    def test_extrapolation_or_unknown_domain_rejects_finite_points(self):
        for warning in ("feature_extrapolation:source_bytes", "unseen_domain"):
            selected = choose_forecast(replace(VALID, warnings=(warning,)), 20, 60)
            self.assertEqual(selected.source, "prior")
            self.assertEqual(selected.reason, "forecast_warning")

    def test_zero_negative_nonfinite_missing_and_bool_heads_fall_back(self):
        for head in ("first_call_mean", "total_policy_mean"):
            for value in (None, 0, -1, math.inf, -math.inf, math.nan, True, "invalid"):
                with self.subTest(head=head, value=value):
                    selected = choose_forecast(replace(VALID, **{head: value}), 20, 60)
                    self.assertEqual(selected.source, "prior")
                    self.assertEqual(selected.reason, "invalid_forecast_points")
                    self.assertEqual(selected.total_policy, 60)

    def test_missing_one_head_does_not_mix_learned_and_prior(self):
        selected = choose_forecast(replace(VALID, first_call_mean=None), 20, 60)
        self.assertEqual((selected.first_call, selected.total_policy), (20, 60))

    def test_total_below_first_is_rejected(self):
        selected = choose_forecast(replace(VALID, first_call_mean=101), 20, 60)
        self.assertEqual(selected.reason, "inconsistent_forecast_points")

    def test_full_pipeline_threshold_is_not_a_value_clip(self):
        at = choose_forecast(replace(VALID, total_policy_mean=100000), 20, 60)
        above = choose_forecast(replace(VALID, total_policy_mean=100000.1), 20, 60)
        huge = choose_forecast(replace(VALID, total_policy_mean=1e65), 20, 60)
        self.assertEqual(at.source, "learned")
        for selected in (above, huge):
            self.assertEqual(selected.source, "prior")
            self.assertEqual(selected.total_policy, 60)
            self.assertEqual(selected.reason, "forecast_exceeds_pipeline_budget")

    def test_large_prior_is_not_clipped_to_a_task_cap(self):
        selected = choose_forecast(None, 110000, 220000)
        self.assertEqual((selected.first_call, selected.total_policy), (110000, 220000))

    def test_invalid_priors_raise_even_when_forecast_is_usable(self):
        for first, total in ((0, 50), (-1, 50), (math.nan, 50), (math.inf, 50),
                             (10, math.inf), (10, None), (10, 9), (True, 50)):
            with self.subTest(first=first, total=total):
                with self.assertRaises(ValueError):
                    choose_forecast(VALID, first, total)

    def test_invalid_pipeline_budget_raises(self):
        for value in (0, -1, math.inf, math.nan, None, True):
            with self.assertRaises(ValueError):
                choose_forecast(VALID, 20, 60, pipeline_budget=value)

    def test_raw_dataclass_and_json_mapping_are_not_mutated(self):
        mapping = asdict(VALID)
        before = copy.deepcopy(mapping)
        self.assertEqual(choose_forecast(mapping, 20, 60), choose_forecast(VALID, 20, 60))
        self.assertEqual(mapping, before)
        self.assertEqual(VALID.total_policy_mean, 100)

    def test_budget_parameter_is_the_whole_pipeline_threshold(self):
        selected = choose_forecast(VALID, 20, 60, pipeline_budget=90)
        self.assertEqual(selected.reason, "forecast_exceeds_pipeline_budget")
        self.assertEqual(selected.total_policy, 60)


if __name__ == "__main__":
    unittest.main()

