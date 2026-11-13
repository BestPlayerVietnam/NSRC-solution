"""Synthetic fixtures verify statistics and provenance, not model performance."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.forecast_uncertainty import (
    analyze_objects, build_report, paired_group_bootstrap, percentile, sha,
    upper_diagnostics,
)


def pair(group, *, actual=100, deployed=12, heuristic=10, prior=8):
    return {"group": group, "actual": actual, "deployed": deployed,
            "heuristic": heuristic, "prompt_prior": prior}


def fixture():
    arm = "adaptive"
    records = [{"id": "a", "cost": 20, "trace": {"execution_complete": True}},
               {"id": "b", "cost": 30, "trace": {"execution_complete": True}}]
    evaluation = {"comparison_complete": True,
        "protocol": {"arms": {arm: {"initial_model": "model", "retry_policy": "adaptive"},
                              "classic": {"initial_model": "model", "retry_policy": "classic"}}},
        "arms": {arm: {"rows": records}, "classic": {"rows": copy.deepcopy(records)}}}
    forecasts = [{"id": ident, "model": "model", "heuristic": 15, "fallback": [10, 25],
                   "forecast": {"total_policy_mean": 20, "total_policy_upper": 40, "warnings": []},
                   "deployed": {"total_policy": 20, "source": "learned", "reason": None}}
                  for ident in ("a", "b")]
    metadata = {"a": {"group": "g1"}, "b": {"group": "g2"}}
    events = []
    for ident, cost in (("a", 20), ("b", 30)):
        events.extend([
            {"id": ident, "event": "reserved", "arm": arm, "task": ident, "reserved": 100},
            {"id": ident, "event": "completed", "arm": arm, "task": ident,
             "accounting": "provider_reconciled", "cost": cost}])
    return evaluation, forecasts, metadata, events


class ForecastUncertaintyTests(unittest.TestCase):
    def test_paired_identical_gain_has_degenerate_correct_interval(self):
        rows = [pair(f"g{i}", heuristic=10+i, deployed=12+i) for i in range(8)]
        result = paired_group_bootstrap(rows, "heuristic")
        self.assertEqual(result["primary_group_macro"]["delta_mae"], -2)
        self.assertEqual(result["primary_group_macro"]["ci95"], [-2, -2])
        self.assertEqual(result["secondary_row_mae"]["ci95"], [-2, -2])
        self.assertEqual(result["resamples"], 2000)

    def test_groups_unweighted_primary_rows_weighted_secondary(self):
        rows = [pair("many", actual=10, deployed=11, heuristic=10) for _ in range(10)]
        rows += [pair("one", actual=10, deployed=10, heuristic=0)]
        result = paired_group_bootstrap(rows, "heuristic")
        self.assertEqual(result["primary_group_macro"]["delta_mae"], -4.5)
        self.assertAlmostEqual(result["secondary_row_mae"]["delta_mae"], 0)
        self.assertEqual(result["primary_group_macro"]["ci95"], [-10, 1])

    def test_fixed_seed_is_reproducible(self):
        rows = [pair(f"g{i}", deployed=3*i) for i in range(7)]
        self.assertEqual(paired_group_bootstrap(rows, "prompt_prior"),
                         paired_group_bootstrap(rows, "prompt_prior"))

    def test_zero_or_one_group_does_not_claim_precision(self):
        for rows in ([], [pair("one"), pair("one")]):
            result = paired_group_bootstrap(rows, "heuristic")
            self.assertIsNone(result["primary_group_macro"]["ci95"])
            self.assertEqual(result["interval_status"], "insufficient_source_groups")
            self.assertEqual(result["resamples"], 0)

    def test_percentile_boundaries_and_interpolation(self):
        self.assertEqual(percentile([10, 0], 0), 0)
        self.assertEqual(percentile([10, 0], 1), 10)
        self.assertAlmostEqual(percentile([10, 0], .025), .25)
        self.assertAlmostEqual(percentile([10, 0], .975), 9.75)
        self.assertIsNone(percentile([], .5))
        with self.assertRaises(ValueError):
            percentile([1], -1)

    def test_nonfinite_paired_forecast_is_not_silently_excluded(self):
        with self.assertRaises(ValueError):
            paired_group_bootstrap([pair("g", deployed=math.inf)], "heuristic")
        with self.assertRaises(ValueError):
            paired_group_bootstrap([pair("")], "heuristic")

    def test_population_uses_actual_reconciled_cost_not_reservation(self):
        report = analyze_objects(*fixture())
        model = report["models"]["model"]
        self.assertEqual(model["population_audit"]["complete_known_total"], 2)
        self.assertEqual(model["comparisons"]["heuristic"]["primary_group_macro"]["deployed_mae"], 5)
        self.assertEqual(len(report["models"]), 1)  # classic was excluded

    def test_unknown_usage_partial_execution_and_unattempted_excluded(self):
        e, f, meta, calls = fixture()
        # One known first call does not make a later reserved/error total known.
        calls.append({"id": "extra", "event": "reserved", "arm": "adaptive", "task": "a", "reserved": 50})
        e["arms"]["adaptive"]["rows"][0]["cost"] += 50
        e["arms"]["adaptive"]["rows"][1]["trace"]["execution_complete"] = False
        e["arms"]["adaptive"]["rows"].append({"id": "never", "cost": 0, "trace": None})
        result = analyze_objects(e, f, meta, calls)["models"]["model"]
        self.assertEqual(result["population_audit"]["attempted"], 2)
        self.assertEqual(result["population_audit"]["excluded_total"], 2)
        self.assertEqual(result["population_audit"]["not_attempted"], 1)
        self.assertEqual(result["comparisons"]["heuristic"]["n"], 0)

    def test_fallback_fraction_uses_same_eligible_population(self):
        e, f, meta, calls = fixture()
        f[0]["deployed"] = {"total_policy": 25, "source": "prior", "reason": "forecast_warning"}
        e["arms"]["adaptive"]["rows"][1]["trace"]["execution_complete"] = False
        result = analyze_objects(e, f, meta, calls)["models"]["model"]["deployment_fallback"]
        self.assertEqual((result["n"], result["count"], result["fraction"]), (1, 1, 1))

    def test_upper_magnitude_and_cluster_coverage_are_distinct(self):
        rows = [{"group": "g1", "actual": 9, "raw_forecast": {"total_policy_mean": 3, "total_policy_upper": 10}},
                {"group": "g1", "actual": 11, "raw_forecast": {"total_policy_mean": 3, "total_policy_upper": 10}},
                {"group": "g2", "actual": 5, "raw_forecast": {"total_policy_mean": 3, "total_policy_upper": 1000}}]
        result = upper_diagnostics(rows)
        self.assertEqual(result["upper_tokens"]["max"], 1000)
        self.assertEqual(result["upper_minus_raw_point_tokens"]["max"], 997)
        self.assertAlmostEqual(result["row_empirical_coverage"], 2/3)
        self.assertEqual(result["group_macro_empirical_coverage"], .75)
        self.assertEqual(result["all_rows_covered_group_fraction"], .5)
        self.assertFalse(result["coverage_guarantee"])

    def test_invalid_or_missing_uppers_do_not_count_as_covered(self):
        rows = [{"group": f"g{i}", "actual": 9, "raw_forecast": raw} for i, raw in enumerate([
            {"total_policy_mean": 3, "total_policy_upper": None},
            {"total_policy_mean": 3, "total_policy_upper": math.inf},
            {"total_policy_mean": 3, "total_policy_upper": 2}])]
        result = upper_diagnostics(rows)
        self.assertEqual(result["available_finite_consistent_bounds"], 0)
        self.assertEqual((result["missing_upper"], result["nonfinite_or_missing_point"],
                          result["upper_below_point"]), (1, 1, 1))
        self.assertIsNone(result["row_empirical_coverage"])

    def test_incomplete_evaluation_and_cost_disagreement_fail(self):
        e, f, meta, calls = fixture()
        e["comparison_complete"] = False
        with self.assertRaises(RuntimeError):
            analyze_objects(e, f, meta, calls)
        e["comparison_complete"] = True
        e["arms"]["adaptive"]["rows"][0]["cost"] = 19
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyze_objects(e, f, meta, calls)

    def test_frozen_forecast_hash_is_verified(self):
        e, f, meta, calls = fixture()
        with tempfile.TemporaryDirectory() as directory:
            run, suite = Path(directory)/"run", Path(directory)/"suite"
            run.mkdir(); suite.mkdir()
            (run/"forecasts.json").write_text(json.dumps(f), encoding="utf-8")
            (run/"calls.jsonl").write_text("\n".join(json.dumps(r) for r in calls), encoding="utf-8")
            (suite/"research_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            (suite/"manifest.json").write_text("{}", encoding="utf-8")
            for key, path in (("forecasts_sha256", run/"forecasts.json"),
                              ("metadata_sha256", suite/"research_metadata.json"),
                              ("manifest_sha256", suite/"manifest.json")):
                e["protocol"][key] = sha(path)
            (run/"evaluation.json").write_text(json.dumps(e), encoding="utf-8")
            self.assertTrue(build_report(run, suite)["frozen_forecast_hash_verified"])
            (run/"forecasts.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                build_report(run, suite)
            self.assertFalse((run/"uncertainty_report.json").exists())


if __name__ == "__main__":
    unittest.main()

