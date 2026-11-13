"""Contract tests only; synthetic fixtures are never experimental evidence."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import offline_policy_search as search

POLICY = "offline-contract-test"


def row(i, *, model="plain", prefix="train", cost=None, success=True):
    cost = 20.0 + i if cost is None else float(cost)
    return {"id": f"{prefix}-{i}", "task_group": f"{prefix}-group-{i}",
            "model": model, "policy_id": POLICY, "domain": "math",
            "source_type": "real_api", "reward": 1 + i % 3,
            "features": {"size": float(i), "kind_math": 1.0},
            "first_call_cost": cost / 2, "total_policy_cost": cost,
            "first_usage_known": True, "total_usage_known": True,
            "execution_complete": True, "final_success": success, "verifier_kind": "oracle"}


def fit(rows=None, calibration=(), **kwargs):
    return search.fit_offline(rows or [row(i) for i in range(12)], calibration,
                              policy_id=POLICY, **kwargs)


class OfflinePolicySearchTests(unittest.TestCase):
    def test_explicit_real_observations_and_policy_are_required(self):
        r = row(0)
        r["source_type"] = "mock"
        with self.assertRaisesRegex(ValueError, "real_api"):
            search.validate_rows([r], POLICY, ["size"])
        r["source_type"] = "real_api"; r["policy_id"] = "different"
        with self.assertRaisesRegex(ValueError, "policies"):
            search.validate_rows([r], POLICY, ["size"])

    def test_labels_and_reward_cannot_be_pre_execution_features(self):
        for name in ("answerable", "correct", "reward", "actual_tokens", "final_success"):
            r = row(0); r["features"][name] = 1
            with self.assertRaisesRegex(ValueError, "Feature schema"):
                search.validate_rows([r], POLICY, ["size", name])

    def test_public_success_is_not_hidden_correctness(self):
        rows = [dict(row(i), verifier_kind="public_tests") for i in range(12)]
        result = search.fit_success(rows)
        self.assertEqual(result["rows"], 0)
        self.assertEqual(result["global"], 0.5)

    def test_success_shrinkage_balances_source_groups(self):
        rows = [row(i, success=True) for i in range(3)] + [row(3, success=False)]
        for r in rows[:3]:
            r["task_group"] = "one-repeated-source"
        result = search.fit_success(rows)
        self.assertAlmostEqual(result["global"], 0.5)
        self.assertEqual(result["groups"], 2)

    def test_scheduler_does_not_read_real_cost_or_success(self):
        choices = [{"id": str(i), "reward": 1, "predicted_cost": 40,
                    "success_probability": 0.7, "actual_cost": 1, "correct": True}
                   for i in range(4)]
        before = search.select_portfolio(choices, search.BASELINE, 100)
        for r in choices:
            r["actual_cost"] = 100000; r["correct"] = False
        self.assertEqual(before, search.select_portfolio(choices, search.BASELINE, 100))
        self.assertEqual(len(before["selected_ids"]), 2)

    def test_over_budget_bundle_never_gets_fabricated_prefix_reward(self):
        rows = [row(0, cost=60), row(1, cost=60)]
        selection = {"selected_ids": [r["id"] for r in rows], "budget": 100,
                     "predicted_reserved_cost": 80}
        measured = search.evaluate_bundle(selection, rows)
        self.assertFalse(measured["within_budget"])
        self.assertEqual(measured["measured_full_trace_cost"], 120)
        self.assertEqual(measured["overrun"], 20)
        self.assertIsNone(measured["budget_valid_reward"])

    def test_unknown_outcome_is_not_an_observed_failure(self):
        r = dict(row(0), final_success=None, verifier_kind="unknown")
        measured = search.evaluate_bundle(
            {"selected_ids": [r["id"]], "budget": 100, "predicted_reserved_cost": 20}, [r])
        self.assertFalse(measured["complete_observation"])
        self.assertEqual(measured["unknown_ids"], [r["id"]])
        self.assertIsNone(measured["budget_valid_reward"])

    def test_task_order_cannot_break_identical_bundle_tie(self):
        rows = [row(i, cost=c) for i, c in enumerate((1e4, 0.1, 0.2, 0.3, 0.4))]
        ids = [r["id"] for r in rows]
        a = search.evaluate_bundle({"selected_ids": ids, "budget": 100000}, rows)
        b = search.evaluate_bundle({"selected_ids": ids[::-1], "budget": 100000}, rows)
        self.assertEqual(search.bundle_objective(a), search.bundle_objective(b))

    def test_lexicographic_points_dominate_cost_savings(self):
        high = {"within_budget": True, "budget_valid_reward": 2, "measured_full_trace_cost": 99000}
        low = {"within_budget": True, "budget_valid_reward": 1, "measured_full_trace_cost": 1}
        self.assertGreater(search.bundle_objective(high), search.bundle_objective(low))
        high2 = dict(high, measured_full_trace_cost=98000)
        self.assertGreater(search.bundle_objective(high2), search.bundle_objective(high))

    def test_one_replicate_per_model_task_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            search.validate_rows([row(0), row(0)], POLICY, ["size"])
        search.validate_rows([row(0), row(0, model="other")], POLICY, ["size"])

    def test_oof_holds_out_entire_source_group(self):
        rows = [row(i) for i in range(12)]
        original = search.CostModel
        checked = []

        class GuardedModel(original):
            def predict(self, features, **kwargs):
                g = f"train-group-{int(features['size'])}"
                self_outer.assertNotIn(g, self.artifact["train_groups"])
                checked.append(g)
                return super().predict(features, **kwargs)

        self_outer = self
        with patch.object(search, "CostModel", GuardedModel):
            search.crossfit(rows, ["size", "kind_math"], POLICY)
        self.assertEqual(len(checked), len(rows))

    def test_train_calibration_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit(calibration=[row(0)])

    def test_primary_budgets_are_isolated_per_model(self):
        rows = [row(i, model=m, cost=10000) for m in ("a", "b") for i in range(8)]
        artifact, report = fit(rows)
        self.assertEqual(artifact["primary_budget_per_model"], 100000)
        costs = []
        for m in ("a", "b"):
            winner = report["models"][m]["selected_candidate"]
            bundle = report["models"][m]["candidates"][winner]["bundle"]
            self.assertTrue(bundle["within_budget"])
            costs.append(bundle["measured_full_trace_cost"])
        self.assertEqual(sum(costs), 160000)

    def test_cheap_all_selected_primary_is_reported_as_uninformative(self):
        artifact, report = fit()
        self.assertFalse(report["models"]["plain"]["primary_search_informative"])
        self.assertEqual(report["models"]["plain"]["selected_candidate"], 0)
        self.assertEqual(artifact["models"]["plain"]["scheduler"], search.BASELINE)
        self.assertEqual(artifact["cost_artifact"]["criterion"], "mae")

    def test_stress_budgets_do_not_select_primary_winner(self):
        a, _ = fit(stress_budgets=(100,))
        b, _ = fit(stress_budgets=(50, 10000))
        self.assertEqual(a["models"], b["models"])

    def test_heldout_evaluation_does_not_fit(self):
        artifact, _ = fit()
        with patch.object(search, "fit_cost_artifact", side_effect=AssertionError("refit")):
            report = search.evaluate_offline(artifact, [row(i, prefix="test") for i in range(8)])
        self.assertEqual(report["models"]["plain"]["n"], 8)

    def test_heldout_source_overlap_rejected(self):
        artifact, _ = fit()
        with self.assertRaisesRegex(ValueError, "overlap"):
            search.evaluate_offline(artifact, [row(2)])

    def test_partial_trajectory_not_search_label(self):
        rows = [row(i) for i in range(12)]
        rows[0]["execution_complete"] = False
        rows[1]["verifier_kind"] = "public_tests"
        artifact, report = fit(rows)
        self.assertEqual(report["models"]["plain"]["excluded_unverified_or_censored"], 2)
        self.assertEqual(artifact["models"]["plain"]["search_rows"], 10)

    def test_first_cost_unknown_does_not_discard_known_total(self):
        rows = [dict(row(i), first_usage_known=False, first_call_cost=None) for i in range(12)]
        _, report = fit(rows)
        self.assertEqual(report["models"]["plain"]["search_rows"], 12)
        self.assertEqual(report["models"]["plain"]["oof_cost"]["first_call"]["n"], 0)

    def test_calibration_labels_do_not_select_scheduler(self):
        cal = [row(i, prefix="cal", cost=3000, success=False) for i in range(8)]
        a, _ = fit()
        b, _ = fit(calibration=cal)
        self.assertEqual(a["models"], b["models"])
        for target in ("first_call", "total_policy"):
            self.assertEqual(a["cost_artifact"]["models"]["plain"]["heads"][target]["fitted"],
                             b["cost_artifact"]["models"]["plain"]["heads"][target]["fitted"])

    def test_invalid_primary_or_stress_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, "Primary"):
            fit(primary_budget=5000)
        with self.assertRaisesRegex(ValueError, "Stress"):
            fit(stress_budgets=(100000,))

    def legacy(self, *, split="train", policy=POLICY, replicate=0, answerable=True, correct=True):
        r = row(0)
        return {"id": r["id"], "group": r["task_group"], "model": r["model"],
                "policy": policy, "split": split, "replicate": replicate, "domain": "math",
                "features": {"instruction_chars": 20, "domain": "math"},
                "reward": 3, "actual_tokens": 20, "state": "answered",
                "calls": [{"cost": 20, "accounting": "provider_reconciled"}],
                "answerable": answerable, "correct": correct}

    def test_legacy_correct_abstention_earns_no_points(self):
        old = self.legacy(answerable=False, correct=True); old["state"] = "abstained"
        converted = search.export_legacy_train([old], POLICY)
        self.assertIs(converted[0]["final_success"], False)
        self.assertNotIn("answerable", converted[0]["features"])

    def test_legacy_only_train_primary_replicate_exact_policy(self):
        records = [self.legacy(), self.legacy(split="test"),
                   self.legacy(split="calibration"), self.legacy(replicate=1),
                   self.legacy(policy="other")]
        self.assertEqual(len(search.export_legacy_train(records, POLICY)), 1)

    def test_legacy_accounting_disagreement_is_rejected(self):
        old = self.legacy(); old["actual_tokens"] = 40
        with self.assertRaisesRegex(ValueError, "totals"):
            search.export_legacy_train([old], POLICY)


if __name__ == "__main__":
    unittest.main()

