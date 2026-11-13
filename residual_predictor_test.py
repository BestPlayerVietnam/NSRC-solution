"""Inference contract and exact frozen-visible-forecast parity; no outcome reads."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from residual_predictor import ResidualModel, load_residual_model

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "experiments/runs/2026-11-13-residual-v1"
POLICY = "unit-policy"
MODEL = "unit-noreason"
FEATURES = {"instruction_chars": 350, "context_chars": 0, "kind_math": 1}


def artifact(method="additive", first=10, total=20):
    return {"schema_version": 1, "policy_id": POLICY, "models": {MODEL: {"heads": {
        target: {"method": method, "global": value, "domains": {"math": {"value": value}}}
        for target, value in (("first_call", first), ("total_policy", total))}}}}


def predict(estimator, **kwargs):
    arguments = {"model": MODEL, "policy_id": POLICY, "domain": "math", "prior": (100, 120)}
    arguments.update(kwargs)
    return estimator.predict(FEATURES, **arguments)


class ResidualPredictorTests(unittest.TestCase):
    def test_all_numeric_methods_and_unknown_domain(self):
        for method, expected in (("prior", (100, 120)), ("multiplier", (1000, 2400)),
                                 ("additive", (110, 140)), ("input_remaining", (38, 48))):
            with self.subTest(method=method):
                estimator = ResidualModel(artifact(method))
                result = predict(estimator)
                self.assertEqual((result.first_call, result.total_policy), expected)
                self.assertEqual(predict(estimator, domain="new_domain"), result)

    def test_model_and_policy_fallback_preserve_both_prior_heads(self):
        estimator = ResidualModel(artifact())
        for kwargs, reason in (({"model":"unknown"}, "model_not_trained"),
                               ({"policy_id":"other"}, "policy_mismatch")):
            result = predict(estimator, **kwargs)
            self.assertEqual(asdict(result), {"first_call":100, "total_policy":120,
                                             "source":"prior", "reason":reason})

    def test_nonfinite_and_boolean_coefficients_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ResidualModel(artifact(first=value))

    def test_negative_offsets_allowed_but_negative_scales_rejected(self):
        self.assertEqual(predict(ResidualModel(artifact(first=-20))).first_call, 80)
        for method in ("multiplier", "input_remaining"):
            with self.assertRaises(ValueError):
                ResidualModel(artifact(method, first=-1))

    def test_positive_floor_and_order_match_experimental_projection(self):
        self.assertEqual(predict(ResidualModel(artifact(first=-500, total=-500))).first_call, 1)
        result = predict(ResidualModel(artifact(first=100, total=-500)))
        self.assertEqual((result.first_call, result.total_policy), (200, 200))

    def test_finite_coefficients_overflow_falls_back_whole_pair(self):
        estimator = ResidualModel(artifact("multiplier", first=1e308))
        result = predict(estimator)
        self.assertEqual((result.first_call, result.total_policy), (100, 120))
        self.assertEqual(result.reason, "invalid_numeric_prediction")

    def test_no_unfrozen_budget_or_ood_guard(self):
        estimator = ResidualModel(artifact("multiplier", first=2000, total=2000))
        result = predict(estimator)
        self.assertEqual((result.first_call, result.total_policy), (200000, 240000))
        self.assertEqual(result.source, "residual")

    def test_invalid_prior_is_rejected_before_scope_fallback(self):
        for prior in ((0,1), (2,1), (1,math.inf), (True,2), (1,)):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                predict(ResidualModel(artifact()), model="unknown", prior=prior)

    def test_optional_prior_and_domain_match_frozen_formula(self):
        result = ResidualModel(artifact("prior")).predict(FEATURES, model=MODEL, policy_id=POLICY)
        self.assertEqual(result.first_call, 52)
        self.assertEqual(result.total_policy, 52 + .06 * (28 + 650))

    def test_caller_mutation_does_not_change_loaded_coefficients(self):
        data = artifact()
        estimator = ResidualModel(data)
        expected = predict(estimator)
        data["models"][MODEL]["heads"]["first_call"]["domains"]["math"]["value"] = 999
        self.assertEqual(predict(estimator), expected)

    def test_load_checks_exact_policy_and_utf8_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "artifact.json"
            path.write_text(json.dumps(artifact()), encoding="utf-8-sig")
            self.assertEqual(predict(load_residual_model(path, POLICY)).total_policy, 140)
            with self.assertRaisesRegex(ValueError, "policy_id"):
                load_residual_model(path, "wrong")

    @unittest.skipUnless(FROZEN.exists(), "Frozen development parity fixtures not packaged")
    def test_all_592_frozen_predictions_match_exactly_without_outcome_files(self):
        def read(path):
            return json.loads(path.read_text(encoding="utf-8-sig"))
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        freeze = read(FROZEN / "freeze_record.json")
        artifact_path = FROZEN / "residual_artifact.json"
        self.assertEqual(sha(artifact_path), freeze["output_sha256"]["artifact"])
        estimator = load_residual_model(artifact_path, freeze["policy_id"])
        count = 0
        for name, relative in (("synthetic", "experiments/runs/2026-11-12-policy/test/forecasts.json"),
                               ("public100", "experiments/runs/2026-11-12-public100/test/forecasts.json")):
            visible_path = ROOT / relative
            predictions_path = FROZEN / ("predictions_" + name + ".json")
            self.assertEqual(sha(visible_path), freeze["input_sha256"][name])
            self.assertEqual(sha(predictions_path), freeze["output_sha256"]["predictions_" + name])
            expected = {(r["id"],r["model"]):r for r in read(predictions_path)["predictions"]}
            seen = set()
            for record in read(visible_path):
                key = (record["id"], record["model"])
                self.assertNotIn(key, seen)
                seen.add(key)
                result = estimator.predict(record["features"], model=record["model"],
                    policy_id=freeze["policy_id"], prior=record["fallback"])
                self.assertEqual(asdict(result), expected[key]["prediction"], key)
                count += 1
            self.assertEqual(seen, set(expected))
        self.assertEqual(count, 592)


if __name__ == "__main__":
    unittest.main()
