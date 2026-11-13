"""Synthetic math/leakage contract tests; no benchmark-performance claims."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import residual_forecaster as residual

POLICY="residual-contract-test"
MODEL="unit-noreason"


def row(i, model=MODEL, group=None):
    features={"instruction_chars": 100+17*i, "context_chars":0,
              "kind_math":1,"kind_coding":0,"kind_search":0,"kind_data_analysis":0,
              "source_bytes":0,"contradiction_flag":0}
    prior=residual.prompt_prior(features,model)
    return {"id":f"train-{i}","task_group":group or f"group-{i}","source_type":"real_api",
            "policy_id":POLICY,"model":model,"domain":"math","features":features,
            "first_call_cost":prior[0],"total_policy_cost":prior[1],
            "first_usage_known":True,"total_usage_known":True,"execution_complete":True}


class ResidualForecasterTests(unittest.TestCase):
    def test_ratio_median_uses_prior_times_group_weight(self):
        rows=residual.prepare_rows([row(0),row(1)],POLICY)
        rows[0].update(_prior=(10,20),first_call_cost=10)
        rows[1].update(_prior=(100,200),first_call_cost=300)
        head=residual.fit_head(rows,"first_call",("multiplier",0))
        self.assertEqual(head["global"],3)  # unweighted median would be 2

    def test_duplicate_family_variants_do_not_dominate(self):
        rows=residual.prepare_rows([row(i,group="many") for i in range(10)]+[row(10,group="one")],POLICY)
        for r in rows:
            r["_prior"]=(100,200)
            r["first_call_cost"]=100 if r["task_group"]=="many" else 900
        head=residual.fit_head(rows,"first_call",("multiplier",0))
        self.assertAlmostEqual(head["global"],5)

    def test_domain_median_shrinks_by_group_count(self):
        rows=residual.prepare_rows([row(0),row(1)],POLICY)
        rows[0].update(_prior=(100,200),first_call_cost=100,domain="math")
        rows[1].update(_prior=(100,200),first_call_cost=900,domain="search")
        head=residual.fit_head(rows,"first_call",("multiplier",2))
        self.assertAlmostEqual(head["domains"]["math"]["value"],11/3)
        self.assertAlmostEqual(head["domains"]["search"]["value"],19/3)

    def test_weighted_median_boundaries(self):
        self.assertEqual(residual.weighted_median([1,9],[1,1]),5)
        self.assertEqual(residual.weighted_median([1,9],[0,1]),9)
        for values,weights in (([1],[0]),([1],[math.nan]),([1],[1,2]),([1],[-1])):
            with self.assertRaises(ValueError):
                residual.weighted_median(values,weights)

    def test_cv_scores_final_ordered_points(self):
        r=residual.prepare_rows([row(0)],POLICY)[0]
        r["first_call_cost"]=100;r["total_policy_cost"]=140
        score=residual.pair_scores([r],[150],[20])
        self.assertEqual(score["total_macro_mae"],10)
        self.assertEqual(score["first_macro_mae"],50)
        self.assertEqual(residual.project(-3,-4),(1,1))
        self.assertEqual(residual.project(100,20),(100,100))

    def test_exact_prompt_prior_remains_an_eligible_winner(self):
        artifact,report=residual.fit([row(i) for i in range(12)],policy_id=POLICY)
        self.assertEqual(artifact["models"][MODEL]["selected_candidates"],{"first_call":0,"total_policy":0})
        self.assertEqual(report["models"][MODEL]["chosen"]["total_macro_mae"],0)

    def test_models_are_separate_and_unknown_scope_uses_prior(self):
        rows=[row(i,model=m) for m in (MODEL,"other-noreason") for i in range(12)]
        for r in rows:
            if r["model"]=="other-noreason":
                r["first_call_cost"]*=2;r["total_policy_cost"]*=2
        artifact,_=residual.fit(rows,policy_id=POLICY)
        estimator=residual.ResidualForecaster(artifact)
        features=row(3)["features"]
        a=estimator.predict(features,model=MODEL)
        b=estimator.predict(features,model="other-noreason")
        self.assertAlmostEqual(b.total_policy,2*a.total_policy)
        self.assertEqual(estimator.predict(features,model="missing").reason,"model_not_trained")
        self.assertEqual(estimator.predict(features,model=MODEL,policy_id="other").reason,"policy_mismatch")

    def test_unknown_targets_excluded_independently(self):
        rows=[row(i) for i in range(12)]
        rows[0].update(first_usage_known=False,first_call_cost=None)
        rows[1]["execution_complete"]=False
        _,report=residual.fit(rows,policy_id=POLICY)
        self.assertEqual(report["models"][MODEL]["known_first"],11)
        self.assertEqual(report["models"][MODEL]["known_complete_total"],11)

    def test_oof_excludes_all_variants_of_held_source_group(self):
        rows=[row(i,group=f"family-{i//2}") for i in range(12)]
        lookup={p:r["task_group"] for r in rows for p in residual.prompt_prior(r["features"],MODEL)}
        original_fit,original_point=residual.fit_head,residual.raw_point
        checks=[]
        def traced_fit(training,target,candidate):
            head=original_fit(training,target,candidate)
            head["_train_groups"]=sorted({r["task_group"] for r in training})
            return head
        def guarded_point(head,prior,input_cost,domain):
            self.assertNotIn(lookup[prior],head["_train_groups"])
            checks.append(prior)
            return original_point(head,prior,input_cost,domain)
        with patch.object(residual,"fit_head",traced_fit),patch.object(residual,"raw_point",guarded_point):
            residual.fit(rows,policy_id=POLICY)
        self.assertGreater(len(checks),0)

    def test_one_group_retains_prior_without_fabricated_cv_score(self):
        artifact,report=residual.fit([row(0)],policy_id=POLICY)
        self.assertEqual(artifact["models"][MODEL]["selected_candidates"]["first_call"],0)
        self.assertIsNone(report["models"][MODEL]["chosen"]["total_macro_mae"])

    def test_guarded_comparator_refits_inside_outer_groups_and_applies_guard(self):
        import cost_model
        rows=[row(i,group=f"family-{i//2}") for i in range(12)]
        features_to_group={r["features"]["instruction_chars"]:r["task_group"] for r in rows}
        fitted_groups=[];checked=[]
        def fit_inner(training,**kwargs):
            groups={r["task_group"] for r in training}
            fitted_groups.append(groups)
            self.assertEqual(kwargs["criterion"],"mae")
            return groups
        class FakeEstimator:
            def __init__(self,groups):
                self.groups=groups
            def predict(self,features,**kwargs):
                held=features_to_group[features["instruction_chars"]]
                if held in self.groups:
                    raise AssertionError("Outer held group leaked into guarded fit")
                checked.append(held)
                return {"first_call_mean":1e20,"total_policy_mean":1e20,
                        "warnings":["feature_extrapolation:test"]}
        with patch.object(cost_model,"fit_cost_artifact",fit_inner),patch.object(cost_model,"CostModel",FakeEstimator):
            report=residual.guarded_oof_diagnostic(rows,policy_id=POLICY)
        self.assertEqual(len(fitted_groups),4)
        self.assertEqual(len(checked),len(rows))
        self.assertEqual(report["models"][MODEL]["fallback_count"],len(rows))
        self.assertEqual(report["models"][MODEL]["total_macro_mae"],0)

    def test_visible_freeze_ignores_label_like_extra_fields(self):
        artifact,_=residual.fit([row(i) for i in range(8)],policy_id=POLICY)
        estimator=residual.ResidualForecaster(artifact)
        record={"id":"visible-new","model":MODEL,"features":row(30)["features"]}
        record["fallback"]=list(residual.prompt_prior(record["features"],MODEL))
        a=residual.freeze_visible(estimator,[record],artifact_sha="unit")
        record.update(correct=True,answerable=False,actual_tokens=1e9,oracle="not read")
        b=residual.freeze_visible(estimator,[record],artifact_sha="unit")
        self.assertEqual(a,b)
        record["id"]="train-0"
        with self.assertRaisesRegex(ValueError,"overlap"):
            residual.freeze_visible(estimator,[record],artifact_sha="unit")

    def test_frozen_prior_mismatch_is_rejected(self):
        artifact,_=residual.fit([row(i) for i in range(8)],policy_id=POLICY)
        record={"id":"visible-new","model":MODEL,"features":row(30)["features"],"fallback":[1,2]}
        with self.assertRaisesRegex(ValueError,"prior differs"):
            residual.freeze_visible(residual.ResidualForecaster(artifact),[record],artifact_sha="unit")

    def test_fit_freeze_opens_only_declared_train_and_visible_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);train=base/"train.json";visible=base/"forecasts.json"
            train.write_text(json.dumps([row(i) for i in range(8)]),encoding="utf-8")
            features=row(30)["features"]
            visible.write_text(json.dumps([{"id":"new","model":MODEL,"features":features,
                "fallback":list(residual.prompt_prior(features,MODEL))}]),encoding="utf-8")
            opened=[];original=residual.read
            def traced_read(path):
                opened.append(Path(path));return original(path)
            with patch.object(residual,"read",traced_read):
                _,manifest=residual.fit_and_freeze(train,base/"out",{"public100":visible},POLICY)
            self.assertEqual(opened,[train,visible])
            self.assertFalse(manifest["public_evaluation_or_results_read"])
            self.assertTrue((base/"out/residual_forecaster_frozen.py").exists())
            with self.assertRaises(FileExistsError):
                residual.fit_and_freeze(train,base/"out",{"public100":visible},POLICY)


if __name__=="__main__":
    unittest.main()

