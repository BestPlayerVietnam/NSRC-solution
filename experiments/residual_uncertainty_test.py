"""Synthetic statistical, label-scope and provenance checks; no public outcomes."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import residual_uncertainty as uncertainty

POLICY = "unit-policy"


def fixture():
    artifact = {"policy_id":POLICY,"train_ids":["train"],"train_groups":["train-group"]}
    frozen = {"predictions":[]}
    rows = []
    metadata = {}
    events = []
    evaluation = {"comparison_complete":True, "protocol":{"arms": {
        "adaptive":{"initial_model":"model","retry_policy":"adaptive"}}},
        "arms":{"adaptive":{"rows":[]}}}
    for ident in ("a","b"):
        frozen["predictions"].append({"id":ident,"model":"model",
            "prediction":{"first_call":10,"total_policy":20},
            "original_deployed":{"first_call":8,"total_policy":18},"prompt_prior":[6,16]})
        rows.append({"id":ident,"model":"model","task_group":"group-"+ident,
            "source_type":"real_api","policy_id":POLICY,"first_call_cost":10,"total_policy_cost":20,
            "first_usage_known":True,"total_usage_known":True,"execution_complete":True})
        metadata[ident] = {"group":"group-"+ident}
        evaluation["arms"]["adaptive"]["rows"].append({"id":ident,
            "trace":{"policy_id":POLICY,"execution_complete":True}})
        for j in range(2):
            events.append({"id":ident+str(j),"event":"reserved","arm":"adaptive","task":ident,"reserved":50})
            events.append({"id":ident+str(j),"event":"completed","arm":"adaptive","task":ident,
                "cost":10,"accounting":"provider_reconciled"})
    return artifact, frozen, rows, evaluation, metadata, events


def pair(group, actual=100, residual=12, guarded=10, prior=8):
    return {"group":group,"actual":actual,"residual":residual,"guarded":guarded,"prompt_prior":prior}


class ResidualUncertaintyTests(unittest.TestCase):
    def test_negative_delta_and_relabeling_are_correct(self):
        result = uncertainty.compare([pair("a"),pair("b")],"guarded")
        self.assertEqual(result["primary_group_macro"]["delta_mae"],-2)
        self.assertEqual(result["primary_group_macro"]["ci95"],[-2,-2])
        self.assertEqual(result["primary_group_macro"]["residual_mae"],88)
        self.assertEqual(result["negative_favors"],"residual")
        self.assertNotIn("deployed_mae",result["primary_group_macro"])
        self.assertEqual(result["secondary_wape"]["delta"],-.02)
        self.assertEqual(result["secondary_wape"]["ci95"],[-.02,-.02])
        self.assertEqual(result["resamples"],2000)
        self.assertEqual(result["seed"],20261112)

    def test_macro_row_and_wape_have_distinct_weighting(self):
        rows = [pair("large",actual=10,residual=11,guarded=10) for _ in range(10)]
        rows += [pair("small",actual=10,residual=10,guarded=0)]
        result = uncertainty.compare(rows,"guarded")
        self.assertEqual(result["primary_group_macro"]["delta_mae"],-4.5)
        self.assertAlmostEqual(result["secondary_row_mae"]["delta_mae"],0)
        self.assertEqual(result["secondary_wape"]["delta"],0)
        self.assertEqual(result["primary_group_macro"]["ci95"],[-10,1])

    def test_zero_cost_wape_and_small_group_ci_are_not_fabricated(self):
        for rows in ([],[pair("only")]):
            result = uncertainty.compare(rows,"prompt_prior")
            self.assertIsNone(result["primary_group_macro"]["ci95"])
        result = uncertainty.compare([pair("a",actual=0),pair("b",actual=0)],"guarded")
        self.assertIsNone(result["secondary_wape"]["delta"])
        self.assertIsNone(result["secondary_wape"]["ci95"])
        result = uncertainty.compare([pair("a",actual=0),pair("b",actual=1)],"guarded")
        self.assertEqual(result["secondary_wape"]["interval_status"],"zero_denominator_in_some_resamples")
        self.assertIsNone(result["secondary_wape"]["ci95"])

    def test_seed_reproducible_and_nonfinite_predictions_fail(self):
        rows = [pair(str(i),residual=i*5) for i in range(5)]
        self.assertEqual(uncertainty.compare(rows,"guarded"),uncertainty.compare(rows,"guarded"))
        rows[0]["residual"] = math.inf
        with self.assertRaises(ValueError):
            uncertainty.compare(rows,"guarded")

    def test_head_labels_independent_but_predictors_use_same_rows(self):
        artifact,frozen,rows,*_ = fixture()
        rows[0]["execution_complete"] = False
        report = uncertainty.analyze_rows(artifact,frozen,rows)["models"]["model"]["heads"]
        self.assertEqual(report["first_call"]["n"],2)
        self.assertEqual(report["total_policy"]["n"],1)
        for target in uncertainty.TARGETS:
            self.assertEqual(report[target]["comparisons"]["guarded"]["n"],report[target]["comparisons"]["prompt_prior"]["n"])
        rows[1].update(first_usage_known=False,first_call_cost=None)
        self.assertEqual(uncertainty.analyze_rows(artifact,frozen,rows)["models"]["model"]["heads"]["first_call"]["n"],1)

    def test_missing_baseline_rejected_instead_of_dropping_rows(self):
        artifact,frozen,rows,*_ = fixture()
        frozen["predictions"][0].pop("original_deployed")
        with self.assertRaisesRegex(ValueError,"Guarded forecast"):
            uncertainty.analyze_rows(artifact,frozen,rows)

    def test_policy_duplicate_and_training_overlap_rejected(self):
        artifact,frozen,original,*_ = fixture()
        changes = ({"policy_id":"other"},{"source_type":"mock"},{"task_group":"train-group"},
                   {"first_call_cost":30})
        for change in changes:
            rows = copy.deepcopy(original);rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                uncertainty.analyze_rows(artifact,frozen,rows)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            uncertainty.analyze_rows(artifact,frozen,original+[original[0]])

    def test_normalized_labels_match_actual_calls_and_adaptive_scope(self):
        artifact,frozen,rows,evaluation,metadata,events = fixture()
        uncertainty.validate_labels(rows,evaluation,metadata,events,POLICY)
        rows[0]["total_policy_cost"] += 1
        with self.assertRaisesRegex(ValueError,"cost disagrees"):
            uncertainty.validate_labels(rows,evaluation,metadata,events,POLICY)
        rows[0]["total_policy_cost"] -= 1
        events[-1]["accounting"] = "reserved_error"
        with self.assertRaisesRegex(ValueError,"label mismatch"):
            uncertainty.validate_labels(rows,evaluation,metadata,events,POLICY)

    def test_incomplete_run_is_checked_before_any_outcome_rows_are_opened(self):
        opened = []
        def read(path):
            opened.append(Path(path).name)
            return {"comparison_complete":False}
        with patch.object(uncertainty,"read",read), self.assertRaisesRegex(RuntimeError,"incomplete"):
            uncertainty.build_report(run="run",suite="suite",rows_path="forbidden-test_rows.json",
                                     frozen_dir="frozen",dataset="public100")
        self.assertEqual(opened,["evaluation.json"])

    def test_end_to_end_hash_audit_and_source_paths(self):
        artifact,frozen,rows,evaluation,metadata,events = fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);run=base/"run";suite=base/"suite";saved=base/"frozen"
            for folder in (run,suite,saved):folder.mkdir()
            def write(path,value):path.write_text(json.dumps(value),encoding="utf-8")
            write(saved/"residual_artifact.json",artifact)
            frozen["artifact_sha256"] = uncertainty.sha(saved/"residual_artifact.json")
            write(saved/"predictions_synthetic.json",frozen)
            (saved/"residual_forecaster_frozen.py").write_text("# frozen fixture\n",encoding="utf-8")
            write(run/"forecasts.json",[])
            write(suite/"research_metadata.json",metadata);write(suite/"manifest.json",{})
            write(base/"rows.json",rows)
            (run/"calls.jsonl").write_text("\n".join(json.dumps(r) for r in events),encoding="utf-8")
            for key,path in (("forecasts_sha256",run/"forecasts.json"),
                             ("metadata_sha256",suite/"research_metadata.json"),("manifest_sha256",suite/"manifest.json")):
                evaluation["protocol"][key] = uncertainty.sha(path)
            write(run/"evaluation.json",evaluation)
            freeze = {"frozen_at_utc":"fixture", "evaluation_scope":{"synthetic":"development"},
                "output_sha256":{name:uncertainty.sha(saved/file) for name,file in (
                    ("artifact","residual_artifact.json"),("predictions_synthetic","predictions_synthetic.json"),
                    ("module","residual_forecaster_frozen.py"))},
                "input_sha256":{"synthetic":uncertainty.sha(run/"forecasts.json")}}
            write(saved/"freeze_record.json",freeze)
            kwargs = dict(run=run,suite=suite,rows_path=base/"rows.json",frozen_dir=saved,dataset="synthetic")
            report = uncertainty.build_report(**kwargs)
            self.assertTrue(report["normalized_labels_reconciled"])
            self.assertEqual(len(report["input_sha256"]),10)
            self.assertFalse(report["fit_or_selection"])
            self.assertEqual(report["models"]["model"]["heads"]["total_policy"]["comparisons"]["guarded"]["secondary_wape"]["delta"],-.1)
            write(run/"forecasts.json",["changed"])
            with self.assertRaisesRegex(ValueError,"hash"):
                uncertainty.build_report(**kwargs)


if __name__ == "__main__":
    unittest.main()
