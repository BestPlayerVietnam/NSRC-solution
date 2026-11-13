"""Independent robust residual-cost experiment. Stdlib only; no API or deployment.

Fit uses training cost labels only. CV scores FINAL nonnegative/ordered outputs.
Freeze consumes existing visible forecasts (features + prior), never evaluations.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION = 1
TARGETS = ("first_call", "total_policy")
SHRINK = (0.0, 2.0, 8.0)
CANDIDATES = [("prior", 0.0)] + [(method, strength)
    for method in ("multiplier", "additive", "input_remaining") for strength in SHRINK]


def number(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(name + " must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def average(values):
    values = list(values)
    return math.fsum(v / len(values) for v in values) if values else None


def weighted_median(values, weights):
    values, weights = list(values), list(weights)
    if len(values) != len(weights):
        raise ValueError("Median values and weights differ in length")
    pairs = [(number(v, "median value"), number(w, "median weight", nonnegative=True))
             for v, w in zip(values, weights)]
    pairs = sorted((v,w) for v,w in pairs if w > 0)
    if not pairs:
        raise ValueError("Weighted median needs positive weight")
    half = math.fsum(w for _, w in pairs) / 2
    cumulative = 0.0
    for i, (value, weight) in enumerate(pairs):
        cumulative += weight
        if cumulative >= half:
            if cumulative == half and i + 1 < len(pairs):
                return value / 2 + pairs[i+1][0] / 2
            return value
    return pairs[-1][0]


def source_group(row):
    value = row.get("task_group", row.get("group"))
    if not isinstance(value, str) or not value:
        raise ValueError("Missing source group")
    return value


def group_weights(rows):
    counts = Counter(source_group(r) for r in rows)
    return [1 / counts[source_group(r)] for r in rows]


def domain_from_features(features):
    kinds = [k for k in ("math", "coding", "search", "data_analysis")
             if number(features.get("kind_" + k, 0), "kind", nonnegative=True) == 1]
    if len(kinds) != 1:
        raise ValueError("Expected exactly one domain indicator")
    return kinds[0]


def prompt_input(features):
    return (number(features["instruction_chars"], "instruction_chars", nonnegative=True) +
            number(features["context_chars"], "context_chars", nonnegative=True)) / 35 + 18


def prompt_prior(features, model):
    """Exactly the frozen production prior; freeze verifies stored prior tuples."""
    prompt = prompt_input(features)
    kind = domain_from_features(features)
    if kind == "coding":
        output = 85 + min(1400, number(features.get("source_bytes", 0), "source_bytes", nonnegative=True)/6)
        retry = 0 if features.get("contradiction_flag") else .2
    elif kind == "data_analysis":
        output, retry = 40, .12
    elif kind == "search":
        output, retry = 8, .05
    else:
        output, retry = 24, .06
    first = prompt + output * (1 if model.endswith("-noreason") else 5)
    return first, first + retry * (prompt + 650)


def project(first, total):
    """The same fixed numeric projection is used by CV and inference."""
    first = max(1.0, number(first, "first point"))
    total = max(first, 1.0, number(total, "total point"))
    return first, total


def label(row, target):
    if target == "first_call":
        return (number(row["first_call_cost"], "first cost", nonnegative=True)
                if row.get("first_usage_known") is True else None)
    return (number(row["total_policy_cost"], "total cost", nonnegative=True)
            if row.get("total_usage_known") is True and row.get("execution_complete") is True else None)


def prepare_rows(rows, policy_id):
    output, seen = [], set()
    for original in rows:
        if original.get("source_type") != "real_api" or original.get("policy_id") != policy_id:
            raise ValueError("Expected real API rows from the exact recorded policy")
        row = dict(original)
        key = (row["model"], row["id"])
        if key in seen:
            raise ValueError("Choose one predeclared replicate per model/task")
        seen.add(key)
        source_group(row)
        row["domain"] = domain_from_features(row["features"])
        row["_prior"] = prompt_prior(row["features"], row["model"])
        row["_input"] = prompt_input(row["features"])
        a, b = label(row, "first_call"), label(row, "total_policy")
        if a is not None and b is not None and a > b + 1e-8:
            raise ValueError("First cost exceeds total cost")
        output.append(row)
    return output


def fold_ids(rows, folds=4):
    groups = sorted({source_group(r) for r in rows},
                    key=lambda g: hashlib.sha256(("residual-v1|" + g).encode()).hexdigest())
    n = min(folds, len(groups))
    return {g: i % n for i, g in enumerate(groups)} if n else {}


def fit_head(rows, target, candidate):
    method, strength = candidate
    selected = [r for r in rows if label(r, target) is not None]
    index = TARGETS.index(target)
    if method == "prior" or not selected:
        return {"method": "prior", "strength": 0.0, "groups": len({source_group(r) for r in selected})}
    weights = group_weights(selected)
    values, median_weights = [], []
    for row, weight in zip(selected, weights):
        actual, prior = label(row, target), row["_prior"][index]
        if method == "multiplier":
            value, median_weight = actual / prior, prior * weight
        elif method == "additive":
            value, median_weight = actual - prior, weight
        elif method == "input_remaining":
            value, median_weight = max(0.0, actual - row["_input"]), weight
        else:
            raise ValueError("Unknown residual method")
        values.append(value); median_weights.append(median_weight)
    global_value = weighted_median(values, median_weights)
    domains = {}
    for domain in sorted({r["domain"] for r in selected}):
        indices = [i for i, r in enumerate(selected) if r["domain"] == domain]
        local = weighted_median([values[i] for i in indices], [median_weights[i] for i in indices])
        support = math.fsum(weights[i] for i in indices)
        domains[domain] = {"value": (local * support + strength * global_value) / (support + strength),
                           "effective_groups": support}
    return {"method": method, "strength": strength, "global": global_value,
            "domains": domains, "groups": len({source_group(r) for r in selected})}


def raw_point(head, prior, input_cost, domain):
    method = head["method"]
    if method == "prior":
        return prior
    value = head["domains"].get(domain, {}).get("value", head["global"])
    if method == "multiplier":
        return prior * value
    if method == "additive":
        return prior + value
    if method == "input_remaining":
        return input_cost + value
    raise ValueError("Unknown artifact method")


def macro_mae(rows, predictions, target):
    groups = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        actual = label(row, target)
        if actual is not None:
            groups[source_group(row)].append(abs(prediction - actual))
    return average(average(v) for v in groups.values()) if groups else None


def pair_scores(rows, first, total):
    points = []
    for row, a, b in zip(rows, first, total):
        try:
            points.append(project(a,b))
        except (ValueError,OverflowError):
            points.append(project(*row["_prior"]))
    return {"first_macro_mae": macro_mae(rows, [x[0] for x in points], "first_call"),
            "total_macro_mae": macro_mae(rows, [x[1] for x in points], "total_policy")}


def fit(rows, *, policy_id, folds=4):
    if not isinstance(folds,int) or isinstance(folds,bool) or folds < 2:
        raise ValueError("At least two grouped CV folds are required")
    rows = prepare_rows(list(rows), policy_id)
    if not rows:
        raise ValueError("No training rows")
    models, reports = {}, {}
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        assignments = fold_ids(subset, folds)
        scores = []
        first_choice = total_choice = 0
        if len(set(assignments.values())) >= 2:
            oof = {t: [[None] * len(subset) for _ in CANDIDATES] for t in TARGETS}
            for fold in sorted(set(assignments.values())):
                training = [r for r in subset if assignments[source_group(r)] != fold]
                held = [(i, r) for i, r in enumerate(subset) if assignments[source_group(r)] == fold]
                for target in TARGETS:
                    for c, candidate in enumerate(CANDIDATES):
                        head = fit_head(training, target, candidate)
                        for i, row in held:
                            value = raw_point(head, row["_prior"][TARGETS.index(target)], row["_input"], row["domain"])
                            oof[target][c][i] = value
            for first_choice in range(len(CANDIDATES)):
                for total_choice in range(len(CANDIDATES)):
                    measured = pair_scores(subset, oof["first_call"][first_choice], oof["total_policy"][total_choice])
                    scores.append({"first_candidate": first_choice, "total_candidate": total_choice, **measured})
            def objective(item):
                a, b = item["first_macro_mae"], item["total_macro_mae"]
                complexity = int(item["first_candidate"] != 0) + int(item["total_candidate"] != 0)
                return (math.inf if b is None else b, math.inf if a is None else a,
                        complexity, item["first_candidate"], item["total_candidate"])
            winner = min(scores, key=objective)
            first_choice, total_choice = winner["first_candidate"], winner["total_candidate"]
        else:
            winner = {"first_candidate": 0, "total_candidate": 0,
                      "first_macro_mae": None, "total_macro_mae": None}
        chosen = {"first_call": first_choice, "total_policy": total_choice}
        models[model] = {"heads": {target: fit_head(subset, target, CANDIDATES[chosen[target]])
                                  for target in TARGETS},
                         "selected_candidates": chosen}
        reports[model] = {"rows": len(subset), "groups": len(set(assignments)),
            "known_first": sum(label(r, "first_call") is not None for r in subset),
            "known_complete_total": sum(label(r, "total_policy") is not None for r in subset),
            "chosen": winner, "baseline_prior_pair": scores[0] if scores else winner,
            "candidate_pair_scores": scores,
            "selection_note": "train-only grouped CV; total MAE primary, first MAE tie-break; final numeric projection applied"}
    artifact = {"schema_version": VERSION, "policy_id": policy_id, "models": models,
        "train_groups": sorted({source_group(r) for r in rows}),
        "train_ids": sorted({r["id"] for r in rows}),
        "candidates": [{"method": method, "shrink": strength} for method, strength in CANDIDATES],
        "cv_folds": folds, "selection": "lexicographic (group-macro total MAE, group-macro first MAE, simpler pair)",
        "projection": "first=max(1,first); total=max(first,1,total)",
        "estimator_target": "robust L1 point estimate, not a calibrated conditional expectation",
        "input_remaining_definition": "max(0, observed_cost - estimated_prompt_input_cost)",
        "domain_shrinkage": "blend domain/global weighted medians by effective source-group count and fixed shrink strength",
        "calibration_or_test_labels_used": False}
    ResidualForecaster(artifact)
    return artifact, {"scope": "training model-selection diagnostics, not heldout evidence", "models": reports}


def guarded_oof_diagnostic(rows, *, policy_id, folds=4):
    """Nested grouped-CV comparator, never a residual candidate-selection input.

    Refit the existing cost model inside each outer training split, including
    its own train-only grouped selection. Calibration changes upper bounds,
    not these point predictions, so no calibration labels are needed here.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import cost_model
    import forecast_guard

    rows = prepare_rows(list(rows), policy_id)
    assignments = fold_ids(rows, folds)
    results = {}
    if len(set(assignments.values())) < 2:
        return {"scope": "No independent outer fold available", "models": results}
    names = sorted({name for row in rows for name in row["features"]})
    predictions = {}
    for fold in sorted(set(assignments.values())):
        training = [r for r in rows if assignments[source_group(r)] != fold]
        held = [r for r in rows if assignments[source_group(r)] == fold]
        fitted = cost_model.fit_cost_artifact(training, feature_names=names,
            policy_id=policy_id, criterion="mae")
        estimator = cost_model.CostModel(fitted)
        for row in held:
            forecast = estimator.predict(row["features"], model=row["model"],
                policy_id=policy_id, domain=row["domain"])
            predictions[(row["model"], row["id"])] = forecast_guard.choose_forecast(
                forecast, *row["_prior"], pipeline_budget=100000)
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        points = [predictions[(model, r["id"])] for r in subset]
        results[model] = {**pair_scores(subset, [p.first_call for p in points],
            [p.total_policy for p in points]), "rows": len(subset),
            "fallback_count": sum(p.source == "prior" for p in points),
            "fallback_reasons": dict(Counter(p.reason for p in points if p.reason))}
    return {"scope": "Train-only OOF diagnostic; same outer source-group folds, inner grouped CV; not used to select residual candidates",
        "outer_folds": len(set(assignments.values())), "criterion": "mae",
        "calibration_labels_used": False, "models": results,
        "code_sha256": {"cost_model.py": sha(cost_model.__file__),
                        "forecast_guard.py": sha(forecast_guard.__file__)}}


@dataclass(frozen=True)
class ResidualPrediction:
    first_call: float
    total_policy: float
    source: str
    reason: str | None


class ResidualForecaster:
    def __init__(self, artifact):
        self.artifact = json.loads(json.dumps(artifact, allow_nan=False))
        if self.artifact.get("schema_version") != VERSION:
            raise ValueError("Unknown residual artifact")
        for scope in self.artifact["models"].values():
            for head in scope["heads"].values():
                if head["method"] not in {c[0] for c in CANDIDATES}:
                    raise ValueError("Unknown residual method")
                if head["method"] != "prior":
                    values = [head["global"]] + [x["value"] for x in head["domains"].values()]
                    for value in values:
                        number(value, "coefficient", nonnegative=head["method"] != "additive")

    def predict(self, features, *, model, domain=None, prior=None, policy_id=None):
        prior = prompt_prior(features, model) if prior is None else tuple(prior)
        if len(prior) != 2 or any(number(v, "prior", nonnegative=True) <= 0 for v in prior) or prior[1] < prior[0]:
            raise ValueError("Invalid prompt prior")
        domain = domain or domain_from_features(features)
        if policy_id is not None and policy_id != self.artifact["policy_id"]:
            return ResidualPrediction(*prior, "prior", "policy_mismatch")
        scope = self.artifact["models"].get(model)
        if scope is None:
            return ResidualPrediction(*prior, "prior", "model_not_trained")
        try:
            input_cost = prompt_input(features)
            raw = [raw_point(scope["heads"][target], prior[i], input_cost, domain)
                   for i, target in enumerate(TARGETS)]
            first, total = project(*raw)
        except (ValueError, OverflowError):
            return ResidualPrediction(*prior, "prior", "invalid_numeric_prediction")
        return ResidualPrediction(first, total, "residual", None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_visible(predictor, records, *, artifact_sha):
    output = []
    seen = set()
    for record in records:
        key = (record["id"], record["model"])
        if key in seen or record["id"] in predictor.artifact["train_ids"]:
            raise ValueError("Duplicate visible forecast or training task ID overlap")
        seen.add(key)
        features, model = record["features"], record["model"]
        prior = record["fallback"]
        computed = prompt_prior(features, model)
        if len(prior) != 2 or any(abs(a-b) > 1e-8 * max(1, abs(a)) for a,b in zip(computed, prior)):
            raise ValueError("Stored frozen prior differs from residual experiment formula")
        forecast = predictor.predict(features, model=model, prior=prior)
        output.append({"id": record["id"], "model": model,
            "prediction": asdict(forecast), "prompt_prior": list(prior),
            "heuristic": record.get("heuristic"), "original_deployed": record.get("deployed"),
            "visible_features_sha256": hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()})
    return {"artifact_sha256": artifact_sha, "labels_read": False,
        "scope": "Predictions frozen after pipeline execution began, before this experiment accesses grading or outcome labels.",
        "predictions": output}


def fit_and_freeze(train_path, out_dir, visible_paths, policy_id):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new directory; residual experiment is frozen once written")
    if not visible_paths or len(visible_paths) != len(set(visible_paths)):
        raise ValueError("Distinct visible forecast labels are required")
    training = read(train_path)
    artifact, report = fit(training, policy_id=policy_id)
    report["existing_guarded_oof_diagnostic"] = guarded_oof_diagnostic(training, policy_id=policy_id)
    # Validate every visible input before writing any immutable output.
    predictor = ResidualForecaster(artifact)
    visible = {}
    for name, path in visible_paths.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Visible forecast label must be alphanumeric")
        visible[name] = freeze_visible(predictor, read(path), artifact_sha="pending")
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "residual_artifact.json"
    write_new(artifact_path, artifact)
    write_new(out / "train_cv_report.json", report)
    frozen_source = out / "residual_forecaster_frozen.py"
    with frozen_source.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    written = {"artifact": artifact_path, "train_cv_report": out/"train_cv_report.json",
               "module": frozen_source}
    for name, path in visible_paths.items():
        destination = out / ("predictions_" + name + ".json")
        visible[name]["artifact_sha256"] = sha(artifact_path)
        write_new(destination, visible[name])
        written["predictions_" + name] = destination
    manifest = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "policy_id": policy_id, "training_only_selection": True,
        "calibration_labels_read": False, "public_evaluation_or_results_read": False,
        "public_collection_already_started": True, "api_calls": 0, "runtime_applied": False,
        "input_sha256": {"training_rows": sha(train_path), **{name: sha(path) for name,path in visible_paths.items()}},
        "output_sha256": {name: sha(path) for name,path in written.items()},
        "evaluation_scope": {"synthetic": "development diagnostic: earlier results have already been seen",
                             "public100": "independent labels-unread forecast test; freeze is after collection started, not before collection"}}
    write_new(out / "freeze_record.json", manifest)
    return report, manifest


def evaluate_frozen(artifact_path, predictions_path, rows_path):
    artifact, frozen = read(artifact_path), read(predictions_path)
    if frozen["artifact_sha256"] != sha(artifact_path):
        raise ValueError("Frozen residual artifact changed")
    lookup = {(r["id"], r["model"]):r for r in frozen["predictions"]}
    rows = prepare_rows(read(rows_path), artifact["policy_id"])
    if set(artifact["train_groups"]) & {source_group(r) for r in rows}:
        raise ValueError("Evaluation source groups overlap training")
    report = {}
    for model in sorted({r["model"] for r in rows}):
        report[model] = {}
        for target in TARGETS:
            by_method = defaultdict(list)
            for row in rows:
                if row["model"] != model:
                    continue
                actual = label(row, target)
                if actual is None:
                    continue
                saved = lookup[(row["id"], model)]
                values = {"residual": saved["prediction"][target],
                          "prompt_prior": saved["prompt_prior"][TARGETS.index(target)]}
                if saved.get("original_deployed"):
                    values["original_guarded"] = saved["original_deployed"][target]
                if target == "total_policy" and saved.get("heuristic") is not None:
                    values["old_heuristic"] = saved["heuristic"]
                for method, value in values.items():
                    by_method[method].append((source_group(row), abs(value-actual)))
            for method, pairs in by_method.items():
                groups = defaultdict(list)
                for group, error in pairs:
                    groups[group].append(error)
                report[model].setdefault(target, {})[method] = {
                    "n":len(pairs), "groups":len(groups), "row_mae":average(e for _,e in pairs),
                    "group_macro_mae":average(average(v) for v in groups.values())}
    return {"artifact_sha256":sha(artifact_path), "predictions_sha256":sha(predictions_path),
            "rows_sha256":sha(rows_path), "fit_or_selection":False, "models":report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    prepare=sub.add_parser("fit-freeze")
    prepare.add_argument("--train",required=True);prepare.add_argument("--out-dir",required=True)
    prepare.add_argument("--policy-id",required=True);prepare.add_argument("--visible",action="append",required=True,
        help="LABEL=path/to/existing/forecasts.json; contains visible features, no labels")
    evaluate=sub.add_parser("evaluate")
    evaluate.add_argument("--artifact",required=True);evaluate.add_argument("--predictions",required=True)
    evaluate.add_argument("--rows",required=True);evaluate.add_argument("--out",required=True)
    args=parser.parse_args()
    if args.action=="fit-freeze":
        pairs=[entry.split("=",1) for entry in args.visible]
        if any(len(p)!=2 for p in pairs) or len({p[0] for p in pairs})!=len(pairs):
            raise ValueError("Use distinct LABEL=PATH arguments")
        report,manifest=fit_and_freeze(args.train,args.out_dir,dict(pairs),args.policy_id)
        print(json.dumps({"frozen":manifest,"cv_choices":{m:{"chosen":r["chosen"],
              "prior":r["baseline_prior_pair"]} for m,r in report["models"].items()}},indent=2))
    else:
        result=evaluate_frozen(args.artifact,args.predictions,args.rows)
        write_new(args.out,result)
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

