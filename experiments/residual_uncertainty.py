"""Post-run paired uncertainty for frozen residual forecasts. No fit or selection.

Reads normalized rows only after evaluation.json says comparison_complete.
Compares frozen residual/guarded/prior on identical observations for each head.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.forecast_uncertainty import (
    SEED, RESAMPLES, finite, mean, percentile, paired_group_bootstrap,
    charged_calls, read, sha,
)

TARGETS = ("first_call", "total_policy")
BASELINES = ("guarded", "prompt_prior")


def compare(rows, baseline):
    """Reuse the paired MAE helper with explicit semantic relabeling."""
    if baseline not in BASELINES:
        raise ValueError("Unknown residual comparison baseline")
    mapped = [{"group": r["group"], "actual": r["actual"],
               "deployed": r["residual"], "heuristic": r["guarded"],
               "prompt_prior": r["prompt_prior"]} for r in rows]
    result = paired_group_bootstrap(mapped, "heuristic" if baseline == "guarded" else baseline)
    result["baseline"] = baseline
    result["negative_favors"] = "residual"
    result["delta_definition"] = "absolute_error_residual minus absolute_error_" + baseline
    for name in ("primary_group_macro", "secondary_row_mae"):
        result[name]["residual_mae"] = result[name].pop("deployed_mae")

    groups = defaultdict(list)
    for row in rows:
        actual = finite(row["actual"], "actual")
        groups[row["group"]].append((actual,
            abs(finite(row["residual"], "residual") - actual),
            abs(finite(row[baseline], baseline) - actual)))
    cells = [tuple(math.fsum(r[i] for r in groups[g]) for i in range(3)) for g in sorted(groups)]
    denominator = math.fsum(c[0] for c in cells)
    residual_error = math.fsum(c[1] for c in cells)
    baseline_error = math.fsum(c[2] for c in cells)
    wape = {"residual": residual_error / denominator if denominator else None,
        "baseline": baseline_error / denominator if denominator else None,
        "delta": (residual_error - baseline_error) / denominator if denominator else None,
        "actual_cost_sum": denominator, "ci95": None, "valid_resamples": 0,
        "definition": "sum_absolute_error / sum_actual_cost; ratio, not percent",
        "interval_status": "zero_actual_denominator" if not denominator else "insufficient_source_groups"}
    if denominator and len(cells) >= 2:
        rng = random.Random(SEED)
        samples = []
        for _ in range(RESAMPLES):
            draw = [cells[rng.randrange(len(cells))] for _ in cells]
            divisor = math.fsum(c[0] for c in draw)
            if divisor:
                samples.append(math.fsum(c[1] - c[2] for c in draw) / divisor)
        wape["valid_resamples"] = len(samples)
        if len(samples) == RESAMPLES:
            wape["ci95"] = [percentile(samples, .025), percentile(samples, .975)]
            wape["interval_status"] = "ok"
        else:
            # Do not silently condition on bootstrap draws having positive cost.
            wape["interval_status"] = "zero_denominator_in_some_resamples"
    result["secondary_wape"] = wape
    return result


def analyze_rows(artifact, frozen, rows):
    lookup = {}
    for record in frozen["predictions"]:
        key = (record["id"], record["model"])
        if key in lookup:
            raise ValueError("Duplicate frozen prediction")
        lookup[key] = record
    seen = set()
    by_model = defaultdict(list)
    for row in rows:
        if row.get("source_type") != "real_api" or row.get("policy_id") != artifact["policy_id"]:
            raise ValueError("Rows must be real API observations from the exact frozen policy")
        key = (row["id"], row["model"])
        if key in seen:
            raise ValueError("Duplicate normalized observation")
        seen.add(key)
        if key not in lookup:
            raise ValueError("Observation has no frozen prediction")
        group = row.get("task_group")
        if not isinstance(group, str) or not group:
            raise ValueError("Missing source group")
        if group in artifact["train_groups"] or row["id"] in artifact["train_ids"]:
            raise ValueError("Evaluation overlaps residual training")
        if row.get("first_usage_known") is True and row.get("total_usage_known") is True:
            if finite(row["first_call_cost"], "first cost") > finite(row["total_policy_cost"], "total cost") + 1e-8:
                raise ValueError("First cost exceeds total cost")
        by_model[row["model"]].append(row)
    models = {}
    for model in sorted({key[1] for key in lookup}):
        subset = by_model[model]
        heads = {}
        for target in TARGETS:
            eligible = []
            for row in subset:
                usable = row.get("first_usage_known") is True if target == "first_call" else (
                    row.get("total_usage_known") is True and row.get("execution_complete") is True)
                if not usable:
                    continue
                saved = lookup[(row["id"], model)]
                original = saved.get("original_deployed")
                if not isinstance(original, dict):
                    raise ValueError("Guarded forecast missing; cannot silently change comparison population")
                eligible.append({"id": row["id"], "group": row["task_group"],
                    "actual": finite(row[target + "_cost"], "actual cost"),
                    "residual": finite(saved["prediction"][target], "residual"),
                    "guarded": finite(original[target], "guarded"),
                    "prompt_prior": finite(saved["prompt_prior"][TARGETS.index(target)], "prior")})
            heads[target] = {"n": len(eligible), "groups": len({r["group"] for r in eligible}),
                "eligible_task_ids": [r["id"] for r in eligible],
                "label_scope": ("Attempted first call with provider-known usage, even if later execution is interrupted"
                    if target == "first_call" else "Attempted trajectory with provider-known full usage and complete policy execution"),
                "comparisons": {baseline: compare(eligible, baseline) for baseline in BASELINES}}
        frozen_tasks = sum(key[1] == model for key in lookup)
        models[model] = {"population_audit": {"frozen_tasks": frozen_tasks,
            "attempted": len(subset), "not_attempted": frozen_tasks - len(subset),
            "unknown_first": sum(r.get("first_usage_known") is not True for r in subset),
            "unknown_total": sum(r.get("total_usage_known") is not True for r in subset),
            "incomplete_execution": sum(r.get("execution_complete") is not True for r in subset),
            "excluded_total": len(subset) - heads["total_policy"]["n"]}, "heads": heads}
    return {"analysis": "Paired frozen residual versus guarded and prompt prior on identical observed trajectories",
        "policy_id": artifact["policy_id"], "fit_or_selection": False, "api_calls": 0,
        "cost_unit": "output_tokens + input_tokens / 10", "models": models,
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "confidence": .95,
            "unit": "source group, paired between predictors", "primary": "equal-group macro MAE delta",
            "secondary": ["row MAE delta", "WAPE delta"], "negative_favors": "residual"},
        "limitations": [
            "First and total heads have explicitly different eligibility; predictors within a head share exactly the same rows.",
            "Skips are not treated as zero-cost solve labels; incomplete/unknown totals are excluded, not imputed.",
            "Frozen pre-run forecasts are evaluated; later online rescaling is not substituted for the guarded baseline.",
            "Paired percentile group bootstrap gives empirical sampling intervals, not a generalization or coverage guarantee.",
            "Few source groups and correlated tasks limit inference; intervals are unadjusted for multiple comparisons.",
            "WAPE uses total actual cost as its denominator and is undefined if that denominator is zero.",
            "This is an offline forecast comparison, not an evaluation of changed admission, ordering, retry or solver policy."]}


def validate_labels(rows, evaluation, metadata, events, policy_id):
    """Audit normalized labels against reconciled calls, adaptive arms and metadata."""
    calls = defaultdict(list)
    for call in charged_calls(events):
        calls[(call["arm"], call["task"])].append(call)
    expected, models = {}, set()
    for arm, config in evaluation["protocol"]["arms"].items():
        if config.get("retry_policy") != "adaptive":
            continue
        model = config["initial_model"]
        if model in models:
            raise ValueError("Multiple adaptive pipelines for one model")
        models.add(model)
        for result in evaluation["arms"][arm]["rows"]:
            cs = calls.get((arm, result["id"]), [])
            if not cs:
                continue
            trace = result.get("trace") or {}
            if trace.get("policy_id") != policy_id:
                raise ValueError("Observed trace policy differs from residual artifact")
            known = all(c.get("accounting") == "provider_reconciled" for c in cs)
            meta = metadata[result["id"]]
            expected[(result["id"], model)] = {
                "task_group": meta.get("group", meta.get("family")),
                "first_call_cost": finite(cs[0]["cost"], "first call"),
                "total_policy_cost": math.fsum(finite(c["cost"], "call cost") for c in cs),
                "first_usage_known": cs[0].get("accounting") == "provider_reconciled",
                "total_usage_known": known,
                "execution_complete": known and trace.get("execution_complete") is True}
    observed = {(r["id"], r["model"]): r for r in rows}
    if len(observed) != len(rows) or set(observed) != set(expected):
        raise ValueError("Normalized attempted population differs from reconciled adaptive calls")
    for key, wanted in expected.items():
        got = observed[key]
        for field in ("task_group", "first_usage_known", "total_usage_known", "execution_complete"):
            if got.get(field) != wanted[field]:
                raise ValueError("Normalized label mismatch: " + field)
        for field in ("first_call_cost", "total_policy_cost"):
            if abs(finite(got[field], field) - wanted[field]) > .011:
                raise ValueError("Normalized cost disagrees with reconciled calls")


def build_report(*, run, suite, rows_path, frozen_dir, dataset):
    run, suite, frozen_dir = Path(run), Path(suite), Path(frozen_dir)
    paths = {"evaluation": run / "evaluation.json"}
    evaluation = read(paths["evaluation"])
    if evaluation.get("comparison_complete") is not True:
        raise RuntimeError("Run is incomplete; normalized outcome rows have not been opened")
    if dataset not in {"synthetic", "public100"}:
        raise ValueError("Use the frozen dataset label synthetic or public100")
    paths.update({"normalized_rows": Path(rows_path), "calls": run / "calls.jsonl",
        "visible_forecasts": run / "forecasts.json", "metadata": suite / "research_metadata.json",
        "manifest": suite / "manifest.json", "residual_artifact": frozen_dir / "residual_artifact.json",
        "residual_predictions": frozen_dir / ("predictions_" + dataset + ".json"),
        "freeze_record": frozen_dir / "freeze_record.json",
        "frozen_experiment_module": frozen_dir / "residual_forecaster_frozen.py"})
    freeze = read(paths["freeze_record"])
    checks = [(paths["residual_artifact"], freeze["output_sha256"]["artifact"]),
              (paths["residual_predictions"], freeze["output_sha256"]["predictions_" + dataset]),
              (paths["frozen_experiment_module"], freeze["output_sha256"]["module"]),
              (paths["visible_forecasts"], freeze["input_sha256"][dataset])]
    for key in ("visible_forecasts", "metadata", "manifest"):
        protocol_key = "forecasts_sha256" if key == "visible_forecasts" else key + "_sha256"
        checks.append((paths[key], evaluation["protocol"].get(protocol_key)))
    if any(sha(path) != wanted for path, wanted in checks):
        raise ValueError("Frozen source hash mismatch")
    artifact = read(paths["residual_artifact"])
    frozen = read(paths["residual_predictions"])
    if frozen["artifact_sha256"] != sha(paths["residual_artifact"]):
        raise ValueError("Prediction artifact provenance mismatch")
    rows = read(paths["normalized_rows"])
    events = [json.loads(line) for line in paths["calls"].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    validate_labels(rows, evaluation, read(paths["metadata"]), events, artifact["policy_id"])
    report = analyze_rows(artifact, frozen, rows)
    report.update({"dataset": dataset, "evaluation_scope": freeze["evaluation_scope"][dataset],
        "frozen_at_utc": freeze["frozen_at_utc"], "normalized_labels_reconciled": True,
        "input_paths": {name: str(path.resolve()) for name,path in paths.items()},
        "input_sha256": {name: sha(path) for name,path in paths.items()},
        "analysis_source_sha256": sha(__file__),
        "bootstrap_helper_sha256": sha(ROOT / "experiments/forecast_uncertainty.py")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "suite", "rows", "frozen-dir", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "public100"), required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing uncertainty report")
    report = build_report(run=args.run, suite=args.suite, rows_path=args.rows,
        frozen_dir=args.frozen_dir, dataset=args.dataset)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "models": {model: {
        target: {baseline: values["comparisons"][baseline]["primary_group_macro"] for baseline in BASELINES}
        for target, values in scope["heads"].items()}
        for model,scope in report["models"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
