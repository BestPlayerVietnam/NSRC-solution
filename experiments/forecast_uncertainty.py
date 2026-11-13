"""Post-run uncertainty for frozen forecasts; no fitting or policy selection.

CLI: python -B experiments/forecast_uncertainty.py --run RUN/test --suite SUITE
Requires a complete evaluation and frozen forecasts. Writes a separate new
uncertainty_report.json; never modifies forecasts, policy, runtime or ledgers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261112
RESAMPLES = 2000
CONFIDENCE = 0.95


def finite(value, name, *, nonnegative=True):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric, not bool")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return result


def mean(values):
    values = list(values)
    return math.fsum(v / len(values) for v in values) if values else None


def percentile(values, probability):
    """Linear empirical percentile, including exact endpoints."""
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be in [0, 1]")
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    fraction = position - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def summary(values):
    values = list(values)
    return {"n": len(values), "mean": mean(values), "median": percentile(values, .5),
            "p90": percentile(values, .9), "max": max(values) if values else None}


def paired_group_bootstrap(rows, baseline, *, seed=SEED, resamples=RESAMPLES):
    """Paired cluster bootstrap: same group draw for both forecast errors.

    Macro estimand weights source groups equally. Secondary row estimand weights
    each observation equally; its bootstrap still resamples whole source groups.
    A negative delta favors the guarded deployed prediction.
    """
    if baseline not in {"heuristic", "prompt_prior"}:
        raise ValueError("Unsupported baseline")
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    groups = defaultdict(list)
    for row in rows:
        ident = row.get("group")
        if not isinstance(ident, str) or not ident:
            raise ValueError("Missing source group")
        actual = finite(row["actual"], "actual")
        deployed = finite(row["deployed"], "deployed")
        reference = finite(row[baseline], baseline)
        groups[ident].append((abs(deployed - actual), abs(reference - actual)))
    cells = []
    for ident in sorted(groups):
        pairs = groups[ident]
        cells.append({"n": len(pairs), "deployed": mean(d for d, _ in pairs),
                      "baseline": mean(b for _, b in pairs),
                      "delta": mean(d - b for d, b in pairs)})
    n = sum(c["n"] for c in cells)
    primary = {"deployed_mae": mean(c["deployed"] for c in cells),
               "baseline_mae": mean(c["baseline"] for c in cells),
               "delta_mae": mean(c["delta"] for c in cells), "ci95": None}
    secondary = {
        "deployed_mae": math.fsum(c["deployed"] * (c["n"]/n) for c in cells) if n else None,
        "baseline_mae": math.fsum(c["baseline"] * (c["n"]/n) for c in cells) if n else None,
        "delta_mae": math.fsum(c["delta"] * (c["n"]/n) for c in cells) if n else None,
        "ci95": None}
    status = "insufficient_source_groups" if len(cells) < 2 else "ok"
    if len(cells) >= 2:
        rng = random.Random(seed)
        macro_samples, row_samples = [], []
        for _ in range(resamples):
            draw = [cells[rng.randrange(len(cells))] for _ in cells]
            macro_samples.append(mean(c["delta"] for c in draw))
            size = sum(c["n"] for c in draw)
            row_samples.append(math.fsum(c["delta"] * (c["n"]/size) for c in draw))
        primary["ci95"] = [percentile(macro_samples, .025), percentile(macro_samples, .975)]
        secondary["ci95"] = [percentile(row_samples, .025), percentile(row_samples, .975)]
    return {"baseline": baseline, "n": n, "groups": len(cells),
            "delta_definition": "absolute_error_guarded_deployed minus absolute_error_baseline",
            "negative_favors": "guarded_deployed", "primary_group_macro": primary,
            "secondary_row_mae": secondary, "seed": seed,
            "resamples": resamples if status == "ok" else 0,
            "requested_resamples": resamples, "confidence": CONFIDENCE,
            "interval_method": "paired source-group percentile bootstrap",
            "interval_status": status}


def charged_calls(events):
    """Reproduce frozen runner's per-physical-call reconciliation, without imports."""
    grouped = {}
    for event in events:
        if event.get("id"):
            grouped.setdefault(event["id"], []).append(event)
    charged = []
    for call_id, records in grouped.items():
        initial = next((r for r in records if r["event"] == "reserved"), records[0])
        terminal = next((r for r in records if r["event"] == "completed"), None)
        if terminal is None:
            terminal = next((r for r in records if r["event"] == "error"), None)
        charged.append({**initial, **(terminal or {}), "id": call_id,
            "arm": initial["arm"],
            "cost": terminal["cost"] if terminal else initial["reserved"],
            "accounting": terminal.get("accounting", "reserved_error") if terminal else "reserved_pending"})
    return charged


def upper_diagnostics(rows):
    """Raw one-sided upper magnitude/headroom; not a two-sided interval width."""
    uppers, headrooms, ratios = [], [], []
    coverage = []
    coverage_by_group = defaultdict(list)
    eligible_groups = Counter(r["group"] for r in rows)
    missing = nonfinite = inconsistent = zero_point = 0
    for row in rows:
        raw = row["raw_forecast"]
        upper, point = raw.get("total_policy_upper"), raw.get("total_policy_mean")
        if upper is None:
            missing += 1
            continue
        try:
            upper = finite(upper, "total_policy_upper")
            point = finite(point, "total_policy_mean")
        except ValueError:
            nonfinite += 1
            continue
        if upper < point:
            inconsistent += 1
            continue
        covered = row["actual"] <= upper
        uppers.append(upper)
        headrooms.append(upper - point)
        coverage.append(covered)
        coverage_by_group[row["group"]].append(covered)
        if point > 0:
            ratio = upper / point
            if math.isfinite(ratio):
                ratios.append(ratio)
        else:
            zero_point += 1
    complete_groups = [values for g, values in coverage_by_group.items()
                       if len(values) == eligible_groups[g]]
    return {"scope": "raw total-policy one-sided upper, before deployment fallback",
            "eligible_rows": len(rows), "available_finite_consistent_bounds": len(uppers),
            "missing_upper": missing, "nonfinite_or_missing_point": nonfinite,
            "upper_below_point": inconsistent, "zero_raw_point": zero_point,
            "upper_tokens": summary(uppers), "upper_minus_raw_point_tokens": summary(headrooms),
            "upper_over_raw_point": summary(ratios),
            "row_empirical_coverage": mean(float(x) for x in coverage),
            "group_macro_empirical_coverage": mean(mean(float(x) for x in values)
                                                    for values in coverage_by_group.values()),
            "fully_observed_groups": len(complete_groups),
            "all_rows_covered_group_fraction": mean(float(all(values)) for values in complete_groups),
            "coverage_guarantee": False,
            "interpretation": "Large uppers can attain high coverage trivially; inspect magnitude and headroom."}


def analyze_objects(evaluation, forecasts, metadata, events):
    if evaluation.get("comparison_complete") is not True:
        raise RuntimeError("Evaluation is incomplete; run after the pipeline report is complete")
    protocol = evaluation["protocol"]
    lookup = {}
    for forecast in forecasts:
        key = (forecast["id"], forecast["model"])
        if key in lookup:
            raise ValueError("Duplicate frozen forecast")
        lookup[key] = forecast
    calls_by_task = defaultdict(list)
    for call in charged_calls(events):
        calls_by_task[(call["arm"], call["task"])].append(call)
    models = {}
    for arm, config in protocol["arms"].items():
        if config.get("retry_policy") != "adaptive":
            continue
        model = config["initial_model"]
        if model in models:
            raise ValueError("Multiple adaptive arms for one model; keep pipelines separate")
        rows, audit = [], Counter({name: 0 for name in (
            "all_tasks", "not_attempted", "attempted", "unknown_usage",
            "incomplete_execution", "excluded_total", "complete_known_total")})
        seen = set()
        for row in evaluation["arms"][arm]["rows"]:
            ident = row["id"]
            if ident in seen:
                raise ValueError("Duplicate evaluation task")
            seen.add(ident)
            audit["all_tasks"] += 1
            calls = calls_by_task.get((arm, ident), [])
            if not calls:
                audit["not_attempted"] += 1
                continue
            audit["attempted"] += 1
            known = all(c.get("accounting") == "provider_reconciled" for c in calls)
            trace = row.get("trace") or {}
            if not known:
                audit["unknown_usage"] += 1
            if trace.get("execution_complete") is not True:
                audit["incomplete_execution"] += 1
            if not known or trace.get("execution_complete") is not True:
                audit["excluded_total"] += 1
                continue
            actual = math.fsum(finite(c["cost"], "call_cost") for c in calls)
            if abs(actual - finite(row["cost"], "evaluation_cost")) > .011:
                raise ValueError("Evaluation cost and reconciled physical calls disagree")
            record = lookup.get((ident, model))
            if record is None:
                raise ValueError("No frozen forecast for an eligible observation")
            meta = metadata[ident]
            group = meta.get("group", meta.get("family"))
            if not isinstance(group, str) or not group:
                raise ValueError("Missing source-group metadata")
            deployed = record.get("deployed")
            if not isinstance(deployed, dict) or deployed.get("source") not in {"learned", "prior"}:
                raise ValueError("Frozen guarded deployed forecast is missing")
            fallback = record.get("fallback")
            if not isinstance(fallback, list) or len(fallback) != 2:
                raise ValueError("Frozen prompt-aware prior is missing")
            rows.append({"id": ident, "group": group, "actual": actual,
                "deployed": finite(deployed.get("total_policy"), "deployed_total"),
                "heuristic": finite(record.get("heuristic"), "heuristic"),
                "prompt_prior": finite(fallback[1], "prompt_prior"),
                "source": deployed["source"], "fallback_reason": deployed.get("reason"),
                "raw_forecast": record.get("forecast") or {}})
        audit["complete_known_total"] = len(rows)
        fallback_reasons = Counter(str(r["fallback_reason"] or "unspecified")
                                   for r in rows if r["source"] == "prior")
        models[model] = {"arm": arm, "population_audit": dict(audit),
            "eligible_task_ids": [r["id"] for r in rows],
            "groups": len({r["group"] for r in rows}),
            "comparisons": {baseline: paired_group_bootstrap(rows, baseline)
                            for baseline in ("heuristic", "prompt_prior")},
            "deployment_fallback": {"n": len(rows), "count": sum(fallback_reasons.values()),
                "fraction": sum(fallback_reasons.values()) / len(rows) if rows else None,
                "reasons": dict(fallback_reasons),
                "raw_warning_count": sum(bool(r["raw_forecast"].get("warnings")) for r in rows)},
            "upper_diagnostics": upper_diagnostics(rows)}
    return {"analysis": "paired frozen pre-run total-cost forecasts on the same observed trajectories",
        "policy_or_model_fit": False, "models": models,
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "confidence": CONFIDENCE,
            "primary_estimand": "equal source-group macro mean absolute error difference",
            "secondary_estimand": "row mean absolute error difference, resampling whole source groups"},
        "limitations": [
            "Only adaptive attempted, provider-reconciled, complete trajectories are included; skips and censored observations are excluded.",
            "Guarded deployed means the frozen pre-run selector output, before later online cost rescaling.",
            "Percentile cluster bootstrap describes sampling uncertainty under the supplied source grouping, not a coverage guarantee.",
            "Few source groups and correlated benchmark families limit inference; these are not independent model reruns.",
            "Intervals are unadjusted for multiple model/baseline comparisons.",
            "Raw one-sided upper coverage is empirical; upper magnitude and headroom must also be inspected.",
            "This report neither chooses a policy nor changes forecasts, fitting data, execution or budget."]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_report(run, suite):
    run, suite = Path(run), Path(suite)
    evaluation = read(run / "evaluation.json")
    if evaluation.get("comparison_complete") is not True:
        raise RuntimeError("Evaluation is not complete yet")
    protocol = evaluation["protocol"]
    paths = {"evaluation": run / "evaluation.json", "forecasts": run / "forecasts.json",
             "calls": run / "calls.jsonl", "metadata": suite / "research_metadata.json",
             "manifest": suite / "manifest.json"}
    for key, source in (("forecasts_sha256", "forecasts"), ("metadata_sha256", "metadata"),
                        ("manifest_sha256", "manifest")):
        if protocol.get(key) != sha(paths[source]):
            raise RuntimeError("Frozen input hash mismatch or absent: " + key)
    events = [json.loads(line) for line in paths["calls"].read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    result = analyze_objects(evaluation, read(paths["forecasts"]), read(paths["metadata"]), events)
    result["input_sha256"] = {name: sha(path) for name, path in paths.items()}
    result["analysis_source_sha256"] = sha(Path(__file__))
    result["frozen_forecast_hash_verified"] = True
    result["run_directory"] = str(run.resolve())
    result["suite_directory"] = str(suite.resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    destination = Path(args.run) / "uncertainty_report.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a completed uncertainty report")
    report = build_report(args.run, args.suite)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "models": {
        model: {"groups": value["groups"],
                "delta_macro_vs_heuristic": value["comparisons"]["heuristic"]["primary_group_macro"],
                "delta_macro_vs_prompt_prior": value["comparisons"]["prompt_prior"]["primary_group_macro"],
                "fallback": value["deployment_fallback"]["fraction"]}
        for model, value in report["models"].items()}, "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()

