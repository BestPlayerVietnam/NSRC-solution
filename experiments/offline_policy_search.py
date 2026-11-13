"""Offline calibration and small scheduler search on recorded real API trajectories.

No simulator, API client, unseen retries, cap changes or model switches.
Training search is exploratory; only evaluate() on new source groups is heldout.
Portfolio costs/outcomes are measured complete trajectories. An over-budget
bundle is infeasible, NOT a fabricated prefix with invented solved tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cost_model import CostModel, evaluate_cost_model, fit_cost_artifact, grouped_fold_ids

SCHEMA_VERSION = 1
TRUSTED_VERIFIERS = {"oracle", "hidden_tests", "evaluation_groundtruth"}
FORBIDDEN_FEATURES = {"answerable", "correct", "final_success", "first_success",
    "hidden_pass", "public_pass", "verdict", "actual_tokens", "first_call_cost",
    "total_policy_cost", "reward", "replicate", "split"}
BASELINE = {"cost_exponent": 1.0, "cost_multiplier": 1.0,
            "overhead": 0.0, "success_exponent": 1.0}
LIMITATIONS = [
    "Recorded full-trajectory portfolio diagnostic, not an end-to-end contest replay.",
    "No counterfactual retries, token caps, model switches, prefixes or outcomes are imputed.",
    "Past task cost/outcome is assumed reusable only for this descriptive bundle calculation.",
    "100000 weighted tokens is a separate primary budget for each model's pipeline.",
    "Smaller budgets are secondary stress diagnostics and never select the primary winner.",
    "Unknown usage, incomplete executions and unverified outcomes are excluded from search labels.",
    "Search on OOF train is model selection, not an unbiased estimate of selected-policy performance.",
    "Public-test success alone is not a trusted hidden-correctness label.",
    "Empirical calibration and fitted success probabilities carry no finite-sample guarantee.",
]


def number(value, name, *, nonnegative=True):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    try:
        out = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(name + " must be numeric") from exc
    if not math.isfinite(out) or (nonnegative and out < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return out


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def group(row):
    value = row.get("task_group", row.get("group"))
    if not isinstance(value, str) or not value:
        raise ValueError("Every row needs a nonempty source task_group")
    return value


def trusted_success(row):
    if (row.get("verifier_kind") in TRUSTED_VERIFIERS
            and isinstance(row.get("final_success"), bool)
            and row.get("execution_complete") is True):
        return row["final_success"]
    return None


def known_total(row):
    if row.get("total_usage_known") is True and row.get("execution_complete") is True:
        return number(row.get("total_policy_cost"), "total_policy_cost")
    return None


def validate_rows(rows, policy_id, feature_names):
    names = list(feature_names)
    if not names or len(names) != len(set(names)) or any(n in FORBIDDEN_FEATURES for n in names):
        raise ValueError("Feature schema must be unique and exclude outcomes, reward and split")
    seen = set()
    for row in rows:
        if row.get("source_type") != "real_api":
            raise ValueError("Only explicitly identified real_api observations are accepted")
        if row.get("policy_id") != policy_id:
            raise ValueError("Refusing to mix behavior policies")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every row needs an id")
        if not isinstance(row.get("model"), str) or not row["model"]:
            raise ValueError("Every row needs a model")
        group(row)
        key = (row["model"], row["id"])
        if key in seen:
            raise ValueError("Duplicate model/task: choose one predeclared replicate per task")
        seen.add(key)
        number(row.get("reward"), "reward")
        for name in names:
            number(row.get("features", {}).get(name), "feature:" + name, nonnegative=False)
        known_total(row)
    return rows


def infer_feature_names(rows):
    if not rows:
        raise ValueError("Training rows are empty")
    return sorted(rows[0]["features"])


def fit_success(rows, strength=4.0):
    """Group-balanced domain shrinkage from verified final outcomes only."""
    rows = [r for r in rows if trusted_success(r) is not None]
    if not rows:
        return {"global": 0.5, "domains": {}, "groups": 0, "rows": 0,
                "strength": strength, "verified_only": True}
    counts = Counter(group(r) for r in rows)
    total = math.fsum(1 / counts[group(r)] for r in rows)
    successes = math.fsum(float(trusted_success(r)) / counts[group(r)] for r in rows)
    global_probability = (successes + 1.0) / (total + 2.0)
    cells = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        weight = 1 / counts[group(row)]
        cell = cells[row.get("domain", "")]
        cell[0] += weight * float(trusted_success(row))
        cell[1] += weight
    return {"global": global_probability,
            "domains": {d: (s + strength * global_probability) / (n + strength)
                        for d, (s, n) in cells.items()},
            "groups": len(counts), "rows": len(rows), "strength": strength,
            "verified_only": True}


def success_probability(artifact, domain):
    return max(0.0, min(1.0, artifact["domains"].get(domain, artifact["global"])))


def candidate_grid():
    # Baseline first: exact ties retain it instead of manufacturing a winner.
    candidates = [dict(BASELINE)]
    for exponent, margin, overhead, success in itertools.product(
            (0.0, 1.0), (1.0, 1.25, 1.5), (0.0, 50.0), (0.0, 1.0)):
        candidate = dict(cost_exponent=exponent, cost_multiplier=margin,
                         overhead=overhead, success_exponent=success)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def decision_row(row, forecast, success):
    """An explicit boundary: scheduler input contains no observed labels."""
    if forecast.total_policy_mean is None:
        return None
    return {"id": row["id"], "model": row["model"], "domain": row.get("domain", ""),
            "task_group": group(row), "reward": number(row["reward"], "reward"),
            "predicted_cost": max(0.0, forecast.total_policy_mean),
            "predicted_first": forecast.first_call_mean,
            "empirical_upper": forecast.total_policy_upper,
            "success_probability": success_probability(success, row.get("domain", ""))}


def select_portfolio(decisions, config, budget):
    """Greedy expected-value ordering/admission; never reads actual cost or verdict."""
    budget = number(budget, "budget")
    margin = number(config["cost_multiplier"], "cost_multiplier")
    if margin < 1:
        raise ValueError("Cost multiplier is a reserve margin and must be >= 1")
    exponent = number(config["cost_exponent"], "cost_exponent")
    overhead = number(config["overhead"], "overhead")
    success_exponent = number(config["success_exponent"], "success_exponent")
    scored = []
    for item in decisions:
        cost = number(item["predicted_cost"], "predicted_cost")
        probability = number(item["success_probability"], "success_probability")
        if probability > 1:
            raise ValueError("Probability exceeds one")
        reward = number(item["reward"], "reward")
        expected = reward * probability ** success_exponent
        reserved = max(0.1, cost * margin)
        score = expected / max(1.0, reserved + overhead) ** exponent
        scored.append((score, expected, -reserved, item["id"], reserved))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    used, selected = 0.0, []
    for _, expected, _, ident, reserved in scored:
        if expected > 0 and used + reserved <= budget + 1e-9:
            used += reserved
            selected.append(ident)
    return {"selected_ids": selected, "predicted_reserved_cost": used, "budget": budget,
            "algorithm": "greedy_full_trajectory_portfolio"}


def evaluate_bundle(selection, measurements):
    """Measured totals only. Over-budget bundle has no valid budget-constrained reward."""
    byid = {r["id"]: r for r in measurements}
    if len(byid) != len(measurements):
        raise ValueError("Bundle measurements must have one row per task")
    ids = selection["selected_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate selected task")
    unknown, costs, rewards, solved = [], [], [], 0
    for ident in ids:
        row = byid.get(ident)
        if row is None:
            unknown.append(ident)
            continue
        cost, outcome = known_total(row), trusted_success(row)
        if cost is not None:
            costs.append(cost)
        if outcome is not None:
            rewards.append(number(row["reward"], "reward") * outcome)
            solved += int(outcome)
        if cost is None or outcome is None:
            unknown.append(ident)
    # Summation must not create a spurious winner when only task order differs.
    known_cost, known_reward = math.fsum(costs), math.fsum(rewards)
    complete = not unknown
    within = complete and known_cost <= selection["budget"] + 1e-8
    return {**selection, "selected": len(ids), "complete_observation": complete,
            "unknown_ids": unknown, "measured_full_trace_cost": known_cost,
            "measured_full_trace_reward": known_reward, "measured_solved": solved,
            "within_budget": within,
            "overrun": max(0.0, known_cost - selection["budget"]),
            "budget_valid_reward": known_reward if within else None}


def bundle_objective(bundle):
    """Lexicographic points, then lower cost. Infeasible bundles are disqualified."""
    if not bundle["within_budget"]:
        return (0, 0.0, -math.inf)
    return (1, bundle["budget_valid_reward"], -bundle["measured_full_trace_cost"])


def point_metrics(rows, predictions, target):
    pairs = []
    for row, prediction in zip(rows, predictions):
        if target == "total_policy":
            value, predicted = known_total(row), prediction.total_policy_mean
        else:
            value = (number(row.get("first_call_cost"), "first_call_cost")
                     if row.get("first_usage_known") is True else None)
            predicted = prediction.first_call_mean
        if value is not None and predicted is not None:
            pairs.append((group(row), value, predicted))
    if not pairs:
        return {"n": 0, "groups": 0}
    counts = Counter(g for g, _, _ in pairs)
    weights = [1 / counts[g] for g, _, _ in pairs]
    denom = sum(weights)
    ae = [abs(y - p) for _, y, p in pairs]
    return {"n": len(pairs), "groups": len(counts),
            "row_mae": sum(ae) / len(ae),
            "group_mae": sum(w * a for w, a in zip(weights, ae)) / denom,
            "row_bias": sum(p - y for _, y, p in pairs) / len(pairs),
            "group_bias": sum(w * (p-y) for w, (_, y, p) in zip(weights, pairs)) / denom}


def crossfit(train, feature_names, policy_id, folds=4):
    """Every OOF row is predicted without any variant of its source group."""
    assignment = grouped_fold_ids(train, folds)
    if len(set(assignment.values())) < 2:
        raise ValueError("At least two source groups are required for OOF search")
    decisions, forecasts = {}, {}
    for fold in sorted(set(assignment.values())):
        learning = [r for r in train if assignment[group(r)] != fold]
        validation = [r for r in train if assignment[group(r)] == fold]
        fitted = fit_cost_artifact(learning, feature_names=feature_names,
                                  policy_id=policy_id, criterion="mae", folds=3)
        estimator = CostModel(fitted)
        successes = {m: fit_success([r for r in learning if r["model"] == m])
                     for m in {r["model"] for r in learning}}
        for row in validation:
            forecast = estimator.predict(row["features"], model=row["model"],
                                         policy_id=policy_id, domain=row.get("domain", ""))
            key = (row["model"], row["id"])
            forecasts[key] = forecast
            successes_for_model = successes.get(row["model"], fit_success([]))
            decisions[key] = decision_row(row, forecast, successes_for_model)
    return decisions, forecasts


def fit_offline(train, calibration=(), *, policy_id, feature_names=None,
                primary_budget=100000.0, stress_budgets=(1000.0, 5000.0, 10000.0), folds=4):
    train, calibration = list(train), list(calibration)
    feature_names = list(feature_names or infer_feature_names(train))
    validate_rows(train, policy_id, feature_names)
    validate_rows(calibration, policy_id, feature_names)
    primary_budget = number(primary_budget, "primary_budget")
    if primary_budget != 100000:
        raise ValueError("Primary experiment budget must be 100000 per model; use stress budgets otherwise")
    stress_budgets = sorted(set(number(x, "stress_budget") for x in stress_budgets))
    if any(x <= 0 or x >= primary_budget for x in stress_budgets):
        raise ValueError("Stress budgets must be positive and below the primary budget")
    # This also validates policy labels, known costs and source-group separation.
    cost_artifact = fit_cost_artifact(train, calibration, feature_names=feature_names,
                                     policy_id=policy_id, criterion="mae")
    oof_decisions, oof_forecasts = crossfit(train, feature_names, policy_id, folds)
    configurations = candidate_grid()
    models, reports = {}, {}
    for name in sorted({r["model"] for r in train}):
        all_rows = [r for r in train if r["model"] == name]
        eligible = [r for r in all_rows if known_total(r) is not None
                    and trusted_success(r) is not None]
        if not eligible:
            raise ValueError("No complete verified search trajectories for " + name)
        decisions = [oof_decisions[(name, r["id"])] for r in eligible]
        if any(d is None for d in decisions):
            raise ValueError("OOF model has insufficient total-cost support for " + name)
        candidates = []
        for config in configurations:
            bundle = evaluate_bundle(select_portfolio(decisions, config, primary_budget), eligible)
            candidates.append({"config": config, "bundle": bundle})
        winner = max(range(len(candidates)), key=lambda i: bundle_objective(candidates[i]["bundle"]))
        chosen = configurations[winner]
        successes = fit_success(all_rows)
        models[name] = {"scheduler": chosen, "success": successes,
                        "training_rows": len(all_rows), "search_rows": len(eligible),
                        "search_groups": len({group(r) for r in eligible})}
        unique_bundles = {tuple(sorted(item["bundle"]["selected_ids"])) for item in candidates}
        forecasts = [oof_forecasts[(name, r["id"])] for r in all_rows]
        reports[name] = {"primary_budget": primary_budget, "selected_candidate": winner,
            "primary_search_informative": len(unique_bundles) > 1,
            "distinct_selected_bundles": len(unique_bundles),
            "candidate_count": len(candidates), "training_rows": len(all_rows),
            "search_rows": len(eligible), "excluded_unverified_or_censored": len(all_rows)-len(eligible),
            "oof_cost": {t: point_metrics(all_rows, forecasts, t)
                         for t in ("first_call", "total_policy")},
            "candidates": candidates,
            "secondary_stress_not_used_for_selection": [
                {"budget": budget,
                 "baseline": evaluate_bundle(select_portfolio(decisions, BASELINE, budget), eligible),
                 "selected_primary": evaluate_bundle(select_portfolio(decisions, chosen, budget), eligible)}
                for budget in stress_budgets]}
    artifact = {"schema_version": SCHEMA_VERSION, "policy_id": policy_id,
        "feature_names": feature_names, "primary_budget_per_model": primary_budget,
        "stress_budgets": stress_budgets, "cost_artifact": cost_artifact, "models": models,
        "train_groups": sorted({group(r) for r in train}),
        "calibration_groups": sorted({group(r) for r in calibration}),
        "train_rows_sha256": digest(train), "calibration_rows_sha256": digest(calibration),
        "criterion": "lexicographic_budget_feasible_points_then_lower_cost",
        "cost_cv_criterion": "mae", "test_labels_read": False,
        "limitations": list(LIMITATIONS)}
    report = {"scope": "exploratory OOF training search; not heldout evidence",
        "models": reports, "calibration_point_diagnostics": evaluate_cost_model(
            CostModel(cost_artifact), calibration, require_unseen=False),
        "artifact_sha256": digest(artifact), "limitations": list(LIMITATIONS)}
    return artifact, report


def evaluate_offline(artifact, test):
    """Evaluate frozen search once on new source groups; no fitting or selection."""
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported offline artifact schema")
    test = validate_rows(list(test), artifact["policy_id"], artifact["feature_names"])
    used = set(artifact["train_groups"]) | set(artifact["calibration_groups"])
    if used.intersection(group(r) for r in test):
        raise ValueError("Heldout source groups overlap train/calibration")
    unknown_models = {r["model"] for r in test} - set(artifact["models"])
    if unknown_models:
        raise ValueError("Untrained heldout models: " + ",".join(sorted(unknown_models)))
    cost = CostModel(artifact["cost_artifact"])
    reports = {}
    for model, data in artifact["models"].items():
        measured = [r for r in test if r["model"] == model]
        if not measured:
            continue
        predictions = [cost.predict(r["features"], model=model, policy_id=artifact["policy_id"],
                                    domain=r.get("domain", "")) for r in measured]
        decisions = [decision_row(r, p, data["success"]) for r, p in zip(measured, predictions)]
        if any(d is None for d in decisions):
            raise ValueError("Frozen estimator cannot predict a heldout task")
        # Decisions are built from public inputs only before labels are scored.
        budget = artifact["primary_budget_per_model"]
        plans = {"baseline": select_portfolio(decisions, BASELINE, budget),
                 "selected_primary": select_portfolio(decisions, data["scheduler"], budget)}
        prediction_hash = digest(decisions)
        reports[model] = {"n": len(measured), "groups": len({group(r) for r in measured}),
            "predictions_sha256": prediction_hash,
            "first_call": point_metrics(measured, predictions, "first_call"),
            "total_policy": point_metrics(measured, predictions, "total_policy"),
            "primary": {k: evaluate_bundle(v, measured) for k, v in plans.items()},
            "secondary_stress_not_used_for_selection": [
                {"budget": b,
                 "baseline": evaluate_bundle(select_portfolio(decisions, BASELINE, b), measured),
                 "selected_primary": evaluate_bundle(select_portfolio(decisions, data["scheduler"], b), measured)}
                for b in artifact["stress_budgets"]]}
    return {"scope": "heldout full-trajectory portfolio diagnostic",
            "artifact_sha256": digest(artifact), "test_rows_sha256": digest(test),
            "models": reports, "limitations": list(LIMITATIONS)}


def export_legacy_train(rows, policy_id):
    """Historical utility run only: explicit policy, train split, primary replicate.

    Legacy 'correct' includes correct abstentions. They earn NO contest points,
    so final_success = answerable AND correct; answerability never is a feature.
    """
    output = []
    completed_states = {"answered", "abstained", "sql_error", "calculation_error",
                        "empty_output", "public_test_failed"}
    for old in rows:
        if (old.get("split") != "train" or old.get("replicate") != 0
                or old.get("policy") != policy_id):
            continue
        calls = old.get("calls", [])
        if not calls:
            continue
        known = all(c.get("accounting") == "provider_reconciled" for c in calls)
        complete = old.get("state") in completed_states and known
        features = {k: float(v) for k, v in old.get("features", {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and k not in FORBIDDEN_FEATURES}
        domain = old["domain"]
        features.update({f"kind_{d}": float(domain == d)
                         for d in ("math", "search", "coding", "data_analysis")})
        total = sum(number(c.get("cost", 0), "cost") for c in calls)
        if known and abs(total - number(old["actual_tokens"], "actual_tokens")) > 0.11:
            raise ValueError("Legacy call totals disagree with actual_tokens")
        output.append({"source_type": "real_api", "id": old["id"], "task_group": group(old),
            "model": old["model"], "policy_id": policy_id, "domain": domain,
            "features": features, "reward": old["reward"],
            "first_call_cost": calls[0].get("cost"),
            "first_usage_known": calls[0].get("accounting") == "provider_reconciled",
            "total_policy_cost": total, "total_usage_known": known,
            "execution_complete": complete,
            "final_success": bool(old.get("answerable") and old.get("correct")) if complete else None,
            "verifier_kind": ("hidden_tests" if domain == "coding" else "oracle"),
            "historical_train_only": True, "original_split": old["split"],
            "original_key": old.get("key")})
    if not output:
        raise ValueError("No train/replicate=0 rows for the requested historical policy")
    return output


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else value["rows"]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    exp = sub.add_parser("export-legacy")
    exp.add_argument("--input", required=True); exp.add_argument("--policy-id", required=True)
    exp.add_argument("--out", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--train", required=True); fit.add_argument("--calibration")
    fit.add_argument("--policy-id", required=True); fit.add_argument("--out-dir", required=True)
    fit.add_argument("--stress-budgets", default="1000,5000,10000")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--artifact", required=True); evaluate.add_argument("--test", required=True)
    evaluate.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.action == "export-legacy":
        rows = export_legacy_train(load_rows(args.input), args.policy_id)
        write_new(args.out, rows)
        print(json.dumps({"rows": len(rows), "scope": "historical train only", "api_calls": 0}))
    elif args.action == "fit":
        out = Path(args.out_dir)
        if (out / "offline_artifact.json").exists() or (out / "search_report.json").exists():
            raise FileExistsError("Refusing to overwrite a frozen offline experiment")
        artifact, report = fit_offline(load_rows(args.train),
            load_rows(args.calibration) if args.calibration else (), policy_id=args.policy_id,
            stress_budgets=[float(x) for x in args.stress_budgets.split(",") if x])
        write_new(out / "offline_artifact.json", artifact)
        write_new(out / "search_report.json", report)
        print(json.dumps({"models": {m: {"candidate": r["selected_candidate"],
            "informative": r["primary_search_informative"], "search_rows": r["search_rows"],
            "oof_cost": r["oof_cost"]} for m, r in report["models"].items()}, "api_calls": 0}))
    else:
        artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8-sig"))
        report = evaluate_offline(artifact, load_rows(args.test))
        write_new(args.out, report)
        print(json.dumps({"models": list(report["models"]), "scope": report["scope"], "api_calls": 0}))


if __name__ == "__main__":
    main()

