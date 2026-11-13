"""CPU-only pre-run cost models; training and inference require only stdlib.

Rows contain task_group (or group), model, policy_id, domain, features, and:
  first_call_cost + first_usage_known=True
  total_policy_cost + total_usage_known=True + execution_complete=True
Unknown labels are excluded independently. A normal cap stop is a completed
policy; an interrupted/abandoned policy is not a total-policy cost label.

Model selection uses grouped CV on train only. Calibration groups must be
disjoint. Empirical upper estimates are diagnostics, NOT budget guarantees.
No per-task execution cap, model API, hidden verifier, or filesystem feature
extraction is implemented here. The caller owns its frozen feature extractor.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

SCHEMA_VERSION = 1
TARGETS = ("first_call", "total_policy")
MAX_FEATURES = 128
DEFAULT_CANDIDATES = (
    ("domain_mean", 2.0), ("domain_mean", 8.0),
    ("ridge", 1.0), ("ridge", 10.0), ("ridge", 100.0),
    ("log_ridge", 1.0), ("log_ridge", 10.0), ("log_ridge", 100.0),
)


@dataclass(frozen=True)
class CostForecast:
    model: str
    policy_id: str
    first_call_mean: float | None = None
    total_policy_mean: float | None = None
    first_call_upper: float | None = None
    total_policy_upper: float | None = None
    support: dict = field(default_factory=dict)
    calibrated: bool = False
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()
    continuation_probability: float | None = None
    extra_cost_if_continued: float | None = None
    hurdle_total_mean: float | None = None

    def as_dict(self):
        return asdict(self)


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _features(names):
    if isinstance(names, str):
        raise ValueError("feature_names must be a sequence of names")
    names = list(names)
    if (not names or len(names) > MAX_FEATURES or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x for x in names)):
        raise ValueError("feature_names must contain 1..128 unique nonempty strings")
    return names


def _group(row):
    value = row.get("task_group", row.get("group"))
    if not isinstance(value, str) or not value:
        raise ValueError("Each row requires a nonempty source task_group")
    return value


def _domain(row):
    value = row.get("domain", "")
    if not isinstance(value, str):
        raise ValueError("domain must be a string")
    return value


def _vector(features, names):
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    return [_number(features[name], f"feature:{name}") for name in names]


def _target(row, target):
    if target == "first_call":
        if row.get("first_usage_known") is not True:
            return None
        return _number(row.get("first_call_cost"), "first_call_cost", nonnegative=True)
    if row.get("total_usage_known") is not True or row.get("execution_complete") is not True:
        return None
    return _number(row.get("total_policy_cost"), "total_policy_cost", nonnegative=True)


def _validate_rows(rows, names, policy_id):
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every observation must be a mapping")
        _group(row)
        _domain(row)
        if not isinstance(row.get("model"), str) or not row["model"]:
            raise ValueError("Every observation requires model")
        if row.get("policy_id") != policy_id:
            raise ValueError("Training/calibration policy_id mismatch")
        _vector(row.get("features"), names)
        first, total = (_target(row, name) for name in TARGETS)
        if first is not None and total is not None and total + 1e-8 < first:
            raise ValueError("total_policy_cost cannot be smaller than first_call_cost")
        result.append(dict(row))
    return result


def _weights(rows):
    counts = Counter(_group(row) for row in rows)
    return [1.0 / counts[_group(row)] for row in rows]


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _solve(matrix, rhs):
    """Partial-pivot Gaussian elimination, sufficient for tiny ridge systems."""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(aug[i][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular ridge system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            ratio = aug[row][col] / aug[col][col]
            for k in range(col + 1, n + 1):
                aug[row][k] -= ratio * aug[col][k]
            aug[row][col] = 0.0
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = (aug[row][n] - math.fsum(
            aug[row][k] * solution[k] for k in range(row + 1, n))) / aug[row][row]
    return solution


def _domain_fit(rows, values, strength):
    weights = _weights(rows)
    total_weight = math.fsum(weights)
    global_mean = _dot(values, weights) / total_weight
    amounts, counts = defaultdict(float), defaultdict(float)
    for row, y, weight in zip(rows, values, weights):
        domain = _domain(row)
        amounts[domain] += weight * y
        counts[domain] += weight
    return {"global": global_mean,
            "domains": {d: (amounts[d] + strength * global_mean) / (counts[d] + strength)
                        for d in counts},
            "strength": strength}


def _fit(rows, values, names, method, regularization):
    if method == "domain_mean":
        return {"method": method, **_domain_fit(rows, values, regularization)}
    weights = _weights(rows)
    raw = [_vector(row["features"], names) for row in rows]
    weight_sum = math.fsum(weights)
    means = [math.fsum(w * x[j] for x, w in zip(raw, weights)) / weight_sum
             for j in range(len(names))]
    scales = [math.sqrt(max(0.0, math.fsum(
        w * (x[j] - means[j]) ** 2 for x, w in zip(raw, weights)) / weight_sum))
        for j in range(len(names))]
    scales = [s if s > 1e-8 else 1.0 for s in scales]
    design = [[1.0] + [(v - m) / s for v, m, s in zip(x, means, scales)] for x in raw]
    target = [math.log1p(y) for y in values] if method == "log_ridge" else values
    width = len(names) + 1
    matrix = [[math.fsum(w * x[i] * x[j] for x, w in zip(design, weights))
               for j in range(width)] for i in range(width)]
    for j in range(1, width):
        matrix[j][j] += regularization
    rhs = [math.fsum(w * x[j] * y for x, y, w in zip(design, target, weights))
           for j in range(width)]
    coefficients = _solve(matrix, rhs)
    result = {"method": method, "regularization": regularization,
              "mean": means, "scale": scales, "coefficients": coefficients}
    if method == "log_ridge":
        # Do not floor the latent log prediction. Retransformation targets a
        # mean estimate, but global/domain smearing is not a correctness proof
        # under heteroscedasticity or distribution shift.
        smear = [math.exp(y - _dot(x, coefficients)) for x, y in zip(design, target)]
        result["smearing"] = _domain_fit(rows, smear, 8.0)
    return result


def _predict_fitted(fitted, features, domain, names):
    method = fitted["method"]
    if method == "domain_mean":
        return fitted["domains"].get(domain, fitted["global"])
    raw = _vector(features, names)
    vector = [1.0] + [(v - m) / s for v, m, s in zip(raw, fitted["mean"], fitted["scale"])]
    latent = _dot(vector, fitted["coefficients"])
    if method == "log_ridge":
        smear = fitted["smearing"]
        factor = smear["domains"].get(domain, smear["global"])
        value = math.exp(latent) * factor - 1.0
    else:
        value = latent
    if not math.isfinite(value):
        raise ValueError("Non-finite cost prediction")
    return max(0.0, value)


def _metrics(rows, actual, predicted):
    if not rows:
        return {"n": 0, "groups": 0}
    weights = _weights(rows)
    denominator = math.fsum(weights)
    mae = math.fsum(w * abs(y - p) for y, p, w in zip(actual, predicted, weights)) / denominator
    mse = math.fsum(w * (y - p) ** 2 for y, p, w in zip(actual, predicted, weights)) / denominator
    actual_mean = _dot(actual, weights) / denominator
    predicted_mean = _dot(predicted, weights) / denominator
    return {"n": len(rows), "groups": len(set(map(_group, rows))), "mae": mae,
            "mse": mse, "wape": mae / actual_mean if actual_mean > 0 else None,
            "bias": predicted_mean - actual_mean, "actual_mean": actual_mean,
            "predicted_mean": predicted_mean}


def grouped_fold_ids(rows, folds=5):
    groups = sorted(set(map(_group, rows)),
                    key=lambda g: hashlib.sha256(("cost-cv-v1|" + g).encode()).hexdigest())
    count = min(folds, len(groups))
    return {group: index % count for index, group in enumerate(groups)} if count else {}


def _select_head(rows, target, names, criterion, folds, min_cv_groups, candidates):
    eligible = [row for row in rows if _target(row, target) is not None]
    if not eligible:
        return None
    group_count = len(set(map(_group, eligible)))
    reports = []
    if group_count >= min_cv_groups:
        assignments = grouped_fold_ids(eligible, folds)
        for method, regularization in candidates:
            evaluation_rows, actual, predicted = [], [], []
            for fold in sorted(set(assignments.values())):
                train = [r for r in eligible if assignments[_group(r)] != fold]
                held = [r for r in eligible if assignments[_group(r)] == fold]
                fitted = _fit(train, [_target(r, target) for r in train],
                              names, method, regularization)
                for row in held:
                    evaluation_rows.append(row)
                    actual.append(_target(row, target))
                    predicted.append(_predict_fitted(fitted, row["features"], _domain(row), names))
            reports.append({"method": method, "regularization": regularization,
                            **_metrics(evaluation_rows, actual, predicted)})
        best = min(reports, key=lambda report: (report[criterion],
                   0 if report["method"] == "domain_mean" else 1))
        method, regularization = best["method"], best["regularization"]
    else:
        method, regularization = "domain_mean", 8.0
    fitted = _fit(eligible, [_target(r, target) for r in eligible],
                  names, method, regularization)
    return {"fitted": fitted, "target": target, "n": len(eligible), "groups": group_count,
            "cv_criterion": criterion, "cv": reports,
            "selection_note": "grouped_cv_train_only" if reports else "small_sample_domain_shrinkage",
            "calibration": None}


def _points(scope, features, domain, names):
    points = {}
    for target in TARGETS:
        head = scope["heads"].get(target)
        points[target] = (None if head is None else
                          _predict_fitted(head["fitted"], features, domain, names))
    if points["first_call"] is not None and points["total_policy"] is not None:
        points["total_policy"] = max(points["first_call"], points["total_policy"])
    return points


def _hurdle_fit(rows):
    eligible = [r for r in rows if _target(r, "first_call") is not None
                and _target(r, "total_policy") is not None]
    if not eligible:
        return None
    extra = [max(0.0, _target(r, "total_policy") - _target(r, "first_call")) for r in eligible]
    positive = [(r, y) for r, y in zip(eligible, extra) if y > 1e-8]
    return {"event": _domain_fit(eligible, [float(y > 1e-8) for y in extra], 8.0),
            "positive_cost": (_domain_fit([r for r, _ in positive],
                              [y for _, y in positive], 8.0) if positive else None),
            "n": len(eligible), "groups": len(set(map(_group, eligible))),
            "positive_groups": len({_group(r) for r, _ in positive}),
            "note": "Extra-call continuation under the recorded policy; not arbitrary retry cost."}


def fit_cost_artifact(train_rows, calibration_rows=(), *, feature_names, policy_id,
                      criterion="mse", folds=5, min_cv_groups=8,
                      min_calibration_groups=8, upper_quantile=.9,
                      candidates=DEFAULT_CANDIDATES,
                      feature_schema_id="explicit-prerun-features-v1"):
    """Fit separate heads within each exact model+policy, returning plain JSON data.

    criterion='mse' targets mean-oriented selection; criterion='mae' favors
    absolute error. Neither choice promises that a fitted 'mean' is calibrated.
    No hyperparameters are selected using calibration or final heldout labels.
    """
    names = _features(feature_names)
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id must be nonempty")
    if criterion not in {"mae", "mse"}:
        raise ValueError("criterion must be mae or mse")
    if folds < 2 or min_cv_groups < 2 or min_calibration_groups < 1:
        raise ValueError("Invalid fold or support requirement")
    if not 0 < upper_quantile < 1:
        raise ValueError("upper_quantile must be between zero and one")
    candidates = [(str(method), _number(reg, "regularization", nonnegative=True))
                  for method, reg in candidates]
    if not candidates or any(m not in {"domain_mean", "ridge", "log_ridge"} or r <= 0
                             for m, r in candidates):
        raise ValueError("Candidates require a supported method and positive regularization")
    train = _validate_rows(list(train_rows), names, policy_id)
    calibration = _validate_rows(list(calibration_rows), names, policy_id)
    if not train:
        raise ValueError("No training rows")
    train_groups = set(map(_group, train))
    calibration_groups = set(map(_group, calibration))
    if train_groups & calibration_groups:
        raise ValueError("Train and calibration source groups overlap")
    models = sorted({r["model"] for r in train})
    if {r["model"] for r in calibration} - set(models):
        raise ValueError("Calibration model has no training scope")
    artifact = {"schema_version": SCHEMA_VERSION, "policy_id": policy_id,
                "feature_schema_id": feature_schema_id, "feature_names": names,
                "criterion": criterion, "upper_quantile": upper_quantile,
                "min_calibration_groups": min_calibration_groups,
                "train_groups": sorted(train_groups),
                "calibration_groups": sorted(calibration_groups),
                "scope": "Pre-run features; model selection uses train groups only.",
                "uncertainty": "Empirical relative residuals; no coverage or budget guarantee.",
                "models": {}}
    for model in models:
        subset = [row for row in train if row["model"] == model]
        raw = [_vector(row["features"], names) for row in subset]
        scope = {"heads": {target: _select_head(subset, target, names, criterion, folds,
                      min_cv_groups, candidates) for target in TARGETS},
                 "feature_min": [min(x[j] for x in raw) for j in range(len(names))],
                 "feature_max": [max(x[j] for x in raw) for j in range(len(names))],
                 "domains": sorted({_domain(row) for row in subset}),
                 "hurdle": _hurdle_fit(subset)}
        for target in TARGETS:
            head = scope["heads"][target]
            if head is None:
                continue
            residuals = {}
            n = 0
            for row in calibration:
                if row["model"] != model or _target(row, target) is None:
                    continue
                prediction = _points(scope, row["features"], _domain(row), names)[target]
                score = max(0.0, (_target(row, target) - prediction) / max(1.0, prediction))
                group = _group(row)
                residuals[group] = max(residuals.get(group, 0.0), score)
                n += 1
            sorted_scores = sorted(residuals.values())
            correction = (sorted_scores[max(0, math.ceil(upper_quantile * len(sorted_scores)) - 1)]
                          if len(sorted_scores) >= min_calibration_groups else None)
            head["calibration"] = {"n": n, "groups": len(sorted_scores),
                                   "upper_ratio": correction, "quantile": upper_quantile,
                                   "method": "empirical_group_max_relative_residual",
                                   "guarantee": False}
        artifact["models"][model] = scope
    # Validate serialization and the inference contract before handing it out.
    CostModel(artifact)
    return artifact


def _validate_fitted(fitted, width):
    method = fitted.get("method")
    def check_domain(data, positive=False):
        if not isinstance(data.get("domains"), dict):
            raise ValueError("Invalid domain model")
        values = [data.get("global")] + list(data["domains"].values())
        for value in values:
            number = _number(value, "domain value", nonnegative=True)
            if positive and number <= 0:
                raise ValueError("Smearing factor must be positive")
    if method == "domain_mean":
        check_domain(fitted)
    elif method in {"ridge", "log_ridge"}:
        for key, size in (("mean", width), ("scale", width), ("coefficients", width + 1)):
            values = fitted.get(key)
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"Invalid {key} vector")
            for value in values:
                number = _number(value, key)
                if key == "scale" and number <= 0:
                    raise ValueError("Feature scale must be positive")
        if method == "log_ridge":
            check_domain(fitted.get("smearing", {}), positive=True)
    else:
        raise ValueError("Unsupported artifact model method")


class CostModel:
    def __init__(self, artifact):
        # JSON roundtrip rejects NaN and prevents mutation by the caller.
        self.artifact = json.loads(json.dumps(artifact, allow_nan=False))
        data = self.artifact
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported cost artifact schema")
        self.feature_names = _features(data.get("feature_names", []))
        self.policy_id = data.get("policy_id")
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("Invalid artifact policy")
        if not isinstance(data.get("models"), dict):
            raise ValueError("Invalid model scopes")
        for scope in data["models"].values():
            for bound in ("feature_min", "feature_max"):
                if len(scope.get(bound, [])) != len(self.feature_names):
                    raise ValueError("Invalid feature bounds")
                for value in scope[bound]:
                    _number(value, bound)
            if any(lo > hi for lo, hi in zip(scope["feature_min"], scope["feature_max"])):
                raise ValueError("Reversed feature bounds")
            for target in TARGETS:
                head = scope.get("heads", {}).get(target)
                if head is None:
                    continue
                _validate_fitted(head["fitted"], len(self.feature_names))
                cal = head.get("calibration")
                if cal and cal.get("upper_ratio") is not None:
                    _number(cal["upper_ratio"], "upper_ratio", nonnegative=True)
            hurdle = scope.get("hurdle")
            if hurdle:
                _validate_fitted({"method": "domain_mean", **hurdle["event"]}, 0)
                if hurdle.get("positive_cost") is not None:
                    _validate_fitted({"method": "domain_mean", **hurdle["positive_cost"]}, 0)

    def predict(self, features, *, model, policy_id, domain=None):
        empty = dict(model=model, policy_id=policy_id)
        if policy_id != self.policy_id:
            return CostForecast(**empty, fallback_reason="policy_mismatch")
        scope = self.artifact["models"].get(model)
        if scope is None:
            return CostForecast(**empty, fallback_reason="model_not_trained")
        domain = "" if domain is None else domain
        try:
            raw = _vector(features, self.feature_names)
            points = _points(scope, features, domain, self.feature_names)
        except (ValueError, OverflowError, TypeError) as exc:
            return CostForecast(**empty, fallback_reason=f"invalid_prediction:{exc}")
        warnings = []
        outside = [name for name, value, lo, hi in zip(self.feature_names, raw,
                   scope["feature_min"], scope["feature_max"]) if value < lo - 1e-8 or value > hi + 1e-8]
        if outside:
            warnings.append("feature_extrapolation:" + ",".join(outside))
        if domain not in scope["domains"]:
            warnings.append("unseen_domain")
        uppers, support = {}, {}
        for target in TARGETS:
            head = scope["heads"].get(target)
            support[target + "_groups"] = 0 if head is None else head["groups"]
            support[target + "_rows"] = 0 if head is None else head["n"]
            cal = {} if head is None else (head.get("calibration") or {})
            support[target + "_calibration_groups"] = cal.get("groups", 0)
            correction = cal.get("upper_ratio")
            uppers[target] = (None if points[target] is None or correction is None else
                              points[target] + correction * max(1.0, points[target]))
        if uppers["total_policy"] is not None and uppers["first_call"] is not None:
            uppers["total_policy"] = max(uppers["total_policy"], uppers["first_call"])
        probability = conditional_extra = hurdle_total = None
        hurdle = scope.get("hurdle")
        if hurdle:
            event = hurdle["event"]
            probability = max(0.0, min(1.0, event["domains"].get(domain, event["global"])))
            positive = hurdle.get("positive_cost")
            conditional_extra = (None if positive is None else
                                 positive["domains"].get(domain, positive["global"]))
            support["continuation_groups"] = hurdle["groups"]
            support["positive_extra_groups"] = hurdle["positive_groups"]
            if points["first_call"] is not None:
                if conditional_extra is not None:
                    hurdle_total = points["first_call"] + probability * conditional_extra
                elif probability == 0:
                    hurdle_total = points["first_call"]
        reason = "no_known_cost_labels" if all(v is None for v in points.values()) else None
        return CostForecast(**empty, first_call_mean=points["first_call"],
             total_policy_mean=points["total_policy"], first_call_upper=uppers["first_call"],
             total_policy_upper=uppers["total_policy"], support=support,
             calibrated=all(v is not None for v in uppers.values()) and not warnings,
             fallback_reason=reason, warnings=tuple(warnings),
             continuation_probability=probability,
             extra_cost_if_continued=conditional_extra, hurdle_total_mean=hurdle_total)


def load_cost_model(path, *, expected_policy_id=None, expected_feature_schema_id=None):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Cost artifact is unexpectedly large")
    model = CostModel(json.loads(path.read_text(encoding="utf-8-sig")))
    if expected_policy_id is not None and model.policy_id != expected_policy_id:
        raise ValueError("Loaded cost artifact has a different policy_id")
    if (expected_feature_schema_id is not None and
            model.artifact.get("feature_schema_id") != expected_feature_schema_id):
        raise ValueError("Loaded cost artifact has a different feature schema")
    return model


def save_cost_artifact(artifact, path):
    validated = CostModel(artifact).artifact
    Path(path).write_text(json.dumps(validated, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")


def assert_unseen_groups(rows, artifact):
    rows = list(rows)
    used = set(artifact["train_groups"]) | set(artifact["calibration_groups"])
    if used & {_group(row) for row in rows}:
        raise ValueError("Evaluation groups overlap fitting/calibration groups")


def evaluate_cost_model(model, rows, *, require_unseen=True):
    rows = _validate_rows(list(rows), model.feature_names, model.policy_id)
    if require_unseen:
        assert_unseen_groups(rows, model.artifact)
    report = {}
    for name in sorted({r["model"] for r in rows}):
        report[name] = {}
        for target in TARGETS:
            used, actual, predicted, covered = [], [], [], []
            for row in rows:
                if row["model"] != name or _target(row, target) is None:
                    continue
                forecast = model.predict(row["features"], model=name,
                                         policy_id=model.policy_id, domain=_domain(row))
                point = getattr(forecast, target + "_mean")
                if point is None:
                    continue
                used.append(row)
                actual.append(_target(row, target))
                predicted.append(point)
                upper = getattr(forecast, target + "_upper")
                if upper is not None:
                    covered.append(_target(row, target) <= upper)
            report[name][target] = {**_metrics(used, actual, predicted),
                "upper_coverage_n": len(covered),
                "upper_coverage": sum(covered) / len(covered) if covered else None}
    return report


def retry_marginal_utility(*, reward, success_gain, extra_cost,
                           alternative_reward_per_token):
    """Pure decision aid: E[extra reward] minus opportunity cost.

    success_gain must come from justified conditional evidence; None returns
    None. This function does not estimate correctness from self-confidence,
    enforce a per-task cap, reserve tokens, or make an API call.
    """
    if success_gain is None or extra_cost is None:
        return None
    gain = _number(success_gain, "success_gain", nonnegative=True)
    if gain > 1:
        raise ValueError("success_gain must be a probability")
    return (_number(reward, "reward", nonnegative=True) * gain -
            _number(alternative_reward_per_token, "alternative_reward_per_token",
                    nonnegative=True) * _number(extra_cost, "extra_cost", nonnegative=True))

