"""Stdlib inference for the frozen residual-cost artifact; no fit or API calls.

Matches experiments/residual_forecaster.py v1 point predictions and fallbacks.
No extra OOD, budget or clipping rule is imposed by this inference adapter.
These are robust MAE point estimates, not calibrated expected costs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("first_call", "total_policy")
METHODS = {"prior", "multiplier", "additive", "input_remaining"}


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(name + " must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _domain(features):
    kinds = [kind for kind in ("math", "coding", "search", "data_analysis")
             if _number(features.get("kind_" + kind, 0), "kind", nonnegative=True) == 1]
    if len(kinds) != 1:
        raise ValueError("Expected exactly one domain indicator")
    return kinds[0]


def _prompt_input(features):
    return (_number(features["instruction_chars"], "instruction_chars", nonnegative=True) +
            _number(features["context_chars"], "context_chars", nonnegative=True)) / 35 + 18


def _prompt_prior(features, model):
    prompt = _prompt_input(features)
    kind = _domain(features)
    if kind == "coding":
        output = 85 + min(1400, _number(features.get("source_bytes", 0), "source_bytes", nonnegative=True)/6)
        retry = 0 if features.get("contradiction_flag") else .2
    elif kind == "data_analysis":
        output, retry = 40, .12
    elif kind == "search":
        output, retry = 8, .05
    else:
        output, retry = 24, .06
    first = prompt + output * (1 if model.endswith("-noreason") else 5)
    return first, first + retry * (prompt + 650)


def _raw_point(head, prior, input_cost, domain):
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


@dataclass(frozen=True)
class ResidualPrediction:
    first_call: float
    total_policy: float
    source: str
    reason: str | None


class ResidualModel:
    def __init__(self, artifact):
        self.artifact = json.loads(json.dumps(artifact, allow_nan=False))
        if self.artifact.get("schema_version") != 1:
            raise ValueError("Unknown residual artifact")
        self.policy_id = self.artifact["policy_id"]
        for scope in self.artifact["models"].values():
            for head in scope["heads"].values():
                if head["method"] not in METHODS:
                    raise ValueError("Unknown residual method")
                if head["method"] != "prior":
                    values = [head["global"]] + [x["value"] for x in head["domains"].values()]
                    for value in values:
                        _number(value, "coefficient", nonnegative=head["method"] != "additive")

    def predict(self, features, *, model, policy_id=None, domain=None, prior=None):
        prior = _prompt_prior(features, model) if prior is None else tuple(prior)
        if len(prior) != 2 or any(_number(v, "prior", nonnegative=True) <= 0 for v in prior) or prior[1] < prior[0]:
            raise ValueError("Invalid prompt prior")
        domain = domain or _domain(features)
        if policy_id is not None and policy_id != self.policy_id:
            return ResidualPrediction(*prior, "prior", "policy_mismatch")
        scope = self.artifact["models"].get(model)
        if scope is None:
            return ResidualPrediction(*prior, "prior", "model_not_trained")
        try:
            input_cost = _prompt_input(features)
            raw = [_raw_point(scope["heads"][target], prior[i], input_cost, domain)
                   for i, target in enumerate(TARGETS)]
            first = max(1.0, _number(raw[0], "first point"))
            total = max(first, 1.0, _number(raw[1], "total point"))
        except (ValueError, OverflowError):
            return ResidualPrediction(*prior, "prior", "invalid_numeric_prediction")
        return ResidualPrediction(first, total, "residual", None)


def load_residual_model(path, expected_policy_id):
    """Load the frozen JSON artifact and reject a mismatched policy scope."""
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Residual artifact is unexpectedly large")
    model = ResidualModel(json.loads(path.read_text(encoding="utf-8-sig")))
    if model.policy_id != expected_policy_id:
        raise ValueError("Loaded residual artifact has a different policy_id")
    return model
