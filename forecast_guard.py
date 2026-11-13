"""Choose a deployable point estimate without changing the raw cost forecast.

This is a prediction trust guard, not a task execution cap or correctness test.
It makes no model/API calls. Keep the raw forecast alongside this decision when
evaluating prediction accuracy. Empirical interval calibration is not required
to use otherwise valid point estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class ForecastChoice:
    first_call: float
    total_policy: float
    source: str
    reason: str | None


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " must be finite and positive")
    return result


def _field(forecast, name, default=None):
    if isinstance(forecast, Mapping):
        return forecast.get(name, default)
    return getattr(forecast, name, default)


def choose_forecast(forecast, prior_first, prior_total, *, pipeline_budget=100000):
    """Use learned points only when their scope and numeric values are usable.

    Reasons for falling back: missing/unsupported scope, any model warning,
    missing/nonfinite/nonpositive heads, total below first, or predicted total
    exceeding the whole pipeline budget. No values are clipped to that budget.
    A valid prior may itself exceed it; normal shared-budget admission decides
    whether such a task can be attempted.

    Both prior heads are validated even when the learned forecast is accepted,
    so an invalid fallback cannot remain latent until deployment.
    """
    first = _positive(prior_first, "prior_first")
    total = _positive(prior_total, "prior_total")
    budget = _positive(pipeline_budget, "pipeline_budget")
    if total < first:
        raise ValueError("prior_total must be >= prior_first")

    def fallback(reason):
        return ForecastChoice(first, total, "prior", reason)

    if forecast is None:
        return fallback("forecast_missing")
    failure = _field(forecast, "fallback_reason")
    if failure:
        return fallback(str(failure))
    if _field(forecast, "warnings", ()):
        return fallback("forecast_warning")
    try:
        predicted_first = _positive(_field(forecast, "first_call_mean"), "first_call_mean")
        predicted_total = _positive(_field(forecast, "total_policy_mean"), "total_policy_mean")
    except ValueError:
        return fallback("invalid_forecast_points")
    if predicted_total < predicted_first:
        return fallback("inconsistent_forecast_points")
    if predicted_total > budget:
        return fallback("forecast_exceeds_pipeline_budget")
    return ForecastChoice(predicted_first, predicted_total, "learned", None)

