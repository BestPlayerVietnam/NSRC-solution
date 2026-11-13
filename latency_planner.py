"""CPU-only latency forecasting and joint token/deadline allocation.

The initial request priors are rounded robust fits to 989 completed public100
calls observed before final-release validation. They model HTTP response latency,
not TTFT. Upper multipliers are empirical buffers, not coverage guarantees.
Tool priors include planning/disk/SDK gaps and must not be called pure CPU time.
No prediction imposes an individual task time quota: admission compares with the
remaining global deadline. Token-only mode preserves the existing queue scores.
SLA-v3 allocates expected reward using point durations; upper estimates remain
diagnostics. The runtime enforces the actual shared deadline independently.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Mapping


DOMAINS = frozenset({'math', 'coding', 'search', 'data_analysis'})
MAX_HISTORY = 32
MAX_OBSERVATION_SECONDS = 3600.0
PRIOR_PROVENANCE = {
    'source': 'public100_completed_known_usage_before_final_release',
    'calls': 989,
    'calls_sha256': '7ab4ea7dd0dc13db7ef8c1d5d864ac456e7f3af6657be13f0688cd4e41a3c883',
    'method': 'Rounded robust intercept+slope fits; empirical actual/prediction p90 buffers.',
    'limitations': 'Observed API latency depends on load; tool priors include inter-request gaps. '
                   'No final-release labels or timings were used. No coverage guarantee.',
}


@dataclass(frozen=True)
class LatencyPrior:
    request_overhead_seconds: float
    seconds_per_weighted_token: float
    upper_multiplier: float
    source: str = 'public100_rounded_prior'


DEFAULT_PRIORS = {
    'qwen3.6-35b-a3b-noreason': LatencyPrior(0.5, 0.0122, 3.2),
    'qwen3.8-27b-noreason': LatencyPrior(0.5, 0.0090, 3.9),
    'qwen3.6-35b-a3b': LatencyPrior(2.0, 0.0069, 1.8),
    'qwen3.8-27b': LatencyPrior(1.0, 0.0104, 2.4),
}
_FALLBACK_PRIOR = LatencyPrior(2.0, 0.02, 4.0, 'unsupported_scope_conservative_prior')
_TOOL_PRIORS = {'coding': 3.0, 'math': 2.5, 'search': 2.5, 'data_analysis': 2.5}


@dataclass(frozen=True)
class LatencyForecast:
    seconds: float
    conservative_seconds: float
    estimated_calls: float
    source: str
    observations: int
    censored_observations: int = 0
    unknown_usage_observations: int = 0
    tool_observations: int = 0


@dataclass(frozen=True)
class ResourceCandidate:
    key: str
    expected_reward: float
    predicted_tokens: float
    predicted_seconds: float
    conservative_seconds: float
    token_priority: float
    continuation: bool = False


@dataclass(frozen=True)
class ResourceChoice:
    key: str
    eligible: bool
    score: float
    reason: str
    token_pressure: bool
    time_pressure: bool


@dataclass
class _Observations:
    ratios: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    censored_seconds: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    unknown_seconds: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    tool_seconds: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))


def _finite(value: object, *, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _quantile(values, fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _bounded_product(left: float, right: float) -> float:
    # Saturation protects trace JSON from infinity. A billion seconds already
    # exceeds every supported run deadline; this is not an execution time quota.
    return min(1_000_000_000.0, left * right)


class LatencyEstimator:
    """Per-model/domain rolling calibration; no task text or answers retained."""

    def __init__(self, priors: Mapping | None = None):
        self._priors = dict(DEFAULT_PRIORS)
        self._domain_priors = {}
        for key, value in (priors or {}).items():
            if not isinstance(value, LatencyPrior):
                raise ValueError('priors must contain LatencyPrior values')
            if (not _finite(value.request_overhead_seconds)
                    or not _finite(value.seconds_per_weighted_token, positive=True)
                    or not _finite(value.upper_multiplier, positive=True)
                    or value.upper_multiplier < 1 or type(value.source) is not str):
                raise ValueError('invalid latency prior')
            if type(key) is str and key in DEFAULT_PRIORS:
                self._priors[key] = value
            elif (type(key) is tuple and len(key) == 2 and type(key[0]) is str
                  and key[0] in DEFAULT_PRIORS and type(key[1]) is str and key[1] in DOMAINS):
                self._domain_priors[key] = value
            else:
                raise ValueError('unsupported prior scope')
        self._observations: dict[tuple[str, str], _Observations] = {}

    @staticmethod
    def _supported(model, domain) -> bool:
        return type(model) is str and model in DEFAULT_PRIORS and type(domain) is str and domain in DOMAINS

    def _prior(self, model, domain) -> LatencyPrior:
        if not self._supported(model, domain):
            return _FALLBACK_PRIOR
        return self._domain_priors.get((model, domain), self._priors[model])

    def observe_call(self, *, model: str, domain: str, elapsed_seconds: float,
                     weighted_tokens: float | None = None, usage_known: bool = False,
                     completed: bool = True, censored: bool = False) -> bool:
        """Record active HTTP latency, excluding transport backoff and queue waits.

        Unknown usage and failed/censored calls never update a token-rate sample.
        Their durations can only raise the conservative request-time estimate.
        """
        if (not self._supported(model, domain) or not _finite(elapsed_seconds, positive=True)
                or elapsed_seconds > MAX_OBSERVATION_SECONDS
                or any(type(value) is not bool for value in (usage_known, completed, censored))):
            return False
        observation = self._observations.setdefault((model, domain), _Observations())
        if censored or not completed:
            observation.censored_seconds.append(float(elapsed_seconds))
        elif usage_known and _finite(weighted_tokens, positive=True):
            prior = self._prior(model, domain)
            baseline = prior.request_overhead_seconds + prior.seconds_per_weighted_token * weighted_tokens
            ratio = elapsed_seconds / baseline
            # Robust calibration has bounded influence. Raw censored durations
            # remain separate instead of masquerading as quick completions.
            observation.ratios.append(min(16.0, max(0.125, ratio)))
        else:
            observation.unknown_seconds.append(float(elapsed_seconds))
        return True

    def observe_tool(self, *, model: str, domain: str, elapsed_seconds: float) -> bool:
        """Record completed ACTIVE-visit overhead after API latency/backoff removal.

        The caller must exclude waiting behind other tasks and incomplete visits.
        """
        if (not self._supported(model, domain) or not _finite(elapsed_seconds)
                or elapsed_seconds > MAX_OBSERVATION_SECONDS):
            return False
        self._observations.setdefault((model, domain), _Observations()).tool_seconds.append(float(elapsed_seconds))
        return True

    def forecast(self, *, model: str, domain: str, predicted_tokens: float,
                 first_call_tokens: float | None = None, estimated_calls: float | None = None,
                 include_tools: bool = True) -> LatencyForecast:
        if not _finite(predicted_tokens) or type(include_tools) is not bool:
            raise ValueError('predicted_tokens must be finite and nonnegative')
        if estimated_calls is not None and not _finite(estimated_calls, positive=True):
            raise ValueError('estimated_calls must be positive and finite')
        if estimated_calls is None:
            calls = predicted_tokens / first_call_tokens if _finite(first_call_tokens, positive=True) else 1.0
        else:
            calls = float(estimated_calls)
        calls = min(1_000_000.0, max(1.0, calls))
        prior = self._prior(model, domain)
        observation = self._observations.get((model, domain)) if self._supported(model, domain) else None
        ratios = observation.ratios if observation else ()
        count = len(ratios)
        scale = 1.0
        if count:
            weight = count / (count + 8.0)
            scale = min(4.0, max(0.25, 1.0 + weight * (median(ratios) - 1.0)))
        baseline = min(1_000_000_000.0, _bounded_product(prior.request_overhead_seconds, calls)
                       + _bounded_product(prior.seconds_per_weighted_token, predicted_tokens))
        point = _bounded_product(baseline, scale)
        upper_factor = max(prior.upper_multiplier * scale, _quantile(ratios, 0.9) if count else 1.0)
        upper = _bounded_product(baseline, upper_factor)
        unknown = observation.unknown_seconds if observation else ()
        censored = observation.censored_seconds if observation else ()
        # These are lower-bound/wall-clock signals, not successful-rate labels.
        for durations in (unknown, censored):
            if durations:
                upper = max(upper, _bounded_product(_quantile(durations, 0.75), calls))
        tools = observation.tool_seconds if observation else ()
        if include_tools:
            tool_prior = _TOOL_PRIORS.get(domain, 3.0) if type(domain) is str else 3.0
            tool_point = tool_prior
            if tools:
                weight = len(tools) / (len(tools) + 8.0)
                tool_point = tool_prior * (1 - weight) + median(tools) * weight
            point += tool_point
            upper += max(tool_prior, tool_point, _quantile(tools, 0.9) if tools else 0.0)
        source = prior.source + ('+online' if count or tools or unknown or censored else '')
        return LatencyForecast(min(point, 1_000_000_000.0), min(max(point, upper), 1_000_000_000.0),
                               calls, source, count, len(censored), len(unknown), len(tools))


def _valid_candidate(candidate: ResourceCandidate, *, require_time: bool) -> bool:
    if not isinstance(candidate, ResourceCandidate) or type(candidate.key) is not str:
        return False
    if type(candidate.continuation) is not bool:
        return False
    if not all(_finite(value) for value in (candidate.expected_reward, candidate.predicted_tokens, candidate.token_priority)):
        return False
    return not require_time or (all(_finite(value) for value in
        (candidate.predicted_seconds, candidate.conservative_seconds))
        and candidate.conservative_seconds >= candidate.predicted_seconds)


def _exceeds(values, remaining: float) -> bool:
    used = 0.0
    for value in values:
        if value > remaining - used:
            return True
        used += value
    return False


def _joint_score(reward: float, tokens: float, seconds: float,
                 remaining_tokens: float, remaining_seconds: float) -> float:
    share = max(tokens / remaining_tokens, seconds / remaining_seconds, 1e-12)
    return min(1e12, reward / share)


def rank_candidates(candidates, *, remaining_tokens: float, remaining_seconds: float,
                    mode: str = 'joint') -> tuple[ResourceChoice, ...]:
    """Return all choices, eligible first; stable ties preserve original order.

    New tasks retain the existing strict 1.25 token affordability guard.
    Continuations use the retry controller's inclusive 1.1 guard, so an allowed
    retry participates in the same aggregate pressure as competing new tasks.
    With ample time, scores are exactly the caller's token priorities. Under
    time pressure, reward is divided by the larger normalized resource share.
    SLA-v3 uses expected duration for admission and scarce-time ranking. Upper
    estimates remain validated diagnostics, not expected resource consumption.
    Callers must recompute alternatives after elapsed time or spending, and
    enforce the actual global deadline during execution.
    """
    if mode not in ('joint', 'token'):
        raise ValueError('mode must be joint or token')
    joint = mode == 'joint'
    rows = []
    for candidate in candidates:
        key = candidate.key if isinstance(candidate, ResourceCandidate) and type(candidate.key) is str else ''
        if not _valid_candidate(candidate, require_time=joint):
            reason = 'invalid_forecast'
        elif candidate.expected_reward <= 0 or candidate.token_priority <= 0:
            reason = 'zero_expected_value'
        elif (not _finite(remaining_tokens, positive=True)
              or (candidate.continuation and candidate.predicted_tokens * 1.1 > remaining_tokens)
              or (not candidate.continuation and candidate.predicted_tokens * 1.25 >= remaining_tokens)):
            reason = 'token_budget_insufficient'
        elif joint and (not _finite(remaining_seconds, positive=True) or candidate.predicted_seconds > remaining_seconds):
            reason = 'global_deadline_insufficient'
        else:
            reason = 'eligible'
        rows.append((candidate, key, reason))
    eligible = [candidate for candidate, _, reason in rows if reason == 'eligible']
    token_pressure = bool(eligible) and _exceeds((item.predicted_tokens for item in eligible), remaining_tokens)
    time_pressure = joint and bool(eligible) and _exceeds((item.predicted_seconds for item in eligible), remaining_seconds)
    choices = []
    for candidate, key, reason in rows:
        allowed = reason == 'eligible'
        score = 0.0
        if allowed:
            score = candidate.token_priority
            if time_pressure:
                score = _joint_score(candidate.expected_reward, candidate.predicted_tokens,
                                     candidate.predicted_seconds, remaining_tokens, remaining_seconds)
                reason = 'joint_scarce_time_value'
            else:
                reason = 'token_priority_preserved'
        choices.append(ResourceChoice(key, allowed, score, reason, token_pressure, time_pressure))
    return tuple(sorted(choices, key=lambda choice: (not choice.eligible, -choice.score)))


def continuation_value(*, expected_reward: float, predicted_tokens: float,
                       forecast: LatencyForecast, remaining_tokens: float,
                       remaining_seconds: float, opportunity: float = 0.0,
                       time_pressure: bool = False, mode: str = 'joint') -> ResourceChoice:
    """A marginal check AFTER existing retry guards, never a hard impossibility.

    Pass freshly rescored alternative opportunity and its time_pressure state.
    This function cannot authorize a retry rejected by the existing controller.
    Token-only mode leaves that controller's decision untouched.
    SLA-v3 uses the point duration, consistently with fresh-queue value units;
    upper-duration uncertainty does not independently forbid expected value.
    """
    if mode not in ('joint', 'token'):
        raise ValueError('mode must be joint or token')
    if mode == 'token':
        return ResourceChoice('continuation', True, 0.0, 'token_policy_unchanged', False, False)
    if (not isinstance(forecast, LatencyForecast) or not _finite(expected_reward)
            or not _finite(predicted_tokens) or not _finite(opportunity)
            or not _finite(forecast.seconds) or not _finite(forecast.conservative_seconds)
            or forecast.conservative_seconds < forecast.seconds
            or type(time_pressure) is not bool):
        return ResourceChoice('continuation', False, 0.0, 'invalid_forecast', False, False)
    if expected_reward <= 0:
        return ResourceChoice('continuation', False, 0.0, 'zero_expected_value', False, time_pressure)
    if not _finite(remaining_tokens, positive=True) or predicted_tokens * 1.1 > remaining_tokens:
        return ResourceChoice('continuation', False, 0.0, 'token_budget_insufficient', True, time_pressure)
    if not _finite(remaining_seconds, positive=True) or forecast.seconds > remaining_seconds:
        return ResourceChoice('continuation', False, 0.0, 'global_deadline_insufficient', False, time_pressure)
    score = expected_reward / max(1.0, predicted_tokens)
    if time_pressure:
        score = _joint_score(expected_reward, predicted_tokens, forecast.seconds,
                             remaining_tokens, remaining_seconds)
    # The old retry controller owns token-only competition. Do not add a second
    # token-value rejection while time is plentiful.
    allowed = not time_pressure or score >= opportunity
    return ResourceChoice('continuation', allowed, score,
                          'continuation_value_sufficient' if allowed else 'other_tasks_have_higher_joint_value',
                          False, time_pressure)
