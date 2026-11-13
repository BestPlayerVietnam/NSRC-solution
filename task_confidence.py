"""Cheap, uncalibrated confidence adjustments from visible coverage signals.

This is a ranking heuristic, not an impossibility proof or a calibrated success
probability. Its fixed multipliers are policy choices, not fitted measurements.
Only the strongest adjustment applies: overlapping coverage/conflict signals
must not be counted as independent evidence. Task length is not difficulty.

No task identifiers, rewards, answers, oracle labels, or model responses are
read. A positive floor deliberately prevents this score from being a hard skip.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


_MODELS = frozenset({
    'qwen3.6-35b-a3b', 'qwen3.6-35b-a3b-noreason',
    'qwen3.8-27b', 'qwen3.8-27b-noreason',
})
_DOMAINS = frozenset({'math', 'coding', 'search', 'data_analysis'})
_FEATURE_FLAGS = ('context_partial', 'missing_evidence_flag', 'contradiction_flag')
_AMBIGUOUS_CONFLICT = frozenset({
    'explicit_stateful', 'mutable_arguments', 'incomplete_analysis',
})
_RISK_FLAGS = _AMBIGUOUS_CONFLICT | {'missing_evidence', 'conflicting_outputs'}
_MIN_PROBABILITY = 0.01
_MAX_PROBABILITY = 0.99
_DEFAULT_PRIOR = 0.40


@dataclass(frozen=True)
class ConfidenceEstimate:
    probability: float
    source: str
    reasons: tuple[str, ...]


def _prior(value: object, reasons: list[str]) -> float:
    if type(value) not in (int, float):
        reasons.append('invalid_prior_default_0.40')
        return _DEFAULT_PRIOR
    try:
        number = float(value)
    except (OverflowError, ValueError):
        number = float('nan')
    if not math.isfinite(number):
        reasons.append('invalid_prior_default_0.40')
        return _DEFAULT_PRIOR
    bounded = min(_MAX_PROBABILITY, max(_MIN_PROBABILITY, number))
    if bounded != number:
        reasons.append('prior_clipped_to_0.01_0.99')
    return bounded


def _visible_flags(features: object, reasons: list[str]) -> dict[str, bool]:
    flags = dict.fromkeys(_FEATURE_FLAGS, False)
    if not isinstance(features, Mapping):
        reasons.append('invalid_features_ignored')
        return flags
    for key in _FEATURE_FLAGS:
        value = features.get(key, 0.0)
        # Extracted flags are binary; never interpret NaN, strings, or an OOD
        # magnitude as strong evidence merely because Python considers it true.
        if type(value) in (int, float, bool) and value in (0, 1):
            flags[key] = bool(value)
        else:
            reasons.append(f'invalid_feature_ignored:{key}')
    return flags


def _risk_flags(risk: object, reasons: list[str]) -> frozenset[str]:
    if risk is None:
        return frozenset()
    raw = risk.get('flags', ()) if isinstance(risk, Mapping) else getattr(risk, 'flags', ())
    if not isinstance(raw, (tuple, list, set, frozenset)) or len(raw) > 32:
        reasons.append('invalid_risk_flags_ignored')
        return frozenset()
    # Do not consume arbitrary iterators or the report's hard_impossible field.
    return frozenset(v for v in raw if type(v) is str and v in _RISK_FLAGS)


def estimate_confidence(task: Mapping, features: Mapping,
                        prior_probability: float, model: str, *,
                        risk: object = None) -> ConfidenceEstimate:
    """Return a bounded soft score using a caller-owned, model-specific prior.

    ``risk`` can be a FeasibilityReport or a mapping with ``flags``. Providing it
    avoids repeating AST/filesystem analysis. Unsupported domains/models retain
    the bounded prior rather than transferring these hand-set adjustments.
    Missing or malformed flags are unknown, not evidence of impossibility.
    """
    reasons: list[str] = []
    prior = _prior(prior_probability, reasons)
    domain = task.get('task_type') if isinstance(task, Mapping) else None
    if type(model) is not str or model not in _MODELS:
        reasons.append('prior_only_unknown_model')
    if type(domain) is not str or domain not in _DOMAINS:
        reasons.append('prior_only_unknown_domain')
    if any(reason.startswith('prior_only_') for reason in reasons):
        return ConfidenceEstimate(prior, 'prior_only_v1', tuple(reasons))

    flags = _visible_flags(features, reasons)
    risks = _risk_flags(risk, reasons)
    factors = [1.0]
    if flags['context_partial']:
        factors.append(0.85)
        reasons.append('context_partial')
    if flags['missing_evidence_flag'] or 'missing_evidence' in risks:
        factors.append(0.70)
        reasons.append('missing_evidence')
    if flags['contradiction_flag'] or 'conflicting_outputs' in risks:
        if risks & _AMBIGUOUS_CONFLICT:
            factors.append(0.85)
            reasons.append('conflicting_outputs_ambiguous')
        else:
            factors.append(0.60)
            reasons.append('conflicting_outputs_unproven')
    if len(factors) == 1:
        reasons.append('no_visible_risk_adjustment')
    probability = max(_MIN_PROBABILITY, prior * min(factors))
    return ConfidenceEstimate(probability, 'heuristic_visible_v1', tuple(reasons))
