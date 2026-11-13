"""Choose additional work from observed feedback and the shared run budget."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    model: str
    max_tokens: int
    reason: str
    predicted_cost: float
    success_probability: float
    action: str = "continue"


def failure_count(feedback: str) -> int | None:
    match = re.search(r'\b(\d+) failed\b', feedback)
    return int(match.group(1)) if match else None


def decide_retry(*, kind: str, state: str, feedback: str, attempts: int,
                 last_model: str, reasoning_model: str, last_cap: int,
                 finish_reason: str | None, remaining: float, reward: float,
                 opportunity: float = 0, competing_cost: float = 0,
                 flags=(), repeated: bool = False, improved: bool = False,
                 allow_reasoning: bool = True, confidence_enabled: bool = False,
                 assessment=None, new_evidence: bool = False,
                 observed_cost: float | None = None,
                 task_confidence: float | None = None) -> RetryDecision:
    def stop(reason):
        return RetryDecision(False, last_model, 0, reason, 0, 0, "stop")
    if state in {'abstained', 'unsupported', 'unsupported_answer', 'public_tests_passed', 'answered', 'stalled'}:
        return stop(state)
    truncated = str(finish_reason).lower() in {'length', 'max_tokens', 'max_output_tokens'}
    if repeated and not truncated:
        return stop('same_candidate_or_failure')
    if (not truncated and 'conflicting_outputs' in flags and
            not set(flags).intersection({'explicit_stateful', 'mutable_arguments', 'incomplete_analysis'})):
        return stop('conflicting_test_requirements_after_failed_attempt')
    if not truncated and attempts >= 2 and not improved:
        return stop('no_verifiable_progress')
    syntax = bool(re.search(r'JSON|SyntaxError|invalid syntax|unique|no such (?:table|column)|parse|syntax error', feedback, re.I))
    if confidence_enabled and feedback == 'Invalid arithmetic expression':
        syntax = True
    if truncated:
        # A generation cap controls ONE call, never the accumulated task cost.
        model = last_model
        cap = min(32768, max(last_cap + 256, last_cap * 2))
        if cap == last_cap:
            return stop('provider_generation_limit')
        probability, cost = 0.65, cap * (0.5 if model.endswith('-noreason') else 0.8)
        reason = 'extend_truncated_generation'
    elif syntax or not allow_reasoning:
        model = last_model if last_model.endswith('-noreason') else reasoning_model
        cap = max(512, last_cap)
        probability, cost, reason = 0.6, 150.0, 'repair_tool_error'
    else:
        model = reasoning_model
        cap = max(512,last_cap) if model.endswith('-noreason') else max(3072,last_cap if not last_model.endswith('-noreason') else 0)
        probability, cost, reason = (0.5 if improved else 0.3), (250.0 if model.endswith('-noreason') else 1000.0), 'repair_verified_failure'
    if confidence_enabled:
        from confidence_policy import evaluate_continuation
        if (type(observed_cost) in (int, float) and math.isfinite(observed_cost)
                and observed_cost > 0 and model == last_model):
            cost = max(cost, observed_cost * (cap / max(1, last_cap) if truncated else 1))
        if (type(task_confidence) in (int, float) and math.isfinite(task_confidence)
                and 0 < task_confidence <= 1):
            probability = min(probability, task_confidence)
        confidence = evaluate_continuation(
            assessment=assessment, base_allowed=True, base_probability=probability,
            attempts=attempts, repeated=repeated, improved=improved,
            tool_error=syntax, new_evidence=new_evidence, truncated=truncated,
            reward=reward, expected_cost=cost, remaining=remaining,
            competing_cost=competing_cost, opportunity=opportunity)
        probability = confidence.probability
        if not confidence.retry:
            return RetryDecision(False, model, 0, confidence.reason, cost,
                                 probability, confidence.action)
        if confidence.assessment_used:
            reason = confidence.reason
    # Compare marginal gain only when tasks actually compete for remaining funds.
    # Surplus budget is not divided into arbitrary equal per-task quotas.
    if remaining < max(64, cost * 1.1):
        return stop('shared_budget_insufficient')
    if competing_cost + cost > remaining * 0.8 and opportunity > 0:
        if reward * probability / max(1, cost) < opportunity:
            return stop('other_tasks_have_higher_expected_value')
    return RetryDecision(True, model, int(cap), reason, cost, probability)
