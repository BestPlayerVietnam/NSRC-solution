"""Local cost and feasibility planner for the 100,000-token contest run.

No model call is made to choose a task. Estimates are intentionally conservative
until real per-task usage becomes available during the run.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


MODELS = {
    "small_plain": "qwen3.6-35b-a3b-noreason",
    "small_reason": "qwen3.6-35b-a3b",
    "large_plain": "qwen3.8-27b-noreason",
    "large_reason": "qwen3.8-27b",
}
TOKEN_BUDGET = 100_000
PRIOR_COST = {"math": 15.0, "search": 20.0, "data_analysis": 45.0,
              "coding": 45.0}
PRIOR_SUCCESS = {"math": 0.78, "search": 0.73, "data_analysis": 0.68,
                 "coding": 0.53}
PURE_BUILTINS = {"abs", "all", "any", "enumerate", "float", "int", "len",
                 "max", "min", "range", "reversed", "sorted", "str", "sum"}


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    kind: str
    reward: float
    predicted_tokens: float
    success_probability: float
    model: str
    impossible_reason: str | None = None
    raw_predicted_tokens: float | None = None
    policy_id: str = "run-agent-v1"

    @property
    def priority(self) -> float:
        if self.impossible_reason or self.reward <= 0:
            return 0.0
        return self.reward * self.success_probability / max(1.0, self.predicted_tokens)


def obviously_pure_functions(workspace: Path) -> set[str]:
    source = workspace / "mylib.py"
    if not source.is_file():
        return set()
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return set()
    pure = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        nodes = list(ast.walk(node))
        if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.Attribute)) for n in nodes):
            continue
        if any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) for n in nodes):
            continue
        if any(isinstance(n, ast.Call) and
               (not isinstance(n.func, ast.Name) or n.func.id not in PURE_BUILTINS)
               for n in nodes):
            continue
        arguments = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        assigned = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and
               n.id not in arguments | assigned | PURE_BUILTINS for n in nodes):
            continue
        pure.add(node.name)
    return pure


def impossible_reason(task: dict, workspace: Path) -> str | None:
    """Hard-skip only a contradiction proved under an explicit task contract.

    Missing familiar filenames/columns or an unsupported local template are
    reasons to use the general solver, not proofs that no solution exists.
    Current source purity alone also does not constrain a repaired function.
    """
    from feasibility import analyze_task
    report = analyze_task(task, workspace)
    return report.reason if report.hard_impossible else None



class TaskEstimator:
    """Calibrate a frozen heuristic within one model and execution policy.

    Costs are ratios to the *unadapted* estimate saved before execution. Answer
    availability and correctness of an available answer are separate targets.
    Public tests are useful diagnostics, but are not a hidden correctness label.
    """

    WINDOW = 12
    COST_PRIOR_WEIGHT = 3.0
    OUTCOME_PRIOR_WEIGHT = 4.0
    TRUSTED_VERIFIERS = frozenset({"official", "oracle", "hidden_tests", "ground_truth"})

    def __init__(self, policy_id: str = "run-agent-v1",
                 allow_reasoning_retry: bool = True,
                 initial_model: str = MODELS["large_plain"]) -> None:
        # Change policy_id when prompts, caps, tools, or retry rules change.
        self.policy_id = policy_id
        if initial_model not in MODELS.values():
            raise ValueError("Unknown initial model")
        self.initial_model = initial_model
        self.allow_reasoning_retry = allow_reasoning_retry
        self.cost_observations: dict[tuple[str, str, str], list[float]] = {}
        self.outcome_observations: dict[tuple[str, str, str], list[bool]] = {}
        self.answer_observations: dict[tuple[str, str, str], list[bool]] = {}
        self._last_plans: dict[str, TaskPlan] = {}

    @staticmethod
    def _key(plan: TaskPlan) -> tuple[str, str, str]:
        return plan.kind, plan.model, plan.policy_id

    def _append(self, observations: dict, key: tuple[str, str, str], value) -> None:
        window = observations.setdefault(key, [])
        window.append(value)
        del window[:-self.WINDOW]

    def estimate(self, task: dict, workspace: Path) -> TaskPlan:
        kind = str(task.get("task_type", ""))
        instruction = str(task.get("instruction", ""))
        reward = max(0.0, float(task.get("reward", 0)))
        reason = impossible_reason(task, workspace)
        model = self.initial_model
        base = PRIOR_COST.get(kind, 500.0)

        # This is a ranking feature, not an upper bound for budget admission.
        input_cost = len(instruction) / 40.0
        if kind == "coding":
            source_size = sum(min(p.stat().st_size, 7000) for p in workspace.rglob("*.py")
                              if not p.is_symlink() and not p.name.startswith("test_") and not any(x in {"tests","test","__pycache__"} or x.startswith(".") for x in p.relative_to(workspace).parts))
            test_size = sum(min(p.stat().st_size, 4500) for p in workspace.rglob("test_*.py") if not p.is_symlink())
            input_cost += (source_size + test_size) / 40.0
            base += min(2500.0, source_size / 2.5)
        elif kind == "search":
            context_bytes = sum(min(p.stat().st_size, 1500) for p in workspace.rglob("*")
                                if p.is_file() and p.suffix in {".md", ".txt", ".json"})
            input_cost += min(8500, context_bytes) / 40.0
        elif kind == "data_analysis":
            input_cost += 100.0 / 40.0

        # Reward is not a difficulty feature. These thresholds only describe
        # the legacy executor's explicit eligibility for a reasoning retry.
        retry_allowed = self.allow_reasoning_retry and (
            (kind == "coding" and reward >= 2) or
            (kind in {"math", "search", "data_analysis"} and reward >= 4)
        )
        raw_prediction = base + input_cost
        if retry_allowed:
            if kind == "coding":
                raw_prediction += 20.0 + min(150.0, source_size / 10.0)
            elif kind == "data_analysis":
                raw_prediction += 45.0
            else:
                raw_prediction += 5.0

        policy_id = self.policy_id if self.policy_id.startswith("production-cpu-v3") else f"{self.policy_id}|reasoning_retry={int(retry_allowed)}"
        key = kind, model, policy_id
        ratios = self.cost_observations.get(key, [])
        scale = (self.COST_PRIOR_WEIGHT + sum(ratios)) / (
            self.COST_PRIOR_WEIGHT + len(ratios))
        predicted = max(1.0, raw_prediction * scale)

        prior_p = PRIOR_SUCCESS.get(kind, 0.40)
        outcomes = self.outcome_observations.get(key, [])
        correctness = (self.OUTCOME_PRIOR_WEIGHT * prior_p + sum(outcomes)) / (
            self.OUTCOME_PRIOR_WEIGHT + len(outcomes))
        answers = self.answer_observations.get(key, [])
        availability = (self.OUTCOME_PRIOR_WEIGHT + sum(answers)) / (
            self.OUTCOME_PRIOR_WEIGHT + len(answers))
        plan = TaskPlan(
            str(task.get("task_id", "")), kind, reward, predicted,
            correctness * availability, model, reason,
            raw_predicted_tokens=raw_prediction, policy_id=policy_id)
        self._last_plans[kind] = plan
        return plan

    def observe_plan(self, snapshot: TaskPlan, weighted_tokens: float | None, *,
                     usage_status: str = "provider_reconciled",
                     verdict: bool | None = None,
                     verifier_kind: str = "unknown",
                     produced_answer: bool | None = None) -> None:
        """Observe the selected, pre-execution plan without inventing labels.

        Only reconciled provider usage calibrates cost. A retained reservation
        remains a budget charge but is not an observed consumption label.
        Correctness is conditional on an available answer; a known missing
        answer only updates availability, avoiding two penalties for one event.
        """
        if produced_answer is not None and not isinstance(produced_answer, bool):
            raise ValueError("produced_answer must be bool or None")
        if verdict is not None and not isinstance(verdict, bool):
            raise ValueError("verdict must be bool or None")
        key = self._key(snapshot)
        if usage_status == "provider_reconciled" and weighted_tokens is not None:
            raw = snapshot.raw_predicted_tokens
            if raw is None:
                raw = snapshot.predicted_tokens  # compatibility with old TaskPlan
            try:
                cost, baseline = float(weighted_tokens), float(raw)
                ratio = cost / baseline if baseline > 0 else float("nan")
            except (TypeError, ValueError, OverflowError):
                ratio = float("nan")
                cost = baseline = float("nan")
            if (math.isfinite(cost) and cost >= 0 and math.isfinite(baseline)
                    and baseline > 0 and math.isfinite(ratio)):
                self._append(self.cost_observations, key, ratio)

        if produced_answer is not None:
            self._append(self.answer_observations, key, produced_answer)
        if (verdict is not None and verifier_kind in self.TRUSTED_VERIFIERS
                and produced_answer is not False):
            self._append(self.outcome_observations, key, verdict)

    def observe(self, kind: str, weighted_tokens: float, produced_answer: bool,
                verified: bool = False) -> None:
        """Legacy adapter; new callers must pass their selected plan explicitly.

        This API can only recover the most recent estimate of this kind.
        Legacy verified meant public tests, so it never supplies a hidden label.
        """
        snapshot = self._last_plans.get(kind)
        if snapshot is None:
            model = MODELS["large_plain"]
            raw = PRIOR_COST.get(kind, 500.0)
            snapshot = TaskPlan(
                "", kind, 0.0, raw, PRIOR_SUCCESS.get(kind, 0.40), model,
                raw_predicted_tokens=raw,
                policy_id=f"{self.policy_id}|reasoning_retry=0")
        self.observe_plan(
            snapshot, weighted_tokens, produced_answer=produced_answer,
            verifier_kind="public_tests" if verified else "unknown")
