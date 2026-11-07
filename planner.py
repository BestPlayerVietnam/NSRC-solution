"""Local cost and feasibility planner for the 100,000-token contest run.

No model call is made to choose a task. Estimates are intentionally conservative
until real per-task usage becomes available during the run.
"""

from __future__ import annotations

import ast
import csv
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


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    kind: str
    reward: float
    predicted_tokens: float
    success_probability: float
    model: str
    impossible_reason: str | None = None

    @property
    def priority(self) -> float:
        if self.impossible_reason or self.reward <= 0:
            return 0.0
        return self.reward * self.success_probability / max(1.0, self.predicted_tokens)


def impossible_reason(task: dict, workspace: Path) -> str | None:
    """Return only reasons that are evident from local files and task wording."""
    kind = str(task.get("task_type", ""))
    instruction = str(task.get("instruction", ""))
    if kind == "coding":
        tests = list(workspace.glob("test_*.py"))
        if not tests:
            return "missing tests"
        if not any(p for p in workspace.glob("*.py") if not p.name.startswith("test_")):
            return "missing source"
        # A pure module function cannot return two different literal results
        # for the same literal arguments. This catches a common impossible
        # test construction before any model call.
        expected_by_call: dict[str, object] = {}
        for path in tests:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "assertEqual" or len(node.args) < 2:
                    continue
                tested = node.args[0]
                if (not isinstance(tested, ast.Call) or
                        not isinstance(tested.func, ast.Attribute) or
                        not isinstance(tested.func.value, ast.Name) or
                        tested.func.value.id != "m"):
                    continue
                try:
                    args = tuple(ast.literal_eval(arg) for arg in tested.args)
                    expected = ast.literal_eval(node.args[1])
                except (ValueError, TypeError):
                    continue
                key = repr((tested.func.attr, args))
                if key in expected_by_call and expected_by_call[key] != expected:
                    return "contradictory tests"
                expected_by_call[key] = expected
    elif kind == "data_analysis":
        files = list(workspace.glob("*.csv"))
        if not files:
            return "missing CSV"
        named = re.findall(r"\b([A-Za-z_][\w-]*\.csv)\b", instruction, re.I)
        if named and not all((workspace / name).is_file() for name in named):
            return "named CSV missing"
        query = re.search(r"How many units of product ['\"][^'\"]+['\"] were sold in months "
                          r"\d+ through \d+ inclusive\?", instruction, re.I)
        if query and len(files) == 1:
            with files[0].open(newline="", encoding="utf-8-sig") as stream:
                columns = csv.DictReader(stream).fieldnames or []
            if not {"product", "month", "units"}.issubset(columns):
                return "required CSV columns missing"
    elif kind == "search":
        kb = workspace / "kb"
        if not kb.is_dir():
            return "missing knowledge base"
        born = re.search(r"(?:year was|when was) ([A-Za-z]+ [A-Za-z]+) born", instruction, re.I)
        employer_hq = re.search(r"headquarters of ([A-Za-z]+ [A-Za-z]+)'s employer", instruction, re.I)
        match = born or employer_hq
        if match:
            path = kb / ("person_" + match.group(1).lower().replace(" ", "_") + ".md")
            if not path.is_file():
                return "named person missing"
            body = path.read_text(encoding="utf-8", errors="replace")
            if born and not re.search(r"^Born:\s*\d{4}\s*$", body, re.M | re.I):
                return "birth year missing"
            if employer_hq:
                employer = re.search(r"^Employer:\s*(.+?)\s*$", body, re.M | re.I)
                if not employer:
                    return "employer missing"
                company_name = re.sub(r"[^a-z0-9]+", "_", employer.group(1).lower()).strip("_")
                company = kb / f"company_{company_name}.md"
                if not company.is_file():
                    return "employer file missing"
                if not re.search(r"^Headquarters:\s*.+$", company.read_text(encoding="utf-8", errors="replace"), re.M | re.I):
                    return "headquarters missing"
    return None


class TaskEstimator:
    def __init__(self) -> None:
        self.cost_observations: dict[str, list[float]] = {}
        self.outcome_observations: dict[str, list[bool]] = {}

    def estimate(self, task: dict, workspace: Path) -> TaskPlan:
        kind = str(task.get("task_type", ""))
        instruction = str(task.get("instruction", ""))
        reward = max(0.0, float(task.get("reward", 0)))
        reason = impossible_reason(task, workspace)
        base = PRIOR_COST.get(kind, 500.0)

        # Input tokens are discounted 10:1. A conservative four characters
        # per input token gives about one weighted token per 40 characters.
        input_cost = len(instruction) / 40.0
        if kind == "coding":
            source_size = sum(min(p.stat().st_size, 7000) for p in workspace.glob("*.py")
                              if not p.name.startswith("test_"))
            test_size = sum(min(p.stat().st_size, 4500) for p in workspace.glob("test_*.py"))
            input_cost += (source_size + test_size) / 40.0
            base += min(2500.0, source_size / 2.5)
        elif kind == "search":
            context_bytes = sum(min(p.stat().st_size, 1500) for p in workspace.rglob("*")
                                if p.is_file() and p.suffix in {".md", ".txt", ".json"})
            input_cost += min(8500, context_bytes) / 40.0
        elif kind == "data_analysis":
            input_cost += 100.0 / 40.0  # schema rather than full CSV rows

        predicted = base + input_cost
        if kind == "coding" and reward >= 2:
            predicted += 20.0 + min(150.0, source_size / 10.0)  # possible reasoning retry
        elif kind == "data_analysis" and reward >= 4:
            predicted += 45.0
        elif kind in {"math", "search"} and reward >= 4:
            predicted += 5.0
        observed = self.cost_observations.get(kind, [])[-12:]
        if observed:
            # Three virtual observations preserve a useful prior with tiny
            # sample sizes, while adapting to the actual proxy's usage data.
            predicted = (3 * predicted + sum(observed)) / (3 + len(observed))
        prior_p = PRIOR_SUCCESS.get(kind, 0.40)
        outcomes = self.outcome_observations.get(kind, [])[-12:]
        probability = (4 * prior_p + sum(outcomes)) / (4 + len(outcomes))
        model = MODELS["large_plain"] if kind == "coding" else MODELS["small_plain"]
        return TaskPlan(str(task.get("task_id", "")), kind, reward,
                        max(1.0, predicted), probability, model, reason)

    def observe(self, kind: str, weighted_tokens: float, produced_answer: bool,
                verified: bool = False) -> None:
        if weighted_tokens > 0:
            self.cost_observations.setdefault(kind, []).append(weighted_tokens)
            if not produced_answer or verified:
                self.outcome_observations.setdefault(kind, []).append(produced_answer)
