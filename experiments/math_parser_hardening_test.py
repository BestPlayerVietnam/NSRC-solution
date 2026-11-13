"""Offline current-runtime math regressions and historical differential.

python -B experiments/math_parser_hardening_test.py
python -B experiments/math_parser_hardening_test.py --differential

No runtime edits, model calls, oracle files, or dataset answer labels are used.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_agent as runtime

PATCH = ROOT / "experiments/math_parser_hardening.patch"
QUESTION = "How many boxes does it have at the end?"
SUFFIX = (
    "Compute it with the bash tool (python3 -c). Work only inside the current directory. "
    "When you are done, write ONLY the final answer to the file answer.txt in the current "
    "directory (for example with bash: echo 42 > answer.txt)."
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def historical_math_source():
    """Recover complete old function from old/context lines of its recorded diff."""
    old_lines = {}
    old_number = None
    for line in PATCH.read_text(encoding="utf-8").splitlines(keepends=True):
        header = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if header:
            old_number = int(header.group(1))
            continue
        if old_number is None or line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith((" ", "-")):
            if old_number in old_lines and old_lines[old_number] != line[1:]:
                raise AssertionError("Historical patch has inconsistent old context")
            old_lines[old_number] = line[1:]
            old_number += 1
        elif line.startswith("+"):
            continue
        elif line.startswith("\\ No newline"):
            continue
        else:
            raise AssertionError("Unsupported historical diff record")
    starts = [number for number, line in old_lines.items()
              if line.startswith("def math_answer(")]
    if len(starts) != 1:
        raise AssertionError("Historical patch must contain exactly one math_answer")
    start = starts[0]
    following = [number for number, line in old_lines.items()
                 if number > start and line.startswith(("def ", "async def ", "class "))]
    if not following:
        raise AssertionError("Historical function boundary is not recorded")
    end = min(following)
    if any(number not in old_lines for number in range(start, end)):
        raise AssertionError("Historical function is incomplete in patch context")
    source = "".join(old_lines[number] for number in range(start, end))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise AssertionError("Historical fixture must contain one function only")
    return source


def load_math_versions():
    namespace = {"re": re}
    # Only the recorded historical arithmetic function is compiled. The new
    # function is the real, already integrated runtime import.
    exec(compile(historical_math_source(), "<historical-math-parser>", "exec"), namespace)
    return namespace["math_answer"], runtime.math_answer

def problem(operations, question=QUESTION, suffix=""):
    text = "A warehouse starts with 10 boxes. " + operations + " " + question
    return text + (" " + suffix if suffix else "")

class MathParserHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_sha = sha(ROOT / "run_agent.py")
        old, new = load_math_versions()
        cls.old = staticmethod(old)
        cls.new = staticmethod(new)

    @classmethod
    def tearDownClass(cls):
        if sha(ROOT / "run_agent.py") != cls.runtime_sha:
            raise AssertionError("Frozen runtime changed during test")

    def test_two_official_templates_with_full_suffix(self):
        cases = [
            ("math_0000", ((((1752 // 2) * 2) // 3) * 2) // 3),
            ("math_0004", 1669 - 36 * 3 + 2 * 87),
        ]
        for name, independently_computed in cases:
            task = json.loads((ROOT / "TASK/task-dir" / name / "task.json").read_text(encoding="utf-8"))
            instruction = task["instruction"]  # Never use task["expected"].
            with self.subTest(name=name):
                self.assertEqual(self.new(instruction), str(independently_computed))
                self.assertEqual(self.old(instruction), self.new(instruction))

    def test_official_template_without_technical_suffix(self):
        task = json.loads((ROOT / "TASK/task-dir/math_0004/task.json").read_text(encoding="utf-8"))
        instruction = task["instruction"].split(" Compute it with", 1)[0]
        self.assertEqual(self.new(instruction), str(1669 - 36 * 3 + 2 * 87))

    def test_whitespace_and_case_are_semantically_irrelevant(self):
        text = problem("It receives 2 deliveries of 5 boxes each.", suffix=SUFFIX)
        self.assertEqual(self.new("\n  " + text.upper().replace(" ", "\t \n") + " \n"), "20")

    def test_ordered_known_operations_and_floor(self):
        text = problem(
            "It ships 3 boxes per day for 2 days. "
            "It receives 2 deliveries of 5 boxes each. "
            "Then the stock is doubled by merging with a sister warehouse. "
            "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down)."
        )
        self.assertEqual(self.new(text), str(((10 - 3 * 2 + 2 * 5) * 2) // 3))

    def test_donation_never_returns_partial_stock(self):
        text = problem("It receives 2 deliveries of 5 boxes each. It donates 3 boxes.")
        self.assertEqual(self.old(text), "20")
        self.assertIsNone(self.new(text))  # The full arithmetic answer would be 17.

    def test_half_operation_is_unrecognized(self):
        text = problem("It receives 2 deliveries of 5 boxes each. Then half the stock is donated.")
        self.assertEqual(self.old(text), "20")
        self.assertIsNone(self.new(text))

    def test_negated_events_are_not_positive_events(self):
        for operation in (
            "It does not receive 2 deliveries of 5 boxes each.",
            "It does not ship 3 boxes per day for 1 day.",
            "Then the stock is not doubled by merging with a sister warehouse.",
        ):
            with self.subTest(operation=operation):
                self.assertIsNone(self.new(problem(operation)))

    def test_different_target_question_is_not_final_stock(self):
        text = problem("It receives 2 deliveries of 5 boxes each.",
                       question="How many boxes were delivered?")
        self.assertEqual(self.old(text), "20")
        self.assertIsNone(self.new(text))  # Delivered quantity is 10, not final stock.

    def test_extra_unrelated_quantity_invalidates_closed_parse(self):
        text = problem("It receives 2 deliveries of 5 boxes each. The building has 3 doors.")
        self.assertEqual(self.old(text), "20")
        self.assertIsNone(self.new(text))

    def test_conditional_event_and_extra_suffix_are_unrecognized(self):
        self.assertIsNone(self.new(problem(
            "It receives 2 deliveries of 5 boxes each only if the trucks arrive.")))
        self.assertIsNone(self.new(problem(
            "It receives 2 deliveries of 5 boxes each.") + " The year is 2026."))
        self.assertIsNone(self.new(problem(
            "It receives 2 deliveries of 5 boxes each.", suffix=SUFFIX.replace("echo 42", "echo 17"))))

    def test_no_operations_or_invalid_split(self):
        self.assertIsNone(self.new("A warehouse starts with 10 boxes. " + QUESTION))
        self.assertIsNone(self.new(problem(
            "Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).")))

    def test_invalid_physical_stock_and_oversized_numbers_defer(self):
        self.assertIsNone(self.new(problem("It ships 100 boxes per day for 2 days.")))
        self.assertIsNone(self.new(problem(
            "It receives 2 deliveries of 5 boxes each.").replace("10 boxes", "9" * 100 + " boxes")))

def observed(function, instruction):
    try:
        return {"answer": function(instruction), "error": None}
    except Exception as exc:
        return {"answer": None, "error": type(exc).__name__}

def differential():
    """Only visible task.json instructions. Never read metadata or oracle files."""
    runtime_sha = sha(ROOT / "run_agent.py")
    old, new = load_math_versions()
    summaries = {}
    changed = []
    for name in ("suite_policy", "suite_public100"):
        tasks = ROOT / "experiments" / name / "tasks"
        if not tasks.is_dir():
            raise RuntimeError("Required visible suite is missing: " + name)
        summary = {
            "task_json_files": 0, "math_instructions": 0, "old_local_answers": 0,
            "new_local_answers": 0, "answer_to_none": 0, "none_to_answer": 0,
            "changed_nonempty_answer": 0, "old_errors": 0, "new_errors": 0,
            "changed": 0,
        }
        input_hashes = {}
        for path in sorted(tasks.glob("*/task.json")):
            summary["task_json_files"] += 1
            task = json.loads(path.read_text(encoding="utf-8-sig"))
            if task.get("task_type") != "math":
                continue
            summary["math_instructions"] += 1
            instruction = task["instruction"]
            ident = task.get("task_id", task.get("id", path.parent.name))
            input_hashes[ident] = hashlib.sha256(instruction.encode()).hexdigest()
            before, after = observed(old, instruction), observed(new, instruction)
            a, b = before["answer"], after["answer"]
            summary["old_local_answers"] += a is not None
            summary["new_local_answers"] += b is not None
            summary["answer_to_none"] += a is not None and b is None
            summary["none_to_answer"] += a is None and b is not None
            summary["changed_nonempty_answer"] += a is not None and b is not None and a != b
            summary["old_errors"] += before["error"] is not None
            summary["new_errors"] += after["error"] is not None
            if before != after:
                summary["changed"] += 1
                changed.append({"suite": name, "id": ident, "old": before, "new": after})
        summaries[name] = {**summary, "instruction_sha256": input_hashes}
    if sha(ROOT / "run_agent.py") != runtime_sha:
        raise AssertionError("Frozen runtime changed")
    report = {
        "scope": "Visible math task.json instructions only; no oracle/metadata/gold read",
        "api_calls": 0, "runtime_changed": False,
        "runtime_sha256": runtime_sha, "patch_sha256": sha(PATCH),
        "summaries": summaries, "changed_tasks": changed,
        "limitations": [
            "This is a local-answer differential, not a correctness evaluation.",
            "No live run was replayed and no model-token saving is estimated.",
            "The current parser is compared with its recorded historical baseline; neither was fitted to these inputs.",
        ],
    }
    output = ROOT / "experiments/math_parser_differential.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({name: {k: v for k, v in summary.items() if k != "instruction_sha256"}
                      for name, summary in summaries.items()}, indent=2))

if __name__ == "__main__":
    if "--differential" in sys.argv:
        differential()
    else:
        unittest.main()
