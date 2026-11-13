"""Exercise proven impossibility through the normal CLI before tests or model calls."""
import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_agent
from feasibility import analyze_task


INTEGRITY = (
    "All conditions apply to the same returned object, using ordinary Python "
    "built-in semantics. Modifying tests, built-ins, globals or the test "
    "framework is forbidden."
)


def constraints(kind, possible):
    if kind == "letters":
        upper = "A" if possible else "B"
        return ("assert type(value) is str\n"
                "assert len(value) == 1\n"
                "assert 'a' <= value <= 'z' or 'A' <= value <= 'Z'\n"
                "assert value.lower() == 'a'\n"
                f"assert value.upper() == {upper!r}")
    if kind == "bytes":
        length = 2 if possible else 1
        return ("assert type(value) is bytes\n"
                "assert len(value) == 1\n"
                f"assert len(value.hex()) == {length}")
    composition = "value[value[value[i]]]" if possible else "value[value[value[value[i]]]]"
    return ("assert type(value) is tuple and len(value) == 3\n"
            "assert all(type(item) is int for item in value)\n"
            "assert set(value) == {0, 1, 2}\n"
            f"assert all({composition} == i for i in range(3))\n"
            "assert any(value[value[i]] != i for i in range(3))")


class FiniteFeasibilityIntegrationTests(unittest.TestCase):
    def make_tasks(self, root, possible):
        witnesses = {"letters": "'a'", "bytes": "b'\\xaf'", "permutation": "(1, 2, 0)"}
        for kind, witness in witnesses.items():
            directory = root / kind
            workspace = directory / "workspace"
            workspace.mkdir(parents=True)
            (directory / "task.json").write_text(json.dumps({
                "task_id": kind, "task_type": "coding", "reward": 3,
                "instruction": "Implement factory() in mylib.py. " + INTEGRITY,
            }), encoding="utf-8")
            expression = witness if possible else "None"
            (workspace / "mylib.py").write_text(
                f"def factory():\n    return {expression}\n", encoding="utf-8")
            body = constraints(kind, possible)
            test = ("from mylib import factory\n\ndef test_contract():\n"
                    "    value = factory()\n" +
                    "\n".join("    " + line for line in body.splitlines()) + "\n")
            (workspace / "test_contract.py").write_text(test, encoding="utf-8")

    def launch(self, root):
        argv = ["run_agent.py", "--tasks-dir", str(root), "--budget", "5000",
                "--trace-file", str(root / "trace.jsonl")]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {
                "LLM_BASE_URL": "https://fixture.invalid", "LLM_API_KEY": "fixture"}), \
                redirect_stdout(io.StringIO()):
            asyncio.run(run_agent.main())

    def test_impossible_contracts_skip_before_tests_and_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tasks(root, possible=False)
            originals = {p: p.read_bytes() for p in root.rglob("*.py")}
            with patch.object(run_agent, "run_tests", side_effect=AssertionError("Task code executed")), \
                    patch.object(run_agent, "JiuwenLLM", side_effect=AssertionError("Model constructed")):
                self.launch(root)
            self.assertFalse(list(root.rglob("answer.txt")))
            rows = [json.loads(s) for s in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["task_id"] for r in rows}, {"letters", "bytes", "permutation"})
            self.assertTrue(all(r["outcome"] == "impossible" and r["actual_tokens"] == 0 for r in rows))
            self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_satisfiable_neighbors_reach_public_tests_and_save_done(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tasks(root, possible=True)
            with patch.object(run_agent, "JiuwenLLM", side_effect=AssertionError("Model constructed")):
                self.launch(root)
            for kind in ("letters", "bytes", "permutation"):
                self.assertEqual((root / kind / "answer.txt").read_text(encoding="utf-8"), "DONE\n")
            rows = [json.loads(s) for s in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(r["outcome"] == "solved_local" for r in rows))

    def test_requested_file_limit_prevents_a_finite_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tasks(root, possible=False)
            for kind in ("letters", "bytes", "permutation"):
                task = json.loads((root / kind / "task.json").read_text(encoding="utf-8"))
                result = analyze_task(task, root / kind / "workspace", max_file_bytes=10)
                self.assertFalse(result.hard_impossible)

    def test_finite_verifier_failure_keeps_the_case_unproven(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tasks(root, possible=False)
            task = json.loads((root / "letters/task.json").read_text(encoding="utf-8"))
            with patch("finite_constraints.analyze_task", side_effect=RuntimeError("Verifier failure")):
                result = analyze_task(task, root / "letters/workspace")
            self.assertFalse(result.hard_impossible)


if __name__ == "__main__":
    unittest.main()
