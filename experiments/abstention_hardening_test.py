"""Offline strict-abstention regressions against the current runtime; temporary workspaces only."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_agent as runtime

REPEATED = "UNSOLVABLE\n\n\nUNSOLVABLE"

class FakeLLM:
    def __init__(self, *answers, model, finish_reason="stop"):
        self.answers = iter(answers)
        self.calls = []
        self.initial_model = model
        self.fixed_model = True
        self.retry_policy = "adaptive"
        self.last_finish_reason = finish_reason
        self.last_cap = 0
        self.remaining = 100000
        self.opportunity = 0
        self.competing_cost = 0
        self.retry_decisions = []
        self.task_state = "unknown"

    async def ask(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        self.last_cap = kwargs.get("max_tokens", 0)
        return next(self.answers)

class AbstentionHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_path = ROOT / "run_agent.py"
        cls.original = cls.original_path.read_bytes()
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="abstention-regression-", dir=ROOT / ".tools")
        cls.stage = Path(cls.temporary.name).resolve()
        if not cls.stage.is_relative_to((ROOT / ".tools").resolve()):
            raise AssertionError("Temporary directory outside intended scope")
        cls.agent = runtime

    @classmethod
    def tearDownClass(cls):
        if cls.original_path.read_bytes() != cls.original:
            raise AssertionError("Frozen runtime changed")
        cls.temporary.cleanup()

    def workspace(self, files):
        temporary = tempfile.TemporaryDirectory(prefix="case-", dir=self.stage)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def fake(self, *answers, **kwargs):
        return FakeLLM(*answers, model=self.agent.MODELS["large_plain"], **kwargs)

    def broken_coding(self):
        original = "def plus(a, b):\n    return a - b\n"
        tests = "from mylib import plus\n\ndef test_plus():\n    assert plus(2, 3) == 5\n"
        workspace = self.workspace({"mylib.py": original, "test_mylib.py": tests})
        task = {"task_type": "coding", "instruction": "Fix plus in mylib.py.", "reward": 5}
        return workspace, task, original, tests

    def test_strict_repeated_markers(self):
        for text in ("UNSOLVABLE", REPEATED, " unsolvable.\r\n\r\nUNSOLVABLE! ",
                     "\nUNSOLVABLE\n", "UNSOLVABLE.\nUNSOLVABLE!\nunsolvable"):
            with self.subTest(text=text):
                self.assertTrue(self.agent.is_explicit_abstention(text))

    def test_code_strings_mixed_text_and_empty_are_not_markers(self):
        tick = chr(96)
        cases = (
            "", " \n\t", None, 'print("UNSOLVABLE")', '"UNSOLVABLE"', "'UNSOLVABLE'",
            "UNSOLVABLE\n42", "UNSOLVABLE\nCALC: 2 + 2",
            "UNSOLVABLE because information is missing", "UNSOLVABLE\n# explanation",
            tick + "UNSOLVABLE" + tick,
            tick * 3 + "python\nUNSOLVABLE\n" + tick * 3,
            "UNSOLVABLE!!", "UNSOLVABLE?", "UNSOLVABLE...", "UNSOLVABLE\u200b",
            "UNſOLVABLE",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(self.agent.is_explicit_abstention(text))

    def test_coding_repeated_abstention_one_call_and_original_source(self):
        workspace, task, original, tests = self.broken_coding()
        llm = self.fake(REPEATED)
        solved = asyncio.run(self.agent.solve_coding(llm, task, workspace))
        self.assertFalse(solved)
        self.assertEqual(llm.task_state, "abstained")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual((workspace / "mylib.py").read_text(), original)
        self.assertEqual((workspace / "test_mylib.py").read_text(), tests)

    def test_abstention_after_failed_patch_rolls_back_actual_change(self):
        workspace, task, original, tests = self.broken_coding()
        candidate = json.dumps({"files": {"mylib.py": "def plus(a, b):\n    return 0\n"}})
        llm = self.fake(candidate, REPEATED)
        llm.retry_policy = "classic"
        self.assertFalse(asyncio.run(self.agent.solve_coding(llm, task, workspace)))
        self.assertEqual(llm.task_state, "abstained")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual((workspace / "mylib.py").read_text(), original)
        self.assertEqual((workspace / "test_mylib.py").read_text(), tests)

    def test_code_returning_marker_string_is_still_a_solution(self):
        workspace = self.workspace({
            "mylib.py": "def label():\n    return 'missing'\n",
            "test_mylib.py": "from mylib import label\n\ndef test_label():\n    assert label() == 'UNSOLVABLE'\n",
        })
        task = {"task_type": "coding", "instruction": "Make label return the required string.", "reward": 5}
        llm = self.fake("def label():\n    return 'UNSOLVABLE'\n")
        self.assertTrue(asyncio.run(self.agent.solve_coding(llm, task, workspace)))
        self.assertEqual(llm.task_state, "public_tests_passed")
        self.assertEqual(len(llm.calls), 1)

    def test_all_three_non_coding_sites_use_repeated_marker(self):
        workspace = self.workspace({"data.csv": "x\n1\n", "kb/doc.md": "The city is Riga.\n"})
        for kind in ("math", "data_analysis", "search"):
            with self.subTest(kind=kind):
                llm = self.fake(REPEATED)
                answer = asyncio.run(self.agent.model_answer(
                    llm, {"task_type": kind, "instruction": "Find the missing value."},
                    workspace, llm.initial_model))
                self.assertIsNone(answer)
                self.assertEqual(llm.task_state, "abstained")
                self.assertEqual(len(llm.calls), 1)

    def test_non_coding_truncation_checks_keep_precedence(self):
        workspace = self.workspace({"data.csv": "x\n1\n", "kb/doc.md": "The city is Riga.\n"})
        for kind in ("math", "data_analysis", "search"):
            with self.subTest(kind=kind):
                llm = self.fake(REPEATED, finish_reason="length")
                self.assertIsNone(asyncio.run(self.agent.model_answer(
                    llm, {"task_type": kind, "instruction": "Find the missing value."},
                    workspace, llm.initial_model)))
                self.assertEqual(llm.task_state, "check_failed")
                self.assertEqual(llm.last_feedback, "Generation truncated")
                self.assertEqual(len(llm.calls), 1)

    def test_coding_existing_abstention_before_truncation_order_is_preserved(self):
        workspace, task, original, _ = self.broken_coding()
        llm = self.fake(REPEATED, finish_reason="length")
        self.assertFalse(asyncio.run(self.agent.solve_coding(llm, task, workspace)))
        self.assertEqual(llm.task_state, "abstained")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual((workspace / "mylib.py").read_text(), original)

    def test_search_does_not_strip_code_into_abstention(self):
        workspace = self.workspace({"kb/doc.md": "The city is Riga.\n"})
        tick = chr(96)
        for text in (tick + "UNSOLVABLE" + tick, tick * 3 + "\nUNSOLVABLE\n" + tick * 3):
            with self.subTest(text=text):
                llm = self.fake(text)
                self.assertIsNone(asyncio.run(self.agent.model_answer(
                    llm, {"task_type": "search", "instruction": "Which city?"},
                    workspace, llm.initial_model)))
                self.assertNotEqual(llm.task_state, "abstained")

if __name__ == "__main__":
    unittest.main()
