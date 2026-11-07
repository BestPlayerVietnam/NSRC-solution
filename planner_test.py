import tempfile
import unittest
import asyncio
from pathlib import Path

from planner import MODELS, TaskEstimator, impossible_reason
from run_agent import csv_answer, model_answer


class PlannerTests(unittest.TestCase):
    def test_all_four_models_are_available_to_router(self):
        self.assertEqual(len(set(MODELS.values())), 4)
        self.assertTrue(all(name.startswith("qwen3.") for name in MODELS.values()))

    def test_missing_person_is_skipped_before_spending_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kb").mkdir()
            task = {"task_id": "missing", "task_type": "search", "reward": 5,
                    "instruction": "In which year was Irina Smith born?"}
            self.assertEqual(impossible_reason(task, workspace), "named person missing")
            self.assertEqual(TaskEstimator().estimate(task, workspace).priority, 0)

    def test_missing_coding_source_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "test_mylib.py").write_text("pass\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Fix mylib.py", "reward": 5}
            self.assertEqual(impossible_reason(task, workspace), "missing source")

    def test_contradictory_coding_assertions_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "mylib.py").write_text("def f(x): return x\n", encoding="utf-8")
            (workspace / "test_mylib.py").write_text(
                "import mylib as m\n"
                "def test(self):\n"
                "    self.assertEqual(m.f(1), 1)\n"
                "    self.assertEqual(m.f(1), 2)\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Fix mylib.py", "reward": 5}
            self.assertEqual(impossible_reason(task, workspace), "contradictory tests")

    def test_stateful_function_is_not_declared_impossible(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "mylib.py").write_text(
                "counter = 0\ndef f(x):\n    global counter\n    counter += 1\n    return counter\n",
                encoding="utf-8")
            (workspace / "test_mylib.py").write_text(
                "import mylib as m\n"
                "def test(self):\n"
                "    self.assertEqual(m.f(1), 1)\n"
                "    self.assertEqual(m.f(1), 2)\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Fix mylib.py", "reward": 5}
            self.assertIsNone(impossible_reason(task, workspace))

    def test_missing_required_csv_column_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "sales.csv").write_text("product,month\nwidget,7\n", encoding="utf-8")
            task = {"task_type": "data_analysis", "reward": 3,
                    "instruction": "How many units of product 'widget' were sold in months 7 through 8 inclusive?"}
            self.assertEqual(impossible_reason(task, workspace), "required CSV columns missing")

    def test_csv_aggregates_do_not_need_model_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "sales.csv").write_text(
                "order_id,region,product,month,units,price\n"
                "1,west,widget,1,3,10\n2,east,widget,1,5,30\n"
                "3,west,widget,2,4,20\n", encoding="utf-8")
            self.assertEqual(csv_answer("What is the average price of widget orders?", workspace), "20")
            self.assertEqual(csv_answer("How many units of widget were sold in the west region?", workspace), "7")

    def test_cost_and_answer_rate_adapt_after_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = {"task_type": "math", "instruction": "Calculate a value", "reward": 5}
            estimator = TaskEstimator()
            initial = estimator.estimate(task, workspace)
            estimator.observe("math", 1000, False)
            updated = estimator.estimate(task, workspace)
            self.assertGreater(updated.predicted_tokens, initial.predicted_tokens)
            self.assertLess(updated.success_probability, initial.success_probability)
            self.assertLess(updated.priority, initial.priority)

    def test_unverified_model_answer_does_not_inflate_success_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = {"task_type": "math", "instruction": "Calculate a value", "reward": 5}
            estimator = TaskEstimator()
            initial = estimator.estimate(task, workspace)
            estimator.observe("math", 10, True, verified=False)
            updated = estimator.estimate(task, workspace)
            self.assertEqual(updated.success_probability, initial.success_probability)

    def test_search_rejects_answer_absent_from_evidence(self):
        class FakeLLM:
            async def ask(self, *_args, **_kwargs):
                return "Paris"

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            kb = workspace / "kb"
            kb.mkdir()
            (kb / "company_example.md").write_text("Headquarters: Osaka\n", encoding="utf-8")
            task = {"task_type": "search", "instruction": "Where is Example headquartered?"}
            self.assertIsNone(asyncio.run(model_answer(FakeLLM(), task, workspace)))


if __name__ == "__main__":
    unittest.main()
