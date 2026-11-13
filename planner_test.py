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

    def test_missing_expected_person_filename_is_not_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kb").mkdir()
            task = {"task_id": "missing", "task_type": "search", "reward": 5,
                    "instruction": "In which year was Irina Smith born?"}
            self.assertIsNone(impossible_reason(task, workspace))
            self.assertGreater(TaskEstimator().estimate(task, workspace).priority, 0)

    def test_missing_coding_source_can_be_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "test_mylib.py").write_text("pass\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Fix mylib.py", "reward": 5}
            self.assertIsNone(impossible_reason(task, workspace))

    def test_contradictory_coding_assertions_require_explicit_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "mylib.py").write_text("def f(x): return x\n", encoding="utf-8")
            (workspace / "test_mylib.py").write_text(
                "import mylib as m\nimport unittest\n"
                "class TestContract(unittest.TestCase):\n"
                "    def test(self):\n"
                "        self.assertEqual(m.f(1), 1)\n"
                "        self.assertEqual(m.f(1), 2)\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Fix mylib.py", "reward": 5}
            self.assertIsNone(impossible_reason(task, workspace))
            task["instruction"] += ". Functions must be stateless and deterministic."
            self.assertIsNone(impossible_reason(task, workspace))
            task["instruction"] += " Functions must return built-in integers."
            self.assertIsNotNone(impossible_reason(task, workspace))

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

    def test_missing_expected_csv_column_is_not_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "sales.csv").write_text("product,month\nwidget,7\n", encoding="utf-8")
            task = {"task_type": "data_analysis", "reward": 3,
                    "instruction": "How many units of product 'widget' were sold in months 7 through 8 inclusive?"}
            self.assertIsNone(impossible_reason(task, workspace))

    def test_alternate_layouts_and_derived_data_are_not_hard_skipped(self):
        fixtures = [
            ("search", "In which year was Lina Zorina born?",
             {"people/biography.md": "Lina Zorina was born on 14 March 1991."}),
            ("search", "In which city is the headquarters of Sergei Garcia's employer?",
             {"kb/person_sergei_garcia.md": "Employer: Klen Lab", "kb/notes.md": "Klen Lab headquarters: Tomsk"}),
            ("data_analysis", "How many units of product 'A' were sold in months 1 through 2 inclusive? "
             "Quantity equals total divided by unit_price; no discounts.",
             {"transactions.csv": "product,month,total,unit_price\nA,1,120,30\nA,2,175,25\n"}),
            ("data_analysis", "How many units of product 'A' were sold in months 1 through 2 inclusive? "
             "item means product, period means month, quantity means units.",
             {"orders.csv": "item,period,quantity\nA,1,4\nA,2,7\n"}),
            ("coding", "Fix mylib.clamp and run python -m unittest discover -s tests.",
             {"mylib/__init__.py": "def clamp(x, lo, hi): return min(lo, max(hi, x))\n",
              "tests/test_clamp.py": "from mylib import clamp\nassert clamp(4,0,10)==4\n"}),
        ]
        for kind, instruction, files in fixtures:
            with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                for name, text in files.items():
                    path = workspace / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                task = {"task_type": kind, "instruction": instruction, "reward": 2}
                self.assertIsNone(impossible_reason(task, workspace))
                self.assertGreater(TaskEstimator().estimate(task, workspace).priority, 0)

    def test_contradiction_check_respects_keywords_and_contract_negation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "mylib.py").write_text("def f(x=0): return x\n", encoding="utf-8")
            tests = workspace / "test_mylib.py"
            tests.write_text("import mylib as m\ndef test(self):\n"
                             "    self.assertEqual(m.f(x=1), 1)\n"
                             "    self.assertEqual(m.f(x=2), 2)\n", encoding="utf-8")
            task = {"task_type": "coding", "instruction": "Functions must be stateless and deterministic."}
            self.assertIsNone(impossible_reason(task, workspace))
            tests.write_text(tests.read_text(encoding="utf-8").replace("x=2", "x=1"), encoding="utf-8")
            task["instruction"] = "No functions must be stateless and deterministic. Stateful repairs are allowed."
            self.assertIsNone(impossible_reason(task, workspace))

    def test_csv_aggregates_do_not_need_model_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "sales.csv").write_text(
                "order_id,region,product,month,units,price\n"
                "1,west,widget,1,3,10\n2,east,widget,1,5,30\n"
                "3,west,widget,2,4,20\n", encoding="utf-8")
            self.assertEqual(csv_answer("What is the average price of widget orders?", workspace), "20")
            self.assertEqual(csv_answer("How many units of widget were sold in the west region?", workspace), "7")

    def test_single_csv_accepts_natural_table_names_without_rewriting_values(self):
        class FakeLLM:
            def __init__(self, query):
                self.query = query
            async def ask(self, *_args, **_kwargs):
                return self.query

        for filename in ("transactions.csv", "Sales.csv", "DATA.csv", "sqlite_sales.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / filename).write_text(
                    "product,total,unit_price\nFROM sales,120,30\nFROM sales,175,25\nother,90,10\n",
                    encoding="utf-8")
                task = {"task_type": "data_analysis", "instruction": "Compute quantity as total/unit_price for product FROM sales."}
                table_names = ["data", "sales", "main"]
                if not Path(filename).stem.casefold().startswith("sqlite_"):
                    table_names.append(Path(filename).stem)
                for table in table_names:
                    query = 'SELECT SUM(total/unit_price) FROM "' + table + '" WHERE product=\'FROM sales\''
                    with self.subTest(table=table):
                        self.assertEqual(asyncio.run(model_answer(FakeLLM(query), task, workspace)), "11")

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
