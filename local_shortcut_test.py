"""Local shortcuts must understand the complete question or defer to the model."""
import tempfile
import unittest
from pathlib import Path

from run_agent import csv_answer, search_answer


TAIL = ("Work only inside the current directory. When you are done, write ONLY "
        "the final answer to the file answer.txt in the current directory "
        "(for example with bash: echo 42 > answer.txt).")


class CompleteLocalQuestions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "sales.csv").write_text(
            "product,region,month,units,price,returned\n"
            "widget,East,7,5,10,no\nwidget,East,8,3,20,yes\n"
            "widget,West,9,4,30,no\n", encoding="utf-8")
        self.kb = self.root / "kb"
        self.kb.mkdir()
        self.person = self.kb / "person_jane_doe.md"
        self.person.write_text("Name: Jane Doe\nBorn: 1980\nEmployer: Example Labs\n", encoding="utf-8")
        self.company = self.kb / "company_example_labs.md"
        self.company.write_text("Headquarters: Novosibirsk\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_simple_csv_queries(self):
        for question, answer in [
            ("How many units of product 'widget' were sold in months 7 through 8 inclusive?", "8"),
            ("How many units of widget were sold in the East region?", "8"),
            ("What is the average price of widget orders?", "20"),
        ]:
            with self.subTest(question=question):
                self.assertEqual(csv_answer(question, self.root), answer)

    def test_official_csv_boilerplate(self):
        question = ("The file sales.csv has columns product,region,month,units,price,returned (3 rows). "
                    "How many units of product 'widget' were sold in months 7 through 8 inclusive? "
                    "Use bash with python3 to compute it. " + TAIL)
        self.assertEqual(csv_answer(question, self.root), "8")

    def test_csv_extra_filter_is_not_ignored(self):
        question = "How many units of widget were sold in the East region?"
        for extra in ["Exclude returned orders.", "Count only month 7.", "Then double that number."]:
            for extended in [question + " " + extra, extra + " " + question]:
                with self.subTest(question=extended):
                    self.assertIsNone(csv_answer(extended, self.root))

    def test_csv_unknown_tool_suffix_cannot_hide_semantics(self):
        question = ("How many units of widget were sold in the East region? "
                    "Use bash with python3 to compute it, excluding returned orders. " + TAIL)
        self.assertIsNone(csv_answer(question, self.root))

    def test_csv_declared_file_must_match(self):
        question = ("The file refunds.csv has columns product,region,units (3 rows). "
                    "How many units of widget were sold in the East region?")
        self.assertIsNone(csv_answer(question, self.root))

    def test_search_simple_questions(self):
        self.assertEqual(search_answer("In which year was Jane Doe born?", self.root), "1980")
        self.assertEqual(search_answer("In which city is the headquarters of Jane Doe's employer?",
                                       self.root), "Novosibirsk")

    def test_official_search_boilerplate(self):
        question = ("The directory kb/ contains a small knowledge base of markdown files. "
                    "In which year was Jane Doe born? "
                    "Use the file tools or bash (ls, grep, cat) to find it. " + TAIL)
        self.assertEqual(search_answer(question, self.root), "1980")

    def test_search_extra_operations_are_not_ignored(self):
        question = "In which year was Jane Doe born?"
        for extra in ["Add ten to the year.", "Return her employer's founding year instead."]:
            for extended in [question + " " + extra, extra + " " + question]:
                with self.subTest(question=extended):
                    self.assertIsNone(search_answer(extended, self.root))

    def test_search_distinct_birth_years_defer(self):
        self.person.write_text("Born: 1980\nBorn: 1981\n", encoding="utf-8")
        self.assertIsNone(search_answer("In which year was Jane Doe born?", self.root))

    def test_search_duplicate_identical_year_is_consistent(self):
        self.person.write_text("Born: 1980\nBorn: 1980\n", encoding="utf-8")
        self.assertEqual(search_answer("In which year was Jane Doe born?", self.root), "1980")

    def test_search_multiple_employers_defer(self):
        self.person.write_text("Employer: Example Labs\nEmployer: Other Labs\n", encoding="utf-8")
        self.assertIsNone(search_answer("In which city is the headquarters of Jane Doe's employer?", self.root))

    def test_search_conflicting_headquarters_defer(self):
        self.company.write_text("Headquarters: Novosibirsk\nHeadquarters: Tomsk\n", encoding="utf-8")
        self.assertIsNone(search_answer("In which city is the headquarters of Jane Doe's employer?", self.root))


if __name__ == "__main__":
    unittest.main()
