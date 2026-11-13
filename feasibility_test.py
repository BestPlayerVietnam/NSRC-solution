"""Behavioral counterexamples for AST-only feasibility; fixtures are never run."""
import tempfile
import unittest
from pathlib import Path
from feasibility import analyze_task

CONTRACT = ("All functions must be stateless and deterministic. "
            "All functions must return a built-in int.")
DEFAULT = "def f(x, **kwargs):\n    return 0\n"

class FeasibilityTests(unittest.TestCase):
    def analyze(self, tests, instruction=CONTRACT, sources=None, **limits):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {"mylib.py": DEFAULT, "test_main.py": tests}
            files.update(sources or {})
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return analyze_task({"task_type": "coding", "instruction": instruction}, root, **limits)

    def test_plain_pytest_same_input_is_proved(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(3)==10\n assert f(3)==11\n")
        self.assertTrue(r.hard_impossible)
        self.assertEqual(r.retry_hint, "formal_skip")
        self.assertIn("conflicting_outputs", r.flags)
        self.assertEqual({e["expected"] for e in r.evidence}, {10, 11})

    def test_unittest_alias_and_separate_methods(self):
        r = self.analyze("import unittest as ut\nimport mylib as m\n"
                         "class Checks(ut.TestCase):\n"
                         " def test_a(self):\n  self.assertEqual(m.f(5),7)\n"
                         " def test_b(self):\n  self.assertEqual(8,m.f(5))\n")
        self.assertTrue(r.hard_impossible)

    def test_pytest_parameterized_immutable_cases(self):
        r = self.analyze("import pytest as p\nfrom mylib import f as run\n"
                         "@p.mark.parametrize('x, expected', [(3, 8), (3, 9)])\n"
                         "def test_cases(x, expected):\n assert run(x)==expected\n")
        self.assertTrue(r.hard_impossible)

    def test_local_alias_and_result_variable(self):
        r = self.analyze("from mylib import f\n"
                         "def test_f():\n run=f\n x=(1,'a')\n y=run(x)\n assert y==7\n assert y==8\n")
        self.assertTrue(r.hard_impossible)

    def test_multifile_reexport_alias_resolves_to_same_callable(self):
        r = self.analyze("from api import perform\n"
                         "def test_a():\n assert perform(2)==3\n",
                         sources={"api.py": "from mylib import f as perform\n",
                                  "test_more.py": "import mylib as m\ndef test_b():\n assert m.f(2)==4\n"})
        self.assertTrue(r.hard_impossible)

    def test_relative_package_reexport(self):
        r = self.analyze("from pkg import f\nfrom pkg.impl import f as g\n"
                         "def test_f():\n assert f(1)==1\n assert g(1)==2\n",
                         sources={"pkg/__init__.py":"from .impl import f\n",
                                  "pkg/impl.py":"def f(x):\n return 0\n"})
        self.assertTrue(r.hard_impossible)

    def test_contract_for_named_function(self):
        instruction = "Function f must be stateless and deterministic. Function f must return a built-in int."
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==2\n assert f(1)==3\n", instruction)
        self.assertTrue(r.hard_impossible)

    def test_no_int_contract_custom_equality_only_soft(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==2\n assert f(1)==3\n",
                         "All functions must be stateless and deterministic.",
                         {"mylib.py": "class AnyValue:\n def __eq__(self, other):\n  return True\n"
                                      "def f(x):\n return AnyValue()\n"})
        self.assertFalse(r.hard_impossible)
        self.assertIn("conflicting_outputs", r.flags)

    def test_explicit_stateful_is_never_hard(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==2\n assert f(1)==3\n",
                         "A stateful counter is allowed. Return a built-in int.")
        self.assertFalse(r.hard_impossible)
        self.assertIn("explicit_stateful", r.flags)

    def test_absent_determinism_contract_is_soft(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==2\n assert f(1)==3\n",
                         "All functions must return a built-in int.")
        self.assertFalse(r.hard_impossible)
        self.assertIn("conflicting_outputs", r.flags)

    def test_mutable_arguments_are_not_conflated(self):
        r = self.analyze("from mylib import f\ndef test_f():\n x=[1]\n assert f(x)==2\n"
                         " x.append(3)\n assert f(x)==5\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("mutable_arguments", r.flags)
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_bool_int_float_and_signed_zero_inputs_stay_distinct(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " assert f(True)==1\n assert f(1)==2\n assert f(1.0)==3\n"
                         " assert f(0.0)==4\n assert f(-0.0)==5\n")
        self.assertFalse(r.hard_impossible)
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_equivalent_numeric_labels_are_not_a_conflict(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " assert f(1)==True\n assert f(1)==1\n assert f(1)==1.0\n")
        self.assertFalse(r.hard_impossible)
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_kwargs_order_and_call_style_are_observable(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " assert f(x=1,y=2)==1\n assert f(y=2,x=1)==2\n"
                         " assert f(1)==3\n assert f(x=1)==4\n")
        self.assertFalse(r.hard_impossible)
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_identical_kwargs_conflict_is_supported(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " assert f(x=1,y=True)==2\n assert f(x=1,y=True)==3\n")
        self.assertTrue(r.hard_impossible)

    def test_different_imported_functions_do_not_merge(self):
        r = self.analyze("from mylib import f as a\nfrom other import f as b\n"
                         "def test_f():\n assert a(1)==1\n assert b(1)==2\n",
                         sources={"other.py": DEFAULT})
        self.assertNotIn("conflicting_outputs", r.flags)
        self.assertFalse(r.hard_impossible)

    def test_rebinding_export_does_not_merge_distinct_functions(self):
        r = self.analyze("from mylib import old, f\n"
                         "def test_f():\n assert old(1)==1\n assert f(1)==2\n",
                         sources={"mylib.py": "def f(x):\n return 1\nold=f\n"
                                             "def g(x):\n return 2\nf=g\n"})
        self.assertNotIn("conflicting_outputs", r.flags)
        self.assertFalse(r.hard_impossible)

    def test_mutually_exclusive_branches_are_not_merged(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " if True:\n  assert f(1)==1\n else:\n  assert f(1)==2\n")
        self.assertFalse(r.hard_impossible)
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_unreachable_assertions_are_not_collected(self):
        r = self.analyze("from mylib import f\ndef test_f():\n"
                         " assert f(1)==1\n return\n assert f(1)==2\n")
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_shadowed_test_definition_is_not_collected(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==1\n"
                         "def test_f():\n assert f(1)==2\n")
        self.assertNotIn("conflicting_outputs", r.flags)

    def test_skipped_parameter_cases_do_not_become_proof(self):
        r = self.analyze("import pytest\nfrom mylib import f\n"
                         "@pytest.mark.parametrize('x,y', [(1,2),(1,3)], indirect=True)\n"
                         "def test_f(x,y):\n assert f(x)==y\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_fixture_mutation_and_unittest_overrides_disable_proof(self):
        r = self.analyze("import unittest\nfrom mylib import f\n"
                         "class T(unittest.TestCase):\n"
                         " def assertEqual(self,a,b):\n  pass\n"
                         " def test_f(self):\n  self.assertEqual(f(1),2)\n  self.assertEqual(f(1),3)\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_quoted_contract_is_not_authority(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==1\n assert f(1)==2\n",
                         'Example: "All functions must be stateless and deterministic." '
                         '"All functions must return a built-in int."')
        self.assertFalse(r.hard_impossible)

    def test_contract_inside_quoted_paragraph_allows_a_stateful_solution(self):
        instruction = ('The documentation quotes this passage: "Example. '
                       'All functions must be stateless and deterministic. '
                       'All functions must return built-in integers. End." '
                       'Implement f to satisfy the tests.')
        result = self.analyze(
            "from mylib import f\ndef test_f():\n assert f(1)==0\n assert f(1)==1\n",
            instruction)
        values = iter((0, 1))
        implementation = lambda value: next(values)
        self.assertEqual((implementation(1), implementation(1)), (0, 1))
        self.assertFalse(result.hard_impossible)

    def test_real_contract_after_a_quoted_example_still_proves_conflict(self):
        instruction = ('The documentation quotes "Example. Return anything. End."\n'
                       + CONTRACT)
        result = self.analyze(
            "from mylib import f\ndef test_f():\n assert f(1)==0\n assert f(1)==1\n",
            instruction)
        self.assertTrue(result.hard_impossible)

    def test_missing_evidence_is_soft(self):
        r = self.analyze("pass\n", "Repair absent.py using table.csv.")
        self.assertFalse(r.hard_impossible)
        self.assertIn("missing_evidence", r.flags)

    def test_resource_limits_disable_hard_conclusion(self):
        r = self.analyze("from mylib import f\ndef test_f():\n assert f(1)==1\n assert f(1)==2\n",
                         max_total_bytes=10)
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_task_code_is_never_executed(self):
        r = self.analyze("raise RuntimeError('MUST NOT RUN')\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_import_time_monkeypatch_disables_formal_proof(self):
        r = self.analyze("from mylib import f as old\nimport switcher\nfrom mylib import f as new\n"
                         "def test_f():\n assert old(1)==1\n assert new(1)==2\n",
                         sources={"switcher.py":"import mylib\ndef alternate(x):\n return 2\nmylib.f=alternate\n"})
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_unknown_import_side_effects_disable_formal_proof(self):
        r = self.analyze("from mylib import f\nimport unknown_plugin\n"
                         "def test_f():\n assert f(1)==1\n assert f(1)==2\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_dynamic_parametrize_ids_are_not_executed_or_trusted(self):
        r = self.analyze("import pytest\nfrom mylib import f\n"
                         "@pytest.mark.parametrize('x,y',[(1,2),(1,3)],ids=lambda v: str(v))\n"
                         "def test_f(x,y):\n assert f(x)==y\n")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

if __name__ == "__main__":
    unittest.main()

