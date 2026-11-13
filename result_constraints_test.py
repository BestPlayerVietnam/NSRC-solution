"""Soundness/counterexample tests for result_constraints; no fixture is executed."""
import tempfile
import json
import unittest
from pathlib import Path
from result_constraints import analyze_task, _contract_text

INTEGRITY = ("Do not modify tests, builtins, standard libraries, import hooks "
             "or the test framework.")

class ResultConstraintsTests(unittest.TestCase):
    def inspect(self, body, impl="def f(x):\n    return 0\n", extra=None, prefix="", instruction=INTEGRITY):
        with tempfile.TemporaryDirectory(prefix="result_constraints_test_") as name:
            root = Path(name)
            (root / "mylib.py").write_text(impl, encoding="utf-8")
            source = "from mylib import f\n" + prefix + "\ndef test_result():\n"
            source += "\n".join("    " + line for line in body.splitlines()) + "\n"
            (root / "test_mylib.py").write_text(source, encoding="utf-8")
            for path, text in (extra or {}).items():
                (root / path).write_text(text, encoding="utf-8")
            return analyze_task({"task_type": "coding", "instruction": instruction}, root)

    def test_same_builtin_int_zero_and_positive(self):
        r = self.inspect("x = f([])\nassert type(x) is int\nassert x == 0\nassert x > 0")
        self.assertTrue(r.hard_impossible)
        self.assertEqual(r.evidence[0]["lower_inclusive"], 1)
        self.assertEqual(r.evidence[0]["upper_inclusive"], 0)

    def test_strict_integer_gap(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert x > 3\nassert x < 4")
        self.assertTrue(r.hard_impossible)

    def test_reversed_signed_inequalities(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert -2 <= x\nassert -2 > x")
        self.assertTrue(r.hard_impossible)

    def test_finite_domain_excluded_completely(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert x >= -1\nassert x <= 1\nassert x != -1\nassert x != 0\nassert x != 1")
        self.assertTrue(r.hard_impossible)

    def test_nonempty_bounded_interval(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert x >= -1\nassert x <= 1\nassert x != 0")
        self.assertFalse(r.hard_impossible)

    def test_bool_zero_label_is_compatible(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert x == 0\nassert x == False")
        self.assertFalse(r.hard_impossible)

    def test_bool_one_label_is_compatible(self):
        r = self.inspect("x = f(1)\nassert type(x) is int\nassert x == True\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_list_length_not_stable_under_future_implementation(self):
        r = self.inspect("r = f([])\nassert type(r) is list\nassert len(r) == 1\nassert len(r) == 2",
                         "def f(x):\n    return []\n")
        self.assertFalse(r.hard_impossible)

    def test_tuple_negative_length(self):
        r = self.inspect("r = f(0)\nassert type(r) is tuple\nassert len(r) < 0",
                         "def f(x):\n    return ()\n")
        self.assertTrue(r.hard_impossible)

    def test_mutation_between_length_asserts(self):
        r = self.inspect("r = f([])\nassert type(r) is list\nassert len(r) == 1\nr.append(3)\nassert len(r) == 2")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_subscript_mutation_is_not_assumed_pure(self):
        r = self.inspect("r = f([])\nassert type(r) is list\nassert len(r) == 1\nr[:] = [1, 2]\nassert len(r) == 2")
        self.assertFalse(r.hard_impossible)

    def test_separate_calls_are_never_merged(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nx = f(0)\nassert type(x) is int\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_other_call_invalidates_old_result(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\ny = f(1)\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_literal_binding_replaces_old_result(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nx = 1\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_type_guard_required(self):
        r = self.inspect("x = f(0)\nassert x == 0\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_isinstance_is_not_exact_builtin_type(self):
        r = self.inspect("x = f(0)\nassert isinstance(x, int)\nassert x == 0\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_custom_class_not_assumed_builtin(self):
        impl = "class Odd(int):\n    pass\n\ndef f(x):\n    return Odd(0)\n"
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0", impl)
        self.assertFalse(r.hard_impossible)

    def test_builtin_shadowing(self):
        r = self.inspect("int = 0\nx = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_custom_len_binding(self):
        r = self.inspect("r = f(0)\nassert type(r) is list\nassert len(r) == 1\nassert len(r) == 2",
                         prefix="def len(value):\n    return 1\n")
        self.assertFalse(r.hard_impossible)

    def test_builtin_monkeypatch_import(self):
        impl = "import builtins\n\ndef f(x):\n    builtins.int = str\n    return 0\n"
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0", impl)
        self.assertFalse(r.hard_impossible)

    def test_global_mutation_is_unknown(self):
        impl = "counter = 0\n\ndef f(x):\n    global counter\n    counter += 1\n    return counter\n"
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0", impl)
        self.assertFalse(r.hard_impossible)

    def test_fixture_conftest_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         extra={"conftest.py": "# Potential autouse fixture\n"})
        self.assertFalse(r.hard_impossible)

    def test_unknown_import_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         impl="import package_not_available\n\ndef f(x):\n    return 0\n")
        self.assertFalse(r.hard_impossible)

    def test_dead_branch_is_not_proof(self):
        r = self.inspect("x = f(0)\nif False:\n    assert type(x) is int\n    assert x == 0\n    assert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_assert_message_effects_break_block(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0, f(1)\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_float_semantics_outside_subset(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0.0\nassert x > 0")
        self.assertFalse(r.hard_impossible)

    def test_simple_import_alias_is_supported(self):
        with tempfile.TemporaryDirectory(prefix="result_constraint_alias_") as name:
            root = Path(name)
            (root / "mylib.py").write_text("def f(x):\n    return 0\n", encoding="utf-8")
            (root / "test_mylib.py").write_text(
                "import mylib as m\n\ndef test_result():\n    x = m.f(1)\n    assert type(x) is int\n    assert x == 0\n    assert x > 0\n",
                encoding="utf-8")
            self.assertTrue(analyze_task({"task_type": "coding", "instruction": INTEGRITY}, root).hard_impossible)

    def test_file_budget_returns_unknown(self):
        with tempfile.TemporaryDirectory(prefix="result_constraint_bound_") as name:
            root = Path(name)
            (root / "test_x.py").write_text("pass\n", encoding="utf-8")
            self.assertFalse(analyze_task({"task_type": "coding", "instruction": INTEGRITY}, root,
                                         max_total_bytes=1).hard_impossible)


    def test_pytest_plugin_global_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         prefix="pytest_plugins = ['unknown_plugin']\n")
        self.assertFalse(r.hard_impossible)

    def test_pytest_configuration_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         extra={"pytest.ini": "[pytest]\naddopts = -p unknown_plugin\n"})
        self.assertFalse(r.hard_impossible)

    def test_train_single_list_result_regression(self):
        # Explicitly authorized TRAIN fixture only; no manifest/cal/test/public100.
        task_dir = (Path(__file__).resolve().parent / "experiments" / "suite_policy"
                    / "tasks" / "policy_62ab3bfa6706510a")
        if not task_dir.is_dir():
            self.skipTest("Optional local TRAIN fixture is unavailable")
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        r = analyze_task(task, task_dir / "workspace")
        # A future implementation can mutate an exact list asynchronously.
        # The existing TRAIN wording is not a result-stability contract.
        self.assertFalse(r.hard_impossible, r)


    def test_missing_integrity_contract_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         instruction="Implement the function.")
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_quoted_integrity_sentence_is_not_contract(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         instruction='Example: "' + INTEGRITY + '"')
        self.assertFalse(r.hard_impossible)


    def test_quoted_multisentence_paragraph_is_not_an_integrity_contract(self):
        for opener, closer in (('"', '"'), ('“', '”'), ('‘', '’'), ('«', '»')):
            instruction = 'Documentation quotes: ' + opener + 'Example. ' + INTEGRITY + ' End.' + closer
            result = self.inspect("x=f(0)\nassert type(x) is int\nassert x==0\nassert x>0", instruction=instruction)
            self.assertFalse(result.hard_impossible)
            self.assertIn('incomplete_analysis', result.flags)

    def test_quote_removal_cannot_assemble_a_different_normative_sentence(self):
        text = _contract_text("All functions must 'not' be stateless and deterministic.")
        self.assertIn('[quoted content]', text)
        instruction = INTEGRITY.replace('Do not modify', "Do not 'always' modify")
        result = self.inspect("x=f(0)\nassert type(x) is int\nassert x==0\nassert x>0", instruction=instruction)
        self.assertFalse(result.hard_impossible)
        self.assertIn('incomplete_analysis', result.flags)

    def test_overridden_test_definition_is_unknown(self):
        old = ("def test_result():\n    x = f(0)\n    assert type(x) is int\n"
               "    assert x < 0\n    assert x >= 0\n")
        r = self.inspect("assert f(0) == 0", prefix=old)
        self.assertFalse(r.hard_impossible)
        self.assertIn("incomplete_analysis", r.flags)

    def test_import_and_definition_binding_collision_is_unknown(self):
        r = self.inspect("x = f(0)\nassert type(x) is int\nassert x == 0\nassert x > 0",
                         prefix="from mylib import f as test_result\n")
        self.assertFalse(r.hard_impossible)

if __name__ == "__main__":
    unittest.main()
