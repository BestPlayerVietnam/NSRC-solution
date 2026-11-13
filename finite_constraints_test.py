"""Independent witnesses and guard counterexamples; no fixture code is executed."""
import itertools
import tempfile
import unittest
from pathlib import Path

from finite_constraints import analyze_task
from result_constraints import analyze_task as legacy_analyze

OLD_CONTRACT = 'Do not modify tests, builtins, standard libraries, import hooks or the test framework.'
PUBLIC_CONTRACT = ('All conditions apply to the same returned object, using ordinary Python built-in semantics. '
                   'Modifying tests, built-ins, globals or the test framework is forbidden.')
STR_DOMAIN = "y = f()\nassert type(y) is str\nassert len(y)==1 and ('a'<=y<='z' or 'A'<=y<='Z')\n"
STR_BAD = STR_DOMAIN + "assert y.lower()=='a'\nassert y.upper()=='B'"
BYTES_BAD = "y=f()\nassert type(y) is bytes\nassert len(y)==1\nassert len(y.hex())==1"
TUPLE_DOMAIN = ('y=f()\nassert type(y) is tuple and len(y)==3\n'
                'assert all(type(v) is int for v in y)\nassert set(y)=={0,1,2}\n')
TUPLE_BAD = TUPLE_DOMAIN + ('assert all(y[y[y[y[i]]]]==i for i in range(3))\n'
                          'assert any(y[y[i]]!=i for i in range(3))')


class FiniteConstraintTests(unittest.TestCase):
    def inspect(self, body, *, implementation='def f():\n    return None\n',
                extra=None, prefix='', instruction=PUBLIC_CONTRACT, checker=analyze_task, **limits):
        with tempfile.TemporaryDirectory(prefix='finite-constraint-test-') as name:
            root = Path(name)
            (root / 'mylib.py').write_text(implementation, encoding='utf-8')
            source = 'from mylib import f\n' + prefix + '\ndef test_contract():\n'
            source += '\n'.join('    ' + line for line in body.splitlines()) + '\n'
            (root / 'test_mylib.py').write_text(source, encoding='utf-8')
            for relative, content in (extra or {}).items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            return checker({'task_type': 'coding', 'instruction': instruction}, root, **limits)

    def assert_unknown(self, result):
        self.assertFalse(result.hard_impossible, result)
        self.assertIn('incomplete_analysis', result.flags)

    def test_ascii_lower_upper_contradiction_exhausts_52_candidates(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.assertFalse(any(c.lower() == 'a' and c.upper() == 'B' for c in alphabet))
        result = self.inspect(STR_BAD)
        self.assertTrue(result.hard_impossible, result)
        self.assertEqual(result.evidence[0]['domain_size'], 52)
        self.assertEqual(result.evidence[0]['exhausted_count'], 52)
        self.assertEqual(result.evidence[0]['builtin_type'], 'str')

    def test_different_ascii_parameters_use_same_finite_proof(self):
        body = STR_DOMAIN + "assert y.lower()=='m'\nassert y.upper()=='N'"
        self.assertTrue(self.inspect(body).hard_impossible)
        witness = 'm'
        self.assertEqual(witness.lower(), 'm')
        self.assertEqual(witness.upper(), 'M')
        self.assertFalse(self.inspect(body.replace("=='N'", "=='M'")).hard_impossible)

    def test_bytes_hex_contradiction_exhausts_256_candidates(self):
        self.assertTrue(all(len(bytes([value]).hex()) == 2 for value in range(256)))
        result = self.inspect(BYTES_BAD)
        self.assertTrue(result.hard_impossible, result)
        self.assertEqual(result.evidence[0]['domain_size'], 256)
        self.assertEqual(result.evidence[0]['exhausted_count'], 256)

    def test_zero_length_bytes_and_satisfiable_hex_variants(self):
        self.assertEqual(b''.hex(), '')
        bad = BYTES_BAD.replace('len(y)==1', 'len(y)==0')
        self.assertTrue(self.inspect(bad).hard_impossible)
        self.assertFalse(self.inspect(bad.replace('len(y.hex())==1', 'len(y.hex())==0')).hard_impossible)
        self.assertFalse(self.inspect(BYTES_BAD.replace('len(y.hex())==1', 'len(y.hex())==2')).hard_impossible)

    def test_tuple_permutation_contradiction_exhausts_six(self):
        for candidate in itertools.permutations((0, 1, 2)):
            fourth = all(candidate[candidate[candidate[candidate[i]]]] == i for i in range(3))
            not_second = any(candidate[candidate[i]] != i for i in range(3))
            self.assertFalse(fourth and not_second)
        result = self.inspect(TUPLE_BAD)
        self.assertTrue(result.hard_impossible, result)
        self.assertEqual(result.evidence[0]['domain_size'], 6)
        self.assertEqual(result.evidence[0]['exhausted_count'], 6)

    def test_four_elements_can_satisfy_fourth_but_not_second_identity(self):
        candidate = (1, 2, 3, 0)
        self.assertTrue(all(candidate[candidate[candidate[candidate[i]]]] == i for i in range(4)))
        self.assertTrue(any(candidate[candidate[i]] != i for i in range(4)))
        body = TUPLE_BAD.replace('len(y)==3', 'len(y)==4').replace('{0,1,2}', '{0,1,2,3}').replace('range(3)', 'range(4)')
        self.assertFalse(self.inspect(body).hard_impossible)

    def test_fifth_power_nonidentity_impossible_on_four_elements(self):
        body = ('y=f()\nassert type(y) is tuple and len(y)==4\n'
                'assert all(type(v) is int for v in y)\nassert set(y)=={0,1,2,3}\n'
                'assert all(y[y[y[y[y[i]]]]]==i for i in range(4))\n'
                'assert any(y[i]!=i for i in range(4))')
        self.assertTrue(self.inspect(body).hard_impossible)
        positive = body.replace('len(y)==4', 'len(y)==5').replace('{0,1,2,3}', '{0,1,2,3,4}').replace('range(4)', 'range(5)')
        self.assertFalse(self.inspect(positive).hard_impossible)

    def test_builtin_int_zero_is_equal_to_false(self):
        body = TUPLE_DOMAIN + 'assert y[0]==False'
        self.assertTrue((0, 1, 2)[0] == False)
        self.assertFalse(self.inspect(body).hard_impossible)

    def test_unicode_expansion_is_a_valid_finite_witness(self):
        body = ("y=f()\nassert type(y) is str\nassert len(y)==1 and y=='ß'\n"
                "assert y.upper()=='SS'\nassert y.lower()=='ß'")
        self.assertEqual('ß'.upper(), 'SS')
        self.assertFalse(self.inspect(body).hard_impossible)

    def test_or_union_includes_unicode_branch(self):
        body = ("y=f()\nassert type(y) is str\nassert len(y)==1\n"
                "assert 'a'<=y<='z' or y=='ß'\nassert y.upper()=='SS'")
        self.assertFalse(self.inspect(body).hard_impossible)

    def test_or_with_unbounded_branch_never_assumes_ascii(self):
        body = ("y=f()\nassert type(y) is str\nassert len(y)==1\n"
                "assert 'a'<=y<='z' or True\nassert y.upper()=='SS'")
        self.assert_unknown(self.inspect(body))

    def test_len_one_and_isalpha_do_not_bound_unicode(self):
        body = "y=f()\nassert type(y) is str\nassert len(y)==1\nassert y.isalpha()\nassert y.upper()=='SS'"
        self.assert_unknown(self.inspect(body))

    def test_bounds_on_lower_method_are_not_bounds_on_original_character(self):
        body = "y=f()\nassert type(y) is str\nassert len(y)==1\nassert 'a'<=y.lower()<='z'\nassert y.upper()=='SS'"
        self.assert_unknown(self.inspect(body))

    def test_isinstance_and_missing_type_guard_are_unknown(self):
        for body in (STR_BAD.replace('type(y) is str', 'isinstance(y, str)'),
                     BYTES_BAD.replace('assert type(y) is bytes\n', ''),
                     TUPLE_BAD.replace('type(y) is tuple', 'isinstance(y, tuple)')):
            with self.subTest(body=body): self.assert_unknown(self.inspect(body))

    def test_subclass_implementation_and_custom_methods_are_unknown(self):
        implementation = "class Custom(str):\n    def upper(self): return 'B'\ndef f(): return Custom('a')\n"
        self.assert_unknown(self.inspect(STR_BAD, implementation=implementation))

    def test_mutable_list_never_produces_finite_certificate(self):
        self.assert_unknown(self.inspect(TUPLE_BAD.replace('type(y) is tuple', 'type(y) is list')))

    def test_missing_exact_element_type_does_not_enumerate_only_builtin_ints(self):
        body = TUPLE_BAD.replace('assert all(type(v) is int for v in y)\n', '')
        self.assert_unknown(self.inspect(body))

    def test_subset_domain_does_not_mean_permutations(self):
        body = TUPLE_BAD.replace('set(y)=={0,1,2}', 'set(y)<={0,1,2}')
        self.assert_unknown(self.inspect(body))

    def test_wrong_length_does_not_mean_permutations(self):
        self.assert_unknown(self.inspect(TUPLE_BAD.replace('len(y)==3', 'len(y)==4')))

    def test_bool_or_float_set_literals_are_conservatively_unknown(self):
        for literal in ('{False,True,2}', '{0.0,1,2}'):
            with self.subTest(literal=literal):
                self.assert_unknown(self.inspect(TUPLE_BAD.replace('{0,1,2}', literal)))

    def test_arbitrary_identity_is_not_a_value_property(self):
        for extra in ('assert y is "a"', 'assert y is not "a"'):
            with self.subTest(extra=extra): self.assert_unknown(self.inspect(STR_BAD + '\n' + extra))
        self.assert_unknown(self.inspect(TUPLE_BAD + '\nassert y[0] is 257'))

    def test_named_expression_and_generator_rebinding_are_unknown(self):
        self.assert_unknown(self.inspect(STR_BAD + '\nassert (y := "a")'))
        self.assert_unknown(self.inspect(TUPLE_BAD.replace('type(v) is int for v in y', 'type(y) is int for y in y')))

    def test_filtered_generator_cannot_prove_all_element_types(self):
        self.assert_unknown(self.inspect(TUPLE_BAD.replace('for v in y)', 'for v in y if v>=0)')))

    def test_mutation_unknown_call_and_controlflow_break_linear_block(self):
        for inserted in ('y = "a"', 'f()', 'y.append(1)', 'if False:\n    assert False'):
            with self.subTest(inserted=inserted):
                self.assert_unknown(self.inspect(STR_BAD.replace("assert y.upper()=='B'", inserted + "\nassert y.upper()=='B'")))

    def test_separate_calls_are_not_merged(self):
        body = STR_DOMAIN + "assert y.lower()=='a'\ny=f()\nassert y.upper()=='B'"
        self.assert_unknown(self.inspect(body))

    def test_stateful_source_and_framework_imports_are_unknown(self):
        for implementation in ('counter=0\ndef f():\n    global counter\n    counter+=1\n    return str(counter)\n',
                               'import builtins\ndef f(): return None\n',
                               'import pytest\ndef f(): return None\n'):
            with self.subTest(implementation=implementation):
                self.assert_unknown(self.inspect(STR_BAD, implementation=implementation))

    def test_builtin_shadowing_including_bytes_is_unknown(self):
        for prefix in ('bytes = str\n', 'bytes = 1\n', 'str = 1\n',
                       'def len(value): return 1\n', 'from mylib import f as bytes\n'):
            with self.subTest(prefix=prefix): self.assert_unknown(self.inspect(BYTES_BAD, prefix=prefix))

    def test_shadowed_generator_builtin_is_unknown(self):
        self.assert_unknown(self.inspect(TUPLE_BAD.replace('type(v) is int for v in y', 'type(int) is int for int in y')))

    def test_fixtures_configuration_hidden_source_and_module_collision_are_unknown(self):
        extras = ({'conftest.py': ''}, {'pytest.ini': '[pytest]\n'}, {'test_hidden/.hidden/part.py': ''},
                  {'mylib/__init__.py': 'def f(): return None\n'}, {'pytest.pyc': 'opaque'},
                  {'sitecustomize.py': '# startup hook\n'}, {'extension.pyd': 'opaque'})
        for extra in extras:
            with self.subTest(extra=extra): self.assert_unknown(self.inspect(STR_BAD, extra=extra))

    def test_fixture_argument_and_duplicate_test_binding_are_unknown(self):
        prefix = 'def test_contract():\n    assert True\n'
        self.assert_unknown(self.inspect(STR_BAD, prefix=prefix))
        extra = {'test_fixture.py': 'def test_fixture(monkeypatch):\n    assert True\n'}
        # An unrelated fixture could mutate global bindings before our test.
        self.assert_unknown(self.inspect(STR_BAD, extra=extra))

    def test_assert_message_call_and_untrusted_method_are_unknown(self):
        self.assert_unknown(self.inspect(STR_BAD.replace("assert y.lower()=='a'", "assert y.lower()=='a', f()")))
        self.assert_unknown(self.inspect(STR_BAD + "\nassert y.replace('a','b')=='b'"))

    def test_unsupported_short_circuit_branch_is_still_unknown(self):
        self.assert_unknown(self.inspect(STR_BAD + '\nassert True or unknown_callback()'))

    def test_limits_never_count_as_candidate_failure(self):
        self.assert_unknown(self.inspect(STR_BAD, max_candidates=51))
        self.assert_unknown(self.inspect(BYTES_BAD, max_operations=1))
        huge = TUPLE_DOMAIN + 'assert all(i>=0 for i in range(1000000))'
        self.assert_unknown(self.inspect(huge))
        self.assert_unknown(self.inspect(STR_BAD, max_total_bytes=1))

    def test_unsupported_arithmetic_cannot_allocate_large_values(self):
        self.assert_unknown(self.inspect(STR_BAD + "\nassert y * 1000000000 == 'a'"))

    def test_integrity_contracts_and_quoted_examples(self):
        self.assertTrue(self.inspect(STR_BAD, instruction=OLD_CONTRACT).hard_impossible)
        for instruction in ('Implement the function.', 'Example: "' + PUBLIC_CONTRACT + '"',
                            chr(96)*3 + PUBLIC_CONTRACT + chr(96)*3,
                            'Modifying tests, built-ins, globals or the test framework is forbidden.'):
            with self.subTest(instruction=instruction): self.assert_unknown(self.inspect(STR_BAD, instruction=instruction))

    def test_integrity_sentences_inside_multisentence_quotes_are_not_normative(self):
        quoted_old = 'Example: "This is quoted prose. ' + OLD_CONTRACT + '"'
        quoted_public = 'Example: "This is quoted prose. ' + PUBLIC_CONTRACT + '"'
        for instruction in (quoted_old, quoted_public, '"Unclosed quotation. ' + PUBLIC_CONTRACT):
            with self.subTest(instruction=instruction):
                self.assert_unknown(self.inspect(STR_BAD, instruction=instruction))

    def test_unicode_quotes_blockquotes_and_contract_exceptions_are_unknown(self):
        instructions = [
            'Example: “This is quoted prose. ' + PUBLIC_CONTRACT + '”',
            'Example: ‘This is quoted prose. ' + PUBLIC_CONTRACT + '’',
            '> Example. ' + PUBLIC_CONTRACT,
            'Example: «Text. ' + PUBLIC_CONTRACT + '»',
            chr(96)*2 + 'Text. ' + PUBLIC_CONTRACT + chr(96)*2,
            '<code>Text. ' + PUBLIC_CONTRACT + '</code>',
            '<!--Text. ' + PUBLIC_CONTRACT + '-->',
            'Example\n\n    ' + PUBLIC_CONTRACT,
            PUBLIC_CONTRACT + ' Except when impossible, modifying builtins is allowed.',
        ]
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                self.assert_unknown(self.inspect(STR_BAD, instruction=instruction))

    def test_candidate_limit_is_task_wide_across_supported_blocks(self):
        positive = "y=f()\nassert type(y) is bytes\nassert len(y)==1\nassert y.hex()=='ff'"
        prefix = ''
        for name in ('test_first', 'test_second'):
            prefix += 'def ' + name + '():\n' + '\n'.join('    ' + line for line in positive.splitlines()) + '\n'
        report = self.inspect(BYTES_BAD, prefix=prefix, max_candidates=512)
        self.assert_unknown(report)
        self.assertIn('Task-wide finite candidate limit', report.evidence[-1]['detail'])
        self.assertEqual(sum(item.get('exhausted_count', 0) for item in report.evidence), 512)

    def test_input_function_is_never_executed(self):
        result = self.inspect(STR_BAD, implementation="def f():\n    raise AssertionError('must never execute')\n")
        self.assertTrue(result.hard_impossible, result)

    def test_legacy_method_policy_stays_unchanged(self):
        self.assert_unknown(self.inspect(STR_BAD, instruction=OLD_CONTRACT, checker=legacy_analyze))


if __name__ == '__main__':
    unittest.main()