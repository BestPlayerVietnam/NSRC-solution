"""SQL feedback survives outer retries, without leaking between tasks; no API."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from planner import MODELS
from run_agent import model_answer


class FakeLLM:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []
        self.last_finish_reason = 'stop'

    async def ask(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        return next(self.answers)


class SQLRetryContextTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.make_workspace('first')
        self.task = {'task_type': 'data_analysis', 'instruction': 'Total value?'}

    def make_workspace(self, name):
        workspace = self.root / name
        workspace.mkdir()
        (workspace / 'values.csv').write_text('value\n2\n3\n4\n', encoding='utf-8')
        return workspace

    def call(self, llm, *, task=None, workspace=None, model=None):
        return asyncio.run(model_answer(llm, self.task if task is None else task,
                                       self.workspace if workspace is None else workspace,
                                       MODELS['large_reason'] if model is None else model))

    def test_reasoning_outer_retry_receives_query_and_actual_database_error(self):
        llm = FakeLLM('SELECT SUM(missing) FROM data', 'SELECT SUM(value) FROM data')
        self.assertIsNone(self.call(llm))
        self.assertEqual(llm.task_state, 'check_failed')
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(self.call(llm), '9')
        self.assertIn('Previous query: SELECT SUM(missing) FROM data', llm.calls[1][1])
        self.assertIn('Execution error: SQLite: no such column: missing', llm.calls[1][1])
        self.assertNotIn('Previous query:', llm.calls[0][1])
        self.assertEqual(llm.calls[0][2]['max_tokens'], 3072)
        self.assertEqual(llm.calls[1][2]['max_tokens'], 3072)

    def test_plain_outer_retry_detects_failed_query_seen_in_earlier_inner_attempt(self):
        first = 'SELECT first_missing FROM data'
        second = 'SELECT second_missing FROM data'
        llm = FakeLLM(first, second, first)
        self.assertIsNone(self.call(llm, model=MODELS['large_plain']))
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('first_missing', llm.calls[1][1])
        self.assertIsNone(self.call(llm, model=MODELS['large_plain']))
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(llm.task_state, 'stalled')
        self.assertIn('Previous query: ' + second, llm.calls[2][1])
        self.assertIn('no such column: second_missing', llm.calls[2][1])

    def test_different_workspace_resets_feedback_and_seen_query_set(self):
        bad = 'SELECT absent FROM data'
        llm = FakeLLM(bad, bad)
        self.assertIsNone(self.call(llm))
        self.assertIsNone(self.call(llm, workspace=self.make_workspace('second')))
        self.assertEqual(llm.task_state, 'check_failed')
        self.assertNotIn('Previous query:', llm.calls[1][1])
        self.assertNotIn('Execution error:', llm.calls[1][1])

    def test_changed_instruction_in_same_workspace_resets_state(self):
        bad = 'SELECT absent FROM data'
        llm = FakeLLM(bad, bad)
        self.assertIsNone(self.call(llm))
        self.assertIsNone(self.call(llm, task={'task_type': 'data_analysis', 'instruction': 'Maximum value?'}))
        self.assertEqual(llm.task_state, 'check_failed')
        self.assertNotIn('Previous query:', llm.calls[1][1])
        self.assertEqual(llm._sql_retry_context['key'][1], 'Maximum value?')

    def test_reasoning_repeat_stalls_across_calls_and_abstention_remains_one_call(self):
        bad = 'SELECT missing FROM data'
        llm = FakeLLM(bad, bad)
        self.assertIsNone(self.call(llm))
        self.assertIsNone(self.call(llm))
        self.assertEqual(llm.task_state, 'stalled')
        self.assertEqual(len(llm.calls), 2)
        abstain = FakeLLM('UNSOLVABLE')
        self.assertIsNone(self.call(abstain))
        self.assertEqual(abstain.task_state, 'abstained')
        self.assertEqual(len(abstain.calls), 1)


    def test_sql_scalar_abstention_is_not_persisted_as_an_answer(self):
        llm = FakeLLM("SELECT 'UNSOLVABLE'")
        self.assertIsNone(self.call(llm))
        self.assertEqual(llm.task_state, 'abstained')
        self.assertEqual(len(llm.calls), 1)

    def test_valid_text_scalar_and_near_abstention_are_not_fuzzy_matched(self):
        for value in ('Riga', 'UNSOLVABLE as a title', 'UNSOLEvable'):
            with self.subTest(value=value):
                llm = FakeLLM("SELECT '" + value + "'")
                self.assertEqual(self.call(llm), value)
                self.assertEqual(llm.task_state, 'answered')

    def test_exact_int_check_supplies_feedback_to_outer_correction(self):
        task = dict(self.task, check='exact_int')
        llm = FakeLLM('SELECT 9.5', 'SELECT SUM(value) FROM data')
        self.assertIsNone(self.call(llm, task=task))
        self.assertEqual(llm.task_state, 'check_failed')
        self.assertEqual(self.call(llm, task=task), '9')
        self.assertIn('Previous query: SELECT 9.5', llm.calls[1][1])
        self.assertIn('Expected an integer result, got: 9.5', llm.calls[1][1])

    def test_exact_int_check_also_works_with_plain_inner_correction(self):
        llm = FakeLLM("SELECT 'not a number'", 'SELECT -3')
        result = self.call(llm, task=dict(self.task, check='exact_int'), model=MODELS['large_plain'])
        self.assertEqual(result, '-3')
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('Expected an integer result, got: not a number', llm.calls[1][1])

    def test_fractional_result_is_allowed_without_exact_int_and_guidance_is_present(self):
        llm = FakeLLM('SELECT 1.0 * 68 / 5')
        self.assertEqual(self.call(llm), '13.6')
        self.assertIn('SQLite division of two integers truncates', llm.calls[0][0])
        self.assertIn('CAST(numerator AS REAL)', llm.calls[0][0])


if __name__ == '__main__':
    unittest.main()