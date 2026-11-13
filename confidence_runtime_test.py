"""Independent offline checks of confidence allocation and useful continuation.

All model responses are scripted. No network, historical labels, or research
fixtures are used. Only temporary task workspaces are changed by these tests.
"""
import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as runtime
from confidence_policy import ModelAssessment
from planner import MODELS
from retry_policy import decide_retry
from task_confidence import ConfidenceEstimate


MODEL = MODELS['large_reason']


def envelope(*, action='continue', confidence='low', reason='uncertain',
             next_step='Compute the supplied expression.', result='CALC: 17+25'):
    return json.dumps({
        'protocol': 'nsrc-assessment-v1',
        'assessment': {'action': action, 'confidence': confidence,
                       'reason_code': reason, 'next_step': next_step},
        'result': result,
    })


class ScriptedLLM:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []
        self.confidence_enabled = True
        self.last_finish_reason = 'stop'
        self.last_feedback = ''
        self.task_progress = {}
        self.last_assessment = None
        self.assessment_records = []

    async def ask(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        return next(self.answers)


def context(text, *, partial=True, marker='source.md'):
    return SimpleNamespace(
        partial=partial, text=text,
        snippets=(SimpleNamespace(path=marker, text=text),))


class ConfidenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def workspace(self, name='workspace', files=None):
        directory = self.root / name
        directory.mkdir(parents=True)
        for filename, text in (files or {}).items():
            path = directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        return directory

    def run_cli(self, tasks, *, extra=(), responses=None, budget=100000):
        folder = self.root / ('cli_' + str(len(list(self.root.iterdir()))))
        folder.mkdir()
        for name, score in tasks:
            task_dir = folder / name
            (task_dir / 'workspace').mkdir(parents=True)
            (task_dir / 'task.json').write_text(json.dumps({
                'task_type': 'math', 'instruction': name + ' Compute seventeen plus twenty-five.',
                'reward': 1, 'check': 'exact_int',
            }), encoding='utf-8')
        scores = dict(tasks)
        instances = []
        scripted = {key: iter(values) for key, values in (responses or {}).items()}

        class OfflineLLM:
            def __init__(self, budget, **kwargs):
                self.budget = float(budget)
                self.spent = 0.0
                self.call_records = []
                self.calls = []
                self.abort_run = False
                self.last_finish_reason = 'stop'
                instances.append(self)

            @property
            def remaining(self):
                return self.budget - self.spent

            async def ask(self, system, user, **kwargs):
                names = [name for name in scores if name in user]
                if len(names) != 1:
                    raise AssertionError('Exactly one synthetic task must own a call')
                name = names[0]
                self.calls.append({'name': name, 'system': system, 'user': user,
                                   'opportunity': self.opportunity,
                                   'competing_cost': self.competing_cost,
                                   'attempts_before_call': self.task_progress.get('attempts', 0)})
                self.last_cap = kwargs['max_tokens']
                self.spent += 10
                self.call_records.append({'status': 'provider_reconciled',
                                          'weighted_tokens': 10, 'model': kwargs['model']})
                return next(scripted[name]) if name in scripted else 'CALC: 17+25'

        def confidence(task, features, prior_probability, model, **kwargs):
            name = task['instruction'].split()[0]
            return ConfidenceEstimate(scores[name], 'independent_test_score', ('test_visible_signal',))

        chosen = SimpleNamespace(first_call=500.0, total_policy=1000.0,
                                 source='independent_test_cost', reason=None)
        trace = folder / 'trace.jsonl'
        argv = ['run_agent.py', '--tasks-dir', str(folder), '--trace-file', str(trace),
                '--initial-model', MODEL, '--budget', str(budget), *extra]
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.object(runtime, 'JiuwenLLM', OfflineLLM))
            stack.enter_context(patch.object(runtime, 'deterministic_answer', return_value=None))
            stack.enter_context(patch.object(runtime, 'load_task_cost_model', return_value=(None, 'test')))
            stack.enter_context(patch.object(runtime, 'task_cost_forecast', return_value=(chosen, None)))
            scorer = stack.enter_context(patch.object(runtime, 'estimate_confidence', side_effect=confidence))
            stack.enter_context(patch.dict(os.environ, {'LLM_BASE_URL': 'https://offline.invalid',
                                                        'LLM_API_KEY': 'offline-test'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            asyncio.run(runtime.main())
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(instances), 1)
        return folder, instances[0], rows, scorer

    def test_normal_cli_ranks_task_confidence_and_still_attempts_low_score(self):
        folder, llm, rows, scorer = self.run_cli([('task_001', 0.01), ('task_002', 0.90)])
        self.assertTrue(scorer.called)
        self.assertEqual([call['name'] for call in llm.calls], ['task_002', 'task_001'])
        self.assertEqual({row['outcome'] for row in rows}, {'answered_unverified'})
        for name in ('task_001', 'task_002'):
            self.assertEqual((folder / name / 'answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_cli_alternative_value_uses_the_other_tasks_confidence(self):
        _, low, _, _ = self.run_cli([('task_001', 0.90), ('task_002', 0.01)])
        _, high, _, _ = self.run_cli([('task_001', 0.90), ('task_002', 0.80)])
        self.assertEqual(low.calls[0]['name'], 'task_001')
        self.assertEqual(high.calls[0]['name'], 'task_001')
        self.assertGreater(high.calls[0]['opportunity'], low.calls[0]['opportunity'] * 10)
        self.assertAlmostEqual(high.calls[0]['competing_cost'], low.calls[0]['competing_cost'])

    def test_cli_off_and_classic_do_not_apply_new_score_or_skill(self):
        for extra in (('--confidence-policy', 'off'), ('--retry-policy', 'classic')):
            with self.subTest(extra=extra):
                _, llm, _, scorer = self.run_cli([('task_001', 0.01), ('task_002', 0.90)], extra=extra)
                self.assertFalse(scorer.called)
                self.assertEqual([call['name'] for call in llm.calls], ['task_001', 'task_002'])
                self.assertNotIn('nsrc-assessment-v1', llm.calls[0]['system'])

    def test_cli_surplus_budget_allows_tool_feedback_repair(self):
        _, llm, rows, _ = self.run_cli(
            [('task_001', 0.01)], responses={'task_001': ['CALC: undefined(17)', 'CALC: 17+25']})
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('Invalid arithmetic expression', llm.calls[1]['user'])
        self.assertEqual(rows[-1]['outcome'], 'answered_unverified')
        self.assertNotEqual(rows[-1]['outcome'], 'impossible')

    def test_cli_model_stop_without_a_next_step_is_not_a_formal_proof(self):
        blocked = envelope(action='stop', reason='missing_information', next_step='', result=None)
        folder, llm, rows, _ = self.run_cli(
            [('task_001', 0.20)], responses={'task_001': [blocked]})
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(rows[-1]['outcome'], 'skipped')
        self.assertFalse((folder / 'task_001' / 'answer.txt').exists())
        self.assertNotIn('impossible', {row.get('outcome') for row in rows})

    def test_cli_defers_once_then_resumes_with_attempts_and_total_cost_preserved(self):
        blocked = envelope(action='defer', reason='uncertain',
                           next_step='Recheck the arithmetic operations in the stated total.', result=None)
        folder, llm, rows, _ = self.run_cli(
            [('task_001', 0.90), ('task_002', 0.80)], budget=2200,
            responses={'task_001': [blocked, 'CALC: 17+25']})
        self.assertEqual([call['name'] for call in llm.calls], ['task_001', 'task_002', 'task_001'])
        self.assertEqual(llm.calls[-1]['attempts_before_call'], 1)
        visits = [row for row in rows if row['task_id'] == 'task_001']
        self.assertEqual(len(visits), 2)
        self.assertEqual([row['visit'] for row in visits], [1, 2])
        self.assertEqual(visits[0]['outcome'], 'deferred')
        self.assertFalse(visits[0]['execution_complete'])
        self.assertEqual(visits[1]['outcome'], 'answered_unverified')
        self.assertTrue(visits[1]['execution_complete'])
        self.assertEqual(visits[1]['task_actual_tokens'], 20)
        self.assertEqual([row['actual_tokens'] for row in visits], [10, 10])
        self.assertEqual(llm.spent, 30)
        self.assertEqual((folder / 'task_001' / 'answer.txt').read_text(encoding='utf-8'), '42\n')
        self.assertNotIn('impossible', {row.get('outcome') for row in rows})

    def test_cli_deferred_failure_is_not_requeued_or_given_a_fresh_attempt_budget(self):
        blocked = envelope(action='defer', reason='uncertain',
                           next_step='Recheck the arithmetic operations in the stated total.', result=None)
        _, llm, rows, _ = self.run_cli(
            [('task_001', 0.90), ('task_002', 0.80)], budget=2200,
            responses={'task_001': [blocked, blocked]})
        self.assertEqual([call['name'] for call in llm.calls], ['task_001', 'task_002', 'task_001'])
        visits = [row for row in rows if row['task_id'] == 'task_001']
        self.assertEqual([row['outcome'] for row in visits], ['deferred', 'skipped'])
        self.assertEqual(visits[-1]['task_actual_tokens'], 20)
        self.assertEqual(visits[-1]['retry_decisions'][-1]['reason'], 'no_verifiable_progress')

    def test_low_model_confidence_does_not_discard_locally_valid_answer(self):
        llm = ScriptedLLM(envelope(confidence='low'))
        answer = asyncio.run(runtime.model_answer(llm, {
            'task_type': 'math', 'instruction': 'Calculate the total.', 'check': 'exact_int'},
            self.workspace(), MODEL))
        self.assertEqual(answer, '42')
        self.assertEqual(llm.task_state, 'answered')
        self.assertEqual(len(llm.calls), 1)

    def test_math_repair_receives_tool_error_and_not_an_identical_prompt(self):
        llm = ScriptedLLM('CALC: unknown(17)', 'CALC: 17+25')
        task = {'task_type': 'math', 'instruction': 'Calculate the total.'}
        workspace = self.workspace()
        self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)))
        self.assertEqual(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)), '42')
        self.assertNotEqual(llm.calls[0][1], llm.calls[1][1])
        self.assertIn('Invalid arithmetic expression', llm.calls[1][1])

    def test_sql_replay_still_stops_with_confidence_enabled(self):
        workspace = self.workspace(files={'values.csv': 'amount\n17\n25\n'})
        llm = ScriptedLLM('SELECT missing FROM data', 'SELECT missing FROM data')
        task = {'task_type': 'data_analysis', 'instruction': 'Total amount?'}
        self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)))
        self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)))
        self.assertEqual(llm.task_state, 'stalled')
        self.assertIn('no such column', llm.calls[1][1])
        self.assertEqual(len(llm.calls), 2)

    def test_sql_history_survives_another_sql_task_before_deferred_resume(self):
        first = self.workspace('first', {'values.csv': 'amount\n17\n25\n'})
        second = self.workspace('second', {'values.csv': 'amount\n1\n2\n'})
        task_a = {'task_type': 'data_analysis', 'instruction': 'Total amount of the first task?'}
        task_b = {'task_type': 'data_analysis', 'instruction': 'Total amount of the second task?'}
        llm = ScriptedLLM('SELECT missing FROM data', 'SELECT SUM(amount) FROM data',
                         'SELECT missing FROM data')
        self.assertIsNone(asyncio.run(runtime.model_answer(llm, task_a, first, MODEL)))
        saved_progress, saved_feedback = llm.task_progress, llm.last_feedback
        llm.task_progress, llm.last_feedback = {}, ''
        self.assertEqual(asyncio.run(runtime.model_answer(llm, task_b, second, MODEL)), '3')
        llm.task_progress, llm.last_feedback = saved_progress, saved_feedback
        self.assertIsNone(asyncio.run(runtime.model_answer(llm, task_a, first, MODEL)))
        self.assertEqual(llm.task_state, 'stalled')
        self.assertIn('Previous query: SELECT missing FROM data', llm.calls[-1][1])
        self.assertNotIn('Total amount of the second task?', llm.calls[-1][1])

    def test_partial_search_expands_once_and_supplies_new_evidence(self):
        initial = context('The document describes an expedition.')
        expanded = context('The expedition destination was Riga.', partial=False)
        llm = ScriptedLLM('UNSOLVABLE', 'Riga')
        task = {'task_type': 'search', 'instruction': 'What was the expedition destination?'}
        workspace = self.workspace()
        with patch.object(runtime, 'retrieve_context', side_effect=[initial, expanded]) as retrieve:
            self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)))
            self.assertTrue(llm.new_evidence)
            self.assertEqual(llm.task_state, 'check_failed')
            self.assertEqual(asyncio.run(runtime.model_answer(llm, task, workspace, MODEL)), 'Riga')
            self.assertEqual(retrieve.call_count, 2)
        self.assertIn('destination was Riga', llm.calls[1][1])
        self.assertEqual(len(llm.calls), 2)

    def test_search_metadata_change_is_not_new_evidence(self):
        llm = ScriptedLLM('UNSOLVABLE')
        initial = context('The destination is not recorded.')
        expanded = context('The destination is not recorded.', partial=False)
        task = {'task_type': 'search', 'instruction': 'What is the destination?'}
        with patch.object(runtime, 'retrieve_context', side_effect=[initial, expanded]):
            self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, self.workspace(), MODEL)))
        self.assertFalse(llm.new_evidence)
        self.assertEqual(llm.task_state, 'abstained')
        self.assertEqual(len(llm.calls), 1)

    def test_complete_search_abstention_does_not_retrieve_again(self):
        llm = ScriptedLLM('UNSOLVABLE')
        task = {'task_type': 'search', 'instruction': 'What is the destination?'}
        with patch.object(runtime, 'retrieve_context', return_value=context('No destination.', partial=False)) as retrieve:
            self.assertIsNone(asyncio.run(runtime.model_answer(llm, task, self.workspace(), MODEL)))
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(llm.task_state, 'abstained')

    def retry(self, **overrides):
        options = dict(kind='math', state='check_failed', feedback='Invalid arithmetic expression',
                       attempts=1, last_model=MODEL, reasoning_model=MODEL, last_cap=3072,
                       finish_reason='stop', remaining=100000, reward=1,
                       opportunity=0, competing_cost=0, confidence_enabled=True)
        options.update(overrides)
        return decide_retry(**options)

    def test_high_confidence_cannot_override_repeated_or_terminal_state(self):
        assessment = ModelAssessment('continue', 'high', 'solvable', 'Repeat the same calculation.')
        for extra in ({'repeated': True}, {'state': 'stalled'}, {'state': 'abstained'}):
            with self.subTest(extra=extra):
                self.assertFalse(self.retry(assessment=assessment, **extra).retry)

    def test_budget_pressure_respects_marginal_opportunity_but_surplus_repairs(self):
        assessment = ModelAssessment('continue', 'low', 'tool_error', 'Correct the arithmetic syntax.')
        pressured = self.retry(assessment=assessment, remaining=1600, competing_cost=1500, opportunity=0.10)
        surplus = self.retry(assessment=assessment, remaining=100000, competing_cost=1500, opportunity=0.10)
        self.assertFalse(pressured.retry)
        self.assertTrue(surplus.retry)

    def test_observed_call_cost_changes_marginal_retry_allocation(self):
        assessment = ModelAssessment('continue', 'high', 'tool_error', 'Correct the arithmetic syntax.')
        cheap = self.retry(assessment=assessment, observed_cost=100, remaining=3000,
                           competing_cost=2000, opportunity=0.001)
        costly = self.retry(assessment=assessment, observed_cost=900, remaining=3000,
                            competing_cost=2000, opportunity=0.001)
        self.assertTrue(cheap.retry)
        self.assertEqual(cheap.predicted_cost, 150)
        self.assertFalse(costly.retry)
        self.assertEqual(costly.action, 'defer')
        self.assertEqual(costly.predicted_cost, 900)

    def test_model_stop_without_plan_is_not_requeued_under_pressure(self):
        blocked = envelope(action='stop', confidence='low', reason='missing_information',
                           next_step='', result=None)
        _, llm, rows, _ = self.run_cli(
            [('task_001', 0.90), ('task_002', 0.80)], budget=2200,
            responses={'task_001': [blocked], 'task_002': ['CALC: 17+25']})
        self.assertEqual([call['name'] for call in llm.calls], ['task_001', 'task_002'])
        first = next(row for row in rows if row['task_id'] == 'task_001')
        self.assertEqual(first['outcome'], 'skipped')
        self.assertEqual(first['task_actual_tokens'], 10)
        self.assertEqual(first['retry_decisions'][-1]['reason'], 'no_actionable_next_step')

    def test_generation_limit_is_extendable_without_calling_it_impossible(self):
        decision = self.retry(finish_reason='length', repeated=True, last_cap=512)
        self.assertTrue(decision.retry)
        self.assertGreater(decision.max_tokens, 512)
        self.assertEqual(decision.model, MODEL)


if __name__ == '__main__':
    unittest.main()
