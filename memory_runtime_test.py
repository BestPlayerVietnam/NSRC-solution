"""Independent CPU-only integration checks for bounded cross-task tool lessons.

All model answers are scripted. The tests change only temporary workspaces and
never invoke a remote model or inspect archived contest labels.
"""
import asyncio
import ast
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as runtime
from experience_memory import ExperienceMemory
from planner import MODELS


MODEL = MODELS['large_reason']


class OfflineLLM:
    def __init__(self, *answers, memory=None, model=MODEL):
        self.answers = iter(answers)
        self.calls = []
        self.experience_memory = memory if memory is not None else ExperienceMemory(mode='revise')
        self.memory_records = []
        self.confidence_enabled = False
        self.last_finish_reason = 'stop'
        self.last_feedback = ''
        self.task_progress = {}
        self.last_assessment = None
        self.assessment_records = []
        self.initial_model = model
        self.fixed_model = True
        self.retry_policy = 'adaptive'
        self.remaining = 100000
        self.retry_decisions = []
        self.call_records = []
        self.opportunity = 0
        self.competing_cost = 0
        self.defer_requested = False

    async def ask(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        self.last_cap = kwargs['max_tokens']
        self.call_records.append({'status': 'provider_reconciled',
                                  'weighted_tokens': 10, 'model': kwargs['model']})
        return next(self.answers)


class MemoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def workspace(self, name, files=None):
        folder = self.root / name
        folder.mkdir(parents=True)
        for relative, text in (files or {}).items():
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        return folder

    def call(self, llm, task, workspace, *, model=MODEL):
        return asyncio.run(runtime.model_answer(llm, task, workspace, model))

    def lessons(self, llm, *, model=MODEL, domain='math', instruction='Compute the arithmetic expression.'):
        return llm.experience_memory.retrieve(model=model, domain=domain,
                                              instruction=instruction, source_ref='independent-next-task')

    def teach_math(self, llm=None, name='teach_math'):
        llm = llm or OfflineLLM('CALC: unresolved(17)', 'CALC: 17+25')
        workspace = self.workspace(name)
        task = {'task_type': 'math', 'instruction': 'Compute the arithmetic expression for the total.',
                'check': 'exact_int'}
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertEqual(self.call(llm, task, workspace), '42')
        return llm, task, workspace

    def test_cross_task_math_repair_is_reused_without_a_memory_model_call(self):
        llm = OfflineLLM('CALC: unresolved(17)', 'CALC: 17+25', 'CALC: 6*7')
        self.teach_math(llm)
        lessons = self.lessons(llm)
        self.assertTrue(lessons)
        rendered = llm.experience_memory.render(lessons)
        self.assertTrue(rendered)
        self.assertNotIn(rendered, '\n'.join(llm.calls[0][:2]))
        self.assertNotIn(rendered, '\n'.join(llm.calls[1][:2]))
        self.assertEqual(self.call(llm, {'task_type': 'math', 'instruction': 'Compute a new arithmetic expression.'},
                                   self.workspace('reuse')), '42')
        self.assertIn(rendered, '\n'.join(llm.calls[2][:2]))
        self.assertEqual(len(llm.calls), 3)
        self.assertFalse(llm.confidence_enabled)

    def test_matching_tool_lesson_isolated_by_actual_model_and_domain(self):
        llm, _, _ = self.teach_math()
        self.assertTrue(self.lessons(llm))
        for model in MODELS.values():
            if model != MODEL:
                with self.subTest(model=model):
                    self.assertEqual(self.lessons(llm, model=model), ())
        for domain in ('coding', 'data_analysis', 'search'):
            with self.subTest(domain=domain):
                self.assertEqual(self.lessons(llm, domain=domain), ())

    def test_runtime_injection_uses_requested_model_instead_of_initial_model(self):
        llm = OfflineLLM('CALC: unresolved(17)', 'CALC: 17+25', 'CALC: 6*7')
        self.teach_math(llm)
        rendered = llm.experience_memory.render(self.lessons(llm))
        self.assertTrue(rendered)
        llm.fixed_model = False
        self.assertEqual(self.call(llm, {'task_type': 'math', 'instruction': 'Compute arithmetic on a different model.'},
                                   self.workspace('other_model'), model=MODELS['small_reason']), '42')
        self.assertEqual(llm.initial_model, MODEL)
        self.assertNotIn(rendered, '\n'.join(llm.calls[-1][:2]))

    def test_another_models_success_cannot_promote_the_first_models_error(self):
        llm = OfflineLLM('CALC: unknown(17)', 'CALC: 17+25')
        llm.fixed_model = False
        task = {'task_type': 'math', 'instruction': 'Compute arithmetic.'}
        workspace = self.workspace('routed_repair')
        self.assertIsNone(self.call(llm, task, workspace, model=MODEL))
        self.assertEqual(self.call(llm, task, workspace, model=MODELS['small_reason']), '42')
        self.assertEqual(self.lessons(llm), ())
        self.assertEqual(self.lessons(llm, model=MODELS['small_reason']), ())
        self.assertEqual(llm.experience_memory.summary()['cards'], 0)
        runtime.close_memory_task(llm)
        self.assertEqual(llm.experience_memory.summary()['pending_tasks'], 0)

    def test_append_keeps_old_card_while_revision_updates_exposed_lesson(self):
        for mode in ('append', 'revise'):
            with self.subTest(mode=mode):
                llm = OfflineLLM('CALC: unresolved(17)', 'CALC: 17+25',
                                 'CALC: another_unknown(19)', 'CALC: 5+6',
                                 memory=ExperienceMemory(mode=mode))
                self.teach_math(llm, name=mode + '_first')
                before = self.lessons(llm)[0]
                task = {'task_type': 'math', 'instruction': 'Compute another arithmetic expression.'}
                workspace = self.workspace(mode + '_second')
                self.assertIsNone(self.call(llm, task, workspace))
                self.assertEqual(self.call(llm, task, workspace), '11')
                after = self.lessons(llm)[0]
                self.assertEqual(after.id, before.id)
                self.assertEqual(llm.experience_memory.summary()['cards'], 1)
                self.assertEqual(after.version, before.version + (mode == 'revise'))
                self.assertEqual(after.support_count, before.support_count + (mode == 'revise'))
                self.assertEqual(llm.experience_memory.summary()['pending_tasks'], 0)

    def test_success_without_a_prior_tool_error_creates_no_lesson(self):
        llm = OfflineLLM('CALC: 17+25')
        self.assertEqual(self.call(llm, {'task_type': 'math', 'instruction': 'Compute arithmetic.'},
                                   self.workspace('first_success')), '42')
        self.assertEqual(self.lessons(llm), ())

    def test_failure_and_plain_abstention_never_promote_a_lesson(self):
        task = {'task_type': 'math', 'instruction': 'Compute the arithmetic expression.'}
        for index, second in enumerate(('CALC: unknown_again(17)', 'UNSOLVABLE')):
            with self.subTest(second=second):
                llm = OfflineLLM('CALC: unknown(17)', second)
                workspace = self.workspace('failed_' + str(index))
                self.assertIsNone(self.call(llm, task, workspace))
                self.assertIsNone(self.call(llm, task, workspace))
                self.assertEqual(self.lessons(llm), ())

    def test_self_reported_tool_error_is_not_a_tool_observation(self):
        assessment = json.dumps({'protocol': 'nsrc-assessment-v1', 'assessment': {
            'action': 'continue', 'confidence': 'low', 'reason_code': 'tool_error',
            'next_step': 'Correct the Invalid arithmetic expression in the next check.'}, 'result': None})
        llm = OfflineLLM(assessment, 'CALC: 17+25')
        llm.confidence_enabled = True
        task = {'task_type': 'math', 'instruction': 'Compute the arithmetic expression.'}
        workspace = self.workspace('model_claim')
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertEqual(self.call(llm, task, workspace), '42')
        self.assertEqual(self.lessons(llm), ())

    def test_structured_abstention_after_a_real_error_does_not_become_success(self):
        abstain = json.dumps({'protocol': 'nsrc-assessment-v1', 'assessment': {
            'action': 'stop', 'confidence': 'low', 'reason_code': 'missing_information',
            'next_step': ''}, 'result': None})
        llm = OfflineLLM('CALC: unknown(17)', abstain)
        llm.confidence_enabled = True
        task = {'task_type': 'math', 'instruction': 'Compute arithmetic.'}
        workspace = self.workspace('structured_abstain')
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertEqual(self.lessons(llm), ())

    def test_sql_lesson_contains_no_query_schema_values_or_raw_source_reference(self):
        private_column = 'private_invoice_marker_7391'
        private_missing = 'missing_customer_marker_8193'
        workspace = self.workspace('private_workspace_3179', {
            'private_table_2981.csv': private_column + '\n17291\n24907\n'})
        task = {'task_type': 'data_analysis', 'instruction': 'Sum the available numeric column.'}
        llm = OfflineLLM('SELECT ' + private_missing + ' FROM data',
                         'SELECT SUM(' + private_column + ') FROM data')
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertEqual(self.call(llm, task, workspace), '42198')
        lessons = self.lessons(llm, domain='data_analysis', instruction='Query a column using SQLite SELECT.')
        self.assertTrue(lessons)
        stored = json.dumps([asdict(lesson) for lesson in lessons], ensure_ascii=False)
        rendered = llm.experience_memory.render(lessons)
        for forbidden in (private_column, private_missing,
                          'private_table_2981', 'private_workspace_3179', str(workspace),
                          'SELECT ' + private_missing):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, stored + rendered)
        for number in ('17291', '24907', '42198'):
            # A digit substring inside an opaque source hash is not copied data.
            self.assertIsNone(re.search(r'(?<![A-Za-z0-9_])' + number + r'(?![A-Za-z0-9_])', stored + rendered))

    def test_sql_execution_failure_with_unsupported_category_is_not_learned(self):
        workspace = self.workspace('unknown_sql_error', {'values.csv': 'amount\n17\n25\n'})
        llm = OfflineLLM('SELECT unknown_function(amount) FROM data', 'SELECT SUM(amount) FROM data')
        task = {'task_type': 'data_analysis', 'instruction': 'Total amount using SQLite.'}
        self.assertIsNone(self.call(llm, task, workspace))
        self.assertEqual(self.call(llm, task, workspace), '42')
        self.assertEqual(self.lessons(llm, domain='data_analysis', instruction='Query SQLite SELECT column.'), ())

    def test_old_task_error_cannot_be_promoted_by_another_tasks_success(self):
        llm = OfflineLLM('CALC: unknown(17)', 'CALC: 17+25')
        task = {'task_type': 'math', 'instruction': 'Compute an arithmetic expression.'}
        self.assertIsNone(self.call(llm, task, self.workspace('failed_source')))
        self.assertEqual(self.call(llm, task, self.workspace('unrelated_success')), '42')
        self.assertEqual(self.lessons(llm), ())

    def test_memory_render_is_bounded_and_retrieves_at_most_two_cards(self):
        memory = ExperienceMemory(mode='revise')
        for index, failure in enumerate(('SQLite: no such column: missing',
                                         'Expected an integer result, got: 1.5',
                                         'Query must return exactly one scalar value')):
            source = 'observed-tool-source-' + str(index)
            memory.record_failure(model=MODEL, domain='data_analysis', source_ref=source,
                                  call_seq=1, feedback=failure)
            memory.finish_task(model=MODEL, domain='data_analysis', source_ref=source,
                               call_seq=2, verification_level='tool_execution')
        lessons = memory.retrieve(model=MODEL, domain='data_analysis',
                                  instruction='SQLite SELECT integer column query returning one scalar.',
                                  source_ref='new-query-source')
        self.assertEqual(len(lessons), 2)
        self.assertLessEqual(len(memory.render(lessons)), 1000)

    def test_existing_lesson_never_executes_code_or_marks_a_task_impossible(self):
        llm = OfflineLLM('CALC: unresolved(17)', 'CALC: 17+25', 'CALC: 5+6')
        self.teach_math(llm)
        workspace = self.workspace('safe_reuse', {'untouched.py': 'raise RuntimeError("must not run")\n'})
        before = (workspace / 'untouched.py').read_bytes()
        with patch.object(runtime, 'run_tests', side_effect=AssertionError('memory may not execute code')):
            answer = self.call(llm, {'task_type': 'math', 'instruction': 'Compute a new arithmetic total.'}, workspace)
        self.assertEqual(answer, '11')
        self.assertEqual(llm.task_state, 'answered')
        self.assertEqual((workspace / 'untouched.py').read_bytes(), before)
        self.assertEqual(len(llm.calls), 3)

    def test_coding_tool_repair_requires_public_tests_before_cross_task_reuse(self):
        files = {'calc.py': 'def add(a, b):\n    return a-b\n',
                 'tests/test_calc.py': 'from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n'}
        wrong_edit = json.dumps({'edits': [{'path': 'calc.py', 'old': 'a nonexistent span', 'new': 'return a+b'}]})
        correct_edit = json.dumps({'edits': [{'path': 'calc.py', 'old': 'return a-b', 'new': 'return a+b'}]})
        llm = OfflineLLM(wrong_edit, correct_edit, correct_edit)
        task = {'task_type': 'coding', 'instruction': 'Repair the add function in calc.py without changing tests.'}
        first = self.workspace('coding_first', files)
        original = {name: (first / name).read_bytes() for name in files}
        self.assertTrue(asyncio.run(runtime.solve_coding(llm, task, first)))
        lessons = self.lessons(llm, domain='coding', instruction='Repair the function with exact edits.')
        self.assertTrue(lessons)
        self.assertEqual({lesson.verification_level for lesson in lessons}, {'public_tests'})
        rendered = llm.experience_memory.render(lessons)
        second = self.workspace('coding_second', files)
        self.assertTrue(asyncio.run(runtime.solve_coding(llm, task, second)))
        self.assertIn(rendered, '\n'.join(llm.calls[-1][:2]))
        self.assertEqual(len(llm.calls), 3)
        self.assertIn('return a+b', (second / 'calc.py').read_text(encoding='utf-8'))
        for workspace in (first, second):
            self.assertEqual({path.relative_to(workspace).as_posix() for path in workspace.rglob('*') if path.is_file()},
                             set(files))
            self.assertEqual((workspace / 'tests/test_calc.py').read_bytes(), original['tests/test_calc.py'])
            self.assertEqual(ast.dump(ast.parse((workspace / 'calc.py').read_text(encoding='utf-8'))),
                             ast.dump(ast.parse(files['calc.py'].replace('a-b', 'a+b'))))

    def test_routed_coding_error_then_another_models_repair_does_not_promote_either_model(self):
        files = {'calc.py': 'def add(a, b):\n    return a-b\n',
                 'tests/test_calc.py': 'from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n'}
        forbidden_edit = json.dumps({'edits': [{'path': 'tests/test_calc.py',
                                               'old': '== 5', 'new': '== -1'}]})
        correct_edit = json.dumps({'edits': [{'path': 'calc.py', 'old': 'return a-b', 'new': 'return a+b'}]})
        first_model = MODELS['small_plain']
        second_model = MODELS['large_reason']
        llm = OfflineLLM(forbidden_edit, correct_edit, model=first_model)
        llm.fixed_model = False
        workspace = self.workspace('routed_coding', files)
        tests_before = (workspace / 'tests/test_calc.py').read_bytes()
        task = {'task_type': 'coding', 'instruction': 'Repair the add function in calc.py without changing tests.'}
        self.assertTrue(asyncio.run(runtime.solve_coding(llm, task, workspace)))
        self.assertEqual([call[2]['model'] for call in llm.calls], [first_model, second_model])
        self.assertEqual(len(llm.calls), 2)
        for model in (first_model, second_model):
            self.assertEqual(self.lessons(llm, model=model, domain='coding', instruction='Repair allowed source paths.'), ())
        self.assertEqual(llm.experience_memory.summary()['cards'], 0)
        runtime.close_memory_task(llm)
        self.assertEqual(llm.experience_memory.summary()['pending_tasks'], 0)
        self.assertEqual((workspace / 'tests/test_calc.py').read_bytes(), tests_before)
        self.assertEqual({path.relative_to(workspace).as_posix() for path in workspace.rglob('*') if path.is_file()}, set(files))
        self.assertIn('return a+b', (workspace / 'calc.py').read_text(encoding='utf-8'))

    def run_cli(self, *, extra=(), answers=None):
        folder = self.workspace('cli_' + str(len(list(self.root.iterdir()))))
        for index, reward in ((1, 10), (2, 1)):
            task_dir = folder / ('task_00' + str(index))
            (task_dir / 'workspace').mkdir(parents=True)
            (task_dir / 'task.json').write_text(json.dumps({
                'task_type': 'math', 'instruction': 'task_00' + str(index) + ' Compute the arithmetic expression.',
                'reward': reward, 'check': 'exact_int'}), encoding='utf-8')
        instances = []

        class CLILLM(OfflineLLM):
            def __init__(self, budget, **kwargs):
                super().__init__(*(answers or ('CALC: unknown(17)', 'CALC: 17+25', 'CALC: 6*7')))
                self.budget = float(budget)
                self.spent = 0
                self.abort_run = False
                instances.append(self)

            async def ask(self, system, user, **kwargs):
                answer = await super().ask(system, user, **kwargs)
                self.spent += 10
                self.remaining = self.budget - self.spent
                return answer

        forecast = SimpleNamespace(first_call=500.0, total_policy=1000.0, source='independent_test', reason=None)
        trace = folder / 'trace.jsonl'
        argv = ['run_agent.py', '--tasks-dir', str(folder), '--trace-file', str(trace),
                '--initial-model', MODEL, '--budget', '100000', '--confidence-policy', 'off',
                '--sla-policy', 'adaptive', *extra]
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.object(runtime, 'JiuwenLLM', CLILLM))
            stack.enter_context(patch.object(runtime, 'deterministic_answer', return_value=None))
            stack.enter_context(patch.object(runtime, 'load_task_cost_model', return_value=(None, 'test')))
            stack.enter_context(patch.object(runtime, 'task_cost_forecast', return_value=(forecast, None)))
            stack.enter_context(patch.dict(os.environ, {'LLM_BASE_URL': 'https://offline.invalid',
                                                        'LLM_API_KEY': 'offline-test'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            asyncio.run(runtime.main())
        self.assertEqual(len(instances), 1)
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        return folder, instances[0], rows

    def test_normal_cli_reuses_memory_with_confidence_disabled_and_three_calls(self):
        folder, llm, rows = self.run_cli()
        self.assertFalse(llm.confidence_enabled)
        self.assertEqual(len(llm.calls), 3)
        self.assertIn('task_002', llm.calls[-1][1])
        rendered = llm.experience_memory.render(self.lessons(llm))
        self.assertTrue(rendered)
        self.assertIn(rendered, '\n'.join(llm.calls[-1][:2]))
        self.assertEqual({row['outcome'] for row in rows}, {'answered_unverified'})
        for name in ('task_001', 'task_002'):
            self.assertEqual((folder / name / 'answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_cli_trace_identifies_memory_policy_and_per_call_injection(self):
        for mode in ('append', 'revise'):
            with self.subTest(mode=mode):
                _, llm, rows = self.run_cli(extra=('--memory-policy', mode))
                first = next(row for row in rows if row['task_id'] == 'task_001')
                second = next(row for row in rows if row['task_id'] == 'task_002')
                self.assertEqual(first['memory_policy'], mode)
                self.assertIn('memory-v1-' + mode, first['policy_id'].split('|'))
                self.assertIn('sla-v3', first['policy_id'].split('|'))
                self.assertNotIn('|confidence-v1', first['policy_id'])
                self.assertTrue(first['prediction_details']['policy_transfer'])
                self.assertEqual(sum(call['physical_calls'] for row in rows for call in row['memory_calls']), 3)
                self.assertEqual([call['memory_cards'] for call in first['memory_calls']], [0, 0])
                self.assertEqual(len(second['memory_calls']), 1)
                self.assertEqual(second['memory_calls'][0]['memory_cards'], 1)
                self.assertGreater(second['memory_calls'][0]['memory_injected_utf8_bytes'], 0)
                self.assertEqual(second['calls'][0]['memory_versions'], second['memory_calls'][0]['memory_versions'])
                self.assertEqual(first['memory_errors'], 0)
                self.assertEqual(second['memory_errors'], 0)
                self.assertEqual(llm.experience_memory.summary()['pending_tasks'], 0)

    def test_cli_final_failure_clears_pending_without_training_on_next_success(self):
        folder, llm, rows = self.run_cli(answers=('CALC: unknown(17)', 'UNSOLVABLE', 'CALC: 6*7'))
        first = next(row for row in rows if row['task_id'] == 'task_001')
        second = next(row for row in rows if row['task_id'] == 'task_002')
        self.assertEqual(first['outcome'], 'skipped')
        self.assertEqual(first['memory_state']['pending_tasks'], 0)
        self.assertEqual(first['memory_state']['cards'], 0)
        self.assertEqual(second['outcome'], 'answered_unverified')
        self.assertEqual(second['memory_calls'][0]['memory_cards'], 0)
        self.assertEqual(llm.experience_memory.summary()['cards'], 0)
        self.assertEqual(llm.experience_memory.summary()['pending_tasks'], 0)
        self.assertFalse((folder / 'task_001' / 'answer.txt').exists())

    def test_cli_off_and_classic_disable_memory_learning_and_injection(self):
        for extra in (('--memory-policy', 'off'), ('--retry-policy', 'classic')):
            with self.subTest(extra=extra):
                _, llm, rows = self.run_cli(extra=extra)
                self.assertEqual(len(llm.calls), 3)
                self.assertEqual(self.lessons(llm), ())
                self.assertEqual({row['outcome'] for row in rows}, {'answered_unverified'})


if __name__ == '__main__':
    unittest.main()
