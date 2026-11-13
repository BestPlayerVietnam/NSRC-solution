"""Independent SLA integration regressions with a local scripted SDK boundary.

No remote model, archived label or real sleep is used. Task files exist only in
temporary directories; the production ledger, CLI and solver control flow run.
"""
import asyncio
import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_agent as runtime
import coding_tools
from budget_config_test import FakeSDKModel, fake_sdk_modules
from budget_runtime import BudgetLedger, DeadlineExceeded
from latency_planner import LatencyEstimator, LatencyForecast
from planner import MODELS, TaskPlan
from retry_policy import RetryDecision


MODEL = MODELS['large_plain']


class Clock:
    def __init__(self, now=100.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FixtureLatencyEstimator(LatencyEstimator):
    """Known unequal durations isolate scheduling from estimation accuracy."""
    def __init__(self, durations=None):
        super().__init__()
        self.durations = durations or {'math': 110.0, 'search': 20.0}
        self.tool_events = []

    def forecast(self, *, model, domain, **kwargs):
        seconds = self.durations.get(domain, 10.0)
        return LatencyForecast(seconds, seconds, 1.0, 'independent_scheduling_fixture', 0)

    def observe_tool(self, **kwargs):
        self.tool_events.append(dict(kwargs))
        return super().observe_tool(**kwargs)


class LinearLatencyEstimator(LatencyEstimator):
    def forecast(self, *, predicted_tokens, **kwargs):
        seconds = predicted_tokens / 10
        return LatencyForecast(seconds, seconds, 1.0, 'independent_time_cap_fixture', 0)


def response(content='CALC: 17+25', usage=True):
    return SimpleNamespace(content=content, finish_reason='stop',
                           usage_metadata=({'input_tokens': 10, 'output_tokens': 4,
                                            'total_tokens': 14} if usage else {}))


class SLARuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @contextmanager
    def transport(self, *, clock=None, deadline=None, behavior=None):
        clock = clock or Clock()
        behavior = behavior or (lambda messages, kwargs: response())
        FakeSDKModel.instances = []

        async def invoke(sdk, messages, **kwargs):
            sdk.calls.append((messages, kwargs))
            value = behavior(messages, kwargs)
            if isinstance(value, BaseException):
                raise value
            return await value if inspect.isawaitable(value) else value

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, fake_sdk_modules()))
            stack.enter_context(patch.object(FakeSDKModel, 'invoke', new=invoke))
            # Patching the module reference avoids changing asyncio's own clock.
            stack.enter_context(patch.object(runtime, 'time', SimpleNamespace(monotonic=clock)))
            stack.enter_context(patch.object(runtime, 'BudgetLedger',
                                            side_effect=lambda budget: BudgetLedger(budget, clock=clock)))
            stack.enter_context(patch.dict(os.environ, {'LLM_BASE_URL': 'https://offline.invalid',
                                                        'LLM_API_KEY': 'offline-test'}))
            llm = runtime.JiuwenLLM(100000, deadline=deadline)
            yield llm, clock

    def ask(self, llm):
        return asyncio.run(llm.ask('Return a short result.', 'Compute seventeen plus twenty-five.',
                                  model=MODEL, max_tokens=128))

    def test_expired_deadline_never_reserves_or_calls_provider(self):
        with self.transport(deadline=100) as (llm, clock):
            with self.assertRaises(DeadlineExceeded):
                self.ask(llm)
            self.assertEqual(llm.call_records, [])
            self.assertEqual(llm.remaining, 100000)
            self.assertEqual(FakeSDKModel.instances[0].calls, [])

    def test_provider_timeout_is_clamped_to_global_remaining_seconds(self):
        with self.transport(deadline=102.5) as (llm, clock):
            self.assertEqual(self.ask(llm), 'CALC: 17+25')
            self.assertAlmostEqual(FakeSDKModel.instances[0].calls[0][1]['timeout'], 2.5)
            self.assertAlmostEqual(llm.call_records[0]['timeout'], 2.5)
            self.assertEqual(llm.call_records[0]['status'], 'provider_reconciled')

    def test_transport_retry_rechecks_deadline_after_wait_and_keeps_first_reserve(self):
        clock = Clock()
        error = RuntimeError('synthetic request rate limit')
        error.status_code = 429

        def fail(messages, kwargs):
            clock.advance(1)
            return error

        async def oversleep(seconds):
            # A queued wakeup can happen later than the requested sleep.
            clock.advance(seconds + 100)

        with self.transport(clock=clock, deadline=200, behavior=fail) as (llm, _):
            with patch.object(runtime.asyncio, 'sleep', side_effect=oversleep) as sleep:
                with self.assertRaises(DeadlineExceeded):
                    self.ask(llm)
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            self.assertEqual(len(llm.call_records), 1)
            self.assertEqual(llm.call_records[0]['status'], 'reserved_rate_limit')
            self.assertGreater(llm.spent, 0)

    def test_timeout_with_no_retry_time_keeps_reservation_and_error_latency(self):
        clock = Clock()

        def fail(messages, kwargs):
            clock.advance(1.75)
            return TimeoutError('synthetic transport timeout')

        with self.transport(clock=clock, deadline=102, behavior=fail) as (llm, _):
            with self.assertRaises(TimeoutError):
                self.ask(llm)
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            self.assertEqual(llm.call_records[0]['status'], 'reserved_timeout')
            self.assertGreater(llm.spent, 0)
            self.assertAlmostEqual(llm.call_records[0]['latency'], 1.75)

    def test_cancelled_request_never_releases_unknown_provider_cost(self):
        with self.transport(deadline=200, behavior=lambda messages, kwargs: asyncio.CancelledError()) as (llm, _):
            with self.assertRaises(asyncio.CancelledError):
                self.ask(llm)
            self.assertEqual(len(llm.call_records), 1)
            self.assertEqual(llm.call_records[0]['status'], 'reserved_cancelled')
            self.assertGreater(llm.spent, 0)
            self.assertLess(llm.remaining, 100000)

    def test_missing_usage_at_deadline_keeps_reserve(self):
        clock = Clock()

        def answer(messages, kwargs):
            clock.advance(2)
            return response(usage=False)

        with self.transport(clock=clock, deadline=102, behavior=answer) as (llm, _):
            self.ask(llm)
            self.assertEqual(len(llm.call_records), 1)
            self.assertNotEqual(llm.call_records[0]['status'], 'provider_reconciled')
            self.assertGreater(llm.spent, 0)
            spent = llm.spent
            with self.assertRaises(DeadlineExceeded):
                self.ask(llm)
            self.assertEqual(llm.spent, spent)
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)

    def test_public_test_timeout_is_clamped_and_expired_work_never_starts(self):
        clock = Clock()
        with patch.object(runtime, 'time', SimpleNamespace(monotonic=clock)), \
                patch.object(runtime, 'run_tests', return_value=(False, 'fixture failure')) as checks:
            self.assertEqual(runtime.run_tests_with_deadline(self.root, deadline=100.25),
                             (False, 'fixture failure'))
            self.assertGreater(checks.call_args.kwargs['timeout'], 0)
            self.assertLessEqual(checks.call_args.kwargs['timeout'], 0.25)
            self.assertEqual(checks.call_args.kwargs['deadline'], 100.25)
            checks.reset_mock()
            with self.assertRaises(DeadlineExceeded):
                runtime.run_tests_with_deadline(self.root, deadline=100)
            checks.assert_not_called()

    def test_public_test_copy_deadline_prevents_subprocess_creation(self):
        (self.root / 'solution.py').write_text('def result(): return 42\n', encoding='utf-8')
        test = self.root / 'test_solution.py'
        test.write_text('from solution import result\ndef test_result(): assert result() == 42\n',
                        encoding='utf-8')
        original = test.read_bytes()
        clock, copyfile = Clock(), coding_tools.shutil.copyfile

        def delayed_copy(source, target):
            value = copyfile(source, target)
            clock.advance(2)
            return value

        with patch.object(coding_tools, 'time', SimpleNamespace(monotonic=clock)), \
                patch.object(coding_tools.shutil, 'copyfile', side_effect=delayed_copy), \
                patch.object(coding_tools.subprocess, 'Popen') as spawn:
            passing, feedback = coding_tools.run_tests(self.root, timeout=18, deadline=101)
        self.assertFalse(passing)
        self.assertIn('deadline', feedback.lower())
        spawn.assert_not_called()
        self.assertEqual(test.read_bytes(), original)

    def test_public_test_wait_budget_subtracts_copy_preparation_time(self):
        (self.root / 'solution.py').write_text('def result(): return 42\n', encoding='utf-8')
        (self.root / 'test_solution.py').write_text(
            'from solution import result\ndef test_result(): assert result() == 42\n', encoding='utf-8')
        clock, copyfile = Clock(), coding_tools.shutil.copyfile

        def delayed_copy(source, target):
            value = copyfile(source, target)
            clock.advance(0.3)
            return value

        process = SimpleNamespace(wait=Mock(return_value=0), returncode=0)
        with patch.object(coding_tools, 'time', SimpleNamespace(monotonic=clock)), \
                patch.object(coding_tools.shutil, 'copyfile', side_effect=delayed_copy), \
                patch.object(coding_tools.subprocess, 'Popen', return_value=process) as spawn:
            coding_tools.run_tests(self.root, timeout=18, deadline=102)
        spawn.assert_called_once()
        self.assertAlmostEqual(process.wait.call_args.kwargs['timeout'], 1.4)

    def test_coding_candidate_after_deadline_is_not_applied_and_sources_stay_intact(self):
        source = self.root / 'solution.py'
        source.write_text('def result(): return 0\n', encoding='utf-8')
        (self.root / 'test_solution.py').write_text(
            'from solution import result\ndef test_result(): assert result() == 42\n', encoding='utf-8')
        original, clock = source.read_bytes(), Clock()

        def answer(messages, kwargs):
            clock.advance(2)
            return response('def result(): return 42\n')

        with self.transport(clock=clock, deadline=101, behavior=answer) as (llm, _), \
                patch.object(runtime, 'run_tests', return_value=(False, '1 failed')), \
                patch.object(runtime, 'apply_candidate') as apply:
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(runtime.solve_coding(llm, {'task_type': 'coding', 'reward': 5,
                    'instruction': 'Repair solution.py so result returns forty-two.'}, self.root))
            apply.assert_not_called()
        self.assertEqual(source.read_bytes(), original)

    def test_sql_response_after_deadline_does_not_start_database_query(self):
        clock = Clock()
        database = SimpleNamespace(prompt=lambda: 'table data(amount)',
                                   execute=Mock(return_value=('42', None)), close=Mock())

        def answer(messages, kwargs):
            clock.advance(2)
            return response('SELECT 42')

        with self.transport(clock=clock, deadline=101, behavior=answer) as (llm, _), \
                patch.object(runtime, 'CSVDatabase', return_value=database):
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(runtime.model_answer(llm, {'task_type': 'data_analysis',
                    'instruction': 'Return the requested aggregate.'}, self.root, MODEL))
            database.execute.assert_not_called()
            database.close.assert_called_once()
            self.assertEqual(llm.call_records[0]['status'], 'provider_reconciled')

    def test_math_response_after_deadline_does_not_start_local_evaluation(self):
        clock = Clock()

        def answer(messages, kwargs):
            clock.advance(2)
            return response()

        with self.transport(clock=clock, deadline=101, behavior=answer) as (llm, _), \
                patch.object(runtime, 'arithmetic_answer', return_value='42') as evaluate:
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(runtime.model_answer(llm, {'task_type': 'math',
                    'instruction': 'Return the requested total.'}, self.root, MODEL))
            evaluate.assert_not_called()

    def launch_cli(self, *, extra=(), advance_metadata=0, expected_error=False):
        task = self.root / 'task_001'
        (task / 'workspace').mkdir(parents=True, exist_ok=True)
        (task / 'task.json').write_text(json.dumps({'task_type': 'math',
            'instruction': 'Compute seventeen plus twenty-five.', 'reward': 5, 'check': 'exact_int'}),
            encoding='utf-8')
        trace = self.root / 'trace.jsonl'
        clock, llms = Clock(), []
        constructor, read_text = runtime.JiuwenLLM, Path.read_text
        FakeSDKModel.instances = []

        def construct(*args, **kwargs):
            llm = constructor(*args, **kwargs)
            llms.append(llm)
            return llm

        def read(path, *args, **kwargs):
            if path == task / 'task.json':
                clock.advance(advance_metadata)
            return read_text(path, *args, **kwargs)

        argv = ['run_agent.py', '--tasks-dir', str(self.root), '--trace-file', str(trace),
                '--initial-model', MODEL, '--budget', '100000', '--confidence-policy', 'off',
                '--memory-policy', 'off', *extra]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.dict(sys.modules, fake_sdk_modules()))
            stack.enter_context(patch.object(runtime, 'time', SimpleNamespace(monotonic=clock)))
            stack.enter_context(patch.object(runtime, 'BudgetLedger',
                                            side_effect=lambda budget: BudgetLedger(budget, clock=clock)))
            stack.enter_context(patch.object(runtime, 'JiuwenLLM', side_effect=construct))
            stack.enter_context(patch.object(Path, 'read_text', new=read))
            local = stack.enter_context(patch.object(runtime, 'deterministic_answer', return_value=None))
            stack.enter_context(patch.object(runtime, 'load_task_cost_model', return_value=(None, 'fixture')))
            choice = SimpleNamespace(total_policy=100, first_call=100, source='fixture', reason=None)
            stack.enter_context(patch.object(runtime, 'task_cost_forecast', return_value=(choice, None)))
            stack.enter_context(patch.dict(os.environ, {'LLM_BASE_URL': 'https://offline.invalid',
                                                        'LLM_API_KEY': 'offline-test'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            if expected_error:
                with self.assertRaises(SystemExit) as error:
                    asyncio.run(runtime.main())
                self.assertEqual(error.exception.code, 2)
                local.assert_not_called()
                self.assertEqual(llms, [])
                self.assertEqual(FakeSDKModel.instances, [])
                self.assertFalse(trace.exists())
                self.assertFalse(list(self.root.rglob('answer.txt')))
            else:
                asyncio.run(runtime.main())
        return llms, task, trace

    def test_invalid_time_limit_rejected_before_any_task_or_sdk_work(self):
        for value in ('0', '-1', 'nan', 'inf', '-inf', '3300.01'):
            with self.subTest(value=value):
                self.launch_cli(extra=('--time-limit=' + value,), expected_error=True)

    def test_default_time_limit_uses_3300_seconds_from_before_metadata(self):
        llms, task, trace = self.launch_cli(advance_metadata=3)
        self.assertEqual(len(llms), 1)
        self.assertEqual(llms[0].deadline, 3400)
        self.assertEqual((task / 'answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_fractional_time_limit_bounds_actual_cli_transport(self):
        llms, task, trace = self.launch_cli(extra=('--time-limit', '120.5'), advance_metadata=3)
        self.assertEqual(len(llms), 1)
        self.assertEqual(llms[0].deadline, 220.5)
        self.assertTrue(llms[0].call_records)
        self.assertLessEqual(llms[0].call_records[0]['timeout'], 117.5)
        self.assertEqual((task / 'answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_normal_cli_defaults_sla_off_and_explicit_adaptive_remains_available(self):
        llms, task, trace = self.launch_cli()
        self.assertEqual(len(llms), 1)
        self.assertFalse(llms[0].sla_enabled)
        self.assertEqual(llms[0].deadline, 3400)
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['timing']['sla_policy'], 'off')
        self.assertNotIn('|sla-v3', rows[0]['policy_id'])
        self.assertEqual((task / 'answer.txt').read_text(encoding='utf-8'), '42\n')

        self.root = self.root / 'explicit_adaptive'
        self.root.mkdir()
        llms, task, trace = self.launch_cli(extra=('--sla-policy', 'adaptive'))
        self.assertEqual(len(llms), 1)
        self.assertTrue(llms[0].sla_enabled)
        self.assertEqual(llms[0].deadline, 3400)
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['timing']['sla_policy'], 'adaptive')
        self.assertIn('|sla-v3', rows[0]['policy_id'])
        self.assertEqual((task / 'answer.txt').read_text(encoding='utf-8'), '42\n')

    def launch_queue(self, *, sla='adaptive', retry='adaptive', elapsed=1.0, time_limit=120,
                     durations=None, rewards=None, scripted=None, elapsed_by_task=None):
        folder = self.root / ('queue_' + sla + '_' + retry)
        folder.mkdir()
        for ident, domain in (('task_a', 'math'), ('task_b', 'search')):
            task = folder / ident
            (task / 'workspace').mkdir(parents=True)
            (task / 'task.json').write_text(json.dumps({'task_type': domain,
                'instruction': ident + (' Compute the requested total.' if domain == 'math'
                                       else ' Which destination is recorded?'),
                'reward': (rewards or {}).get(ident, 5)}), encoding='utf-8')
            if domain == 'search':
                (task / 'workspace' / 'notes.txt').write_text('The destination was Riga.\n', encoding='utf-8')
        clock, estimator = Clock(), FixtureLatencyEstimator(durations)
        trace = folder / 'trace.jsonl'
        answers = {key: iter(values) for key, values in (scripted or {}).items()}

        def answer(messages, kwargs):
            ident = 'task_b' if 'task_b' in messages[-1]['content'] else 'task_a'
            clock.advance((elapsed_by_task or {}).get(ident, elapsed))
            text = next(answers[ident]) if ident in answers else ('Riga' if ident == 'task_b' else 'CALC: 17+25')
            return response(text)

        argv = ['run_agent.py', '--tasks-dir', str(folder), '--trace-file', str(trace),
                '--initial-model', MODEL, '--budget', '100000', '--confidence-policy', 'off',
                '--memory-policy', 'off', '--time-limit', str(time_limit), '--sla-policy', sla,
                '--retry-policy', retry]
        choice = SimpleNamespace(total_policy=100, first_call=100, source='fixture', reason=None)
        with self.transport(clock=clock, behavior=answer) as (llm, _), ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(runtime, 'JiuwenLLM', return_value=llm))
            stack.enter_context(patch.object(runtime, 'LatencyEstimator', return_value=estimator))
            stack.enter_context(patch.object(runtime, 'deterministic_answer', return_value=None))
            stack.enter_context(patch.object(runtime, 'impossible_reason', return_value=None))
            stack.enter_context(patch.object(runtime, 'load_task_cost_model', return_value=(None, 'fixture')))
            stack.enter_context(patch.object(runtime, 'task_cost_forecast', return_value=(choice, None)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            asyncio.run(runtime.main())
        calls = [('task_b' if 'task_b' in messages[-1]['content'] else 'task_a')
                 for messages, kwargs in FakeSDKModel.instances[0].calls]
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        return folder, llm, calls, rows, estimator

    def test_normal_cli_joint_priority_prefers_short_task_only_under_time_pressure(self):
        _, _, adaptive, _, _ = self.launch_queue()
        _, _, off, _, _ = self.launch_queue(sla='off')
        self.assertEqual(adaptive, ['task_b', 'task_a'])
        self.assertEqual(off, ['task_a', 'task_b'])

    def test_normal_cli_refreshes_admission_after_elapsed_task_time(self):
        folder, llm, calls, rows, _ = self.launch_queue(elapsed=20)
        self.assertEqual(calls, ['task_b'])
        self.assertTrue((folder / 'task_b' / 'answer.txt').exists())
        self.assertFalse((folder / 'task_a' / 'answer.txt').exists())
        self.assertGreater(llm.remaining, 99000)
        self.assertFalse(any(row.get('outcome') == 'impossible' for row in rows))

    def test_classic_disables_joint_policy_without_removing_global_deadline(self):
        _, llm, calls, _, _ = self.launch_queue(retry='classic')
        self.assertEqual(calls, ['task_a', 'task_b'])
        self.assertFalse(llm.sla_enabled)
        self.assertEqual(llm.deadline, 220)

    def test_plentiful_time_preserves_token_priority(self):
        _, _, calls, _, _ = self.launch_queue(time_limit=3300)
        self.assertEqual(calls, ['task_a', 'task_b'])

    def test_sla_deferral_without_confidence_preserves_attempts_and_excludes_queue_wait(self):
        folder, llm, calls, rows, estimator = self.launch_queue(
            time_limit=90, durations={'math': 60, 'search': 40}, rewards={'task_a': 10},
            scripted={'task_a': ['CALC: unknown(17)', 'CALC: unknown(17)', 'UNSOLVABLE']},
            elapsed_by_task={'task_a': 1, 'task_b': 15})
        self.assertEqual(calls, ['task_a', 'task_b', 'task_a'])
        visits = [row for row in rows if row['task_id'] == 'task_a']
        self.assertEqual([row['outcome'] for row in visits], ['deferred', 'skipped'])
        self.assertFalse(visits[0]['execution_complete'])
        self.assertEqual([row['timing']['visit_elapsed_seconds'] for row in visits], [1, 1])
        self.assertEqual(visits[-1]['task_actual_tokens'], 10)
        self.assertEqual(llm.task_progress['attempts'], 2)
        self.assertTrue(all(event['elapsed_seconds'] == 0 for event in estimator.tool_events))
        self.assertFalse((folder / 'task_a' / 'answer.txt').exists())
        self.assertEqual((folder / 'task_b' / 'answer.txt').read_text(encoding='utf-8'), 'Riga\n')

    def test_sla_deferred_sql_history_survives_other_task_with_confidence_off(self):
        first, second = self.root / 'first', self.root / 'second'
        for workspace, amounts in ((first, '17\n25\n'), (second, '1\n2\n')):
            workspace.mkdir()
            (workspace / 'values.csv').write_text('amount\n' + amounts, encoding='utf-8')
        answers = iter(['SELECT missing FROM data', 'SELECT SUM(amount) FROM data',
                        'SELECT missing FROM data'])
        task_a = {'task_type': 'data_analysis', 'instruction': 'Total for the first dataset?'}
        task_b = {'task_type': 'data_analysis', 'instruction': 'Total for the second dataset?'}
        with self.transport(deadline=300, behavior=lambda messages, kwargs: response(next(answers))) as (llm, _):
            llm.sla_enabled = True
            llm.latency_estimator = FixtureLatencyEstimator({'data_analysis': 1})
            llm.task_kind = 'data_analysis'
            reasoning = MODELS['large_reason']
            self.assertFalse(llm.confidence_enabled)
            self.assertIsNone(asyncio.run(runtime.model_answer(llm, task_a, first, reasoning)))
            saved = llm.task_progress
            llm.task_progress = {}
            self.assertEqual(asyncio.run(runtime.model_answer(llm, task_b, second, reasoning)), '3')
            llm.task_progress = saved
            self.assertIsNone(asyncio.run(runtime.model_answer(llm, task_a, first, reasoning)))
            self.assertEqual(llm.task_state, 'stalled')
            last_prompt = FakeSDKModel.instances[0].calls[-1][0][-1]['content']
            self.assertIn('Previous query: SELECT missing FROM data', last_prompt)
            self.assertIn('no such column', last_prompt)
            self.assertNotIn('second dataset', last_prompt)

    def test_plain_sql_sla_exposes_each_repair_to_outer_controller_with_confidence_off(self):
        (self.root / 'values.csv').write_text('amount\n17\n25\n', encoding='utf-8')
        task = {'task_type': 'data_analysis', 'instruction': 'Compute the total amount.',
                'check': 'exact_int'}
        for enabled in (True, False):
            with self.subTest(sla_enabled=enabled):
                answers = iter(['SELECT missing FROM data', 'SELECT SUM(amount) FROM data'])
                with self.transport(deadline=300, behavior=lambda messages, kwargs: response(next(answers))) as (llm, _):
                    llm.sla_enabled = enabled
                    llm.latency_estimator = FixtureLatencyEstimator({'data_analysis': 1})
                    llm.task_kind = 'data_analysis'
                    self.assertFalse(llm.confidence_enabled)
                    first = asyncio.run(runtime.model_answer(llm, task, self.root, MODEL))
                    if enabled:
                        self.assertIsNone(first)
                        self.assertEqual(llm.task_state, 'check_failed')
                        self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
                        self.assertEqual(len(llm.call_records), 1)
                        repaired = asyncio.run(runtime.model_answer(llm, task, self.root, MODEL))
                        self.assertEqual(repaired, '42')
                    else:
                        self.assertEqual(first, '42')
                    self.assertEqual(len(FakeSDKModel.instances[0].calls), 2)
                    last_prompt = FakeSDKModel.instances[0].calls[-1][0][-1]['content']
                    self.assertIn('Previous query: SELECT missing FROM data', last_prompt)
                    self.assertIn('no such column', last_prompt)

    def test_time_cap_reduces_actual_generation_before_reservation(self):
        with self.transport(deadline=120) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'math'
            llm.latency_estimator = LinearLatencyEstimator()
            asyncio.run(llm.ask('', '', max_tokens=400, model=MODEL, min_output_tokens=32))
            cap = FakeSDKModel.instances[0].calls[0][1]['max_tokens']
            self.assertGreaterEqual(cap, 32)
            self.assertLess(cap, 400)
            self.assertEqual(llm.call_records[0]['max_tokens'], cap)
            self.assertTrue(any(event['reason'] == 'generation_cap_reduced_for_time'
                                for event in llm.sla_decisions))

    def test_long_english_prompt_uses_expected_latency_but_keeps_full_budget_reserve(self):
        system, user = 'Return only the final answer.', 'A' * 8500
        prompt_bytes = len(system.encode('utf-8')) + len(user.encode('utf-8'))
        estimator = LatencyEstimator()
        old_bound = estimator.forecast(model=MODEL, domain='search',
            predicted_tokens=prompt_bytes / 10 + 128 + 32, estimated_calls=1)
        expected = estimator.forecast(model=MODEL, domain='search',
            predicted_tokens=(len(system) + len(user)) / 35 + 18 + 32, estimated_calls=1)
        self.assertGreater(old_bound.seconds, 7)
        self.assertLess(expected.seconds, 7)
        with self.transport(deadline=107, behavior=lambda messages, kwargs: response(usage=False)) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'search'
            llm.latency_estimator = estimator
            asyncio.run(llm.ask(system, user, max_tokens=400, model=MODEL, min_output_tokens=32))
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            record = llm.call_records[0]
            self.assertGreaterEqual(record['max_tokens'], 32)
            self.assertLess(record['max_tokens'], 400)
            reserve_units = prompt_bytes + 1280 + 20 * record['max_tokens']
            self.assertEqual(record['reserved_units'], reserve_units)
            self.assertEqual(record['charged_units'], reserve_units)
            self.assertAlmostEqual(llm.spent, reserve_units / 10)

    def test_unicode_changes_byte_reserve_not_character_time_cap_and_more_time_allows_growth(self):
        caps, reserves = {}, {}
        for label, user in (('ascii', 'A' * 8500), ('unicode', '\u0416' * 8500)):
            for seconds in (7, 10):
                with self.subTest(text=label, seconds=seconds), self.transport(
                        deadline=100 + seconds, behavior=lambda messages, kwargs: response(usage=False)) as (llm, _):
                    llm.sla_enabled = True
                    llm.task_kind = 'search'
                    llm.latency_estimator = LatencyEstimator()
                    asyncio.run(llm.ask('', user, max_tokens=400, model=MODEL, min_output_tokens=32))
                    record = llm.call_records[0]
                    caps[label, seconds] = record['max_tokens']
                    reserves[label, seconds] = record['charged_units']
                    self.assertEqual(record['reserved_units'],
                                     len(user.encode('utf-8')) + 1280 + 20 * record['max_tokens'])
                    self.assertEqual(record['charged_units'], record['reserved_units'])
        for seconds in (7, 10):
            self.assertEqual(caps['ascii', seconds], caps['unicode', seconds])
            self.assertEqual(reserves['unicode', seconds] - reserves['ascii', seconds], 8500)
        self.assertGreater(caps['ascii', 10], caps['ascii', 7])
        self.assertLessEqual(caps['ascii', 10], 400)

    def test_long_prompt_truly_below_minimum_time_never_calls_or_reserves(self):
        with self.transport(deadline=105) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'search'
            llm.latency_estimator = LatencyEstimator()
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(llm.ask('Return only the final answer.', 'A' * 8500,
                                    max_tokens=400, model=MODEL, min_output_tokens=32))
            self.assertEqual(llm.call_records, [])
            self.assertEqual(llm.spent, 0)
            self.assertEqual(FakeSDKModel.instances[0].calls, [])
            self.assertEqual(llm.sla_decisions[-1]['reason'], 'minimum_generation_exceeds_time')

    def test_expected_time_admission_keeps_upper_diagnostic_and_actual_timeout_enforced(self):
        estimator = LatencyEstimator()
        forecast = estimator.forecast(model=MODEL, domain='search',
                                       predicted_tokens=18 + 128, estimated_calls=1)
        self.assertLess(forecast.seconds, 5)
        self.assertGreater(forecast.conservative_seconds, 5)
        with self.transport(deadline=105, behavior=lambda messages, kwargs: response(usage=False)) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'search'
            llm.latency_estimator = estimator
            asyncio.run(llm.ask('', '', max_tokens=128, model=MODEL, min_output_tokens=32))
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            self.assertEqual(llm.call_records[0]['max_tokens'], 128)
            self.assertEqual(llm.call_records[0]['reserved_units'], 1280 + 20 * 128)
            self.assertEqual(llm.call_records[0]['charged_units'], 1280 + 20 * 128)

        cancelled = []

        async def delayed_response():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return response()

        optimistic = SimpleNamespace(
            forecast=lambda **kwargs: LatencyForecast(0.001, 100, 1, 'optimistic_fixture', 0),
            observe_call=lambda **kwargs: True)
        with self.transport(deadline=100.02, behavior=lambda messages, kwargs: delayed_response()) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'search'
            llm.latency_estimator = optimistic
            with self.assertRaises(TimeoutError):
                asyncio.run(llm.ask('', '', max_tokens=128, model=MODEL, min_output_tokens=32))
            self.assertEqual(cancelled, [True])
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            record = llm.call_records[0]
            self.assertAlmostEqual(record['timeout'], 0.02)
            self.assertEqual(record['status'], 'reserved_timeout')
            self.assertTrue(record['latency_censored'])
            self.assertEqual(record['charged_units'], record['reserved_units'])
            self.assertGreater(llm.spent, 0)

    def test_minimum_generation_not_fitting_time_never_reserves_or_calls(self):
        with self.transport(deadline=101) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'math'
            llm.latency_estimator = LinearLatencyEstimator()
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(llm.ask('', '', max_tokens=400, model=MODEL, min_output_tokens=32))
            self.assertEqual(llm.call_records, [])
            self.assertEqual(llm.spent, 0)
            self.assertEqual(FakeSDKModel.instances[0].calls, [])

    def test_truncated_reply_with_no_possible_cap_growth_stops_before_second_charge(self):
        reply = response()
        reply.finish_reason = 'length'
        with self.transport(deadline=105, behavior=lambda messages, kwargs: reply) as (llm, _):
            llm.sla_enabled = True
            llm.task_kind = 'math'
            llm.latency_estimator = LinearLatencyEstimator()
            asyncio.run(llm.ask('', '', max_tokens=400, model=MODEL, min_output_tokens=32))
            self.assertEqual(llm.last_cap, 32)
            spent = llm.spent
            with self.assertRaises(DeadlineExceeded):
                asyncio.run(llm.ask('', '', max_tokens=800, model=MODEL, min_output_tokens=32))
            self.assertEqual(len(FakeSDKModel.instances[0].calls), 1)
            self.assertEqual(len(llm.call_records), 1)
            self.assertEqual(llm.spent, spent)

    def test_sla_never_revives_a_retry_rejected_by_existing_guards(self):
        decision = RetryDecision(False, MODEL, 0, 'same_candidate_or_failure', 0, 0, 'stop')
        llm = SimpleNamespace(sla_enabled=True,
                              latency_estimator=Mock(side_effect=AssertionError('must not estimate')))
        self.assertIs(runtime.apply_sla_retry(llm, decision, kind='math', reward=5), decision)

    def test_token_affordable_retry_stops_if_remaining_time_is_insufficient(self):
        decision = RetryDecision(True, MODEL, 512, 'repair_tool_error', 150, 0.6)
        with self.transport(deadline=105) as (llm, _):
            llm.sla_enabled = True
            llm.latency_estimator = FixtureLatencyEstimator({'math': 10})
            stopped = runtime.apply_sla_retry(llm, decision, kind='math', reward=5)
            self.assertFalse(stopped.retry)
            self.assertEqual(stopped.action, 'stop')
            self.assertEqual(stopped.reason, 'sla_global_deadline_insufficient')
            self.assertEqual(llm.spent, 0)

    def test_marginal_retry_defers_for_faster_alternative_only_under_time_pressure(self):
        decision = RetryDecision(True, MODEL, 512, 'repair_tool_error', 150, 0.6)
        with self.transport(deadline=190) as (llm, _):
            llm.sla_enabled = True
            llm.latency_estimator = FixtureLatencyEstimator({'math': 60, 'search': 40})
            alternative = TaskPlan('other', 'search', 5, 100, 0.73, MODEL)
            llm.sla_plans = [('other', alternative, {'first_call': 100, 'estimated_request_count': 1})]
            deferred = runtime.apply_sla_retry(llm, decision, kind='math', reward=5)
            self.assertFalse(deferred.retry)
            self.assertEqual(deferred.action, 'defer')
            self.assertEqual(deferred.reason, 'sla_other_tasks_have_higher_joint_value')
            llm.deadline = 3400
            self.assertIs(runtime.apply_sla_retry(llm, decision, kind='math', reward=5), decision)

    def test_real_call_timing_updates_only_actual_model_and_current_domain(self):
        clock = Clock()

        def answer(messages, kwargs):
            clock.advance(1)
            return response()

        with self.transport(clock=clock, deadline=500, behavior=answer) as (llm, _):
            estimator = LatencyEstimator()
            llm.latency_estimator = estimator
            llm.task_kind = 'math'
            self.ask(llm)
            llm.task_kind = 'search'
            reasoning = MODELS['large_reason']
            asyncio.run(llm.ask('', '', model=reasoning, max_tokens=128))
            for model, domain, count in ((MODEL, 'math', 1), (MODEL, 'search', 0),
                                         (reasoning, 'search', 1), (reasoning, 'math', 0)):
                with self.subTest(model=model, domain=domain):
                    self.assertEqual(estimator.forecast(model=model, domain=domain,
                                                        predicted_tokens=100).observations, count)


if __name__ == '__main__':
    unittest.main()
