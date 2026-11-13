"""Exercise the real CLI, ledger and admission with a network-free SDK model."""
import asyncio
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from unittest.mock import patch

import run_agent as runtime
from budget_runtime import BudgetLedger
from forecast_guard import ForecastChoice

ABSENT = object()
REAL_LLM = runtime.JiuwenLLM
REAL_FORECAST = runtime.task_cost_forecast


class FakeSDKModel:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        type(self).instances.append(self)

    async def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return types.SimpleNamespace(content='CALC: 17+25', finish_reason='stop',
                                     usage_metadata={'input_tokens': 10, 'output_tokens': 4,
                                                     'total_tokens': 14})


def fake_sdk_modules():
    """Only replace the SDK constructor/invoke boundary; real budget code runs."""
    leaves = {
        'openjiuwen.core.common.logging.log_config': {
            'configure_log_config': lambda value: None},
        'openjiuwen.core.foundation.llm.model': {'Model': FakeSDKModel},
        'openjiuwen.core.foundation.llm.schema.config': {
            'ModelClientConfig': lambda **kwargs: types.SimpleNamespace(**kwargs),
            'ModelRequestConfig': lambda **kwargs: types.SimpleNamespace(**kwargs)},
    }
    modules = {}
    for leaf, fields in leaves.items():
        parts = leaf.split('.')
        for count in range(1, len(parts) + 1):
            name = '.'.join(parts[:count])
            if name not in modules:
                modules[name] = types.ModuleType(name)
                modules[name].__path__ = []
            if count > 1:
                setattr(modules['.'.join(parts[:count - 1])], parts[count - 1], modules[name])
        for name, value in fields.items():
            setattr(modules[leaf], name, value)
    return modules


class BudgetConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sequence = 0

    def launch(self, *, env_budget=ABSENT, cli=(), expected_error=False,
               forecast_total=None, task_count=1, offline=False):
        self.sequence += 1
        root = self.root / str(self.sequence)
        for index in range(task_count):
            task_dir = root / ('task_' + str(index))
            (task_dir / 'workspace').mkdir(parents=True)
            (task_dir / 'task.json').write_text(json.dumps({
                'task_type': 'math', 'instruction': 'Compute 17 plus 25.',
                'reward': 5, 'check': 'exact_int'}), encoding='utf-8')
        trace = root / 'trace.jsonl'
        argv = ['run_agent.py', '--tasks-dir', str(root), '--trace-file', str(trace)] + list(cli)
        if offline:
            argv.append('--offline')
        environment = dict(os.environ)
        environment.pop('BUDGET', None)
        if env_budget is not ABSENT:
            environment['BUDGET'] = env_budget
        environment.update(LLM_BASE_URL='https://offline.invalid', LLM_API_KEY='offline-test')
        llms, initial_remaining, forecast_budgets = [], [], []
        FakeSDKModel.instances = []

        def construct(budget, **kwargs):
            llm = REAL_LLM(budget, **kwargs)
            llms.append(llm)
            initial_remaining.append(llm.remaining)
            return llm

        def forecast(*args, **kwargs):
            forecast_budgets.append(kwargs['pipeline_budget'])
            if forecast_total is not None:
                return ForecastChoice(forecast_total, forecast_total, 'test', None), None
            return REAL_FORECAST(*args, **kwargs)

        stderr, stdout = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, fake_sdk_modules()), \
                patch.object(sys, 'argv', argv), patch.object(runtime, 'JiuwenLLM', side_effect=construct), \
                patch.object(runtime, 'BudgetLedger', wraps=BudgetLedger) as ledger, \
                patch.object(runtime, 'task_cost_forecast', side_effect=forecast), \
                patch.object(runtime, 'deterministic_answer', wraps=runtime.deterministic_answer) as local, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            if expected_error:
                with self.assertRaises(SystemExit) as error:
                    asyncio.run(runtime.main())
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(local.call_count, 0, 'invalid budget reached task processing')
                self.assertEqual(ledger.call_count, 0)
                self.assertFalse(llms)
                self.assertFalse(FakeSDKModel.instances)
                self.assertFalse(list(root.rglob('answer.txt')))
                self.assertFalse(list(root.rglob('*.tmp')))
                self.assertFalse(trace.exists())
                self.assertEqual(stdout.getvalue(), '')
            else:
                asyncio.run(runtime.main())
            ledger_args = ledger.call_args_list
        return types.SimpleNamespace(root=root, trace=trace, llms=llms,
                                     initial_remaining=initial_remaining,
                                     forecast_budgets=forecast_budgets,
                                     ledger_args=ledger_args, stderr=stderr.getvalue(), stdout=stdout.getvalue(),
                                     calls=sum(len(model.calls) for model in FakeSDKModel.instances))

    def assert_selected_budget(self, result, value):
        value = Decimal(str(value))
        self.assertEqual(len(result.llms), 1)
        self.assertEqual(result.llms[0].budget, value)
        self.assertEqual(result.ledger_args[0].args, (value,))
        units = int((Decimal(str(value)) * 10).to_integral_value(rounding=ROUND_FLOOR))
        self.assertEqual(result.llms[0].ledger.limit_units, units)
        self.assertEqual(result.initial_remaining, [units / 10])
        self.assertTrue(result.forecast_budgets)
        self.assertTrue(all(budget == value for budget in result.forecast_budgets))

    def test_absent_env_and_cli_use_full_100000_fallback(self):
        result = self.launch()
        self.assert_selected_budget(result, 100000)
        self.assertIn('token budget: 100000 (source: default)', result.stdout)
        self.assertEqual(result.calls, 1)
        self.assertEqual((result.root / 'task_0/answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_env_smaller_and_larger_than_100k_reach_ledger_and_forecast(self):
        for value in (50000, 250000):
            with self.subTest(value=value):
                result = self.launch(env_budget=str(value))
                self.assert_selected_budget(result, value)
                self.assertIn('(source: BUDGET)', result.stdout)
                self.assertEqual(result.calls, 1)

    def test_fractional_env_keeps_existing_ledger_tenth_token_precision(self):
        for value in ('73000.0', '73000.5', '73000.59'):
            with self.subTest(value=value):
                result = self.launch(env_budget=value)
                self.assert_selected_budget(result, Decimal(value))
                self.assertEqual(result.calls, 1)

    def test_env_and_cli_do_not_round_up_across_decitoken_boundary(self):
        text = '73000.09999999999999'
        for settings in ({'env_budget': text}, {'env_budget': 'NaN', 'cli': ['--budget', text]}):
            with self.subTest(settings=settings):
                result = self.launch(**settings)
                self.assert_selected_budget(result, Decimal(text))
                self.assertIsInstance(result.ledger_args[0].args[0], Decimal)
                self.assertEqual(result.llms[0].ledger.limit_units, 730000)
                self.assertEqual(result.initial_remaining, [73000.0])
                self.assertEqual(result.calls, 1)
                self.assertIn(text, result.stdout)

    def test_parser_has_no_arbitrary_maximum_or_float_underflow(self):
        self.assertEqual(runtime.positive_budget('1e309'), Decimal('1e309'))
        self.assertEqual(runtime.positive_budget('1e-1000'), Decimal('1e-1000'))

    def test_explicit_cli_overrides_valid_env(self):
        result = self.launch(env_budget='50000', cli=['--budget', '175000.5'])
        self.assert_selected_budget(result, 175000.5)
        self.assertIn('token budget: 175000.5 (source: CLI --budget)', result.stdout)

    def test_valid_cli_overrides_even_invalid_or_empty_env(self):
        for value in ('', 'NaN', '0', '-2', 'not-a-budget'):
            with self.subTest(value=value):
                result = self.launch(env_budget=value, cli=['--budget', '73000.5'])
                self.assert_selected_budget(result, 73000.5)
                self.assertEqual(result.calls, 1)

    def test_invalid_env_fails_before_api_local_solver_or_any_answer_write(self):
        for value in ('', ' ', 'NaN', 'Inf', '-Infinity', '0', '-0.5', '1,000', 'words', '1e'):
            with self.subTest(value=value):
                result = self.launch(env_budget=value, expected_error=True)
                self.assertIn('BUDGET must be a positive finite number', result.stderr)

    def test_invalid_env_is_rejected_even_for_offline_run(self):
        result = self.launch(env_budget='', offline=True, expected_error=True)
        self.assertIn('BUDGET must be a positive finite number', result.stderr)

    def test_invalid_cli_does_not_fall_back_to_valid_env(self):
        for value in ('NaN', 'inf', '0', '-1', '', '1,000', 'words'):
            with self.subTest(value=value):
                result = self.launch(env_budget='100000', cli=['--budget', value], expected_error=True)
                self.assertIn('argument --budget: must be a positive finite number', result.stderr)

    def test_admission_uses_selected_budget_not_hardcoded_100000(self):
        small = self.launch(env_budget='50000', forecast_total=120000)
        large = self.launch(env_budget='200000', forecast_total=120000)
        self.assert_selected_budget(small, 50000)
        self.assert_selected_budget(large, 200000)
        self.assertEqual(small.calls, 0)
        self.assertFalse(list(small.root.rglob('answer.txt')))
        self.assertEqual(large.calls, 1)
        self.assertEqual((large.root / 'task_0/answer.txt').read_text(encoding='utf-8'), '42\n')

    def test_budget_is_shared_by_tasks_with_no_fixed_reserve(self):
        result = self.launch(env_budget='2000.5', task_count=2)
        self.assert_selected_budget(result, 2000.5)
        self.assertEqual(len(result.ledger_args), 1)
        self.assertEqual(result.calls, 2)
        self.assertEqual(result.llms[0].spent, 10)
        self.assertEqual(result.llms[0].remaining, 1990.5)
        self.assertEqual(len(list(result.root.glob('*/answer.txt'))), 2)

    def test_positive_sub_tenth_budget_is_valid_but_admits_no_paid_task(self):
        result = self.launch(env_budget='0.01')
        self.assert_selected_budget(result, 0.01)
        self.assertEqual(result.calls, 0)
        self.assertEqual(result.llms[0].remaining, 0)


if __name__ == '__main__':
    unittest.main()