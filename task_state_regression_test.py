"""Task-boundary and repair rollback regressions using fake transports only."""
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as agent
from planner import MODELS, TaskPlan


class FakeEstimator:
    def __init__(self, *args, **kwargs):
        self.cost_observations = {}

    def estimate(self, task, workspace):
        return TaskPlan(task['task_id'], 'math', 4, 100, .8,
                        MODELS['large_reason'], raw_predicted_tokens=100)

    def observe_plan(self, *args, **kwargs):
        pass


class FailingRetryLLM:
    def __init__(self, error):
        self.error = error
        self.spent = 0
        self.remaining = 100000
        self.call_records = []
        self.abort_run = False
        # Deliberately dirty startup fields also have to be cleared by main.
        self.last_finish_reason = 'length'
        self.last_feedback = 'old task error'
        self.last_cap = 32768
        self.output_cap_override = 32768
        self.caps = []
        self.before_calls = []
        self.trace_overrides = []

    async def ask(self, system, user, **kwargs):
        self.before_calls.append((self.last_finish_reason, self.last_feedback,
                                  self.last_cap, self.output_cap_override))
        self.caps.append((user, kwargs['max_tokens']))
        self.last_cap = kwargs['max_tokens']
        if len(self.caps) == 1:
            self.last_finish_reason = 'length'
            return 'CALC: 1'
        if len(self.caps) == 2:
            raise self.error
        self.last_finish_reason = 'stop'
        return 'CALC: 7'


class TaskStateRegressionTests(unittest.TestCase):
    def run_two_tasks(self, error):
        llm = FailingRetryLLM(error)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ident in ('a', 'b'):
                folder = root / ident
                (folder / 'workspace').mkdir(parents=True)
                (folder / 'task.json').write_text(json.dumps({
                    'task_id': ident, 'task_type': 'math', 'instruction': ident, 'reward': 4}),
                    encoding='utf-8')
            argv = ['run_agent', '--tasks-dir', str(root), '--initial-model', MODELS['large_reason']]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.dict(os.environ, {
                    'LLM_BASE_URL': 'http://fixture.invalid', 'LLM_API_KEY': 'fixture', 'BUDGET': '100000'}))
                stack.enter_context(patch.object(agent, 'JiuwenLLM', return_value=llm))
                stack.enter_context(patch.object(agent, 'TaskEstimator', FakeEstimator))
                stack.enter_context(patch.object(agent, 'deterministic_answer', return_value=None))
                stack.enter_context(patch.object(agent, 'impossible_reason', return_value=None))
                stack.enter_context(patch.object(agent, 'record_trace',
                    side_effect=lambda *args: llm.trace_overrides.append(llm.output_cap_override)))
                stack.enter_context(patch.object(agent, 'load_task_cost_model', return_value=(None, 'fixture')))
                stack.enter_context(patch.object(agent, 'analyze_task', return_value=SimpleNamespace(flags=())))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                asyncio.run(agent.main())
            self.assertFalse((root / 'a/workspace/answer.txt').exists())
            self.assertEqual((root / 'b/workspace/answer.txt').read_text(encoding='utf-8'), '7\n')
        return llm

    def test_transport_exception_does_not_leak_retry_cap_or_feedback_to_next_task(self):
        llm = self.run_two_tasks(RuntimeError('fixture transport failure'))
        self.assertEqual(llm.caps, [('a', 3072), ('a', 6144), ('b', 3072)])
        self.assertEqual(llm.before_calls[0], (None, '', 0, None))
        self.assertEqual(llm.before_calls[2], (None, '', 0, None))
        # The failing task's own final trace is emitted before the next reset.
        self.assertIsNone(llm.trace_overrides[0])
        self.assertIsNone(llm.output_cap_override)

    def test_budget_exception_also_restores_override_before_next_task(self):
        llm = self.run_two_tasks(agent.BudgetExhausted('fixture budget guard'))
        self.assertEqual(llm.caps[-1], ('b', 3072))
        self.assertIsNone(llm.trace_overrides[0])
        self.assertIsNone(llm.output_cap_override)

    def test_local_repair_rolls_back_original_bytes_if_checker_raises(self):
        original = b'def total(xs):\r\n    s = 0\r\n    for x in xs:\r\n        s -= x\r\n    return s\r\n'
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / 'mylib.py'
            source.write_bytes(original)
            with patch.object(agent, 'run_tests', side_effect=OSError('fixture checker failure')):
                with self.assertRaisesRegex(OSError, 'fixture checker failure'):
                    agent.try_known_repairs(workspace)
            self.assertEqual(source.read_bytes(), original)

    def test_local_repair_rolls_back_failed_check_and_keeps_confirmed_success(self):
        original = b'def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n'
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / 'mylib.py'
            source.write_bytes(original)
            with patch.object(agent, 'run_tests', return_value=(False, 'test failure')):
                self.assertFalse(agent.try_known_repairs(workspace))
            self.assertEqual(source.read_bytes(), original)
            with patch.object(agent, 'run_tests', return_value=(True, 'one test passed')):
                self.assertTrue(agent.try_known_repairs(workspace))
            self.assertIn('return sum(xs)', source.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
