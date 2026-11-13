"""Offline coding confidence integration with real isolated pytest checks."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import run_agent as runtime
from planner import MODELS


def envelope(result, *, confidence='low', action='continue'):
    return json.dumps({
        'protocol': 'nsrc-assessment-v1',
        'assessment': {
            'action': action,
            'confidence': confidence,
            'reason_code': 'uncertain' if confidence == 'low' else 'solvable',
            'next_step': 'Repair arithmetic in both source modules.',
        },
        'result': result,
    })


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.confidence_enabled = True
        self.last_finish_reason = 'stop'
        self.last_feedback = ''
        self.last_assessment = None
        self.task_progress = {}
        self.assessment_records = []
        self.retry_decisions = []
        self.defer_requested = False
        self.initial_model = MODELS['large_plain']
        self.fixed_model = True
        self.retry_policy = 'adaptive'
        self.remaining = 100000.0
        self.competing_cost = 0.0
        self.opportunity = 0.0
        self.before_call = None

    async def ask(self, system, user, **kwargs):
        if self.before_call is not None:
            self.before_call(len(self.calls))
        self.calls.append({
            'system': system,
            'user': user,
            'attempts_before_call': self.task_progress.get('attempts', 0),
            **kwargs,
        })
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError('Unexpected additional model call') from None


class NewConfidenceCodingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        # Deliberate CRLF bytes make rollback stronger than text equality.
        self.originals = {
            'pkg/__init__.py': b'',
            'pkg/adding.py': b'def plus(a, b):\r\n    return a - b\r\n',
            'pkg/scaling.py': b'def times(a, b):\r\n    return a + b\r\n',
            'tests/test_arithmetic.py': (
                b'from pkg.adding import plus\r\n'
                b'from pkg.scaling import times\r\n'
                b'def test_plus():\r\n    assert plus(2, 3) == 5\r\n'
                b'def test_times():\r\n    assert times(2, 3) == 6\r\n'),
        }
        for name, data in self.originals.items():
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.task = {
            'task_type': 'coding',
            'instruction': 'Fix plus in pkg/adding.py and times in pkg/scaling.py.',
            'reward': 1,
        }
        self.correct_files = {
            'pkg/adding.py': 'def plus(a, b):\n    return a + b\n',
            'pkg/scaling.py': 'def times(a, b):\n    return a * b\n',
        }
        self.wrong_files = {
            'pkg/adding.py': 'def plus(a, b):\n    return 0\n',
            'pkg/scaling.py': 'def times(a, b):\n    return 0\n',
        }

    def solve(self, llm):
        return asyncio.run(runtime.solve_coding(llm, self.task, self.workspace))

    def assert_originals(self):
        for name, expected in self.originals.items():
            self.assertEqual((self.workspace / name).read_bytes(), expected, name)

    def test_low_assessment_with_multifile_edits_passes_real_tests_once(self):
        edits = [
            {'path': 'pkg/adding.py', 'old': 'return a - b', 'new': 'return a + b'},
            {'path': 'pkg/scaling.py', 'old': 'return a + b', 'new': 'return a * b'},
        ]
        llm = FakeLLM(envelope({'edits': edits}))
        self.assertTrue(self.solve(llm))
        self.assertEqual(llm.task_state, 'public_tests_passed')
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.task_progress['attempts'], 1)
        self.assertEqual(llm.last_assessment.confidence, 'low')
        for name, source in self.correct_files.items():
            self.assertEqual((self.workspace / name).read_text(encoding='utf-8').strip(), source.strip())
        self.assertEqual((self.workspace / 'tests/test_arithmetic.py').read_bytes(),
                         self.originals['tests/test_arithmetic.py'])

    def test_high_assessment_repeated_wrong_files_stops_and_restores_bytes(self):
        response = envelope({'files': self.wrong_files}, confidence='high')
        llm = FakeLLM(response, response)
        self.assertFalse(self.solve(llm))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.task_progress['attempts'], 2)
        self.assertEqual(llm.task_state, 'retry_stopped')
        self.assertEqual(llm.retry_decisions[-1]['reason'], 'same_candidate_or_failure')
        self.assert_originals()

    def test_deferred_null_assessment_resumes_once_with_preserved_attempts(self):
        llm = FakeLLM(envelope(None, action='defer'), envelope({'files': self.correct_files}))
        llm.remaining = 1000.0
        llm.competing_cost = 900.0
        llm.opportunity = 0.01
        self.assertFalse(self.solve(llm))
        self.assertEqual(llm.task_state, 'deferred')
        self.assertTrue(llm.defer_requested)
        self.assertEqual(llm.task_progress['attempts'], 1)
        progress = llm.task_progress
        self.assert_originals()
        # Match the scheduler's resumed task boundary without clearing history.
        llm.defer_requested = False
        llm.competing_cost = 0.0
        llm.opportunity = 0.0
        llm.before_call = lambda _: self.assert_originals()
        self.assertTrue(self.solve(llm))
        self.assertIs(llm.task_progress, progress)
        self.assertEqual(llm.task_progress['attempts'], 2)
        self.assertEqual([call['attempts_before_call'] for call in llm.calls], [0, 1])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.task_state, 'public_tests_passed')
        self.assertEqual((self.workspace / 'tests/test_arithmetic.py').read_bytes(),
                         self.originals['tests/test_arithmetic.py'])

    def test_malformed_envelope_cannot_modify_protected_or_outside_files(self):
        outside = self.root / 'outside.py'
        outside.write_bytes(b'# outside workspace\r\n')
        malformed = json.loads(envelope(None, action='defer'))
        # An extra field invalidates the assessment envelope. Legacy parsing must
        # still apply its own allowed-path checks to this attempted source map.
        malformed['files'] = {
            'pkg/adding.py': self.correct_files['pkg/adding.py'],
            'tests/test_arithmetic.py': 'def test_bypass(): assert True\n',
            '../outside.py': '# overwritten\n',
        }
        llm = FakeLLM(envelope({'files': self.wrong_files}, confidence='high'),
                      json.dumps(malformed))
        def check_prior_mutation(index):
            if index == 1:
                self.assertNotEqual((self.workspace / 'pkg/adding.py').read_bytes(),
                                    self.originals['pkg/adding.py'])
        llm.before_call = check_prior_mutation
        self.assertFalse(self.solve(llm))
        self.assertEqual(len(llm.calls), 2)
        self.assertIsNone(llm.last_assessment)
        self.assertIn('outside allowed repair set', llm.last_feedback)
        self.assertEqual(llm.task_state, 'retry_stopped')
        self.assertFalse(llm.defer_requested)
        self.assert_originals()
        self.assertEqual(outside.read_bytes(), b'# outside workspace\r\n')
        self.assertFalse(list(self.workspace.rglob('.nsrc-*.tmp')))


if __name__ == '__main__':
    unittest.main()
