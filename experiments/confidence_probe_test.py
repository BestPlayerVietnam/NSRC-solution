"""No-network checks for fresh paired-arm accounting and oracle isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import confidence_probe as probe


class ConfidenceProbeTests(unittest.TestCase):
    def test_four_positive_and_four_certified_negative_cases(self):
        cases = probe.cases()
        self.assertEqual(len(cases), 8)
        for domain in ('math', 'search', 'data_analysis', 'coding'):
            selected = [case for case in cases if case['domain'] == domain]
            self.assertEqual(len(selected), 2)
            self.assertEqual(sum(case['expected'] is not None for case in selected), 1)
            self.assertTrue(next(case for case in selected if case['expected'] is None)['certificate'])

    def test_grading_requires_all_completed_arms_before_oracle_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(probe, 'verify', return_value={'arms': {'pending': {}}}):
                with self.assertRaisesRegex(RuntimeError, 'All eight arms'):
                    probe.report(root/'absent-oracle', root, 'unused')

    def test_report_sums_all_deferred_visits_and_keeps_first_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, run = root/'suite', root/'run'
            suite.mkdir();run.mkdir()
            ident = 'task-a'
            task = {'id': ident, 'domain': 'math', 'task_path': 'tasks/'+ident}
            probe.write_new(suite/'manifest.json', {'tasks': [task]})
            probe.write_new(suite/'oracle.json', {ident: {'answerable': True, 'expected': '42'}})
            protocol = {'arms': {}}
            events = []
            for confidence in ('off', 'heuristic'):
                for alias, model in probe.base.MODEL_ALIASES.items():
                    arm = confidence+'_'+alias
                    protocol['arms'][arm] = {'initial_model': model, 'confidence_policy': confidence}
                    folder = run/arm
                    (folder/'tasks'/ident).mkdir(parents=True)
                    (folder/'tasks'/ident/'answer.txt').write_text('42\n', encoding='utf-8')
                    probe.write_new(folder/'completed.json', {'returncode': 0})
                    probe.write_new(folder/'ledger_end.json', {'models': {model: {'charged_units': 300}}})
                    visits = [{'task_id': ident, 'actual_tokens': 10, 'predicted_tokens': 100,
                               'outcome': 'deferred', 'self_assessments': [{}]},
                              {'task_id': ident, 'actual_tokens': 20, 'predicted_tokens': 999,
                               'outcome': 'answered_unverified', 'self_assessments': [{}]}]
                    (folder/'tasks/trace.jsonl').write_text(
                        '\n'.join(json.dumps(visit) for visit in visits), encoding='utf-8')
                    for index, cost in enumerate((10, 20)):
                        call_id = arm+str(index)
                        common = {'id': call_id, 'arm': arm, 'task': ident, 'model': model}
                        events.append({**common, 'event': 'reserved', 'reserved': 500})
                        events.append({**common, 'event': 'completed', 'cost': cost,
                                       'accounting': 'provider_reconciled'})
            probe.write_new(run/'protocol.json', protocol)
            (run/'calls.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
            with patch.object(probe, 'verify', return_value=protocol), contextlib.redirect_stdout(io.StringIO()):
                probe.report(suite, run, 'unused')
            summary = probe.read(run/'evaluation.json')
            for arm in summary['arms'].values():
                row = arm['rows'][0]
                self.assertEqual((row['cost'], row['traced_cost_all_visits'], row['initial_forecast']),
                                 (30, 30, 100))
                self.assertEqual((row['visits'], row['calls'], row['assessments']), (2, 2, 2))
                self.assertEqual(arm['total_cost'], 30)
            self.assertTrue(summary['comparison_complete'])

    def test_frozen_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'output.json'
            probe.write_new(path, {'first': True})
            with self.assertRaises(FileExistsError):
                probe.write_new(path, {'replacement': True})
            self.assertEqual(probe.read(path), {'first': True})


if __name__ == '__main__':
    unittest.main()
