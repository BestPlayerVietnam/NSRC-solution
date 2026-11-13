"""Offline integration of selectable frozen cost estimators; never calls an API."""
import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as runtime
from cost_model import CostForecast, load_cost_model
from forecast_guard import choose_forecast
from planner import MODELS
from policy_features import POLICY_ID, extract_features, fallback_cost
from residual_predictor import ResidualModel, load_residual_model

ROOT = Path(runtime.__file__).resolve().parent
MODEL = MODELS['large_plain']
FEATURES = {'instruction_chars': 140, 'context_chars': 350, 'kind_math': 1,
            'source_bytes': 0, 'contradiction_flag': 0}


def artifact(first=20, total=60):
    def head(value):
        return {'method': 'input_remaining', 'global': value, 'domains': {}}
    return {'schema_version': 1, 'policy_id': POLICY_ID,
            'models': {MODEL: {'heads': {'first_call': head(first), 'total_policy': head(total)}}}}


class OfflineLLM:
    def __init__(self, budget, **kwargs):
        self.budget = budget
        self.spent = 0
        self.call_records = []
        self.abort_run = False
        self.last_finish_reason = 'stop'

    @property
    def remaining(self):
        return self.budget - self.spent

    async def ask(self, system, user, **kwargs):
        self.spent += 10
        self.call_records.append({'status': 'provider_reconciled', 'weighted_tokens': 10,
                                  'model': kwargs.get('model')})
        return 'CALC: 17+25'


class ResidualRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def load(self, selector='residual', **kwargs):
        options = dict(selector=selector, guarded_artifact=ROOT / 'cost_artifact.json',
                       residual_artifact=ROOT / 'residual_cost_artifact.json',
                       fixed_model=True, retry_policy='adaptive')
        options.update(kwargs)
        return runtime.load_task_cost_model(**options)

    def forecast(self, predictor, selector='residual', features=None, **kwargs):
        return runtime.task_cost_forecast(
            FEATURES if features is None else features, model=MODEL, domain='math',
            selector=selector, cost_model=predictor, **kwargs)

    def test_selected_residual_matches_frozen_adapter_all_four_models(self):
        predictor, reason = self.load()
        self.assertIsNone(reason)
        direct = load_residual_model(ROOT / 'residual_cost_artifact.json', POLICY_ID)
        for model in MODELS.values():
            with self.subTest(model=model):
                chosen, raw = runtime.task_cost_forecast(
                    FEATURES, model=model, domain='math', selector='residual', cost_model=predictor)
                expected = direct.predict(FEATURES, model=model, policy_id=POLICY_ID,
                                          domain='math', prior=fallback_cost(FEATURES, model))
                self.assertEqual(chosen, expected)
                self.assertEqual(raw, expected)
                self.assertEqual(chosen.source, 'residual')

    def test_residual_points_are_not_clipped_or_replaced_by_budget_guard(self):
        predictor = ResidualModel(artifact(100000, 200000))
        chosen, raw = self.forecast(predictor, pipeline_budget=100)
        self.assertGreater(chosen.total_policy, 200000)
        self.assertEqual(chosen, raw)
        self.assertEqual(chosen.source, 'residual')

    def test_missing_artifact_uses_prior_and_records_reason(self):
        predictor, reason = self.load(residual_artifact=self.root / 'missing.json')
        chosen, raw = self.forecast(predictor, fallback_reason=reason)
        self.assertEqual((chosen.first_call, chosen.total_policy), fallback_cost(FEATURES, MODEL))
        self.assertEqual((chosen.source, chosen.reason), ('prior', 'residual_artifact_missing'))
        self.assertIsNone(raw)

    def test_invalid_json_schema_and_policy_are_safe_prior_fallbacks(self):
        wrong_policy = artifact()
        wrong_policy['policy_id'] = 'different-policy'
        invalid_schema = artifact()
        invalid_schema['schema_version'] = 99
        for value in ('{invalid', json.dumps(wrong_policy), json.dumps(invalid_schema)):
            with self.subTest(value=value):
                path = self.root / 'bad.json'
                path.write_text(value, encoding='utf-8')
                predictor, reason = self.load(residual_artifact=path)
                chosen, raw = self.forecast(predictor, fallback_reason=reason)
                self.assertEqual((chosen.source, chosen.reason), ('prior', 'residual_artifact_invalid'))
                self.assertEqual(chosen.total_policy, fallback_cost(FEATURES, MODEL)[1])
                self.assertIsNone(raw)

    def test_incomplete_artifact_prediction_error_does_not_drop_task(self):
        value = artifact()
        del value['models'][MODEL]['heads']['total_policy']
        chosen, raw = self.forecast(ResidualModel(value))
        self.assertEqual((chosen.source, chosen.reason), ('prior', 'residual_prediction_error'))
        self.assertIsNone(raw)

    def test_untrained_model_keeps_adapter_fallback_reason(self):
        value = artifact()
        value['models'] = {}
        chosen, raw = self.forecast(ResidualModel(value))
        self.assertEqual((chosen.source, chosen.reason), ('prior', 'model_not_trained'))
        self.assertEqual(chosen.total_policy, fallback_cost(FEATURES, MODEL)[1])
        self.assertEqual(chosen, raw)

    def test_routed_or_classic_scope_never_loads_residual_artifact(self):
        for scope in ({'fixed_model': False}, {'retry_policy': 'classic'}):
            with self.subTest(scope=scope):
                with patch('residual_predictor.load_residual_model', side_effect=AssertionError('must not load')):
                    predictor, reason = self.load(**scope)
                chosen, _ = self.forecast(predictor, fallback_reason=reason)
                self.assertEqual((chosen.source, chosen.reason), ('prior', 'unsupported_runtime_policy'))

    def test_guarded_valid_artifact_keeps_previous_selector_result(self):
        predictor, reason = self.load('guarded')
        self.assertIsNone(reason)
        workspace = self.root / 'workspace'
        workspace.mkdir()
        features = extract_features({'task_type': 'math', 'instruction': 'Compute 17 plus 25.'}, workspace)
        chosen, raw = self.forecast(predictor, selector='guarded', features=features)
        old = load_cost_model(ROOT / 'cost_artifact.json', expected_policy_id=POLICY_ID)
        expected_raw = old.predict(features, model=MODEL, policy_id=POLICY_ID, domain='math')
        self.assertEqual(raw, expected_raw)
        self.assertEqual(chosen, choose_forecast(expected_raw, *fallback_cost(features, MODEL)))

    def test_guarded_warning_still_falls_back(self):
        forecast = CostForecast(model=MODEL, policy_id=POLICY_ID, first_call_mean=40,
                                total_policy_mean=100, warnings=('feature_extrapolation:context_chars',))
        predictor = SimpleNamespace(predict=lambda *args, **kwargs: forecast)
        chosen, raw = self.forecast(predictor, selector='guarded')
        self.assertEqual((chosen.source, chosen.reason), ('prior', 'forecast_warning'))
        self.assertEqual(raw, forecast)

    def run_main(self, extra, initial_model=MODEL):
        folder = self.root / ('case_' + str(len(list(self.root.iterdir()))))
        task_dir = folder / 'task_001'
        (task_dir / 'workspace').mkdir(parents=True)
        (task_dir / 'task.json').write_text(json.dumps({
            'task_type': 'math', 'instruction': 'Compute 17 plus 25.',
            'reward': 10, 'check': 'exact_int'}), encoding='utf-8')
        trace = folder / 'trace.jsonl'
        # Isolate the frozen estimator policy from adaptive controllers.
        argv = ['run_agent.py', '--tasks-dir', str(folder), '--trace-file', str(trace),
                '--confidence-policy', 'off', '--memory-policy', 'off', '--sla-policy', 'off']
        if initial_model is not None:
            argv += ['--initial-model', initial_model]
        argv += extra
        with patch('sys.argv', argv), patch.object(runtime, 'JiuwenLLM', OfflineLLM), \
                patch.dict(os.environ, {'LLM_BASE_URL': 'https://offline.invalid', 'LLM_API_KEY': 'offline-test'}), \
                redirect_stdout(io.StringIO()):
            asyncio.run(runtime.main())
        self.assertEqual((task_dir / 'answer.txt').read_text(encoding='utf-8'), '42\n')
        self.assertEqual((task_dir / 'workspace' / 'answer.txt').read_text(encoding='utf-8'), '42\n')
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        event = next(row for row in rows if row.get('outcome') == 'answered_unverified')
        self.assertEqual(event['policy_id'], POLICY_ID)
        self.assertEqual(len(event['calls']), 1)
        return event

    def test_cli_default_keeps_guarded_trace_selector(self):
        event = self.run_main([])
        self.assertEqual(event['prediction_details']['selector'], 'guarded-v1')

    def test_cli_without_model_uses_validated_reasoning_default(self):
        event = self.run_main([], initial_model=None)
        self.assertEqual(event['model'], MODELS['large_reason'])
        self.assertEqual(event['calls'][0]['model'], MODELS['large_reason'])
        self.assertEqual(event['prediction_details']['selector'], 'guarded-v1')

    def test_cli_residual_selects_new_artifact_and_traces_frozen_point(self):
        event = self.run_main(['--cost-estimator', 'residual'])
        details = event['prediction_details']
        self.assertEqual((details['selector'], details['forecast_source']), ('residual-v1', 'residual'))
        self.assertEqual(details['frozen_total'], details['raw_learned']['total_policy'])
        self.assertIsNone(details['fallback_reason'])

    def test_cli_missing_residual_artifact_still_executes_task(self):
        event = self.run_main(['--cost-estimator', 'residual', '--residual-cost-artifact', str(self.root / 'absent.json')])
        details = event['prediction_details']
        self.assertEqual(details['selector'], 'residual-v1')
        self.assertEqual((details['forecast_source'], details['fallback_reason']), ('prior', 'residual_artifact_missing'))
        self.assertIsNone(details['raw_learned'])


if __name__ == '__main__':
    unittest.main()
