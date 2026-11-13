"""Confidence contract tests: no model calls, filesystem parsing, or labels."""
import copy
import math
import unittest
from types import SimpleNamespace

from task_confidence import ConfidenceEstimate, estimate_confidence


MODEL = 'qwen3.8-27b-noreason'


def score(features=None, prior=0.8, *, risk=None, model=MODEL, task=None):
    return estimate_confidence({'task_type': 'coding'} if task is None else task,
                               {} if features is None else features,
                               prior, model, risk=risk)


class AllowlistedMapping(dict):
    def __init__(self, allowed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = allowed

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError(f'Forbidden feature access: {key}')
        return super().get(key, default)

    def __iter__(self):
        raise AssertionError('Must not enumerate task or features')


class TaskConfidenceTests(unittest.TestCase):
    def test_no_risk_preserves_caller_model_prior(self):
        for model in ('qwen3.6-35b-a3b', 'qwen3.6-35b-a3b-noreason',
                      'qwen3.8-27b', MODEL):
            result = score(prior=.73, model=model)
            self.assertEqual(result.probability, .73)
            self.assertEqual(result.source, 'heuristic_visible_v1')

    def test_partial_missing_and_conflict_are_soft_ordered_signals(self):
        partial = score({'context_partial': 1}).probability
        missing = score({'missing_evidence_flag': 1}).probability
        conflict = score({'contradiction_flag': 1}).probability
        self.assertGreater(.8, partial)
        self.assertGreater(partial, missing)
        self.assertGreater(missing, conflict)
        self.assertGreater(conflict, 0)
        self.assertNotIn('hard_impossible', ConfidenceEstimate.__dataclass_fields__)

    def test_correlated_evidence_does_not_compound(self):
        together = score(dict(context_partial=1, missing_evidence_flag=1,
                              contradiction_flag=1))
        self.assertEqual(together.probability,
                         score({'contradiction_flag': 1}).probability)
        self.assertIn('context_partial', together.reasons)
        self.assertIn('missing_evidence', together.reasons)

    def test_state_or_mutability_weakens_unproved_conflict(self):
        baseline = score({'contradiction_flag': 1})
        for flag in ('explicit_stateful', 'mutable_arguments', 'incomplete_analysis'):
            result = score({'contradiction_flag': 1},
                           risk=SimpleNamespace(flags=(flag,)))
            self.assertGreater(result.probability, baseline.probability)
            self.assertLess(result.probability, .8)
            self.assertIn('conflicting_outputs_ambiguous', result.reasons)

    def test_unsupported_syntax_alone_does_not_mean_difficult(self):
        self.assertEqual(score(risk={'flags': ['incomplete_analysis']}).probability, .8)

    def test_risk_report_mapping_and_object_agree(self):
        flags = ('conflicting_outputs', 'mutable_arguments')
        self.assertEqual(score(risk={'flags': flags}),
                         score(risk=SimpleNamespace(flags=flags)))
        self.assertEqual(score(risk={'flags': ('missing_evidence',)}).probability,
                         score({'missing_evidence_flag': 1}).probability)

    def test_identifiers_rewards_labels_lengths_and_hard_status_are_not_read(self):
        task = AllowlistedMapping({'task_type'}, task_type='coding',
                                  task_id='negative_123', reward=100,
                                  correct=False, answerable=False)
        features = AllowlistedMapping(
            {'context_partial', 'missing_evidence_flag', 'contradiction_flag'},
            instruction_chars=10**20, context_chars=10**20, source_bytes=10**20,
            legacy_raw_cost=10**20, oracle_answer='secret', success=0)
        risk = AllowlistedMapping({'flags'}, flags=(), hard_impossible=True,
                                  reason='An oracle says impossible')
        self.assertEqual(score(features, task=task, risk=risk).probability, .8)

    def test_inputs_are_not_mutated(self):
        task = {'task_type': 'coding', 'instruction': 'text', 'nested': [1]}
        features = {'context_partial': 1.0, 'contradiction_flag': 0.0}
        risk = {'flags': ['missing_evidence'], 'evidence': [{'x': 1}]}
        before = copy.deepcopy((task, features, risk))
        score(features, task=task, risk=risk)
        self.assertEqual((task, features, risk), before)

    def test_invalid_priors_and_boundaries_always_finite_strictly_between_zero_and_one(self):
        for prior in (float('nan'), float('inf'), -float('inf'), None, '0.9',
                      True, 10**1000, -100, 0, 1, 100, .001):
            for features in ({}, {'contradiction_flag': 1}):
                value = score(features, prior).probability
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, .01)
                self.assertLessEqual(value, .99)
        self.assertEqual(score(prior=float('nan')).probability, .4)

    def test_unknown_model_or_domain_does_not_transfer_adjustments(self):
        features = {'missing_evidence_flag': 1}
        for model in ('unknown', None, []):
            result = score(features, model=model)
            self.assertEqual(result.probability, .8)
            self.assertEqual(result.source, 'prior_only_v1')
            self.assertIn('prior_only_unknown_model', result.reasons)
        for task in ({'task_type': 'unknown'}, {}, {'task_type': []}):
            result = score(features, task=task)
            self.assertEqual(result.probability, .8)
            self.assertIn('prior_only_unknown_domain', result.reasons)

    def test_malformed_flags_are_unknown_not_truthy_risk(self):
        for value in (float('nan'), float('inf'), -1, 2, 'false', '1', None, []):
            result = score({'contradiction_flag': value})
            self.assertEqual(result.probability, .8)
            self.assertIn('invalid_feature_ignored:contradiction_flag', result.reasons)
        self.assertEqual(estimate_confidence({'task_type': 'math'}, None, .8,
                                            MODEL).probability, .8)
        for raw in ('conflicting_outputs', ['conflicting_outputs'] * 33):
            self.assertEqual(score(risk={'flags': raw}).probability, .8)


if __name__ == '__main__':
    unittest.main()
