"""Latency/queue counterexamples without clocks, models or task execution."""
import dataclasses
import math
import unittest

from latency_planner import (
    DEFAULT_PRIORS, MAX_HISTORY, MAX_OBSERVATION_SECONDS, LatencyEstimator,
    LatencyForecast, LatencyPrior, ResourceCandidate, continuation_value,
    rank_candidates,
)


MODEL = 'qwen3.8-27b'
OTHER = 'qwen3.8-27b-noreason'


def candidate(key, reward, tokens, seconds, *, priority=None, upper=None):
    return ResourceCandidate(key, reward, tokens, seconds,
                             seconds if upper is None else upper,
                             reward / max(1, tokens) if priority is None else priority)


class LatencyEstimatorTests(unittest.TestCase):
    def forecast(self, estimator, *, model=MODEL, domain='coding', tokens=100,
                 first=None, calls=None, tools=True):
        return estimator.forecast(model=model, domain=domain, predicted_tokens=tokens,
                                  first_call_tokens=first, estimated_calls=calls,
                                  include_tools=tools)

    def observe(self, estimator, elapsed, *, model=MODEL, domain='coding', tokens=100,
                known=True, completed=True, censored=False):
        return estimator.observe_call(model=model, domain=domain,
            elapsed_seconds=elapsed, weighted_tokens=tokens, usage_known=known,
            completed=completed, censored=censored)

    def test_prior_includes_request_count_and_active_tool_overhead(self):
        estimator = LatencyEstimator()
        single = self.forecast(estimator, calls=1, tools=False)
        multi = self.forecast(estimator, tokens=300, first=100, tools=False)
        self.assertAlmostEqual(single.seconds, 1 + 100 * .0104)
        self.assertAlmostEqual(multi.seconds, 3 * single.seconds)
        self.assertEqual(multi.estimated_calls, 3)
        self.assertAlmostEqual(self.forecast(estimator, tokens=300, first=100).seconds,
                               multi.seconds + 3)
        self.assertGreaterEqual(multi.conservative_seconds, multi.seconds)

    def test_explicit_single_call_overrides_total_first_ratio(self):
        estimator = LatencyEstimator()
        forecast = self.forecast(estimator, tokens=500, first=1, calls=1)
        self.assertEqual(forecast.estimated_calls, 1)

    def test_completed_known_usage_calibrates_only_matching_scope(self):
        estimator = LatencyEstimator()
        before = self.forecast(estimator)
        other_model = self.forecast(estimator, model=OTHER)
        other_domain = self.forecast(estimator, domain='math')
        for _ in range(8):
            self.assertTrue(self.observe(estimator, 20))
        after = self.forecast(estimator)
        self.assertGreater(after.seconds, before.seconds)
        self.assertEqual(after.observations, 8)
        self.assertEqual(self.forecast(estimator, model=OTHER), other_model)
        self.assertEqual(self.forecast(estimator, domain='math'), other_domain)

    def test_unknown_usage_informs_upper_duration_without_learning_rate(self):
        estimator = LatencyEstimator()
        before = self.forecast(estimator, tools=False)
        self.observe(estimator, 40, tokens=0.1, known=False)
        after = self.forecast(estimator, tools=False)
        self.assertEqual(after.seconds, before.seconds)
        self.assertGreaterEqual(after.conservative_seconds, 40)
        self.assertEqual(after.observations, 0)
        self.assertEqual(after.unknown_usage_observations, 1)

    def test_censored_and_failed_calls_never_become_fast_successes(self):
        for completed, censored in ((False, True), (False, False), (True, True)):
            with self.subTest(completed=completed, censored=censored):
                estimator = LatencyEstimator()
                before = self.forecast(estimator)
                self.observe(estimator, .001, tokens=100000,
                             completed=completed, censored=censored)
                after = self.forecast(estimator)
                self.assertEqual(after.seconds, before.seconds)
                self.assertEqual(after.observations, 0)
                self.assertEqual(after.censored_observations, 1)
                self.assertGreaterEqual(after.conservative_seconds, before.conservative_seconds)

    def test_long_timeout_only_raises_conservative_estimate(self):
        estimator = LatencyEstimator()
        before = self.forecast(estimator, tools=False)
        self.observe(estimator, 75, completed=False, censored=True)
        after = self.forecast(estimator, tools=False)
        self.assertEqual(before.seconds, after.seconds)
        self.assertGreaterEqual(after.conservative_seconds, 75)

    def test_completed_zero_or_invalid_usage_is_duration_only(self):
        for cost in (None, 0, float('nan'), -1, True):
            with self.subTest(cost=cost):
                estimator = LatencyEstimator()
                self.observe(estimator, 3, tokens=cost)
                forecast = self.forecast(estimator)
                self.assertEqual(forecast.observations, 0)
                self.assertEqual(forecast.unknown_usage_observations, 1)

    def test_tool_observations_are_separate_scoped_and_not_token_rates(self):
        estimator = LatencyEstimator()
        before = self.forecast(estimator)
        without_tools = self.forecast(estimator, tools=False)
        for _ in range(8):
            self.assertTrue(estimator.observe_tool(model=MODEL, domain='coding', elapsed_seconds=6))
        after = self.forecast(estimator)
        self.assertGreater(after.seconds, before.seconds)
        self.assertEqual(after.observations, 0)
        self.assertEqual(after.tool_observations, 8)
        self.assertEqual(self.forecast(estimator, tools=False).seconds, without_tools.seconds)
        self.assertEqual(self.forecast(estimator, domain='search').tool_observations, 0)

    def test_outlier_influence_and_observation_buffers_are_bounded(self):
        estimator = LatencyEstimator()
        original = self.forecast(estimator, tools=False)
        for _ in range(100):
            self.observe(estimator, 3000, tokens=.001)
            estimator.observe_tool(model=MODEL, domain='coding', elapsed_seconds=1)
        forecast = self.forecast(estimator, tools=False)
        self.assertLessEqual(forecast.seconds, original.seconds * 4)
        self.assertEqual(forecast.observations, MAX_HISTORY)
        self.assertEqual(forecast.tool_observations, MAX_HISTORY)
        self.assertTrue(math.isfinite(forecast.conservative_seconds))

    def test_invalid_samples_do_not_allocate_state(self):
        estimator = LatencyEstimator()
        for elapsed in (-1, 0, float('nan'), float('inf'), True, MAX_OBSERVATION_SECONDS + 1):
            self.assertFalse(self.observe(estimator, elapsed))
        self.assertFalse(self.observe(estimator, 1, model='foreign-model'))
        self.assertFalse(self.observe(estimator, 1, domain=[]))
        self.assertEqual(estimator._observations, {})

    def test_single_call_forecast_is_monotone_for_deadline_cap_search(self):
        estimator = LatencyEstimator()
        self.observe(estimator, 50, known=False)
        self.observe(estimator, 4)
        estimator.observe_tool(model=MODEL, domain='coding', elapsed_seconds=2)
        forecasts = [self.forecast(estimator, tokens=tokens, first=tokens, calls=1)
                     for tokens in (0, 1, 32, 64, 128, 500, 3072, 6144, 32768, 1e308)]
        self.assertEqual([f.seconds for f in forecasts], sorted(f.seconds for f in forecasts))
        self.assertEqual([f.conservative_seconds for f in forecasts],
                         sorted(f.conservative_seconds for f in forecasts))
        self.assertTrue(all(math.isfinite(f.conservative_seconds) for f in forecasts))

    def test_domain_override_and_unknown_scope_use_explicit_priors(self):
        estimator = LatencyEstimator({(MODEL, 'math'): LatencyPrior(1, .1, 2, 'test')})
        self.assertAlmostEqual(self.forecast(estimator, domain='math', tools=False).seconds, 11)
        self.assertNotEqual(self.forecast(estimator, tools=False).source, 'test')
        unknown = self.forecast(estimator, model='other', tools=False)
        self.assertEqual(unknown.source, 'unsupported_scope_conservative_prior')

    def test_invalid_priors_and_forecast_arguments_fail_explicitly(self):
        for prior in (LatencyPrior(1, 0, 2), LatencyPrior(1, .1, .5), LatencyPrior(float('nan'), .1, 2)):
            with self.assertRaises(ValueError):
                LatencyEstimator({MODEL: prior})
        estimator = LatencyEstimator()
        for tokens in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                self.forecast(estimator, tokens=tokens)
        with self.assertRaises(ValueError):
            self.forecast(estimator, calls=0)


class JointResourceTests(unittest.TestCase):
    def test_permitted_retry_contributes_to_pressure_and_joint_opportunity(self):
        retry = dataclasses.replace(candidate('retry', 1.2, 250, 59), continuation=True)
        alternative = candidate('alternative', .228, 50, 10)
        choices = rank_candidates([retry, alternative], remaining_tokens=300,
                                  remaining_seconds=60)
        self.assertTrue(all(choice.eligible and choice.time_pressure for choice in choices))
        self.assertEqual([choice.key for choice in choices], ['alternative', 'retry'])
        opportunity = choices[0].score
        decision = continuation_value(expected_reward=retry.expected_reward,
            predicted_tokens=retry.predicted_tokens,
            forecast=LatencyForecast(59, 59, 1, 'test', 0),
            remaining_tokens=300, remaining_seconds=60,
            opportunity=opportunity, time_pressure=choices[0].time_pressure)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, 'other_tasks_have_higher_joint_value')
        self.assertAlmostEqual(decision.score, choices[1].score)

    def test_retry_inclusive_boundary_does_not_relax_new_task_admission(self):
        retry = dataclasses.replace(candidate('retry', 1, 20, 1), continuation=True)
        fresh = candidate('fresh', 1, 20, 1)
        for mode in ('joint', 'token'):
            with self.subTest(mode=mode):
                choices = rank_candidates([retry, fresh], remaining_tokens=22,
                                          remaining_seconds=60, mode=mode)
                self.assertTrue(choices[0].eligible)
                self.assertEqual(choices[0].key, 'retry')
                self.assertFalse(choices[1].eligible)
                below = rank_candidates([retry], remaining_tokens=21.99,
                                        remaining_seconds=60, mode=mode)
                self.assertFalse(below[0].eligible)
        fresh_boundary = rank_candidates([fresh], remaining_tokens=25,
                                         remaining_seconds=60)
        self.assertFalse(fresh_boundary[0].eligible)

    def test_non_boolean_continuation_marker_is_invalid(self):
        forged = dataclasses.replace(candidate('retry', 1, 20, 1), continuation='yes')
        choices = rank_candidates([forged], remaining_tokens=22, remaining_seconds=60)
        self.assertEqual(choices[0].reason, 'invalid_forecast')

    def test_spare_time_exactly_preserves_original_token_priorities(self):
        items = [candidate('slow_valuable', 2, 10, 40, priority=9.123),
                 candidate('fast', 1, 20, 1, priority=8.456)]
        choices = rank_candidates(items, remaining_tokens=100, remaining_seconds=100)
        self.assertEqual([item.key for item in choices], ['slow_valuable', 'fast'])
        self.assertEqual([item.score for item in choices], [9.123, 8.456])
        self.assertTrue(all(not item.time_pressure for item in choices))

    def test_tight_time_can_prefer_fast_task_over_token_efficient_slow_task(self):
        items = [candidate('slow', 1, 10, 55), candidate('fast', 1, 20, 6)]
        choices = rank_candidates(items, remaining_tokens=1000, remaining_seconds=60)
        self.assertEqual([item.key for item in choices], ['fast', 'slow'])
        self.assertTrue(all(item.time_pressure for item in choices))
        self.assertAlmostEqual(choices[0].score, 1 / max(20 / 1000, 6 / 60))

    def test_joint_score_prices_both_scarce_resources(self):
        items = [candidate('token_heavy', 1, 70, 10), candidate('balanced', 1, 20, 20),
                 candidate('slow', 1, 10, 40)]
        choices = rank_candidates(items, remaining_tokens=100, remaining_seconds=60)
        self.assertEqual(choices[0].key, 'balanced')
        self.assertTrue(all(item.time_pressure for item in choices))

    def test_token_only_ignores_time_and_preserves_legacy_strict_affordability(self):
        items = [candidate('long', 1, 79, 10000), candidate('at_boundary', 100, 80, 0),
                 candidate('invalid_time', 1, 10, float('nan'), priority=.001)]
        choices = rank_candidates(items, remaining_tokens=100, remaining_seconds=-1, mode='token')
        self.assertEqual([item.key for item in choices], ['long', 'invalid_time', 'at_boundary'])
        self.assertTrue(choices[0].eligible)
        self.assertTrue(choices[1].eligible)
        self.assertFalse(choices[2].eligible)
        self.assertTrue(all(not item.time_pressure for item in choices))

    def test_deadline_admission_uses_point_and_does_not_claim_impossibility(self):
        choices = rank_candidates([candidate('too_slow', 1, 10, 61, upper=70),
                                   candidate('fits', 1, 10, 60, upper=90)],
                                  remaining_tokens=100, remaining_seconds=60)
        self.assertEqual(choices[0].key, 'fits')
        self.assertTrue(choices[0].eligible)
        self.assertEqual(choices[1].reason, 'global_deadline_insufficient')

    def test_expected_duration_fits_despite_upper_tail_exceeding_deadline(self):
        item = candidate('uncertain', 1, 10, 20, upper=90)
        choice = rank_candidates([item], remaining_tokens=100,
                                 remaining_seconds=30)[0]
        self.assertTrue(choice.eligible)
        self.assertFalse(choice.time_pressure)
        retry = continuation_value(expected_reward=1, predicted_tokens=10,
            forecast=LatencyForecast(20, 90, 1, 'test', 0),
            remaining_tokens=100, remaining_seconds=30)
        self.assertTrue(retry.eligible)

    def test_scarce_expected_time_drives_joint_score_not_tail_size(self):
        items = [candidate('slow', 1, 10, 50, upper=55),
                 candidate('fast_uncertain', 1, 10, 20, upper=200)]
        choices = rank_candidates(items, remaining_tokens=1000, remaining_seconds=60)
        self.assertTrue(all(choice.eligible and choice.time_pressure for choice in choices))
        self.assertEqual([choice.key for choice in choices], ['fast_uncertain', 'slow'])
        self.assertAlmostEqual(choices[0].score, 3)
        self.assertAlmostEqual(choices[1].score, 1.2)
        ample = rank_candidates(items, remaining_tokens=1000, remaining_seconds=80)
        self.assertTrue(all(not choice.time_pressure for choice in ample))
        self.assertEqual([choice.key for choice in ample], ['slow', 'fast_uncertain'])

    def test_point_beyond_global_time_still_rejects_task_and_retry(self):
        choice = rank_candidates([candidate('slow', 1, 10, 61, upper=80)],
            remaining_tokens=1000, remaining_seconds=60)[0]
        self.assertFalse(choice.eligible)
        self.assertEqual(choice.reason, 'global_deadline_insufficient')
        retry = continuation_value(expected_reward=1, predicted_tokens=10,
            forecast=LatencyForecast(61, 80, 1, 'test', 0),
            remaining_tokens=1000, remaining_seconds=60)
        self.assertFalse(retry.eligible)
        self.assertEqual(retry.reason, 'global_deadline_insufficient')

    def test_ineligible_and_zero_reward_do_not_manufacture_time_pressure(self):
        items = [candidate('free_value', 1, 10, 1), candidate('zero', 0, 1, 60, priority=99),
                 candidate('over_tokens', 1, 1000, 60), candidate('over_time', 1, 10, 1000)]
        choices = rank_candidates(items, remaining_tokens=100, remaining_seconds=60)
        self.assertEqual(choices[0].key, 'free_value')
        self.assertTrue(all(not choice.time_pressure for choice in choices))
        self.assertEqual(sum(choice.eligible for choice in choices), 1)

    def test_token_pressure_alone_does_not_change_old_order(self):
        choices = rank_candidates([candidate('a', 1, 60, 1, priority=3),
                                   candidate('b', 2, 60, 1, priority=1)],
                                  remaining_tokens=100, remaining_seconds=60)
        self.assertEqual([choice.key for choice in choices], ['a', 'b'])
        self.assertTrue(all(choice.token_pressure and not choice.time_pressure for choice in choices))

    def test_rank_is_rescored_when_remaining_time_changes(self):
        items = [candidate('slow', 1, 10, 55), candidate('fast', 1, 20, 6)]
        ample = rank_candidates(items, remaining_tokens=1000, remaining_seconds=600)
        tight = rank_candidates(items, remaining_tokens=1000, remaining_seconds=60)
        self.assertEqual(ample[0].key, 'slow')
        self.assertEqual(tight[0].key, 'fast')

    def test_online_calibration_can_change_next_queue_admission(self):
        estimator = LatencyEstimator()
        def choices():
            forecast = estimator.forecast(model=MODEL, domain='math', predicted_tokens=100,
                                          estimated_calls=1, include_tools=False)
            return rank_candidates([candidate('work', 1, 100, forecast.seconds,
                                               upper=forecast.conservative_seconds)],
                                   remaining_tokens=1000, remaining_seconds=3)
        self.assertTrue(choices()[0].eligible)
        estimator.observe_call(model=MODEL, domain='math', elapsed_seconds=20,
                               weighted_tokens=100, usage_known=True)
        self.assertFalse(choices()[0].eligible)

    def test_empty_and_invalid_inputs_fail_closed_without_nan_scores(self):
        self.assertEqual(rank_candidates([], remaining_tokens=0, remaining_seconds=0), ())
        item = candidate('a', 1, 10, 1)
        for tokens, seconds in ((0, 100), (100, 0), (float('nan'), 10), (10, float('inf'))):
            choices = rank_candidates([item], remaining_tokens=tokens, remaining_seconds=seconds)
            self.assertFalse(choices[0].eligible)
            self.assertEqual(choices[0].score, 0)
        forged = dataclasses.replace(item, expected_reward=float('nan'))
        self.assertEqual(rank_candidates([forged], remaining_tokens=100,
                                         remaining_seconds=100)[0].reason, 'invalid_forecast')

    def test_equal_scores_have_stable_tie_order(self):
        items = [candidate('z', 1, 10, 1), candidate('a', 1, 10, 1)]
        self.assertEqual([choice.key for choice in rank_candidates(items,
            remaining_tokens=100, remaining_seconds=100)], ['z', 'a'])

    def test_more_global_time_can_admit_a_task_without_individual_time_quota(self):
        item = candidate('long_task', 100, 100, 900, upper=1000)
        self.assertFalse(rank_candidates([item], remaining_tokens=10000,
                                          remaining_seconds=600)[0].eligible)
        self.assertTrue(rank_candidates([item], remaining_tokens=10000,
                                         remaining_seconds=1200)[0].eligible)


class ContinuationTests(unittest.TestCase):
    def check(self, **kwargs):
        defaults = dict(expected_reward=1.0, predicted_tokens=20,
                        forecast=LatencyForecast(5, 6, 1, 'test', 0),
                        remaining_tokens=1000, remaining_seconds=60)
        defaults.update(kwargs)
        return continuation_value(**defaults)

    def test_retry_uses_same_joint_value_units_as_fresh_queue(self):
        retry = self.check(time_pressure=True, opportunity=11)
        self.assertTrue(retry.eligible)
        self.assertAlmostEqual(retry.score, 12)
        stopped = self.check(time_pressure=True, opportunity=13)
        self.assertFalse(stopped.eligible)
        self.assertEqual(stopped.reason, 'other_tasks_have_higher_joint_value')

    def test_surplus_does_not_add_new_token_only_opportunity_guard(self):
        self.assertTrue(self.check(opportunity=1e6).eligible)

    def test_retry_cannot_fit_after_time_elapsed_even_with_surplus_tokens(self):
        decision = self.check(remaining_seconds=4.99)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, 'global_deadline_insufficient')

    def test_retry_preserves_existing_token_headroom_boundary(self):
        self.assertTrue(self.check(remaining_tokens=22.0).eligible)
        self.assertFalse(self.check(remaining_tokens=21.99).eligible)

    def test_token_mode_leaves_existing_retry_controller_unchanged(self):
        decision = self.check(mode='token', remaining_seconds=0, opportunity=1e6)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, 'token_policy_unchanged')

    def test_invalid_forecast_or_zero_value_cannot_authorize_joint_retry(self):
        self.assertFalse(self.check(forecast=LatencyForecast(10, 1, 1, 'bad', 0)).eligible)
        self.assertFalse(self.check(expected_reward=0).eligible)
        self.assertFalse(self.check(opportunity=float('nan')).eligible)


if __name__ == '__main__':
    unittest.main()
