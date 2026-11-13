import unittest
from retry_policy import decide_retry


class RetryPolicyTests(unittest.TestCase):
    def decide(self, **kw):
        base=dict(kind='coding', state='check_failed', feedback='1 failed', attempts=1,
                  last_model='qwen3.8-27b-noreason', reasoning_model='qwen3.8-27b',
                  last_cap=512, finish_reason='stop', remaining=80000, reward=1)
        base.update(kw)
        return decide_retry(**base)

    def test_low_reward_can_retry_with_surplus_budget(self):
        self.assertTrue(self.decide().retry)

    def test_contradiction_does_not_trigger_expensive_repeat(self):
        self.assertFalse(self.decide(flags=['conflicting_outputs']).retry)

    def test_stateful_or_incomplete_analysis_does_not_prove_retry_hopeless(self):
        for flag in ('explicit_stateful','mutable_arguments','incomplete_analysis'):
            self.assertTrue(self.decide(flags=['conflicting_outputs',flag]).retry)

    def test_truncation_extends_cap_even_after_multiple_calls(self):
        d=self.decide(attempts=4,last_cap=4096,finish_reason='length')
        self.assertTrue(d.retry)
        self.assertEqual(d.max_tokens,8192)

    def test_empty_reasoning_content_can_be_truncated_more_than_once(self):
        self.assertTrue(self.decide(repeated=True,finish_reason='length').retry)

    def test_same_failure_stops_but_improvement_can_continue(self):
        self.assertFalse(self.decide(attempts=2).retry)
        self.assertTrue(self.decide(attempts=4,improved=True).retry)

    def test_shared_budget_pressure_compares_opportunity(self):
        self.assertFalse(self.decide(remaining=2000,competing_cost=2000,opportunity=0.1).retry)
        self.assertTrue(self.decide(remaining=80000,competing_cost=2000,opportunity=0.1).retry)

    def test_explicit_abstention_never_repeats(self):
        self.assertFalse(self.decide(state='abstained',finish_reason='length').retry)

    def test_sql_error_uses_cheaper_repair(self):
        d=self.decide(kind='data_analysis',feedback='no such column: x')
        self.assertTrue(d.model.endswith('-noreason'))


if __name__=='__main__': unittest.main()
