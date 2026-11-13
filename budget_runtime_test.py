"""Local accounting tests only; no SDK imports and no network calls."""

import concurrent.futures
import unittest
from types import SimpleNamespace

from budget_runtime import (BudgetExhausted, BudgetLedger, DeadlineExceeded,
                            classify_api_error, normalize_usage)


class UsageTests(unittest.TestCase):
    def test_inclusive_and_separate_reasoning_same_charge(self):
        inclusive = normalize_usage(dict(prompt_tokens=10, completion_tokens=12,
            completion_tokens_details={"reasoning_tokens": 10}, total_tokens=22))
        separate = normalize_usage(SimpleNamespace(input_tokens=10, output_tokens=2,
            reasoning_tokens=10, total_tokens=22))
        self.assertEqual(inclusive.units, 130)
        self.assertEqual(separate.units, 130)
        self.assertEqual(inclusive.mode, "reasoning_included")
        self.assertEqual(separate.mode, "reasoning_separate")

    def test_plain_dummy_usage_and_cache(self):
        usage = dict(input_tokens=10, output_tokens=2, total_tokens=12,
                     input_tokens_details={"cached_tokens": 10})
        self.assertEqual(normalize_usage(usage).weighted_tokens, 3)

    def test_missing_ambiguous_and_malformed_usage_are_unknown(self):
        samples = [None, {}, dict(input_tokens=0, output_tokens=0, total_tokens=0),
            dict(input_tokens=10, output_tokens=2),
            dict(input_tokens=10, output_tokens=2, total_tokens=23, reasoning_tokens=10),
            dict(input_tokens=10, output_tokens=2, total_tokens=12, reasoning_tokens=10),
            dict(input_tokens=-1, output_tokens=2, total_tokens=1),
            dict(input_tokens=True, output_tokens=2, total_tokens=3),
            dict(input_tokens="10", output_tokens=2, total_tokens=12),
            dict(input_tokens=10, prompt_tokens=11, output_tokens=2, total_tokens=12),
            dict(input_tokens=10, output_tokens=12, total_tokens=22, reasoning_tokens=10,
                 completion_tokens_details={"reasoning_tokens": 11})]
        for usage in samples:
            with self.subTest(usage=usage):
                self.assertIsNone(normalize_usage(usage))


class LedgerTests(unittest.TestCase):
    def test_tiny_budget_blocks_without_charge(self):
        ledger = BudgetLedger(30)
        with self.assertRaises(BudgetExhausted):
            ledger.prepare("", "", max_tokens=400)
        self.assertEqual(ledger.spent, 0)
        self.assertEqual(ledger.call_records, [])

    def test_shrink_then_reconcile_exact_dummy_charge(self):
        ledger = BudgetLedger(260)
        call = ledger.prepare("system", "user", max_tokens=400)
        self.assertGreaterEqual(call.max_tokens, 32)
        self.assertLess(call.max_tokens, 400)
        self.assertEqual(call.reserved_units, 1280 + 10 + 20 * call.max_tokens)
        self.assertLessEqual(ledger.spent, 260)
        record = ledger.reconcile(call, dict(input_tokens=10, output_tokens=2, total_tokens=12))
        self.assertEqual(ledger.spent, 3)
        self.assertEqual(ledger.remaining, 257)
        self.assertEqual(record["status"], "provider_reconciled")
        self.assertEqual(record["raw_usage"]["input_tokens"], 10)

    def test_unknown_and_error_keep_full_reserve(self):
        for mode in ("missing", "error"):
            ledger = BudgetLedger(300)
            call = ledger.prepare("", "", max_tokens=50)
            original = ledger.spent
            if mode == "missing":
                ledger.reconcile(call, None)
            else:
                ledger.hold(call, "timeout")
            self.assertEqual(ledger.spent, original)
            with self.assertRaises(ValueError):
                ledger.reconcile(call, dict(input_tokens=10, output_tokens=2, total_tokens=12))
            with self.assertRaises(BudgetExhausted):
                ledger.prepare("", "", max_tokens=50)

    def test_utf8_extra_input_and_no_fractional_drift(self):
        ledger = BudgetLedger(1000)
        call = ledger.prepare("я", "🙂", max_tokens=32, extra_input_bytes=7)
        self.assertEqual(call.reserved_units, 2 + 4 + 7 + 1280 + 640)
        ledger.reconcile(call, dict(input_tokens=1, output_tokens=1, total_tokens=2))
        for _ in range(9):
            call = ledger.prepare("", "", max_tokens=32)
            ledger.reconcile(call, dict(input_tokens=1, output_tokens=1, total_tokens=2))
        self.assertEqual(ledger.spent, 11)

    def test_deadline_blocks_or_limits_before_reservation(self):
        ledger = BudgetLedger(1000, clock=lambda: 100.0)
        with self.assertRaises(DeadlineExceeded):
            ledger.prepare("", "", deadline=100.0)
        self.assertEqual(ledger.spent, 0)
        call = ledger.prepare("", "", deadline=102.5)
        self.assertEqual(call.timeout, 2.5)

    def test_provider_overrun_is_not_clipped_and_stops_new_calls(self):
        ledger = BudgetLedger(1000)
        call = ledger.prepare("", "", max_tokens=32)
        record = ledger.reconcile(call, dict(input_tokens=100, output_tokens=1000, total_tokens=1100))
        self.assertEqual(ledger.spent, 1010)
        self.assertEqual(ledger.remaining, 0)
        self.assertTrue(ledger.blocked)
        self.assertEqual(record["status"], "provider_exceeded_reservation")
        with self.assertRaises(BudgetExhausted):
            ledger.prepare("", "")

    def test_partial_usage_cannot_hide_an_obvious_reservation_breach(self):
        ledger = BudgetLedger(5000)
        call = ledger.prepare("", "", max_tokens=32)
        record = ledger.reconcile(call, dict(input_tokens=10, output_tokens=1000))
        self.assertEqual(ledger.spent, 1001)
        self.assertEqual(record["status"], "untrusted_usage_exceeded_reservation")
        self.assertTrue(ledger.blocked)
        with self.assertRaises(BudgetExhausted):
            ledger.prepare("", "")

    def test_concurrent_reservations_cannot_spend_same_balance(self):
        ledger = BudgetLedger(400)
        def prepare(_):
            try:
                return ledger.prepare("", "", max_tokens=32)
            except BudgetExhausted:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            calls = [call for call in pool.map(prepare, range(40)) if call is not None]
        self.assertEqual(len(calls), 2)
        self.assertEqual(ledger.spent, 384)

    def test_foreign_reservations_and_record_mutations_are_rejected(self):
        first, second = BudgetLedger(1000), BudgetLedger(1000)
        call = first.prepare("", "", max_tokens=32)
        second.prepare("", "", max_tokens=32)
        with self.assertRaises(ValueError):
            second.hold(call)
        first.reconcile(call, dict(input_tokens=10, output_tokens=2, total_tokens=12))
        records = first.call_records
        records[0]["raw_usage"]["input_tokens"] = 999
        self.assertEqual(first.call_records[0]["raw_usage"]["input_tokens"], 10)


class ErrorTests(unittest.TestCase):
    def test_retryable_429_is_distinct_from_quota(self):
        rate = RuntimeError("Error code: 429 - rate limit requests per minute")
        quota = RuntimeError("Error code: 429 - insufficient_quota")
        self.assertTrue(classify_api_error(rate).retryable)
        self.assertFalse(classify_api_error(rate).abort_run)
        self.assertTrue(classify_api_error(quota).abort_run)
        self.assertFalse(classify_api_error(quota).retryable)

    def test_wrapped_structured_status_timeout_and_auth(self):
        cause = RuntimeError("temporarily overloaded")
        cause.status_code = 503
        outer = RuntimeError("SDK invocation failed")
        outer.__cause__ = cause
        self.assertEqual(classify_api_error(outer).kind, "transient")
        self.assertTrue(classify_api_error(TimeoutError()).retryable)
        error = RuntimeError("unauthorized")
        error.status_code = 401
        self.assertTrue(classify_api_error(error).abort_run)
        self.assertFalse(classify_api_error(RuntimeError("answer mentions 429 items")).retryable)

    def test_structured_quota_code_overrides_generic_429_text(self):
        error = RuntimeError("requests exceeded")
        error.status_code = 429
        error.body = {"error": {"code": "insufficient_quota"}}
        self.assertEqual(classify_api_error(error).kind, "quota")
        self.assertTrue(classify_api_error(error).abort_run)


if __name__ == "__main__":
    unittest.main()
