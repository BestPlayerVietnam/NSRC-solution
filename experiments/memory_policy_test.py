"""Software-behavior QA only: no task-accuracy or token-savings experiment.

Tool events and provider usage are fixtures. The real memory API and ledger are
used; the SDK transport is replaced before construction. No network calls occur.
"""
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from budget_runtime import BudgetExhausted
from budget_config_test import FakeSDKModel, fake_sdk_modules
from experience_memory import ExperienceMemory
import run_agent as runtime


MODEL = 'qwen3.8-27b-noreason'
OTHER_MODEL = 'qwen3.6-35b-a3b-noreason'
MATH_ERROR = 'Invalid arithmetic expression'
SQL_ERROR = 'SQLite: no such column: missing_field'
INSTRUCTION = 'Compute a supported arithmetic expression for the supplied numbers.'


def teach(memory, *, model=MODEL, domain='math', source='earlier-task',
          feedback=MATH_ERROR, verifier='tool_execution'):
    memory.record_failure(model=model, domain=domain, source_ref=source,
                          call_seq=1, feedback=feedback)
    return memory.finish_task(model=model, domain=domain, source_ref=source,
                              call_seq=2, verification_level=verifier, success=True)


def recall(memory, *, model=MODEL, domain='math', source='later-task', feedback=''):
    instruction = INSTRUCTION if domain == 'math' else 'Query the current SQLite table schema.'
    return memory.retrieve(model=model, domain=domain, instruction=instruction,
                           feedback=feedback, source_ref=source)


class MemoryStreamContractTests(unittest.TestCase):
    def test_verified_repair_is_scoped_and_never_retrieved_for_its_source_task(self):
        memory = ExperienceMemory('revise')
        self.assertFalse(recall(memory))
        self.assertEqual(len(teach(memory)), 1)
        self.assertEqual(len(recall(memory)), 1)
        self.assertFalse(recall(memory, source='earlier-task'))
        self.assertFalse(recall(memory, model=OTHER_MODEL))
        self.assertFalse(recall(memory, domain='data_analysis'))
        self.assertEqual(recall(memory)[0].verification_level, 'tool_execution')

    def test_unknown_success_and_transport_failure_do_not_create_lessons(self):
        for feedback, verifier in ((MATH_ERROR, 'unknown'),
                                   ('Network timeout', 'tool_execution'),
                                   ('HTTP 429 rate limit', 'tool_execution'),
                                   ('Model says the answer is correct', 'tool_execution')):
            with self.subTest(feedback=feedback, verifier=verifier):
                memory = ExperienceMemory('revise')
                self.assertFalse(teach(memory, feedback=feedback, verifier=verifier))
                self.assertFalse(recall(memory))
        memory = ExperienceMemory('revise')
        memory.record_failure(model=MODEL, domain='math', source_ref='same-call',
                              call_seq=1, feedback=MATH_ERROR)
        self.assertFalse(memory.finish_task(model=MODEL, domain='math', source_ref='same-call',
                                            call_seq=1, verification_level='tool_execution', success=True))

    def test_only_revision_mode_narrows_a_used_card_after_repeated_matching_errors(self):
        for mode in ('off', 'append', 'revise'):
            with self.subTest(mode=mode):
                memory = ExperienceMemory(mode)
                teach(memory, domain='data_analysis', feedback=SQL_ERROR)
                used = recall(memory, domain='data_analysis', source='challenging-task')
                if mode == 'off':
                    self.assertFalse(used)
                    self.assertEqual(memory.render(used), '')
                    continue
                before = used[0]
                for seq in (1, 2):
                    memory.record_failure(model=MODEL, domain='data_analysis',
                                          source_ref='challenging-task', call_seq=seq,
                                          feedback=SQL_ERROR, used_ids=(before.id,))
                current = memory.history(before.id)[-1]
                if mode == 'append':
                    self.assertEqual(current.version, before.version)
                    self.assertTrue(recall(memory, domain='data_analysis', source='third-task'))
                else:
                    self.assertGreater(current.version, before.version)
                    self.assertGreater(current.caution_count, before.caution_count)
                    self.assertFalse(recall(memory, domain='data_analysis', source='third-task'))
                    self.assertTrue(recall(memory, domain='data_analysis', source='third-task', feedback=SQL_ERROR))
                    self.assertEqual(memory.render((before,)), '')
                memory.finish_task(model=MODEL, domain='data_analysis', source_ref='challenging-task',
                                   call_seq=2, verification_level='unknown', used_ids=(before.id,), success=False)

    def test_unused_card_is_not_blame_target_for_another_task_failure(self):
        memory = ExperienceMemory('revise')
        teach(memory, domain='data_analysis', feedback=SQL_ERROR)
        card = recall(memory, domain='data_analysis')[0]
        for seq in (1, 2, 3):
            memory.record_failure(model=MODEL, domain='data_analysis', source_ref='unrelated-failure',
                                  call_seq=seq, feedback=SQL_ERROR, used_ids=())
        memory.finish_task(model=MODEL, domain='data_analysis', source_ref='unrelated-failure',
                           call_seq=3, verification_level='unknown', used_ids=(), success=False)
        self.assertEqual(memory.history(card.id)[-1], card)

    def test_raw_source_paths_and_answer_like_error_details_never_render(self):
        secret = 'PRIVATE_SOURCE_AND_ANSWER_987654321'
        memory = ExperienceMemory('revise')
        learned = teach(memory, domain='data_analysis', source=secret,
                        feedback='SQLite: no such column: '+secret)
        self.assertEqual(len(learned), 1)
        visible = recall(memory, domain='data_analysis')
        self.assertTrue(visible)
        self.assertNotIn(secret, memory.render(visible))
        self.assertNotIn(secret, json.dumps([asdict(card) for card in learned]))
        self.assertNotIn(secret, json.dumps(memory.summary()))


class MemoryPromptAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)/'current-task'

    def make_llm(self, mode, budget=100000):
        llm = runtime.JiuwenLLM(budget)
        llm.confidence_enabled = False
        llm.fixed_model = True
        llm.initial_model = MODEL
        llm.experience_memory = ExperienceMemory(mode)
        llm.memory_records = []
        teach(llm.experience_memory)
        runtime.task_progress(llm, {'task_type': 'math', 'instruction': INSTRUCTION}, self.workspace)
        return llm

    def test_actual_memory_prompt_is_reserved_on_every_call_without_extra_calls(self):
        with patch.dict(sys.modules, fake_sdk_modules()), patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://offline.invalid', 'LLM_API_KEY': 'offline-test'}):
            llms = {mode: self.make_llm(mode) for mode in ('off', 'append', 'revise')}
            for llm in llms.values():
                for _ in range(2):
                    asyncio.run(runtime.ask_task(llm, 'Numeric arithmetic only.', INSTRUCTION,
                                                 model=MODEL, max_tokens=64))
                self.assertEqual(len(llm.model.calls), 2)
                self.assertEqual(len(llm.call_records), 2)
                for (messages, kwargs), record in zip(llm.model.calls, llm.call_records):
                    actual_bytes = sum(len(message['content'].encode('utf-8')) for message in messages)
                    self.assertEqual(record['reserved_units'], 1280 + actual_bytes + 20*kwargs['max_tokens'])
                    self.assertEqual(record['status'], 'provider_reconciled')
                    # Fixture counters, not a tokenizer or efficiency estimate.
                    self.assertEqual(record['weighted_tokens'], 5)
            baseline = llms['off']
            for mode in ('append', 'revise'):
                for index, record in enumerate(llms[mode].call_records):
                    delta = record['reserved_units'] - baseline.call_records[index]['reserved_units']
                    self.assertGreater(delta, 0)
                    self.assertEqual(record['memory_injected_utf8_bytes'], delta)
                    self.assertTrue(record['memory_cards'])

    def test_memory_can_exhaust_the_shared_reservation_before_any_transport(self):
        system, user = 'Arithmetic.', INSTRUCTION
        minimum_units = len(system.encode('utf-8'))+len(user.encode('utf-8'))+1280+20*32
        budget = Decimal(minimum_units)/10
        with patch.dict(sys.modules, fake_sdk_modules()), patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://offline.invalid', 'LLM_API_KEY': 'offline-test'}):
            off, memory = self.make_llm('off', budget), self.make_llm('revise', budget)
            asyncio.run(runtime.ask_task(off, system, user, model=MODEL,
                                         max_tokens=32, min_output_tokens=32))
            with self.assertRaises(BudgetExhausted):
                asyncio.run(runtime.ask_task(memory, system, user, model=MODEL,
                                             max_tokens=32, min_output_tokens=32))
            self.assertEqual(len(off.model.calls), 1)
            self.assertEqual(memory.model.calls, [])
            self.assertEqual(memory.call_records, [])
            self.assertEqual(memory.spent, 0)

    def test_transport_retry_retains_memory_accounting_on_both_physical_records(self):
        with patch.dict(sys.modules, fake_sdk_modules()), patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://offline.invalid', 'LLM_API_KEY': 'offline-test'}):
            llm = self.make_llm('revise')
            invoke = llm.model.invoke
            attempts = []

            async def transient_then_success(messages, **kwargs):
                attempts.append((messages, kwargs))
                if len(attempts) == 1:
                    raise TimeoutError('Synthetic transport timeout')
                return await invoke(messages, **kwargs)

            with patch.object(llm.model, 'invoke', transient_then_success), \
                    patch.object(runtime.asyncio, 'sleep', new=AsyncMock()):
                asyncio.run(runtime.ask_task(llm, 'Arithmetic.', INSTRUCTION,
                                             model=MODEL, max_tokens=64))
            self.assertEqual(len(attempts), 2)
            records = llm.call_records
            self.assertEqual(len(records), 2)
            self.assertEqual(llm.memory_records[-1]['physical_calls'], 2)
            for record in records:
                self.assertGreater(record['memory_injected_utf8_bytes'], 0)
                self.assertEqual(record['memory_versions'], records[0]['memory_versions'])
            self.assertEqual(records[0]['charged_units'], records[0]['reserved_units'])
            self.assertNotEqual(records[0]['status'], 'provider_reconciled')
            self.assertEqual(records[1]['status'], 'provider_reconciled')


if __name__ == '__main__':
    unittest.main()
