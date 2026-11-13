"""No-network checks of fresh-panel, budget and censoring contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from decimal import Decimal

from budget_runtime import BudgetLedger

from experiments import final_release_validation as final


class FinalValidationTests(unittest.TestCase):
    def test_eight_primary_and_two_decimal_env_stress_arms(self):
        arms=final.arm_specs('frozen-image')
        primary=[item for item in arms.values() if item['scope']=='primary']
        stress=[item for item in arms.values() if item['scope']!='primary']
        self.assertEqual(len(primary),8)
        self.assertEqual(len(stress),2)
        self.assertEqual({item['initial_model'] for item in primary},set(final.base.MODELS))
        for item in arms.values():
            command=final.docker_command('frozen-image',item,Path('visible-tasks'),
                                        'container-name',12345,'synthetic-token')
            self.assertEqual(command[command.index('--cpus')+1],'4')
            self.assertIn('--fixed-model',command)
            self.assertEqual(command[command.index('--confidence-policy')+1],'heuristic')
            self.assertNotIn('oracle.json',' '.join(command))
            if item['scope']=='primary':
                self.assertEqual(command[command.index('--budget')+1],'100000')
                self.assertFalse(any(x.startswith('BUDGET=') for x in command))
            else:
                self.assertIn('BUDGET=5000.5',command)
                self.assertNotIn('--budget',command)

    def test_proxy_denial_plus_local_reserve_is_not_known_zero_consumption(self):
        held=[{'calls':[{'status':'reserved_error_quota','weighted_tokens':600}]}]
        self.assertFalse(final.trusted_task_usage([],held))
        self.assertTrue(final.trusted_task_usage([],[]))
        physical=[{'accounting':'provider_reconciled'},{'accounting':'provider_reconciled'}]
        visits=[{'calls':[{'status':'provider_reconciled'}]},
                {'calls':[{'status':'provider_reconciled'}]}]
        self.assertTrue(final.trusted_task_usage(physical,visits))
        self.assertFalse(final.trusted_task_usage(physical,visits[:1]))

    def test_count_sum_and_omitted_predicate_annotations_are_rejected(self):
        def item(question,agg,conditions):
            return {'oracle':{'source_question':question,
                              'source_sql_ast':{'agg':agg,'conds':conditions}}}
        self.assertFalse(final.sql_question_unambiguous(item('How many goals did Blue score?',4,[[0,0,'Blue']])))
        self.assertFalse(final.sql_question_unambiguous(item('What is the total score for Blue?',3,[[0,0,'Blue']])))
        self.assertFalse(final.sql_question_unambiguous(item('What is the highest score for Blue?',1,[[0,0,'Blue'],[1,0,'2019']])))
        self.assertTrue(final.sql_question_unambiguous(item('What is the sum of scores for Blue in 2019?',4,[[0,0,'Blue'],[1,0,'2019']])))
        self.assertFalse(final.sql_question_unambiguous(item('What is the sum for Blue in 2019?',4,[[0,0,'Blue'],[1,1,'2019']])))

    def test_legacy_windows_csv_is_validated_not_waived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);suite=root/'experiments/suite';folder=suite/'tasks/old';work=folder/'workspace'
            work.mkdir(parents=True)
            original='x,y\r\n1,2\r\n'
            (work/'sales.csv').write_bytes(original.replace('\n','\r\n').encode())
            task={'task_type':'data_analysis','instruction':'Look up y for x=1.'}
            dump=lambda value:json.dumps(value,ensure_ascii=False,indent=2)
            (folder/'task.json').write_text(dump(task),encoding='utf-8')
            expected=hashlib.sha256((dump(task)+dump({'sales.csv':original})).encode()).hexdigest()
            final.write_new(suite/'manifest.json',{'tasks':[{'id':'old','task_path':'tasks/old',
                'group':'wikisql:old','input_sha256':expected}]})
            with patch.object(final,'ROOT',root):
                excluded=final.Exclusions(root/'experiments/new-suite')
                self.assertEqual(excluded.hash_methods['legacy_combined_reverse_windows_write'],1)
                self.assertIn('wikisql:old',excluded.groups)
                (work/'sales.csv').write_bytes(b'x,y\n1,999\n')
                with self.assertRaisesRegex(ValueError,'combined input hash mismatch'):
                    final.Exclusions(root/'experiments/new-suite')

    def test_authored_cases_have_four_positive_and_four_certified_negative(self):
        cases=final.authored_cases()
        self.assertEqual(len(cases),8)
        for domain in final.public.DOMAINS:
            selected=[case for case in cases if case[0]==domain]
            self.assertEqual(len(selected),2)
            self.assertEqual(sum(case[4] is not None for case in selected),1)
            self.assertTrue(next(case[5]['certificate'] for case in selected if case[4] is None))

    def test_reward_hash_uses_only_seed_and_task_id(self):
        ident='opaque-task-id'
        expected=1+int(hashlib.sha256(f'{final.SEED}:{ident}'.encode()).hexdigest(),16)%5
        self.assertEqual(final.task_reward(ident),expected)
        values=[final.task_reward('opaque-'+str(index)) for index in range(100)]
        self.assertEqual(set(values),{1,2,3,4,5})
        self.assertEqual(values,[final.task_reward('opaque-'+str(index)) for index in range(100)])

    def test_weighted_reward_is_separate_from_solved_count(self):
        score=final.score_summary([
            {'answerable':True,'correct':True,'reward':4},
            {'answerable':True,'correct':True,'reward':2},
            {'answerable':True,'correct':False,'reward':3},
            {'answerable':False,'correct':False,'reward':5}])
        self.assertEqual(score['solved_count'],2)
        self.assertEqual(score['reward'],6)
        self.assertEqual(score['max_reward'],9)
        self.assertEqual(score['answerable_count'],3)

    def test_decimal_stress_clipped_runtime_cap_also_fits_aligned_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            model=final.base.MODELS[3]
            system,user='Числовой ответ.','Compute 17 + 25.\nMemory: use a supported expression.'
            runtime=BudgetLedger(Decimal('5000.5'))
            reservation=runtime.prepare(system,user,max_tokens=3072,model=model)
            messages=json.dumps([{'role':'system','content':system},{'role':'user','content':user}],ensure_ascii=False)
            proxy=final.AlignedLedger(Path(temporary)/'ledger.json',limit=5000.5)
            reserved=proxy.reserve(model,reservation.max_tokens,messages,ceiling=5000.5)
            self.assertIsNotNone(reserved)
            self.assertEqual(reserved[1],reservation.reserved_units)
            self.assertLessEqual(proxy.spent(model),5000.5)
            usage={'input_tokens':100,'output_tokens':25,'total_tokens':125}
            runtime.reconcile(reservation,usage)
            proxy.reconcile(model,*reserved,usage)
            self.assertEqual(proxy.spent(model),runtime.spent)

    def test_unrecognized_message_shape_retains_conservative_serialized_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger=final.AlignedLedger(Path(temporary)/'ledger.json')
            messages=json.dumps([{'role':'user','content':[{'type':'text','text':'hello'}]}])
            reserved=ledger.reserve(final.base.MODELS[0],100,messages)
            self.assertEqual(reserved[1],2000+len(messages.encode('utf-8'))+1280)

    def test_forecast_diagnostics_use_identical_complete_cohort_and_whole_policy_cost(self):
        def row(ident,group,cost,**updates):
            result={'id':ident,'group':group,'domain':'math','cost':cost,'calls':2,
                'unattempted':False,'usage_known':True,'policy_complete':True,
                'first_provider_cost':30,'initial_forecasts':{'frozen_total':100,
                    'online_total':120,'legacy_heuristic':80,'first_call':40}}
            result.update(updates)
            return result
        rows=[row('a','g1',150),row('b','g1',100),row('c','g2',200),
              row('held','g3',300,usage_known=False),row('cut','g4',400,policy_complete=False),
              row('local','g5',0,calls=0),row('waiting','g6',0,unattempted=True),
              row('invalid','g7',500,initial_forecasts={'frozen_total':float('nan')})]
        diagnostic=final.prediction_diagnostics(rows)
        self.assertEqual(diagnostic['eligible_ids'],['a','b','c'])
        self.assertEqual(diagnostic['eligible_count'],3)
        self.assertEqual(diagnostic['group_count'],2)
        self.assertEqual(diagnostic['total']['frozen_total']['mae'],50)
        self.assertEqual(diagnostic['total']['frozen_total']['group_macro_mae'],62.5)
        self.assertAlmostEqual(diagnostic['total']['online_total']['mae'],130/3)
        self.assertEqual(diagnostic['first_call']['mae'],10)
        self.assertEqual(sum(diagnostic['excluded_counts'].values()),5)
        self.assertEqual({v['count'] for v in diagnostic['total'].values()},{3})

    def test_empty_forecast_cohort_has_no_zero_error_claim(self):
        diagnostic=final.prediction_diagnostics([])
        self.assertEqual(diagnostic['eligible_count'],0)
        self.assertIsNone(diagnostic['total']['online_total']['mae'])
        self.assertIsNone(diagnostic['first_call']['mae'])


if __name__=='__main__':unittest.main()
