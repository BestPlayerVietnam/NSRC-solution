"""Software checks only; no simulated quality or timing improvement claim."""
from pathlib import Path
import unittest

from experiments import sla_release_validation as sla


class SLAExperimentTests(unittest.TestCase):
    def test_sixteen_paired_arms_change_only_policy_and_deadline(self):
        arms=sla.arm_specs('frozen-image')
        self.assertEqual(len(arms),16)
        tuples={(a['initial_model'],a['sla_policy'],a['time_limit']) for a in arms.values()}
        self.assertEqual(tuples,{(model,policy,seconds) for model in sla.base.MODELS
                                for policy in ('off','adaptive') for seconds in (90,3300)})
        for arm,item in arms.items():
            cmd=sla.docker_command('frozen-image',item,Path('visible-tasks'),'name',12345,'synthetic-token')
            for flag,value in [('--cpus','4'),('--budget','100000'),('--memory-policy','revise'),
                               ('--confidence-policy','heuristic'),('--sla-policy',item['sla_policy']),
                               ('--time-limit',str(item['time_limit']))]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertIn('--fixed-model',cmd)
            self.assertNotIn('oracle.json',' '.join(cmd))
        self.assertEqual(list(arms),list(sla.arm_specs('frozen-image')))

    def test_deadline_score_uses_actual_emission_not_finish_of_last_api(self):
        self.assertEqual(sla.answer_deadline_status('42',[{'timing':{'answer_elapsed_seconds':90}}],90),('on_time',90))
        self.assertEqual(sla.answer_deadline_status('42',[{'timing':{'answer_elapsed_seconds':90.001}}],90),('late',90.001))
        self.assertEqual(sla.answer_deadline_status(None,[{'timing':{'answer_elapsed_seconds':5}}],90),('no_answer',None))
        self.assertEqual(sla.answer_deadline_status('42',[{'timing':{'run_elapsed_seconds':40}}],90),('unknown_timestamp',None))

    def test_current_runtime_latency_forecast_field_is_supported(self):
        self.assertEqual(sla.time_prediction({'seconds':12.5,'conservative_seconds':30}),12.5)
        self.assertEqual(sla.time_prediction({'predicted_seconds':14,'seconds':12.5}),14)
        self.assertIsNone(sla.time_prediction({}))

    def test_terminal_deadline_stop_is_censored_without_interrupted_request(self):
        terminals=[{'solver_state':'time_limit_stopped'},
            {'retry_decisions':[{'reason':'sla_global_deadline_insufficient'}]},
            {'sla_decisions':[{'phase':'continuation','eligible':False,'reason':'global_deadline_insufficient'}]}]
        for terminal in terminals:
            self.assertTrue(sla.task_deadline_limited([terminal]))
            self.assertFalse(sla.task_deadline_limited([terminal],answered=True))

    def test_resumed_defer_or_successful_reduced_cap_is_not_permanent_censoring(self):
        deferred={'outcome':'deferred','sla_decisions':[{'phase':'continuation','eligible':False,'reason':'global_deadline_insufficient'}]}
        resumed={'execution_complete':True,'outcome':'answered_unverified','solver_state':'answered'}
        self.assertFalse(sla.task_deadline_limited([deferred,resumed],answered=True))
        self.assertFalse(sla.task_deadline_limited([deferred,{'execution_complete':True,'outcome':'skipped','solver_state':'abstained'}]))
        cap_reduced={'sla_decisions':[{'phase':'generation','reason':'generation_cap_reduced_for_time'}]}
        self.assertFalse(sla.task_deadline_limited([cap_reduced],answered=True))
        self.assertFalse(sla.task_deadline_limited([cap_reduced]))
        timed_out={'calls':[{'latency_censored':True}]}
        self.assertTrue(sla.task_deadline_limited([timed_out,resumed],answered=True))

    def test_overwritten_answer_uses_latest_emission_and_invalid_clocks_are_unknown(self):
        visits=[{'timing':{'answer_elapsed_seconds':40}},{'timing':{'answer_elapsed_seconds':91}}]
        self.assertEqual(sla.answer_deadline_status('latest',visits,90),('late',91))
        for value in (True,-1,float('nan'),float('inf'),'30'):
            self.assertEqual(sla.answer_deadline_status('42',[{'timing':{'answer_elapsed_seconds':value}}],90),('unknown_timestamp',None))

    def test_late_provider_charge_is_kept_when_local_deadline_holds_reservation(self):
        physical=[{'cost':67.5,'accounting':'provider_reconciled'}]
        visits=[{'calls':[{'status':'reserved_error_timeout','weighted_tokens':1000}]}]
        value=sla.task_usage(physical,visits)
        self.assertEqual(value['cost'],67.5)
        self.assertTrue(value['provider_usage_known'])
        self.assertFalse(value['local_usage_reconciled'])
        held=sla.task_usage([{'cost':1000,'accounting':'reserved_error'}],visits)
        self.assertEqual(held['cost'],1000)
        self.assertFalse(held['provider_usage_known'])

    def test_local_hold_without_proxy_request_is_not_measured_zero(self):
        self.assertFalse(sla.task_usage([],[{'calls':[{'status':'reserved_error_timeout'}]}])['provider_usage_known'])
        self.assertTrue(sla.task_usage([],[])['provider_usage_known'])

    def test_negative_panel_has_one_certified_case_per_domain(self):
        cases=sla.negative_cases()
        self.assertEqual({row[0] for row in cases},set(sla.public.DOMAINS))
        self.assertEqual(len(cases),4)
        coding=next(row for row in cases if row[0]=='coding')
        self.assertIn('type(r) is tuple',coding[3]['test_mylib.py'])
        self.assertIn('len(r) == 2',coding[3]['test_mylib.py'])
        self.assertIn('len(r) == 3',coding[3]['test_mylib.py'])
        self.assertIn(sla.helper.INTEGRITY,coding[2])

    def test_count_annotations_are_excluded_before_model_execution(self):
        row={'oracle':{'source_question':'What is the total number of bronze medals for Blue?',
             'source_sql_ast':{'agg':3,'conds':[[0,0,'Blue']]}}}
        self.assertFalse(sla.sql_candidate_valid(row))
        row['oracle']['source_question']='What is the sum of scores for Blue?'
        row['oracle']['source_sql_ast']['agg']=4
        self.assertTrue(sla.sql_candidate_valid(row))


if __name__=='__main__':unittest.main()
