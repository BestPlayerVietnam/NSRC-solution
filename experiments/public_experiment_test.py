"""No API, Docker or actual public gold access: grading and freeze guard tests."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import public_experiment as public


class GradingTests(unittest.TestCase):
    def gold(self):
        return {'answerable':True,'expected':'The Harbour','accepted_answers':['The Harbour','Harbor']}

    def test_all_gold_variants_are_accepted(self):
        scores=public.search_scores('HARBOR!',self.gold())
        self.assertTrue(scores['normalized'])
        self.assertFalse(scores['exact'])
        self.assertFalse(scores['canonical'])
        self.assertTrue(public.search_scores(' Harbor ',self.gold())['exact'])

    def test_official_normalization_order(self):
        self.assertEqual(public.normalize_squad('  THE, Harbour!\n'),'harbour')
        self.assertEqual(public.normalize_squad('An apple; a pear.'),'apple pear')
        self.assertNotEqual(public.normalize_squad('New-York'),public.normalize_squad('New York'))
        self.assertNotEqual(public.normalize_squad('Straße'),public.normalize_squad('STRASSE'))
        self.assertNotEqual(public.normalize_squad('“Harbor”'),public.normalize_squad('Harbor'))

    def test_missing_and_negative_never_earn_reward(self):
        self.assertFalse(public.search_scores(None,self.gold())['normalized'])
        negative={'answerable':False,'expected':None}
        self.assertFalse(public.search_scores('UNSOLVABLE',negative)['normalized'])
        self.assertFalse(public.search_scores('',negative)['normalized'])

    def test_official_empty_normalized_gold_fallback(self):
        gold={'answerable':True,'expected':'the','accepted_answers':['the','a']}
        self.assertTrue(public.search_scores('',gold)['normalized'])

    def test_regrade_preserves_other_domains_and_all_costs(self):
        rows=[
            {'id':'s','domain':'search','answer':'HARBOR!','correct':False,'reward':4,'cost':12.3},
            {'id':'m','domain':'math','answer':'1,000','correct':False,'reward':2,'cost':5},
            {'id':'c','domain':'coding','answer':'DONE','correct':True,'reward':3,'cost':7},
        ]
        evaluation={'arms':{'a':{'rows':rows,'solved':1,'reward':3,'cost':24.3,'calls':4}},'budget':{'spent':24.3}}
        original=copy.deepcopy(evaluation)
        result=public.regrade_search(evaluation,{'s':self.gold()})
        self.assertEqual(evaluation,original)
        self.assertEqual(result['arms']['a']['solved'],2)
        self.assertEqual(result['arms']['a']['reward'],7)
        self.assertEqual(result['arms']['a']['strict_exact_solved'],1)
        self.assertEqual(result['arms']['a']['cost'],24.3)
        self.assertEqual(result['budget'],original['budget'])
        self.assertFalse(result['arms']['a']['rows'][1]['correct'])
        self.assertTrue(result['arms']['a']['rows'][2]['correct'])

    def test_malformed_positive_gold_fails_closed(self):
        with self.assertRaises(ValueError):
            public.search_scores('x',{'answerable':True,'expected':'x','accepted_answers':[]})


class ExecutorTests(unittest.TestCase):
    def test_same_provider_family_is_serialized_before_run_arm(self):
        import threading, time
        active=0;peak=0;guard=threading.Lock()
        def job(arm,item):
            nonlocal active,peak
            with guard:active+=1;peak=max(peak,active)
            time.sleep(0.01)
            with guard:active-=1
            return arm
        with public.FamilyExecutor(max_workers=4) as pool:
            futures=[pool.submit(job,str(i),{'initial_model':'qwen3.6-35b-a3b'+('-noreason' if i%2 else '')}) for i in range(8)]
            self.assertEqual([f.result(timeout=5) for f in futures],[str(i) for i in range(8)])
        self.assertEqual(peak,1)

    def test_different_families_can_progress_together(self):
        import threading
        barrier=threading.Barrier(2)
        def job(arm,item):
            barrier.wait(timeout=3)
            return arm
        with public.FamilyExecutor(max_workers=2) as pool:
            futures=[pool.submit(job,str(i),{'initial_model':m}) for i,m in enumerate(['qwen3.6-35b-a3b','qwen3.8-27b-noreason'])]
            self.assertEqual([f.result(timeout=5) for f in futures],['0','1'])


class FreezeTests(unittest.TestCase):
    def test_modified_wrapper_or_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper=Path(tmp)/'wrapper.py';runner=Path(tmp)/'runner.py'
            wrapper.write_text('wrapper-v1');runner.write_text('runner-v1')
            paths={'wrapper_sha256':wrapper,'runner_sha256':runner}
            protocol={key:public.sha(path) for key,path in paths.items()}
            public.verify_hashes(protocol,paths)
            wrapper.write_text('wrapper-v2')
            with self.assertRaisesRegex(RuntimeError,'wrapper_sha256'):public.verify_hashes(protocol,paths)
            wrapper.write_text('wrapper-v1');runner.write_text('runner-v2')
            with self.assertRaisesRegex(RuntimeError,'runner_sha256'):public.verify_hashes(protocol,paths)

    def test_missing_hash_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x';path.write_text('x')
            with self.assertRaises(RuntimeError):public.verify_hashes({}, {'wrapper_sha256':path})

    def test_run_guard_precedes_any_base_execution(self):
        with patch.object(public,'configure'),patch.object(public,'verify_protocol',side_effect=RuntimeError('changed')),patch.object(public.base,'run') as run:
            with self.assertRaises(RuntimeError):public.run()
            run.assert_not_called()

    def test_report_guard_precedes_gold_or_docker(self):
        with patch.object(public,'configure'),patch.object(public,'verify_protocol',side_effect=RuntimeError('changed')),patch.object(public.base,'report') as report:
            with self.assertRaises(RuntimeError):public.report()
            report.assert_not_called()


if __name__=='__main__':unittest.main()
