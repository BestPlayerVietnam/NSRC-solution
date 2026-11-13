"""Retest configuration checks, no API and no simulated quality claims."""
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import sla_release_validation_v2 as v2


class SLARetestTests(unittest.TestCase):
    def test_eight_arms_are_short_only_same_image_and_independent_model_scopes(self):
        arms=v2.arm_specs('frozen-v2')
        self.assertEqual(len(arms),8)
        self.assertEqual({(a['initial_model'],a['sla_policy']) for a in arms.values()},
                         {(model,policy) for model in v2.v1.base.MODELS for policy in ('off','adaptive')})
        for item in arms.values():
            self.assertEqual(item['image'],'frozen-v2');self.assertEqual(item['time_limit'],90)
            self.assertEqual(item['budget'],100000);self.assertEqual(item['phase'],'short')

    def test_control_reuse_exception_cannot_waive_public_groups(self):
        def initialize(obj,destination):
            obj.groups={'gsm8k:new','humaneval:HumanEval/99','squad2:article',*v2.REUSED_GROUPS}
        with patch.object(v2.ORIGINAL_EXCLUSIONS,'__init__',initialize):
            exclusions=v2.ReuseNegativeExclusions(Path('new-suite'))
        self.assertEqual(exclusions.groups,{'gsm8k:new','humaneval:HumanEval/99','squad2:article'})
        self.assertEqual(len(v2.REUSED_GROUPS),4)

    def test_report_pair_suppression_restores_model_aliases_on_failure(self):
        aliases=v2.v1.base.MODEL_ALIASES
        with patch.object(v2.v1,'report',side_effect=RuntimeError('test checker failure')):
            with self.assertRaisesRegex(RuntimeError,'test checker failure'):
                v2.report(Path('suite'),Path('run'),'image')
        self.assertIs(v2.v1.base.MODEL_ALIASES,aliases)

    def test_source_freeze_includes_wrapper_and_frozen_grader(self):
        sources=v2.source_hashes()
        for relative in ('experiments/sla_release_validation_v2.py',
                         'experiments/sla_release_validation.py',
                         'experiments/final_release_validation.py'):
            self.assertEqual(sources[relative],v2.v1.sha(v2.ROOT/relative))


if __name__=='__main__':unittest.main()
