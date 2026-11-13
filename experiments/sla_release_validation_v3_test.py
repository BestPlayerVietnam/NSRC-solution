"""Protocol/configuration checks; no API or simulated quality claims."""
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import sla_release_validation_v3 as v3


class ExpectedTimeRetestTests(unittest.TestCase):
    def test_eight_independent_short_arms_keep_identical_non_sla_configuration(self):
        arms=v3.arm_specs('frozen-v3')
        self.assertEqual(len(arms),8)
        self.assertEqual({(a['initial_model'],a['sla_policy']) for a in arms.values()},
                         {(m,p) for m in v3.v1.base.MODELS for p in ('off','adaptive')})
        for a in arms.values():
            self.assertEqual((a['image'],a['time_limit'],a['budget'],a['phase']),('frozen-v3',90,100000,'short'))

    def test_source_freeze_includes_all_wrapper_generations(self):
        hashes=v3.source_hashes()
        for relative in ('experiments/sla_release_validation.py','experiments/sla_release_validation_v2.py',
                         'experiments/sla_release_validation_v3.py'):
            self.assertEqual(hashes[relative],v3.v1.sha(v3.ROOT/relative))

    def test_builder_uses_v2_parent_then_restores_immutable_helpers(self):
        original_parent=v3.v2.PARENT_SUITE;original_label=v3.v2.LABEL
        def check(suite):
            self.assertEqual(v3.v2.PARENT_SUITE,v3.PARENT_SUITE)
            self.assertEqual(v3.v2.LABEL,v3.LABEL)
            raise RuntimeError('stop before any write')
        with patch.object(v3.v2,'build',side_effect=check):
            with self.assertRaisesRegex(RuntimeError,'before any write'):v3.build(Path('fresh'))
        self.assertEqual(v3.v2.PARENT_SUITE,original_parent);self.assertEqual(v3.v2.LABEL,original_label)

    def test_report_restores_alias_map_on_grader_error(self):
        original=v3.v1.base.MODEL_ALIASES
        with patch.object(v3.v1,'report',side_effect=RuntimeError('grader failed')):
            with self.assertRaisesRegex(RuntimeError,'grader failed'):v3.report(Path('suite'),Path('run'),'image')
        self.assertIs(v3.v1.base.MODEL_ALIASES,original)


if __name__=='__main__':unittest.main()
