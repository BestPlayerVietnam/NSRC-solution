"""SLA-v2 fresh-public retest: eight90-second arms on one new frozen image.

All twelve public groups are new. The four authored negatives are explicitly
reused controls. This wrapper never rewrites the frozen v1 runner or protocol.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import sla_release_validation as v1

SUITE=ROOT/'experiments/suite_sla_release_v2'
RUN=ROOT/'experiments/runs/2026-11-14-sla-release-v2'
PARENT_SUITE=ROOT/'experiments/suite_sla_release'
LABEL='SLA-v2 short-deadline retest; twelve fresh public groups and four reused negative controls'
ORIGINAL_SOURCE_HASHES=v1.source_hashes
ORIGINAL_EXCLUSIONS=v1.previous.Exclusions
REUSED_GROUPS={'sla_authored:'+row[1] for row in v1.negative_cases()}
ADDITIONAL_REJECTIONS={
    'gsm8k:57b66f30c31f87969bd1e81dcb82893d29d0d9a624040ffc3d023f0fbb409b14':
        'Preflight: weekly sweater total states only Monday through Friday, without constraining the weekend.'}


class ReuseNegativeExclusions(ORIGINAL_EXCLUSIONS):
    def __init__(self,destination):
        super().__init__(destination)
        # An explicit control reuse exception, never a public-group waiver.
        self.groups.difference_update(REUSED_GROUPS)


def source_hashes():
    return {**ORIGINAL_SOURCE_HASHES(),Path(__file__).relative_to(ROOT).as_posix():v1.sha(__file__)}


def configure():
    v1.source_hashes=source_hashes


def arm_specs(image):
    return {name:item for name,item in v1.arm_specs(image).items() if item['phase']=='short'}


def build(suite):
    with patch.object(v1,'LABEL',LABEL),patch.object(v1.previous,'Exclusions',ReuseNegativeExclusions),\
         patch.object(v1,'REJECTED_GROUPS',{**v1.REJECTED_GROUPS,**ADDITIONAL_REJECTIONS}):
        v1.build(suite)
    current=v1.read(suite/'research_metadata.json');old=v1.read(PARENT_SUITE/'research_metadata.json')
    prior_public={value['group'] for value in old.values() if value['source_family']!='authored'}
    new_public={value['group'] for value in current.values() if value['source_family']!='authored'}
    controls={value['group'] for value in current.values() if value['source_family']=='authored'}
    if len(new_public)!=12 or prior_public&new_public or controls!=REUSED_GROUPS:
        raise ValueError('Public freshness or declared control reuse mismatch')
    v1.write_new(suite/'reuse_provenance.json',{'parent_suite':PARENT_SUITE.relative_to(ROOT).as_posix(),
        'parent_manifest_sha256':v1.sha(PARENT_SUITE/'manifest.json'),
        'fresh_public_group_count':12,'reused_negative_group_count':4,'reused_groups':sorted(REUSED_GROUPS),
        'scope':'Negative controls were already evaluated in v1; do not claim the entire panel is independent.',
        'public_group_overlap':[]})


def prepare(suite,run,image):
    configure()
    if run.exists():raise FileExistsError('Fresh run directory required')
    runtime=v1.image_runtime(image)
    if v1.read(suite/'validation.json')['passed']!=16:raise ValueError('Validate all16 oracles first')
    v1.helper.verify_visible(suite,v1.read(suite/'manifest.json'))
    v1.write_new(run/'protocol.json',{'label':LABEL,'version':'sla-v2','created':time.time(),
        'image':image,'arms':arm_specs(image),'source_sha256':source_hashes(),'runtime_sha256':runtime,
        'suite_sha256':{path.relative_to(suite).as_posix():v1.sha(path) for path in suite.rglob('*') if path.is_file()},
        'task_count':16,'positive_count':12,'negative_count':4,'fresh_public_groups':12,'reused_negative_groups':4,
        'ceilings':{model:100000 for model in v1.base.MODELS},'max_parallel_pipelines':2,'serialize_provider_family':True,
        'cpus_per_pipeline':4,'fixed_configuration':{'confidence_policy':'heuristic','memory_policy':'revise','retry_policy':'adaptive','fixed_model':True},
        'short_deadline_seconds':90,'full_phase_executed':False,'external_watchdog_grace_seconds':30,
        'primary_objective':'Maximize verified reward emitted by the agent90-second deadline; then minimize charged cost.',
        'deadline_clock':'Agent run-relative elapsed starts before entry scan; container startup and relay drain are separate.',
        'deadline_score':'Only recorded answer_elapsed_seconds <=90 earns on-time points; unknown timestamps reported separately.',
        'unknown_usage_policy':'Retain reservations and charge late provider completions; exclude censored tasks from forecast accuracy.',
        'paired_report':'comparison.json references immutable evaluation.json; v1 fixed two-phase pairing is disabled only in memory.',
        'limitations':['SLA-v1 data diagnosed the unit mismatch and is now development evidence.',
            'Twelve public groups are fresh; four authored negative controls are reused and disclosed.',
            'Only the90-second comparison is repeated. Full3300-second v1 results remain historical, not measurements of the new image.',
            'One realization per arm; service/model variability prevents universal gain or latency guarantees.',
            'No outcome-driven fitting or threshold selection within this retest.']})
    print({'prepared':True,'arms':8,'scheduled_episodes':128,'image':image})


def report(suite,run,image):
    configure()
    # Reuse the exact frozen grading/accounting implementation. Its final
    # two-phase pair loop cannot handle a short-only panel, so suppress that
    # loop in memory and write the comparison to a separate immutable file.
    aliases=dict(v1.base.MODEL_ALIASES)
    with patch.object(v1.base,'MODEL_ALIASES',{}),redirect_stdout(io.StringIO()):
        v1.report(suite,run,image)
    evaluation=v1.read(run/'evaluation.json');paired={}
    for alias in aliases:
        off=evaluation['arms']['short_off_'+alias];on=evaluation['arms']['short_adaptive_'+alias]
        paired[alias]={'delta_on_time_reward':on['on_time_reward']-off['on_time_reward'],
            'delta_on_time_solved':on['on_time_solved']-off['on_time_solved'],
            'delta_cost':round(on['cost']-off['cost'],1),
            'delta_container_seconds':on['container_seconds']-off['container_seconds'],
            'different_coverage':on['unattempted']!=off['unattempted']}
    v1.write_new(run/'comparison.json',{'version':'sla-v2','image':image,
        'evaluation_sha256':v1.sha(run/'evaluation.json'),'protocol_sha256':v1.sha(run/'protocol.json'),
        'fresh_public_groups':12,'reused_negative_groups':4,'paired':paired})
    print({'all_processes_exit_zero':evaluation['all_processes_exit_zero'],'paired':paired})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','validate','prepare','run','report'])
    parser.add_argument('--suite',type=Path,default=SUITE);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--image');args=parser.parse_args();configure()
    if args.action=='build':build(args.suite.resolve())
    elif not args.image:parser.error('--image required')
    elif args.action=='validate':v1.validate(args.suite.resolve(),args.image)
    elif args.action=='run':v1.run_experiment(args.suite.resolve(),args.run.resolve(),args.image)
    else:{'prepare':prepare,'report':report}[args.action](args.suite.resolve(),args.run.resolve(),args.image)


if __name__=='__main__':main()
