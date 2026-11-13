"""Expected-time SLA retest: eight paired90-second arms, no new fitting.

Twelve fresh public groups and four reused authored negative controls. Frozen
v1/v2 source files and their paid histories remain unchanged.
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
from experiments import sla_release_validation_v2 as v2

v1=v2.v1
SUITE=ROOT/'experiments/suite_sla_release_v3'
RUN=ROOT/'experiments/runs/2026-11-14-sla-release-v3'
PARENT_SUITE=v2.SUITE
LABEL='SLA-v3 expected-time retest; twelve fresh public groups and four reused negative controls'
ORIGINAL_SOURCE_HASHES=v2.source_hashes
ADDITIONAL_REJECTIONS={
    'gsm8k:41061160a62df06fc3797150d515a73070554ece1a534003441c04790f2c1c89':
        'Preflight: elapsed cooking time requires unstated sequential rather than overlapping popcorn/fries preparation.',
    'gsm8k:cd01ef3096493fef344180280ac98f298cd2c33fc4a3bc26beb64e5d87a51871':
        'Preflight: source gold treats five30-pen boxes as five6-pen packages; literal profit is95, not the supplied115.'}


def source_hashes():
    return {**ORIGINAL_SOURCE_HASHES(),Path(__file__).relative_to(ROOT).as_posix():v1.sha(__file__)}


def configure():
    v1.source_hashes=source_hashes


def arm_specs(image):
    return v2.arm_specs(image)


def build(suite):
    with patch.object(v2,'LABEL',LABEL),patch.object(v2,'PARENT_SUITE',PARENT_SUITE),\
         patch.object(v2,'ADDITIONAL_REJECTIONS',{**v2.ADDITIONAL_REJECTIONS,**ADDITIONAL_REJECTIONS}):
        v2.build(suite)
    v1.write_new(suite/'retest_scope.json',{'version':'sla-v3','fresh_public_groups':12,
        'reused_authored_negative_groups':4,'previous_development_versions':['sla-v1','sla-v2'],
        'scope':'All public groups excluded against previous suites; four negative controls deliberately reused.'})


def prepare(suite,run,image):
    configure()
    if run.exists():raise FileExistsError('Fresh run directory required')
    runtime=v1.image_runtime(image)
    if v1.read(suite/'validation.json')['passed']!=16:raise ValueError('Validate all16 oracles first')
    v1.helper.verify_visible(suite,v1.read(suite/'manifest.json'))
    v1.write_new(run/'protocol.json',{'label':LABEL,'version':'sla-v3','created':time.time(),
        'image':image,'arms':arm_specs(image),'source_sha256':source_hashes(),'runtime_sha256':runtime,
        'suite_sha256':{path.relative_to(suite).as_posix():v1.sha(path) for path in suite.rglob('*') if path.is_file()},
        'task_count':16,'positive_count':12,'negative_count':4,'fresh_public_groups':12,'reused_negative_groups':4,
        'ceilings':{model:100000 for model in v1.base.MODELS},'max_parallel_pipelines':2,'serialize_provider_family':True,
        'cpus_per_pipeline':4,'fixed_configuration':{'confidence_policy':'heuristic','memory_policy':'revise','retry_policy':'adaptive','fixed_model':True},
        'short_deadline_seconds':90,'full_phase_executed':False,'external_watchdog_grace_seconds':30,
        'primary_objective':'Maximize verified reward emitted by the agent90-second deadline; then minimize charged cost.',
        'candidate_change':'Expected seconds drive joint selection/admission/generation caps; upper seconds remain diagnostic. Absolute SDK/local deadlines remain strict. Priors are unchanged.',
        'deadline_clock':'Agent run-relative elapsed starts before entry scan; container startup and relay drain are separate.',
        'deadline_score':'Only recorded answer_elapsed_seconds <=90 earns on-time points; unknown timestamps reported separately.',
        'unknown_usage_policy':'Retain reservations and charge late provider completions; exclude censored tasks from forecast accuracy.',
        'paired_report':'comparison.json references immutable evaluation.json; v1 fixed two-phase pairing is disabled only in memory.',
        'limitations':['SLA-v1/v2 are development evidence; V3 changes a generic resource objective without model-specific thresholds or fitted priors.',
            'Twelve public groups are fresh; four authored negative controls are reused and disclosed.',
            'Only the90-second comparison is repeated; no new3300-second measurement is claimed.',
            'One realization per arm; service/model variability prevents universal gain or latency guarantees.',
            'No outcome-driven fitting or threshold selection within this retest.']})
    print({'prepared':True,'arms':8,'scheduled_episodes':128,'image':image})


def report(suite,run,image):
    configure()
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
    v1.write_new(run/'comparison.json',{'version':'sla-v3','image':image,
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
