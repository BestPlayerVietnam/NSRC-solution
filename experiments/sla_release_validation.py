"""Fresh, frozen SLA-policy comparison; no API calls during build/prepare/report.

Sixteen tasks and sixteen independent arms compare off/adaptive on the SAME
image at3300s and90s. The short deadline is a secondary pressure diagnostic.
All reward assignment, deadlines and grading are fixed before paid execution.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import as_completed
from decimal import Decimal
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import final_release_validation as previous
from experiments import build_public_panel as public
from experiments import confidence_probe as helper
from experiments import policy_experiment as base
from experiments.public_experiment import FamilyExecutor

SUITE = ROOT/'experiments/suite_sla_release'
RUN = ROOT/'experiments/runs/2026-11-13-sla-release'
SEED = 202611131701
TASK_COUNT = 16
FULL_SECONDS = 3300
SHORT_SECONDS = 90
LABEL = 'Fresh SLA development comparison; no fitting on this panel'
sha, read, write_new = helper.sha, helper.read, helper.write_new
REJECTED_GROUPS = {
    **previous.REJECTED_SOURCE_GROUPS,
    'gsm8k:580b134d77b755152688d891863fbd03405d099741fc022dcf5802de8e6e2afb':
        'One-third less white cheese leaves the comparison quantity unclear.',
    'gsm8k:66c24c652546626d4e86e77443877f98581e75535372f913ba10c232351460a0':
        'Savings calculation requires an unstated52-weeks-per-year convention.'}


def reward_for(ident):
    return 1+int(hashlib.sha256(f'{SEED}:{ident}'.encode()).hexdigest(),16)%5


def sql_candidate_valid(row):
    # Count-of-rows versus numerical quantity is ambiguous in several source
    # questions. This panel excludes COUNT before any model execution.
    return row['oracle']['source_sql_ast']['agg']!=3 and previous.sql_question_unambiguous(row)


class CandidateBuilder(previous.CandidateBuilder):
    def add(self,domain,group,question,files,expected,**kwargs):
        if group not in REJECTED_GROUPS:
            super().add(domain,group,question,files,expected,**kwargs)


def negative_cases():
    return [
        ('math','unrecorded_second_tank',
         'Tank A initially holds64 litres and loses exactly one quarter of its contents. '
         'Tank B initially holds an unrecorded nonnegative quantity, then receives12 litres. '
         'The tanks are independent and there is no capacity or total-volume constraint. '
         'What is the combined final volume? Return UNSOLVABLE if it cannot be uniquely determined.',{},
         'Initial B=0 gives60 litres; B=20 gives80 litres. Both satisfy every stated fact.'),
        ('search','visitor_desk_mapping',
         'Using only the supplied complete register, give the contact code of the desk assigned to visitor OMEGA-63. '
         'Visitor assignments are independent; there is no default desk. Return UNSOLVABLE if the assignment is absent.',
         {'kb/register.md':'Desk East has contact code ACORN-26. Desk West has contact code WILLOW-71. '
          'The assignment for visitor OMEGA-63 was not recorded. No other fact determines that assignment.\n'},
         'Assigning OMEGA-63 to East or West leaves the register unchanged and yields different contact codes.'),
        ('data_analysis','unrecorded_hourly_rate',
         'Compute the exact labour invoice for project oak: sum its hours and multiply by its single hourly rate. '
         'The hourly rate is nonnegative, was not recorded, and is independent of all table values. '
         'There are no discounts or fixed fees. Return UNSOLVABLE if the exact invoice is undetermined.',
         {'work.csv':'project,hours\noak,4\noak,5\nash,8\n'},
         'Nine oak hours at rate10 yield90; rate20 yields180. Both rates satisfy all visible constraints.'),
        ('coding','immutable_tuple_lengths',
         'Implement dimensions() returning one exact built-in tuple r satisfying all public assertions. '
         'Functions must be stateless and deterministic under ordinary Python semantics. '+helper.INTEGRITY,
         {'mylib.py':'def dimensions():\n    return (0, 0)\n',
          'test_mylib.py':'from mylib import dimensions\n\ndef test_contract():\n    r = dimensions()\n    assert type(r) is tuple\n    assert len(r) == 2\n    assert len(r) == 3\n'},
         'One exact builtin tuple has a fixed length; it cannot have lengths2 and3 under the integrity contract.')]


def coding_cases(builder):
    """Reuse the source filter, requiring only this panel's three fresh groups."""
    rows=[json.loads(line) for line in gzip.decompress((public.CACHE/public.ASSETS['humaneval'][0]).read_bytes()).decode().splitlines()]
    candidates=[]
    for row in rows:
        if 'humaneval:'+row['task_id'] in builder.excluded.groups:continue
        complete=row['prompt']+row['canonical_solution']
        try:tree=ast.parse(complete);test_tree=ast.parse(row['test'])
        except SyntaxError:continue
        nodes=list(ast.walk(tree));branches=sum(isinstance(n,(ast.For,ast.While,ast.If,ast.IfExp,ast.comprehension)) for n in nodes)
        if len(nodes)<85 or branches<2 or len(row['prompt'])<220 or len(complete)>8500:continue
        if any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(name.split('.')[0] not in public.ALLOWED_IMPORTS
            for name in ([n.module or ''] if isinstance(n,ast.ImportFrom) else [alias.name for alias in n.names])) for n in nodes):continue
        if any(isinstance(n,ast.Name) and n.id in {'eval','exec','compile','open','input','__import__'} for n in nodes):continue
        assertions=[n for n in ast.walk(test_tree) if isinstance(n,ast.Assert)
            and any(isinstance(x,ast.Name) and x.id=='candidate' for x in ast.walk(n))
            and {x.id for x in ast.walk(n) if isinstance(x,ast.Name)}<=public.PUBLIC_NAMES]
        if assertions:candidates.append((row,complete,len(nodes),branches,assertions))
    candidates.sort(key=lambda item:public.digest(f'{SEED}:{item[0]["task_id"]}'))
    if len(candidates)<3:raise ValueError('Insufficient unused HumanEval cases')
    for index,(row,complete,nodes,branches,assertions) in enumerate(candidates[:3]):
        nested=index%2==1;target='library/implementation.py' if nested else 'mylib.py'
        module='library.implementation' if nested else 'mylib';entry=row['entry_point']
        tests=f'from {module} import {entry} as candidate\n\n'
        for number,node in enumerate(assertions[:2]):tests+=f'def test_public_example_{number}():\n    '+ast.unparse(node)+'\n\n'
        files={target:row['prompt']+"    raise NotImplementedError('Restore the documented function')\n",
               'tests/test_public.py' if nested else 'test_mylib.py':tests}
        if nested:files['library/__init__.py']=''
        hidden=f'from {module} import {entry}\n\n'+row['test']+f'\n\ndef test_original_hidden_checks():\n    check({entry})\n'
        question=f'Implement {entry} in {target}. Preserve its public signature. The COMPLETE specification follows; examples do not replace the general contract. Do not modify tests, monkeypatch built-ins or inspect hidden tests. Additional tests will check the implementation.\n\n'+row['prompt']+'\nWhen the implementation is complete and public tests pass, write DONE to answer.txt.'
        builder.add('coding','humaneval:'+row['task_id'],question,files,'DONE',source_name='humaneval',source_id=row['task_id'],
            validation_kind='coding',hidden_files={'tests/test_hidden_sla.py':hidden},reference_files={target:complete},
            public_profile='public_fail',source_ast_nodes=nodes,source_branch_count=branches,source_prompt=row['prompt'])


def build(suite):
    if suite.exists():raise FileExistsError('Fresh suite path required')
    provenance=public.sources(False)
    excluded=previous.Exclusions(suite)
    candidates=CandidateBuilder(excluded)
    original_seed=public.SEED
    try:
        public.SEED=SEED
        public.math_cases(candidates);coding_cases(candidates)
        public.search_cases(candidates);public.data_cases(candidates)
    finally:public.SEED=original_seed
    chosen=[];rejected_sql=[]
    for domain in public.DOMAINS:
        options=[row for row in candidates.items if row['domain']==domain]
        if domain=='data_analysis':
            rejected_sql=[row['group'] for row in options if not sql_candidate_valid(row)]
            options=[row for row in options if sql_candidate_valid(row)]
        if len(options)<3:raise ValueError('Insufficient fresh source groups for '+domain)
        chosen.extend(options[:3])
    for domain,name,instruction,files,certificate in negative_cases():
        chosen.append({'domain':domain,'group':'sla_authored:'+name,'instruction':instruction,
            'files':files,'expected':None,'source_name':'authored','source_id':name,
            'oracle':{'validation_kind':'negative','certificate':certificate}})
    suite.mkdir(parents=True);manifest=[];gold={};metadata={}
    for row in sorted(chosen,key=lambda item:public.digest(f'{SEED}:{item["group"]}')):
        ident='sla_'+public.digest(f'{SEED}:{row["group"]}')[:16]
        folder=suite/'tasks'/ident;workspace=folder/'workspace';workspace.mkdir(parents=True)
        task={'task_type':row['domain'],'reward':reward_for(ident),
              'instruction':row['instruction']+'\nCase identifier: '+ident+'.'}
        write_new(folder/'task.json',task);helper.put_files(workspace,row['files'])
        manifest.append({'id':ident,'domain':row['domain'],'reward':task['reward'],'task_path':'tasks/'+ident,
            'task_sha256':sha(folder/'task.json'),'files_sha256':{name:sha(workspace/name) for name in row['files']}})
        gold[ident]={'expected':row['expected'],'answerable':row['expected'] is not None,
                    'hidden_files':{},'reference_files':{},**row['oracle']}
        metadata[ident]={'group':row['group'],'source_family':row['source_name'],'source_id':row['source_id']}
    assert len(manifest)==TASK_COUNT and sum(row['answerable'] for row in gold.values())==12
    assert not excluded.groups & {row['group'] for row in chosen}
    write_new(suite/'manifest.json',{'label':LABEL,'seed':SEED,'tasks':manifest})
    write_new(suite/'oracle.json',gold);write_new(suite/'research_metadata.json',metadata)
    write_new(suite/'provenance.json',provenance)
    for name,info in provenance.items():
        target=suite/info['license_file'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(public.CACHE/'public100'/(name+'.LICENSE.txt'),target)
    write_new(suite/'deduplication.json',{'previous_suites':excluded.snapshots,
        'verified_hash_methods':dict(excluded.hash_methods),'overlapping_groups':[],
        'rejected_sql_groups':rejected_sql,'preflight_source_rejections':REJECTED_GROUPS,
        'sql_scope':'COUNT excluded before execution because row counts and numerical measures can be confused in source annotations.',
        'scope':'Source-group freshness, not a claim about pretraining or semantic novelty of negative families.'})
    print(json.dumps({'built':TASK_COUNT,'positive':12,'negative':4,'rejected_sql':len(rejected_sql)}))


def image_runtime(image):
    helper.require_image(image)
    paths=previous.runtime_paths()
    script='import json,hashlib,pathlib,sys;p=json.load(sys.stdin);print(json.dumps({k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in p.items()}))'
    result=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only',image,'python','-B','-c',script],
        input=json.dumps(paths),text=True,encoding='utf-8',capture_output=True,check=True,timeout=60)
    hashes=json.loads(result.stdout)
    if len(hashes)<20 or any(hashes[name]!=sha(ROOT/name) for name in paths):
        raise ValueError('Image differs from frozen runtime source/artifact files')
    return hashes


def validate(suite,image):
    """Reference code runs only in a networkless Docker container."""
    image_runtime(image);manifest=read(suite/'manifest.json');oracle=read(suite/'oracle.json')
    helper.verify_visible(suite,manifest);jobs=[]
    for row in manifest['tasks']:
        jobs.append({'id':row['id'],'domain':row['domain'],'gold':oracle[row['id']],
            'files':{name:(suite/row['task_path']/'workspace'/name).read_text(encoding='utf-8') for name in row['files_sha256']}})
    script=r'''
import json,pathlib,tempfile,sys
sys.path.insert(0,'/app')
from coding_tools import run_tests
jobs=json.load(sys.stdin);out=[]
for job in jobs:
 with tempfile.TemporaryDirectory() as tmp:
  work=pathlib.Path(tmp);g=job['gold']
  for name,text in job['files'].items():
   path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
  if job['domain']=='coding':
   assert not run_tests(work)[0], 'stub unexpectedly passes'
   if g['answerable']:
    for name,text in {**g['reference_files'],**g['hidden_files']}.items():
     path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    assert run_tests(work)[0], 'reference failed public/hidden tests'
  out.append({'id':job['id'],'passed':True})
print(json.dumps(out))
'''
    result=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=512m',
        '--cpus','4','--memory','2g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
        image,'python','-B','-c',script],input=json.dumps(jobs),text=True,encoding='utf-8',capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError('Reference validation failed: '+result.stderr[-1500:])
    checked=json.loads(result.stdout.strip().splitlines()[-1])
    for row in manifest['tasks']:
        gold=oracle[row['id']];kind=gold['validation_kind']
        if kind=='math_source':
            for step in gold['arithmetic_steps']:
                lhs,rhs=step.rsplit('=',1)
                assert math.isclose(public.arithmetic(lhs),float(rhs.replace(',','')),rel_tol=1e-8,abs_tol=1e-7)
            assert math.isclose(float(gold['arithmetic_steps'][-1].rsplit('=',1)[1]),float(gold['expected']))
        elif kind=='search_source':
            for answer in gold['source_answers']:
                assert gold['source_context'][answer['answer_start']:answer['answer_start']+len(answer['text'])]==answer['text']
        elif kind=='sql_source':
            body=(suite/row['task_path']/'workspace'/gold['csv_file']).read_text(encoding='utf-8')
            one=public.sql_result(body,gold['column_types'],gold['query'],gold['params'])
            two=public.python_sql_result(body,gold['column_types'],gold['source_sql_ast'],gold['params'])
            assert math.isclose(float(one),float(two),rel_tol=1e-10) and one==gold['expected']
        elif not gold['answerable']:assert gold['certificate']
    write_new(suite/'validation.json',{'image':image,'api_calls':0,'network':'none','total':TASK_COUNT,
        'passed':len(checked),'coding_reference_execution':'isolated_docker','cases':checked})
    print(json.dumps({'offline_validated':len(checked),'api_calls':0}))


def arm_specs(image):
    arms={}
    # Opposite initial policy order across model families reduces a single
    # off-then-on time trend; provider families remain serialized throughout.
    for phase,seconds in (('full',FULL_SECONDS),('short',SHORT_SECONDS)):
        for cycle in (0,1):
            for index,(alias,model) in enumerate(base.MODEL_ALIASES.items()):
                policy=('off','adaptive')[(index+cycle)%2]
                arms[f'{phase}_{policy}_{alias}']={'image':image,'initial_model':model,
                    'sla_policy':policy,'time_limit':seconds,'budget':100000,'phase':phase}
    return arms


def source_hashes():
    paths={Path(__file__),Path(previous.__file__),*previous.DEPENDENCIES,
           ROOT/'experiments/SLA_LATENCY_PRIORS_2026-11-13.json'}
    return {path.relative_to(ROOT).as_posix():sha(path) for path in sorted(paths)}


def prepare(suite,run,image):
    if run.exists():raise FileExistsError('Fresh run path required')
    runtime=image_runtime(image)
    if read(suite/'validation.json')['passed']!=TASK_COUNT:raise ValueError('Validate all oracles first')
    helper.verify_visible(suite,read(suite/'manifest.json'))
    write_new(run/'protocol.json',{'label':LABEL,'created':time.time(),'image':image,'arms':arm_specs(image),
        'source_sha256':source_hashes(),'runtime_sha256':runtime,
        'suite_sha256':{p.relative_to(suite).as_posix():sha(p) for p in suite.rglob('*') if p.is_file()},
        'task_count':TASK_COUNT,'positive_count':12,'negative_count':4,'ceilings':{model:100000 for model in base.MODELS},
        'max_parallel_pipelines':2,'serialize_provider_family':True,'cpus_per_pipeline':4,
        'fixed_configuration':{'confidence_policy':'heuristic','memory_policy':'revise','retry_policy':'adaptive','fixed_model':True},
        'primary_objective':'Maximize verified reward emitted by the agent deadline; then minimize charged cost.',
        'deadline_clock':'Agent run-relative elapsed; starts before entry scan. Docker startup and relay drain reported separately.',
        'short_deadline_seconds':SHORT_SECONDS,'full_deadline_seconds':FULL_SECONDS,
        'external_watchdog_grace_seconds':30,'unknown_usage_policy':'Retain charged reservations; exclude unknown usage from cost forecast metrics.',
        'deadline_score':'Only answers with recorded answer_elapsed_seconds <= time_limit earn on-time points; unknown timestamps reported separately.',
        'limitations':['Small development panel; no fitting or post-outcome threshold selection.',
            'No causal memory comparison; only SLA-policy off/adaptive differs.',
            'Short runs can leave tasks unattempted or interrupt a call. Late provider usage remains charged.',
            'Search scoring is frozen exact accepted span, not semantic judgment.',
            'Observed timings are service-dependent; no universal deadline or throughput guarantee.']})
    print(json.dumps({'prepared':True,'arms':16,'scheduled_episodes':16*TASK_COUNT,'image':image}))


def verify(suite,run,image):
    p=read(run/'protocol.json')
    if p['image']!=image or p['source_sha256']!=source_hashes() or p['runtime_sha256']!=image_runtime(image):
        raise ValueError('Frozen source/runtime/image changed')
    if p['suite_sha256']!={x.relative_to(suite).as_posix():sha(x) for x in suite.rglob('*') if x.is_file()}:
        raise ValueError('Frozen suite changed')
    return p


def docker_command(image,item,tasks,name,port,token):
    return ['docker','run','--rm','--name',name,'--read-only','--tmpfs','/tmp:rw,size=2g','--cpus','4',
        '--memory','16g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges','-e','HOME=/tmp',
        '-e',f'LLM_BASE_URL=http://host.docker.internal:{port}/v1','-e','LLM_API_KEY='+token,
        '--mount',f'type=bind,source={tasks.resolve()},target=/data/tasks',image,'python','/app/run_agent.py',
        '--tasks-dir','/data/tasks','--trace-file','/data/tasks/trace.jsonl','--initial-model',item['initial_model'],
        '--fixed-model','--budget','100000','--confidence-policy','heuristic','--memory-policy','revise',
        '--retry-policy','adaptive','--sla-policy',item['sla_policy'],'--time-limit',str(item['time_limit'])]


def run_experiment(suite,run,image):
    p=verify(suite,run,image)
    if not os.environ.get('LLM_API_KEY'):raise RuntimeError('Inject API key into process environment')
    helper.configure(suite,run);manifest=read(suite/'manifest.json')
    def run_arm(arm,item):
        folder=run/arm
        if (folder/'completed.json').exists():
            if read(folder/'completed.json')['returncode']:raise RuntimeError('Review failed arm before resume')
            return
        if folder.exists():raise RuntimeError('Refusing partial-arm reset')
        tasks=folder/'tasks';tasks.mkdir(parents=True)
        for row in manifest['tasks']:shutil.copytree(suite/row['task_path'],tasks/row['id'])
        relay=base.Relay(p);relay.arm=arm;relay.selected_model=item['initial_model']
        relay.ledger=previous.AlignedLedger(folder/'ledger.json',limit=100000);relay.serve()
        name='nsrc-sla-'+secrets.token_hex(8);started=time.monotonic();started_unix=time.time();returncode=124;watchdog=False
        try:
            with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
                returncode=subprocess.run(docker_command(image,item,tasks,name,relay.server.server_port,relay.token),
                    stdout=out,stderr=err,timeout=item['time_limit']+30).returncode
        except subprocess.TimeoutExpired:watchdog=True
        finally:
            container_seconds=time.monotonic()-started
            container_exit_unix=time.time()
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
            draining=time.monotonic();relay.close();drain_seconds=time.monotonic()-draining
            write_new(folder/'ledger_end.json',relay.ledger.data)
        charged=sum(v['charged_units'] for v in relay.ledger.data['models'].values())/10
        write_new(folder/'completed.json',{'returncode':returncode,'watchdog_triggered':watchdog,
            'container_seconds':container_seconds,'relay_drain_seconds':drain_seconds,
            'container_started_unix':started_unix,'container_exit_unix':container_exit_unix,
            'seconds':time.monotonic()-started,'charged':charged,'image':image,
            'time_limit':item['time_limit'],'sla_policy':item['sla_policy'],'finished':time.time()})
        print(json.dumps({'arm':arm,'returncode':returncode,'charged':charged,'seconds':container_seconds}),flush=True)
        if charged>100000:raise RuntimeError('Token budget exceeded')
        if returncode and not watchdog:raise RuntimeError('Unexpected container failure')
    with base.portalocker.Lock(str(run/'process.lock'),timeout=0):
        with FamilyExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_arm,arm,item) for arm,item in p['arms'].items()]
            for future in as_completed(futures):future.result()


def answer_deadline_status(answer,visits,limit):
    if answer is None:return 'no_answer',None
    times=[t.get('timing',{}).get('answer_elapsed_seconds') for t in visits]
    times=[t for t in times if type(t) in (int,float) and math.isfinite(t) and t>=0]
    if not times:return 'unknown_timestamp',None
    elapsed=max(times)
    return ('on_time' if elapsed<=limit else 'late'),elapsed


def time_prediction(timing):
    # Current runtime serializes LatencyForecast.seconds; tolerate the explicit
    # planned alias without silently discarding all timing diagnostics.
    return timing.get('predicted_seconds',timing.get('seconds'))


def task_deadline_limited(visits,*,answered=False):
    """Distinguish a terminal deadline stop from an earlier deferred visit."""
    if any(call.get('latency_censored',False) for visit in visits for call in visit.get('calls',[])):
        return True
    if not visits or answered:return False
    terminal=visits[-1]
    if terminal.get('timing',{}).get('deadline_limited') is True:return True
    if terminal.get('solver_state')=='time_limit_stopped':return True
    if any(decision.get('reason')=='sla_global_deadline_insufficient'
           for decision in terminal.get('retry_decisions',[])):
        return True
    return any(decision.get('phase')=='continuation' and decision.get('eligible') is False
               and decision.get('reason')=='global_deadline_insufficient'
               for decision in terminal.get('sla_decisions',[]))


def task_usage(physical,visits):
    local=[call for visit in visits for call in visit.get('calls',[])]
    return {'cost':sum(call['cost'] for call in physical),
        'provider_usage_known':(bool(physical) or not local) and all(call['accounting']=='provider_reconciled' for call in physical),
        'local_usage_reconciled':previous.trusted_task_usage(physical,visits)}


def hidden_checks(suite,folder,image,manifest,oracle):
    from coding_tools import is_test
    jobs=[]
    for row in manifest['tasks']:
        gold=oracle[row['id']]
        if not gold.get('hidden_files'):continue
        workspace=folder/row['task_path']/'workspace';original=suite/row['task_path']/'workspace'
        files={path.relative_to(workspace).as_posix():path.read_text(encoding='utf-8')
               for path in workspace.rglob('*.py') if '__pycache__' not in path.parts}
        files.update({path.relative_to(original).as_posix():path.read_text(encoding='utf-8')
                      for path in original.rglob('*.py') if is_test(path.relative_to(original))})
        files.update(gold['hidden_files']);jobs.append({'id':row['id'],'files':files})
    if not jobs:return {}
    check=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=512m',
        '--cpus','4','--memory','2g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
        image,'python','-B','-c',base.CHECK_CODE],input=json.dumps(jobs),text=True,encoding='utf-8',capture_output=True,timeout=180)
    if check.returncode:raise RuntimeError('Hidden checker failed')
    return json.loads(check.stdout.strip().splitlines()[-1])


def report(suite,run,image):
    p=verify(suite,run,image)
    if not all((run/arm/'completed.json').exists() for arm in p['arms']):raise RuntimeError('Finish all16 arms before grading')
    manifest=read(suite/'manifest.json');oracle=read(suite/'oracle.json');metadata=read(suite/'research_metadata.json')
    events=helper.json_lines(run/'calls.jsonl');calls=base.charged_calls(events)
    result={'label':LABEL,'image':image,'protocol_sha256':sha(run/'protocol.json'),'arms':{},'paired':{},
        'fitting_or_selection_on_results':False,'clock_scope':p['deadline_clock']}
    for arm,spec in p['arms'].items():
        folder=run/arm;completion=read(folder/'completed.json');ledger=read(folder/'ledger_end.json')
        traces=helper.json_lines(folder/'tasks/trace.jsonl');physical=[call for call in calls if call['arm']==arm]
        charged=sum(v['charged_units'] for v in ledger['models'].values())/10
        if not math.isclose(charged,sum(call['cost'] for call in physical),abs_tol=.01):raise ValueError('Ledger/proxy mismatch')
        if charged>100000:raise ValueError('Token ceiling breach')
        if any(v['charged_units'] and model!=spec['initial_model'] for model,v in ledger['models'].items()):raise ValueError('Fixed-model scope changed')
        checked=hidden_checks(suite,folder,image,manifest,oracle);rows=[]
        for row in manifest['tasks']:
            ident=row['id'];gold=oracle[ident];visits=[trace for trace in traces if trace.get('task_id')==ident]
            task_calls=[call for call in physical if call['task']==ident];usage=task_usage(task_calls,visits)
            path=folder/row['task_path']/'answer.txt';answer=path.read_text(encoding='utf-8').strip() if path.exists() else None
            status,emitted=answer_deadline_status(answer,visits,spec['time_limit'])
            correct=False
            if answer is not None and gold['answerable']:
                if row['domain']=='coding':correct=answer=='DONE' and checked.get(ident,{}).get('passed',False)
                elif row['domain'] in {'math','data_analysis'}:correct=base.numeric_equal(answer,gold['expected'])
                else:correct=answer in gold.get('accepted_answers',[gold['expected']])
            trace_charge=sum(visit.get('actual_tokens',0) for visit in visits)
            if usage['local_usage_reconciled'] and not math.isclose(trace_charge,usage['cost'],abs_tol=.11):
                raise ValueError('Reconciled all-visit usage mismatch')
            first=visits[0] if visits else {};detail=first.get('prediction_details') or {};timing=first.get('timing') or {}
            complete=bool(visits) and visits[-1].get('execution_complete') is True and all(
                visit.get('execution_complete') is True or visit.get('outcome')=='deferred' for visit in visits)
            elapsed=[visit.get('timing',{}).get('visit_elapsed_seconds') for visit in visits]
            elapsed_known=bool(elapsed) and all(previous.finite_nonnegative(value) for value in elapsed)
            rows.append({'id':ident,'group':metadata[ident]['group'],'domain':row['domain'],'reward':row['reward'],
                'answerable':gold['answerable'],'answered':answer is not None,'correct':correct,
                'on_time_correct':correct and status=='on_time','answer_deadline_status':status,'answer_elapsed_seconds':emitted,
                'calls':len(task_calls),'visits':len(visits),'unattempted':not visits,'policy_complete':complete,
                **usage,'usage_known':usage['provider_usage_known'] and usage['local_usage_reconciled'],
                'runtime_all_visit_charge':trace_charge,'first_provider_cost':task_calls[0]['cost'] if task_calls else None,
                'initial_forecasts':{'frozen_total':detail.get('frozen_total'),'online_total':first.get('predicted_tokens'),
                    'legacy_heuristic':first.get('heuristic_predicted_tokens'),'first_call':detail.get('first_call')},
                'initial_predicted_seconds':time_prediction(timing),
                'initial_conservative_seconds':timing.get('conservative_seconds'),
                'actual_visit_seconds':sum(elapsed) if elapsed_known else None,
                'deadline_limited':task_deadline_limited(visits,answered=answer is not None),
                'final_outcome':visits[-1].get('outcome') if visits else 'unattempted'})
        metric_rows=[row for row in rows if row['calls'] and row['policy_complete'] and row['usage_known'] and not row['deadline_limited']
                     and all(previous.finite_nonnegative(row[key]) for key in
                             ('actual_visit_seconds','initial_predicted_seconds','initial_conservative_seconds'))]
        n=len(metric_rows);decisions=[decision for trace in traces for decision in trace.get('sla_decisions',[])]
        on_time_reward=sum(row['reward'] for row in rows if row['on_time_correct'])
        records=[call for trace in traces for call in trace.get('calls',[])]
        result['arms'][arm]={'model':spec['initial_model'],'sla_policy':spec['sla_policy'],'phase':spec['phase'],
            'time_limit_seconds':spec['time_limit'],'scheduled':TASK_COUNT,'visited':sum(not row['unattempted'] for row in rows),
            'unattempted':sum(row['unattempted'] for row in rows),'provider_attempted_tasks':sum(row['calls']>0 for row in rows),
            'raw_graded_reward':sum(row['reward'] for row in rows if row['correct']),
            'on_time_reward':on_time_reward,'maximum_reward':sum(row['reward'] for row in rows if row['answerable']),
            'on_time_solved':sum(row['on_time_correct'] for row in rows),'raw_solved':sum(row['correct'] for row in rows),
            'negative_answered':sum(not row['answerable'] and row['answered'] for row in rows),
            'positive_wrong_answer':sum(row['answerable'] and row['answered'] and not row['correct'] for row in rows),
            'late_answers':sum(row['answer_deadline_status']=='late' for row in rows),
            'unknown_answer_timestamps':sum(row['answer_deadline_status']=='unknown_timestamp' for row in rows),
            'cost':charged,'calls':len(physical),'usage_unknown_calls':sum(call['accounting']!='provider_reconciled' for call in physical),
            'local_nonreconciled_calls':sum(call.get('status')!='provider_reconciled' for call in records),
            'pending_calls':sum(len(v['pending']) for v in ledger['models'].values()),
            'provider_responses_after_container_exit':sum(call.get('accounting')=='provider_reconciled'
                and call.get('time',0)+call.get('elapsed',0)>completion['container_exit_unix']+.1 for call in physical),
            'container_seconds':completion['container_seconds'],'relay_drain_seconds':completion['relay_drain_seconds'],
            'total_observed_seconds':completion['seconds'],'on_time_reward_per_container_second':
                on_time_reward/completion['container_seconds'] if completion['container_seconds']>0 else None,
            'watchdog_triggered':completion['watchdog_triggered'],'returncode':completion['returncode'],
            'event_counts':dict(Counter(event['event'] for event in events if event.get('arm')==arm)),
            'sla_decision_reasons':dict(Counter(str(decision.get('reason','unknown')) for decision in decisions)),
            'memory_injected_calls':sum(record.get('memory_cards',0)>0 for record in records),
            'time_forecast':{'same_complete_cohort_count':n,'point_mae':
                sum(abs(row['actual_visit_seconds']-row['initial_predicted_seconds']) for row in metric_rows)/n if n else None,
                'empirical_upper_coverage':sum(row['actual_visit_seconds']<=row['initial_conservative_seconds'] for row in metric_rows)/n if n else None,
                'coverage_guarantee':False},
            'cost_forecast':previous.prediction_diagnostics([
                {**row,'policy_complete':row['policy_complete'] and not row['deadline_limited']} for row in rows])
                if spec['phase']=='full' else {'scope':'Not reported for the budget-censored short panel.'},'rows':rows}
    for phase in ('full','short'):
        for alias in base.MODEL_ALIASES:
            off=result['arms'][f'{phase}_off_{alias}'];on=result['arms'][f'{phase}_adaptive_{alias}']
            result['paired'][phase+'_'+alias]={
                'delta_on_time_reward':on['on_time_reward']-off['on_time_reward'],
                'delta_on_time_solved':on['on_time_solved']-off['on_time_solved'],
                'delta_cost':round(on['cost']-off['cost'],1),
                'delta_container_seconds':on['container_seconds']-off['container_seconds'],
                'different_coverage':on['unattempted']!=off['unattempted']}
    result['all_processes_exit_zero']=all(value['returncode']==0 for value in result['arms'].values())
    write_new(run/'evaluation.json',result)
    print(json.dumps({'all_processes_exit_zero':result['all_processes_exit_zero'],'paired':result['paired']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','validate','prepare','run','report'])
    parser.add_argument('--suite',type=Path,default=SUITE);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--image');args=parser.parse_args()
    if args.action=='build':build(args.suite.resolve())
    elif not args.image:parser.error('--image required')
    elif args.action=='validate':validate(args.suite.resolve(),args.image)
    else:{'prepare':prepare,'run':run_experiment,'report':report}[args.action](args.suite.resolve(),args.run.resolve(),args.image)


if __name__=='__main__':main()
