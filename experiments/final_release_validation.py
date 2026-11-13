"""Fresh 20-task release validation, eight primary and two budget-stress arms.

Four domains each contain three unused public groups, one authored structural
case and one certified negative. This small development panel is not a contest
accuracy estimate. No fitting or extra model calls are used by the evaluator.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import as_completed
from decimal import Decimal
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import build_public_panel as public
from experiments import confidence_probe as helper
from experiments import policy_experiment as base
from experiments.public_experiment import FamilyExecutor

SUITE = ROOT/'experiments/suite_final_release_v2'
RUN = ROOT/'experiments/runs/2026-11-13-final-release-v2'
SEED = 20261113917
LABEL = 'Fresh release development validation; unused public groups plus authored cases'
REJECTED_SOURCE_GROUPS = {
    'gsm8k:bda22ba8f66f1dfa7c73d1dae729a8f82c27f9bc75cb99c5aef3ce9f2ef99e12':
        'Preflight ambiguity: every later season has 50% more episodes may refer to first or previous season.',
    'gsm8k:b3fea71586b314ba2ca04aabf915f4545ef755c24ba5d763abae1ae7bdf1a51f':
        'Preflight ambiguity: doubled sales may mean revenue, total quantity or each flower quantity.'}
DEPENDENCIES = (Path(__file__), Path(public.__file__), Path(helper.__file__), Path(base.__file__),
                ROOT/'experiments/public_experiment.py', ROOT/'experiments/run_experiments.py')
sha, read, write_new = helper.sha, helper.read, helper.write_new


def task_reward(ident):
    """Visible reward assignment depends only on the seed and opaque task ID."""
    return 1+int(hashlib.sha256(f'{SEED}:{ident}'.encode()).hexdigest(),16)%5


def normalized_bytes(path):
    return Path(path).read_bytes().replace(b'\r\n', b'\n')


def verify_legacy_hash(path, expected, *, json_task=False):
    raw, lf = Path(path).read_bytes(), normalized_bytes(path)
    candidates = [('raw', raw), ('lf', lf)]
    if json_task:
        obj = read(path)
        candidates += [('canonical_source_json', public.dumps(obj).encode()),
                       ('legacy_source_json', json.dumps(obj, ensure_ascii=False, indent=2).encode())]
    for method, data in candidates:
        if hashlib.sha256(data).hexdigest() == expected:
            return method
    raise ValueError('Prior fixture hash mismatch: '+str(path))


class Exclusions:
    """Validate old manifests, including declared LF normalization, before use."""
    def __init__(self, destination):
        self.groups, self.instructions, self.passages, self.files = set(), set(), set(), set()
        self.snapshots, self.hash_methods = {}, Counter()
        for suite in sorted((ROOT/'experiments').glob('suite*')):
            manifest_path = suite/'manifest.json'
            if suite.resolve() == destination.resolve() or not manifest_path.is_file():
                continue
            manifest = read(manifest_path)
            metadata_path = suite/'research_metadata.json'
            metadata = read(metadata_path) if metadata_path.exists() else {}
            self.snapshots[suite.name] = {'manifest_sha256': sha(manifest_path),
                'metadata_sha256': sha(metadata_path) if metadata_path.exists() else None}
            for row in manifest['tasks']:
                folder = suite/row['task_path']
                task = read(folder/'task.json')
                info = metadata.get(row['id'], {})
                group = row.get('group') or info.get('group') or info.get('task_group')
                if group:
                    self.groups.add(group)
                self.instructions.add(public.norm(task['instruction']))
                files = {p.relative_to(folder/'workspace').as_posix(): p.read_text(encoding='utf-8')
                         for p in (folder/'workspace').rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts}
                if 'files_sha256' in row:
                    if set(files) != set(row['files_sha256']):
                        raise ValueError('Prior file inventory changed: '+str(folder))
                    for name, expected in row['files_sha256'].items():
                        self.hash_methods[verify_legacy_hash(folder/'workspace'/name, expected)] += 1
                if 'task_sha256' in row:
                    self.hash_methods[verify_legacy_hash(folder/'task.json', row['task_sha256'], json_task=True)] += 1
                if 'input_sha256' in row:
                    # Older combined hashes include dictionary insertion order.
                    # Check all bounded file orders; do not waive mismatches.
                    if len(files) > 6:
                        raise ValueError('Unrecognized legacy combined-hash inventory')
                    dumps = lambda value: json.dumps(value, ensure_ascii=False, indent=2)
                    # Windows write_text expanded CSV's existing CRLF into
                    # CRCRLF. Reverse that single write expansion, without
                    # universal-newline conversion that would add blank rows.
                    original_strings = {name:normalized_bytes(folder/'workspace'/name).decode('utf-8')
                                        for name in files}
                    matched_method = None
                    for method, values in [('universal_lf',files),('reverse_windows_write',original_strings)]:
                        if any(public.digest(dumps(task)+dumps(dict(order))) == row['input_sha256']
                               for order in itertools.permutations(values.items())):
                            matched_method = method
                            break
                    if matched_method is None:
                        raise ValueError('Prior combined input hash mismatch: '+str(folder))
                    self.hash_methods['legacy_combined_'+matched_method] += 1
                for name, text in files.items():
                    self.files.add(public.digest(public.norm(text)))
                    if Path(name).suffix in {'.md', '.txt'}:
                        self.passages.add(public.norm(text))


class CandidateBuilder(public.Builder):
    def add(self, domain, group, question, files, expected, **kwargs):
        # Public helper also enumerates its old authored negatives; never reuse them.
        if expected is None or group in self.excluded.groups or group in REJECTED_SOURCE_GROUPS:
            return
        super().add(domain, group, question, files, expected, **kwargs)


def sql_question_unambiguous(item):
    """Conservative mechanical rejection, not a universal semantic oracle."""
    oracle = item['oracle']
    question = public.norm(oracle['source_question'])
    spec = oracle['source_sql_ast']
    markers = {1:r'\b(maximum|highest|largest|greatest|most|longest)\b',
               2:r'\b(minimum|lowest|smallest|least|fewest|shortest)\b',
               3:r'\b(how many|number of|count)\b',
               4:r'\b(sum|total|combined)\b', 5:r'\b(average|mean)\b'}
    aggregate = spec['agg']
    if aggregate and not re.search(markers[aggregate], question):
        return False
    if aggregate != 3 and re.search(r'\b(how many|count|number of)\b', question):
        return False
    if aggregate == 3 and re.search(r'\b(sum|average|mean|highest|lowest)\b', question):
        return False
    if aggregate == 0 and any(re.search(pattern, question) for pattern in markers.values()):
        return False
    for _, operator, value in spec['conds']:
        literal = public.norm(value)
        if literal.endswith('.0'):
            literal = literal[:-2]
        if literal not in question:
            return False
        if operator == 1 and not re.search(r'\b(over|more than|greater than|after|above)\b', question):
            return False
        if operator == 2 and not re.search(r'\b(under|less than|before|below|fewer than)\b', question):
            return False
    return True


def authored_cases():
    integrity = helper.INTEGRITY
    sources = {'mylib.py': 'from engine.runs import longest_run\n', 'engine/__init__.py': '',
               'engine/runs.py': 'def longest_run(values):\n    return (0, len(values))\n',
               'tests/test_runs.py': 'from mylib import longest_run\n\ndef test_examples():\n    assert longest_run([]) == (0, 0)\n    assert longest_run([4, 2, 2, 5, 1]) == (1, 3)\n'}
    reference = 'def longest_run(values):\n    best = (0, 0)\n    start = 0\n    for i, value in enumerate(values):\n        if i and value < values[i-1]:\n            start = i\n        length = i-start+1\n        if length > best[1]:\n            best = (start, length)\n    return best\n'
    hidden = {'tests/test_hidden_runs.py': 'from mylib import longest_run\n\ndef test_hidden():\n    for a, expected in [([3,2,1],(0,1)), ([1,2,0,1],(0,2)), ([7],(0,1)), ([1,1,1],(0,3)), ([9,1,2,3,0],(1,3))]:\n        before = list(a)\n        assert longest_run(a) == expected\n        assert a == before\n'}
    sql_files = {'customers.csv':'customer_id,region\n1,West\n2,East\n3,West\n',
                 'orders.csv':'order_id,customer_id,net amount\n11,1,40\n12,1,20\n13,2,90\n14,3,50\n',
                 'refunds.csv':'order_id,refund\n11,3\n11,2\n14,10\n'}
    query = ('WITH r AS (SELECT order_id,SUM(refund) AS refund FROM refunds GROUP BY order_id) '
             'SELECT 1.0*SUM(o."net amount"-COALESCE(r.refund,0))/COUNT(*) FROM orders o '
             "JOIN customers c ON c.customer_id=o.customer_id LEFT JOIN r ON r.order_id=o.order_id WHERE c.region='West'")
    return [
        ('math','fraction_then_floor','A workshop has 37 metres of ribbon. It spends 3/8 of the original length on decorations, then exactly 5/4 metres on labels. Each finished parcel requires 7/6 metres. How many complete parcels can be wrapped from the remaining ribbon? Keep any leftover; do not round until taking the final whole-parcel count.',{},'18',{'validation_kind':'authored_math','expression':'(37*(1-3/8)-5/4)//(7/6)','manual':'(185/8-10/8)/(7/6)=525/28=18.75, floor=18'}),
        ('math','unrecorded_rate','A van travels for exactly 3 hours at a constant positive speed that was not recorded, then travels 24 kilometres. What total distance does it cover? No arrival time, starting distance or speed constraint supplies the missing speed. Return UNSOLVABLE if there is no unique answer.',{},None,{'validation_kind':'negative','certificate':'Speed 30 gives114km; speed50 gives174km; both fit every fact.'}),
        ('search','latest_revision','Using only the supplied bulletin, what is the explicitly current contact code for observatory IVORY-19? Use the largest published revision number, not archive ordering. Copy only the contact code.',{'kb/bulletin.md':'Observatory IVORY-19 official bulletin. Revision 2, superseded: contact code PINE-70. Revision 5, current: contact code CEDAR-42. Revision 3, superseded: contact code ELM-16.\n'},'CEDAR-42',{'validation_kind':'authored_search'}),
        ('search','unassigned_signatory','What is the signatory assigned to docket MINT-24? The complete supplied register has no default signatory and assignments are independent of all other records. Return UNSOLVABLE if the supplied register does not identify one.',{'register.json':'{"docket":"MINT-24","assignment_recorded":false,"other_assignments":{"MINT-25":"Mira","MINT-26":"Niko"}}\n'},None,{'validation_kind':'negative','certificate':'MINT-24 may be assigned Mira orNiko without changing any observed field; no assignment is recorded.'}),
        ('data_analysis','refund_join_mean','For orders belonging to West-region customers, compute the average net amount per order after subtracting all recorded refunds for that order. Orders without refunds keep their full net amount. Join the complete tables by their explicit IDs, count each eligible order once, and return the fractional numeric result.',sql_files,str(95/3),{'validation_kind':'authored_sql','reference_sql':query,'manual':'((40-3-2)+20+(50-10))/3=95/3'}),
        ('data_analysis','unrecorded_fixed_cost','Return exact profit for category lab, defined as its total revenue minus a separate fixed expense. This complete revenue table contains no expense or profit-margin information; the expense is nonnegative and otherwise unconstrained. Return UNSOLVABLE if exact profit is undetermined.',{'revenue.csv':'category,revenue\nlab,38\nlab,27\nshop,99\n'},None,{'validation_kind':'negative','certificate':'Lab revenue65; expense10 gives55 andexpense30 gives35, bothconsistent.'}),
        ('coding','nondecreasing_multifile','Repair longest_run(values) exposed by mylib.py. Return (start_index, length) of the longest contiguous nondecreasing run. Ties choose the earliest start. Empty input returns (0,0); do not mutate the input list. Implement the shared helper in engine/runs.py. '+integrity,sources,'DONE',{'validation_kind':'coding','reference_files':{'engine/runs.py':reference},'hidden_files':hidden}),
        ('coding','integer_negative_square','Implement make_value() returning an exact built-in int r with r*r == -1, under ordinary Python arithmetic. A single returned object must satisfy the public tests. All functions must be stateless and deterministic. '+integrity,{'mylib.py':'def make_value():\n    return 0\n','test_mylib.py':'from mylib import make_value\n\ndef test_contract():\n    r = make_value()\n    assert type(r) is int\n    assert r*r == -1\n'},None,{'validation_kind':'negative','certificate':'Every Python integer square is nonnegative; exact type and integrity contract exclude overloaded arithmetic.'}),
    ]


def build(suite):
    if suite.exists():
        raise FileExistsError('Fresh suite path required')
    provenance = public.sources(False)
    excluded = Exclusions(suite)
    candidates = CandidateBuilder(excluded)
    # Globals affect this imported builder only in this process, never source files.
    old_seed = public.SEED
    try:
        public.SEED = SEED
        public.math_cases(candidates); public.humaneval_cases(candidates)
        public.search_cases(candidates); public.data_cases(candidates)
    finally:
        public.SEED = old_seed
    chosen, rejected_sql = [], []
    for domain in public.DOMAINS:
        options = [item for item in candidates.items if item['domain'] == domain]
        if domain == 'data_analysis':
            rejected_sql = [item['group'] for item in options if not sql_question_unambiguous(item)]
            options = [item for item in options if sql_question_unambiguous(item)]
        if len(options) < 3:
            raise ValueError('Insufficient unambiguous fresh public cases for '+domain)
        chosen.extend(options[:3])
    for domain, name, question, files, expected, oracle in authored_cases():
        chosen.append({'domain':domain,'group':'release_authored:'+name,'instruction':question,
                       'files':files,'expected':expected,'oracle':oracle,'source_name':'authored','source_id':name})
    suite.mkdir(parents=True)
    manifest, labels, metadata = [], {}, {}
    for item in sorted(chosen, key=lambda row:public.digest(str(SEED)+row['group'])):
        ident = 'release_'+public.digest(str(SEED)+item['group'])[:16]
        folder = suite/'tasks'/ident; work = folder/'workspace'; work.mkdir(parents=True)
        reward=task_reward(ident)
        task = {'task_type':item['domain'],'reward':reward,'instruction':item['instruction']+'\nCase identifier: '+ident+'.'}
        write_new(folder/'task.json',task); helper.put_files(work,item['files'])
        manifest.append({'id':ident,'domain':item['domain'],'reward':reward,'task_path':'tasks/'+ident,
                         'task_sha256':sha(folder/'task.json'),'files_sha256':{name:sha(work/name) for name in item['files']}})
        labels[ident] = {'expected':item['expected'],'answerable':item['expected'] is not None,
                         'hidden_files':{},'reference_files':{},**item['oracle']}
        metadata[ident] = {'group':item['group'],'source_family':item['source_name'],'source_id':item['source_id']}
    assert len(manifest)==20 and sum(g['answerable'] for g in labels.values())==16
    assert not excluded.groups & {item['group'] for item in chosen}
    write_new(suite/'manifest.json',{'label':LABEL,'seed':SEED,'tasks':manifest})
    write_new(suite/'oracle.json',labels); write_new(suite/'research_metadata.json',metadata)
    write_new(suite/'provenance.json',provenance)
    for name, info in provenance.items():
        path=suite/info['license_file'];path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(public.CACHE/'public100'/(name+'.LICENSE.txt'),path)
    write_new(suite/'deduplication.json',{'previous_suites':excluded.snapshots,
        'verified_hash_methods':dict(excluded.hash_methods),'prior_groups':len(excluded.groups),
        'overlapping_groups':[],'rejected_ambiguous_sql_groups':rejected_sql,
        'preflight_source_rejections':REJECTED_SOURCE_GROUPS,
        'limitations':'Source groups and normalized instructions/passages checked, not exhaustive semantic equivalence or pretraining contamination.'})
    print(json.dumps({'built':20,'positive':16,'negative':4,'rejected_sql':len(rejected_sql)}))


def revise_rewards(source,suite):
    """Preserve the unpaid first candidate while changing only visible rewards."""
    if suite.exists():raise FileExistsError('Fresh reward-version suite required')
    previous_run=ROOT/'experiments/runs/2026-11-13-final-release'
    if (previous_run/'calls.jsonl').exists() or any(previous_run.glob('*/completed.json')):
        raise ValueError('This revision is permitted only before any paid run')
    manifest=read(source/'manifest.json')
    helper.verify_visible(source,manifest)
    parent_protocol=previous_run/'protocol.json'
    original_task_hashes={row['id']:row['task_sha256'] for row in manifest['tasks']}
    for path in source.rglob('*'):
        if path.is_file() and path.name not in {'manifest.json','task.json','validation.json'}:
            target=suite/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    changes=[]
    for row in manifest['tasks']:
        task=read(source/row['task_path']/'task.json');old_reward=task['reward']
        task['reward']=row['reward']=task_reward(row['id'])
        target=suite/row['task_path']/'task.json';write_new(target,task)
        row['task_sha256']=sha(target)
        changes.append({'id':row['id'],'old_reward':old_reward,'new_reward':row['reward']})
    write_new(suite/'manifest.json',manifest)
    write_new(suite/'unpaid_revision.json',{'parent_suite':str(source.relative_to(ROOT)),
        'parent_manifest_sha256':sha(source/'manifest.json'),
        'parent_protocol_sha256':sha(parent_protocol),'parent_api_calls':0,
        'unchanged_oracle_sha256':sha(source/'oracle.json'),
        'parent_task_sha256':original_task_hashes,'changes':changes,
        'formula':"1 + int(SHA256(UTF8(str(seed) + ':' + task_id)),16) % 5",
        'selection_uses_labels':False,'seed':SEED})
    print(json.dumps({'reward_version':2,'tasks':len(changes),'label_based_assignment':False}))


def runtime_paths():
    result = {}
    for line in (ROOT/'Dockerfile').read_text().splitlines():
        if not line.startswith('COPY '):
            continue
        parts=shlex.split(line)[1:]; destination=parts[-1]
        for name in parts[:-1]:
            result[name] = destination+Path(name).name if destination.endswith('/') else destination
    return result


def image_runtime(image):
    helper.require_image(image)
    paths=runtime_paths()
    code='import json,hashlib,pathlib,sys;p=json.load(sys.stdin);print(json.dumps({k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in p.items()}))'
    result=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only',image,'python','-B','-c',code],
        input=json.dumps(paths),text=True,encoding='utf-8',capture_output=True,check=True,timeout=60)
    hashes=json.loads(result.stdout)
    if len(hashes)!=20 or any(hashes[name]!=sha(ROOT/name) for name in paths):
        raise ValueError('Image differs from the20 frozen runtime source/artifact files')
    return hashes


def validate(suite, image):
    """All reference execution, including source code, is isolated in Docker."""
    image_runtime(image)
    manifest, oracle = read(suite/'manifest.json'), read(suite/'oracle.json')
    helper.verify_visible(suite,manifest)
    jobs=[]
    for row in manifest['tasks']:
        gold=oracle[row['id']]
        files={name:(suite/row['task_path']/'workspace'/name).read_text(encoding='utf-8') for name in row['files_sha256']}
        jobs.append({'id':row['id'],'domain':row['domain'],'files':files,'gold':gold})
    code=r'''
import json,math,pathlib,tempfile,sys
sys.path.insert(0,'/app')
from coding_tools import run_tests
from analytical_tools import CSVDatabase
jobs=json.load(sys.stdin);out=[]
for job in jobs:
 g=job['gold'];passed=True
 with tempfile.TemporaryDirectory() as tmp:
  work=pathlib.Path(tmp)
  for name,text in job['files'].items():
   p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
  if job['domain']=='coding':
   assert not run_tests(work)[0], 'stub unexpectedly passes'
   if g['answerable']:
    for name,text in {**g['reference_files'],**g['hidden_files']}.items():
     p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
    assert run_tests(work)[0], 'reference public/hidden failure'
  if g.get('reference_sql'):
   db=CSVDatabase(work)
   try:answer,error=db.execute(g['reference_sql'])
   finally:db.close()
   assert error is None and math.isclose(float(answer),float(g['expected']),rel_tol=1e-10)
 out.append({'id':job['id'],'passed':True})
print(json.dumps(out))
'''
    result=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=512m',
        '--cpus','4','--memory','2g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
        image,'python','-B','-c',code],input=json.dumps(jobs),text=True,encoding='utf-8',capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError('Isolated reference validation failed: '+result.stderr[-1800:])
    results=json.loads(result.stdout.strip().splitlines()[-1])
    for row in manifest['tasks']:
        gold=oracle[row['id']];kind=gold['validation_kind']
        if kind=='math_source':
            for step in gold['arithmetic_steps']:
                lhs,rhs=step.rsplit('=',1)
                assert math.isclose(public.arithmetic(lhs),float(rhs.replace(',','')),rel_tol=1e-8,abs_tol=1e-7)
        elif kind=='authored_math':
            assert public.scalar(public.arithmetic(gold['expression']))==gold['expected']
        elif kind=='search_source':
            for answer in gold['source_answers']:
                assert gold['source_context'][answer['answer_start']:answer['answer_start']+len(answer['text'])]==answer['text']
        elif kind=='sql_source':
            body=(suite/row['task_path']/'workspace'/gold['csv_file']).read_text(encoding='utf-8')
            a=public.sql_result(body,gold['column_types'],gold['query'],gold['params'])
            b=public.python_sql_result(body,gold['column_types'],gold['source_sql_ast'],gold['params'])
            assert math.isclose(float(a),float(b),rel_tol=1e-10) and a==gold['expected']
        elif not gold['answerable']:
            assert gold.get('certificate')
    write_new(suite/'validation.json',{'image':image,'api_calls':0,'network':'none','total':20,
        'passed':len(results),'public_source_coding_executed_only_in_docker':True,'cases':results})
    print('20 offline oracle checks passed; public references executed only in networkless Docker.')


def source_hashes():
    return {path.relative_to(ROOT).as_posix():sha(path) for path in DEPENDENCIES}


def arm_specs(image):
    arms={}
    for memory in ('off','revise'):
        for alias,model in base.MODEL_ALIASES.items():
            arms['primary_'+memory+'_'+alias]={'initial_model':model,'memory_policy':memory,'image':image,
                'budget':'100000','budget_source':'cli','scope':'primary'}
    for memory in ('off','revise'):
        arms['stress_'+memory+'_large_reason']={'initial_model':base.MODEL_ALIASES['large_reason'],
            'memory_policy':memory,'image':image,'budget':'5000.5','budget_source':'env','scope':'secondary_budget_stress'}
    return arms


FORECAST_DIAGNOSTICS = {
    'scope':'Primary arms only; descriptive, no fitting or selection.',
    'eligible':'Attempted tasks with at least one provider request, all usage reconciled, '
               'and completed terminal policy; exclude local-only, unattempted and censored tasks.',
    'cohort':'All three finite nonnegative initial predictions must exist; identical rows for all comparisons.',
    'total_actual':'Sum charged provider request costs over every visit of the same task.',
    'predictions':{
        'frozen_total':'First visit prediction_details.frozen_total, including estimated skill/memory prompt overhead.',
        'online_total':'First visit predicted_tokens, including online scale at initial execution.',
        'legacy_heuristic':'First visit heuristic_predicted_tokens from a new TaskEstimator.'},
    'metrics':'Row MAE and equal-group macro MAE; cost is output + input/10.',
    'first_call':'Same eligible cohort with finite first_call: initial first_call versus first provider request only.',
}


def prepare(suite,run,image):
    runtime=image_runtime(image)
    if run.exists():raise FileExistsError('Fresh run directory required')
    if read(suite/'validation.json')['passed']!=20:raise ValueError('Validate all20 oracles first')
    helper.verify_visible(suite,read(suite/'manifest.json'))
    files={p.relative_to(suite).as_posix():sha(p) for p in suite.rglob('*') if p.is_file()}
    write_new(run/'protocol.json',{'label':LABEL,'created':time.time(),'image':image,'arms':arm_specs(image),
        'source_sha256':source_hashes(),'runtime_sha256':runtime,'suite_sha256':files,
        'ceilings':{model:100000 for model in base.MODELS},'task_count':20,
        'answerable_count':16,'negative_count':4,'forecast_diagnostics':FORECAST_DIAGNOSTICS,
        'reward_assignment':"1 + int(SHA256(UTF8(str(seed) + ':' + task_id)),16) % 5; independent of labels/domain",
        'proxy_reservation':'Known two plain system/user messages use summed UTF-8 content bytes '
                            'plus shared framing allowance, matching runtime; other shapes retain serialized reservation.',
        'max_parallel_pipelines':2,'cpus_per_pipeline':4,'confidence_policy':'heuristic',
        'primary_objective':['maximize_correct_points','then_minimize_total_cost'],
        'limitations':['Small development validation; source group freshness is not pretraining independence.',
            'Memory may be sparse: log creation/retrieval/revision and do not claim causal benefit without activity.',
            'Stress arms have separate5000.5 budgets and may leave tasks unattempted; report censored coverage.',
            'Public tests/tool execution do not certify hidden correctness. Gold is outside solver mounts.',
            'No predictor fitting, policy tuning or model selection uses these outcomes.']})
    print(json.dumps({'prepared':True,'arms':10,'scheduled_episodes':200,'image':image}))


def verify(suite,run,image):
    p=read(run/'protocol.json')
    if p['image']!=image or source_hashes()!=p['source_sha256'] or image_runtime(image)!=p['runtime_sha256']:
        raise ValueError('Frozen image/runtime/runner changed')
    actual={path.relative_to(suite).as_posix():sha(path) for path in suite.rglob('*') if path.is_file()}
    if actual!=p['suite_sha256']:raise ValueError('Frozen suite changed')
    return p


def docker_command(image,item,tasks,name,port,token):
    command=['docker','run','--rm','--name',name,'--read-only','--tmpfs','/tmp:rw,size=2g','--cpus','4',
        '--memory','16g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges',
        '-e','HOME=/tmp','-e',f'LLM_BASE_URL=http://host.docker.internal:{port}/v1','-e','LLM_API_KEY='+token]
    if item['budget_source']=='env':command+=['-e','BUDGET='+item['budget']]
    command+=['--mount',f'type=bind,source={tasks.resolve()},target=/data/tasks',image,'python','/app/run_agent.py',
        '--tasks-dir','/data/tasks','--trace-file','/data/tasks/trace.jsonl','--initial-model',item['initial_model'],
        '--fixed-model','--retry-policy','adaptive','--confidence-policy','heuristic','--memory-policy',item['memory_policy']]
    if item['budget_source']=='cli':command+=['--budget',item['budget']]
    return command


def trusted_task_usage(physical, visits):
    local=[record for visit in visits for record in visit.get('calls',[])]
    return (len(local)==len(physical)
            and all(call['accounting']=='provider_reconciled' for call in physical)
            and all(record.get('status')=='provider_reconciled' for record in local))


def finite_nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def prediction_diagnostics(rows):
    """Frozen descriptive formulas; labels/correctness never select the cohort."""
    names=('frozen_total','online_total','legacy_heuristic')
    selected=[];excluded=Counter()
    for row in rows:
        if row['unattempted']: reason='unattempted'
        elif not row['calls']: reason='no_provider_request'
        elif not row['usage_known']: reason='unknown_usage'
        elif not row['policy_complete']: reason='censored_policy'
        elif not all(finite_nonnegative(row['initial_forecasts'].get(name)) for name in names):
            reason='missing_or_invalid_prediction'
        else: selected.append(row); continue
        excluded[reason]+=1
    def metrics(cohort,prediction,actual):
        errors=[abs(prediction(row)-actual(row)) for row in cohort]
        groups={row['group'] for row in cohort}
        macro=[sum(abs(prediction(row)-actual(row)) for row in cohort if row['group']==group)
               /sum(row['group']==group for row in cohort) for group in groups]
        return {'count':len(cohort),'group_count':len(groups),
                'mae':sum(errors)/len(errors) if errors else None,
                'group_macro_mae':sum(macro)/len(macro) if macro else None}
    first=[row for row in selected if finite_nonnegative(row['initial_forecasts'].get('first_call'))
           and finite_nonnegative(row.get('first_provider_cost'))]
    return {'scope':FORECAST_DIAGNOSTICS,'eligible_ids':[row['id'] for row in selected],
        'eligible_count':len(selected),'group_count':len({row['group'] for row in selected}),
        'excluded_counts':dict(excluded),'domain_counts':dict(Counter(row['domain'] for row in selected)),
        'total':{name:metrics(selected,lambda row,n=name:row['initial_forecasts'][n],lambda row:row['cost']) for name in names},
        'first_call':metrics(first,lambda row:row['initial_forecasts']['first_call'],lambda row:row['first_provider_cost'])}


def score_summary(rows):
    reward=sum(row['reward'] for row in rows if row['answerable'] and row['correct'])
    maximum=sum(row['reward'] for row in rows if row['answerable'])
    return {'solved_count':sum(row['answerable'] and row['correct'] for row in rows),
            'answerable_count':sum(row['answerable'] for row in rows),
            'reward':reward,'max_reward':maximum,'points':reward,'max_points':maximum}


class AlignedLedger(base.Ledger):
    """Match the runtime's reservation for its two plain chat messages.

    The shared128-token framing allowance already covers roles. Counting JSON
    syntax again can reject a cap just clipped to the same remaining budget.
    Unexpected message shapes retain the older, larger serialized reservation.
    Reconciliation and the actual per-pipeline spending ceiling are unchanged.
    """
    def reserve(self, model, cap, text, ceiling=100000):
        try:
            messages=json.loads(text)
            plain=(isinstance(messages,list) and len(messages)==2
                   and all(type(message) is dict and set(message)=={'role','content'}
                           and type(message['content']) is str for message in messages)
                   and [message['role'] for message in messages]==['system','user'])
        except (ValueError,TypeError,KeyError):
            plain=False
        content=''.join(message['content'] for message in messages) if plain else text
        return super().reserve(model,cap,content,ceiling)


def run_experiment(suite,run,image):
    p=verify(suite,run,image)
    if not os.environ.get('LLM_API_KEY'):raise RuntimeError('Inject API key into process environment')
    helper.configure(suite,run)
    manifest=read(suite/'manifest.json')
    def run_arm(arm,item):
        folder=run/arm
        if (folder/'completed.json').exists():
            if read(folder/'completed.json')['returncode']:raise RuntimeError('Review failed arm before resume')
            return
        if folder.exists():raise RuntimeError('Refusing partial-arm budget/answer reset')
        tasks=folder/'tasks';tasks.mkdir(parents=True)
        for row in manifest['tasks']:shutil.copytree(suite/row['task_path'],tasks/row['id'])
        scope=dict(p);scope['ceilings']={model:float(item['budget']) for model in base.MODELS}
        relay=base.Relay(scope);relay.arm=arm;relay.selected_model=item['initial_model']
        relay.ledger=AlignedLedger(folder/'ledger.json',limit=float(item['budget']));relay.serve()
        name='nsrc-final-'+secrets.token_hex(8)
        command=docker_command(image,item,tasks,name,relay.server.server_port,relay.token)
        started,returncode=time.monotonic(),124
        try:
            with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
                returncode=subprocess.run(command,stdout=out,stderr=err,timeout=3500).returncode
        except subprocess.TimeoutExpired:pass
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
            relay.close();write_new(folder/'ledger_end.json',relay.ledger.data)
        charged=sum(value['charged_units'] for value in relay.ledger.data['models'].values())/10
        source='CLI --budget' if item['budget_source']=='cli' else 'BUDGET'
        budget_line=f"token budget: {item['budget']} (source: {source})"
        budget_startup_verified=budget_line in (folder/'stdout.txt').read_text(encoding='utf-8',errors='replace')
        write_new(folder/'completed.json',{'returncode':returncode,'charged':charged,'seconds':time.monotonic()-started,
            'image':image,'budget':item['budget'],'budget_source':item['budget_source'],
            'budget_startup_verified':budget_startup_verified})
        print(json.dumps({'arm':arm,'returncode':returncode,'charged':charged}),flush=True)
        if returncode or not budget_startup_verified or Decimal(str(charged))>Decimal(item['budget']):raise RuntimeError('Failed/breached/misconfigured arm: '+arm)
    with base.portalocker.Lock(str(run/'process.lock'),timeout=0):
        with FamilyExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_arm,arm,item) for arm,item in p['arms'].items()]
            for future in as_completed(futures):future.result()


def report(suite,run,image):
    p=verify(suite,run,image)
    if not all((run/arm/'completed.json').exists() for arm in p['arms']):raise RuntimeError('Finish all10 arms before grading')
    manifest,oracle=read(suite/'manifest.json'),read(suite/'oracle.json')
    metadata=read(suite/'research_metadata.json')
    events=helper.json_lines(run/'calls.jsonl');calls=base.charged_calls(events)
    result={'label':LABEL,'protocol_sha256':sha(run/'protocol.json'),'image':image,'arms':{},'paired_primary':{},
        'secondary_scope':'Budget-pressure censoring, not a20-task completeness claim','prediction_generalization_claim':False}
    for arm,item in p['arms'].items():
        folder=run/arm;traces=helper.json_lines(folder/'tasks/trace.jsonl')
        arm_calls=[call for call in calls if call['arm']==arm]
        ledger=read(folder/'ledger_end.json');charged=sum(v['charged_units'] for v in ledger['models'].values())/10
        if not math.isclose(charged,sum(call['cost'] for call in arm_calls),abs_tol=.01):raise ValueError('Ledger/proxy mismatch')
        if Decimal(str(charged))>Decimal(item['budget']):raise ValueError('Budget breach')
        if any(v['charged_units'] and model!=item['initial_model'] for model,v in ledger['models'].items()):raise ValueError('Model switch')
        jobs=[]
        for row in manifest['tasks']:
            gold=oracle[row['id']]
            if not gold.get('hidden_files'):continue
            work=folder/row['task_path']/'workspace';original=suite/row['task_path']/'workspace'
            files={x.relative_to(work).as_posix():x.read_text(encoding='utf-8') for x in work.rglob('*.py') if '__pycache__' not in x.parts}
            from coding_tools import is_test
            files.update({x.relative_to(original).as_posix():x.read_text(encoding='utf-8') for x in original.rglob('*.py') if is_test(x.relative_to(original))})
            files.update(gold['hidden_files']);jobs.append({'id':row['id'],'files':files})
        checked={}
        if jobs:
            check=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=512m',
                '--cpus','4','--memory','2g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
                image,'python','-B','-c',base.CHECK_CODE],input=json.dumps(jobs),text=True,encoding='utf-8',capture_output=True,timeout=180)
            if check.returncode:raise RuntimeError('Hidden checker failed')
            checked=json.loads(check.stdout.strip().splitlines()[-1])
        rows=[]
        for row in manifest['tasks']:
            ident=row['id'];gold=oracle[ident];visits=[t for t in traces if t.get('task_id')==ident]
            physical=[c for c in arm_calls if c['task']==ident];cost=sum(c['cost'] for c in physical)
            path=folder/row['task_path']/'answer.txt';answer=path.read_text(encoding='utf-8').strip() if path.exists() else None
            known=trusted_task_usage(physical,visits)
            tracecost=sum(t.get('actual_tokens',0) for t in visits)
            if known and not math.isclose(cost,tracecost,abs_tol=.11):raise ValueError('All-visit trace usage mismatch')
            initial=visits[0] if visits else {}
            detail=initial.get('prediction_details') or {}
            complete=bool(visits) and visits[-1].get('execution_complete') is True and all(
                visit.get('execution_complete') is True or visit.get('outcome')=='deferred' for visit in visits)
            correct=False
            if gold['answerable'] and answer is not None:
                if row['domain']=='coding':correct=answer=='DONE' and checked.get(ident,{}).get('passed',False)
                elif row['domain'] in {'math','data_analysis'}:correct=base.numeric_equal(answer,gold['expected'])
                else:correct=answer in gold.get('accepted_answers',[gold['expected']])
            rows.append({'id':ident,'domain':row['domain'],'group':metadata[ident]['group'],'reward':row['reward'],
                'answerable':gold['answerable'],'correct':correct,
                'answered':answer is not None,'cost':round(cost,1),'calls':len(physical),'visits':len(visits),
                'runtime_all_visit_charge':round(tracecost,1),
                'usage_known':known,'unattempted':not visits,'initial_prediction':visits[0].get('predicted_tokens') if visits else None,
                'policy_complete':complete,'first_provider_cost':physical[0]['cost'] if physical else None,
                'initial_forecasts':{'frozen_total':detail.get('frozen_total'),
                    'online_total':initial.get('predicted_tokens'),
                    'legacy_heuristic':initial.get('heuristic_predicted_tokens'),
                    'first_call':detail.get('first_call')},
                'final_state':visits[-1].get('solver_state',visits[-1].get('outcome')) if visits else 'unattempted',
                'deferred_visits':sum(t.get('outcome')=='deferred' for t in visits)})
        records=[record for trace in traces for record in trace.get('calls',[])]
        memories=[trace.get('memory_state',{}) for trace in traces if trace.get('memory_state')]
        versions={}
        for memory in memories:
            for ident,history in memory.get('history_ids',{}).items():versions[ident]=max(versions.get(ident,0),max(history,default=0))
        summary={'scope':item['scope'],'model':item['initial_model'],'memory_policy':item['memory_policy'],
            'budget':item['budget'],'budget_source':item['budget_source'],'scheduled':20,
            'budget_startup_verified':read(folder/'completed.json').get('budget_startup_verified',False),
            'visited':sum(not row['unattempted'] for row in rows),'visits':len(traces),'unattempted':sum(row['unattempted'] for row in rows),
            **score_summary(rows),
            'positive_wrong_answer':sum(row['answerable'] and row['answered'] and not row['correct'] for row in rows),
            'positive_no_answer':sum(row['answerable'] and not row['answered'] for row in rows),
            'negative_answered':sum(not row['answerable'] and row['answered'] for row in rows),
            'negative_skipped_after_visit':sum(not row['answerable'] and not row['answered'] and not row['unattempted'] for row in rows),
            'cost':charged,'negative_cost':round(sum(row['cost'] for row in rows if not row['answerable']),1),
            'calls':len(arm_calls),'usage_unknown_calls':sum(c['accounting']!='provider_reconciled' for c in arm_calls),
            'pending_calls':sum(len(v['pending']) for v in ledger['models'].values()),
            'event_counts':dict(Counter(e['event'] for e in events if e.get('arm')==arm)),
            'memory':{'cards_seen':len(versions),'version_updates_seen':sum(max(0,v-1) for v in versions.values()),
                'injected_physical_calls':sum(r.get('memory_cards',0)>0 for r in records),
                'injected_utf8_bytes':sum(r.get('memory_injected_utf8_bytes',0) for r in records),
                'errors':sum(t.get('memory_errors',0) for t in traces)},'rows':rows}
        summary['primary_all20_visited']=summary['visited']==20 if item['scope']=='primary' else None
        if item['scope']=='primary':summary['forecast_diagnostics']=prediction_diagnostics(rows)
        result['arms'][arm]=summary
    for alias in base.MODEL_ALIASES:
        off,on=result['arms']['primary_off_'+alias],result['arms']['primary_revise_'+alias]
        result['paired_primary'][alias]={'delta_points_revise_minus_off':on['points']-off['points'],
            'delta_solved_revise_minus_off':on['solved_count']-off['solved_count'],
            'delta_cost_revise_minus_off':round(on['cost']-off['cost'],1),
            'memory_retrieval_present':on['memory']['injected_physical_calls']>0}
    result['all10_processes_success']=all(read(run/arm/'completed.json')['returncode']==0 for arm in p['arms'])
    write_new(run/'evaluation.json',result)
    print(json.dumps({'all_processes_success':result['all10_processes_success'],'primary':result['paired_primary']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','revise-rewards','validate','prepare','run','report'])
    parser.add_argument('--suite',type=Path,default=SUITE);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--image')
    parser.add_argument('--source-suite',type=Path,default=ROOT/'experiments/suite_final_release')
    args=parser.parse_args()
    if args.action=='build':build(args.suite.resolve())
    elif args.action=='revise-rewards':revise_rewards(args.source_suite.resolve(),args.suite.resolve())
    elif not args.image:parser.error('--image is required')
    elif args.action=='validate':validate(args.suite.resolve(),args.image)
    else:{'prepare':prepare,'run':run_experiment,'report':report}[args.action](args.suite.resolve(),args.run.resolve(),args.image)


if __name__=='__main__':main()
