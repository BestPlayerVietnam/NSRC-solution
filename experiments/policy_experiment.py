"""Isolated 100k-token, single-model pipelines through real openJiuwen.

The local relay reserves EACH physical HTTP request before forwarding. Solver
containers receive only a random relay token; the NeuralDeep key stays on host.
Gold truth is opened only by the post-run evaluator.
"""
from __future__ import annotations
import argparse,copy,hashlib,http.server,json,os,secrets,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.run_experiments import Ledger,append_json,write_json,MODELS,numeric_equal,norm_answer
import httpx,portalocker
from concurrent.futures import ThreadPoolExecutor,as_completed
_EVENT_LOCK=threading.Lock()
_append_json=append_json
def append_json(path,obj):
 with _EVENT_LOCK:_append_json(path,obj)
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'experiments/suite_policy'
RUN_ROOT=ROOT/'experiments/runs/2026-11-12-policy'
RUN=RUN_ROOT/'train'
PHASE='train'
HISTORY=ROOT/'experiments/runs/2026-11-12-v1/ledger.json'
BUDGET=RUN_ROOT/'unused-ledger.json'
MODEL_ALIASES={'small_plain':MODELS[0],'large_plain':MODELS[1],'small_reason':MODELS[2],'large_reason':MODELS[3]}
ARMS={}
NEW_TAG='nsrc-policy-v3-fitted:20261112'
BASE_TAG='nsrc-cpu-candidate:20261112'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def image_id(tag):return subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
def prepare():
 RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'protocol.json').exists():raise RuntimeError('Protocol already frozen')
 suite=read(SUITE/'manifest.json'); meta=read(SUITE/'research_metadata.json')
 # The metadata schema is validated by the builder; no gold enters a solver.
 selected=[r['id'] for r in suite['tasks'] if meta[r['id']]['split']==PHASE]
 if not selected:raise RuntimeError('Empty phase')
 arms={}
 for alias,model in MODEL_ALIASES.items():
  if PHASE=='test':arms['classic_'+alias]={'tag':NEW_TAG,'initial_model':model,'retry_policy':'classic'}
  arms['adaptive_'+alias]={'tag':NEW_TAG,'initial_model':model,'retry_policy':'adaptive'}
 for item in arms.values():item['image']=image_id(item['tag'])
 protocol={'created':time.time(),'phase':PHASE,'label':'new family-disjoint policy study',
  'arms':arms,'manifest_sha256':sha(SUITE/'manifest.json'),
  'oracle_sha256':sha(SUITE/'oracle.json'),'metadata_sha256':sha(SUITE/'research_metadata.json'),
  'runner_sha256':sha(Path(__file__)),
  'budget_start':{arm:0 for arm in arms},
  'ceilings':{m:100000 for m in MODELS},'pipeline_budget':100000,'max_parallel_pipelines':4 if PHASE=='test' else 2,
  'historic_ledger_sha256':sha(HISTORY),
  'task_ids':selected,
  'notes':['No per-task token allowance; each container has a shared 100k budget.',
  'User clarified: unlimited experiments; EACH isolated model/policy pipeline has its own 100k ledger.',
  'One HTTP request at a time; output caps are per-call only.',
  'All four models participate; historic ledgers are preserved, not charged against new pipelines.',
  'Test families are absent from train/calibration; artifacts frozen before test.']}
 if PHASE=='test':
  artifact=ROOT/'cost_artifact.json'
  if not artifact.is_file():raise RuntimeError('Freeze cost artifact before test')
  protocol['artifact_sha256']=sha(artifact)
  protocol['artifact_in_image_sha256']={}
  for frozen_image in {item['image'] for item in arms.values()}:
   command=['docker','run','--rm','--network','none','--read-only',frozen_image,'python','-B','-c',
    'import hashlib,pathlib;print(hashlib.sha256(pathlib.Path("/app/cost_artifact.json").read_bytes()).hexdigest())']
   image_hash=subprocess.check_output(command,text=True).strip()
   if image_hash!=protocol['artifact_sha256']:raise RuntimeError('Image artifact differs from fitted artifact')
   protocol['artifact_in_image_sha256'][frozen_image]=image_hash
  from dataclasses import asdict
  from policy_features import extract_features,fallback_cost,POLICY_ID
  from cost_model import load_cost_model
  from forecast_guard import choose_forecast
  from planner import TaskEstimator
  from feasibility import analyze_task
  predictor=load_cost_model(artifact,expected_policy_id=POLICY_ID)
  forecasts=[]
  for row in suite['tasks']:
   if row['id'] not in selected:continue
   folder=SUITE/row['task_path'];task=read(folder/'task.json');workspace=folder/'workspace'
   features=extract_features(task,workspace)
   heuristic=TaskEstimator(policy_id=POLICY_ID).estimate(task,workspace)
   hard=analyze_task(task,workspace).hard_impossible
   for model in MODELS:
    forecast=predictor.predict(features,model=model,policy_id=POLICY_ID,domain=row['domain'])
    prior=fallback_cost(features,model)
    deployed=choose_forecast(forecast,*prior,pipeline_budget=100000)
    forecasts.append({'id':row['id'],'model':model,'features':features,
      'heuristic':heuristic.predicted_tokens,'fallback':prior,
      'forecast':asdict(forecast),'deployed':asdict(deployed),'hard_skip':hard})
  protocol['forecast_selector']='guarded-v1'
  protocol['forecast_guard_sha256']=sha(ROOT/'forecast_guard.py')
  write_json(RUN/'forecasts.json',forecasts)
  protocol['forecasts_sha256']=sha(RUN/'forecasts.json')
 write_json(RUN/'protocol.json',protocol)
 print(json.dumps({'phase':PHASE,'tasks':len(selected),'arms':list(arms),'budget_start':protocol['budget_start']},ensure_ascii=False))

class Relay:
 def __init__(self,protocol):
  self.protocol=copy.deepcopy(protocol);self.ledger=Ledger(BUDGET);self.token=secrets.token_hex(24);self.arm='';self.selected_model=None
  self.lock=threading.Lock();self.client=httpx.Client(timeout=910,trust_env=False)
  self.base=os.environ.get('LLM_BASE_URL','https://api.neuraldeep.ru/v1').rstrip('/')
  self.key=os.environ['LLM_API_KEY']
  self.instructions={r['id']:read(SUITE/r['task_path']/'task.json')['instruction'] for r in read(SUITE/'manifest.json')['tasks']}
 def serve(self):
  parent=self
  class Handler(http.server.BaseHTTPRequestHandler):
   def log_message(self,*_):pass
   def respond(self,status,data):
    body=data if isinstance(data,bytes) else json.dumps(data).encode()
    self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers()
    try:self.wfile.write(body)
    except (BrokenPipeError,ConnectionResetError):pass
   def do_POST(self):
    if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer '+parent.token:
     self.respond(403,{'error':{'message':'Invalid relay request'}});return
    size=int(self.headers.get('Content-Length',0))
    if not 0<size<2000000:self.respond(400,{'error':{'message':'Request size'}});return
    try:
     payload=json.loads(self.rfile.read(size));model=payload['model'];cap=int(payload.get('max_tokens',0))
     if model not in MODELS or not 1<=cap<=32768 or payload.get('stream'):raise ValueError('Request policy')
     if parent.selected_model and model!=parent.selected_model:raise ValueError('Fixed model mismatch')
     text='\n'.join(str(m.get('content','')) for m in payload['messages'])
    except (ValueError,KeyError,TypeError):self.respond(400,{'error':{'message':'Invalid request'}});return
    task=next((k for k,v in parent.instructions.items() if v in text),'unknown')
    if task=='unknown':
     append_json(RUN/'calls.jsonl',{'event':'invalid_attribution','arm':parent.arm,'model':model,'time':time.time()})
     self.respond(400,{'error':{'message':'Unattributed experiment request'}});return
    arm=parent.arm
    with parent.lock:
     reserve=parent.ledger.reserve(model,cap,json.dumps(payload['messages'],ensure_ascii=False),parent.protocol['ceilings'][model])
     if reserve is None:
      append_json(RUN/'calls.jsonl',{'event':'not_started','arm':arm,'task':task,'model':model,'time':time.time()})
      self.respond(402,{'error':{'message':'Experimental budget exhausted','code':'insufficient_quota'}});return
     cid,amount=reserve;start=time.monotonic()
     append_json(RUN/'calls.jsonl',{'event':'reserved','arm':arm,'task':task,'model':model,'id':cid,'cap':cap,'reserved':amount/10,'time':time.time(),'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
     try:
      response=parent.client.post(parent.base+'/chat/completions',headers={'Authorization':'Bearer '+parent.key},json=payload)
      if response.status_code!=200:
       parent.ledger.fail(model,cid)
       append_json(RUN/'calls.jsonl',{'event':'error','arm':arm,'task':task,'model':model,'id':cid,'http_status':response.status_code,'cost':amount/10,'elapsed':time.monotonic()-start})
       self.respond(response.status_code,response.content);return
      result=response.json();u=result.get('usage') or {};reason=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',u.get('reasoning_tokens',0))
      usage={'input_tokens':u.get('prompt_tokens',0),'output_tokens':u.get('completion_tokens',0),'reasoning_tokens':reason,'total_tokens':u.get('total_tokens',0)}
      cost,accounting,breach=parent.ledger.reconcile(model,cid,amount,usage)
      append_json(RUN/'calls.jsonl',{'event':'completed','arm':arm,'task':task,'model':model,'id':cid,'cost':cost,'usage':usage,'accounting':accounting,'breach':breach,'elapsed':time.monotonic()-start,'response':result.get('choices')})
      if breach:parent.protocol['ceilings'][model]=parent.ledger.spent(model)
      self.respond(200,response.content)
     except Exception as exc:
      # Reconcile may already have finalized; never refund an unknown outcome.
      if cid in parent.ledger.data['models'][model]['pending']:parent.ledger.fail(model,cid)
      append_json(RUN/'calls.jsonl',{'event':'error','arm':arm,'task':task,'model':model,'id':cid,'type':type(exc).__name__,'cost':amount/10,'elapsed':time.monotonic()-start})
      self.respond(503,{'error':{'message':'Temporary relay transport error'}})
  self.server=http.server.ThreadingHTTPServer(('0.0.0.0',0),Handler)
  self.server.daemon_threads=False
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
 def close(self):self.server.shutdown();self.server.server_close();self.client.close()

def run():
 p=read(RUN/'protocol.json')
 if 'artifact_sha256' in p and sha(ROOT/'cost_artifact.json')!=p['artifact_sha256']:raise RuntimeError('Frozen artifact changed')
 if 'forecasts_sha256' in p and sha(RUN/'forecasts.json')!=p['forecasts_sha256']:raise RuntimeError('Frozen forecasts changed')
 if 'forecast_guard_sha256' in p and sha(ROOT/'forecast_guard.py')!=p['forecast_guard_sha256']:raise RuntimeError('Frozen forecast guard changed')
 for key,path in [('manifest_sha256',SUITE/'manifest.json'),('oracle_sha256',SUITE/'oracle.json'),('runner_sha256',Path(__file__)),('metadata_sha256',SUITE/'research_metadata.json')]:
  if p[key]!=sha(path):raise RuntimeError('Frozen input changed: '+key)
 from experiments.build_production_challenge import dumps,digest
 manifest=read(SUITE/'manifest.json')
 expected_dirs={r['id'] for r in manifest['tasks']}
 if {x.name for x in (SUITE/'tasks').iterdir() if x.is_dir()}!=expected_dirs:raise RuntimeError('Task directory set changed')
 for row in manifest['tasks']:
  folder=SUITE/row['task_path']
  actual_files={x.relative_to(folder/'workspace').as_posix() for x in (folder/'workspace').rglob('*') if x.is_file()}
  if actual_files!=set(row['files_sha256']):raise RuntimeError('Visible file inventory changed')
  if digest(dumps(read(folder/'task.json')))!=row['task_sha256']:raise RuntimeError('Visible task changed')
  for name,expected_hash in row['files_sha256'].items():
   if digest((folder/'workspace'/name).read_text(encoding='utf-8'))!=expected_hash:raise RuntimeError('Visible file changed')
 for a,item in p['arms'].items():
  if image_id(item['tag'])!=item['image']:raise RuntimeError('Frozen image changed: '+a)
 def run_arm(arm,item):
  folder=RUN/arm
  if (folder/'completed.json').exists():return
  tasks=folder/'tasks'
  if tasks.exists():raise RuntimeError('Partial pipeline needs review; refusing to reset its answers or budget')
  tasks.mkdir(parents=True)
  for row in manifest['tasks']:
   if row['id'] in p['task_ids']:shutil.copytree(SUITE/row['task_path'],tasks/row['id'])
  relay=Relay(p);relay.arm=arm;relay.ledger=Ledger(folder/'ledger.json');relay.selected_model=item['initial_model']
  relay.serve()
  name='nsrc-policy-'+arm+'-'+secrets.token_hex(4)
  command=['docker','run','--rm','--name',name,'--read-only','--tmpfs','/tmp:rw,size=2g','--cpus','8','--memory','16g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges',
   '-e','HOME=/tmp','-e',f'LLM_BASE_URL=http://host.docker.internal:{relay.server.server_port}/v1','-e','LLM_API_KEY='+relay.token,
   '--mount',f'type=bind,source={tasks.resolve()},target=/data/tasks',item['image'],'python','/app/run_agent.py',
   '--tasks-dir','/data/tasks','--budget','100000','--trace-file','/data/tasks/trace.jsonl',
   '--initial-model',item['initial_model'],'--fixed-model','--retry-policy',item['retry_policy']]
  start=time.monotonic();returncode=124
  try:
   with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
    returncode=subprocess.run(command,stdout=out,stderr=err,timeout=3500).returncode
  except subprocess.TimeoutExpired:pass
  finally:
   subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
   relay.close()
   write_json(folder/'ledger_end.json',relay.ledger.data)
  used=sum(v['charged_units'] for v in relay.ledger.data['models'].values())/10
  if used>100000:raise RuntimeError('Pipeline budget exceeded')
  write_json(folder/'completed.json',{'returncode':returncode,'seconds':time.monotonic()-start,
   'image':item['image'],'model':item['initial_model'],'charged':used,'finished':time.time()})
  print(json.dumps({'phase':PHASE,'arm':arm,'returncode':returncode,'seconds':round(time.monotonic()-start,1),'charged':used}),flush=True)
  if returncode:raise RuntimeError('Production pipeline failed: '+arm)
 with portalocker.Lock(str(RUN/'process.lock'),timeout=0):
  with ThreadPoolExecutor(max_workers=p.get('max_parallel_pipelines',1)) as pool:
   futures=[pool.submit(run_arm,arm,item) for arm,item in p['arms'].items()]
   for future in as_completed(futures):future.result()
 if sha(HISTORY)!=p['historic_ledger_sha256']:raise RuntimeError('Historic ledger changed unexpectedly')

CHECK_CODE=r'''
import json,tempfile,pathlib,sys
sys.path.insert(0,'/app')
from coding_tools import run_tests
jobs=json.load(sys.stdin);results={}
for job in jobs:
 with tempfile.TemporaryDirectory() as tmp:
  p=pathlib.Path(tmp)
  for name,text in job['files'].items():
   q=p/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
  ok,feedback=run_tests(p);results[job['id']]={'passed':ok,'feedback':feedback}
print(json.dumps(results))
'''

def charged_calls(events):
 grouped={}
 for event in events:
  if event.get('id'):grouped.setdefault(event['id'],[]).append(event)
 charged=[]
 for call_id,records in grouped.items():
  initial=next((r for r in records if r['event']=='reserved'),records[0])
  terminal=next((r for r in records if r['event']=='completed'),None)
  if terminal is None:terminal=next((r for r in records if r['event']=='error'),None)
  charged.append({**initial,**(terminal or {}),'id':call_id,'arm':initial['arm'],
                  'cost':terminal['cost'] if terminal else initial['reserved'],
                  'accounting':terminal.get('accounting','reserved_error') if terminal else 'reserved_pending'})
 return charged

def report():
 manifest=read(SUITE/'manifest.json');oracle=read(SUITE/'oracle.json');p=read(RUN/'protocol.json')
 manifest={**manifest,'tasks':[r for r in manifest['tasks'] if r['id'] in p['task_ids']]}
 calls=[json.loads(x) for x in (RUN/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
 charged=charged_calls(calls)
 budgets={arm:read(RUN/arm/'ledger_end.json') for arm in p['arms'] if (RUN/arm/'ledger_end.json').exists()}
 delta=sum(v['charged_units']/10 for data in budgets.values() for v in data['models'].values())
 if abs(delta-sum(c['cost'] for c in charged))>0.01:raise RuntimeError('Call accounting differs from pipeline ledgers')
 for arm,data in budgets.items():
  model=p['arms'][arm]['initial_model']
  if sum(v['charged_units'] for v in data['models'].values())>1000000:raise RuntimeError('Pipeline exceeded 100k')
  if any(v['charged_units'] and m!=model for m,v in data['models'].items()):raise RuntimeError('Unexpected model switch')
 summary={'protocol':p,'arms':{},'budget':budgets,'ledger_delta':round(delta,1),
  'budget_denials':sum(c['event']=='not_started' for c in calls),
  'comparison_complete':not any(c['event'] in {'not_started','invalid_attribution'} for c in calls) and all((RUN/a/'completed.json').exists() and read(RUN/a/'completed.json')['returncode']==0 for a in p['arms'])}
 for arm in p['arms']:
  folder=RUN/arm
  if not (folder/'completed.json').exists():continue
  jobs=[]
  for row in manifest['tasks']:
   gold=oracle[row['id']];work=folder/row['task_path']/'workspace'
   if gold.get('hidden_files'):
    files={x.relative_to(work).as_posix():x.read_text(encoding='utf-8') for x in work.rglob('*') if x.is_file() and x.name!='answer.txt' and '__pycache__' not in x.parts}
    files.update(gold['hidden_files']);jobs.append({'id':row['id'],'files':files})
  checked={}
  if jobs:
   command=['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=256m','--cpus','2','--memory','1g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',next(iter(p['arms'].values()))['image'],'python','-B','-c',CHECK_CODE]
   result=subprocess.run(command,input=json.dumps(jobs),text=True,encoding='utf-8',capture_output=True,timeout=150)
   if result.returncode:raise RuntimeError('Hidden validator failed: '+result.stderr[-1000:])
   checked=json.loads(result.stdout.strip().splitlines()[-1])
  traces=[json.loads(x) for x in (folder/'tasks/trace.jsonl').read_text(encoding='utf-8').splitlines()]
  byid={x['task_id']:x for x in traces};rows=[]
  for row in manifest['tasks']:
   gold=oracle[row['id']];answerpath=folder/row['task_path']/'answer.txt';answer=answerpath.read_text(encoding='utf-8').strip() if answerpath.exists() else None
   correct=bool(gold['answerable'] and answer is not None and (answer=='DONE' and checked.get(row['id'],{}).get('passed',False) if row['domain']=='coding' else (numeric_equal(answer,gold['expected']) if row['domain'] in {'math','data_analysis'} else answer.strip()==str(gold['expected']).strip())))
   records=[c for c in charged if c.get('arm')==arm and c.get('task')==row['id']]
   cost=sum(c['cost'] for c in records)
   rows.append({'id':row['id'],'domain':row['domain'],'answerable':gold['answerable'],'correct':correct,'answered':answer is not None,'answer':answer,'cost':cost,'reward':row['reward'],'trace':byid.get(row['id'])})
  summary['arms'][arm]={'solved':sum(r['correct'] for r in rows),'answerable':sum(r['answerable'] for r in rows),'unanswerable_answered':sum(not r['answerable'] and r['answered'] for r in rows),'reward':sum(r['reward'] for r in rows if r['correct']),'max_reward':sum(r['reward'] for r in rows if r['answerable']),
   'cost':sum(c['cost'] for c in charged if c.get('arm')==arm),'calls':sum(c.get('arm')==arm for c in charged),'rows':rows}
 write_json(RUN/'evaluation.json',summary)
 print(json.dumps({a:{k:v for k,v in data.items() if k!='rows'} for a,data in summary['arms'].items()},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('action',choices=['prepare','run','report'])
 parser.add_argument('--phase',choices=['train','calibration','test'],required=True)
 args=parser.parse_args();PHASE=args.phase;RUN=RUN_ROOT/PHASE
 globals()[args.action]()
