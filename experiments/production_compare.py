"""Paired real production-container evaluation with the existing per-model ledger.

The local relay reserves EACH physical HTTP request before forwarding. Solver
containers receive only a random relay token; the NeuralDeep key stays on host.
Gold truth is opened only by the post-run evaluator.
"""
from __future__ import annotations
import argparse,hashlib,http.server,json,os,secrets,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.run_experiments import Ledger,append_json,write_json,MODELS,numeric_equal,norm_answer
import httpx,portalocker
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'experiments/suite_production'
RUN=ROOT/'experiments/runs/2026-11-12-production'
BUDGET=ROOT/'experiments/runs/2026-11-12-v1/ledger.json'
ARMS={'before':'nsrc-before-autonomy:20261112','candidate':'nsrc-cpu-candidate:20261112'}

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def image_id(tag):return subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
def prepare():
 RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'protocol.json').exists():raise RuntimeError('Protocol already frozen')
 ledger=Ledger(BUDGET)
 limits={'qwen3.6-35b-a3b-noreason':12000,'qwen3.8-27b-noreason':18000,'qwen3.6-35b-a3b':0,'qwen3.8-27b':12000}
 suite=read(SUITE/'manifest.json')
 protocol={'created':time.time(),'label':'paired end-to-end structural stress; not representative hidden-final evaluation',
  'arms':{a:{'tag':t,'image':image_id(t)} for a,t in ARMS.items()},'manifest_sha256':sha(SUITE/'manifest.json'),
  'oracle_sha256':sha(SUITE/'oracle.json'),'runner_sha256':sha(Path(__file__)),
  'budget_start':{m:ledger.spent(m) for m in MODELS},'ceilings':{m:min(100000,ledger.spent(m)+limits[m]) for m in MODELS},
  'task_ids':[r['id'] for r in suite['tasks']],
  'notes':['Same tasks and rewards; full production policies including routing differ.',
   'No oracle fields enter prompts or solver containers. Baseline image is unchanged.',
   'No resetting original ledger. Qwen3.6 reasoning has no new experimental allocation.',
   'Single relay request at a time, no HTTP client retries. API errors retain reserves.']}
 write_json(RUN/'protocol.json',protocol);print(json.dumps(protocol,ensure_ascii=False,indent=2))

class Relay:
 def __init__(self,protocol):
  self.protocol=protocol;self.ledger=Ledger(BUDGET);self.token=secrets.token_hex(24);self.arm=''
  self.lock=threading.Lock();self.client=httpx.Client(timeout=95,trust_env=False)
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
     if model not in MODELS or not 1<=cap<=8192 or payload.get('stream'):raise ValueError('Request policy')
     text='\n'.join(str(m.get('content','')) for m in payload['messages'])
    except (ValueError,KeyError,TypeError):self.respond(400,{'error':{'message':'Invalid request'}});return
    task=next((k for k,v in parent.instructions.items() if v in text),'unknown')
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
 for key,path in [('manifest_sha256',SUITE/'manifest.json'),('oracle_sha256',SUITE/'oracle.json'),('runner_sha256',Path(__file__))]:
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
 with portalocker.Lock(str(BUDGET.parent/'process.lock'),timeout=0):
  relay=Relay(p);relay.serve()
  try:
   for arm,item in p['arms'].items():
    folder=RUN/arm
    if (folder/'completed.json').exists():continue
    tasks=folder/'tasks'
    if tasks.exists():raise RuntimeError('Partial arm needs review; refusing to reset answers')
    shutil.copytree(SUITE/'tasks',tasks)
    with relay.lock:relay.arm=arm
    name='nsrc-compare-'+arm+'-'+secrets.token_hex(4)
    command=['docker','run','--rm','--name',name,'--read-only','--tmpfs','/tmp:rw,size=2g','--cpus','8','--memory','16g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges',
     '-e','HOME=/tmp','-e',f'LLM_BASE_URL=http://host.docker.internal:{relay.server.server_port}/v1','-e','LLM_API_KEY='+relay.token,
     '--mount',f'type=bind,source={tasks.resolve()},target=/data/tasks',item['tag'],'python','/app/run_agent.py','--tasks-dir','/data/tasks','--budget','30000','--trace-file','/data/tasks/trace.jsonl']
    start=time.monotonic()
    returncode=124
    try:
     with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
      returncode=subprocess.run(command,stdout=out,stderr=err,timeout=3500).returncode
    except subprocess.TimeoutExpired:pass
    finally:
     subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
     with relay.lock:pass
    write_json(folder/'completed.json',{'returncode':returncode,'seconds':time.monotonic()-start,'image':item['image'],'finished':time.time()})
    print(json.dumps({'arm':arm,'returncode':returncode,'seconds':round(time.monotonic()-start,1),'budget':{m:relay.ledger.spent(m) for m in MODELS}}),flush=True)
    if returncode:raise RuntimeError('Production arm failed: '+arm)
  finally:relay.close()

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
 calls=[json.loads(x) for x in (RUN/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
 charged=charged_calls(calls)
 ledger=Ledger(BUDGET)
 delta=sum(ledger.spent(m)-p['budget_start'][m] for m in MODELS)
 if abs(delta-sum(c['cost'] for c in charged))>0.01:raise RuntimeError('Call accounting differs from shared ledger delta')
 summary={'protocol':p,'arms':{},'budget':ledger.data,'ledger_delta':round(delta,1),'budget_denials':sum(c['event']=='not_started' for c in calls),'comparison_complete':not any(c['event']=='not_started' for c in calls)}
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
   command=['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=256m','--cpus','2','--memory','1g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',p['arms']['candidate']['tag'],'python','-B','-c',CHECK_CODE]
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
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','report']);args=parser.parse_args();globals()[args.action]()
