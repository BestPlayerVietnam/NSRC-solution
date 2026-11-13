"""Bounded, resumable four-model evaluation through the real openJiuwen Model client."""
from __future__ import annotations
import argparse, asyncio, ast, csv, hashlib, io, json, math, os, re, sqlite3, statistics, subprocess, sys, time, uuid
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'experiments'/'suite'
RUN=ROOT/'experiments'/'runs'/'2026-11-12-v1'
MODELS=('qwen3.6-35b-a3b-noreason','qwen3.8-27b-noreason','qwen3.6-35b-a3b','qwen3.8-27b')
LIMIT=100000
POLICY='bounded-tool-math-v2'


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def append_json(path,obj):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def redact(s):return re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(s))[:500]

class Ledger:
    """Integer tenths of a weighted token. Pending calls remain fully reserved on restart."""
    def __init__(self,path,limit=LIMIT):
        self.path=path;self.limit=min(LIMIT,limit)
        self.data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version':1,'limit_each':LIMIT,'models':{m:{'charged_units':0,'calls':0,'pending':{}} for m in MODELS}}
        if self.data['limit_each']!=LIMIT:raise ValueError('Unexpected budget history; refusing reset')
    def reserve(self,model,cap,text,ceiling=LIMIT):
        # UTF-8 bytes bound ordinary byte-tokenized prompt payload conservatively;
        # overhead and twice the output cap protect framing/separate reasoning.
        amount=20*cap+len(text.encode('utf-8'))+1280
        rec=self.data['models'][model]
        if rec['charged_units']+amount>10*min(self.limit,ceiling):return None
        call_id=uuid.uuid4().hex
        rec['charged_units']+=amount;rec['calls']+=1
        rec['pending'][call_id]=amount;write_json(self.path,self.data)
        return call_id,amount
    def reconcile(self,model,call_id,amount,usage):
        input_t=int(usage.get('input_tokens',0));out=int(usage.get('output_tokens',0));reason=int(usage.get('reasoning_tokens',0));total=int(usage.get('total_tokens',0))
        trusted=False;output=out
        if input_t>=0 and out>=0 and total>0:
            if total==input_t+out and 0<=reason<=out:trusted=True
            elif reason>=0 and total==input_t+out+reason:output=out+reason;trusted=True
        charge=input_t+10*output if trusted else max(amount,input_t+10*max(out+reason,total-input_t,0))
        rec=self.data['models'][model];rec['charged_units']+=charge-amount;rec['pending'].pop(call_id,None)
        write_json(self.path,self.data)
        return charge/10,'provider_reconciled' if trusted else 'conservative_reserved',charge>amount
    def fail(self,model,call_id):
        # Do not release money after an ambiguous timeout or missing usage.
        self.data['models'][model]['pending'].pop(call_id,None);write_json(self.path,self.data)
    def spent(self,model):return self.data['models'][model]['charged_units']/10


def norm_answer(s):
    s=str(s).casefold().strip()
    s=re.sub(r'\b(a|an|the)\b',' ',s)
    s=re.sub(r'[^\w\s]','',s)
    return ' '.join(s.split())
def abstains(text):return bool(re.fullmatch(r'\s*(?:UNSOLVABLE|INSUFFICIENT_DATA|ABSTAIN)[.!]?\s*',text,re.I))
def clean_fence(text):return re.sub(r'^```(?:python|sql|sqlite)?\s*\n?|\n?```\s*$','',text.strip(),flags=re.I).strip()
def numeric_equal(a,b):
    try:return math.isclose(float(str(a).replace(',','')),float(str(b).replace(',','')),rel_tol=1e-9,abs_tol=1e-8)
    except (ValueError,TypeError):return False

# Runs in a container with no network, no mounted host files and no credentials.
CONTAINER_CHECK=r'''
import json,subprocess,sys,tempfile,pathlib,os
job=json.load(sys.stdin)
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);(p/'mylib.py').write_text(job['code'],encoding='utf-8')
 code="import sys;sys.path.insert(0,"+repr(d)+");from mylib import *\n"+'\n'.join(job.get('imports',[]))+'\n'+job['tests']
 try:
  cmd=[sys.executable,'-I','-B',str(p/'mylib.py')] if job.get('mode')=='script' else [sys.executable,'-I','-B','-c',code]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=4,cwd=d,env={'PATH':os.environ.get('PATH',''),'HOME':'/tmp','PYTHONIOENCODING':'utf-8'})
  print(json.dumps({'passed':r.returncode==0,'feedback':(r.stderr or r.stdout)[-800:]}))
 except subprocess.TimeoutExpired:print(json.dumps({'passed':False,'feedback':'Test timed out'}))
'''

def check_code(code,tests,imports=None,mode=None):
    try:ast.parse(code)
    except SyntaxError as e:return {'passed':False,'feedback':'SyntaxError: '+str(e)[:200]}
    cmd=['docker','run','--rm','-i','--network','none','--read-only','--tmpfs','/tmp:rw,size=64m','--memory','256m','--cpus','1','--pids-limit','64','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--workdir','/tmp','nsrc-harness:latest','python','-I','-B','-c',CONTAINER_CHECK]
    try:
        r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':tests,'imports':imports or [],'mode':mode}),text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=25)
        if r.returncode:return {'passed':False,'feedback':'Container check error: '+r.stderr[-350:],'infrastructure_error':True}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired,ValueError,IndexError) as e:return {'passed':False,'feedback':type(e).__name__,'infrastructure_error':True}


def execute_sql(text,workspace):
    sql=clean_fence(text).removesuffix(';').strip()
    if not re.match(r'^SELECT\b',sql,re.I) or ';' in sql:return None,'Expected one SELECT or UNSOLVABLE'
    with (workspace/'sales.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not fields or any(not re.fullmatch(r'[A-Za-z_]\w*',x) for x in fields):return None,'Unsupported schema'
    con=sqlite3.connect(':memory:')
    try:
        def value(x):
            if re.fullmatch(r'-?\d+(?:\.\d+)?',x or ''):return float(x)
            return x
        con.execute('CREATE TABLE data ('+','.join('"'+f+'" COLLATE NOCASE' for f in fields)+')')
        con.executemany('INSERT INTO data VALUES ('+','.join('?' for _ in fields)+')',[[value(r[f]) for f in fields] for r in rows])
        allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION}
        con.set_authorizer(lambda action,*_:sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        steps=0
        def progress():
            nonlocal steps
            steps+=1;return int(steps>2000)
        con.set_progress_handler(progress,1000)
        out=con.execute(sql).fetchmany(3)
        if len(out)!=1 or len(out[0])!=1:return None,'Query must return one scalar'
        v=out[0][0]
        if v is None:return 'NULL',None
        if isinstance(v,float) and v.is_integer():v=int(v)
        return str(v),None
    except sqlite3.Error as e:return None,'SQL error: '+str(e)[:250]
    finally:con.close()


def features(task,workspace):
    files=[p for p in workspace.rglob('*') if p.is_file()]
    f={'domain':task['task_type'],'instruction_chars':len(task['instruction']),'instruction_words':len(task['instruction'].split()),'file_count':len(files),'bytes':sum(p.stat().st_size for p in files),'ast_nodes':0,'asserts':0,'csv_rows':0,'csv_cols':0,'kb_docs':sum(p.suffix=='.md' for p in files)}
    for p in files:
        if p.suffix=='.py':
            try:
                nodes=list(ast.walk(ast.parse(p.read_text(encoding='utf-8'))));f['ast_nodes']+=len(nodes);f['asserts']+=sum(isinstance(n,ast.Assert) for n in nodes)
            except SyntaxError:pass
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as s:
                rr=list(csv.reader(s));f['csv_rows']=max(0,len(rr)-1);f['csv_cols']=len(rr[0]) if rr else 0
    return f

def prompt_for(task,workspace):
    kind=task['task_type'];q=task['instruction']
    if kind=='coding':
        return 'Restore the Python module. Output only complete source code, or UNSOLVABLE for contradictory requirements. Do not modify tests.',q+'\n\nmylib.py:\n'+(workspace/'mylib.py').read_text(encoding='utf-8')+'\nPublic tests:\n'+(workspace/'test_mylib.py').read_text(encoding='utf-8')
    if kind=='data_analysis':
        with (workspace/'sales.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.reader(f))
        return 'Output exactly one SQLite SELECT query over table data, or UNSOLVABLE if essential data are absent. Column names are exactly as supplied. Text comparisons are case insensitive. Numeric-looking values are stored numerically. The shown rows are ONLY a preview, not the full table: your query executes on ALL rows. Never abstain merely because a queried value is absent from the preview. No markdown.',q+'\nCSV header and sample rows:\n'+json.dumps(rows[:4],ensure_ascii=False)
    if kind=='search':
        return 'Answer only from the named evidence document. Return a short copied answer, or UNSOLVABLE if the evidence does not support an answer.',q+'\n\nkb/source.md:\n'+(workspace/'kb/source.md').read_text(encoding='utf-8')
    return 'Solve the arithmetic task by writing a short standalone Python 3 program that computes and prints ONLY the final numeric answer. Return code without markdown. If essential facts are missing, return UNSOLVABLE instead. The program is executed locally; do not use external packages.',q


class Experiment:
    def __init__(self,args):
        self.args=args;RUN.mkdir(parents=True,exist_ok=True)
        self.ledger=Ledger(RUN/'ledger.json');self.sem=asyncio.Semaphore(1)
        self.manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'));self.oracle=json.loads((SUITE/'oracle.json').read_text(encoding='utf-8'))
        self.done={}
        if (RUN/'results.jsonl').exists():
            for line in (RUN/'results.jsonl').read_text(encoding='utf-8').splitlines():
                r=json.loads(line);self.done[r['key']]=r
        self.clients={};self.halted=set();self.next_allowed_time=0.0
        self.image_id=subprocess.check_output(['docker','image','inspect','nsrc-harness:latest','--format','{{.Id}}'],text=True).strip()
        self.predictions={}
        for split in ('calibration','test'):
            path=RUN/(split+'_predictions.jsonl')
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    item=json.loads(line);self.predictions[(item['id'],item['model'])]=item
        panel_path=RUN/'evaluation_panel.json'
        self.panel=json.loads(panel_path.read_text(encoding='utf-8'))['ids'] if panel_path.exists() else {}
    def client(self,model):
        if model in self.clients:return self.clients[model]
        from openjiuwen.core.common.logging.log_config import configure_log_config
        configure_log_config({'backend':'default','level':'ERROR','output':['console'],'interface_output':['console'],'performance_output':['console']})
        from openjiuwen.core.foundation.llm.model import Model
        from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig,ModelRequestConfig
        base=os.environ.get('LLM_BASE_URL','https://api.neuraldeep.ru/v1').rstrip('/')
        cfg=ModelClientConfig(client_provider='OpenAI',api_base=base,api_key=os.environ['LLM_API_KEY'],timeout=90,max_retries=0)
        self.clients[model]=Model(cfg,ModelRequestConfig(model=model))
        return self.clients[model]
    async def ask(self,model,system,user,cap,ceiling,taskid):
        for retry in range(3):
            async with self.sem:
                if time.monotonic()<self.next_allowed_time:await asyncio.sleep(self.next_allowed_time-time.monotonic())
                reservation=self.ledger.reserve(model,cap,system+user,ceiling)
                if reservation is None:return {'content':'','status':'budget_not_started','cost':0,'finish_reason':None}
                call_id,amount=reservation;t=time.monotonic()
                append_json(RUN/'calls.jsonl',{'event':'reserved','id':call_id,'model':model,'task':taskid,'cap':cap,'reserved':amount/10,'time':time.time()})
                try:
                    resp=await asyncio.wait_for(self.client(model).invoke([{'role':'system','content':system},{'role':'user','content':user}],model=model,max_tokens=cap,temperature=0,timeout=90),timeout=100)
                    usage=resp.usage_metadata.model_dump() if resp.usage_metadata else {}
                    usage={k:v for k,v in usage.items() if k in ('input_tokens','output_tokens','reasoning_tokens','total_tokens','cache_tokens')}
                    cost,accounting,breach=self.ledger.reconcile(model,call_id,amount,usage)
                    out={'content':str(resp.content or ''),'status':'ok','cost':cost,'finish_reason':resp.finish_reason,'usage':usage,'accounting':accounting,'latency':time.monotonic()-t,'cap':cap}
                    append_json(RUN/'calls.jsonl',{'event':'completed','id':call_id,'model':model,'task':taskid,**out})
                    if breach:self.halted.add(model);out['budget_bound_breach']=True
                    return out
                except Exception as e:
                    self.ledger.fail(model,call_id);detail=redact(e)
                    append_json(RUN/'calls.jsonl',{'event':'error','id':call_id,'model':model,'task':taskid,'detail':detail,'cost_reserved':amount/10,'latency':time.monotonic()-t})
                    transient=bool(re.search(r'429|502|503|timeout|timed out',detail,re.I))
                    wait_match=re.search(r'(?:через|after)\s+(\d+)',detail,re.I)
                    retry_wait=max(65 if '429' in detail else 5*2**retry, int(wait_match.group(1))+2 if wait_match else 0)
                    self.next_allowed_time=time.monotonic()+retry_wait
                    if not transient or retry==2:
                        self.halted.add(model)
                        return {'content':'','status':'infrastructure_error','cost':amount/10,'finish_reason':None,'error':detail}
            await asyncio.sleep(retry_wait)
        raise AssertionError('unreachable')
    async def one(self,model,r,replicate=0,ceiling=LIMIT):
        key=f'{POLICY}:{model}:{r["id"]}:{replicate}'
        if key in self.done:return
        folder=SUITE/r['task_path'];workspace=folder/'workspace';task=json.loads((folder/'task.json').read_text(encoding='utf-8'))
        visible_files={p.relative_to(workspace).as_posix():p.read_bytes().decode('utf-8') for p in sorted(workspace.rglob('*')) if p.is_file()}
        canonical=lambda obj:json.dumps(obj,ensure_ascii=False,indent=2)
        # Builder text output expands LF on Windows; reverse that once, including CSV CRLF.
        variants=(visible_files,{k:v.replace('\r\n','\n') for k,v in visible_files.items()})
        fingerprints={hashlib.sha256((canonical(task)+canonical(files)).encode('utf-8')).hexdigest() for files in variants}
        if r['input_sha256'] not in fingerprints:raise RuntimeError('Task inputs changed after suite freeze: '+r['id'])
        f=features(task,workspace);system,user=prompt_for(task,workspace)
        reason=not model.endswith('-noreason')
        cap=1536 if reason else (768 if r['domain']=='coding' else 384)
        if replicate==1:cap=max(192,cap//2)
        if replicate==2:cap=min(2048,cap*2)
        before=self.ledger.spent(model);started=time.monotonic();calls=[];answer=None;public_pass=None;hidden_pass=None;feedback='';state=''
        baseline=(30+f['instruction_chars']/40+f['bytes']/40+(70 if r['domain']=='coding' else 0))*(3 if reason else 1)
        frozen_prediction=None
        if r['split'] in ('calibration','test'):
            frozen_prediction=self.predictions.get((r['id'],model))
            if not frozen_prediction or frozen_prediction['policy']!=POLICY or frozen_prediction['features']!=f:
                raise RuntimeError('Missing or mismatched pre-run prediction for '+r['id'])
            artifact=RUN/('predictor_train.json' if r['split']=='calibration' else 'predictor_calibrated.json')
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=frozen_prediction['artifact_sha256']:
                raise RuntimeError('Predictor artifact changed after planning')
            bundle=json.loads(artifact.read_text(encoding='utf-8'))
            if bundle['docker_image_id']!=self.image_id:raise RuntimeError('Docker image changed after predictor freeze')
            if bundle['runner_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError('Policy code changed after predictor freeze')
            append_json(RUN/'precall_predictions.jsonl',{'key':key,'time':time.time(),**frozen_prediction})
        # Gold truth is not used to select prompts, model, retries, or cap.
        for attempt in range(2):
            response=await self.ask(model,system,user if not attempt else user+'\n\nPrevious attempt failed the available check: '+feedback+'\nCorrect it; return only the required output.',cap,ceiling,r['id'])
            calls.append({k:v for k,v in response.items() if k!='content'})
            text=response['content'].strip()
            if response['status']!='ok':state=response['status'];break
            if abstains(text):answer='UNSOLVABLE';state='abstained';break
            if not text:feedback='No final answer was returned';state='empty_output'
            elif r['domain']=='coding':
                answer=clean_fence(text)
                check=await asyncio.to_thread(check_code,answer,(workspace/'test_mylib.py').read_text(encoding='utf-8'))
                public_pass=check['passed'];feedback=check['feedback'];state='answered' if public_pass else 'public_test_failed'
                if check.get('infrastructure_error'):state='validator_error';self.halted.add(model);break
                if public_pass:break
            elif r['domain']=='data_analysis':
                answer,feedback=execute_sql(text,workspace);state='answered' if feedback is None else 'sql_error'
                if feedback is None:break
            elif r['domain']=='math':
                if re.fullmatch(r'-?\d+(?:\.\d+)?',text):answer=text;state='answered';break
                check=await asyncio.to_thread(check_code,clean_fence(text),'',None,'script')
                feedback=check['feedback'];value=feedback.strip()
                if check.get('infrastructure_error'):state='validator_error';self.halted.add(model);break
                if check['passed'] and re.fullmatch(r'-?\d+(?:\.\d+)?',value):answer=value;state='answered';break
                state='calculation_error'
            else:
                answer=text;state='answered';break
            if model in self.halted:break
        if state=='budget_not_started' and len(calls)==1:
            append_json(RUN/'not_run.jsonl',{'key':key,'model':model,'id':r['id'],'split':r['split'],'reason':'stage_budget_reserve','ceiling':ceiling,'charged':self.ledger.spent(model)})
            return
        gold=self.oracle[r['id']]
        if not r['answerable']:
            correct=state=='abstained'
        elif state!='answered':correct=False
        elif r['domain']=='coding':
            check=await asyncio.to_thread(check_code,answer,'\n'.join(gold['hidden_tests']),gold.get('test_imports',[]))
            hidden_pass=check['passed'];correct=bool(public_pass and hidden_pass)
            if check.get('infrastructure_error'):state='validator_error';correct=False;self.halted.add(model)
        elif r['domain']=='search':correct=any(norm_answer(answer)==norm_answer(g) for g in gold['answer'])
        else:correct=numeric_equal(answer,gold['answer']) or str(answer).strip().casefold()==str(gold['answer']).strip().casefold()
        result={'key':key,'policy':POLICY,'id':r['id'],'source':r['source'],'group':r['group'],'split':r['split'],'domain':r['domain'],'model':model,'answerable':r['answerable'],'replicate':replicate,'reward':task['reward'],'correct':correct,'state':state,'answer':answer,'public_pass':public_pass,'hidden_pass':hidden_pass,'features':f,'predicted_tokens':baseline,'frozen_prediction':frozen_prediction,'provider_tokens':round(sum(c.get('cost',0) for c in calls if c.get('accounting')=='provider_reconciled'),1),'actual_tokens':round(self.ledger.spent(model)-before,1),'calls':calls,'elapsed':round(time.monotonic()-started,2)}
        append_json(RUN/'results.jsonl',result);self.done[key]=result
        print(json.dumps({'model':model,'task':r['id'],'split':r['split'],'correct':correct,'state':state,'cost':result['actual_tokens'],'spent':self.ledger.spent(model)},ensure_ascii=False),flush=True)
    def pilot(self):
        rows=[]
        for domain in ('math','coding','search','data_analysis'):
            selected=[r for r in self.manifest['tasks'] if r['split']=='train' and r['domain']==domain]
            rows.extend([r for r in selected if r['answerable']][:3]+[r for r in selected if not r['answerable']][:1])
        return rows
    async def worker(self,model):
        pilot=self.pilot()
        if self.args.phase=='pilot':schedule=[(r,0,25000) for r in pilot]
        else:
            schedule=[]
            for split,ceiling in [('train',55000),('calibration',75000),('test',LIMIT)]:
                if self.args.phase!='full' and split!=self.args.phase:continue
                rows=[r for r in self.manifest['tasks'] if r['split']==split]
                # Fixed panel first; hash order for additional observations is identical across models.
                primary=self.panel.get(split,[])
                order={task_id:i for i,task_id in enumerate(primary)}
                rows.sort(key=lambda r:(0,order[r['id']]) if r['id'] in order else (1,int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)))
                schedule.extend((r,0,ceiling) for r in rows)
            # Cap experiments must finish on training data before any held-out evaluation.
            ablations=[(r,rep,55000) for rep in (1,2) for r in pilot[:2]]
            first_holdout=next((i for i,x in enumerate(schedule) if x[0]['split']!='train'),len(schedule))
            if self.args.phase in ('train','full'):schedule[first_holdout:first_holdout]=ablations
        for r,rep,ceiling in schedule:
            if model in self.halted:break
            await self.one(model,r,rep,ceiling)
    async def run(self):
        phases=('train','calibration','test') if self.args.phase=='full' else (self.args.phase,)
        for phase in phases:
            self.args.phase=phase
            await asyncio.gather(*(self.worker(m) for m in MODELS if not self.args.model or m==self.args.model))
            self.report()
    def report(self):
        out={'budget_each':LIMIT,'models':{},'policy':POLICY}
        for m in MODELS:
            rows=[r for r in self.done.values() if r['model']==m and r['replicate']==0 and r['policy']==POLICY]
            groups={}
            for split in ('train','calibration','test'):
                for domain in ('math','coding','search','data_analysis'):
                    rr=[r for r in rows if r['split']==split and r['domain']==domain]
                    if rr:
                        solv=[r for r in rr if r['answerable']];uns=[r for r in rr if not r['answerable']]
                        groups[split+'/'+domain]={'n':len(rr),'solved':sum(r['correct'] for r in solv),'answerable':len(solv),'correct_abstentions':sum(r['correct'] for r in uns),'unanswerable':len(uns),'false_abstentions':sum(r['state']=='abstained' for r in solv),'cost':round(sum(r['actual_tokens'] for r in rr),1)}
            out['models'][m]={'spent_including_retries_and_conservative_errors':self.ledger.spent(m),'completed_core':len(rows),'breakdown':groups,'halted_this_process':m in self.halted}
        write_json(RUN/'summary.json',out);print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


def validate_suite():
    manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'));oracle=json.loads((SUITE/'oracle.json').read_text(encoding='utf-8'));errors=[]
    for r in manifest['tasks']:
        folder=SUITE/r['task_path'];workspace=folder/'workspace';g=oracle[r['id']]
        if r['domain']=='coding' and r['answerable']:
            public=(workspace/'test_mylib.py').read_text(encoding='utf-8')
            ref=check_code(g['reference_code'],public+'\n'+'\n'.join(g['hidden_tests']),g.get('test_imports'))
            bad=check_code((workspace/'mylib.py').read_text(encoding='utf-8'),public)
            if not ref['passed'] or bad['passed']:errors.append({'id':r['id'],'reference':ref,'broken':bad})
        elif r['domain']=='data_analysis' and r['answerable']:
            query=g['gold_sql']
            for p in g['gold_params']:
                literal=str(p) if isinstance(p,(int,float)) else "'"+str(p).replace("'","''")+"'"
                query=query.replace('?',literal,1)
            actual,error=execute_sql(query,workspace)
            if error or not (numeric_equal(actual,g['answer']) or str(actual)==str(g['answer'])):errors.append({'id':r['id'],'actual':actual,'gold':g['answer'],'error':error})
        elif r['domain']=='math' and not r['answerable']:
            if g['witness']['answers'][0]==g['witness']['answers'][1]:errors.append({'id':r['id'],'error':'invalid witness'})
        print('validated '+r['id'],flush=True)
    write_json(SUITE/'validation.json',{'total':len(manifest['tasks']),'errors':errors,'passed':not errors,'manifest_sha256':hashlib.sha256((SUITE/'manifest.json').read_bytes()).hexdigest()})
    print(json.dumps({'total':len(manifest['tasks']),'errors':errors},ensure_ascii=False))
    return int(bool(errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','train','calibration','test','full'],default='pilot');parser.add_argument('--model',choices=MODELS);parser.add_argument('--validate-only',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.validate_only:return validate_suite()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'process.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    exp=Experiment(args)
    if args.report_only:exp.report();return 0
    valid=json.loads((SUITE/'validation.json').read_text(encoding='utf-8'))
    if not valid['passed']:raise RuntimeError('Dataset validation failed')
    if valid.get('manifest_sha256')!=hashlib.sha256((SUITE/'manifest.json').read_bytes()).hexdigest():raise RuntimeError('Manifest changed after validation')
    if not os.environ.get('LLM_API_KEY'):raise RuntimeError('LLM_API_KEY environment variable is required')
    asyncio.run(exp.run());return 0
if __name__=='__main__':raise SystemExit(main())
