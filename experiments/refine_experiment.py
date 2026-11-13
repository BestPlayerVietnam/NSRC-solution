"""Fresh held-out validation, sharing the original per-model budget ledger."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,math,msvcrt,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import analyze as original
import predictor as base
import predictor_refined as refined
import run_experiments as runner

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'experiments'/'runs'/'2026-11-12-v1'
RUN=ROOT/'experiments'/'runs'/'2026-11-12-refinement';SUITE=ROOT/'experiments'/'suite_fresh'
TARGETS=[m for m in base.MODELS if m!='qwen3.6-35b-a3b']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x] if p.exists() else []
def prepare():
 RUN.mkdir(parents=True,exist_ok=True)
 artifact=RUN/'predictor_calibrated.json'
 if artifact.exists():raise RuntimeError('Refined predictor already frozen')
 old=json.loads((OLD/'predictor_train.json').read_text(encoding='utf-8'))
 rows=original.all_rows();train=[r for r in rows if r['split']=='train' and original.eligible(r)]
 calibration_ids=json.loads((OLD/'evaluation_panel.json').read_text())['ids']['calibration']
 bundle={'version':2,'policy':runner.POLICY,'created':time.time(),'manifest_sha256':sha(SUITE/'manifest.json'),
  'training_manifest_sha256':sha(original.SUITE/'manifest.json'),'oracle_sha256':sha(SUITE/'oracle.json'),'runner_sha256':sha(Path(runner.__file__)),
  'wrapper_sha256':sha(Path(__file__)),'refined_code_sha256':sha(Path(refined.__file__)),
  'docker_image_id':subprocess.check_output(['docker','image','inspect','nsrc-harness:latest','--format','{{.Id}}'],text=True).strip(),
  'cost_models':{},'selection':{},'calibration':{},'success':old['success'],'old_cost':old['cost'],
  'budget_ledger':str(OLD/'ledger.json'),'note':'Candidates selected per model using only original train groups. First test prompted this refinement; all evaluation here uses new source groups. No new calls to qwen3.6 reasoning.'}
 for model in TARGETS:
  fitted,scores=refined.select([r for r in train if r['model']==model]);bundle['cost_models'][model]=fitted;bundle['selection'][model]=scores
  cal=[r for r in rows if r['model']==model and r['split']=='calibration' and r['id'] in calibration_ids and original.eligible(r)]
  grouped={}
  for r in cal:
   p=refined.predict(fitted,r);grouped.setdefault(r['group'],[]).append((r['provider_tokens']-p)/max(1,p))
  residual=sorted(max(v) for v in grouped.values());k=math.ceil((len(residual)+1)*.9)
  bundle['calibration'][model]={'n':len(cal),'groups':len(residual),'q90':max(0,residual[k-1]) if 0<k<=len(residual) else None}
 runner.write_json(artifact,bundle)
 manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'));predictions=[]
 for r in manifest['tasks']:
  folder=SUITE/r['task_path'];task=json.loads((folder/'task.json').read_text(encoding='utf-8'));f=runner.features(task,folder/'workspace')
  for model in TARGETS:
   row={'id':r['id'],'model':model,'domain':r['domain'],'features':f};cost=refined.predict(bundle['cost_models'][model],row);q=bundle['calibration'][model]['q90']
   predictions.append({**row,'policy':runner.POLICY,'split':'test','primary':True,'created':time.time(),'artifact_sha256':sha(artifact),
    'prediction':{'expected_tokens':cost,'upper90':None if q is None else cost+q*max(1,cost),'p_solve':base.predict(bundle['success'],row),'old_expected_tokens':base.predict(bundle['old_cost'],row)}})
 (RUN/'test_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions),encoding='utf-8')
 ids=[r['id'] for r in manifest['tasks']];ids.sort(key=lambda i:hashlib.sha256(('fresh-order-v1'+i).encode()).hexdigest())
 runner.write_json(RUN/'evaluation_panel.json',{'ids':{'test':ids}})
 print(json.dumps({'new_tasks':len(ids),'models':TARGETS,'selection':{m:min(bundle['selection'][m],key=lambda x:x['mae']) for m in TARGETS}},indent=2))
def configure():runner.RUN=RUN;runner.SUITE=SUITE
async def run():
 bundle=json.loads((RUN/'predictor_calibrated.json').read_text(encoding='utf-8'))
 if bundle['oracle_sha256']!=sha(SUITE/'oracle.json'):raise RuntimeError('Fresh oracle changed after freeze')
 if bundle['wrapper_sha256']!=sha(Path(__file__)) or bundle['refined_code_sha256']!=sha(Path(refined.__file__)):raise RuntimeError('Refinement code changed after freeze')
 validation=json.loads((SUITE/'validation.json').read_text(encoding='utf-8'))
 if not validation['passed'] or validation['manifest_sha256']!=sha(SUITE/'manifest.json'):raise RuntimeError('Fresh suite not validated')
 configure();exp=runner.Experiment(SimpleNamespace(phase='test',model=None))
 # Crucial: no new budget is created for this experiment folder.
 exp.ledger=runner.Ledger(OLD/'ledger.json')
 if (RUN/'ledger.json').exists():raise RuntimeError('Unexpected independent ledger')
 await asyncio.gather(*(exp.worker(m) for m in TARGETS));exp.report()
def report():
 rows={r['key']:r for r in read_lines(RUN/'results.jsonl')};predictions={(r['id'],r['model']):r['prediction'] for r in read_lines(RUN/'test_predictions.jsonl')}
 ledger=json.loads((OLD/'ledger.json').read_text());out={'suite':'fresh-40','models':{},'cost_scope':'Per-model ledger includes BOTH experiment rounds and conservative unknown-usage reservations.'}
 for model in TARGETS:
  rr=[r for r in rows.values() if r['model']==model];valid=[r for r in rr if original.eligible(r)]
  actual=[r['provider_tokens'] for r in valid];new=[predictions[(r['id'],model)]['expected_tokens'] for r in valid];old=[predictions[(r['id'],model)]['old_expected_tokens'] for r in valid]
  covered=[r['provider_tokens']<=predictions[(r['id'],model)]['upper90'] for r in valid if predictions[(r['id'],model)]['upper90'] is not None]
  out['models'][model]={'operational':original.operational(rr,40),'refined_cost':base.cost_metrics(actual,new),'previous_cost':base.cost_metrics(actual,old),
   'heuristic_cost':base.cost_metrics(actual,[r['predicted_tokens'] for r in valid]),'upper90_coverage':sum(covered)/len(covered) if covered else None,
   'charged_total':ledger['models'][model]['charged_units']/10,'remaining':runner.LIMIT-ledger['models'][model]['charged_units']/10}
 runner.write_json(RUN/'evaluation.json',out);print(json.dumps(out,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','prepare','run','report']);args=p.parse_args()
 RUN.mkdir(parents=True,exist_ok=True)
 with (OLD/'process.lock').open('a+b') as lock:
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  if args.action=='validate':configure();raise SystemExit(runner.validate_suite())
  if args.action=='prepare':prepare()
  elif args.action=='run':
   if not os.environ.get('LLM_API_KEY'):raise RuntimeError('LLM_API_KEY required')
   asyncio.run(run())
  else:report()
if __name__=='__main__':main()
