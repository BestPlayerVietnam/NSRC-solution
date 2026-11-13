"""Freeze predictors on train, calibrate once, and evaluate untouched test outcomes."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, statistics, subprocess, time
from pathlib import Path
import predictor as pred
import run_experiments as runner

RUN=runner.RUN;SUITE=runner.SUITE

def read_jsonl(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if path.exists() else []
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def all_rows():return list({r['key']:r for r in read_jsonl(RUN/'results.jsonl') if r.get('policy')==runner.POLICY and r['replicate']==0}.values())
def eligible(r):
    return (r['state'] not in ('budget_not_started','infrastructure_error','validator_error') and r.get('provider_tokens',0)>0
      and abs(r['actual_tokens']-r.get('provider_tokens',0))<.11
      and all(c.get('accounting')=='provider_reconciled' for c in r['calls']))
def freeze():
    dest=RUN/'predictor_train.json'
    if dest.exists():raise RuntimeError('Frozen predictor already exists; refusing overwrite')
    if any(r['split']!='train' for r in all_rows()):raise RuntimeError('Held-out observations already exist; refusing train tuning')
    rows=[r for r in all_rows() if r['split']=='train' and eligible(r)]
    cost,cv_cost=pred.cross_validate(rows,'cost');success,cv_success=pred.cross_validate(rows,'success')
    bundle={'version':1,'policy':runner.POLICY,'created':time.time(),'manifest_sha256':sha(SUITE/'manifest.json'),
      'feature_names':pred.FEATURE_NAMES,'n_train':len(rows),'training_keys':[r['key'] for r in rows],
      'training_rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
      'runner_sha256':sha(Path(runner.__file__)),'predictor_code_sha256':sha(Path(pred.__file__)),
      'docker_image_id':subprocess.check_output(['docker','image','inspect','nsrc-harness:latest','--format','{{.Id}}'],text=True).strip(),
      'cost':cost,'success':success,'cv_cost':cv_cost,'cv_success':cv_success,
      'scope':'Expected cost of the frozen bounded two-call policy, not the cost of guaranteed success. Success means correct answer on an answerable task; abstentions earn no contest reward.'}
    runner.write_json(dest,bundle)
    print(json.dumps({'frozen':str(dest),'n_train':len(rows),'cv_cost':cv_cost,'cv_success':cv_success},indent=2))

def select_panel():
    path=RUN/'evaluation_panel.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    if any(r['split']!='train' for r in all_rows()):raise RuntimeError('Cannot choose panel after held-out outcomes')
    manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'));panel={}
    for split in ('calibration','test'):
        selected=[]
        for d in pred.DOMAINS:
            for answerable,count in ((True,4),(False,1)):
                pool=[r for r in manifest['tasks'] if r['split']==split and r['domain']==d and r['answerable']==answerable]
                pool.sort(key=lambda r:hashlib.sha256(('primary-panel-v1:'+r['id']).encode()).hexdigest())
                selected.extend(r['id'] for r in pool[:count])
        selected.sort(key=lambda x:hashlib.sha256(('panel-order-v1:'+x).encode()).hexdigest());panel[split]=selected
    out={'created':time.time(),'seed':'primary-panel-v1','ids':panel,'description':'Fixed stratified primary panel: 20 per split, 5 per domain including one unanswerable. Chosen before held-out calls. Extra samples are secondary.'}
    runner.write_json(path,out);return out

def plan(split):
    path=RUN/(split+'_predictions.jsonl')
    if path.exists():raise RuntimeError('Predictions already frozen: '+str(path))
    model_path=RUN/('predictor_train.json' if split=='calibration' else 'predictor_calibrated.json')
    bundle=json.loads(model_path.read_text(encoding='utf-8'))
    if bundle['predictor_code_sha256']!=sha(Path(pred.__file__)):raise RuntimeError('Predictor implementation changed after freeze')
    if bundle['runner_sha256']!=sha(Path(runner.__file__)):raise RuntimeError('Solver policy changed after freeze')
    panel=select_panel();manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'))
    if any(r['split']==split for r in all_rows()):raise RuntimeError('Cannot create pre-run predictions after observations')
    if bundle['manifest_sha256']!=sha(SUITE/'manifest.json'):raise RuntimeError('Manifest changed')
    result=[]
    for r in manifest['tasks']:
        if r['split']!=split:continue
        folder=SUITE/r['task_path'];task=json.loads((folder/'task.json').read_text(encoding='utf-8'));f=runner.features(task,folder/'workspace')
        for model in pred.MODELS:
            row={'id':r['id'],'domain':r['domain'],'model':model,'features':f}
            result.append({**row,'policy':runner.POLICY,'split':split,'primary':r['id'] in panel['ids'][split],'prediction':pred.predict_all(bundle,row),'artifact_sha256':sha(model_path),'created':time.time()})
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    print(json.dumps({'planned_split':split,'n':len(result),'artifact_sha256':sha(model_path)},indent=2))
    if split=='test':
        import replay
        replay.make_plan()

def calibrate():
    dest=RUN/'predictor_calibrated.json'
    if dest.exists():raise RuntimeError('Calibration already frozen')
    if any(r['split']=='test' for r in all_rows()):raise RuntimeError('Test outcomes already exist')
    bundle=json.loads((RUN/'predictor_train.json').read_text(encoding='utf-8'));panel=select_panel()
    rows=[r for r in all_rows() if r['split']=='calibration' and r['id'] in panel['ids']['calibration'] and eligible(r)]
    report={'method':'Per-model one-sided relative residual, maxima within source group; finite quantile ceil((G+1)*0.9). Exchangeability is not established for budget-selected subsets.','models':{}}
    for model in pred.MODELS:
        groups=collections.defaultdict(list);selected=[r for r in rows if r['model']==model]
        for r in selected:
            p=pred.predict(bundle['cost'],r);groups[r['group']].append((r['provider_tokens']-p)/max(1,p))
        scores=sorted(max(v) for v in groups.values());k=math.ceil((len(scores)+1)*.9)
        q=max(0,scores[k-1]) if 0<k<=len(scores) else None
        report['models'][model]={'n':len(selected),'groups':len(groups),'q90':q,'complete_primary':len(selected)==len(panel['ids']['calibration'])}
    bundle['calibration']=report;bundle['calibration_keys']=[r['key'] for r in rows];bundle['train_artifact_sha256']=sha(RUN/'predictor_train.json')
    runner.write_json(dest,bundle);print(json.dumps(report,indent=2))

def evaluate(rows,predictions):
    actual=[];estimated=[];baseline=[];ys=[];ps=[];covered=[];uppers=[]
    for r in rows:
        if not eligible(r):continue
        p=predictions.get((r['id'],r['model']))
        if p is None:continue
        p=p['prediction'];a=r['provider_tokens'];actual.append(a);estimated.append(p['expected_tokens']);baseline.append(r['predicted_tokens']);ys.append(int(r['answerable'] and r['correct']));ps.append(p['p_solve'])
        if p.get('upper90') is not None:covered.append(a<=p['upper90']);uppers.append(p['upper90'])
    return {'cost':pred.cost_metrics(actual,estimated),'heuristic_cost':pred.cost_metrics(actual,baseline),'success':pred.success_metrics(ys,ps),
      'upper90':{'n':len(covered),'coverage':sum(covered)/len(covered) if covered else None,'mean_bound':sum(uppers)/len(uppers) if uppers else None}}

def operational(rows,expected):
    valid=[r for r in rows if r['state'] not in ('infrastructure_error','validator_error','budget_not_started')]
    solv=[r for r in valid if r['answerable']];uns=[r for r in valid if not r['answerable']]
    return {'expected':expected,'attempted':len(rows),'valid_outcomes':len(valid),'not_completed':expected-len(valid),
      'solved':sum(r['correct'] for r in solv),'answerable':len(solv),'correct_abstentions':sum(r['state']=='abstained' for r in uns),
      'unanswerable':len(uns),'false_abstentions':sum(r['state']=='abstained' for r in solv),
      'earned_reward':sum(r['reward'] for r in solv if r['correct']),'max_observed_reward':sum(r['reward'] for r in solv),
      'provider_tokens_observed':round(sum(r.get('provider_tokens',0) for r in rows),1),
      'charged_tokens':round(sum(r['actual_tokens'] for r in rows),1),'length_stops':sum(any(c.get('finish_reason')=='length' for c in r['calls']) for r in rows),
      'states':dict(collections.Counter(r['state'] for r in rows))}

def report():
    rows=all_rows();ledger=json.loads((RUN/'ledger.json').read_text(encoding='utf-8'));panel=select_panel();manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'))
    predictions={(p['id'],p['model']):p for split in ('calibration','test') for p in read_jsonl(RUN/(split+'_predictions.jsonl'))}
    out={'policy':runner.POLICY,'budget_each':runner.LIMIT,'primary_panel':panel,'models':{},'limitations':[
      'Public benchmarks may overlap model training data; this is not contamination-free evaluation.',
      'Only 20 primary held-out tasks per split. Repeated model responses are not independent tasks.',
      'Synthetic traps and MBPP reconstruction/single-paragraph QA simplify real contest workflows.',
      'Predictor estimates cost of this bounded policy, not cost to solve an arbitrary task.',
      'Budget-limited omissions are reported; completed-subset metrics can have selection bias.',
      'Unknown-usage errors and interrupted calls remain fully charged to experiment budgets.',
      'Calibration bounds are empirical; distribution shift and family dependence limit guarantees.']}
    completed_calls=[x for x in read_jsonl(RUN/'calls.jsonl') if x.get('event')=='completed']
    errors=[x for x in read_jsonl(RUN/'calls.jsonl') if x.get('event')=='error']
    for model in pred.MODELS:
        rec=ledger['models'][model];charged=rec['charged_units']/10;rr=[r for r in rows if r['model']==model]
        info={'charged_tokens_all_policies':charged,'remaining':runner.LIMIT-charged,
          'provider_reconciled_all_policies':round(sum(c['cost'] for c in completed_calls if c['model']==model and c.get('accounting')=='provider_reconciled'),1),
          'pending_reserved_tokens':sum(rec['pending'].values())/10,'error_calls':sum(c['model']==model for c in errors),'splits':{}}
        for split in ('train','calibration','test'):
            selected=[r for r in rr if r['split']==split];n=sum(r['split']==split for r in manifest['tasks'])
            s={'all':operational(selected,n)}
            if split!='train':
                primary=[r for r in selected if r['id'] in panel['ids'][split]]
                s.update(primary=operational(primary,len(panel['ids'][split])),prediction=evaluate(primary,predictions),secondary_all_predictions=evaluate(selected,predictions))
                s['domains']={d:operational([r for r in primary if r['domain']==d],5) for d in pred.DOMAINS}
            info['splits'][split]=s
        out['models'][model]=info
    # Fixed primary test panel is the main comparison; this intersection is explicitly secondary.
    shared=set(panel['ids']['test'])
    for model in pred.MODELS:shared&={r['id'] for r in rows if r['model']==model and r['split']=='test' and eligible(r)}
    out['secondary_common_test']={'ids':sorted(shared),'n_tasks':len(shared),'models':{m:operational([r for r in rows if r['split']=='test' and r['model']==m and r['id'] in shared],len(shared)) for m in pred.MODELS}}
    review_path=RUN/'answerability_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text(encoding='utf-8'));exceptions=review['exceptions']
        out['source_answerability_diagnostic']={'review':review,'models':{}}
        for model in pred.MODELS:
            counts={}
            for label in ('determinate','convention_dependent','disputed'):
                selected=[r for r in rows if r['model']==model and r['id'] in review['reviewed_ids'] and exceptions.get(r['id'],{}).get('status','determinate')==label]
                counts[label]={'n':len(selected),'official_matches':sum(r['correct'] for r in selected),'refusals':sum(r['state']=='abstained' for r in selected)}
            out['source_answerability_diagnostic']['models'][model]=counts
        out['limitations'].append('Some GSM8K texts depend on conventions or ambiguous wording. Official labels remain unchanged; a separate blinded source review flags these cases and does not automatically credit refusals.')
    runner.write_json(RUN/'evaluation.json',out)
    lines=['# Результат ограниченного эксперимента NSRC','',f'Политика: `{runner.POLICY}`. Лимит — 100 000 приведённых токенов на модель, до 400 000 всего.','',
      '| Модель | Учтено, токенов | Primary test выполнено | Решено выполнимых | Пропущено невыполнимых | MAE прогноза | MAE эвристики |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model,info in out['models'].items():
        s=info['splits']['test'];o=s['primary'];e=s['prediction']
        lines.append(f"| {model} | {info['charged_tokens_all_policies']:.1f} | {o['valid_outcomes']}/20 | {o['solved']}/{o['answerable']} | {o['correct_abstentions']}/{o['unanswerable']} | {e['cost'].get('mae',float('nan')):.1f} | {e['heuristic_cost'].get('mae',float('nan')):.1f} |")
    lines+=['','Правильный пропуск не приносит конкурсных баллов. Неполные вызовы и неизвестный usage не считаются точными метками стоимости. Полная детализация — в evaluation.json.','', '## Ограничения','']+['- '+x for x in out['limitations']]
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({m:{'spent':x['charged_tokens_all_policies'],'remaining':x['remaining'],'test':x['splits']['test']['primary'],'prediction':x['splits']['test']['prediction']} for m,x in out['models'].items()},ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['panel','freeze','plan-calibration','calibrate','plan-test','report']);a=p.parse_args().action
    lock=None
    if a not in ('panel','report') and os.name=='nt':
        import msvcrt
        lock=(RUN/'process.lock').open('a+b');lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if a=='panel':print(json.dumps(select_panel(),indent=2))
    elif a=='freeze':freeze()
    elif a.startswith('plan-'):plan(a[5:])
    elif a=='calibrate':calibrate()
    else:report()
if __name__=='__main__':main()
