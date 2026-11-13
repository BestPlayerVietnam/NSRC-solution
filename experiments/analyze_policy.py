"""Fit on train, calibrate separately, and evaluate frozen forecasts on test."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policy_features import POLICY_ID, extract_features
from experiments.policy_experiment import charged_calls

RUN=ROOT/'experiments/runs/2026-11-12-policy'
SUITE=ROOT/'experiments/suite_policy'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(phase):
    folder=RUN/phase;e=read(folder/'evaluation.json')
    if not e['comparison_complete']:raise RuntimeError('Incomplete phase: '+phase)
    meta=read(SUITE/'research_metadata.json')
    manifest={r['id']:r for r in read(SUITE/'manifest.json')['tasks']}
    events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    calls=charged_calls(events)
    out=[]
    for arm,data in e['arms'].items():
        if e['protocol']['arms'][arm]['retry_policy']!='adaptive':continue
        initial=e['protocol']['arms'][arm]['initial_model']
        for row in data['rows']:
            cs=[c for c in calls if c['arm']==arm and c['task']==row['id']]
            if not cs:continue
            known=all(c['accounting']=='provider_reconciled' for c in cs)
            task_dir=SUITE/manifest[row['id']]['task_path']
            trace=row.get('trace') or {}
            if trace.get('policy_id')!=POLICY_ID:raise RuntimeError('Trace policy differs from fitted policy')
            features=trace.get('features') or extract_features(read(task_dir/'task.json'),task_dir/'workspace')
            group=meta[row['id']].get('group',meta[row['id']].get('family'))
            if group is None:raise RuntimeError('Missing source group')
            out.append({'id':row['id'],'task_group':str(group),'model':initial,
                'policy_id':POLICY_ID,'domain':row['domain'],'features':features,
                'source_type':'real_api','reward':row['reward'],
                'first_call_cost':cs[0]['cost'],'total_policy_cost':sum(c['cost'] for c in cs),
                'first_usage_known':cs[0]['accounting']=='provider_reconciled',
                'total_usage_known':known,'execution_complete':known and trace.get('execution_complete') is True,
                'retry_count':len(cs)-1,'first_success':bool(row['correct'] and len(cs)==1),
                'final_success':bool(row['correct']),'verifier_kind':'hidden_tests' if row['domain']=='coding' else 'oracle',
                'retry_executed':len(cs)>1,'answerable':row['answerable']})
    return out


def fit():
    from cost_model import fit_cost_artifact
    artifact=ROOT/'cost_artifact.json'
    if artifact.exists():raise RuntimeError('Refusing to overwrite frozen cost artifact')
    train=rows_for('train');calibration=rows_for('calibration')
    kwargs={'feature_names':sorted(train[0]['features']),'policy_id':POLICY_ID}
    signature=inspect.signature(fit_cost_artifact)
    if 'selection_metric' in signature.parameters:kwargs['selection_metric']='mae'
    elif 'criterion' in signature.parameters:kwargs['criterion']='mae'
    fitted=fit_cost_artifact(train,calibration,**kwargs)
    write(artifact,fitted)
    write(RUN/'training_rows.json',train);write(RUN/'calibration_rows.json',calibration)
    write(RUN/'fit_record.json',{'train_rows':len(train),'calibration_rows':len(calibration),
        'train_groups':len({r['task_group'] for r in train}),
        'calibration_groups':len({r['task_group'] for r in calibration}),
        'model_sha256':sha(artifact),'fit_kwargs':kwargs,
        'policy':POLICY_ID,'test_labels_read':False})
    print(json.dumps(read(RUN/'fit_record.json'),indent=2))


def errors(pairs):
    if not pairs:return {'n':0}
    from collections import defaultdict
    points=[pair[:2] for pair in pairs];n=len(points)
    ae=[abs(p-y) for p,y in points];denom=sum(abs(y) for p,y in points)
    groups=defaultdict(list)
    for pair in pairs:
        if len(pair)>2:groups[pair[2]].append(abs(pair[0]-pair[1]))
    return {'n':n,'mae':sum(ae)/n,'rmse':math.sqrt(sum((p-y)**2 for p,y in points)/n),
        'bias':sum(p-y for p,y in points)/n,'wape':sum(ae)/denom if denom else None,
        'underprediction_fraction':sum(p<y for p,y in points)/n,
        'group_count':len(groups),'group_macro_mae':statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None}


def report():
    test=read(RUN/'test/evaluation.json');protocol=test['protocol']
    if sha(ROOT/'cost_artifact.json')!=protocol['artifact_sha256']:raise RuntimeError('Artifact changed')
    if sha(RUN/'test/forecasts.json')!=protocol['forecasts_sha256']:raise RuntimeError('Forecasts changed')
    forecasts={(r['id'],r['model']):r for r in read(RUN/'test/forecasts.json')}
    measured=rows_for('test');metrics={}
    for model in sorted({r['model'] for r in measured}):
        scoped=[r for r in measured if r['model']==model]
        rows=[r for r in scoped if r['total_usage_known'] and r['execution_complete']]
        pairs={k:[] for k in ['heuristic','prompt_prior','learned_total','learned_first','deployed_total','deployed_first','online_total']}
        upper=[]
        arm=next(a for a,c in protocol['arms'].items() if c['initial_model']==model and c['retry_policy']=='adaptive')
        byid={r['id']:r for r in test['arms'][arm]['rows']}
        for row in rows:
            f=forecasts[(row['id'],model)];learned=f['forecast'];y=row['total_policy_cost']
            pairs['heuristic'].append((f['heuristic'],y,row['task_group']));pairs['prompt_prior'].append((f['fallback'][1],y,row['task_group']))
            if learned['total_policy_mean'] is not None:pairs['learned_total'].append((learned['total_policy_mean'],y,row['task_group']))
            deployed=f.get('deployed')
            if deployed:pairs['deployed_total'].append((deployed['total_policy'],y,row['task_group']))
            trace=byid[row['id']].get('trace') or {}
            if 'predicted_tokens' in trace:pairs['online_total'].append((trace['predicted_tokens'],y,row['task_group']))
            if learned.get('total_policy_upper') is not None:upper.append(y<=learned['total_policy_upper'])
        # A completed first call is usable even when a later call or the full
        # policy was interrupted. Never require a full-policy label here.
        for row in scoped:
            if not row['first_usage_known']:continue
            learned=forecasts[(row['id'],model)]['forecast']
            if learned['first_call_mean'] is not None:
                pairs['learned_first'].append((learned['first_call_mean'],row['first_call_cost'],row['task_group']))
            deployed=forecasts[(row['id'],model)].get('deployed')
            if deployed:pairs['deployed_first'].append((deployed['first_call'],row['first_call_cost'],row['task_group']))
        metrics[model]={k:errors(v) for k,v in pairs.items()}
        metrics[model]['empirical_upper_coverage']={'n':len(upper),'fraction':sum(upper)/len(upper) if upper else None}
        metrics[model]['label_audit']={'attempted':len(scoped),'complete_known_total':len(rows),
            'known_first':sum(r['first_usage_known'] for r in scoped),
            'excluded_total':len(scoped)-len(rows),
            'extrapolated':sum(bool(forecasts[(r['id'],model)]['forecast'].get('warnings')) for r in rows)}
    execution={}
    call_labels=charged_calls([json.loads(line) for line in (RUN/'test/calls.jsonl').read_text().splitlines()])
    for arm,data in test['arms'].items():
        rows=data['rows'];traps=[r for r in rows if not r['answerable']]
        retry=[r for r in rows if len((r.get('trace') or {}).get('calls',[]))>1]
        execution[arm]={k:v for k,v in data.items() if k!='rows'}
        calls_by_task={r['id']:[c for c in call_labels if c['arm']==arm and c['task']==r['id']] for r in rows}
        execution[arm].update(unanswerable_cost=sum(r['cost'] for r in traps),
            retry_tasks=len(retry),retry_solved=sum(r['correct'] for r in retry),
            continuation_cost=sum(c['cost'] for cs in calls_by_task.values() for c in cs[1:]),
            unanswerable_continuation_cost=sum(c['cost'] for r in traps for c in calls_by_task[r['id']][1:]),
            unknown_usage_held_cost=sum(c['cost'] for c in call_labels if c['arm']==arm and c['accounting']!='provider_reconciled'),
            hard_skips=sum((r.get('trace') or {}).get('outcome')=='impossible' for r in rows),
            unattempted=sum(r.get('trace') is None for r in rows),
            false_hard_skips=sum(r['answerable'] and (r.get('trace') or {}).get('outcome')=='impossible' for r in rows),
            by_domain={d:{'solved':sum(r['correct'] for r in rows if r['domain']==d),
                          'answerable':sum(r['answerable'] for r in rows if r['domain']==d),
                          'cost':sum(r['cost'] for r in rows if r['domain']==d)}
                       for d in sorted({r['domain'] for r in rows})})
    out={'complete':test['comparison_complete'],'prediction':metrics,'execution':execution,
         'test_tasks':len(protocol['task_ids']),'fit':read(RUN/'fit_record.json'),
         'budget':test['budget'],'limitations':['Synthetic family-disjoint test, not representative hidden final',
             'Each model/policy pipeline has an independent 100k budget; historical totals are separate',
             'End-to-end comparison changes solver policy; forecast comparison uses identical candidate trajectories']}
    write(RUN/'test_rows.json',measured)
    write(RUN/'final_report.json',out)
    print(json.dumps({'prediction':metrics,'execution':execution},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','report'])
    globals()[parser.parse_args().action]()
