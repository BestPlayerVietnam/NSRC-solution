"""Offline multiple-choice routing: decisions freeze before test outcomes exist."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments'/'runs'/'2026-11-12-v1';SUITE=ROOT/'experiments'/'suite'
MODELS=('qwen3.6-35b-a3b-noreason','qwen3.8-27b-noreason','qwen3.6-35b-a3b','qwen3.8-27b')
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x] if path.exists() else []
def choose(options,budget,quantum=10):
    # One action per task: skip, or one measured frozen solver configuration.
    slots=int(budget//quantum);states={0:(0.,[])}
    for task_id,choices in options:
        next_states={}
        for used,(value,path) in states.items():
            for c in [None]+choices:
                weight=0 if c is None else max(1,math.ceil(c['selection_cost']/quantum))
                total=used+weight
                if total>slots:continue
                score=value+(0 if c is None else c['expected_reward'])
                previous=next_states.get(total)
                if previous is None or score>previous[0]:next_states[total]=(score,path+[(task_id,c)])
        # Pareto prune dominated budget states without looking at actual outcomes.
        states={};best=-1.
        for used,item in sorted(next_states.items()):
            if item[0]>best+1e-12:states[used]=item;best=item[0]
    used,(value,path)=max(states.items(),key=lambda pair:(pair[1][0],-pair[0]))
    return {'quantum':quantum,'budget':budget,'rounded_prediction_budget':used*quantum,'expected_reward':value,
      'actions':[{'id':i,'action':'skip'} if c is None else {'id':i,'action':'solve',**c} for i,c in path]}
def make_plan():
    target=RUN/'routing_plan.json'
    if target.exists():raise RuntimeError('Routing plan already frozen')
    if any(r['split']=='test' for r in lines(RUN/'results.jsonl')):raise RuntimeError('Cannot plan routing after test observations')
    predictions=lines(RUN/'test_predictions.jsonl');panel=json.loads((RUN/'evaluation_panel.json').read_text())['ids']['test']
    manifest=json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'));reward={}
    for r in manifest['tasks']:
        if r['id'] in panel:reward[r['id']]=json.loads((SUITE/r['task_path']/'task.json').read_text(encoding='utf-8'))['reward']
    plans={}
    for risk in ('expected','upper90'):
        options=[]
        for task_id in panel:
            choices=[]
            for r in predictions:
                if r['id']!=task_id:continue
                p=r['prediction'];cost=p['expected_tokens'] if risk=='expected' else p.get('upper90')
                if cost is None:continue
                choices.append({'model':r['model'],'selection_cost':cost,'expected_tokens':p['expected_tokens'],'p_solve':p['p_solve'],'expected_reward':reward[task_id]*p['p_solve']})
            options.append((task_id,choices))
        plans[risk]=choose(options,1000*len(panel))
    out={'pre_run':True,'scope':'Offline replay on 20 fixed primary tasks, budget scaled to 1000 tokens/task. No conditional model escalation or fabricated unobserved trajectories.',
      'prediction_sha256':hashlib.sha256((RUN/'test_predictions.jsonl').read_bytes()).hexdigest(),'plans':plans}
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'routing_plans_frozen':{k:{'selected':sum(a['action']=='solve' for a in p['actions']),'expected_reward':p['expected_reward'],'predicted_cost':p['rounded_prediction_budget']} for k,p in plans.items()}}))
def evaluate():
    frozen=json.loads((RUN/'routing_plan.json').read_text(encoding='utf-8'));panel=json.loads((RUN/'evaluation_panel.json').read_text())['ids']['test']
    rows={(r['id'],r['model']):r for r in lines(RUN/'results.jsonl') if r.get('policy')=='bounded-tool-math-v2' and r['replicate']==0 and r['split']=='test'}
    def score(actions,budget):
        chosen=[a for a in actions if a['action']=='solve'];known=[];missing=[]
        for a in chosen:
            r=rows.get((a['id'],a['model']))
            if not r or r['state'] in ('infrastructure_error','validator_error','budget_not_started'):missing.append(a)
            else:known.append(r)
        cost=sum(r.get('provider_tokens',0) for r in known);reward=sum(r['reward'] for r in known if r['answerable'] and r['correct'])
        return {'selected':len(chosen),'known_outcomes':len(known),'unknown_outcomes':len(missing),'model_counts':{m:sum(a['model']==m for a in chosen) for m in MODELS},
          'known_solved':sum(r['answerable'] and r['correct'] for r in known),'known_reward':reward,'known_cost':round(cost,1),
          'charged_cost_of_known_traces':round(sum(r['actual_tokens'] for r in known),1),
          'cost_exact_for_known_traces':all(abs(r['actual_tokens']-r.get('provider_tokens',0))<.11 for r in known),
          'budget':budget,'within_budget_on_known_traces':sum(r['actual_tokens'] for r in known)<=budget,
          'budget_valid_reward':reward if not missing and sum(r['actual_tokens'] for r in known)<=budget else None,
          'complete':not missing,'unknown_actions':missing}
    out={'scope':frozen['scope'],'routing':{k:score(p['actions'],p['budget']) for k,p in frozen['plans'].items()},
      'fixed_model_baselines':{m:score([{'id':i,'model':m,'action':'solve'} for i in panel],1000*len(panel)) for m in MODELS},
      'caution':'This reuses measured independent task trajectories. It is not an actual mixed-model 100-task contest run; retries, context reuse, wall time and online budget reservations are not simulated.'}
    # Exploratory sensitivity, kept separate from the two pre-run primary plans.
    predictions=lines(RUN/'test_predictions.jsonl')
    rewards={}
    for r in json.loads((SUITE/'manifest.json').read_text(encoding='utf-8'))['tasks']:
        if r['id'] in panel:rewards[r['id']]=json.loads((SUITE/r['task_path']/'task.json').read_text(encoding='utf-8'))['reward']
    sensitivity=[]
    for risk in ('expected','upper90'):
        options=[]
        for task_id in panel:
            choices=[]
            for item in predictions:
                if item['id']!=task_id:continue
                p=item['prediction'];cost=p['expected_tokens'] if risk=='expected' else p.get('upper90')
                if cost is not None:choices.append({'model':item['model'],'selection_cost':cost,'expected_reward':rewards[task_id]*p['p_solve']})
            options.append((task_id,choices))
        for budget in (1000,2000,5000,10000,20000):
            proposed=choose(options,budget)
            sensitivity.append({'risk':risk,'budget':budget,**score(proposed['actions'],budget)})
    out['exploratory_budget_sensitivity']={'pre_registered':False,'note':'Post-hoc descriptive budget grid using only frozen pre-run predictions for selection; no model or threshold refitted. Over-budget trace totals do not receive a valid reward.','results':sensitivity}
    (RUN/'routing_evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','evaluate']);a=p.parse_args()
    make_plan() if a.action=='plan' else evaluate()
