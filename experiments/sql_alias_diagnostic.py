"""Training-only counterfactual: tolerate natural CSV table names in model SQL."""
import json,re
from pathlib import Path
import analyze as a
import run_experiments as runner
calls=[c for c in a.read_jsonl(a.RUN/'calls.jsonl') if c.get('event')=='completed']
by_call={(c['model'],c['task'],c['latency']):c for c in calls}
manifest={r['id']:r for r in json.loads((a.SUITE/'manifest.json').read_text(encoding='utf-8'))['tasks']}
oracle=json.loads((a.SUITE/'oracle.json').read_text(encoding='utf-8'))
results=[]
for r in a.all_rows():
 if r['split']!='train' or r['domain']!='data_analysis' or not r['answerable'] or not r['calls']:continue
 c=by_call.get((r['model'],r['id'],r['calls'][0].get('latency')))
 if not c:continue
 query=runner.clean_fence(c['content'])
 normalized=re.sub(r'\bFROM\s+(?:sales|main)\b','FROM data',query,flags=re.I)
 workspace=a.SUITE/manifest[r['id']]['task_path']/'workspace'
 value,error=runner.execute_sql(normalized,workspace);gold=oracle[r['id']]['answer']
 correct=error is None and (runner.numeric_equal(value,gold) or str(value).strip().casefold()==str(gold).strip().casefold())
 results.append({'id':r['id'],'model':r['model'],'alias_changed':query!=normalized,'observed_final_correct':r['correct'],'normalized_first_correct':correct,
  'observed_tokens':r.get('provider_tokens',0),'first_response_tokens':c['cost'],'error':error})
summary={}
for m in runner.MODELS:
 rows=[r for r in results if r['model']==m]
 summary[m]={'n':len(rows),'alias_changes':sum(r['alias_changed'] for r in rows),'observed_final_correct':sum(r['observed_final_correct'] for r in rows),
  'normalized_first_correct':sum(r['normalized_first_correct'] for r in rows),
  'potential_saved_tokens_on_correct_first_queries':round(sum(r['observed_tokens']-r['first_response_tokens'] for r in rows if r['normalized_first_correct']),1)}
out={'scope':'Training-only offline replay of already recorded first responses. Only FROM sales/main is normalized to the sole data table. This is not a new live policy score; frozen held-out solver remains unchanged.','summary':summary,'rows':results}
runner.write_json(a.RUN/'sql_alias_diagnostic.json',out)
print(json.dumps(summary,indent=2))
