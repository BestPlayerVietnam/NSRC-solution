"""Checks for leakage, prediction portability, grouping, and probability bounds."""
import copy,json,math
import predictor as p
import analyze as a
rows=[r for r in a.all_rows() if r['split']=='train' and a.eligible(r)]
assert len(rows)>=40
for kind in ('cost','success'):
 for method in ('group_mean','ridge' if kind=='cost' else 'logistic'):
  fitted=p.fit(rows,kind,method,10)
  restored=json.loads(json.dumps(fitted))
  for r in rows[:20]:
   value=p.predict(restored,r);assert math.isfinite(value)
   assert (value>=1 if kind=='cost' else 0<value<1)
   mutated=copy.deepcopy(r)
   for k in ('answerable','correct','reward','answer','state','provider_tokens','actual_tokens','calls','id','group'):
    mutated[k]='DO NOT USE AFTER-RUN TARGET'
   assert p.predict(restored,mutated)==value
   assert p.vector(restored_row:=r)==p.vector(mutated)
# All models/replicas of a source family stay together in cross-validation.
groups={}
for r in rows:
 groups.setdefault(r['group'],set()).add(p.fold(r))
assert all(len(v)==1 for v in groups.values())
manifest=json.loads((a.SUITE/'manifest.json').read_text(encoding='utf-8'))
by_split={s:{r['group'] for r in manifest['tasks'] if r['split']==s} for s in ('train','calibration','test')}
assert not by_split['train']&by_split['calibration'] and not by_split['train']&by_split['test'] and not by_split['calibration']&by_split['test']
panel=a.select_panel()
for s,ids in panel['ids'].items():
 assert len(ids)==20 and len(set(ids))==20
 for d in p.DOMAINS:
  selected=[r for r in manifest['tasks'] if r['id'] in ids and r['domain']==d]
  assert len(selected)==5 and sum(r['answerable'] for r in selected)==4
print('Predictor leakage, grouping, panel balance, and serialization checks passed; training observations:',len(rows))
