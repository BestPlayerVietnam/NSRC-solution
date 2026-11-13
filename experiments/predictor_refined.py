"""Model-specific cost refinements chosen on grouped training folds only."""
import copy,math
import predictor as base

def heuristic(row):
 f=row['features'];reason=not row['model'].endswith('-noreason')
 return (30+f['instruction_chars']/40+f['bytes']/40+(70 if row['domain']=='coding' else 0))*(3 if reason else 1)
def fit(rows,method,reg):
 if method=='heuristic':return {'method':method}
 if method=='log_ridge_smearing':
  transformed=[{**r,'provider_tokens':math.log1p(r['provider_tokens'])} for r in rows]
  inner=base.fit(transformed,'cost','ridge',reg)
  residual=[math.exp(math.log1p(r['provider_tokens'])-base.predict(inner,r)) for r in rows]
  return {'method':method,'inner':inner,'smearing':sum(residual)/len(residual),'regularization':reg}
 return {'method':'base','inner':base.fit(rows,'cost',method,reg),'regularization':reg}
def predict(model,row):
 if model['method']=='heuristic':return heuristic(row)
 if model['method']=='log_ridge_smearing':return max(1,math.exp(min(30,base.predict(model['inner'],row)))*model['smearing']-1)
 return base.predict(model['inner'],row)
def select(rows):
 candidates=[('heuristic',0),('group_mean',2),('group_mean',8)]+[(method,reg) for method in ('ridge','log_ridge_smearing') for reg in (1,10,100)]
 scores=[]
 for method,reg in candidates:
  pairs=[]
  for fold in range(5):
   train=[r for r in rows if base.fold(r)!=fold];held=[r for r in rows if base.fold(r)==fold]
   if not train or not held:continue
   fitted=fit(train,method,reg);pairs.extend((r['provider_tokens'],predict(fitted,r)) for r in held)
  metric=base.cost_metrics([a for a,p in pairs],[p for a,p in pairs])
  scores.append({'method':method,'regularization':reg,**metric})
 best=min(scores,key=lambda s:s['mae'])
 return fit(rows,best['method'],best['regularization']),scores
