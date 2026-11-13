"""Small classical predictors. Training needs NumPy; inference only stdlib."""
from __future__ import annotations
import hashlib, math

MODELS=('qwen3.6-35b-a3b-noreason','qwen3.8-27b-noreason','qwen3.6-35b-a3b','qwen3.8-27b')
DOMAINS=('math','coding','search','data_analysis')
NUMERIC=('instruction_chars','instruction_words','file_count','bytes','ast_nodes','asserts','csv_rows','csv_cols','kb_docs')
FEATURE_NAMES=([f'domain:{d}' for d in DOMAINS]+[f'model:{m}' for m in MODELS]+
 [f'interaction:{m}:{d}' for m in MODELS for d in DOMAINS]+[f'log1p:{k}' for k in NUMERIC]+['log1p:cap'])

def initial_cap(model,domain):return 1536 if not model.endswith('-noreason') else (768 if domain=='coding' else 384)
def vector(row):
    f=row['features'];m=row['model'];d=row['domain']
    return ([float(d==x) for x in DOMAINS]+[float(m==x) for x in MODELS]+
      [float(m==a and d==b) for a in MODELS for b in DOMAINS]+
      [math.log1p(max(0,float(f.get(k,0)))) for k in NUMERIC]+[math.log1p(initial_cap(m,d))])
def fold(row):return int(hashlib.sha256(row['group'].encode()).hexdigest()[:8],16)%5
def sigmoid(x):return 1/(1+math.exp(-max(-35,min(35,x))))
def target(row,kind):return row['provider_tokens'] if kind=='cost' else float(row['answerable'] and row['correct'])
def fit(rows,kind,method,regularization):
    import numpy as np
    y=np.array([target(r,kind) for r in rows],float)
    if not len(y):raise ValueError('No eligible training observations')
    if method=='group_mean':
        global_mean=float(y.mean());by_model={};by_group={}
        for m in MODELS:
            vals=[target(r,kind) for r in rows if r['model']==m]
            mean=(sum(vals)+regularization*global_mean)/(len(vals)+regularization)
            by_model[m]=mean
            for d in DOMAINS:
                vals=[target(r,kind) for r in rows if r['model']==m and r['domain']==d]
                by_group[m+'|'+d]=(sum(vals)+regularization*mean)/(len(vals)+regularization)
        return {'kind':kind,'method':method,'regularization':regularization,'by_model':by_model,'by_group':by_group,'global':global_mean}
    raw=np.array([vector(r) for r in rows]);mean=raw.mean(axis=0);scale=raw.std(axis=0);scale[scale<1e-8]=1
    X=np.column_stack([np.ones(len(raw)),(raw-mean)/scale]);pen=np.eye(X.shape[1])*regularization;pen[0,0]=0
    if kind=='cost':
        coefficients=np.linalg.solve(X.T@X+pen,X.T@y)
    else:
        coefficients=np.zeros(X.shape[1]);p0=float(np.clip(y.mean(),1e-3,1-1e-3));coefficients[0]=math.log(p0/(1-p0))
        def objective(w):
            z=X@w
            return float(np.sum(np.logaddexp(0,z)-y*z)+.5*w@pen@w)
        for _ in range(70):
            z=np.clip(X@coefficients,-35,35);p=1/(1+np.exp(-z));weight=np.maximum(p*(1-p),1e-7)
            gradient=X.T@(p-y)+pen@coefficients
            hessian=X.T@(X*weight[:,None])+pen+np.eye(X.shape[1])*1e-8
            step=np.linalg.solve(hessian,gradient);rate=1.;loss=objective(coefficients)
            while rate>1e-5 and objective(coefficients-rate*step)>loss:rate*=.5
            coefficients-=rate*step
            if np.max(np.abs(rate*step))<1e-7:break
    return {'kind':kind,'method':method,'regularization':regularization,'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist()}
def predict(model,row):
    if model['method']=='group_mean':value=model['by_group'].get(row['model']+'|'+row['domain'],model['by_model'].get(row['model'],model['global']))
    else:
        x=[1.]+[(v-a)/b for v,a,b in zip(vector(row),model['mean'],model['scale'])]
        value=sum(a*b for a,b in zip(x,model['coefficients']))
        if model['kind']=='success':value=sigmoid(value)
    return max(1.,value) if model['kind']=='cost' else max(.005,min(.995,value))
def cost_metrics(actual,pred):
    n=len(actual)
    if not n:return {'n':0}
    return {'n':n,'mae':sum(abs(a-p) for a,p in zip(actual,pred))/n,'wape':sum(abs(a-p) for a,p in zip(actual,pred))/max(1,sum(actual)),
      'bias':sum(p-a for a,p in zip(actual,pred))/n,'actual_mean':sum(actual)/n,'predicted_mean':sum(pred)/n}
def success_metrics(actual,pred):
    n=len(actual)
    if not n:return {'n':0}
    return {'n':n,'brier':sum((a-p)**2 for a,p in zip(actual,pred))/n,
      'log_loss':-sum(a*math.log(p)+(1-a)*math.log(1-p) for a,p in zip(actual,pred))/n,'actual_rate':sum(actual)/n,'predicted_rate':sum(pred)/n}
def cross_validate(rows,kind):
    candidates=[('group_mean',2),('group_mean',8)]+[('ridge' if kind=='cost' else 'logistic',v) for v in (1,10,100)]
    reports=[]
    for method,reg in candidates:
        records=[]
        for k in range(5):
            training=[r for r in rows if fold(r)!=k];validation=[r for r in rows if fold(r)==k]
            if not training or not validation:continue
            fitted=fit(training,kind,method,reg)
            records.extend((r,target(r,kind),predict(fitted,r)) for r in validation)
        model_scores=[]
        for m in MODELS:
            pairs=[(y,p) for r,y,p in records if r['model']==m]
            if not pairs:continue
            metric=(cost_metrics if kind=='cost' else success_metrics)([y for y,p in pairs],[p for y,p in pairs])
            model_scores.append(metric['wape' if kind=='cost' else 'log_loss'])
        reports.append({'method':method,'regularization':reg,'cv_score':sum(model_scores)/len(model_scores),'n':len(records)})
    best=min(reports,key=lambda x:x['cv_score'])
    return fit(rows,kind,best['method'],best['regularization']),reports

def predict_all(bundle,row):
    cost=predict(bundle['cost'],row);success=predict(bundle['success'],row)
    out={'expected_tokens':cost,'p_solve':success,'policy':bundle['policy']}
    if 'calibration' in bundle:
        q=bundle['calibration']['models'].get(row['model'],{}).get('q90')
        out['upper90']=None if q is None else max(0,cost+q*max(1,cost))
    return out
