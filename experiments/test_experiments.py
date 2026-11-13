"""Offline budget/accounting regression checks. No live API calls."""
import importlib.util,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('run_experiments',Path(__file__).with_name('run_experiments.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'ledger.json';ledger=m.Ledger(p,limit=1000);model=m.MODELS[0]
 call,amount=ledger.reserve(model,100,'hello')
 cost,source,breach=ledger.reconcile(model,call,amount,{'input_tokens':100,'output_tokens':200,'reasoning_tokens':150,'total_tokens':300})
 assert cost==210 and source=='provider_reconciled'
 call,amount=ledger.reserve(model,100,'hello')
 cost,_,_=ledger.reconcile(model,call,amount,{'input_tokens':100,'output_tokens':20,'reasoning_tokens':150,'total_tokens':270})
 assert cost==180
 call,amount=ledger.reserve(model,100,'hello');held=ledger.spent(model)
 assert m.Ledger(p,limit=1000).spent(model)==held
 cost,source,_=ledger.reconcile(model,call,amount,{})
 assert cost==amount/10 and source=='conservative_reserved'
 assert ledger.reserve(model,5000,'hello') is None
 assert ledger.spent(m.MODELS[1])==0
assert m.abstains('UNSOLVABLE') and not m.abstains('')
assert m.norm_answer('The Paris.')==m.norm_answer('Paris')
assert m.numeric_equal('42','42.0')
print('Offline accounting checks passed')
