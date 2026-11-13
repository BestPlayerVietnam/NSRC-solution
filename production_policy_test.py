import asyncio,json,tempfile,unittest
from pathlib import Path
from planner import MODELS
from run_agent import model_answer,solve_coding,BudgetExhausted,try_known_repairs,JiuwenLLM

class FakeLLM:
 def __init__(self,*answers):self.answers=iter(answers);self.calls=[]
 async def ask(self,system,user,**kwargs):
  self.calls.append((system,user,kwargs));answer=next(self.answers)
  if isinstance(answer,Exception):raise answer
  return answer

class ProductionPolicyTests(unittest.TestCase):
 def test_fixed_model_rejects_switch_before_any_reservation(self):
  from types import SimpleNamespace
  llm=SimpleNamespace(fixed_model=True,initial_model=MODELS['small_plain'])
  with self.assertRaises(ValueError):
   asyncio.run(JiuwenLLM.ask(llm,'system','user',model=MODELS['large_plain']))
 def workspace(self,files):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)
  for name,text in files.items():
   path=p/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
  return p
 def test_sql_repair_sees_real_error_and_runs_on_all_rows(self):
  p=self.workspace({'values.csv':'unit price\n2\n3\n4\n100\n'});llm=FakeLLM('SELECT SUM(missing) FROM data','SELECT SUM("unit price") FROM data')
  result=asyncio.run(model_answer(llm,{'task_type':'data_analysis','instruction':'Total unit price?'},p,MODELS['large_plain']))
  self.assertEqual(result,'109');self.assertEqual(len(llm.calls),2);self.assertIn('no such column',llm.calls[1][1])
 def test_repeated_sql_stops_and_explicit_abstention_does_not_retry(self):
  p=self.workspace({'x.csv':'x\n1\n2\n'})
  for answers,n in [(('UNSOLVABLE',),1),(('SELECT missing FROM data','SELECT missing FROM data'),2)]:
   llm=FakeLLM(*answers);self.assertIsNone(asyncio.run(model_answer(llm,{'task_type':'data_analysis','instruction':'Find unknown y'},p,MODELS['large_plain'])));self.assertEqual(len(llm.calls),n)
 def test_math_uses_exact_calculator(self):
  p=self.workspace({});llm=FakeLLM('CALC: (0.1+0.2)*100')
  self.assertEqual(asyncio.run(model_answer(llm,{'task_type':'math','instruction':'Compute total','check':'exact_int'},p,MODELS['large_plain'])),'30')
 def test_valid_prefix_of_truncated_response_is_not_accepted(self):
  p=self.workspace({});llm=FakeLLM('CALC: 12')
  llm.last_finish_reason='length'
  self.assertIsNone(asyncio.run(model_answer(llm,{'task_type':'math','instruction':'Compute a long sum'},p,MODELS['large_plain'])))
  self.assertEqual(llm.task_state,'check_failed')
 def test_search_late_evidence_is_in_prompt(self):
  p=self.workspace({'kb/notes.md':'Background unrelated text.\n'*160+'The headquarters of Velorian are in Riga.\n'})
  llm=FakeLLM('Riga');self.assertEqual(asyncio.run(model_answer(llm,{'task_type':'search','instruction':'Where are the headquarters of Velorian?'},p,MODELS['large_plain'])),'Riga')
  self.assertIn('headquarters of Velorian are in Riga',llm.calls[0][1])
 def test_search_evidence_matches_unescaped_content(self):
  p=self.workspace({'kb/source.md':'The author of Lunar Notes is O"Brien.\n'})
  llm=FakeLLM('O"Brien')
  self.assertEqual(asyncio.run(model_answer(llm,{'task_type':'search','instruction':'Who wrote Lunar Notes?'},p,MODELS['large_plain'])),'O"Brien')
 def test_local_repair_preserves_unrelated_same_named_function(self):
  original='def maximum(xs): return max(xs)+1\n\ndef total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n'
  p=self.workspace({'mylib.py':original,'test_mylib.py':'from mylib import total\ndef test_total(): assert total([1,2])==3\n'})
  self.assertTrue(try_known_repairs(p));self.assertIn('return max(xs)+1',(p/'mylib.py').read_text())
 def test_coding_multifile_repair_and_budget_rollback(self):
  p=self.workspace({'pkg/__init__.py':'','pkg/calc.py':'def plus(a,b): return a-b\n','tests/test_calc.py':'from pkg.calc import plus\ndef test_plus(): assert plus(2,3)==5\n'})
  task={'task_type':'coding','instruction':'Fix pkg/calc.py'}
  llm=FakeLLM(json.dumps({'edits':[{'path':'pkg/calc.py','old':'return a-b','new':'return a+b'}]}))
  self.assertTrue(asyncio.run(solve_coding(llm,task,p,False)))
  original='def plus(a,b): return a-b\n';(p/'pkg/calc.py').write_text(original,encoding='utf-8')
  llm=FakeLLM(json.dumps({'files':{'pkg/calc.py':'def plus(a,b): return 0'}}),BudgetExhausted('test'))
  with self.assertRaises(BudgetExhausted):asyncio.run(solve_coding(llm,task,p,False))
  self.assertEqual((p/'pkg/calc.py').read_text(),original)

if __name__=='__main__':unittest.main()
