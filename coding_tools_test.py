import tempfile,unittest,os
from pathlib import Path
from coding_tools import run_tests,repair_snapshot,apply_candidate,restore_sources,repair_context
import json

class CodingToolsTests(unittest.TestCase):
 def workspace(self,files):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)
  for name,text in files.items():
   q=p/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
  return p
 def test_pytest_function_actually_runs_and_repair(self):
  p=self.workspace({'calc.py':'def add(a,b): return a-b\n','tests/test_calc.py':'from calc import add\ndef test_add(): assert add(2,3)==5\n'})
  self.assertFalse(run_tests(p)[0]);s=repair_snapshot(p,'Fix calc.py')
  self.assertTrue(apply_candidate(p,s,'{"edits":[{"path":"calc.py","old":"return a-b","new":"return a+b"}]}')[0])
  self.assertTrue(run_tests(p)[0]);restore_sources(p,s);self.assertIn('a-b',(p/'calc.py').read_text())
 def test_zero_tests_is_not_success(self):
  p=self.workspace({'calc.py':'x=1\n','test_empty.py':'pass\n'})
  self.assertFalse(run_tests(p)[0])
 def test_printed_success_then_early_exit_is_not_pass(self):
  p=self.workspace({'test_fake.py':'import os\nprint("1 passed",flush=True)\nos._exit(0)\n'})
  self.assertFalse(run_tests(p)[0])
 def test_multi_file_protected_paths_atomic_validation(self):
  p=self.workspace({'pkg/__init__.py':'','pkg/impl.py':'x=1\n','test_x.py':'def test_x(): assert True\n'})
  s=repair_snapshot(p,'Fix pkg/impl.py')
  for candidate in ['{"files":{"../escape.py":"x=2"}}','{"files":{"test_x.py":"pass"}}','{"files":{"pkg/impl.py":"x=2","pkg/__init__.py":"not valid python !"}}']:
   self.assertFalse(apply_candidate(p,s,candidate)[0]);self.assertEqual((p/'pkg/impl.py').read_text(),'x=1\n')
 def test_candidate_cannot_mutate_real_tests_or_read_api_environment(self):
  p=self.workspace({'calc.py':'value=2\n','test_x.py':'import os\nfrom pathlib import Path\ndef test_x():\n assert "LLM_API_KEY" not in os.environ\n Path("calc.py").write_text("broken")\n assert True\n'})
  previous=os.environ.get('LLM_API_KEY');os.environ['LLM_API_KEY']='test-only-not-a-real-key'
  try:self.assertTrue(run_tests(p)[0])
  finally:
   if previous is None:os.environ.pop('LLM_API_KEY',None)
   else:os.environ['LLM_API_KEY']=previous
  self.assertEqual((p/'calc.py').read_text(),'value=2\n')
 def test_large_file_only_exact_edits(self):
  original='# retained\n'*900+'x=1\n';p=self.workspace({'big.py':original,'test_big.py':'pass\n'});s=repair_snapshot(p,'Fix big.py')
  self.assertFalse(apply_candidate(p,s,'x=2')[0])
  self.assertFalse(apply_candidate(p,s,'{"files":{"big.py":"x=2"},"edits":null}')[0])
  self.assertTrue(apply_candidate(p,s,'{"edits":[{"path":"big.py","old":"x=1","new":"x=2"}]}')[0])
  self.assertIn('# retained\n'*900,(p/'big.py').read_text())
 def test_large_source_context_contains_tail_and_stays_valid_json(self):
  p=self.workspace({'big.py':'# padding\n'*1200+'def target(x): return x-1\n','test_big.py':'from big import target\ndef test_target(): assert target(3)==4\n'})
  snapshot=repair_snapshot(p,'Fix target in big.py')
  context=repair_context(p,snapshot,limit=5000,instruction='Fix target in big.py')
  self.assertLessEqual(len(context),5000);data=json.loads(context)
  self.assertIn('def target',context);self.assertTrue(data['editable_sources'][0]['truncated'])
 def test_missing_explicit_module_can_be_created_and_rolled_back(self):
  p=self.workspace({'test_new.py':'from newmod import value\ndef test_value(): assert value==3\n'});s=repair_snapshot(p,'Create newmod.py')
  self.assertTrue(apply_candidate(p,s,'value=3')[0]);self.assertTrue(run_tests(p)[0]);restore_sources(p,s);self.assertFalse((p/'newmod.py').exists())

if __name__=='__main__':unittest.main()
