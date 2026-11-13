import os,tempfile,unittest,json,httpx
from pathlib import Path
from unittest.mock import patch
from experiments import production_compare as pc

class RelayTests(unittest.TestCase):
 def test_pending_reservations_and_late_arm_are_not_lost(self):
  events=[{'event':'reserved','id':'a','arm':'before','reserved':100},
          {'event':'completed','id':'a','arm':'candidate','cost':20,'accounting':'provider_reconciled'},
          {'event':'reserved','id':'b','arm':'before','reserved':80}]
  charged=pc.charged_calls(events)
  self.assertEqual(sum(r['cost'] for r in charged),100)
  self.assertEqual([r['arm'] for r in charged],['before','before'])
  self.assertEqual(charged[1]['accounting'],'reserved_pending')
 def test_every_physical_request_is_charged_and_unknown_usage_is_held(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);(base/'run').mkdir();protocol={'ceilings':{m:10000 for m in pc.MODELS}}
   with patch.object(pc,'BUDGET',base/'ledger.json'),patch.object(pc,'RUN',base/'run'),patch.dict(os.environ,{'LLM_API_KEY':'test-only'}):
    relay=pc.Relay(protocol);relay.client.close()
    class Backend:
     n=0
     def post(self,*args,**kwargs):
      self.n+=1
      payload={'choices':[{'message':{'content':'42'},'finish_reason':'stop'}]}
      if self.n==1:payload['usage']={'prompt_tokens':50,'completion_tokens':10,'total_tokens':60}
      return httpx.Response(200,json=payload)
     def close(self):pass
    backend=Backend();relay.client=backend;relay.arm='test';relay.serve()
    model='qwen3.8-27b-noreason';question=next(iter(relay.instructions.values()))
    body={'model':model,'max_tokens':100,'messages':[{'role':'user','content':question}]}
    try:
     with httpx.Client(trust_env=False) as client:
      url=f'http://127.0.0.1:{relay.server.server_port}/v1/chat/completions';headers={'Authorization':'Bearer '+relay.token}
      self.assertEqual(client.post(url,json=body,headers=headers).status_code,200)
      self.assertEqual(relay.ledger.spent(model),15)
      self.assertEqual(client.post(url,json=body,headers=headers).status_code,200)
      self.assertGreater(relay.ledger.spent(model),343)
      self.assertEqual(client.post(url,json=body,headers={'Authorization':'Bearer wrong'}).status_code,403)
      self.assertEqual(backend.n,2)
      self.assertEqual(relay.ledger.data['models'][model]['calls'],2)
      self.assertFalse(relay.ledger.data['models'][model]['pending'])
    finally:relay.close()

if __name__=='__main__':unittest.main()
