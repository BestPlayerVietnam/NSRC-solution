"""Resume the remaining frozen evaluation stages after the active train process exits."""
import json,msvcrt,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments'/'runs'/'2026-11-12-v1'
def status(stage,state='running'):
 obj={'stage':stage,'state':state,'time':time.time()}
 p=RUN/'pipeline_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj),encoding='utf-8');os.replace(temp,p)
 print(json.dumps(obj),flush=True)
def main():
 status('waiting_for_train')
 deadline=time.monotonic()+5400
 while True:
  with (RUN/'process.lock').open('a+b') as lock:
   lock.seek(0)
   try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
   except OSError:
    if time.monotonic()>deadline:raise RuntimeError('Train wait deadline exceeded')
   else:
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);break
  time.sleep(2)
 steps=[
 ('freeze',['analyze.py','freeze'],RUN/'predictor_train.json'),
 ('plan_calibration',['analyze.py','plan-calibration'],RUN/'calibration_predictions.jsonl'),
 ('calibration',['run_experiments.py','--phase','calibration'],None),
 ('calibrate',['analyze.py','calibrate'],RUN/'predictor_calibrated.json'),
 ('plan_test',['analyze.py','plan-test'],RUN/'test_predictions.jsonl'),
 ('test',['run_experiments.py','--phase','test'],None),
 ('report',['analyze.py','report'],None),
 ]
 for stage,args,artifact in steps:
  if artifact and artifact.exists():continue
  status(stage)
  result=subprocess.run([sys.executable,'-B',str(ROOT/'experiments'/args[0]),*args[1:]],cwd=ROOT)
  if result.returncode:status(stage,'failed');raise SystemExit(result.returncode)
 status('complete','complete')
if __name__=='__main__':main()
