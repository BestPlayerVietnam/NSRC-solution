"""Download public research datasets during development, never at contest runtime."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
CACHE=Path(__file__).resolve().parents[1]/'.tools'/'dataset-cache'
SOURCES=[
 ('gsm8k_train.jsonl','https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl','17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'),
 ('mbpp_sanitized.json','https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json','ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9'),
 ('squad_dev_v2.json','https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json','80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'),
 ('wikisql_data.tar.bz2','https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2','755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881'),
]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
 for name,url,expected in SOURCES:
  dest=CACHE/name
  if dest.exists():
   if digest(dest)!=expected:raise RuntimeError('Existing source hash mismatch: '+name)
  else:
   if not args.download:raise RuntimeError('Missing source; rerun with --download: '+name)
   CACHE.mkdir(parents=True,exist_ok=True);part=dest.with_suffix(dest.suffix+'.part')
   with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as out:
    size=0
    while True:
     chunk=response.read(1024*1024)
     if not chunk:break
     size+=len(chunk)
     if size>100_000_000:raise RuntimeError('Unexpected source size')
     out.write(chunk)
   if digest(part)!=expected:raise RuntimeError('Downloaded source changed; keep previous protocol frozen: '+name)
   os.replace(part,dest)
  print(json.dumps({'file':name,'bytes':dest.stat().st_size,'verified_sha256':expected}))
if __name__=='__main__':main()
