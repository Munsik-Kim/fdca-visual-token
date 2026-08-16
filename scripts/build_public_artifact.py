#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'release/FDCA_v1.0.0_Public_Research_Artifact.zip'
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 try:rels=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
 except Exception:rels=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and 'build' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and '.venv' not in p.parts]
 rels=sorted(r for r in rels if r!=OUT.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for r in rels:z.write(ROOT/r,r)
 with zipfile.ZipFile(OUT) as z:
  if z.testzip():raise RuntimeError('CRC failure')
 print(json.dumps({'path':str(OUT),'files':len(rels),'size':OUT.stat().st_size,'sha256':sha(OUT),'crc_pass':True}))
if __name__=='__main__':main()
