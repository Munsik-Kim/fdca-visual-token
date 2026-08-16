#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('asset_dir',type=Path);a=ap.parse_args();ledger=json.loads((ROOT/'release/excluded_assets.json').read_text());bad=[]
 for x in ledger['assets']:
  p=a.asset_dir/x['filename']
  if not p.is_file() or p.stat().st_size!=x['size'] or sha(p)!=x['sha256']:bad.append(x['filename'])
 print(json.dumps({'checked':len(ledger['assets']),'mismatches':bad,'pass':not bad}));raise SystemExit(0 if not bad else 2)
if __name__=='__main__':main()
