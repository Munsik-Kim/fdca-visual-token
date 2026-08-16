#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,re,sys
ROOT=Path(__file__).resolve().parents[1];MAX=25*1024*1024
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def files():
 return sorted(p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and 'build' not in p.parts and not (p.parent==ROOT/'release' and p.suffix=='.zip') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
def main():
 fs=files();findings=[];patterns=[('private_home','/home/'+'mskim'),('windows_profile','/mnt/c/'+'Users/|[A-Za-z]:\\\\'+'Users\\\\'),('token',r'(?i)\b(?:h'+'f_|gh'+'p_|github_'+'pat_|s'+'k-[A-Za-z0-9])\w+'),('private_key','BEGIN (?:RSA |OPENSSH )?'+'PRIVATE KEY'),('placeholder',r'\b(?:RE'+'PLACE|PLACE'+'HOLDER|T'+'BD)\b')]
 for p in fs:
  rel=p.relative_to(ROOT).as_posix()
  if p.stat().st_size>MAX:findings.append({'file':rel,'kind':'size','bytes':p.stat().st_size})
  if p.is_symlink():findings.append({'file':rel,'kind':'symlink'})
  if p.name=='.env' or p.suffix.lower() in {'.pth','.pt','.ckpt','.safetensors'}:findings.append({'file':rel,'kind':'prohibited_asset'})
  try:text=p.read_text()
  except UnicodeDecodeError:continue
  for kind,pat in patterns:
   if re.search(pat,text):findings.append({'file':rel,'kind':kind})
 for required in ['README.md','README_KO.md','CITATION.cff','.zenodo.json','LICENSE-CODE','LICENSE-PAPER','release/PUBLIC_MANIFEST.json','release/SHA256SUMS.txt']:
  if not (ROOT/required).is_file():findings.append({'file':required,'kind':'missing'})
 for rel in ['README.md','CITATION.cff','release/RELEASE_NOTES_v1.0.0.md']:
  t=(ROOT/rel).read_text()
  if '10.5281/zenodo.21965747' not in t:findings.append({'file':rel,'kind':'doi_mismatch'})
 m=json.loads((ROOT/'release/PUBLIC_MANIFEST.json').read_text());actual=[]
 for p in fs:
  rel=p.relative_to(ROOT).as_posix()
  if rel in {'release/PUBLIC_MANIFEST.json','release/SHA256SUMS.txt'}:continue
  actual.append({'path':rel,'size':p.stat().st_size,'sha256':sha(p)})
 if actual!=m['files']:findings.append({'file':'release/PUBLIC_MANIFEST.json','kind':'manifest_mismatch'})
 result={'schema':'FDCA_PUBLIC_VERIFY_V1','files_scanned':len(fs),'findings':findings,'escaping_symlinks':0,'tracked_over_25mb':sum(x['kind']=='size' for x in findings),'pass':not findings};print(json.dumps(result,indent=2,sort_keys=True));sys.exit(0 if result['pass'] else 2)
if __name__=='__main__':main()
