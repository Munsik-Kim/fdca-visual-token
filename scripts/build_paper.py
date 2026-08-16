#!/usr/bin/env python3
from pathlib import Path
import hashlib,shutil,subprocess,tempfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'paper/source';FROZEN=ROOT/'paper/FDCA_v1.0.0.pdf'
def main():
 exe=shutil.which('tectonic')
 if not exe:raise SystemExit('tectonic is required')
 with tempfile.TemporaryDirectory(prefix='fdca-paper-') as td:
  d=Path(td)
  for p in SRC.iterdir():shutil.copy2(p,d/p.name)
  subprocess.run([exe,'-X','compile','FDCA_v1.0.0.tex','--keep-intermediates'],cwd=d,check=True)
  built=d/'FDCA_v1.0.0.pdf';a=len(PdfReader(str(FROZEN)).pages);b=len(PdfReader(str(built)).pages)
  if a!=b:raise RuntimeError(f'page mismatch {a} != {b}')
  print({'pass':True,'frozen_pages':a,'built_pages':b,'source_sha256':hashlib.sha256((SRC/'FDCA_v1.0.0.tex').read_bytes()).hexdigest()})
if __name__=='__main__':main()
