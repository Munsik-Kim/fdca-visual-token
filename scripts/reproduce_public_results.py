#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def boot(frame,col,seed):
 p=frame.pivot(index='block_id',columns='anchor_target_rho',values=col);d=(p[.5]-p[.3]).to_numpy();rng=np.random.default_rng(seed);draw=d[rng.integers(0,len(d),(20000,len(d)))].mean(1);return {'left':float(p[.5].mean()),'right':float(p[.3].mean()),'difference':float(d.mean()),'ratio':float(p[.5].mean()/p[.3].mean()),'ci_lower':float(np.quantile(draw,.025,method='higher')),'ci_upper':float(np.quantile(draw,.975,method='higher'))}
def close(a,b,tol=1e-12):
 if abs(a-b)>tol:raise AssertionError((a,b))
def main():
 p2=pd.read_csv(ROOT/'public_results/tables/P2P_BLOCK_LEVEL.csv');w=p2[p2.cell_id.eq('W8_L1.00')];h=json.loads((ROOT/'public_results/summaries/P2P_HYPOTHESIS_TESTS.json').read_text());m=pd.read_csv(ROOT/'public_results/tables/P2P_MATCHED_BLOCK_LEVEL.csv');tests={'P2P_H1':boot(w,'lpips_alex',h['H1']['semantic_seed']),'P2P_H2':boot(w,'unconditional_lpips_alex',h['H2']['semantic_seed']),'P2P_H4':boot(w,'dino_cosine_distance',h['H4']['semantic_seed']),'P2P_H5':boot(m,'lpips_alex',h['H5']['semantic_seed'])}
 for k,hk in [('P2P_H1','H1'),('P2P_H2','H2'),('P2P_H4','H4'),('P2P_H5','H5')]:close(tests[k]['difference'],h[hk]['difference_mean']);close(tests[k]['ci_lower'],h[hk]['ci95_lower'])
 p3=pd.read_csv(ROOT/'public_results/tables/P3S_BLOCK_LEVEL.csv');p=p3[p3.panel.eq('primary_cfg3')];r=json.loads((ROOT/'public_results/summaries/P3S_REPLICATION_TESTS.json').read_text());mm=pd.read_csv(ROOT/'public_results/tables/P3S_MATCHED_BLOCK_LEVEL.csv');tests.update({'P3S_R1':boot(p,'lpips_alex',r['R1']['semantic_seed']),'P3S_R2':boot(p,'unconditional_lpips_alex',r['R2']['semantic_seed']),'P3S_R3':boot(p,'dino_cosine_distance',r['R3']['semantic_seed']),'P3S_R4':boot(mm,'lpips_alex',r['R4']['semantic_seed'])})
 for k,rk in [('P3S_R1','R1'),('P3S_R2','R2'),('P3S_R3','R3'),('P3S_R4','R4')]:close(tests[k]['difference'],r[rk]['difference_mean']);close(tests[k]['ci_lower'],r[rk]['ci95_lower'])
 for phase in ['P2P','P3S']:
  o=pd.read_csv(ROOT/f'public_results/tables/{phase}_TOKEN_OOF.csv');ratio=float(o.abs_error_S1.mean()/o.abs_error_S0.mean());tests[phase+'_TOKEN']={'rows':len(o),'mae_ratio':ratio};expected=.5166468599054039 if phase=='P2P' else .5439628777124867;close(ratio,expected)
 out=ROOT/'build';out.mkdir(exist_ok=True);(out/'reproduced_results.json').write_text(json.dumps({'pass':True,'tests':tests},indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':True,'checks':len(tests),'output':str(out/'reproduced_results.json')}))
if __name__=='__main__':main()
