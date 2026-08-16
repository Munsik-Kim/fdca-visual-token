"""Model-free P2P decomposition, block inference, and LOBO prediction."""

from __future__ import annotations

import hashlib
from typing import Any
import numpy as np
import pandas as pd

RHOS=[.1,.3,.5,.7,.9];METRICS=['lpips_alex','dino_cosine_distance','classifier_jsd','conditioning_prob_abs_diff','conditioning_prob_drop','conditioning_logit_abs_diff','ssim','pixel_l1','psnr']

def seed(label):return int.from_bytes(hashlib.sha256(f'FDCA-P2P-BOOTSTRAP|{label}'.encode()).digest()[:8],'big')
def boot_diff(left,right,label,n=20000):
 ids=sorted(set(left)&set(right));l=np.array([left[i] for i in ids],float);r=np.array([right[i] for i in ids],float);d=l-r;rng=np.random.default_rng(seed(label));draw=d[rng.integers(0,len(d),(n,len(d)))].mean(1);lm=float(l.mean());rm=float(r.mean());return {'block_count':len(ids),'left_mean':lm,'right_mean':rm,'difference_mean':float(d.mean()),'ci95_lower':float(np.quantile(draw,.025,method='higher')),'ci95_upper':float(np.quantile(draw,.975,method='higher')),'ratio_of_block_means':lm/rm if rm else float('inf'),'bootstrap_resamples':n,'semantic_seed':seed(label)}

def decomposition(rows,law):
 n=rows[rows.panel_type.eq('natural')].copy();agg={m:(m,'mean') for m in METRICS};agg.update({'replicate_count':('replicate_id','nunique'),'seed_count_cond':('seed_count','mean'),'descendant_count_cond':('descendant_count','mean'),'descendant_rate_cond':('D_desc_rate','mean'),'D_total_cond':('D_total','mean'),'radius_cond':('maximum_seed_to_descendant_radius','mean')})
 d=n.groupby(['block_id','cell_id','anchor_target_rho'],as_index=False).agg(**agg);keep=law[['block_id','cell_id','anchor_target_rho','split_incidence','expected_seed_count','single_seed_probability','multi_seed_probability']];d=d.merge(keep,on=['block_id','cell_id','anchor_target_rho'],validate='one_to_one')
 for m in METRICS:
  if m=='psnr':continue
  d[f'unconditional_{m}']=d.split_incidence*d[m]
 d['unconditional_descendant_rate']=d.split_incidence*d.descendant_rate_cond;return d

def _design(frame,advanced):
 one=np.column_stack([(frame.anchor_target_rho.to_numpy()==r).astype(float) for r in RHOS[1:]])
 base=['seed_count','severity_tv','severity_surprisal','connected_components','bounding_box_area'];extra=['D_desc_rate','descendant_count','first_descendant_step','maximum_seed_to_descendant_radius','descendant_connected_components','descendant_fraction_distance_le_5'];cols=base+(extra if advanced else []);x=frame[cols].astype(float).copy();x['first_descendant_step']=x['first_descendant_step'].fillna(33) if 'first_descendant_step' in x else 0;return np.column_stack([one,x.to_numpy()]),['rho_'+str(r) for r in RHOS[1:]]+cols
def ridge_predict(xtr,ytr,xte):
 mu=xtr.mean(0);sd=xtr.std(0);sd[sd<1e-12]=1;z=(xtr-mu)/sd;zt=(xte-mu)/sd;ym=ytr.mean();beta=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(ytr-ym));return ym+zt@beta,mu,sd
def predictor(rows):
 f=rows[(rows.panel_type=='natural')&rows.cell_id.eq('W8_L1.00')].copy().reset_index(drop=True);x0,c0=_design(f,False);x1,c1=_design(f,True);y=f.lpips_alex.to_numpy(float);p0=np.empty(len(f));p1=np.empty(len(f));fold=[]
 for b in sorted(f.block_id.unique()):
  te=f.block_id.eq(b).to_numpy();tr=~te;p0[te],m0,s0=ridge_predict(x0[tr],y[tr],x0[te]);p1[te],m1,s1=ridge_predict(x1[tr],y[tr],x1[te]);fold.append({'heldout_block':b,'train_block_count':int(f.loc[tr,'block_id'].nunique()),'heldout_row_count':int(te.sum()),'heldout_absent_from_train':b not in set(f.loc[tr,'block_id']),'S0_zero_variance_feature_count':int(np.sum(s0==1)&0),'S1_feature_count':len(c1)})
 f['prediction_S0']=p0;f['prediction_S1']=p1;f['abs_error_S0']=abs(y-p0);f['abs_error_S1']=abs(y-p1);f['propagation_contribution']=p1-p0
 bm=f.groupby('block_id').agg(e0=('abs_error_S0','mean'),e1=('abs_error_S1','mean'));imp=(bm.e0-bm.e1).to_numpy();rng=np.random.default_rng(seed('H3_MAE'));draw=imp[rng.integers(0,len(imp),(20000,len(imp)))].mean(1)
 f['rank_y_within_rho']=f.groupby('anchor_target_rho').lpips_alex.rank(method='average');f['rank_c_within_rho']=f.groupby('anchor_target_rho').propagation_contribution.rank(method='average');x=f.rank_c_within_rho.to_numpy(float);z=f.rank_y_within_rho.to_numpy(float)
 def corr_from_sums(n,sx,sy,sxx,syy,sxy):
  num=sxy-sx*sy/n;den=np.sqrt(np.maximum((sxx-sx*sx/n)*(syy-sy*sy/n),0));return np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den>0)
 bs=[]
 for b,g in f.groupby('block_id'):bs.append([len(g),g.rank_c_within_rho.sum(),g.rank_y_within_rho.sum(),(g.rank_c_within_rho**2).sum(),(g.rank_y_within_rho**2).sum(),(g.rank_c_within_rho*g.rank_y_within_rho).sum()])
 bs=np.array(bs,float);overall=float(corr_from_sums(np.array([len(f)]),np.array([x.sum()]),np.array([z.sum()]),np.array([(x*x).sum()]),np.array([(z*z).sum()]),np.array([(x*z).sum()]))[0]);rng=np.random.default_rng(seed('H3_SPEARMAN'));ix=rng.integers(0,len(bs),(20000,len(bs)));s=bs[ix].sum(1);corr=corr_from_sums(*[s[:,i] for i in range(6)])
 result={'rows':len(f),'blocks':int(f.block_id.nunique()),'mae_S0':float(f.abs_error_S0.mean()),'mae_S1':float(f.abs_error_S1.mean()),'mae_ratio_S1_over_S0':float(f.abs_error_S1.mean()/f.abs_error_S0.mean()),'mae_improvement':float(imp.mean()),'mae_improvement_ci95_lower':float(np.quantile(draw,.025,method='higher')),'mae_improvement_ci95_upper':float(np.quantile(draw,.975,method='higher')),'within_rho_residual_spearman':overall,'within_rho_residual_spearman_ci95_lower':float(np.quantile(corr,.025,method='higher')),'within_rho_residual_spearman_ci95_upper':float(np.quantile(corr,.975,method='higher')),'bootstrap_resamples':20000,'S0_features':c0,'S1_features':c1,'lobo_no_leakage':all(r['heldout_absent_from_train'] for r in fold)};result['pass']=result['mae_ratio_S1_over_S0']<=.90 and result['mae_improvement_ci95_lower']>0 and result['within_rho_residual_spearman']>.20 and result['within_rho_residual_spearman_ci95_lower']>0;return f,pd.DataFrame(fold),result

def evaluate(rows,dec):
 w=dec[dec.cell_id.eq('W8_L1.00')]
 def mp(frame,rho,col):return frame[frame.anchor_target_rho.eq(rho)].set_index('block_id')[col].astype(float).to_dict()
 h1=boot_diff(mp(w,.5,'lpips_alex'),mp(w,.3,'lpips_alex'),'H1_CONDITIONAL_LPIPS');h1['pass']=h1['ci95_lower']>0 and h1['ratio_of_block_means']>=1.5
 h2=boot_diff(mp(w,.5,'unconditional_lpips_alex'),mp(w,.3,'unconditional_lpips_alex'),'H2_UNCONDITIONAL_LPIPS');h2['pass']=h2['ci95_lower']>0 and h2['ratio_of_block_means']>=1.5
 h4=boot_diff(mp(w,.5,'dino_cosine_distance'),mp(w,.3,'dino_cosine_distance'),'H4_DINO');h4['pass']=h4['ci95_lower']>0 and h4['ratio_of_block_means']>=1.5
 m=rows[rows.panel_type.eq('matched_timing')].groupby(['block_id','anchor_target_rho'],as_index=False).lpips_alex.mean();h5=boot_diff(mp(m,.5,'lpips_alex'),mp(m,.3,'lpips_alex'),'H5_MATCHED');h5['pass']=h5['ci95_lower']>0 and h5['ratio_of_block_means']>=1.5
 oof,fold,h3=predictor(rows);return {'H1':h1,'H2':h2,'H3':h3,'H4':h4,'H5':h5},oof,fold
