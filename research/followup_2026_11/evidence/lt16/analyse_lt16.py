"""LT16 frozen saved-state redraw analysis. Inputs are scalar rows; no model imports."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

SELECTORS=['OLD_MAX','PROPOSED_OOF','ENTROPY_OOF','TIME_ONLY_FROZEN','FIXED_OOF']
SCOPE='LT14 reused development classes; saved-state conditional redraw diagnostic'
BINS=np.array([-np.inf,-.01,0,.005,.01,.02,np.inf],dtype=np.float64)

def read_csv(path):
    # CSV records -> list[dict]; preserve integer seed as string until explicit conversion.
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    # records -> stable columns in insertion order, with explicit empty NA.
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text('');return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(path):
    # bytes -> hexadecimal SHA256.
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def truth(v):
    # scalar bool/int/string -> bool, refusing unknown encodings.
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if str(v).lower() in ('true','1'):return True
    if str(v).lower() in ('false','0'):return False
    raise ValueError(('invalid boolean',v))

def unitkey(row):
    # record -> semantic trajectory key, never row position.
    return tuple(int(row[k]) for k in ('class_id','base_seed','replicate_id'))

def statekey(row):
    # record -> semantic original-trajectory/time key.
    return unitkey(row)+(int(row['step']),)

def interval(class_values,draws,q):
    # class_values [C], draws [20000,C] -> estimate, percentile interval.
    a=np.asarray(class_values,dtype=np.float64)
    b=a[draws].mean(axis=1)
    lo,hi=np.quantile(b,q,method='linear')
    return float(a.mean()),float(lo),float(hi)

def mcse(z):
    # z [original trajectories,8 redraws]; coupled selector differences first.
    z=np.asarray(z,dtype=np.float64)
    return float(np.sqrt(z.var(axis=1,ddof=1).sum()/(len(z)**2*z.shape[1])))

def verdict(lo,hi,boundary):
    # interval endpoints -> declared directional/boundary classification.
    if lo>boundary:return 'LOWER_ABOVE_PRACTICAL_BOUNDARY'
    if lo>0:return 'POSITIVE_DIRECTION_BOUNDARY_UNCONFIRMED'
    if hi<0:return 'NEGATIVE_DIRECTION'
    return 'DIRECTION_UNRESOLVED'

def split_half(times,g):
    # times [T] sorted; g [T,8] -> exact-tie early winner and cross-half evaluation.
    a=np.asarray(g)[:,:4].mean(axis=1);b=np.asarray(g)[:,4:].mean(axis=1)
    ia=int(np.argmax(a));ib=int(np.argmax(b))
    return dict(t_A=int(times[ia]),t_B=int(times[ib]),same_winner=ia==ib,
                all_A_equal=bool(np.all(a==a[0])),all_B_equal=bool(np.all(b==b[0])),
                all_both_equal=bool(np.all(a==a[0]) and np.all(b==b[0])),
                candidate_count=len(times),self_value=float((a[ia]+b[ib])/2),
                transfer_value=float((b[ia]+a[ib])/2),
                selection_optimism=float((a[ia]+b[ib]-b[ia]-a[ib])/2))

def compute(selectors,states,drawrows,strict=True):
    # All inputs keyed records. No class or trajectory is silently dropped.
    sm={statekey(r):r for r in states};assert len(sm)==len(states)
    sid={r['state_id']:statekey(r) for r in states};assert len(sid)==len(states)
    sk={(unitkey(r),r['selector']):r for r in selectors};assert len(sk)==len(selectors)
    units=sorted({unitkey(r) for r in selectors});classes=sorted({u[0] for u in units})
    assert len(units)==2*len(classes)
    assert all(sorted(u[2] for u in units if u[0]==c)==[0,1] for c in classes)
    assert set(sk)=={(u,s) for u in units for s in SELECTORS}
    dm={}
    for row in drawrows:
        key=(statekey(row),int(row['redraw_id']));assert key not in dm,('duplicate',key)
        assert sid[row['state_id']]==key[0]
        assert key[0] in sm and 0<=key[1]<8
        g=float(row['G_new']);pq=float(row['lpips_pq']);pr=float(row['lpips_pr'])
        assert np.isfinite([g,pq,pr]).all()
        assert g==pq-pr,('gain differs from current-draw metric pair',key)
        changed=truth(row['post_state_changed'])
        assert changed==(int(row['changed_batch_count'])>0)
        if not changed:
            assert g==0 and truth(row['final_QR_same'])
            assert int(row['final_hamming_qr'])==0
        dm[key]=row
    assert set(dm)=={(s,r) for s in sm for r in range(8)},'incomplete state/redraw panel'
    if strict:
        assert (len(classes),len(units),len(states),len(selectors),len(drawrows))==(24,48,158,240,1264)
        assert sum(32-k[3] for k in sm)==2895
    state_stats=[];gs={};hs={};ms={}
    for key in sorted(sm):
        r=sm[key];rr=[dm[(key,i)] for i in range(8)]
        g=np.array([float(x['G_new']) for x in rr]);I=np.array([truth(x['post_state_changed']) for x in rr])
        mu=float(g.mean());h=float(I.mean());m=float(g[I].mean()) if I.any() else 0.
        assert np.isclose(mu,h*m,rtol=2e-14,atol=2e-16)
        gs[key]=g;hs[key]=h;ms[key]=m
        state_stats.append(dict(trajectory_id=r['trajectory_id'],state_id=r['state_id'],
            class_id=key[0],base_seed=key[1],replicate_id=key[2],step=key[3],
            old_G=float(r['old_G']),mu_hat=mu,std=float(g.std(ddof=1)),MCSE_state=float(g.std(ddof=1)/np.sqrt(8)),
            K=8,changed_count=int(I.sum()),h=h,m=m,m_definition='0_if_no_observed_change',
            hm=h*m,positive_count=int((g>0).sum()),negative_count=int((g<0).sum()),zero_count=int((g==0).sum()),
            changed_zero_gain_count=int((I&(g==0)).sum()),old_prediction=float(r['old_prediction']),
            m_t=float(r['m_t']),scope=SCOPE))
    idx={u:i for i,u in enumerate(units)}
    arr=np.empty((len(units),len(SELECTORS),8));old=np.empty((len(units),len(SELECTORS)))
    hh=np.empty_like(old);mm=np.empty_like(old)
    selector_units=[];selector_draws=[]
    for ui,u in enumerate(units):
        for si,s in enumerate(SELECTORS):
            sr=sk[(u,s)];key=statekey(sr);assert key in sm and sr['state_id']==sm[key]['state_id']
            arr[ui,si]=gs[key];old[ui,si]=float(sr['old_G']);hh[ui,si]=hs[key];mm[ui,si]=ms[key]
            selector_units.append(dict(trajectory_id=sr['trajectory_id'],class_id=u[0],base_seed=u[1],
                replicate_id=u[2],selector=s,state_id=sr['state_id'],step=key[3],old_G=old[ui,si],
                new_G=float(arr[ui,si].mean()),new_minus_old=float(arr[ui,si].mean()-old[ui,si]),
                h=hh[ui,si],m=mm[ui,si],scope=SCOPE))
            for ri in range(8):
                selector_draws.append(dict(class_id=u[0],base_seed=u[1],replicate_id=u[2],
                    trajectory_id=sr['trajectory_id'],selector=s,state_id=sr['state_id'],step=key[3],
                    redraw_id=ri,canonical_unit_id=dm[(key,ri)]['unit_id'],G_new=arr[ui,si,ri],
                    scope=SCOPE))
    primary_draws=np.random.Generator(np.random.PCG64(2026111601)).integers(0,len(classes),(20000,len(classes)),dtype=np.int64)
    auxiliary_draws=np.random.Generator(np.random.PCG64(2026111602)).integers(0,len(classes),(20000,len(classes)),dtype=np.int64)
    def cmean(x):
        # x [U,...] -> [C,...], averaging the two original trajectories first.
        return np.array([np.asarray(x)[[idx[u] for u in units if u[0]==c]].mean(axis=0) for c in classes])
    z={'A':arr[:,0,:],'B':arr[:,0,:]-arr[:,2,:],'C':arr[:,1,:]-arr[:,3,:]}
    definitions={'A':'OLD_MAX','B':'OLD_MAX-ENTROPY_OOF','C':'PROPOSED_OOF-TIME_ONLY_FROZEN'}
    primary=[];class_rows=[];loco=[];rep_rows=[]
    for name,zz in z.items():
        cv=cmean(zz.mean(axis=1));est,lo,hi=interval(cv,primary_draws,[.05/6,1-.05/6])
        boundary=.01 if name=='A' else .005
        primary.append(dict(contrast=name,definition=definitions[name],estimate=est,ci_low=lo,ci_high=hi,
            confidence=1-.05/3,MCSE_fixed_parents=mcse(zz),boundary=boundary,
            point_above_boundary=est>boundary,lower_above_boundary=lo>boundary,status=verdict(lo,hi,boundary),
            n_class=len(classes),n_original_trajectory=len(units),K=8,scope=SCOPE))
        for ci,c in enumerate(classes):
            class_rows.append(dict(class_id=c,contrast=name,value=float(cv[ci]),positive=bool(cv[ci]>0),scope=SCOPE))
            loco.append(dict(contrast=name,excluded_class=c,estimate=float(np.delete(cv,ci).mean()),scope='all-class leave-one-out sensitivity; primary retained'))
        for rep in [0,1]:
            rep_rows.append(dict(contrast=name,replicate_id=rep,estimate=float(zz[[u[2]==rep for u in units]].mean()),n_class=len(classes),scope=SCOPE))
    auxiliary=[];selector_summary=[];class_selector=[];signed=[]
    def aux(name,uv):
        # per-original-trajectory vector [U] -> paired class percentile interval.
        e,l,h=interval(cmean(uv),auxiliary_draws,[.025,.975])
        auxiliary.append(dict(contrast=name,estimate=e,ci_low=l,ci_high=h,confidence=.95,n_class=len(classes),scope=SCOPE))
        return e,l,h
    aux('PROPOSED_OOF-ENTROPY_OOF',(arr[:,1]-arr[:,2]).mean(axis=1))
    aux('OLD_MAX-FIXED_OOF',(arr[:,0]-arr[:,4]).mean(axis=1))
    for si,s in enumerate(SELECTORS):
        avg=arr[:,si].mean(axis=1);e,lo,hi=aux(s,avg)
        de,dl,dh=aux(s+'_NEW_MINUS_OLD',avg-old[:,si])
        H=float(hh[:,si].mean());Meq=float(mm[:,si].mean())
        cov=float(np.mean((hh[:,si]-H)*(mm[:,si]-Meq)))
        assert np.isclose(e,H*Meq+cov,rtol=2e-13,atol=2e-16)
        selector_summary.append(dict(selector=s,old_G=float(old[:,si].mean()),new_G=e,ci_low=lo,ci_high=hi,
            confidence=.95,new_minus_old=de,change_ci_low=dl,change_ci_high=dh,MCSE_fixed_parents=mcse(arr[:,si]),
            H_observed_change=H,M_equal_sample=Meq,covariance_ddof0=cov,H_times_Mequal=H*Meq,
            H_Mequal_plus_cov=H*Meq+cov,M_pooled_changed=e/H if H>0 else '',
            changed_count=int(hh[:,si].sum()*8),logical_selector_draws=len(units)*8,
            zero_change_states=int((hh[:,si]==0).sum()),all_change_states=int((hh[:,si]==1).sum()),n_class=len(classes),scope=SCOPE))
        cv=cmean(avg);ocv=cmean(old[:,si])
        for ci,c in enumerate(classes):
            class_selector.append(dict(class_id=c,selector=s,old_G=float(ocv[ci]),new_G=float(cv[ci]),new_minus_old=float(cv[ci]-ocv[ci]),scope=SCOPE))
        flat=arr[:,si].ravel();pplus=float(np.mean(flat>0));pminus=float(np.mean(flat<0))
        plus=float(flat[flat>0].mean()) if pplus else 0;minus=float(flat[flat<0].mean()) if pminus else 0
        assert np.isclose(e,pplus*plus+pminus*minus,rtol=2e-13,atol=2e-16)
        signed.append(dict(selector=s,p_positive=pplus,p_negative=pminus,p_zero=float(np.mean(flat==0)),
            mean_positive=plus if pplus else '',mean_negative=minus if pminus else '',
            positive_contribution=pplus*plus,negative_contribution=pminus*minus,
            combined=e,class_positive=int((cv>0).sum()),class_negative=int((cv<0).sum()),class_zero=int((cv==0).sum()),scope=SCOPE))
    half=[];pred_unit=[];pred_bins=[];error_terms={};frozen_pred_rows=[]
    for u in units:
        kk=sorted(k for k in sm if k[:3]==u);gg=np.stack([gs[k] for k in kk]);times=[k[3] for k in kk]
        sh=split_half(times,gg);half.append(dict(class_id=u[0],base_seed=u[1],replicate_id=u[2],**sh,scope=SCOPE))
        pred=np.array([float(sm[k]['old_prediction']) for k in kk]);mt=np.array([float(sm[k]['m_t']) for k in kk])
        gold=np.array([float(sm[k]['old_G']) for k in kk]);new=gg.mean(axis=1)
        row=dict(class_id=u[0],base_seed=u[1],replicate_id=u[2],candidate_count=len(kk))
        for target,yy in [('old',gold),('new',new)]:
            a=float(np.mean((pred-yy)**2));b=float(np.mean((mt-yy)**2))
            row[target+'_state_MSE']=a;row[target+'_time_MSE']=b;row[target+'_state_minus_time_MSE']=a-b
        row['mean_state_MCSE']=float(np.mean(gg.std(axis=1,ddof=1)/np.sqrt(8)))
        row['scope']=SCOPE;pred_unit.append(row)
        for i,k in enumerate(kk):
            frozen_pred_rows.append(dict(class_id=u[0],trajectory_id=sm[k]['trajectory_id'],state_id=sm[k]['state_id'],
                step=k[3],weight=1/(len(units)*len(kk)),old_prediction=pred[i],m_t=mt[i],old_G=gold[i],mu_hat=new[i],
                MCSE_state=float(gg[i].std(ddof=1)/np.sqrt(8))))
    for key in ['self_value','transfer_value','selection_optimism']:
        aux('SPLIT_HALF_'+key.upper(),np.array([r[key] for r in half]))
    for key in ['old_state_MSE','old_time_MSE','old_state_minus_time_MSE','new_state_MSE','new_time_MSE','new_state_minus_time_MSE']:
        aux('PREDICTION_'+key,np.array([r[key] for r in pred_unit]))
    for predictor in ['old_prediction','m_t']:
        xx=np.array([r[predictor] for r in frozen_pred_rows]);ww=np.array([r['weight'] for r in frozen_pred_rows])
        for bi in range(len(BINS)-1):
            mask=(xx>=BINS[bi])&(xx<BINS[bi+1]);weights=ww[mask];mass=float(weights.sum())
            pred_bins.append(dict(predictor=predictor,bin_index=bi,lower=str(BINS[bi]),upper=str(BINS[bi+1]),
                states=int(mask.sum()),trajectory_equal_weight_mass=mass,prediction_mean=float(np.sum(xx[mask]*weights)/mass) if mass else '',
                old_G_mean=float(sum(r['old_G']*r['weight'] for r,m in zip(frozen_pred_rows,mask) if m)/mass) if mass else '',
                new_G_mean=float(sum(r['mu_hat']*r['weight'] for r,m in zip(frozen_pred_rows,mask) if m)/mass) if mass else '',scope=SCOPE))
    calibration=[]
    for predictor in ['old_prediction','m_t']:
        x=np.array([r[predictor] for r in frozen_pred_rows]);w=np.array([r['weight'] for r in frozen_pred_rows]);mx=float(w@x)
        for target in ['old_G','mu_hat']:
            y=np.array([r[target] for r in frozen_pred_rows]);my=float(w@y)
            cov=float(w@((x-mx)*(y-my)));vx=float(w@((x-mx)**2));vy=float(w@((y-my)**2))
            calibration.append(dict(predictor=predictor,target=target,weighted_MSE=float(w@((x-y)**2)),
                weighted_correlation=cov/math.sqrt(vx*vy) if vx>0 and vy>0 else '',scope='descriptive, selected unique times only; unit-equal weighting'))
    nums={'primary':primary,'selector_summary':selector_summary,'auxiliary':auxiliary,
        'split_half':{'self_mean':float(np.mean([r['self_value'] for r in half])),
            'transfer_mean':float(np.mean([r['transfer_value'] for r in half])),
            'same_winner_units':sum(r['same_winner'] for r in half),
            'all_A_equal_units':sum(r['all_A_equal'] for r in half),'all_B_equal_units':sum(r['all_B_equal'] for r in half),
            'all_both_equal_units':sum(r['all_both_equal'] for r in half),
            'candidate_count_distribution':{str(k):sum(r['candidate_count']==k for r in half) for k in sorted({r['candidate_count'] for r in half})}},
        'calibration':calibration,'counts':dict(classes=len(classes),original_trajectories=len(units),unique_states=len(states),
            unique_state_draws=len(drawrows),selector_rows=len(selectors),selector_draw_rows=len(selector_draws)),
        'bootstrap':dict(class_order=classes,primary_seed=2026111601,auxiliary_seed=2026111602,shape=list(primary_draws.shape),
            primary_index_bytes_sha256=hashlib.sha256(primary_draws.tobytes(order='C')).hexdigest(),
            auxiliary_index_bytes_sha256=hashlib.sha256(auxiliary_draws.tobytes(order='C')).hexdigest())}
    tables={'state_summary.csv':state_stats,'selector_unit_means.csv':selector_units,'selector_draws.csv':selector_draws,
        'primary_contrasts.csv':primary,'auxiliary_contrasts.csv':auxiliary,'class_contrasts.csv':class_rows,
        'class_selector_means.csv':class_selector,'selector_summary.csv':selector_summary,'signed_contributions.csv':signed,
        'leave_one_class_out.csv':loco,'replicate_comparison.csv':rep_rows,'split_half.csv':half,
        'prediction_unit_errors.csv':pred_unit,'prediction_states.csv':frozen_pred_rows,'prediction_bins.csv':pred_bins,
        'prediction_calibration.csv':calibration}
    return tables,nums,primary_draws,auxiliary_draws

def main():
    # CLI reads 240 selectors/158 states/1264 trials; output is model-free numeric artifacts.
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--draws',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    inputs=args.root/'inputs'
    selectors=read_csv(inputs/'SELECTOR_MANIFEST.csv');states=read_csv(inputs/'UNIQUE_STATE_MANIFEST.csv')
    drawpath=args.draws or args.root/'numeric'/'STATE_DRAWS.csv'
    tables,nums,a,b=compute(selectors,states,read_csv(drawpath))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():write_csv(args.output/name,rows)
    np.save(args.output/'primary_bootstrap_index.npy',a,allow_pickle=False)
    np.save(args.output/'auxiliary_bootstrap_index.npy',b,allow_pickle=False)
    summary={'status':'COMPLETE_NUMERIC','scope':SCOPE,'numerical':nums,
        'input_hashes':{str(p.relative_to(args.root)) if p.is_relative_to(args.root) else str(p):sha(p) for p in [inputs/'SELECTOR_MANIFEST.csv',inputs/'UNIQUE_STATE_MANIFEST.csv',drawpath]},
        'scope_exclusions':['no model/decoder/LPIPS rerun','no independent raw-law audit','no unseen-class evaluation']}
    (args.output/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE_NUMERIC','counts':nums['counts'],'primary':nums['primary']},ensure_ascii=False))

if __name__=='__main__':main()
