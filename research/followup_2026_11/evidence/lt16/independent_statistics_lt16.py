"""Independent LT16 scalar audit; deliberately imports neither generator nor analyzer."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--draws',type=Path);ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();checks=[];n_values=0
    def read(path):
        with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    def mean(x):
        z=list(x);return math.fsum(float(t) for t in z)/len(z)
    def close(x,y,label):
        nonlocal n_values
        x=float(x);y=float(y);n_values+=1
        assert math.isfinite(x) and math.isfinite(y)
        assert math.isclose(x,y,rel_tol=2e-12,abs_tol=2e-14),(label,x,y)
    def key(r):return (int(r['class_id']),int(r['base_seed']),int(r['replicate_id']))
    def state(r):return key(r)+(int(r['step']),)
    selectors=read(a.root/'inputs'/'SELECTOR_MANIFEST.csv')
    states=read(a.root/'inputs'/'UNIQUE_STATE_MANIFEST.csv')
    trials=read(a.draws or a.root/'numeric'/'STATE_DRAWS.csv')
    ss={state(r):r for r in states};assert len(ss)==158
    uu=sorted({key(r) for r in states});cc=sorted({u[0] for u in uu})
    assert len(uu)==48 and len(cc)==24 and len(selectors)==240 and len(trials)==1264
    assert all(len([u for u in uu if u[0]==c])==2 for c in cc)
    dd={}
    for row in trials:
        k=(state(row),int(row['redraw_id']));assert k not in dd;dd[k]=row
        assert row['state_id']==ss[k[0]]['state_id']
    assert set(dd)=={(s,r) for s in ss for r in range(8)}
    sr={(key(r),r['selector']):r for r in selectors};names=['OLD_MAX','PROPOSED_OOF','ENTROPY_OOF','TIME_ONLY_FROZEN','FIXED_OOF']
    assert set(sr)=={(u,s) for u in uu for s in names};checks.append('complete semantic class/seed/rep/time/redraw keys')
    g={};changed={};mu={};h={};m={}
    for st in ss:
        vals=[];flags=[]
        for r in range(8):
            row=dd[(st,r)];x=float(row['lpips_pq'])-float(row['lpips_pr'])
            assert x==float(row['G_new']);vals.append(x)
            flag=str(row['post_state_changed']).lower() in ('true','1');flags.append(flag)
            assert flag==(int(row['changed_batch_count'])>0)
            if not flag:
                assert x==0 and int(row['final_hamming_qr'])==0 and str(row['final_QR_same']).lower() in ('true','1')
        g[st]=vals;changed[st]=flags;mu[st]=mean(vals);h[st]=sum(flags)/8
        m[st]=mean(v for v,f in zip(vals,flags) if f) if any(flags) else 0.
        close(mu[st],h[st]*m[st],('state identity',st))
    checks.append('current-redraw metric difference and unchanged-state zero identity')
    state_rows={state(r):r for r in read(a.analysis/'state_summary.csv')}
    assert set(state_rows)==set(ss)
    for st,row in state_rows.items():
        variance=math.fsum((x-mu[st])**2 for x in g[st])/7
        for col,val in [('mu_hat',mu[st]),('std',math.sqrt(variance)),('MCSE_state',math.sqrt(variance/8)),('h',h[st]),('m',m[st]),('hm',h[st]*m[st])]:close(row[col],val,(st,col))
        for col,val in [('positive_count',sum(x>0 for x in g[st])),('negative_count',sum(x<0 for x in g[st])),('zero_count',sum(x==0 for x in g[st])),('changed_count',sum(changed[st])),('changed_zero_gain_count',sum(f and x==0 for f,x in zip(changed[st],g[st])))]:
            assert int(row[col])==val
    checks.append('all 158 state means/variance/MCSE/positive-negative-zero/decomposition')
    selected={(u,s):state(sr[(u,s)]) for u in uu for s in names}
    U={(u,s):mu[selected[(u,s)]] for u in uu for s in names}
    O={(u,s):float(ss[selected[(u,s)]]['old_G']) for u in uu for s in names}
    unitrows={(key(r),r['selector']):r for r in read(a.analysis/'selector_unit_means.csv')}
    assert set(unitrows)==set(U)
    for k,row in unitrows.items():
        close(row['new_G'],U[k],('selector unit',k));close(row['old_G'],O[k],('old unit',k));close(row['new_minus_old'],U[k]-O[k],('change unit',k))
    aliasrows=read(a.analysis/'selector_draws.csv');assert len(aliasrows)==1920
    keys=set()
    for row in aliasrows:
        k=(key(row),row['selector'],int(row['redraw_id']));assert k not in keys;keys.add(k)
        st=selected[k[:2]];rr=k[2];assert row['state_id']==ss[st]['state_id']
        assert row['canonical_unit_id']==dd[(st,rr)]['unit_id'];close(row['G_new'],g[st][rr],('alias',k))
    checks.append('1920 selector references and 240 original-unit summaries')
    def cv(v):return np.array([mean(v[u] for u in uu if u[0]==c) for c in cc])
    arrays={}
    for nm,seed in [('primary',2026111601),('auxiliary',2026111602)]:
        mat=np.random.Generator(np.random.PCG64(seed)).integers(0,24,size=(20000,24),dtype=np.int64)
        observed=np.load(a.analysis/(nm+'_bootstrap_index.npy'),allow_pickle=False)
        assert np.array_equal(mat,observed);arrays[nm]=mat
    def estimate(v,kind):
        c=cv(v);bootstrap=np.take(c,arrays[kind]).mean(axis=1)
        q=[.05/6,1-.05/6] if kind=='primary' else [.025,.975]
        limits=np.quantile(bootstrap,q,method='linear')
        return mean(v.values()),float(limits[0]),float(limits[1])
    contrasts={
      'A':{u:U[(u,'OLD_MAX')] for u in uu},
      'B':{u:U[(u,'OLD_MAX')]-U[(u,'ENTROPY_OOF')] for u in uu},
      'C':{u:U[(u,'PROPOSED_OOF')]-U[(u,'TIME_ONLY_FROZEN')] for u in uu}}
    perdraw={}
    for n in contrasts:
        perdraw[n]={}
        for u in uu:
            aa='OLD_MAX' if n in ['A','B'] else 'PROPOSED_OOF'
            bb=None if n=='A' else 'ENTROPY_OOF' if n=='B' else 'TIME_ONLY_FROZEN'
            perdraw[n][u]=[g[selected[(u,aa)]][r]-(g[selected[(u,bb)]][r] if bb else 0.) for r in range(8)]
    main_rows={r['contrast']:r for r in read(a.analysis/'primary_contrasts.csv')}
    assert set(main_rows)==set(contrasts)
    expected_primary={}
    for n,v in contrasts.items():
        e,lo,hi=estimate(v,'primary');vv=[]
        for u in uu:
            z=perdraw[n][u];mz=mean(z);vv.append(math.fsum((x-mz)**2 for x in z)/7)
        mc=math.sqrt(math.fsum(vv)/(48*48*8))
        for col,x in [('estimate',e),('ci_low',lo),('ci_high',hi),('MCSE_fixed_parents',mc)]:close(main_rows[n][col],x,(n,col))
        expected_primary[n]=dict(estimate=e,ci_low=lo,ci_high=hi,MCSE_fixed_parents=mc)
    checks.append('three co-primary 98.3333333333% class intervals and paired MCSE independently recomputed')
    clrows={(int(r['class_id']),r['contrast']):r for r in read(a.analysis/'class_contrasts.csv')}
    locorows={(r['contrast'],int(r['excluded_class'])):r for r in read(a.analysis/'leave_one_class_out.csv')}
    reprows={(r['contrast'],int(r['replicate_id'])):r for r in read(a.analysis/'replicate_comparison.csv')}
    for n,v in contrasts.items():
        cvals=cv(v)
        for i,c in enumerate(cc):
            close(clrows[(c,n)]['value'],cvals[i],('class',c,n))
            close(locorows[(n,c)]['estimate'],mean(cvals[j] for j in range(24) if j!=i),('loco',n,c))
        for rep in [0,1]:close(reprows[(n,rep)]['estimate'],mean(v[u] for u in uu if u[2]==rep),('rep',n,rep))
    checks.append('class contrasts, every-class deletion sensitivity, parent replicate summaries')
    aux={}
    aux['PROPOSED_OOF-ENTROPY_OOF']={u:U[(u,'PROPOSED_OOF')]-U[(u,'ENTROPY_OOF')] for u in uu}
    aux['OLD_MAX-FIXED_OOF']={u:U[(u,'OLD_MAX')]-U[(u,'FIXED_OOF')] for u in uu}
    for s in names:
        aux[s]={u:U[(u,s)] for u in uu}
        aux[s+'_NEW_MINUS_OLD']={u:U[(u,s)]-O[(u,s)] for u in uu}
    hsrows={key(r):r for r in read(a.analysis/'split_half.csv')};assert set(hsrows)==set(uu)
    halfterms={name:{} for name in ['self_value','transfer_value','selection_optimism']}
    for u in uu:
        kk=sorted(st for st in ss if st[:3]==u);a4={st:mean(g[st][:4]) for st in kk};b4={st:mean(g[st][4:]) for st in kk}
        ia=min(kk,key=lambda st:(-a4[st],st[3]));ib=min(kk,key=lambda st:(-b4[st],st[3]))
        selfv=(a4[ia]+b4[ib])/2;cross=(b4[ia]+a4[ib])/2
        rr=hsrows[u];assert int(rr['t_A'])==ia[3] and int(rr['t_B'])==ib[3]
        assert int(rr['candidate_count'])==len(kk)
        for name,val in [('self_value',selfv),('transfer_value',cross),('selection_optimism',selfv-cross)]:
            halfterms[name][u]=val;close(rr[name],val,('half',u,name))
    for name,v in halfterms.items():aux['SPLIT_HALF_'+name.upper()]=v
    checks.append('split-half unique-time candidate sets, independent winners, cross-half values')
    perrorrows={key(r):r for r in read(a.analysis/'prediction_unit_errors.csv')}
    ecols=['old_state_MSE','old_time_MSE','old_state_minus_time_MSE','new_state_MSE','new_time_MSE','new_state_minus_time_MSE']
    err={col:{} for col in ecols}
    for u in uu:
        kk=[st for st in ss if st[:3]==u]
        for target in ['old','new']:
            yy={st:float(ss[st]['old_G']) if target=='old' else mu[st] for st in kk}
            ps=mean((float(ss[st]['old_prediction'])-yy[st])**2 for st in kk)
            pt=mean((float(ss[st]['m_t'])-yy[st])**2 for st in kk)
            for tail,val in [('state_MSE',ps),('time_MSE',pt),('state_minus_time_MSE',ps-pt)]:
                col=target+'_'+tail;err[col][u]=val;close(perrorrows[u][col],val,('prediction MSE',u,col))
    for col,v in err.items():aux['PREDICTION_'+col]=v
    auxrows={r['contrast']:r for r in read(a.analysis/'auxiliary_contrasts.csv')};assert set(auxrows)==set(aux)
    for n,v in aux.items():
        e,lo,hi=estimate(v,'auxiliary')
        for col,val in [('estimate',e),('ci_low',lo),('ci_high',hi)]:close(auxrows[n][col],val,('aux interval',n,col))
    checks.append('all auxiliary 95% intervals, including equally weighted prediction-MSE differences')
    sums={r['selector']:r for r in read(a.analysis/'selector_summary.csv')}
    signs={r['selector']:r for r in read(a.analysis/'signed_contributions.csv')}
    class_sel={(int(r['class_id']),r['selector']):r for r in read(a.analysis/'class_selector_means.csv')}
    for s in names:
        HH=mean(h[selected[(u,s)]] for u in uu);MM=mean(m[selected[(u,s)]] for u in uu)
        covariance=mean((h[selected[(u,s)]]-HH)*(m[selected[(u,s)]]-MM) for u in uu)
        total=mean(U[(u,s)] for u in uu);rr=sums[s]
        for col,x in [('new_G',total),('old_G',mean(O[(u,s)] for u in uu)),('H_observed_change',HH),('M_equal_sample',MM),('covariance_ddof0',covariance),('H_times_Mequal',HH*MM),('H_Mequal_plus_cov',HH*MM+covariance)]:
            close(rr[col],x,(s,col))
        if HH:close(rr['M_pooled_changed'],total/HH,(s,'pooled changed'))
        else:assert rr['M_pooled_changed']==''
        flat=[x for u in uu for x in g[selected[(u,s)]]]
        for name,which in [('positive',lambda x:x>0),('negative',lambda x:x<0),('zero',lambda x:x==0)]:
            subset=[x for x in flat if which(x)];close(signs[s]['p_'+name],len(subset)/len(flat),('sign fraction',s,name))
            if name!='zero':
                col='mean_'+name
                if subset:close(signs[s][col],mean(subset),('sign conditional',s,name))
                else:assert signs[s][col]==''
                close(signs[s][name+'_contribution'],math.fsum(subset)/len(flat),('sign contribution',s,name))
        for c in cc:
            rr=class_sel[(c,s)]
            close(rr['new_G'],mean(U[(u,s)] for u in uu if u[0]==c),('class selector',c,s))
    checks.append('all selector H/M/covariance/pooled impact, signed contributions and class means')
    # Independently validate fixed prediction-bin means using unit-equal unique-time mass.
    edges=[-math.inf,-.01,0.,.005,.01,.02,math.inf]
    bins={(r['predictor'],int(r['bin_index'])):r for r in read(a.analysis/'prediction_bins.csv')}
    calibrations={(r['predictor'],r['target']):r for r in read(a.analysis/'prediction_calibration.csv')}
    for predictor in ['old_prediction','m_t']:
        triples=[]
        for u in uu:
            kk=[st for st in ss if st[:3]==u]
            for st in kk:triples.append((float(ss[st][predictor]),float(ss[st]['old_G']),mu[st],1/(48*len(kk))))
        for i in range(6):
            sub=[x for x in triples if edges[i]<=x[0]<edges[i+1]];mass=math.fsum(x[3] for x in sub)
            rr=bins[(predictor,i)];assert int(rr['states'])==len(sub);close(rr['trajectory_equal_weight_mass'],mass,('bin mass',predictor,i))
            for col,index in [('prediction_mean',0),('old_G_mean',1),('new_G_mean',2)]:
                if mass:close(rr[col],math.fsum(x[index]*x[3] for x in sub)/mass,('bin',predictor,i,col))
                else:assert rr[col]==''
        for target,j in [('old_G',1),('mu_hat',2)]:
            row=calibrations[(predictor,target)]
            mse=math.fsum((x[0]-x[j])**2*x[3] for x in triples)
            close(row['weighted_MSE'],mse,('weighted MSE',predictor,target))
    checks.append('frozen absolute prediction bins and unit-equal unique-time calibration')
    analysis=json.load((a.analysis/'analysis.json').open());boot=analysis['numerical']['bootstrap']
    for name in ['primary','auxiliary']:
        assert boot[name+'_index_bytes_sha256']==hashlib.sha256(arrays[name].tobytes()).hexdigest()
    for row in analysis['numerical']['primary']:
        for col,val in expected_primary[row['contrast']].items():close(row[col],val,('JSON primary',row['contrast'],col))
    result={'status':'PASS','scope':'independently coded scalar sampling-output aggregation; no generator/analyzer imports; no model/law/decoder/LPIPS rerun',
        'checks':checks,'numeric_values_compared':n_values,'class_order':cc,'counts':{'classes':24,'trajectories':48,'states':158,'redraws':1264,'selector_draw_references':1920},
        'primary_independent':expected_primary,'primary_bootstrap_sha256':boot['primary_index_bytes_sha256'],'auxiliary_bootstrap_sha256':boot['auxiliary_index_bytes_sha256'],
        'tolerance':{'absolute':2e-14,'relative':2e-12},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','numeric_values_compared':n_values,'checks':len(checks)}))
if __name__=='__main__':main()
