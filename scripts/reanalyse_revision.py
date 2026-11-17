#!/usr/bin/env python3
"""Post-hoc CPU reanalysis of immutable public block tables (v1.0.1).

One draw resamples complete blocks, sharing its indices across rho and metrics.
No model, image metric, feature pipeline, or LOBO model is executed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from reproduce_public_results import paired_values, reproduce

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'public_results/reanalysis_v1_0_1'
TEX = ROOT/'paper/generated/v1.0.1'
RESAMPLES = 100_000
NAMESPACE = 'FDCA-v1.0.1-POSTHOC-20261029'
METRICS = ('H', 'M_equal', 'R', 'M_split', 'mean_coordinate_TV', 'DINO_equal', 'R_DINO')
LABELS = {'Tiny_CFG3_W8':'Tiny CFG3 W8', 'Small_CFG3_W8':'Small CFG3 W8', 'Small_CFG1_W8':'Small CFG1 W8', 'Tiny_CFG3_W4':'Tiny W4 stress'}


def seed(label):
    return int.from_bytes(hashlib.sha256(f'{NAMESPACE}|{label}'.encode('utf-8')).digest()[:8], 'big')


def interval(draw, alpha=.05):
    if not np.isfinite(draw).all():
        return [None, None]
    return [float(x) for x in np.quantile(draw, [alpha/2, 1-alpha/2], method='linear')]


def ratio(numerator, denominator):
    """An absent conditional expectation is undefined, never a fabricated zero."""
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def aggregate(h, m):
    h, m = np.asarray(h, float), np.asarray(m, float)
    if h.shape != m.shape or not np.isfinite(h).all() or np.any((h<0)|(h>1)):
        raise ValueError('invalid split probabilities')
    if not np.isfinite(m[h>0]).all() or np.any(m[np.isfinite(m)]<0):
        raise ValueError('missing or invalid conditional metric at positive h')
    risk = np.zeros_like(h)
    np.multiply(h, m, out=risk, where=h>0)
    H, R = float(h.mean()), float(risk.mean())
    # h=0 does not create an observed conditional m; M_equal is undefined if any are absent.
    M = float(m.mean()) if np.isfinite(m).all() else None
    covariance = float(np.mean((h-H)*(m-M))) if M is not None else None
    return {'H':H, 'M_equal':M, 'R':R, 'M_split':float(R/H) if H>0 else None,
            'product_of_means':H*M if M is not None else None, 'population_covariance':covariance,
            'product_shortfall_percent':100*(R-H*M)/R if M is not None and R>0 else None}


def schedule():
    """NumPy float32 analogue of adapter's torch float32 schedule; no torch import."""
    cumulative = []
    for index in range(32):
        angle = np.arccos(np.float32((index+1)/32))
        # torch.pi*.5 is rounded to the tensor's float32 dtype during division.
        r = np.float32(1) - angle / np.float32(np.pi*.5)
        cumulative.append(max(index+1, int(r*np.float32(256))))
    previous = np.array([0]+cumulative[:-1])
    unresolved = 256-previous
    rows=[]
    for rho in [.1,.3,.5,.7,.9]:
        index = int(np.argmin(abs(unresolved/256-rho)))
        rows.append({'target_rho':rho, 'transition':index+1, 'unresolved_tokens':int(unresolved[index]),
                     'attained_rho':float(unresolved[index]/256), 'commit_batch_c':int(cumulative[index]-previous[index]),
                     'final_flush':index==31})
    expected = [(32,41,41),(30,72,14),(24,126,8),(16,177,6),(6,231,5)]
    if [(r['transition'],r['unresolved_tokens'],r['commit_batch_c']) for r in rows] != expected:
        raise AssertionError('float32 schedule differs from recorded contract')
    return rows


def load_inputs():
    p2 = pd.read_csv(ROOT/'public_results/tables/P2P_BLOCK_LEVEL.csv')
    p3 = pd.read_csv(ROOT/'public_results/tables/P3S_BLOCK_LEVEL.csv')
    panels = {'Tiny_CFG3_W8':p2[p2.cell_id.eq('W8_L1.00')].copy(),
              'Tiny_CFG3_W4':p2[p2.cell_id.eq('W4_L1.00')].copy(),
              'Small_CFG3_W8':p3[p3.panel.eq('primary_cfg3')].copy(),
              'Small_CFG1_W8':p3[p3.panel.eq('official_cfg1')].copy()}
    expected={'Tiny_CFG3_W8':(48,5),'Tiny_CFG3_W4':(24,2),'Small_CFG3_W8':(40,2),'Small_CFG1_W8':(16,2)}
    for name,cfg in [('Small_CFG3_W8',3.0),('Small_CFG1_W8',1.0)]:
        if set(panels[name].cfg_weight.unique())!={cfg}:raise ValueError(f'{name}: unexpected CFG filter')
    schedule_map={r['target_rho']:r['commit_batch_c'] for r in schedule()}
    for name,f in panels.items():
        n,anchors=expected[name]
        if f.duplicated(['block_id','anchor_target_rho']).any() or f.block_id.nunique()!=n or len(f)!=n*anchors:
            raise ValueError(f'{name}: unexpected unique block/anchor schema')
        if len(f.anchor_target_rho.unique()) != anchors:
            raise ValueError(f'{name}: unexpected anchor count')
        for col in ['split_incidence','lpips_alex','unconditional_lpips_alex']:
            paired_values(f,col)
        f['H']=f.split_incidence
        f['M_equal']=f.lpips_alex
        f['R']=f.H*f.M_equal
        if not np.allclose(f.R,f.unconditional_lpips_alex,rtol=1e-12,atol=1e-15):
            raise AssertionError(f'{name}: stored block product differs')
        f['DINO_equal']=f.dino_cosine_distance
        f['R_DINO']=f.H*f.DINO_equal
        f['mean_coordinate_TV']=f.expected_seed_count/f.anchor_target_rho.map(schedule_map)
    if not set(panels['Tiny_CFG3_W4'].block_id)<=set(panels['Tiny_CFG3_W8'].block_id):
        raise ValueError('W4 is not a W8 block subset')
    if not set(panels['Small_CFG1_W8'].block_id)<=set(panels['Small_CFG3_W8'].block_id):
        raise ValueError('CFG1 is not a CFG3 block subset')
    # Cross-check frozen summaries without rewriting them.
    for phase, summary_name in [('P2P','P2P_PERCEPTUAL_SUMMARY.csv'),('P3S','P3S_REPLICATION_SUMMARY.csv')]:
        summary=pd.read_csv(ROOT/'public_results/tables'/summary_name)
        for _, row in summary.iterrows():
            name=('Tiny_CFG3_W8' if row.cell_id=='W8_L1.00' else 'Tiny_CFG3_W4') if phase=='P2P' else ('Small_CFG3_W8' if row.panel=='primary_cfg3' else 'Small_CFG1_W8')
            f=panels[name];g=f[f.anchor_target_rho.eq(row.anchor_target_rho)]
            for source, column in [('split_incidence','H'),('conditional_lpips','M_equal'),('unconditional_lpips','R'),('conditional_dino','DINO_equal'),('unconditional_dino','R_DINO')]:
                if not np.isclose(g[column].mean(),row[source],rtol=1e-12,atol=1e-14):raise AssertionError(f'{phase} frozen summary {source}')
            if len(g)!=int(row.blocks):raise AssertionError('summary block count')
    for phase,name in [('P2P','Tiny_CFG3_W8'),('P3S','Small_CFG3_W8')]:
        oof=pd.read_csv(ROOT/f'public_results/tables/{phase}_TOKEN_OOF.csv')
        counts=oof.groupby(['block_id','anchor_target_rho']).size()
        if not (counts==8).all() or set(counts.index)!=set(zip(panels[name].block_id,panels[name].anchor_target_rho)):
            raise AssertionError('OOF primary conditional-repeat keys differ')
    return panels


def inference(frame,name,resamples=RESAMPLES,rhos=(.3,.5)):
    metrics=[x for x in METRICS if x!='M_split']
    arrays=np.stack([paired_values(frame,x,rhos).to_numpy() for x in metrics],axis=2) # block,rho,metric
    n=len(arrays)
    ix=np.random.default_rng(seed(name)).integers(0,n,(resamples,n))
    means=arrays.mean(axis=0)
    # Bound peak memory: re-use the same draw for all endpoints, metric by metric.
    draws=np.stack([arrays[:,:,j][ix].mean(axis=1) for j in range(len(metrics))],axis=2)
    records=[]
    for metric in METRICS:
        if metric=='M_split':
            values=ratio(means[:,metrics.index('R')],means[:,metrics.index('H')])
            samples=ratio(draws[:,:,metrics.index('R')],draws[:,:,metrics.index('H')])
            block_values=None
        else:
            j=metrics.index(metric);values=means[:,j];samples=draws[:,:,j];block_values=arrays[:,:,j]
        diffs=samples[:,1]-samples[:,0]
        z={'panel':name,'metric':metric,'n':n,'rho_low':rhos[0],'rho_high':rhos[1],
           'low':float(values[0]),'high':float(values[1]),'delta':float(values[1]-values[0]),
           'ci_low':interval(diffs)[0],'ci_high':interval(diffs)[1],
           'low_ci':interval(samples[:,0]),'high_ci':interval(samples[:,1]),
           'ratio_high_over_low':float(ratio(values[1],values[0])),
           'ratio_ci':interval(ratio(samples[:,1],samples[:,0])),
           'resamples':resamples,'quantile':'linear','seed':seed(name)}
        if block_values is not None:
            d=block_values[:,1]-block_values[:,0]
            se=stats.sem(d);t=stats.t.ppf(.975,n-1)
            z.update({'paired_t_ci':[float(d.mean()-t*se),float(d.mean()+t*se)],
                      'loo_delta_min':float(((d.sum()-d)/(n-1)).min()),
                      'loo_delta_max':float(((d.sum()-d)/(n-1)).max()),
                      'trimmed_mean_10pct':float(stats.trim_mean(d,.1)),
                      'blocks_positive':int((d>0).sum()),'blocks_negative':int((d<0).sum()),'blocks_zero':int((d==0).sum())})
        if name in ('Tiny_CFG3_W8','Small_CFG3_W8') and metric in ('H','M_equal','R') and rhos==(.3,.5):
            z['bonferroni_ci_6']=interval(diffs,.05/6)
        records.append(z)
    return records


def difference_interaction(left,right,name,resamples):
    """Both conditions are restricted explicitly to the predeclared common subset."""
    ids=set(left.block_id)
    if not ids<=set(right.block_id):raise ValueError('subset pairing failed')
    b=right[right.block_id.isin(ids)]
    rows=[]
    for metric in ('H','M_equal','R'):
        a=paired_values(left,metric);c=paired_values(b,metric)
        if not a.index.equals(c.index):raise ValueError('interaction keys differ')
        d=(a[.5]-a[.3]).to_numpy()-(c[.5]-c[.3]).to_numpy()
        ix=np.random.default_rng(seed(name)).integers(0,len(d),(resamples,len(d)))
        ci=interval(d[ix].mean(1))
        rows.append({'comparison':name,'metric':metric,'n':len(d),'delta_difference':float(d.mean()),'ci_low':ci[0],'ci_high':ci[1], 'seed':seed(name)})
    return rows


def write_csv(stem,rows):
    pd.DataFrame(rows).to_csv(OUT/'tables'/f'{stem}.csv',index=False,float_format='%.15g')


def tex_table(stem,header,rows,align=None):
    lines=[r'{\small\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+(align or ('l'+'r'*(len(header)-1)))+'}',r'\hline',' & '.join(header)+r' \\',r'\hline']
    lines += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    lines += [r'\hline',r'\end{tabular}',r'}']
    (TEX/f'{stem}.tex').write_text('\n'.join(lines)+'\n')


def num(x):return '--' if x is None else f'{x:.6f}'


def ci(z):return '['+num(z[0])+', '+num(z[1])+']'


def generated_tables(result):
    contrasts=result['contrasts'];legacy=result['legacy_reproduction']
    main=[]
    for name in ('Tiny_CFG3_W8','Small_CFG3_W8'):
        phase='P2P' if name.startswith('Tiny') else 'P3S'
        for metric,key in [('H',None),('M_equal','H1' if phase=='P2P' else 'R1'),('R','H2' if phase=='P2P' else 'R2')]:
            z=next(z for z in contrasts if z['panel']==name and z['metric']==metric)
            interval_legacy=ci([legacy[phase+'_'+key]['ci_lower'],legacy[phase+'_'+key]['ci_upper']]) if key else r'--$^\dagger$'
            main.append([LABELS[name],r'$'+metric.replace('_',r'_{')+('}' if '_' in metric else '')+'$',z['n'],num(z['low']),num(z['high']),num(z['delta']),interval_legacy])
    tex_table('main_panels',['Panel','Metric','$n$',r'$\rho=.3$',r'$\rho=.5$',r'$\Delta$',r'Legacy 95\% CI'],main)
    tex_table('posthoc_contrasts',['Panel','Metric','$n$',r'$\Delta$',r'Post-hoc 95\% CI'],[[LABELS[z['panel']],z['metric'].replace('_',r'\_'),z['n'],num(z['delta']),ci([z['ci_low'],z['ci_high']])] for z in contrasts if z['metric'] in ('H','M_equal','R')])
    tex_table('aggregation',['Panel',r'$\rho$','$M_{equal}$','$M_{split}$','$R$','$H M_{equal}$','Covariance'],[[LABELS[z['panel']],z['rho'],num(z['M_equal']),num(z['M_split']),num(z['R']),num(z['product_of_means']),num(z['population_covariance'])] for z in result['aggregation'] if z['rho'] in (.3,.5) and z['panel'] in ('Tiny_CFG3_W8','Small_CFG3_W8')])
    tex_table('schedule_tv',['Panel',r'$c_{.3}\to c_{.5}$','TV at .3','TV at .5',r'$\Delta$',r'Post-hoc 95\% CI'],[[LABELS[z['panel']],r'$14\to8$',num(z['low']),num(z['high']),num(z['delta']),ci([z['ci_low'],z['ci_high']])] for z in contrasts if z['metric']=='mean_coordinate_TV' and z['panel'] in ('Tiny_CFG3_W8','Small_CFG3_W8')])
    tex_table('schedule',[r'Target $\rho$','Transition','Unresolved',r'Attained $\rho$','$c$'],[[z['target_rho'],z['transition'],z['unresolved_tokens'],z['attained_rho'],z['commit_batch_c']] for z in result['schedule']])
    tex_table('attribution',['Panel','$A_h$','$A_m$',r'$\Delta R$','Joint direction'],[[LABELS[z['panel']],num(z['A_h']),num(z['A_m']),num(z['delta_R']),str(z['joint_H_down_M_up_R_up'])+'/'+str(z['n'])] for z in result['attribution']])
    tex_table('matched',['Panel','$n$','.3 LPIPS','.5 LPIPS',r'$\Delta$',r'Legacy 95\% CI'],[[z['panel'],z['n'],num(z['low']),num(z['high']),num(z['delta']),ci(z['legacy_ci'])] for z in result['matched']])
    robust=[z for z in contrasts if 'bonferroni_ci_6' in z]
    tex_table('robustness',['Panel','Metric',r'Paired $t$ 95\% CI','Bonferroni CI'],[[LABELS[z['panel']],z['metric'].replace('_',r'\_'),ci(z['paired_t_ci']),ci(z['bonferroni_ci_6'])] for z in robust])
    tex_table('robustness_loo',['Panel','Metric','LOO range',r'10\% trim'],[[LABELS[z['panel']],z['metric'].replace('_',r'\_'),ci([z['loo_delta_min'],z['loo_delta_max']]),num(z['trimmed_mean_10pct'])] for z in robust])
    p=TEX/'robustness.tex';p.write_text(p.read_text()+r'\par\medskip'+'\n'+(TEX/'robustness_loo.tex').read_text())
    tex_table('interactions',['Comparison','Metric','$n$',r'Difference in $\Delta$',r'Post-hoc 95\% CI'],[['CFG1--CFG3' if z['comparison'].startswith('CFG1') else 'W4--W8 stress',z['metric'].replace('_',r'\_'),z['n'],num(z['delta_difference']),ci([z['ci_low'],z['ci_high']])] for z in result['interactions']])


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':NAMESPACE})
    colors={'Tiny_CFG3_W8':'#2267A7','Small_CFG3_W8':'#CB572E'}
    def save(fig,stem,foot):
        fig.text(.5,.015,textwrap.fill(foot,width=110),ha='center',va='bottom',fontsize=7)
        fig.tight_layout(rect=(0,.13,1,1))
        for suffix in ['pdf','svg','png']:
            metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else ({'Date':None} if suffix=='svg' else {'Software':'FDCA v1.0.1 CPU reanalysis'})
            fig.savefig(OUT/'figures'/f'{stem}.{suffix}',dpi=190,metadata=metadata)
            if suffix == 'svg':
                path = OUT/'figures'/f'{stem}.svg'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        shutil.copy2(OUT/'figures'/f'{stem}.pdf',TEX/f'{stem}.pdf')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.1))
    for ax,metric,title in zip(axes,['H','M_equal','R'],['Batch split\nprobability H','Equal-block\nconditional LPIPS','Mean block\nproduct R']):
        for name,color in colors.items():
            z=next(x for x in result['contrasts'] if x['panel']==name and x['metric']==metric)
            y=np.array([z['low'],z['high']]);bounds=np.array([z['low_ci'],z['high_ci']]);err=np.array([y-bounds[:,0],bounds[:,1]-y])
            ax.errorbar([.3,.5],y,yerr=err,fmt='o-',capsize=3,color=color,label=LABELS[name]+f", n={z['n']}")
        ax.set(title=title,xlabel='Target rho',xticks=[.3,.5]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='lower left')
    save(fig,'main_panels','Post-hoc 100,000 paired-block bootstrap, linear 95% intervals. Blocks are class-seed units; CFG3 W8 only.')
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.1))
    for ax,metric in zip(axes,['H','M_equal','R']):
        for j,(name,color) in enumerate(colors.items()):
            z=next(x for x in result['contrasts'] if x['panel']==name and x['metric']==metric)
            ax.errorbar(z['delta'],j,xerr=[[z['delta']-z['ci_low']],[z['ci_high']-z['delta']]],fmt='o',capsize=3,color=color)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.axvline(0,color='gray',lw=.8);ax.set(title=metric.replace('_',' '),xlabel='Delta (.5 - .3)',yticks=[0,1],yticklabels=['Tiny n=48','Small n=40']);ax.grid(alpha=.2)
    save(fig,'paired_contrasts','CFG3 W8 panels; post-hoc 100,000 paired-block bootstrap, linear 95% intervals. Panels analysed separately, without cross-checkpoint pairing.')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
    sch=result['schedule'];axes[0].plot([r['target_rho'] for r in sch],[r['commit_batch_c'] for r in sch],'o-',color='#555555');axes[0].set(xlabel='Target rho',ylabel='Commit batch c',title='Fixed 32-step Halton schedule')
    for name,color in colors.items():
        z=next(x for x in result['contrasts'] if x['panel']==name and x['metric']=='mean_coordinate_TV');y=np.array([z['low'],z['high']]);b=np.array([z['low_ci'],z['high_ci']]);axes[1].errorbar([.3,.5],y,yerr=[y-b[:,0],b[:,1]-y],fmt='o-',capsize=3,color=color,label=LABELS[name])
    axes[1].set(xlabel='Target rho',ylabel='sum coordinate TV / c',title='Mean coordinate TV, CFG3 W8',xticks=[.3,.5]);axes[1].legend(fontsize=7)
    save(fig,'schedule_tv','c is deterministic, without sampling CI; TV intervals use post-hoc paired-block bootstrap. rho=.1 is final flush.')
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.1))
    tiny=[r for r in result['aggregation'] if r['panel']=='Tiny_CFG3_W8']
    for ax,metric in zip(axes,['H','M_equal','R']):
        ax.plot([r['rho'] for r in tiny],[r[metric] for r in tiny],'o-',color=colors['Tiny_CFG3_W8']);ax.axvspan(.3,.5,color='#2267A7',alpha=.08);ax.set(xlabel='Target rho',title=metric.replace('_',' '),xticks=[.1,.3,.5,.7,.9]);ax.grid(alpha=.2)
    save(fig,'tiny_all_rho','Tiny CFG3 W8, n=48 paired blocks; means over all five recorded anchors. Shading marks selected .3/.5 contrast; no CI shown.')


def main():
    global OUT, TEX
    import matplotlib
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resamples',type=int,default=RESAMPLES);parser.add_argument('--check',action='store_true',help='Recompute in a temporary directory and compare stored numeric/table artifacts');args=parser.parse_args()
    if args.resamples!=RESAMPLES:raise SystemExit('Published supplement requires exactly 100000 resamples')
    stored_out,stored_tex=OUT,TEX
    temporary=tempfile.TemporaryDirectory(prefix='fdca-reanalysis-check-') if args.check else None
    if temporary is not None:
        OUT=Path(temporary.name)/'results';TEX=Path(temporary.name)/'latex'
    for p in [OUT/'tables',OUT/'figures',TEX]:p.mkdir(parents=True,exist_ok=True)
    panels=load_inputs();contrasts=[];aggregation=[];attribution=[]
    for name,f in panels.items():
        contrasts+=inference(f,name)
        for rho,g in f.groupby('anchor_target_rho'):
            z=aggregate(g.H,g.M_equal);z.update({'panel':name,'rho':float(rho),'n':len(g)});aggregation.append(z)
        h=paired_values(f,'H');m=paired_values(f,'M_equal');r=paired_values(f,'R')
        ah=((h[.5]-h[.3])*(m[.5]+m[.3])/2).to_numpy();am=((m[.5]-m[.3])*(h[.5]+h[.3])/2).to_numpy();dr=(r[.5]-r[.3]).to_numpy()
        if not np.allclose(ah+am,dr,rtol=1e-12,atol=1e-15):raise AssertionError('blockwise attribution identity failed')
        ix=np.random.default_rng(seed(name)).integers(0,len(ah),(RESAMPLES,len(ah)))
        attribution.append({'panel':name,'n':len(h),'A_h':float(ah.mean()),'A_m':float(am.mean()),'delta_R':float(dr.mean()),'A_h_ci':interval(ah[ix].mean(1)),'A_m_ci':interval(am[ix].mean(1)), 'joint_H_down_M_up_R_up':int(((h[.5]<h[.3])&(m[.5]>m[.3])&(r[.5]>r[.3])).sum())})
    # Full-rho diagnostic guards against an unqualified monotonicity claim.
    adjacent=[]
    for low,high in zip([.1,.3,.5,.7],[.3,.5,.7,.9]):
        adjacent+=inference(panels['Tiny_CFG3_W8'],f'Tiny_adjacent_{low}_{high}',rhos=(low,high))
    legacy=reproduce();matched=[]
    for phase,key,expected in [('P2P','H5',24),('P3S','R4',40)]:
        f=pd.read_csv(ROOT/f'public_results/tables/{phase}_MATCHED_BLOCK_LEVEL.csv');p=paired_values(f,'lpips_alex')
        if len(p)!=expected:raise ValueError('matched n mismatch')
        parent=panels['Tiny_CFG3_W8' if phase=='P2P' else 'Small_CFG3_W8']
        if not set(p.index)<=set(parent.block_id):raise ValueError('matched subset mismatch')
        d=(p[.5]-p[.3]).to_numpy();ix=np.random.default_rng(seed(phase+'_matched')).integers(0,len(d),(RESAMPLES,len(d)))
        z=legacy[phase+'_'+key]
        matched.append({'panel':phase+' matched','n':len(d),'low':float(p[.3].mean()),'high':float(p[.5].mean()),'delta':float(d.mean()),'posthoc_ci':interval(d[ix].mean(1)),'legacy_ci':[z['ci_lower'],z['ci_upper']], 'seed':seed(phase+'_matched')})
    interactions=difference_interaction(panels['Small_CFG1_W8'],panels['Small_CFG3_W8'],'CFG1_minus_CFG3_common16',RESAMPLES)+difference_interaction(panels['Tiny_CFG3_W4'],panels['Tiny_CFG3_W8'],'W4_minus_W8_common24_stress',RESAMPLES)
    result={'analysis_status':'POST_HOC_ROBUSTNESS_SUPPLEMENT_NOT_PREREGISTERED','method':{'resamples':RESAMPLES,'quantile':'linear','rng':'NumPy default_rng / PCG64','label_to_seed':'first 8 bytes SHA256(namespace + pipe + UTF-8 label), unsigned big endian','namespace':NAMESPACE,'sample_unit':'complete within-panel class-seed block; all anchors and metrics retained'},'schedule':schedule(),'contrasts':contrasts,'aggregation':aggregation,'attribution':attribution,'matched':matched,'interactions':interactions,'tiny_adjacent':adjacent,'legacy_reproduction':legacy,
            'limitations':['Compact tables lack complete class_id/base_seed manifest: class-cluster inference and independent audit of class/seed or predecessor overlap are unavailable. Block IDs verify only within-panel subsets.', 'P2P compact tables identify W8/W4 cells but omit CFG metadata; CFG3 comes from the preserved protocol and audit rather than independent table filtering.', 'The compact tables do not contain all per-coordinate categorical marginals or terminal images: this analysis cannot independently recompute product-form h, LPIPS/DINO, or within-block rollout Monte Carlo error from first principles.', 'Block bootstrap assumes independent equally weighted class-seed blocks in these selected panels; it does not establish an ImageNet-wide or checkpoint-wide probability sample.','Conditional means have Monte Carlo error from 8 repeats in W8 primary, 4 in CFG1/W4, 2 in matched timing. Raw repeat-level variance is not recoverable from block means.','Matched compact files lack seed token and coordinate identity; execution-audit claims are not an independent raw identity verification.','Stored OOF MAE reaggregation does not retrain LOBO models or reconstruct original features. Descendant features are retrospective, not online inputs.','Coordinate TV is a mean of marginals, not the full coordinate distribution; fixed schedule, state, position, seed composition and remaining horizon co-vary.','Bonferroni family contains exactly six primary CFG3 W8 H/M_equal/R contrasts; it does not correct study-wide post-hoc selection.']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for stem,key in [('contrasts','contrasts'),('aggregation','aggregation'),('attribution','attribution'),('matched','matched'),('interactions','interactions'),('tiny_adjacent','tiny_adjacent'),('schedule','schedule')]:write_csv(stem,result[key])
    write_csv('bonferroni_six',[{k:z[k] for k in ['panel','metric','n','delta','bonferroni_ci_6']} for z in contrasts if 'bonferroni_ci_6' in z])
    generated_tables(result);figures(result)
    headline=['| Panel | Metric | n | rho .3 | rho .5 | Delta (.5-.3) | Legacy 95% CI |', '|---|---|---:|---:|---:|---:|---|']
    for name,phase in [('Tiny_CFG3_W8','P2P'),('Small_CFG3_W8','P3S')]:
        for metric,key in [('H',None),('M_equal','H1' if phase=='P2P' else 'R1'),('R','H2' if phase=='P2P' else 'R2')]:
            z=next(x for x in contrasts if x['panel']==name and x['metric']==metric)
            interval_legacy=ci([legacy[phase+'_'+key]['ci_lower'],legacy[phase+'_'+key]['ci_upper']]) if key else 'Not a frozen headline endpoint; see post-hoc supplement'
            headline.append('| '+' | '.join([LABELS[name],metric,str(z['n']),num(z['low']),num(z['high']),num(z['delta']),interval_legacy])+' |')
    (OUT/'tables/headline.md').write_text('\n'.join(headline)+'\n')
    paths=sorted((ROOT/'public_results/tables').glob('*.csv'))+sorted((ROOT/'public_results/summaries').glob('P2P*.json'))+sorted((ROOT/'public_results/summaries').glob('P3S*.json'))+[ROOT/'scripts/reanalyse_revision.py',ROOT/'scripts/reproduce_public_results.py',ROOT/'fdca/adapters/halton_maskgit_contract.py',ROOT/'fdca/analysis/p2n_natural_shock.py',ROOT/'fdca/analysis/p2p_perceptual.py',ROOT/'fdca/analysis/p3s_replication.py']+sorted((ROOT/'configs/preregistrations').glob('*p2p*.yaml'))+sorted((ROOT/'configs/preregistrations').glob('*p3s*.yaml'))
    audit={'status':'PASS','inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},'counts':{name:{'blocks':int(f.block_id.nunique()),'rows':len(f),'anchors':sorted(f.anchor_target_rho.unique().tolist())} for name,f in panels.items()},'checks':['unique complete block-anchor keys','stored block product identity','mean product/covariance identity','blockwise attribution identity','W4 and CFG1 subset keys','matched subset keys and n','both CI endpoints of eight frozen contrasts','two stored OOF ratios','six endpoint Bonferroni family','float32 adapter-equivalent schedule'], 'seed_by_panel':{name:seed(name) for name in panels},'source_boundaries':result['limitations']}
    for z in aggregation:
        if z['M_equal'] is not None and abs(z['R']-z['product_of_means']-z['population_covariance'])>1e-14:raise AssertionError('aggregation identity')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    (OUT/'README.md').write_text('# v1.0.1 post-hoc robustness supplement\n\nGenerated by `python scripts/reanalyse_revision.py`. This is CPU reanalysis of the immutable compact public tables, not new image generation or original preregistered inference. The original 20,000-resample higher-quantile headline intervals remain in their historical files and are reproduced separately. `results.json` is the shared numeric source for generated CSV, LaTeX and figures. `audit.json` records hashes, environment, counts and limits.\n\nThe 100,000-resample analysis uses linear quantiles and stable SHA-256 panel-label seeds, resampling paired blocks once per draw across metrics. Ratios and M_split are recomputed within every draw. Different seeds from the external sanity-check analysis can change CI endpoints by Monte Carlo error; point estimates are not forced.\n\n`bonferroni_six.csv` contains exactly H, M_equal and R in each primary CFG3 W8 panel; its 99.1667% marginal intervals address only these six endpoints. All other panels, subset interactions and robustness checks are exploratory. `tiny_adjacent.csv` preserves all Tiny adjacent-anchor results. Figures have PDF, SVG and PNG versions.\n\nRead the limitations array in results.json before population, mechanism, quality or reproducibility claims.\n')
    if args.check:
        def compare(actual,expected,path='result'):
            if isinstance(actual,dict):
                if actual.keys()!=expected.keys():raise AssertionError(f'{path}: keys differ')
                for key in actual:compare(actual[key],expected[key],path+'.'+key)
            elif isinstance(actual,list):
                if len(actual)!=len(expected):raise AssertionError(f'{path}: lengths differ')
                for i,(a,e) in enumerate(zip(actual,expected)):compare(a,e,f'{path}[{i}]')
            elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
                if not isinstance(expected,(int,float)) or not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):raise AssertionError(f'{path}: numeric mismatch {actual} != {expected}')
            elif actual!=expected:raise AssertionError(f'{path}: value differs')
        compare(result,json.loads((stored_out/'results.json').read_text()))
        for generated in sorted((OUT/'tables').glob('*')):
            target=stored_out/'tables'/generated.name
            if not target.is_file():raise AssertionError(f'missing stored artifact {target}')
            if generated.suffix=='.csv':
                pd.testing.assert_frame_equal(pd.read_csv(generated),pd.read_csv(target),check_exact=False,rtol=1e-10,atol=1e-12)
            elif generated.read_text()!=target.read_text():raise AssertionError(f'generated table differs: {target}')
        for generated in TEX.glob('*.tex'):
            if generated.read_text()!=(stored_tex/generated.name).read_text():raise AssertionError(f'generated LaTeX differs: {generated.name}')
        recorded_audit=json.loads((stored_out/'audit.json').read_text())
        if audit['inputs_sha256']!=recorded_audit['inputs_sha256']:raise AssertionError('input/source hashes differ: regenerate and review artifacts')
        temporary.cleanup()
    print(json.dumps({'pass':True,'check_mode':args.check,'panels':len(panels),'legacy_checks':len(legacy),'posthoc_contrasts':len(contrasts),'output':str(stored_out.relative_to(ROOT))}))


if __name__=='__main__':main()
