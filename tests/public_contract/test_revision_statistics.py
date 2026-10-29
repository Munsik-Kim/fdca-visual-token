"""Regression checks for pairing, aggregation, and generated statistical contracts."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_public_results import paired_values, reproduce
from reanalyse_revision import aggregate, inference, load_inputs, ratio, schedule


def test_complete_pair_keys_reject_missing_duplicate_and_nan():
    f=pd.DataFrame({'block_id':['a','a','b','b'],'anchor_target_rho':[.3,.5,.3,.5],'value':[1.,2.,3.,4.]})
    assert paired_values(f,'value').shape==(2,2)
    for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(value=[1.,2.,np.nan,4.])]:
        with pytest.raises(ValueError):paired_values(bad,'value')


def test_population_aggregation_and_undefined_conditionals():
    z=aggregate([.1,.9],[1.,3.])
    assert z['R']==pytest.approx(1.4)
    assert z['product_of_means']==pytest.approx(1.)
    assert z['population_covariance']==pytest.approx(.4)
    assert z['M_split']==pytest.approx(2.8)
    absent=aggregate([0.,.5],[np.nan,2.])
    assert absent['R']==pytest.approx(.5)
    assert absent['M_equal'] is None
    assert absent['M_split']==pytest.approx(2.)
    assert aggregate([0.,0.],[np.nan,np.nan])['M_split'] is None
    assert np.isnan(ratio(1.,0.))


def test_joint_block_bootstrap_recomputes_ratio():
    f=pd.DataFrame({'block_id':['a','a','b','b'],'anchor_target_rho':[.3,.5,.3,.5],
        'H':[.1,.4,.9,.6],'M_equal':[1.,3.,5.,2.], 'mean_coordinate_TV':[.1,.2,.2,.3],
        'DINO_equal':[1.,3.,5.,2.]})
    f['R']=f.H*f.M_equal;f['R_DINO']=f.R
    z={x['metric']:x for x in inference(f,'test_contract',resamples=2000)}
    assert z['M_split']['low']==pytest.approx(f[f.anchor_target_rho.eq(.3)].R.mean()/f[f.anchor_target_rho.eq(.3)].H.mean())
    assert z['M_split']['low']!=pytest.approx(z['M_equal']['low'])
    assert z['M_split']['ratio_ci'][0] < z['M_split']['ratio_ci'][1]


def test_panel_counts_schedule_and_legacy_both_endpoints():
    panels=load_inputs()
    assert {k:f.block_id.nunique() for k,f in panels.items()}=={'Tiny_CFG3_W8':48,'Tiny_CFG3_W4':24,'Small_CFG3_W8':40,'Small_CFG1_W8':16}
    assert [x['commit_batch_c'] for x in schedule()]==[41,14,8,6,5]
    result=reproduce()
    assert len(result)==10
    assert result['P2P_H5']['n']==24
    for z in result.values():
        if 'ci_lower' in z:assert z['ci_lower']<z['difference']<z['ci_upper']


def test_generated_family_and_aggregation_identity():
    result=json.loads((ROOT/'public_results/reanalysis_v1_0_1/results.json').read_text())
    family=[(z['panel'],z['metric']) for z in result['contrasts'] if 'bonferroni_ci_6' in z]
    assert set(family)=={(p,m) for p in ['Tiny_CFG3_W8','Small_CFG3_W8'] for m in ['H','M_equal','R']}
    assert len(family)==6
    for z in result['aggregation']:
        assert z['R']==pytest.approx(z['product_of_means']+z['population_covariance'],abs=1e-14)
    for z in result['attribution']:
        assert z['A_h']+z['A_m']==pytest.approx(z['delta_R'],abs=1e-14)
