"""Pure model-free reconstruction helpers for the post-P2N P2A audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
import ast,hashlib,json
from pathlib import Path
from typing import Any,Iterable,Iterator
import numpy as np
import pandas as pd

EXPERIMENT_ID='FDCA_VIS_P2N_NATURAL_SHOCK_BRIDGE';MASK=16384

def canonical(parts:dict[str,Any])->str:return json.dumps(parts,sort_keys=True,separators=(',',':'))
def digest(parts:dict[str,Any])->bytes:return hashlib.sha256(canonical(parts).encode()).digest()

@dataclass(frozen=True)
class TransitionCall:
    cell:str;block:str;anchor:str;replicate:str;step:int;attempt:int;positions:tuple[int,...];site:str
    @property
    def shared_group(self):return (self.block,self.anchor,self.replicate,self.step,self.attempt,'main')

def load_frames(root:Path)->dict[str,pd.DataFrame]:
 raw=root/'raw';derived=root/'derived';audits=root/'audits'
 return {
  'anchors':pd.concat([pd.read_parquet(derived/'p2n_anchor_index.parquet'),pd.read_parquet(raw/'smoke_anchors.parquet')],ignore_index=True),
  'seed_only':pd.concat([pd.read_parquet(raw/'p2n_seed_only_replicates.parquet'),pd.read_parquet(raw/'smoke_seed_only.parquet')],ignore_index=True),
  'seed_manifest':pd.concat([pd.read_parquet(raw/'p2n_natural_seed_manifest.parquet'),pd.read_parquet(raw/'smoke_seed_manifest.parquet')],ignore_index=True),
  'zero':pd.concat([pd.read_parquet(audits/'p2n_zero_seed_controls.parquet'),pd.read_parquet(raw/'smoke_zero_control.parquet')],ignore_index=True),
  'matched_terminal':pd.concat([pd.read_parquet(derived/'p2n_matched_timing_terminal.parquet'),pd.read_parquet(raw/'smoke_matched_terminal.parquet')],ignore_index=True),
 }

class ScheduleIndex:
 def __init__(self,root:Path,anchors:pd.DataFrame):
  self.steps={};self.anchor_step={}
  for row in anchors.itertuples():
   key=(str(row.block_id),str(row.anchor_id));self.anchor_step[key]=int(row.anchor_step)
   if key in self.steps:continue
   directory=root/'raw/p2n_reference_anchors'/str(row.block_id);match=next(p for p in directory.glob('*.npz') if f'R{float(row.anchor_target_rho):.1f}' in p.name);values=np.load(match)['commit_steps'];self.steps[key]={s:tuple(np.flatnonzero(values==s).astype(int).tolist()) for s in range(32)}
 def positions(self,block,anchor,step):return self.steps[(str(block),str(anchor))][int(step)]

def transition_calls(frames:dict[str,pd.DataFrame],schedule:ScheduleIndex)->list[TransitionCall]:
 calls=[]
 for row in frames['seed_only'].itertuples():
  step=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))];calls.append(TransitionCall(str(row.cell_id),str(row.block_id),str(row.anchor_id),f'seed_only_{int(row.replicate_id):03d}',step,0,schedule.positions(row.block_id,row.anchor_id,step),'seed_only_transition'))
 for row in frames['zero'].itertuples():
  step=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))];calls.append(TransitionCall(str(row.cell_id),str(row.block_id),str(row.anchor_id),'zero_control',step,0,schedule.positions(row.block_id,row.anchor_id,step),'zero_control_transition'))
 for row in frames['seed_manifest'].itertuples():
  step=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))]
  for attempt in range(int(row.attempt_count)):calls.append(TransitionCall(str(row.cell_id),str(row.block_id),str(row.anchor_id),f'conditional_{int(row.replicate_id):02d}',step,attempt,schedule.positions(row.block_id,row.anchor_id,step),'conditional_transition_attempt'))
 return calls

def key_parts(*,cell,block,anchor,replicate,step,position,event_kind,attempt=0,draw_id='main',shared=False):
 return {'experiment_id':EXPERIMENT_ID,'approximation_cell':'REFERENCE_SHARED' if shared else cell,'block_id':block,'anchor_id':anchor,'seed_replicate_id':replicate,'step':int(step),'position':int(position),'event_kind':event_kind,'attempt_id':int(attempt),'draw_id':draw_id}

def iter_uses(frames:dict[str,pd.DataFrame],schedule:ScheduleIndex,calls:list[TransitionCall])->Iterator[tuple[dict[str,Any],str,str,int]]:
 group_cells=defaultdict(set)
 for call in calls:group_cells[call.shared_group].add(call.cell)
 for call in calls:
  multiplicity=len(group_cells[call.shared_group])
  for position in call.positions:
   classification='INTENTIONAL_REFERENCE_SHARED_REUSE' if multiplicity>1 else 'UNIQUE'
   yield key_parts(cell=call.cell,block=call.block,anchor=call.anchor,replicate=call.replicate,step=call.step,position=position,event_kind='proposal_x',attempt=call.attempt,shared=True),call.site,classification,multiplicity
   for kind in ['proposal_accept','proposal_residual']:
    yield key_parts(cell=call.cell,block=call.block,anchor=call.anchor,replicate=call.replicate,step=call.step,position=position,event_kind=kind,attempt=call.attempt),call.site,'UNIQUE',1
 # Zero-control common-q suffix: identical branches consume proposal_x only.
 for row in frames['zero'].itertuples():
  start=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))]+1
  for step in range(start,32):
   for position in schedule.positions(row.block_id,row.anchor_id,step):yield key_parts(cell=str(row.cell_id),block=str(row.block_id),anchor=str(row.anchor_id),replicate='zero_control',step=step,position=position,event_kind='proposal_x',draw_id='suffix'),'zero_control_common_q_suffix','UNIQUE',1
 # Conditional natural common-q suffix: persistent seed keeps branches unequal.
 for row in frames['seed_manifest'].itertuples():
  start=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))]+1
  rep=f'conditional_{int(row.replicate_id):02d}'
  for step in range(start,32):
   for position in schedule.positions(row.block_id,row.anchor_id,step):
    for kind in ['proposal_x','proposal_accept','proposal_residual']:yield key_parts(cell=str(row.cell_id),block=str(row.block_id),anchor=str(row.anchor_id),replicate=rep,step=step,position=position,event_kind=kind,draw_id='suffix'),'conditional_common_q_suffix','UNIQUE',1
 # Matched timing starts at the anchor step and always has a persistent seed.
 for row in frames['matched_terminal'].itertuples():
  start=schedule.anchor_step[(str(row.block_id),str(row.anchor_id))];rep=f'matched_{int(row.replicate_id)}'
  for step in range(start,32):
   for position in schedule.positions(row.block_id,row.anchor_id,step):
    for kind in ['proposal_x','proposal_accept','proposal_residual']:yield key_parts(cell='W8_L1.00',block=str(row.block_id),anchor=str(row.anchor_id),replicate=rep,step=step,position=position,event_kind=kind,draw_id='suffix'),'matched_timing_common_q_suffix','UNIQUE',1

def intentional_key_count(calls:list[TransitionCall])->int:
 groups=defaultdict(list)
 for c in calls:groups[c.shared_group].append(c)
 return sum(len(items[0].positions) for items in groups.values() if len({x.cell for x in items})>1)

def extract_call_sites(path:Path)->pd.DataFrame:
 source=path.read_text();tree=ast.parse(source);parents={}
 for node in ast.walk(tree):
  for child in ast.iter_child_nodes(node):parents[child]=node
 rows=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='event_uniform':
   parent=node
   while parent in parents and not isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef)):parent=parents[parent]
   kwargs={k.arg:ast.unparse(k.value) for k in node.keywords}
   rows.append({'source_file':str(path),'line':node.lineno,'function':getattr(parent,'name','module'),'event_kind_expression':kwargs.get('event_kind',''),'key_fields_supplied':json.dumps(kwargs,sort_keys=True),'approximation_cell_mode':'REFERENCE_SHARED when cross_cell_proposal=True' if kwargs.get('cross_cell_proposal')=='True' else 'cell-specific','intended_reuse':kwargs.get('cross_cell_proposal')=='True','scientific_role':'natural transition' if getattr(parent,'name','')=='transition_draw' else 'common approximate suffix'})
 return pd.DataFrame(rows).sort_values('line',ignore_index=True)

def semantic_seed(label:str)->int:return int.from_bytes(hashlib.sha256(f'FDCA-P2A|{label}'.encode()).digest()[:8],'big')
def bootstrap(values:np.ndarray,label:str,resamples:int=20000)->tuple[float,float]:
 rng=np.random.default_rng(semantic_seed(label));draws=rng.choice(np.asarray(values,float),(resamples,len(values)),replace=True).mean(axis=1);return float(np.quantile(draws,.025,method='higher')),float(np.quantile(draws,.975,method='higher'))
def incidence_ci(law:pd.DataFrame,positions:pd.DataFrame)->pd.DataFrame:
 direct=positions.groupby(['block_id','cell_id','anchor_target_rho'],as_index=False).direct_tv_b.mean().rename(columns={'direct_tv_b':'per_position_direct_tv'})
 frame=law.merge(direct,on=['block_id','cell_id','anchor_target_rho'],validate='one_to_one');rows=[]
 for (cell,rho),part in frame.groupby(['cell_id','anchor_target_rho']):
  hci=bootstrap(part.split_incidence.to_numpy(),f'h|{cell}|{rho}');sci=bootstrap(part.expected_seed_count.to_numpy(),f'seed|{cell}|{rho}');dci=bootstrap(part.per_position_direct_tv.to_numpy(),f'direct|{cell}|{rho}')
  rows.append({'row_type':'rho_summary','cell_id':cell,'anchor_target_rho':rho,'blocks':part.block_id.nunique(),'h_split_mean':part.split_incidence.mean(),'h_split_median':part.split_incidence.median(),'h_split_ci95_lower':hci[0],'h_split_ci95_upper':hci[1],'expected_seed_count_mean':part.expected_seed_count.mean(),'expected_seed_count_median':part.expected_seed_count.median(),'expected_seed_count_ci95_lower':sci[0],'expected_seed_count_ci95_upper':sci[1],'single_seed_probability_mean':part.single_seed_probability.mean(),'multi_seed_probability_mean':part.multi_seed_probability.mean(),'per_position_direct_tv_mean':part.per_position_direct_tv.mean(),'per_position_direct_tv_ci95_lower':dci[0],'per_position_direct_tv_ci95_upper':dci[1]})
 for cell,part in frame.groupby('cell_id'):
  wide=part.pivot(index='block_id',columns='anchor_target_rho',values=['split_incidence','expected_seed_count','per_position_direct_tv'])
  vals={name:wide[(name,.5)]-wide[(name,.3)] for name in ['split_incidence','expected_seed_count','per_position_direct_tv']}
  cis={name:bootstrap(value.to_numpy(),f'contrast|{cell}|{name}') for name,value in vals.items()}
  rows.append({'row_type':'rho_0.5_minus_0.3','cell_id':cell,'anchor_target_rho':np.nan,'blocks':len(wide),'h_split_mean':vals['split_incidence'].mean(),'h_split_median':vals['split_incidence'].median(),'h_split_ci95_lower':cis['split_incidence'][0],'h_split_ci95_upper':cis['split_incidence'][1],'expected_seed_count_mean':vals['expected_seed_count'].mean(),'expected_seed_count_median':vals['expected_seed_count'].median(),'expected_seed_count_ci95_lower':cis['expected_seed_count'][0],'expected_seed_count_ci95_upper':cis['expected_seed_count'][1],'single_seed_probability_mean':np.nan,'multi_seed_probability_mean':np.nan,'per_position_direct_tv_mean':vals['per_position_direct_tv'].mean(),'per_position_direct_tv_ci95_lower':cis['per_position_direct_tv'][0],'per_position_direct_tv_ci95_upper':cis['per_position_direct_tv'][1]})
 return pd.DataFrame(rows)
