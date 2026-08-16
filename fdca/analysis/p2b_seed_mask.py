"""Exact model-free arithmetic and compact runtime registry for P2B."""
from __future__ import annotations
import hashlib, json, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from fdca.couplings.p1l_conditional_maximal import probability_vector

MASK=16_384
EXPERIMENT_ID='FDCA_VIS_P2N_NATURAL_SHOCK_BRIDGE'

def canonical(parts:Mapping[str,Any])->str:return json.dumps(dict(parts),sort_keys=True,separators=(',',':'))
def key_parts(*,cell,block,anchor,replicate,step,position,event_kind,attempt=0,shared=False):
 return {'experiment_id':EXPERIMENT_ID,'approximation_cell':'REFERENCE_SHARED' if shared else cell,'block_id':str(block),'anchor_id':str(anchor),'seed_replicate_id':str(replicate),'step':int(step),'position':int(position),'event_kind':event_kind,'attempt_id':int(attempt),'draw_id':'main'}

class MemmapSemanticRegistry:
 def __init__(self,capacity:int,directory:Path):
  directory.mkdir(parents=True,exist_ok=True);self.capacity=int(capacity);self.n=0;self.proposal_n=0
  self.hashes=np.memmap(directory/'all_key_uses.bin',mode='w+',dtype='S32',shape=(capacity,));self.proposals=np.memmap(directory/'proposal_key_uses.bin',mode='w+',dtype='S32',shape=(max(capacity//3,1),))
 def uniform(self,parts:Mapping[str,Any])->tuple[float,str]:
  raw=hashlib.sha256(canonical(parts).encode()).digest()
  if self.n>=self.capacity:raise RuntimeError('registry capacity exceeded')
  self.hashes[self.n]=raw;self.n+=1
  if parts['event_kind']=='proposal_x':self.proposals[self.proposal_n]=raw;self.proposal_n+=1
  return int.from_bytes(raw[:8],'big')/float(1<<64),raw.hex()
 def audit(self)->dict[str,Any]:
  if self.n!=self.capacity:raise RuntimeError(f'missing expected replay keys {self.capacity-self.n}')
  self.hashes.flush();self.proposals.flush();all_values=np.asarray(self.hashes[:self.n]);pvalues=np.asarray(self.proposals[:self.proposal_n]);unique,counts=np.unique(all_values,return_counts=True);pu,pc=np.unique(pvalues,return_counts=True);duplicates=unique[counts>1];allowed=pu[pc>1]
  exact=np.array_equal(duplicates,allowed);unclassified=0 if exact else len(set(map(bytes,duplicates)).symmetric_difference(set(map(bytes,allowed))));maximum=int(counts.max())
  return {'total_key_uses':self.n,'unique_keys':len(unique),'intentional_reference_shared_reuse_keys':len(allowed),'intentional_reference_shared_extra_uses':int(np.sum(pc[pc>1]-1)),'unclassified_duplicate_count':int(unclassified),'missing_expected_key_count':0,'unexpected_replay_key_count':0,'maximum_multiplicity':maximum,'intentional_hash_set_exact_match':bool(exact),'sorted_key_use_multiset_sha256':hashlib.sha256(np.sort(all_values).tobytes()).hexdigest(),'pass':exact and maximum==2}

def prepare_law(p:Sequence[float],q:Sequence[float])->dict[str,Any]:
 left,right=probability_vector(p),probability_vector(q);residual=np.maximum(right-left,0.0);mass=float(residual.sum());rn=probability_vector(residual/mass) if mass>1e-15 else np.zeros_like(residual)
 return {'p':left,'q':right,'p_cdf':np.cumsum(left),'residual_cdf':np.cumsum(rn) if mass>1e-15 else None,'tv':float(np.abs(left-right).sum()/2),'residual_mass':mass}
def inv(cdf:np.ndarray,u:float)->int:return min(int(np.searchsorted(cdf,u,side='right')),len(cdf)-1)
def draw_position(law:Mapping[str,Any],registry:MemmapSemanticRegistry,base:dict[str,Any])->dict[str,Any]:
 up,hp=registry.uniform(key_parts(**base,event_kind='proposal_x',shared=True));x=inv(law['p_cdf'],up);ua,ha=registry.uniform(key_parts(**base,event_kind='proposal_accept'));ur,hr=registry.uniform(key_parts(**base,event_kind='proposal_residual'));ratio=float(law['q'][x]/law['p'][x]) if law['p'][x]>0 else 0.0;accepted=ua<=min(1.0,ratio)
 y=x if accepted or law['residual_mass']<=1e-15 else inv(law['residual_cdf'],ur)
 return {'x':x,'y':y,'accepted_common':bool(accepted or law['residual_mass']<=1e-15),'proposal_key_hash':hp,'semantic_key_hash':hashlib.sha256(bytes.fromhex(hp+ha+hr)).hexdigest()}
def category(x:int,y:int)->str:
 if x==MASK and y==MASK:return 'EQUAL_MASK_MASK'
 if x==MASK:return 'REFERENCE_MASK_ONLY'
 if y==MASK:return 'APPROXIMATE_MASK_ONLY'
 if x==y:return 'EQUAL_NONMASK'
 return 'VALUE_VALUE_MISMATCH'
def poisson_binomial(values:Sequence[float])->np.ndarray:
 law=np.array([1.0],dtype=np.float64)
 for p in np.asarray(values,dtype=np.float64):law=np.convolve(law,np.array([1-p,p],dtype=np.float64))
 return law/law.sum()
def mask_law(pm:Sequence[float],qm:Sequence[float])->dict[str,Any]:
 p=np.asarray(pm,dtype=np.float64);q=np.asarray(qm,dtype=np.float64);equal=np.minimum(p,q);ro=np.maximum(p-q,0);ao=np.maximum(q-p,0);anyp=np.maximum(p,q);le=poisson_binomial(equal);la=poisson_binomial(anyp)
 return {'expected_equal_mask_count':float(equal.sum()),'expected_reference_only_mask_count':float(ro.sum()),'expected_approximate_only_mask_count':float(ao.sum()),'probability_at_least_one_equal_mask':float(1-le[0]),'probability_at_least_one_one_sided_mask':float(1-np.prod(1-ro-ao,dtype=np.float64)),'expected_any_branch_mask_count':float(anyp.sum()),'equal_count_law':le,'any_count_law':la}
def stable_sets(anchor_state:np.ndarray,positions:Sequence[int],future:Sequence[int],x:Sequence[int],y:Sequence[int])->dict[str,Any]:
 a=anchor_state.copy();b=anchor_state.copy();a[list(positions)]=x;b[list(positions)]=y;future=set(map(int,future));immutable=[i for i in range(256) if a[i]==b[i] and i not in future];semantic=[i for i in immutable if a[i]!=MASK]
 return {'reference_state':a,'approximate_state':b,'immutable_equality':immutable,'nonmask_semantic':semantic}
