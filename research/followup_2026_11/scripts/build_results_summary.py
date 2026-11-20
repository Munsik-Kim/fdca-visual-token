#!/usr/bin/env python3
"""Build the current results index from archived tables; perform no new inference."""
from pathlib import Path
import argparse,csv,hashlib,io,json
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/followup_2026_11'
DATE='2026-11-20'

def rows(name):return list(csv.DictReader((BASE/'evidence/tables'/name).open()))
def arr(s):return json.loads(s)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def collect():
 out=[]
 def add(phase,key,metric,estimate,lo,hi,cl,n,cluster,role,panel,source,selector):
  vals=[float(v) if v is not None and v!='' else None for v in [estimate,lo,hi]]
  out.append(dict(phase=phase,contrast=key,metric=metric,estimate=vals[0],ci_low=vals[1],ci_high=vals[2],confidence=float(cl) if cl else None,n_clusters=int(n),cluster=cluster,role=role,panel=panel,source=source,source_row=json.dumps(selector,sort_keys=True,separators=(',',':')),interpretation='NO_INTERVAL' if vals[1] is None else ('POSITIVE_DIRECTION' if vals[1]>0 else 'NEGATIVE_DIRECTION' if vals[2]<0 else 'ZERO_INCLUDED')))
 def s(n):return 'evidence/tables/'+n
 for q in rows('lt1_arms.csv'):
  add('LT1',q['arm_id'],'reference_LPIPS_R',q['R_path'],*arr(q['R_path_ci']),.95,q['classes'],'class','technical_control' if q['arm_id']=='REF_FP32' else 'exploratory','initial_16_classes',s('lt1_arms.csv'),{'arm_id':q['arm_id']})
 for q in rows('lt1r_restoration.csv'):
  add('LT1-R','restore_block_'+q['block'],'reference_gain_G',q['G'],*arr(q['G_ci']),.95,q['classes'],'class','exploratory','reused_8_classes',s('lt1r_restoration.csv'),{'block':q['block']})
 for name,phase in [('lt2_contrasts.csv','LT2'),('lt2c_contrasts.csv','LT2-C'),('lt3_contrasts.csv','LT3'),('lt4_contrasts.csv','LT4-OP5')]:
  for q in rows(name):
   if phase=='LT2' and q['scope']!='MAIN':continue
   if phase=='LT2-C' and q['metric']!='R':continue
   if phase=='LT3' and not(q['metric']=='R' and q['contrast'] in ['C2','C3']):continue
   if phase=='LT4-OP5' and not(q['metric']=='R' and q['kind']=='C5'):continue
   add(phase,q['contrast'],'local_probability_TV_contrast' if phase=='LT2' and q['contrast'].startswith('A') else 'pulse_LPIPS_R_contrast',q['mean'],*arr(q['ci']),q.get('confidence_level',.95),q['class_n'],'class',q['scope'],'new_32_classes' if phase=='LT3' else 'reused_development',s(name),{k:q[k] for k in ['contrast','metric','scope'] if k in q})
 for q in rows('lt5_primary.csv'):
  add('LT5',q['endpoint'],'reference_gain_G',q['mean'],q['ci_low'],q['ci_high'],q['confidence_level'],q['class_n'],'class','co_primary','new_32_classes',s('lt5_primary.csv'),{'endpoint':q['endpoint']})
 # LT6 is recovered in the November 20 extension; its source table is independently replayed.
 p=BASE/'evidence/tables/lt6_primary_20261120.csv'
 if not p.is_file():raise FileNotFoundError('Required recovered LT6 primary input is missing')
 if p.is_file():
  for q in csv.DictReader(p.open()):
   add('LT6',q['contrast'],'W32_new_mismatch_tokens_reduced',float(q['estimate'])*32,float(q['ci_low'])*32,float(q['ci_high'])*32,q['confidence'],q['n_classes'],'class','primary_diagnostic','reused_LT3_32_classes','evidence/tables/lt6_primary_20261120.csv',{'contrast':q['contrast']})
 for q in rows('m_confirm_contrasts.csv'):
  if q['primary']!='True':continue
  add('M-confirm',q['contrast'],'W32_new_mismatch_tokens_reduced',float(q['estimate'])*32,float(q['ci_low'])*32,float(q['ci_high'])*32,q['confidence'],q['class_n'],'class','three_primary','new_32_classes',s('m_confirm_contrasts.csv'),{'contrast':q['contrast']})
 for q in rows('r_test_primary.csv'):
  add('R-test',q['contrast'],'reference_gain_G',q['estimate'],q['lower'],q['upper'],q['confidence'],q['n_classes'],'class','six_primary','new_320_classes',s('r_test_primary.csv'),{'contrast':q['contrast']})
 for q in rows('lt10_13_contrasts.csv'):
  add('LT10',q['contrast'],'paired_reference_gain_G',q['estimate'],q['bonferroni_13_lower'],q['bonferroni_13_upper'],q['bonferroni_confidence'],q['n_classes'],'class','retrospective_13_family','reused_R_test_320_classes',s('lt10_13_contrasts.csv'),{'contrast':q['contrast']})
 for q in rows('lt11_contrasts.csv'):
  if q['horizon']!='W32' or q['role']!='primary':continue
  add('LT11',q['contrast'],'W32_new_mismatch_tokens_reduced',q['token_estimate'],q['token_lower'],q['token_upper'],q['confidence'],q['n_pairs'],'class_pair','two_primary','new_64_classes_in_32_pairs',s('lt11_contrasts.csv'),{'contrast':q['contrast'],'horizon':'W32'})
 for q in rows('lt12_primary.csv'):
  add('LT12',q['contrast'],'reference_gain_G',q['estimate'],q['lower'],q['upper'],q['confidence'],q['n_classes'],'class','four_primary_diagnostic',q['panel'],s('lt12_primary.csv'),{'contrast':q['contrast']})
 for q in rows('lt13_contrasts.csv'):
  add('LT13',q['panel'],'own_minus_swapped_reference_gain_G',q['estimate'],q['lower'],q['upper'],q['confidence'],q['n_pairs'],'class_pair','exploratory','reused_oracle_selected_states',s('lt13_contrasts.csv'),{'panel':q['panel']})
 for q in rows('lt14_policy_summary.csv'):
  add('LT14',q['policy'],'reference_gain_G',q['mean_signed_G'],None,None,None,q['n_classes'],'class','development_outer_OOF','24_development_classes',s('lt14_policy_summary.csv'),{'policy':q['policy']})
 for q in csv.DictReader((BASE/'evidence/lt16/expected/primary_contrasts.csv').open()):
  add('LT16',q['contrast'],'saved_state_reference_gain_G',q['estimate'],q['ci_low'],q['ci_high'],q['confidence'],q['n_class'],'class','three_primary_diagnostic','reused_LT14_24_classes','evidence/lt16/expected/primary_contrasts.csv',{'contrast':q['contrast']})
 for r in out:
  source=BASE/r['source']; key=json.loads(r['source_row'])
  matches=[q for q in csv.DictReader(source.open()) if all(q[k]==v for k,v in key.items())]
  assert len(matches)==1, f'Ambiguous source row: {r["phase"]} / {r["contrast"]}'
  q=matches[0]
  r['definition']=q.get('definition',q.get('direction',q.get('contrast',q.get('endpoint',r['contrast']))))
 assert len({(r['phase'],r['contrast']) for r in out})==len(out)
 return out

def build():
 rs=collect();sio=io.StringIO(newline='');w=csv.DictWriter(sio,fieldnames=list(rs[0]),lineterminator="\n");w.writeheader();w.writerows(rs)
 payload={'data/consolidated_results.csv':sio.getvalue()}
 inputs=sorted({r['source'] for r in rs}|{'data/experiment_ledger.csv','evidence/tables/lt14_dev_gate.json','evidence/tables/lt14_observed_maximum.json','evidence/lt16/expected/selector_summary.csv','evidence/lt16/expected/auxiliary_contrasts.csv'})
 payload['data/consolidation_index.json']=json.dumps({'publication_date':DATE,'scope':'Re-indexing stored results, not a new experiment or new statistical family','normalized_rows':len(rs),'source_files':[{'path':p,'sha256':digest(BASE/p)} for p in inputs],'scripts':'scripts/build_results_summary.py','private_artifacts_excluded':True,'LT15_state':'NOT_STARTED_DEV_CRITERIA'},indent=2)+'\n'
 def fmt(x):return f'{float(x):+.6f}'
 selectors={q['selector']:q for q in csv.DictReader((BASE/'evidence/lt16/expected/selector_summary.csv').open())}
 auxiliary={q['contrast']:q for q in csv.DictReader((BASE/'evidence/lt16/expected/auxiliary_contrasts.csv').open())}
 oldmax=selectors['OLD_MAX']
 split_self=next(q for k,q in auxiliary.items() if 'SPLIT' in k.upper() and 'SELF' in k.upper())
 split_transfer=next(q for k,q in auxiliary.items() if 'SPLIT' in k.upper() and 'TRANSFER' in k.upper())
 labels={('LT6','SPARSE_minus_PREFILL15'):'공통 문맥 추가',('M-confirm','C_ctx'):'문맥 추가',('M-confirm','C_layout'):'원래 배치',('M-confirm','C_dose'):'가시 범위 증가',('LT11','C_shared'):'타 클래스 공통 문맥',('LT11','C_match'):'문맥 출처 일치의 추가 효과',('LT5','A_FDCA_ABSOLUTE'):'FDCA 대 W8',('LT5','B_FDCA_MINUS_LOCAL_TV'):'FDCA 대 국소 TV',('R-test','A_F'):'F 대 W8',('R-test','B_F'):'F 대 국소 TV',('LT12','A_ref'):'참조 토큰 편집 대 W8',('LT12','B_dir'):'참조 편집 대 대체 토큰 편집',('LT12','A_b2'):'oracle 시점 block 2 복원 대 W8',('LT12','A_b3'):'oracle 시점 block 3 복원 대 W8',('LT16','A'):'OLD_MAX 대 W8',('LT16','B'):'OLD_MAX 대 ENTROPY',('LT16','C'):'PROPOSED 대 TIME_ONLY',('LT13','original_draw'):'자기 시점 대 교환 시점, 원 draw',('LT13','redraw'):'자기 시점 대 교환 시점, 새 draw'}
 def table(keys):
  out=['| 단계 / 비교 | 추정치 | 원 신뢰구간 | 독립 집계 단위 |','|---|---:|---|---|']
  for phase,key in keys:
   r=next(r for r in rs if r['phase']==phase and r['contrast']==key)
   src='research/followup_2026_11/'+r['source'];label=phase+' / '+labels.get((phase,key),key)
   out.append(f"| [{label}]({src}) | {fmt(r['estimate'])} | {r['confidence']*100:.8g}% [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['n_clusters']} {'클래스 쌍' if r['cluster']=='class_pair' else '클래스'} |")
  return '\n'.join(out)
 context=table(([(r['phase'],r['contrast']) for r in rs if r['phase']=='LT6'])+[('M-confirm','C_ctx'),('M-confirm','C_layout'),('M-confirm','C_dose'),('LT11','C_shared'),('LT11','C_match')])
 gains=table([('LT5','A_FDCA_ABSOLUTE'),('LT5','B_FDCA_MINUS_LOCAL_TV'),('R-test','A_F'),('R-test','B_F'),('LT12','A_ref'),('LT12','B_dir'),('LT12','A_b2'),('LT12','A_b3'),('LT13','original_draw'),('LT13','redraw'),('LT16','A'),('LT16','B'),('LT16','C')])
 lines=['# FDCA 전체 결과 총정리',f'공개 근거 점검 기준: **{DATE}** · 원 논문부터 LT16까지 · 새 GPU 실험 없음','', '**전파를 통제하고 측정하는 근거는 확보했지만, 그 정보를 이용하는 참조 없는 정밀도 복원 정책의 안정적인 추가 이득은 확인하지 못했다.** 문맥 개입과 참조 정보를 쓰는 토큰 수정의 양의 결과, 정책 평가의 미확정 결과, 개발 단계 중단을 함께 보존한다.','', '[English](RESULTS_SUMMARY.md) · [쉬운 설명](START_HERE_KO.md) · [상세 연구 이력](research/followup_2026_11/REPORT_KO.md) · [모든 주요 수치 CSV](research/followup_2026_11/data/consolidated_results.csv) · [검증 범위](research/followup_2026_11/EVIDENCE_AND_VALIDATION.md)','', '## 먼저 구분할 세 가지 양','', '- **전파:** 시작 차이 뒤 새로 확정한 위치의 토큰 불일치. 문맥 실험의 표는 후속 32좌표 중 감소한 토큰 개수다.','- **Pulse R:** 한 단계 개입 여부를 달리한 뒤 모두 Q로 진행한 두 경로의 종단 LPIPS 기대거리. 분기 조건부 추출에서는 `mean(h*m)`으로 복원한다. 큰 R이 좋은 복원이라는 뜻은 아니다.','- **복원 G:** `LPIPS(FP32 참조, W8) − LPIPS(FP32 참조, 복원 경로)`. 양수는 지정한 결합에서 참조와의 거리 감소다. 이미지 품질·사람 선호·packed INT8 속도는 별도 측정이다.','', '11월 후속 연구는 H-MaskGIT-T의 W8 반올림 가중치를 FP32로 재구성해 실행했다. 초기 논문의 Tiny/Small 비교와 섞어 표본 수나 효과를 합치지 않는다. `mean(h*m)`과 `mean(h)*mean(m)` 사이 covariance를 생략하지 않는다.','', '## 전파에 관한 결과','', context,'', 'LT6는 LT3의 32클래스를 재사용한 512쌍 진단이다. M-confirm은 새 실험 클래스 32개, LT11은 새 클래스 64개를 32개 쌍으로 묶은 별도 연구다. LT11의 C_shared는 다른 클래스 출처 문맥을 공유하는 효과, C_match는 자기 클래스와 출처를 맞춘 추가 효과다. 문맥의 가시성·값·배치가 함께 규정된 개입이므로 일반적인 시간 효과나 모든 문맥의 보편적 효과로 확대하지 않는다.','', 'LT3의 초기 pulse 대비는 block 2 **+0.004121**, block 3 **+0.005000**으로 두 97.5% 구간이 양수였다. 같은 부모를 재사용해 복원 law 노출을 5좌표로 제한한 LT4-OP5에서도 각각 **+0.004806**, **+0.005937**이었다. 이는 pulse 출력 차이의 크기이며 G와 다른 양이다.','', '## 실제 참조 거리 감소와 선택 능력','',gains,'', 'LT5와 R-test의 첫 두 행은 각각 절대 G와 국소 TV 대비 추가 G다. LT16의 B는 OLD_MAX−ENTROPY, C는 PROPOSED−TIME_ONLY다. LT13 원 draw 비교는 32개 클래스 쌍, 새 draw 비교는 8개 쌍이며 두 표본을 합치지 않는다.', '', '표의 LPIPS 단위는 같아도 표본·개입·정보·CI 수준이 다르다. 행 사이 CI 끝점을 빼서 직접 비교하지 않는다. LT12의 시점과 참조 토큰은 FP32 경로에서 얻었으며 R-test 집단을 재사용했다. A_ref/A_b2/A_b3는 트리거 없는 1개를 포함한 128경로, B_dir는 ALT가 가능한 127경로를 클래스 64개에 동일 가중한 결과다. 참조 토큰 수정과 가중치 한 단계 복원은 같은 비용의 행동이 아니다. block 2의 양의 구간과 block 3의 0 포함만으로 두 block의 우열을 선언하지 않는다.','', '**LT14 개발 중단:** 제안 정책 G는 +0.005884, 가장 강한 동일 행동 대조 ENTROPY는 +0.005266이었다. 차이 **+0.000618**은 사전 진행 기준 +0.005에 못 미쳤다. 양의 fold는 3/4였지만 세 조건을 모두 만족해야 하므로 LT15는 `NOT_STARTED_DEV_CRITERIA`다. 예약 265클래스의 식별자는 공개하지 않았으며 해당 확인 실험을 수행했다고 보고하지 않는다.','', f"**LT16 새 난수 진단:** 원 draw에서 32시점 중 최대를 고르면 평균 {fmt(oldmax['old_G'])}이었으나, 선택된 저장 상태에서 8회 새 draw로 측정한 OLD_MAX 평균은 {fmt(oldmax['new_G'])}이었다. 같은 절반으로 선택·평가하면 {fmt(split_self['estimate'])}, 다른 절반으로 평가하면 {fmt(split_transfer['estimate'])}였다. 후자는 보조 95% CI [{fmt(split_transfer['ci_low'])},{fmt(split_transfer['ci_high'])}]다. 현재 특징의 추가 선택 능력을 입증하지 못했으며, 모든 복원 알고리즘이 불가능하다는 결론도 아니다.",'', '다섯 LT16 평균: '+', '.join(f"{name} {fmt(selectors[key]['new_G'])}" for name,key in [('OLD_MAX','OLD_MAX'),('PROPOSED','PROPOSED_OOF'),('ENTROPY','ENTROPY_OOF'),('TIME_ONLY','TIME_ONLY_FROZEN'),('FIXED','FIXED_OOF')])+'. 세 주 구간은 모두 0을 포함한다. 원 48경로·158개 선택 상태·8draw를 재사용한 진단이며 새 클래스 평가가 아니다.','', '## 전체 진행 지도','', '행의 표본은 서로 중첩되므로 더해서 총 독립 표본 수를 만들지 않는다. 기존 요약·음의 결과·미완료는 원 근거에 남긴다.','', '| 단계 | 질문과 설계 | 결과 / 근거 |','|---|---|---|', '| 원 논문 T0–P3S | 합성 법칙, sampler·RNG, 인공·자연 분기, Tiny/Small 출력 차이 | [원 실험 ledger](docs/EXPERIMENT_LEDGER.md). P05R 상한은 유효하지만 실용적으로 무용(vacuous)했고, P1M은 순위 정보만 지지했다. P2A 불완전 실행과 P2B 감사 통과를 구분한다. |']
 descriptions={
 'LT0/LT1':'16클래스 지속 W8 pilot; 층 순위는 기술량', 'LT1-R':'8클래스 재사용·지속 복원; 모든 절대 G 구간 0 포함', 'LT2':'국소 상태 진단·자연 pulse; 희소 분기', 'LT2C':'조건부 pulse를 h로 가중; 초기 차이 탐색', 'LT3':'새 32클래스에서 두 초기 대비 양수', 'LT4/OP5':'LT3 재사용·5좌표 노출; 초기 차이 유지', 'LT5':'새 32클래스·같은 8 block-step; 두 주 G 미확정', 'LT6':'LT3 재사용·같은 최초 차이와 W32; 공통 문맥 효과', 'LT7/M-dev':'후속 비교군 개발; 새 확인으로 세지 않음', 'M-confirm/LT8':'새 32클래스; 문맥·배치·가시 범위 세 방향 확인', 'R0':'재사용 panel의 8-step 대 1-step; 기간 우열 미확정', 'R-cal':'8개 개발 클래스에서 정책 선택; 성능 확인 아님', 'R-test/LT9':'새 320클래스; 여섯 주 구간 모두 0 포함', 'R-alt':'32클래스 subset의 결합 민감도; 결합 동등성 미입증', 'LT10':'R-test 재분석 13개 대비; 보정 구간 모두 0 포함', 'LT11':'새 64클래스/32쌍; 공유·출처 일치 추가 효과', 'LT12':'재사용 64클래스; oracle 토큰 수정과 한 단계 복원', 'LT13':'원 draw 자기 시점 우위; 새 draw에서는 미확정', 'LT14':'24클래스 개발; ENTROPY 대비 진행 기준 실패', 'LT15':'미실행; 새 확인 성과 없음', 'LT16':'24클래스 저장 상태 재사용; 새 난수에서 관측 최대의 양의 이득 미확인'}
 for q in csv.DictReader((BASE/'data/experiment_ledger.csv').open()):
  ph=q['phase'];src='research/followup_2026_11/'+q['numerical_source']
  lines.append(f"| {ph} | {q['classes']} classes; {q['units']} | {descriptions[ph]} · [근거]({src}) |")
 lines += ['', '## 이번 공개 점검에서 보완한 것','', '2026-11-13 공개 당시 LT6 standalone 근거를 확보하지 못했던 범위는 역사적 기록으로 남긴다. 이번에는 실제 보존본에서 최소 단위 수치를 다시 확보해 원 분석 계약으로 수치를 재생하고 집계 결과를 추가했다. 새로운 단위별 입력은 로컬에 보존했다. LT6 집계표의 출처와 변환은 [별도 해시 manifest](research/followup_2026_11/data/aggregate_source_manifest_20261120.json)에 연결한다. 기존 68개 공개 근거 파일과 역사적 수치는 바꾸지 않는다.','', '저장 scalar 수치에서 class/pair 집계와 CI를 재계산하는 범위, 저장된 full categorical law의 token replay, 실제 모델·decoder·LPIPS 재실행은 서로 다르다. 이번 작업은 보존된 수치의 로컬 재생과 저장소 검증이며 새 GPU 실행은 없다. 기존 공개 재생은 유지하고, 이번에 추가한 단위 자료의 로컬 검산과 공개 집계표의 출처 확인을 구분한다. 전체 확률 law·이미지 원자료, 모델, 비공개 전달 링크와 미노출 클래스 목록은 공개 저장소에 복사하지 않았다.','', '```bash','python -B research/followup_2026_11/scripts/replay_public_evidence.py --output build/followup-replay','python -B research/followup_2026_11/scripts/build_results_summary.py --check','make verify test reproduce','```','', '## 논문 상태와 다음 판단','', '기존 v1.0.0 논문·DOI·release는 그대로다. 2026-11-20 확인 시 [정정 PR #2](https://github.com/Munsik-Kim/fdca-visual-token/pull/2)와 [통합 정정 draft PR #4](https://github.com/Munsik-Kim/fdca-visual-token/pull/4)는 OPEN·미병합이었다. 이번 결과 총정리는 해당 PR의 병합이나 새 논문 release를 뜻하지 않는다.','', '현재 근거로 다음에 검토할 한 가지는 **여러 난수 평균으로 개발용 이득 label을 만들고 선택·평가 난수를 분리하는 설계**다. 기존 상태에서의 측정 정밀도, 새 상태의 예측 능력, 새 클래스 확인을 각각 검증해야 한다. 추가 표본·새 정책·LT15 실행은 이번 공개 작업에 포함하지 않았다.']
 payload['../../RESULTS_SUMMARY_KO.md']='\n'.join(lines)+'\n'
 en=['# FDCA consolidated results',f'Public-evidence review: **{DATE}**. Original study through LT16. No new GPU experiment.','', '**Controlled context experiments support context-dependent propagation. The tested reference-free precision-restoration policies have not established reliable incremental gains over their specified controls.** Positive oracle diagnostics and unresolved policy results are both retained.','', '[Complete Korean synthesis](RESULTS_SUMMARY_KO.md) · [Normalized result rows](research/followup_2026_11/data/consolidated_results.csv) · [Detailed chronology](research/followup_2026_11/REPORT_KO.md) · [Methods](research/followup_2026_11/METHODS_KO.md)','', '## Findings and limits','', '| Question | Evidence | Limit |','|---|---|---|','| Does shared context change propagation? | M-confirm: 8.6445 fewer mismatches/32, 98.3333% CI [7.1367,10.1992]. LT11: other-class context 6.9727 fewer, 97.5% CI [5.9570,8.1016]; matching the donor class adds 1.7207 [1.3496,2.1211]. | Different panels; LT11 resamples 32 class pairs. Context visibility, values and arrangement are specified interventions. |','| Does an early pulse have larger consequences? | LT3 early-minus-late contrasts: block 2 +0.004121, block3 +0.005000; both adjusted 97.5% CIs positive. OP5 retains the direction with five exposed coordinates. | Pulse discrepancy R is not restoration gain G. OP5 reuses LT3 classes/parents. |','| Do fixed restoration policies improve reference distance? | LT5 two primary intervals and R-test six primary intervals include zero. LT10 retains all 13 adjusted exploratory comparisons. | Inconclusive direction is not proof of zero effect, equivalence or impossibility. |','| Can reference information help? | LT12 reference-token repair +0.080290 [0.058796,0.103802]; block 2 at oracle-known onset +0.017311 [0.000361,0.035429]; 98.75% CIs. Block 3 unresolved. | Reused 64-class oracle diagnostic. Different information and intervention budgets from a deployable policy. |','| Can current W8 features select a one-step repair? | LT14 proposed-minus-ENTROPY OOF gain +0.000618, below the frozen +0.005 gate. LT15 NOT_STARTED_DEV_CRITERIA; 265 reserved classes. | Development is not independent confirmation; reserved identities are not published. |','| Does the observed best time transfer to fresh randomness? | LT16 OLD_MAX +0.058063 originally, −0.000723 on fresh draws, adjusted 98.3333% CI [−0.004644,+0.003300]. Split-half transfer −0.001295, auxiliary 95% CI [−0.003998,+0.001329]. | Reused 24 classes, 48 saved parent paths, 158 states×8 redraws. Conditional saved-state diagnosis, not online policy performance. |','', 'R is pulse output discrepancy; G is the reduction in LPIPS distance to a generated FP32 reference. Neither is human preference or image quality. W8 is reconstructed in FP32 here, so packed-kernel speed/VRAM benefits were not measured. Studies overlap and their sample counts/effects must not be pooled.','', '## Reproduction and publication update','', 'The previous public replay checks 68 evidence hashes, M-confirm/R-test/LT10 contrasts and LT16 tables plus 4,653 independently recalculated values. This update adds aggregate results and a consolidated index. Newly recovered unit-level inputs were independently reaggregated locally, including LT6, LT11, LT12, LT13 and the LT14 development gate; those new unit-level inputs and their replay code are not redistributed. The added LT6 aggregate is source-hash linked, but its unit-level replay requires the retained local inputs. See the [aggregate source manifest](research/followup_2026_11/data/aggregate_source_manifest_20261120.json) and [validation boundaries](research/followup_2026_11/EVIDENCE_AND_VALIDATION.md).','', '```bash','python -B research/followup_2026_11/scripts/replay_public_evidence.py --output build/followup-replay','python -B research/followup_2026_11/scripts/build_results_summary.py --check','make verify test reproduce','```','', 'No checkpoint, full raw probability/image archive, private delivery URL or reserved holdout identity is redistributed. Scalar numerical replay does not rerun the generator, decoder or LPIPS network.','', 'The original paper, v1.0.0 release and DOI remain historical. [PR #2](https://github.com/Munsik-Kim/fdca-visual-token/pull/2) and the [integration draft PR #4](https://github.com/Munsik-Kim/fdca-visual-token/pull/4) were open and unmerged on this review date; this publication does not merge them or create a release. A next development design would average gain labels over independent redraws and separate selection from evaluation randomness. It has not been executed.']
 payload['../../RESULTS_SUMMARY.md']='\n'.join(en)+'\n'
 return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
 outputs=build()
 for rel,text in outputs.items():
  path=BASE/rel
  if a.check:
   assert path.is_file() and path.read_text()==text, f'Stale generated result index: {rel}'
  else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 print(json.dumps({'status':'PASS','mode':'check' if a.check else 'write','files':len(outputs),'rows':len(collect())}))
if __name__=='__main__':main()
