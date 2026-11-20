# FDCA 전체 결과 총정리
공개 근거 점검 기준: **2026-11-20** · 원 논문부터 LT16까지 · 새 GPU 실험 없음

**전파를 통제하고 측정하는 근거는 확보했지만, 그 정보를 이용하는 참조 없는 정밀도 복원 정책의 안정적인 추가 이득은 확인하지 못했다.** 문맥 개입과 참조 정보를 쓰는 토큰 수정의 양의 결과, 정책 평가의 미확정 결과, 개발 단계 중단을 함께 보존한다.

[English](RESULTS_SUMMARY.md) · [쉬운 설명](START_HERE_KO.md) · [상세 연구 이력](research/followup_2026_11/REPORT_KO.md) · [모든 주요 수치 CSV](research/followup_2026_11/data/consolidated_results.csv) · [검증 범위](research/followup_2026_11/EVIDENCE_AND_VALIDATION.md)

## 먼저 구분할 세 가지 양

- **전파:** 시작 차이 뒤 새로 확정한 위치의 토큰 불일치. 문맥 실험의 표는 후속 32좌표 중 감소한 토큰 개수다.
- **Pulse R:** 한 단계 개입 여부를 달리한 뒤 모두 Q로 진행한 두 경로의 종단 LPIPS 기대거리. 분기 조건부 추출에서는 `mean(h*m)`으로 복원한다. 큰 R이 좋은 복원이라는 뜻은 아니다.
- **복원 G:** `LPIPS(FP32 참조, W8) − LPIPS(FP32 참조, 복원 경로)`. 양수는 지정한 결합에서 참조와의 거리 감소다. 이미지 품질·사람 선호·packed INT8 속도는 별도 측정이다.

11월 후속 연구는 H-MaskGIT-T의 W8 반올림 가중치를 FP32로 재구성해 실행했다. 초기 논문의 Tiny/Small 비교와 섞어 표본 수나 효과를 합치지 않는다. `mean(h*m)`과 `mean(h)*mean(m)` 사이 covariance를 생략하지 않는다.

## 전파에 관한 결과

| 단계 / 비교 | 추정치 | 원 신뢰구간 | 독립 집계 단위 |
|---|---:|---|---|
| [LT6 / 공통 문맥 추가](research/followup_2026_11/evidence/tables/lt6_primary_20261120.csv) | +8.251953 | 95% [+7.261719, +9.222656] | 32 클래스 |
| [M-confirm / 문맥 추가](research/followup_2026_11/evidence/tables/m_confirm_contrasts.csv) | +8.644531 | 98.333333% [+7.136719, +10.199219] | 32 클래스 |
| [M-confirm / 원래 배치](research/followup_2026_11/evidence/tables/m_confirm_contrasts.csv) | +1.238281 | 98.333333% [+0.671875, +1.890625] | 32 클래스 |
| [M-confirm / 가시 범위 증가](research/followup_2026_11/evidence/tables/m_confirm_contrasts.csv) | +1.976562 | 98.333333% [+1.570312, +2.417969] | 32 클래스 |
| [LT11 / 타 클래스 공통 문맥](research/followup_2026_11/evidence/tables/lt11_contrasts.csv) | +6.972656 | 97.5% [+5.957031, +8.101562] | 32 클래스 쌍 |
| [LT11 / 문맥 출처 일치의 추가 효과](research/followup_2026_11/evidence/tables/lt11_contrasts.csv) | +1.720703 | 97.5% [+1.349609, +2.121094] | 32 클래스 쌍 |

LT6는 LT3의 32클래스를 재사용한 512쌍 진단이다. M-confirm은 새 실험 클래스 32개, LT11은 새 클래스 64개를 32개 쌍으로 묶은 별도 연구다. LT11의 C_shared는 다른 클래스 출처 문맥을 공유하는 효과, C_match는 자기 클래스와 출처를 맞춘 추가 효과다. 문맥의 가시성·값·배치가 함께 규정된 개입이므로 일반적인 시간 효과나 모든 문맥의 보편적 효과로 확대하지 않는다.

LT3의 초기 pulse 대비는 block 2 **+0.004121**, block 3 **+0.005000**으로 두 97.5% 구간이 양수였다. 같은 부모를 재사용해 복원 law 노출을 5좌표로 제한한 LT4-OP5에서도 각각 **+0.004806**, **+0.005937**이었다. 이는 pulse 출력 차이의 크기이며 G와 다른 양이다.

## 실제 참조 거리 감소와 선택 능력

| 단계 / 비교 | 추정치 | 원 신뢰구간 | 독립 집계 단위 |
|---|---:|---|---|
| [LT5 / FDCA 대 W8](research/followup_2026_11/evidence/tables/lt5_primary.csv) | -0.004161 | 97.5% [-0.025651, +0.018338] | 32 클래스 |
| [LT5 / FDCA 대 국소 TV](research/followup_2026_11/evidence/tables/lt5_primary.csv) | -0.004573 | 97.5% [-0.026642, +0.018362] | 32 클래스 |
| [R-test / F 대 W8](research/followup_2026_11/evidence/tables/r_test_primary.csv) | +0.007713 | 99.166667% [-0.001659, +0.017488] | 320 클래스 |
| [R-test / F 대 국소 TV](research/followup_2026_11/evidence/tables/r_test_primary.csv) | +0.005103 | 99.166667% [-0.004562, +0.014929] | 320 클래스 |
| [LT12 / 참조 토큰 편집 대 W8](research/followup_2026_11/evidence/tables/lt12_primary.csv) | +0.080290 | 98.75% [+0.058796, +0.103802] | 64 클래스 |
| [LT12 / 참조 편집 대 대체 토큰 편집](research/followup_2026_11/evidence/tables/lt12_primary.csv) | +0.079728 | 98.75% [+0.060338, +0.099947] | 64 클래스 |
| [LT12 / oracle 시점 block 2 복원 대 W8](research/followup_2026_11/evidence/tables/lt12_primary.csv) | +0.017311 | 98.75% [+0.000361, +0.035429] | 64 클래스 |
| [LT12 / oracle 시점 block 3 복원 대 W8](research/followup_2026_11/evidence/tables/lt12_primary.csv) | +0.008156 | 98.75% [-0.005976, +0.022815] | 64 클래스 |
| [LT13 / 자기 시점 대 교환 시점, 원 draw](research/followup_2026_11/evidence/tables/lt13_contrasts.csv) | +0.014858 | 95% [+0.003224, +0.027186] | 32 클래스 쌍 |
| [LT13 / 자기 시점 대 교환 시점, 새 draw](research/followup_2026_11/evidence/tables/lt13_contrasts.csv) | -0.012603 | 95% [-0.038779, +0.008083] | 8 클래스 쌍 |
| [LT16 / OLD_MAX 대 W8](research/followup_2026_11/evidence/lt16/expected/primary_contrasts.csv) | -0.000723 | 98.333333% [-0.004644, +0.003300] | 24 클래스 |
| [LT16 / OLD_MAX 대 ENTROPY](research/followup_2026_11/evidence/lt16/expected/primary_contrasts.csv) | +0.000504 | 98.333333% [-0.004331, +0.005506] | 24 클래스 |
| [LT16 / PROPOSED 대 TIME_ONLY](research/followup_2026_11/evidence/lt16/expected/primary_contrasts.csv) | +0.000162 | 98.333333% [-0.000254, +0.001012] | 24 클래스 |

LT5와 R-test의 첫 두 행은 각각 절대 G와 국소 TV 대비 추가 G다. LT16의 B는 OLD_MAX−ENTROPY, C는 PROPOSED−TIME_ONLY다. LT13 원 draw 비교는 32개 클래스 쌍, 새 draw 비교는 8개 쌍이며 두 표본을 합치지 않는다.

표의 LPIPS 단위는 같아도 표본·개입·정보·CI 수준이 다르다. 행 사이 CI 끝점을 빼서 직접 비교하지 않는다. LT12의 시점과 참조 토큰은 FP32 경로에서 얻었으며 R-test 집단을 재사용했다. A_ref/A_b2/A_b3는 트리거 없는 1개를 포함한 128경로, B_dir는 ALT가 가능한 127경로를 클래스 64개에 동일 가중한 결과다. 참조 토큰 수정과 가중치 한 단계 복원은 같은 비용의 행동이 아니다. block 2의 양의 구간과 block 3의 0 포함만으로 두 block의 우열을 선언하지 않는다.

**LT14 개발 중단:** 제안 정책 G는 +0.005884, 가장 강한 동일 행동 대조 ENTROPY는 +0.005266이었다. 차이 **+0.000618**은 사전 진행 기준 +0.005에 못 미쳤다. 양의 fold는 3/4였지만 세 조건을 모두 만족해야 하므로 LT15는 `NOT_STARTED_DEV_CRITERIA`다. 예약 265클래스의 식별자는 공개하지 않았으며 해당 확인 실험을 수행했다고 보고하지 않는다.

**LT16 새 난수 진단:** 원 draw에서 32시점 중 최대를 고르면 평균 +0.058063이었으나, 선택된 저장 상태에서 8회 새 draw로 측정한 OLD_MAX 평균은 -0.000723이었다. 같은 절반으로 선택·평가하면 +0.009127, 다른 절반으로 평가하면 -0.001295였다. 후자는 보조 95% CI [-0.003998,+0.001329]다. 현재 특징의 추가 선택 능력을 입증하지 못했으며, 모든 복원 알고리즘이 불가능하다는 결론도 아니다.

다섯 LT16 평균: OLD_MAX -0.000723, PROPOSED +0.000953, ENTROPY -0.001228, TIME_ONLY +0.000791, FIXED -0.000434. 세 주 구간은 모두 0을 포함한다. 원 48경로·158개 선택 상태·8draw를 재사용한 진단이며 새 클래스 평가가 아니다.

## 전체 진행 지도

행의 표본은 서로 중첩되므로 더해서 총 독립 표본 수를 만들지 않는다. 기존 요약·음의 결과·미완료는 원 근거에 남긴다.

| 단계 | 질문과 설계 | 결과 / 근거 |
|---|---|---|
| 원 논문 T0–P3S | 합성 법칙, sampler·RNG, 인공·자연 분기, Tiny/Small 출력 차이 | [원 실험 ledger](docs/EXPERIMENT_LEDGER.md). P05R 상한은 유효하지만 실용적으로 무용(vacuous)했고, P1M은 순위 정보만 지지했다. P2A 불완전 실행과 P2B 감사 통과를 구분한다. |
| LT0/LT1 | 16 classes; 32 per arm; 8 arms | 16클래스 지속 W8 pilot; 층 순위는 기술량 · [근거](research/followup_2026_11/evidence/tables/lt1_arms.csv) |
| LT1-R | 8 classes; 16 parents; 96 restoration paths | 8클래스 재사용·지속 복원; 모든 절대 G 구간 0 포함 · [근거](research/followup_2026_11/evidence/tables/lt1r_restoration.csv) |
| LT2 | 8 classes; 256 diagnostics; 1024 pulse pairs | 국소 상태 진단·자연 pulse; 희소 분기 · [근거](research/followup_2026_11/evidence/tables/lt2_contrasts.csv) |
| LT2C | 8 classes; 16 parents; 768 main + 64 terminal pairs | 조건부 pulse를 h로 가중; 초기 차이 탐색 · [근거](research/followup_2026_11/evidence/tables/lt2c_contrasts.csv) |
| LT3 | 32 classes; 64 parents; 1792 conditional pairs | 새 32클래스에서 두 초기 대비 양수 · [근거](research/followup_2026_11/evidence/tables/lt3_contrasts.csv) |
| LT4/OP5 | 32 classes; 1024 new + 512 early reused pairs | LT3 재사용·5좌표 노출; 초기 차이 유지 · [근거](research/followup_2026_11/evidence/tables/lt4_contrasts.csv) |
| LT5 | 32 classes; 64 units; 64 references + 320 candidates | 새 32클래스·같은 8 block-step; 두 주 G 미확정 · [근거](research/followup_2026_11/evidence/tables/lt5_primary.csv) |
| LT6 | 32 classes; 512 pairs | LT3 재사용·같은 최초 차이와 W32; 공통 문맥 효과 · [근거](research/followup_2026_11/evidence/tables/lt6_primary_20261120.csv) |
| LT7/M-dev | 32 classes; 512 paired units in M-dev | 후속 비교군 개발; 새 확인으로 세지 않음 · [근거](research/followup_2026_11/evidence/tables/m_dev_contrasts.csv) |
| M-confirm/LT8 | 32 classes; 256 paired units | 새 32클래스; 문맥·배치·가시 범위 세 방향 확인 · [근거](research/followup_2026_11/evidence/tables/m_confirm_contrasts.csv) |
| R0 | 32 classes; 64 units | 재사용 panel의 8-step 대 1-step; 기간 우열 미확정 · [근거](research/followup_2026_11/evidence/tables/r0_contrasts.csv) |
| R-cal | 8 classes; development policy calibration | 8개 개발 클래스에서 정책 선택; 성능 확인 아님 · [근거](research/followup_2026_11/evidence/tables/r_test_policy.json) |
| R-test/LT9 | 320 classes; 640 units; 5760 endpoints | 새 320클래스; 여섯 주 구간 모두 0 포함 · [근거](research/followup_2026_11/evidence/tables/r_test_primary.csv) |
| R-alt | 32 classes; paired coupling subset | 32클래스 subset의 결합 민감도; 결합 동등성 미입증 · [근거](research/followup_2026_11/evidence/tables/r_alt_contrasts.csv) |
| LT10 | 320 classes; same 640 R-test units | R-test 재분석 13개 대비; 보정 구간 모두 0 포함 · [근거](research/followup_2026_11/evidence/tables/lt10_13_contrasts.csv) |
| LT11 | 64 classes; 32 class pairs; 512 units | 새 64클래스/32쌍; 공유·출처 일치 추가 효과 · [근거](research/followup_2026_11/evidence/tables/lt11_contrasts.csv) |
| LT12 | 64 classes; 128 units; 127 trigger-eligible | 재사용 64클래스; oracle 토큰 수정과 한 단계 복원 · [근거](research/followup_2026_11/evidence/tables/lt12_primary.csv) |
| LT13 | 64 original; 16 redraw classes; 32 original class pairs; 8 redraw pairs | 원 draw 자기 시점 우위; 새 draw에서는 미확정 · [근거](research/followup_2026_11/evidence/tables/lt13_contrasts.csv) |
| LT14 | 24 classes; 48 parents x 32 candidate times | 24클래스 개발; ENTROPY 대비 진행 기준 실패 · [근거](research/followup_2026_11/evidence/tables/lt14_policy_summary.csv) |
| LT15 | 0 opened; 265 reserved classes; 0 model calls | 미실행; 새 확인 성과 없음 · [근거](research/followup_2026_11/evidence/tables/lt14_dev_gate.json) |
| LT16 | 24 classes; 48 parents; 158 states x 8 draws = 1264 | 24클래스 저장 상태 재사용; 새 난수에서 관측 최대의 양의 이득 미확인 · [근거](research/followup_2026_11/evidence/lt16/expected/primary_contrasts.csv) |

## 이번 공개 점검에서 보완한 것

2026-11-13 공개 당시 LT6 standalone 근거를 확보하지 못했던 범위는 역사적 기록으로 남긴다. 이번에는 실제 보존본에서 최소 단위 수치를 다시 확보해 원 분석 계약으로 수치를 재생하고 집계 결과를 추가했다. 새로운 단위별 입력은 로컬에 보존했다. LT6 집계표의 출처와 변환은 [별도 해시 manifest](research/followup_2026_11/data/aggregate_source_manifest_20261120.json)에 연결한다. 기존 68개 공개 근거 파일과 역사적 수치는 바꾸지 않는다.

저장 scalar 수치에서 class/pair 집계와 CI를 재계산하는 범위, 저장된 full categorical law의 token replay, 실제 모델·decoder·LPIPS 재실행은 서로 다르다. 이번 작업은 보존된 수치의 로컬 재생과 저장소 검증이며 새 GPU 실행은 없다. 기존 공개 재생은 유지하고, 이번에 추가한 단위 자료의 로컬 검산과 공개 집계표의 출처 확인을 구분한다. 전체 확률 law·이미지 원자료, 모델, 비공개 전달 링크와 미노출 클래스 목록은 공개 저장소에 복사하지 않았다.

```bash
python -B research/followup_2026_11/scripts/replay_public_evidence.py --output build/followup-replay
python -B research/followup_2026_11/scripts/build_results_summary.py --check
make verify test reproduce
```

## 논문 상태와 다음 판단

기존 v1.0.0 논문·DOI·release는 그대로다. 2026-11-20 확인 시 [정정 PR #2](https://github.com/Munsik-Kim/fdca-visual-token/pull/2)와 [통합 정정 draft PR #4](https://github.com/Munsik-Kim/fdca-visual-token/pull/4)는 OPEN·미병합이었다. 이번 결과 총정리는 해당 PR의 병합이나 새 논문 release를 뜻하지 않는다.

현재 근거로 다음에 검토할 한 가지는 **여러 난수 평균으로 개발용 이득 label을 만들고 선택·평가 난수를 분리하는 설계**다. 기존 상태에서의 측정 정밀도, 새 상태의 예측 능력, 새 클래스 확인을 각각 검증해야 한다. 추가 표본·새 정책·LT15 실행은 이번 공개 작업에 포함하지 않았다.
