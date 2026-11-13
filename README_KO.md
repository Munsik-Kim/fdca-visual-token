# 마스킹 비주얼 생성기를 위한 FDCA

**2026년 11월 추가 연구 포트폴리오:** [전체 연구 보고서](research/followup_2026_11/REPORT_KO.md) · [모델 구조·비교군](research/followup_2026_11/METHODS_KO.md) · [실패 원인 분석](research/followup_2026_11/FAILURE_ANALYSIS_KO.md) · [다음 연구 기준](research/followup_2026_11/NEXT_RESEARCH_KO.md).

LT0–LT16의 목표, 전파·문맥 결과, 복원 정책의 미확정 결과와 개발 기준 실패, LT15 미실행 결정을 함께 정리했습니다. 참조 정보를 쓴 LT12의 양의 복원 결과도 보존했습니다. 원표에 연결된 그래프 4개와 주요 결과의 모델 없는 수치 재생 코드를 제공합니다. 이미지 품질 개선이나 복원 전체의 불가능성을 주장하지 않습니다.

**버전 구분:** 아래 논문과 표는 기존 `v1.0.0` 기록입니다. [과학적 정정 PR #2](https://github.com/Munsik-Kim/fdca-visual-token/pull/2)는 이 추가 보고서 작성 당시 별도의 미병합 수정안입니다. `M_equal`, `M_split`, `R=mean(h*m)`을 구분하며 평균의 곱에는 covariance가 빠진다는 점에 주의해야 합니다. 추가 보고서는 이 정의를 적용하고 기존 논문·release·DOI를 보존합니다.

이미지 생성 모델의 가중치를 적은 비트로 바꿨을 때, 처음 달라지는 선택과 최종 이미지의 변화를 추적하는 도구를 구현했습니다. 같은 조건의 비교 실험, 수학적 분석, CPU 검산 코드를 함께 제공합니다.

**Munsik Kim** · [ORCID](https://orcid.org/0009-0008-9350-2435)

**기술 프리프린트이며 동료평가를 받지 않았습니다.** 최초 공개판은 `v1.0.0`입니다. [논문 PDF](paper/FDCA_v1.0.0.pdf) · [Zenodo](https://zenodo.org/records/21965747)

두 실행의 난수를 맞춰 비교하고, 처음 다른 선택이 발생할 빈도와 그 이후의 영향을 나누어 계산합니다. 이미지 조각의 중간 표현을 채워 넣는 생성 모델 두 개에서 시험했습니다. 구현에는 고정된 선택 순서(Halton), 대응되는 난수 생성, 8비트 반올림 양자화(W8 RTN), 첫 차이 이후의 공통 실행 경로가 포함됩니다.

## 핵심 결과

| Checkpoint | Contrast | rho 0.5 | rho 0.3 | Ratio | Paired 95% CI for difference |
|---|---:|---:|---:|---:|---:|
| H-MaskGIT-T | Conditional LPIPS | 0.03013 | 0.01013 | 2.98 | [0.01719, 0.02287] |
| H-MaskGIT-T | Incidence-weighted LPIPS | 0.00393 | 0.00221 | 1.77 | [0.00137, 0.00207] |
| H-MaskGIT-S | Conditional LPIPS | 0.03076 | 0.01038 | 2.96 | [0.01714, 0.02381] |
| H-MaskGIT-S | Incidence-weighted LPIPS | 0.00411 | 0.00214 | 1.92 | [0.00150, 0.00248] |

이 수치는 두 H-MaskGIT 체크포인트와 운영적 LPIPS/DINO 지표에 한정됩니다. 사람의 지각적 정답, 다른 아키텍처, coupling 비의존성 또는 production controller 효과를 주장하지 않습니다.

## 재현

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-model-free.txt
make verify test reproduce
```

체크포인트, VQ·metric weights, datasets, full trajectories, 외부 clone과 Conda 환경은 포함하지 않습니다. 원본 코드는 MIT, 논문·표·그림·공개 감사 문서는 CC BY 4.0입니다.
