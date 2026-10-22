# 마스킹 비주얼 생성기를 위한 FDCA

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
