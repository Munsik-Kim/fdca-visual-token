# 마스킹 비주얼 생성기를 위한 FDCA

**Munsik Kim** · [ORCID](https://orcid.org/0009-0008-9350-2435)

**기술 프리프린트이며 동료평가를 받지 않았습니다.** 최초 공개판은 `v1.0.0`입니다. [논문 PDF](paper/FDCA_v1.0.0.pdf) · [Zenodo](https://zenodo.org/records/21965747)

FDCA는 기준 디코더와 근사 디코더를 결합해 최초 분기를 식별하고, 위험을 정확한 분기 발생률과 분기 조건부 결과로 분해합니다. 본 저장소는 고정 Halton 스케줄, 조건부 좌표별 maximal coupling, event-addressed RNG, 현실적 W8 RTN, natural seed 이후 공통 W8 suffix를 공개합니다.

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
