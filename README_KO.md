# 마스킹 비주얼 생성기를 위한 FDCA

FDCA는 양자화된 한 번의 전이에서 토큰 선택이 달라질 확률과, 달라졌을 때 최종 두 출력이 얼마나 차이 나는지를 분리해 측정합니다. 자연 양자화 seed, 명시적 결합, 공통 전파 kernel, 감사된 난수, 블록 단위 대응 추론을 하나의 측정 계약으로 제공합니다.

**Munsik Kim** · [ORCID 0009-0008-9350-2435](https://orcid.org/0009-0008-9350-2435)

**동료평가를 받지 않은 기술 프리프린트입니다.** 현재 GitHub 수정 후보는 **v1.0.1, 미릴리스**입니다. [수정 PDF](paper/FDCA_v1.0.1.pdf) · [수정 LaTeX](paper/source/v1.0.1/FDCA_v1.0.1.tex) · [정정 기록](docs/REVISION_NOTES_v1.0.1.md) · [English](README.md)

기존 **v1.0.0**의 [PDF](paper/FDCA_v1.0.0.pdf), [소스](paper/source/FDCA_v1.0.0.tex), [Zenodo 기록](https://zenodo.org/records/21965747)은 보존됩니다. DOI [10.5281/zenodo.21965747](https://doi.org/10.5281/zenodo.21965747)은 기존 보관판을 가리키며 이번 수정 후보의 새 DOI가 아닙니다.

## 측정 계약과 주요 결과

공통 anchor 상태에서 **한 번의 FP32 대 W8 전이**를 비교한 뒤 두 가지 모두 같은 W8 kernel로 진행합니다. 고정된 32-step Halton schedule을 사용하며 rho는 **목표 미해결 토큰 비율**입니다. 클수록 생성 초기입니다. 같은 panel의 동일 class–generation seed 블록 안에서 rho .3과 .5를 대응시킵니다.

`h[b]`는 좌표 독립 결합 아래 계산한 batch split probability이며 이항 발생 빈도가 아닙니다. `m[b]`는 분기에 조건화한 LPIPS 추정값입니다. `H=mean(h)`, `M_equal=mean(m)`, `R=mean(h*m)`을 구분합니다. `M_split=R/H`는 split에 조건화한 가중 평균입니다. 일반적으로 `R=H*M_equal+population_covariance(h,m)`이며 평균 두 개의 곱으로 R을 계산할 수 없습니다.

<!-- GENERATED_HEADLINE_START -->

| Panel | Metric | n | rho .3 | rho .5 | Delta (.5-.3) | Legacy 95% CI |
|---|---|---:|---:|---:|---:|---|
| Tiny CFG3 W8 | H | 48 | 0.213227 | 0.129756 | -0.083471 | Not a frozen headline endpoint; see post-hoc supplement |
| Tiny CFG3 W8 | M_equal | 48 | 0.010126 | 0.030126 | 0.020000 | [0.017189, 0.022874] |
| Tiny CFG3 W8 | R | 48 | 0.002213 | 0.003927 | 0.001714 | [0.001368, 0.002066] |
| Small CFG3 W8 | H | 40 | 0.199666 | 0.127957 | -0.071710 | Not a frozen headline endpoint; see post-hoc supplement |
| Small CFG3 W8 | M_equal | 40 | 0.010380 | 0.030756 | 0.020376 | [0.017142, 0.023806] |
| Small CFG3 W8 | R | 40 | 0.002140 | 0.004106 | 0.001966 | [0.001499, 0.002477] |

<!-- GENERATED_HEADLINE_END -->

표의 기존 headline CI는 **20,000회 / 저장 semantic seed / higher quantile** 대응 블록 bootstrap입니다. H의 CI와 비율·견고성은 **100,000회 / SHA-256 label seed / linear quantile**의 [사후 보조분석](public_results/reanalysis_v1_0_1/README.md)으로 분리했습니다. [H·M_equal·R 그림](public_results/reanalysis_v1_0_1/figures/main_panels.pdf) · [대응 차이](public_results/reanalysis_v1_0_1/figures/paired_contrasts.pdf) · [Tiny 전체 rho](public_results/reanalysis_v1_0_1/figures/tiny_all_rho.pdf).

한 모델 계열의 두 체크포인트에서 H는 감소하고 M_equal과 R은 증가합니다. 동시에 commit batch는 14에서 8로 변합니다. 좌표당 평균 TV는 전반적으로 감소하지 않으므로 H 감소를 local quantization error 감소로 해석할 수 없습니다. LPIPS/DINO는 두 출력의 discrepancy이며 사람의 지각·품질·선호 또는 지속적 FP32 대 W8 생성의 품질 저하를 입증하지 않습니다.

## 재현과 PDF 빌드

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements-revision.txt
make verify test reproduce
make reanalyse
make paper-revision              # Tectonic 필요; 수정 PDF와 새 bbl 저장
python scripts/build_paper.py --version 1.0.1 --check
make paper                       # 기존 v1.0.0 페이지 수 검증은 별도
```

기존 8개 대비의 CI **양쪽 끝점**과 저장된 OOF MAE 비율 2개를 검사합니다. 원 feature pipeline이나 LOBO 모델을 다시 학습한 것은 아닙니다. 재분석 script는 JSON·CSV·LaTeX·PDF/SVG/PNG를 생성합니다. `python scripts/reanalyse_revision.py --check`는 보관 결과를 바꾸지 않고 검산합니다. [재현 범위](docs/REPRODUCIBILITY_TIERS.md) · [실제 compile 기록](paper/build_v1.0.1.json).

## 표본과 한계

주 패널은 Tiny 48 / Small 40 블록, anchor별 조건부 반복 8회입니다. Small CFG1 16 블록·4회와 Tiny W4 stress 24 블록·4회는 겹치는 subset입니다. 별도 matched timing은 Tiny 24 / Small 40 블록·2회입니다. 반복·좌표·anchor·이미지는 독립 표본 수를 늘리지 않습니다. 공개 compact 표에 완전한 class/seed manifest가 없어 class-cluster 분석과 이전 단계 중복의 독립 전수 검증은 불가능합니다.

동결 runner의 양자화 구현은 **per-output-channel symmetric weight-only RTN fake quantization**입니다. P2P YAML의 per-tensor 계획과의 차이는 [deviation 근거](docs/evidence/QUANTIZER_SOURCE_RECEIPT.json)로 남겼습니다. 실행 당시 scale/weight tensor까지 재실행한 검증은 아닙니다. Packed kernel의 속도·VRAM 절감 실측도 아닙니다.

P05R의 공허한 envelope, P1M의 제한적인 추가 설명력, 당시 미완료 P2A 및 이후 P2B 감사 closure를 보존합니다. W4는 stress 결과입니다. [적용 범위](docs/SCIENTIFIC_SCOPE.md) · [실험 이력](docs/EXPERIMENT_LEDGER.md) · [주장 근거 지도](docs/SOURCE_TO_CLAIM_MAP.md) · [증명 ledger](docs/PROOF_LEDGER_v1.0.1.md). 수정 논문의 후속 실험은 **미실행 계획**입니다.

이번 수정에는 새 GPU 추론·학습·이미지 생성·모델 다운로드가 없습니다. 외부 가중치, 데이터셋, 전체 trajectory, 개인 로그는 배포하지 않습니다. 저자·ORCID·라이선스를 유지하며 `CITATION.cff`와 `.zenodo.json`은 기존 v1.0.0 기록을 가리킵니다. 코드는 MIT, 논문·표·그림·공개 감사 자료는 CC BY 4.0입니다. [라이선스 범위](LICENSE_SCOPE.md).
