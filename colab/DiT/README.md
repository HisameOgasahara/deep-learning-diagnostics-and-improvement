# Conditional DiT + Rectified Flow diagnostics

Google Colab T4용 FashionMNIST conditional DiT + Rectified Flow 실습입니다.

## 실험 질문

분류 모델 진단법을 그대로 옮기기보다, diffusion / flow 문헌에서 반복적으로 분석되는 생성 시간 축을 중심으로 다음 연쇄를 봅니다.

```text
training state
→ timestep-conditioned velocity field
→ DiT representation
→ local generative geometry
→ sampling trajectory
→ generated distribution
```

## 파일

- `fashion_mnist_conditional_dit_rectified_flow_diagnostics.ipynb`
  - 전체 실습 노트북
  - 각 진단 단계마다 표/그래프 출력
  - 마지막에 기법별 참고 논문과 활용 방식 정리
- `dit_rf_model.py`
  - 읽기 쉬운 mini conditional DiT
  - adaLN-Zero style conditioning
  - pixel-space Rectified Flow batch construction
- `dit_rf_train.py`
  - 50 epoch 학습 루프
  - timestep-binned loss와 checkpoint 저장
  - FashionMNIST-specific frozen feature encoder 학습
- `dit_rf_diagnostics.py`
  - velocity / divergence
  - layer × timestep probe / class-subspace geometry
  - attention locality
  - local velocity Jacobian spectrum
  - Euler / Heun / RK4 sampling
  - trajectory geometry
  - generated-distribution summary

## 기본 설정

- Dataset: FashionMNIST, Hugging Face 우선 다운로드
- Model: pixel-space mini DiT, patch 4, dim 192, depth 6, heads 6
- Optimizer: AdamW
- Epochs: 50
- Checkpoints: 0 / 10 / 25 / 50
- Main generation-time grid: 10 bins
- Solvers: Euler / Heun / RK4
- Matched NFE: 8 / 16 / 32
- Target runtime: Colab T4 16GB

## 설계 원칙

이 노트북의 목적은 ablation으로 최고 점수를 찾는 것이 아니라, 서로 다른 생성 시간·학습 시점·solver 조건에서 진단량이 어떤 양상을 보이는지 관찰하는 것입니다.

비싼 Jacobian 진단은 대표 timestep만 사용하고, 나머지 진단은 더 촘촘한 timestep grid를 사용합니다.
