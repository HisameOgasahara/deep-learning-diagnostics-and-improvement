https://chatgpt.com/share/6aa8d22d-549c-83e8-a2f0-d68e28792345

# Practice Method Experiment Standard

각 노트북의 목적을 **“방법을 재현하는 튜토리얼”보다 한 단계 더 나아가, 그 방법을 실제로 언제·어떻게 써야 하는지 체득하는 실험장**으로 잡는다.

각 **방법(method)**마다 고정 양식을 강제하지 않고, 필요에 따라 아래 요소를 선택해서 붙인다.

## 1. 구현

- 수식/알고리즘 → 읽을 수 있는 PyTorch 구현.
- 발전사에서 이전 방법과 달라진 부분이 코드상 어디인지 드러나게 함.

## 2. 구현 검증

- 방법마다 reference가 다름.
- PyTorch 지원 방법 → PyTorch API와 output/gradient/update parity.
- SOTA/논문 고유 방법 → 미리 준비한 faithful tiny reference와 구조·tensor path·forward/backward parity.
- 예: 기본 attention은 SDPA, Kimi K3 attention은 K3 reference를 사용.
- 검증은 노트북 단위의 고정 셀이 아니라 **방법마다 각각 다르게** 둔다.
  - 같은 attention 노트북 안에서도 SDPA 기반 attention은 PyTorch reference와 비교하고, Kimi K3 같은 SOTA attention은 해당 모델용 tiny reference와 비교한다.
  - optimizer도 PyTorch 지원 optimizer는 `torch.optim`과 update/state parity를 보고, 논문 구현은 해당 reference와 비교한다.
  - loss도 PyTorch 지원형은 loss 값과 gradient parity, 논문 loss는 reference 구현과 loss/gradient parity를 본다.

## 3. 계산 구조 관찰 — 의미 있을 때

- profiler로 operator graph / CUDA kernel / memory 이동 등을 확인.
- 해당 방법의 장점이 계산효율과 직접 연결된다면 중요도가 올라감.
- 필요하면 작은 CUDA/Triton 구현까지 내려가서 **왜 빨라지는지** 직접 비교.
- 반대로 kernel 차이가 핵심이 아니면 생략.
- profiler는 별도 대분류가 아니라 `00_foundation`에서 한 번 배우고, 이후 각 방법에서 필요한 경우 다음 셀로 붙인다.

## 4. 동작과 효과 관찰

- “모든 방법에서 같은 metric”을 강제하지 않음.
- 그 방법이 바꾼 메커니즘이 실제로 움직였는지를 보는 실험을 선정.
- optimizer면 update dynamics, normalization이면 activation/gradient scale, attention 최적화면 memory/compute처럼 내용에 맞춤.

## 5. 방법 내부의 중요한 자유도

- 옵션, HP, 구현 선택지가 방법의 성질을 크게 바꾸는 경우에만 비교.
- 예: Adam의 $\beta_1,\beta_2$, attention의 head/KV 구성, sampler의 step 수 등.
- 단순 옵션 나열은 하지 않음.

## 6. 사용 조건·레짐·커스터마이징

- 이게 최종적으로 가장 중요한 부분.
- 논문 설정을 그대로 따라 하는 게 목적이 아니라:
  - 어떤 데이터/모델 크기/compute/HP 범위에서 이 방법이 유리한가?
  - 어떤 조건에서는 이점이 사라지는가?
  - 무엇과 조합해야 효과가 나는가?
  - 병목은 어디로 이동하는가?
  - scale을 바꾸면 같은 recipe가 유지되는가?
  - 작은 T4 환경에 가져오려면 무엇을 보존하고 무엇을 조정해야 하는가?
  - 논문이 그 방법이 유리하게 작용하기 위해 상정한 환경은 무엇인가?
  - 잘 작동하는 환경과 약점으로 작용하는 환경은 무엇인가?
  - 동작 과정의 주요 병목은 무엇인가?
  - 실패하는 조건에서 어떤 개량이나 다른 개선을 생각할 수 있는가?
- 필요하면 여기서 조합실험·ablation·scaling sweep을 추가.

그래서 하나의 발전사 노트북은 사실 이런 모습이 된다.

```text
Method A
  구현
  → A에 적절한 검증
  → 필요한 동작/계산 관찰
  → 조건 변화 실험

        ↓ 왜 B가 필요한가?

Method B
  구현
  → B에 적절한 검증
  → A와 달라진 메커니즘 확인
  → B가 이득을 내는 조건 / 잃는 조건 확인

        ↓

Method C (SOTA)
  구현
  → faithful tiny reference 검증
  → 필요하면 profiler / Triton·CUDA
  → HP·scale·조합 변화
  → 실제 사용할 조건과 병목 파악
```

따라서 **발전 순서는 논문사가 뼈대지만, 실험의 기준은 논문을 설명하는 것이 아니라 “이 방법을 내가 다른 환경에 가져가서 쓸 수 있게 만드는 것”**이 된다.

---

# Practice 최종 대분류

```text
practice/
├─ 00_foundation/
├─ 01_architecture/
├─ 02_representation/
├─ 03_loss_objective/
├─ 04_training_recipe/
└─ 05_inference_in_generation_and_control/
```

각 파트는 원칙적으로 **교과서 또는 PyTorch 기본 지원 구조에서 시작해서, 실제 최신 SOTA 모델/논문에서 쓰이는 구조로 발전하는 양상**으로 구성한다.

- PyTorch가 지원하는 방법은 직접 구현과 PyTorch 함수의 구조·출력·gradient/update를 비교해 검증한다.
- 논문/SOTA급 방법은 `paper_implementation_verifier_standard.md`의 전략으로 검증한다.
- 예: 기본 attention은 PyTorch SDPA와 parity, Kimi K3의 attention 같은 구조는 HF metadata/config + 공식/참조 코드에서 만든 tiny reference와 구조/forward/backward parity를 본다.
- 노트북 안에서도 **각 방법마다 검증 셀이 따로 있으며, 그 방법에 맞는 검증 기준을 사용한다.**

---

# 00_foundation

목적은 이후 모든 파트에서 공통으로 사용할 실행·검증 도구를 먼저 익히는 것이다.

```text
00_foundation/
├─ tensor_memory_layout_and_ops.ipynb
├─ pytorch_core_building_blocks.ipynb
├─ pytorch_execution_basics.ipynb
└─ profiler_and_kernel_basics.ipynb
```

핵심:

- Tensor shape / layout / stride / memory
- autograd / forward / backward
- PyTorch operator graph
- profiler 사용
- ATen op → CUDA kernel 관찰
- 작은 Triton/CUDA 예제로 framework op가 실제 kernel로 내려가는 흐름 이해

이후 다른 노트북에서는 **계산 최적화가 중요한 방법에 한해** profiler 셀 또는 작은 Triton/CUDA 구현을 붙인다.

---

# 01_architecture

모델 내부에 들어가는 구조는 normalization, activation, gating 등을 포함해 Architecture로 본다.

Architecture는 크게 다음 두 그룹으로 나눈다.

```text
01_architecture/
├─ 00_common_transformer/
└─ 01_domain_and_compute_specific/
```

## 01-1. Common Transformer

여러 도메인에서 거의 그대로 재사용되는 Transformer 계열 공통 구조를 다룬다.

```text
00_common_transformer/
├─ embedding_and_projection.ipynb
├─ activation_and_gating.ipynb
├─ normalization.ipynb
├─ residual_connections.ipynb
├─ positional_encoding.ipynb
├─ attention.ipynb
├─ ffn_and_gated_mlp.ipynb
├─ moe_and_routing.ipynb
└─ transformer_block_integration.ipynb
```

각 노트북은 실제 발전 순서를 따라간다.

예:

```text
Normalization
BatchNorm → LayerNorm → RMSNorm → QK-Norm 계열

Activation / FFN
ReLU → GELU → SiLU → GLU → GEGLU/SwiGLU → 이후 발전형

Attention
Scaled Dot-Product Attention
→ Multi-Head Attention
→ MQA/GQA
→ 이후 최신 attention 구조
→ MLA/Kimi K3 계열 등
```

각 방법마다:

```text
구현
→ 해당 방법에 맞는 검증
→ 이전 방법과 달라진 구조 확인
→ 동작·효과 관찰
→ 계산 이점이 크면 profiler/kernel 관찰
→ 필요한 경우 옵션/HP/scale/조건 실험
```

## 01-2. Domain and Compute Specific

입력 topology, 데이터 구조, 연산 특성 때문에 도메인에 강하게 묶이는 구조를 둔다.

```text
01_domain_and_compute_specific/
├─ vision/
├─ generative/
├─ sequence/
├─ multimodal/
├─ 3d/
└─ robotics_vla/
```

예시:

- Vision: convolution, patch embedding, hierarchical vision, multi-scale feature
- Generative: U-Net block, DiT conditioning, latent architecture
- Sequence: recurrence, state-space model, scan 구조
- Multimodal: cross-attention, modality projector, fusion architecture
- 3D: point cloud, sparse voxel, implicit representation, Gaussian representation
- Robotics/VLA: action tokenization, temporal/action head, policy architecture

---

# 02_representation

모델의 부품이 아니라 **학습 후 어떤 feature/representation이 만들어졌는가**를 보는 파트다.

주요 대상:

- embedding
- feature geometry
- covariance / SVD / spectrum
- effective rank
- class/semantic direction
- linear probe / k-NN probe
- CKA / representation similarity
- representation drift
- 필요할 때 concept direction / intervention

이 파트도 단순 진단기법 목록이 아니라, 기본적인 representation 측정부터 최신 연구에서 사용하는 분석법으로 확장한다.

또한 Architecture / Loss / Training Recipe의 변경이 **표현을 실제로 어떻게 바꿨는지 확인하는 연결축**으로 사용한다.

---

# 03_loss_objective

무엇을 잘하도록 학습시킬지를 정의하는 축이다.

Training Recipe와의 경계는 다음처럼 둔다.

```text
Loss / Objective
= 무엇을 좋은 해라고 정의할 것인가?

Training Recipe
= 그 objective를 어떤 update 절차와 학습 조건으로 최적화할 것인가?
```

예를 들어 CE를 focal loss로 바꾸거나 auxiliary term을 추가하는 것은 `03_loss_objective`이고, 같은 objective를 AdamW / SGD, 다른 LR / batch / scheduler로 학습시키는 것은 `04_training_recipe`다.

Loss 쪽 regularization은 **objective 또는 target 자체를 바꾸는 것**을 다룬다. 예를 들어 L1/L2 penalty, label smoothing, entropy/confidence penalty, consistency objective가 여기에 들어간다. Dropout, augmentation, stochastic depth, early stopping처럼 학습 절차 쪽에서 작동하는 regularization은 `04_training_recipe`에서 다룬다. L2 penalty와 decoupled weight decay처럼 경계에 걸치는 방법은 두 파트에서 각각 objective 관점과 update-rule 관점으로 비교한다.

## 03-1. Loss 실습의 공통 레벨 구조

Loss는 architecture처럼 바로 큰 모델부터 들어가기보다, **같은 objective를 점점 더 현실적인 레벨로 올리면서 성질을 반복 검증**한다.

```text
Level A. Mathematical Object
수식 자체

        ↓

Level B. Tensor Behavior
임의 tensor / 작은 확률분포

        ↓

Level C. Small-Network Dynamics
작은 MLP/CNN/Transformer에서 실제 학습

        ↓

Level D. Reference-Model Integration
실제 논문·모델이 그 objective를 채택한 조건에서 재검증
[Architecture 실습이 충분히 진행된 뒤 추가]
```

### Level A. Mathematical Object — 수식 자체의 양상

신경망이나 고차원 tensor를 넣기 전에 loss를 하나의 수학적 함수로 본다.

예를 들어 오차 공간을 $E=\mathbb R$이라 하고 MSE를

$$
\ell_{\mathrm{MSE}}:E\to\mathbb R_{\ge 0},
\qquad
\ell_{\mathrm{MSE}}(e)=e^2
$$

로 두면, 입력 $e$를 sweep하면서 다음을 직접 계산·시각화한다.

- loss 값
- 1차 미분 / gradient
- 2차 미분 / curvature
- convexity / non-convexity
- saturation 여부
- outlier에 대한 민감도
- 작은 오차와 큰 오차를 얼마나 다르게 벌하는지
- parameter 또는 weighting 변화에 따라 함수 모양이 어떻게 바뀌는지
- 정의역의 경계나 확률 0/1 부근에서 수치적으로 어떤 문제가 생기는지

Classification loss라면 logit, probability simplex, margin을 직접 sweep하고, distribution loss라면 작은 이산분포나 1D/2D Gaussian을 사용한다.

목적은 공식 자체를 외우는 것이 아니라:

> **이 수식은 어떤 입력에 큰 학습 신호를 만들고, 어떤 입력에서는 약해지며, 어디에서 이점과 병목이 생기는가?**

를 모델 없이 먼저 보는 것이다.

### Level B. Tensor Behavior — 고차원 tensor에서의 양상

다음으로 임의의 tensor와 작은 분포를 만든다.

예:

- logits $Z\in\mathbb R^{B\times K}$
- regression prediction $\hat Y\in\mathbb R^{B\times d}$
- embedding $H\in\mathbb R^{B\times d}$
- class imbalance가 있는 synthetic batch
- noisy-label batch
- 서로 겹치거나 떨어진 Gaussian distribution

여기서는 1차원 수식에서 본 성질이 batch, class, dimension이 생겼을 때 어떻게 변하는지 본다.

주요 관찰:

- sample별 loss와 gradient norm
- class별 gradient contribution
- reduction (`mean` / `sum` / per-sample)의 영향
- 차원 수와 batch size 변화
- imbalance / noise / outlier 변화
- gradient cosine / conflict
- Hessian 또는 top eigenvalue 근사
- numerical stability
- dtype / mixed precision 영향
- PyTorch reference와 loss value / gradient parity

계산 구조가 의미 있을 때만 profiler를 붙인다. 예를 들어 fused cross entropy, log-softmax 안정화, pairwise objective의 $O(B^2)$ 비용처럼 **objective 자체가 operator graph / memory / kernel 비용을 크게 바꾸는 경우**에 본다.

### Level C. Small-Network Dynamics — 실제 학습동역학

같은 작은 모델, 데이터, optimizer, LR, batch를 고정하고 objective만 바꾸어 비교한다.

예:

- toy 2D classifier
- MNIST / FashionMNIST MLP 또는 작은 CNN
- 작은 Transformer
- 작은 metric-learning encoder

주요 질문:

- 수식에서 예상한 gradient 성질이 실제 parameter update에서도 나타나는가?
- loss surface / curvature가 어떻게 달라지는가?
- 어떤 sample / class가 학습을 지배하는가?
- margin과 calibration이 어떻게 달라지는가?
- representation rank / separation / CKA가 어떻게 달라지는가?
- train loss 개선과 validation/generalization이 같은 방향으로 움직이는가?
- 어떤 noise / imbalance / data regime에서 이점이 생기거나 사라지는가?

여기서는 `02_representation`의 측정 도구를 다시 사용해 **objective 변화 → gradient 변화 → 학습 궤적 변화 → representation 변화 → 최종 오류 변화**를 연결한다.

### Level D. Reference-Model Integration — 실제 논문·모델 적용 [후순위]

최종적으로는 해당 objective를 실제로 채택한 논문·모델의 문제 설정을 작은 크기로 보존하여 확인한다.

예:

```text
Focal Loss
→ detection 조건에서 왜 필요한지

Contrastive objective
→ 실제 representation-learning 구조에서 어떻게 쓰이는지

Continual objective
→ sequential task에서 forgetting을 어떻게 줄이는지

RL objective
→ 실제 policy/value/reward 구조 안에서 어떻게 상호작용하는지
```

단, **현재는 Architecture 파트에서 필요한 모델 구성요소를 먼저 끝내야 하므로 이 레벨을 강제하지 않는다.** 노트북 구조에는 확장 위치만 남겨두고, 이후 architecture 구현이 준비되면 실제 모델 reference와 연결한다.

## 03-2. 각 레벨 안에서 반복하는 공통 실험 구조

위 A/B/C/D 레벨마다 기존 Practice Method Standard의 원칙을 그대로 재사용한다.

```text
구현
→ 해당 레벨에 맞는 구현 검증
→ 계산 구조가 중요하면 profiler / kernel 관찰
→ 동작과 효과 관찰
→ 중요한 내부 자유도 sweep
→ 사용 조건·레짐·병목 확인
```

단, 각 레벨에서 검증 대상은 다르다.

- 수학 레벨: analytic result, finite difference, convexity/curvature, 극한값과 비교
- tensor 레벨: PyTorch/reference parity, autograd gradient parity, shape/reduction/numerical stability
- small-network 레벨: loss 감소, backward/update, seed 반복, dynamics / representation / generalization
- reference-model 레벨: 논문·공식 코드가 상정한 구조와 tensor path, loss/gradient, 실제 사용 조건

즉 Loss 파트는 **방법 하나를 한 번 구현하고 끝내는 구조가 아니라, 같은 objective를 여러 관찰 레벨에서 반복해서 찔러보는 구조**로 둔다.

## 03-3. 자유도와 분석 도구의 전체 축

아래 9개는 모두 같은 종류의 loss family가 아니다.

- 1번은 baseline이다.
- 2~3번은 뒤의 objective들을 비교·선택·개량하기 위한 **공통 분석 도구**다.
- 4~7번은 실제로 objective를 조작하는 주요 **설계 자유도**다.
- 8~9번은 앞의 자유도와 분석법을 동시에 사용하는 **종합 문제**다.

전체 누적 구조는 다음과 같다.

```text
1. 기본 loss / objective
        ↓
2. Gradient / Landscape 분석 도구
        ↓
3. Probability / Geometry / Function-Class 분석 도구
        ↓
4. Regularization / Smoothing
        ↓
5. Reweighting / Robust Loss
        ↓
6. Representation-Shaping Objective
        ↓
7. Multi-Objective
        ↓
8. Continual-Learning Objective
        ↓
9. RL Objective / Reward Design
```

중요한 점은 **2~3번을 한 번 배우고 버리는 것이 아니라 4~9번에서 계속 재사용하는 것**이다.

### 1. 기본 Loss / Objective — 기준점

먼저 가장 단순한 prediction error와 likelihood 기반 objective를 기준점으로 둔다.

Regression:

- MSE
- MAE
- Huber / Smooth L1

Classification:

- BCE
- Cross Entropy
- NLL
- softmax / log-softmax와의 관계
- likelihood / maximum likelihood와 loss의 관계

각 방법은 A → B → C 레벨로 진행하면서 다음을 본다.

- error 크기에 따른 penalty 모양
- gradient / curvature 차이
- outlier 민감도
- probability 0/1 부근의 양상
- regression noise distribution 가정과의 관계
- classification confidence / margin과의 관계

이 단계가 뒤의 모든 objective 비교의 baseline이다.

### 2. Gradient / Landscape — 공통 분석 도구 I

목적은 loss를 바꾸었을 때 **parameter가 받는 학습 신호와 local loss geometry가 어떻게 달라지는지 읽는 법**을 먼저 익히는 것이다.

주요 도구:

- sample별 loss
- sample / class / task별 gradient norm
- gradient cosine similarity
- gradient conflict rate
- gradient variance
- update-to-weight ratio
- Hessian top eigenvalue
- Hessian spectrum의 작은 근사
- local curvature
- sharpness
- 방향별 perturbation
- loss interpolation / 1D·2D slice
- margin
- calibration

사용 목적:

```text
loss 변경
→ gradient field가 어떻게 바뀌었는가?
→ local curvature가 어떻게 바뀌었는가?
→ 실제 update trajectory가 어떻게 달라졌는가?
```

2D landscape 그림 하나로 결론내리지 않는다. Gradient → Hessian/curvature → perturbation → 실제 trajectory를 가능한 범위에서 연결한다.

### 3. Probability / Geometry / Function-Class — 공통 분석 도구 II

2번보다 더 근본적이거나 계산·해석 비용이 큰 도구를 둔다. 목적은 수학을 따로 자랑하기 위한 것이 아니라:

> **어떤 objective가 무엇을 거리라고 보고, 어떤 분포 차이를 줄이며, 어떤 함수족에서 generalization을 기대할 수 있는지를 보고 loss를 선택하거나 개량할 근거를 얻는 것**

이다.

#### 3-1. Metric / Divergence / Distribution Geometry

작은 이산분포와 1D/2D Gaussian에서 직접 계산하고 시각화한다.

- KL divergence
- reverse KL
- JS divergence
- MMD
- Wasserstein distance
- Sinkhorn divergence
- cosine / Euclidean / Mahalanobis distance
- margin
- 필요할 때 Fisher information / Fisher geometry

관찰할 것:

- 두 분포가 떨어졌을 때 값과 gradient가 어떻게 변하는가?
- support가 겹치지 않을 때 어떤 방법이 신호를 잃는가?
- mode-covering / mode-seeking 성질이 나타나는가?
- metric과 divergence의 차이가 실제 optimization trajectory에 어떤 차이를 만드는가?
- dimension이 증가할 때 거리의 분별력이 어떻게 변하는가?

예를 들어 두 Gaussian의 위치·분산·겹침을 바꾸면서 KL / MMD / Wasserstein을 objective로 사용하고, 값뿐 아니라 gradient trajectory를 함께 비교한다.

#### 3-2. Empirical Risk / Function-Class Complexity / Generalization

함수족을

$$
\mathcal F=\{f_\theta:\mathcal X\to\mathcal Y\mid\theta\in\Theta\}
$$

라고 하고, loss $\ell$에 대한 empirical risk와 population risk를

$$
\widehat R_n(f)
=
\frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i),y_i),
$$

$$
R(f)
=
\mathbb E_{(X,Y)\sim P}[\ell(f(X),Y)]
$$

로 둔다.

실습 목적은 theorem 증명을 길게 하는 것이 아니라 **train loss와 실제 generalization 사이의 gap이 어떤 조건에서 커지는지 계산과 그래프로 확인**하는 것이다.

볼 수 있는 것:

- sample 수 증가에 따른 train/validation gap
- model width / norm constraint 변화
- margin 변화
- weight norm 또는 간단한 complexity proxy
- Rademacher complexity의 작은 toy 계산 또는 근사
- 동일 empirical risk를 갖는 해들의 validation 차이

이 파트의 도구는 뒤의 regularization, smoothing, robust loss를 왜 쓰는지 판단하는 근거로 재사용한다.

### 4. Regularization / Smoothing — 원래 목표에 좋은 해의 성질을 추가

핵심 질문:

> 정답을 맞추는 것만으로 부족할 때, objective 또는 target에 어떤 제약을 추가할 것인가?

주요 자유도:

- L1 penalty
- L2 penalty
- label smoothing
- entropy regularization
- confidence penalty
- consistency regularization
- 필요할 때 margin / norm regularization

A. 수학 레벨:

- 원래 loss에 penalty를 더했을 때 minimum과 curvature가 어떻게 움직이는지
- smoothing strength가 target distribution과 gradient를 어떻게 바꾸는지
- entropy / confidence term이 probability simplex에서 어떤 힘을 만드는지

B. tensor 레벨:

- sample/class별 gradient 변화
- logit norm / confidence 변화
- calibration 변화
- penalty weight에 따른 gradient budget

C. small-network 레벨:

- train/validation gap
- margin
- calibration
- Hessian / sharpness
- representation rank / separation

을 비교한다.

### 5. Reweighting / Robust Loss — 누구의 오류를 더 중요하게 볼 것인가

모든 sample을 동일하게 취급하지 않는 자유도다.

주요 방법:

- class weighting
- sample weighting
- focal loss
- hard-example weighting
- class-balanced loss 계열
- noisy-label robust loss
- clipping / bounded influence 계열

핵심 질문:

- easy / hard sample 중 누구에게 gradient를 더 주는가?
- majority / minority class의 gradient budget이 어떻게 달라지는가?
- outlier와 label noise가 전체 update를 얼마나 지배하는가?
- weight 또는 focal $\gamma$를 바꿀 때 병목이 어디로 이동하는가?

수학 레벨에서 probability/error를 sweep하고, tensor 레벨에서 synthetic imbalance/noise를 만들고, small-network 레벨에서 class별 recall / margin / calibration / gradient contribution을 연결한다.

### 6. Representation-Shaping Objective — 어떤 feature geometry를 만들 것인가

출력 정답만 맞추는 것을 넘어 **중간 representation의 구조 자체**를 objective로 설계한다.

주요 방법:

- contrastive objective
- pair / triplet objective
- margin-based metric learning
- InfoNCE 계열
- distillation objective
- feature matching
- collapse-prevention / variance-covariance regularization 계열

주요 질문:

- positive pair를 얼마나 당기고 negative pair를 얼마나 미는가?
- distance / similarity 선택이 representation geometry를 어떻게 바꾸는가?
- temperature / margin / negative 수가 gradient를 어떻게 바꾸는가?
- collapse가 언제 발생하는가?
- class 내 응집도와 class 간 분리도가 어떻게 달라지는가?

여기서는 3번의 metric / divergence와 `02_representation`의 spectrum, effective rank, CKA, probe를 적극적으로 재사용한다.

### 7. Multi-Objective — 여러 목표를 어떻게 동시에 최적화할 것인가

여러 loss가 동시에 존재하는 경우를 다룬다.

예를 들어 parameter space를 $\Theta$라 하고 두 objective를

$$
L_1,L_2:\Theta\to\mathbb R
$$

라고 하면 가장 단순한 결합은

$$
L(\theta)=L_1(\theta)+\lambda L_2(\theta)
$$

이다.

그러나 $\lambda$만 정한다고 문제가 끝나는 것은 아니다.

주요 자유도:

- auxiliary loss
- weighted sum
- task weighting
- dynamic weighting
- gradient normalization
- gradient conflict handling
- Pareto trade-off

주요 관찰:

- 각 loss 항의 scale
- 각 항의 gradient norm
- gradient cosine / conflict
- 한 objective 개선이 다른 objective를 악화시키는 구간
- $\lambda$ sweep과 Pareto front
- objective별 representation 변화

이 단계부터 앞의 gradient / geometry 분석을 동시에 사용한다.

### 8. Continual-Learning Objective — 새것을 배우면서 옛것을 어떻게 지킬 것인가

Continual learning은 단순한 새 loss 하나가 아니라 앞의 objective 설계를 종합하는 문제로 본다.

핵심 trade-off:

```text
plasticity
새 task를 잘 배우기

vs

stability
이전 task를 보존하기
```

주요 방법 / 자유도:

- replay + task loss
- distillation
- parameter regularization
- EWC / Fisher 기반 penalty
- representation-preservation objective
- gradient projection / conflict reduction
- task별 weighting

주요 관찰:

- forgetting matrix
- old/new task별 loss와 metric
- task별 gradient conflict
- parameter drift
- representation drift / CKA
- Fisher / parameter importance
- replay ratio / penalty strength

즉 4번 regularization, 6번 representation objective, 7번 multi-objective와 2~3번 분석 도구가 한꺼번에 재사용되는 **종합 실습**으로 둔다.

### 9. RL Objective / Reward Design — objective가 데이터분포까지 바꾸는 종합 문제

RL은 loss라는 이름만으로 묶기보다 **무엇을 보상하고 어떤 정책 변화를 허용할 것인가를 설계하는 objective engineering**으로 본다.

주요 자유도:

- reward definition
- sparse / dense reward
- reward shaping
- return / discount
- policy objective
- value objective
- advantage
- entropy bonus
- KL constraint / trust region
- actor-critic objective balance
- offline RL의 conservative objective
- reward scale / normalization

주요 관찰:

- policy-gradient norm / variance
- actor-critic gradient alignment
- entropy와 exploration
- policy KL
- Fisher geometry
- state-action occupancy distribution
- reward hacking / shortcut
- reward scale에 따른 dynamics
- horizon에 따른 credit assignment

Supervised learning과 달리 policy가 바뀌면 이후에 수집되는 상태·행동 분포 자체가 바뀐다. 따라서 앞에서 배운 **gradient, multi-objective, divergence, Fisher geometry, distribution shift**가 동시에 필요해지는 최종 종합 문제로 둔다.

## 03-4. 4~9번에서 2~3번 도구를 반복 적용하는 방식

예를 들어 Focal Loss를 다룬다면 단순히 CE보다 accuracy가 높은지 보는 것으로 끝내지 않는다.

```text
[수학]
CE와 Focal의 probability별 loss / gradient / curvature 비교

        ↓

[tensor]
synthetic imbalance batch에서
sample/class별 gradient budget 비교

        ↓

[small network]
동일 모델에서 CE ↔ Focal만 교체
→ minority recall
→ margin / calibration
→ Hessian / landscape
→ representation 변화

        ↓

[reference model — later]
detection architecture가 준비되면
실제 detection 조건에서 다시 검증
```

Contrastive objective라면:

```text
distance / similarity 함수 자체
→ pairwise tensor geometry
→ small encoder representation
→ spectrum / rank / probe / CKA
→ 실제 SSL model [later]
```

Multi-objective라면:

```text
두 scalar objective의 기하
→ tensor gradient conflict
→ small multitask network
→ Pareto / representation / task trade-off
→ 실제 multitask model [later]
```

이 구조를 통해 최종적으로 얻고 싶은 능력은:

> **loss 이름을 많이 아는 것이 아니라, 새로운 objective를 보았을 때 수식 자체의 성질 → tensor에서의 gradient 구조 → 실제 학습동역학 → 표현과 generalization → 사용 조건과 병목을 순서대로 검증하고, 현재 문제에 맞는 objective를 선택하거나 개량할 수 있는 것**

이다.

## 03-5. 생성모델 objective와의 경계

Diffusion / Flow의 $\epsilon$-prediction, $x_0$-prediction, $v$-prediction, velocity matching, solver / NFE와 같이 **생성 dynamics 자체와 강하게 묶인 objective**는 `05_inference_in_generation_and_control`의 Diffusion / Flow 발전사에서 구체적으로 다룬다.

`03_loss_objective`에서는 그것들을 별도 도메인 목록으로 중복 구현하지 않고, 다음과 같은 **범용 objective 설계 원리**만 가져온다.

- weighting
- auxiliary objective
- consistency
- representation alignment
- multi-objective interaction
- distribution metric / divergence 선택

즉 `03`은 objective를 분석하고 설계하는 공통 언어를 만들고, `05`는 생성 dynamics라는 실제 도메인에서 그 objective가 어떻게 쓰이는지를 다룬다.

---

# 04_training_recipe

Training Recipe는 단일 기법 모음이 아니라 실제 학습에서 **여러 요소를 조합해서 하나의 동역학 레짐을 만드는 축**으로 본다.

따라서 **개별 기법 실습 + 통합 레짐 실습**의 2층 구조로 둔다.

```text
04_training_recipe/
├─ 01_optimizer/
├─ 02_lr_and_scheduler/
├─ 03_initialization/
├─ 04_batch_and_accumulation/
├─ 05_regularization/
├─ 06_gradient_control/
├─ 07_ema_and_checkpoint/
├─ 08_precision_and_numerics/
├─ 09_hyperparameters/
├─ 10_scaling/
└─ 90_integrated_recipe/
```

## 04-1. 개별 기법

각 분야는 교과서/PyTorch 기본에서 실제 최신 recipe로 발전하는 순서로 본다.

예: optimizer

```text
SGD
→ Momentum
→ Nesterov
→ AdaGrad/RMSProp
→ Adam/AdamW
→ 이후 optimizer
→ Muon 등 최신 계열
```

각 optimizer마다:

- update rule 직접 구현
- PyTorch 지원형은 `torch.optim`과 parameter update/state parity
- 논문형은 reference optimizer와 update parity
- 동일 작은 모델에서 backprop/update dynamics 관찰
- 해당 optimizer에서 중요한 옵션이 있으면 그 안에서 옵션 비교
- 필요하면 optimizer step profiler

Scheduler, initialization, regularization 등도 같은 원칙으로 진행하되 관찰 대상은 내용에 맞게 바꾼다.

## 04-2. Integrated Recipe / Dynamics Regime

Training Recipe의 최종 목적은 개별 기법을 나열하는 것이 아니라 조합에 의해 만들어지는 **training dynamics regime**을 보는 것이다.

```text
90_integrated_recipe/
├─ optimizer_x_lr.ipynb
├─ batch_x_lr.ipynb
├─ weight_decay_x_lr.ipynb
├─ init_x_depth_width.ipynb
├─ clipping_x_optimizer.ipynb
├─ precision_x_scale.ipynb
├─ scaling_recipe_transfer.ipynb
└─ full_recipe_regimes.ipynb
```

궁극적으로 보고 싶은 관계:

$$
(\text{model scale},\text{batch},\text{LR},\text{optimizer},\ldots)
\rightarrow
\text{training dynamics regime}
\rightarrow
\text{stable / unstable / efficient training}
$$

여기서는 필요에 따라:

- loss trajectory
- gradient norm / direction
- update-to-weight ratio
- activation scale
- curvature/Hessian 계열

등을 사용하지만 모든 노트북에 강제하지 않는다.

목적은 **HP를 맞추는 법 자체보다, HP와 scale이 왜 다른 동역학 레짐을 만드는지 이해하고 recipe를 다른 규모에 이전하는 법을 익히는 것**이다.

---

# 05_inference_in_generation_and_control

일반적인 classification/identification과 달리, 생성과 제어는 한 번의 정답 출력으로 끝나지 않는다.

큰 흐름은 다음과 같이 잡는다.

```text
식별
→ 생성
→ 제어
```

- 식별: 주어진 입력에서 정답/상태 추정
- 생성: 가능한 출력의 분포에서 실제 결과를 구성
- 제어: 생성한 action/trajectory가 다시 다음 환경 상태를 바꾸므로 미래를 고려해 연속적으로 결정

현재 생성 모델도 이미지 → 비디오 → world model → action-conditioned world model 방향으로 가면서 생성과 제어의 경계가 가까워지는 흐름을 함께 본다.

```text
05_inference_in_generation_and_control/
├─ 00_common/
├─ 01_generation/
│  ├─ llm_ar/
│  └─ diffusion_flow/
└─ 02_control/
   └─ rl_vla/
```

## 05-1. Common

```text
00_common/
├─ identification_to_generation_to_control.ipynb
├─ sampling_search_and_rollout.ipynb
├─ conditioning_and_conditional_generation_control.ipynb
├─ guidance_constraint_and_selection.ipynb
└─ ar_vs_flow_mathematical_structure.ipynb
```

공통에서 다룰 핵심:

- sampling
- search / selection
- rollout / trajectory
- conditioning
- conditional generation / conditional control
- guidance / constraint
- AR과 Flow의 수학적 생성 구조 비교

AR의 기본 구조:

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

Flow 계열의 기본 구조:

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c)
$$

즉 AR은 순차적 조건부 확률분해, Flow는 연속 동역학으로 분포를 운반하는 구조로 비교한다.

## 05-2. LLM / AR

먼저 기본 생성 제어요소를 직접 다룬다.

- greedy
- ancestral sampling
- temperature
- top-k
- top-p
- beam/search 계열

그다음 AR의 근본적인 **serial token generation 병목을 완화하는 발전사**를 본다.

```text
1-token serial decoding
→ speculative decoding
→ multi-token / tree prediction
→ partial parallel / iterative decoding
```

예:

- speculative decoding
- draft-and-verify
- Medusa류 multi-token prediction
- EAGLE류 tree/draft verification
- lookahead / blockwise / iterative decoding 계열

각 방법에서 구현·reference 검증·실제 token generation 동작·latency/throughput 병목을 필요에 따라 본다.

## 05-3. Diffusion / Flow

기본 구성요소부터 시작한다.

- noise level
- timestep
- scheduler
- guidance / CFG
- conditioning
- solver / integration step
- NFE

이후 생성 dynamics 자체의 발전 흐름을 구현한다.

```text
Diffusion
→ Flow Matching
→ Rectified Flow
→ Mean Flow
→ 이후 관련 최신 방법
```

각 방법은:

- objective / vector field 또는 score 구조 구현
- 해당 reference로 검증
- sampling trajectory/solver 동작 확인
- step 수, straightness, guidance, conditioning 등 중요한 자유도 실험
- NFE/latency/quality trade-off와 profiler를 필요한 경우 확인

## 05-4. RL / VLA Control

기본 제어 구성요소:

- policy distribution
- action sampling
- entropy / temperature
- horizon
- rollout
- action chunk
- goal conditioning
- trajectory selection

이후 policy/control 알고리즘의 발전사를 본다.

예:

```text
REINFORCE
→ actor-critic 계열
→ PPO
→ 이후 policy optimization
→ GRPO류
→ model-based rollout
→ world-model planning/control
```

VLA/robotics에서는 direct action policy에서 action chunk/trajectory policy, world-model rollout과 planning으로 확장되는 흐름을 연결한다.

---

# 실습 철학

이 실습의 목적은 **논문 구현 능력을 기르고, 여러 방법 중 현재 상황에 적합한 방법을 선택한 이유를 설명할 수 있으며, 선택한 방법을 상황에 맞게 커스터마이징하여 실제로 활용할 수 있는 능력을 기르는 것**이다.
