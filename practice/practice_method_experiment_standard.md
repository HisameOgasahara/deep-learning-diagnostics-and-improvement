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

예:

```text
기본 regression/classification objective
→ CE/MSE
→ regularized / reweighted loss
→ focal / label smoothing
→ contrastive / metric objective
→ distillation
→ diffusion/flow objective
→ RL/policy objective
→ 최신 논문 objective
```

각 방법에서는:

- 직접 구현
- PyTorch 지원형은 loss value + gradient parity
- 논문형은 reference implementation과 loss/gradient parity
- 같은 모델에서 objective만 바꾸어 backward gradient와 학습 동작 차이 관찰
- loss 내부 HP/weight가 핵심이면 옵션별 실험
- 어떤 데이터/불균형/노이즈/scale에서 유리하거나 불리한지 확인

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

# 전체 실습 철학

각 파트는 방법을 단순히 많이 구현하는 것이 목적이 아니다.

최종적으로 각 방법에 대해 다음 질문에 답할 수 있어야 한다.

```text
이전 방법과 무엇이 달라졌는가?
왜 그 변화가 필요했는가?
실제로 내부 동작은 어떻게 달라지는가?
공식/reference 구현과 동일하게 구현됐는가?
계산 구조상 이점 또는 병목은 무엇인가?
어떤 조건에서 효과가 커지는가?
어떤 조건에서는 쓰지 않는 것이 좋은가?
어떤 옵션/HP/scale에 민감한가?
다른 방법과 조합하면 어떤 레짐이 생기는가?
내 환경에 가져오려면 무엇을 보존하고 무엇을 바꿔야 하는가?
더 개선할 여지는 어디에 있는가?
```

즉 전체 목표는

```text
기초 구현
→ 검증
→ 발전사 추적
→ 내부 동작 이해
→ 계산 구조 이해
→ 조건/레짐 탐색
→ 실전 커스터마이징
```

으로 이어지는 T4 기반 반복·확장 실습 체계를 만드는 것이다.
