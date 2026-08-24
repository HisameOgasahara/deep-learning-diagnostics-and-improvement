# 딥러닝 학습 개선을 위한 계층형 문제진단 프레임워크 --- 확장판

## 0. 전체상

목표는 모든 분석법을 실행하는 것이 아니라, **값싼 관찰로 문제를 좁힌 뒤
필요한 가지로만 깊게 들어가고, 진단을 실제 개선과 인과 검증까지 연결하는
것**이다.

``` mermaid
flowchart TD
 A["성능이 기대보다 낮다"] --> B["1. Localization"]
 B --> X["입력·데이터 의존"]
 B --> R["2. Representation Diagnosis"]
 B --> D["3. Learning Dynamics"]
 X --> XA["XAI · Perturbation · Counterfactual"]
 R --> G["Global Geometry<br/>Spectrum · Rank · Margin · Probe · CKA"]
 G --> M["Local / Manifold<br/>Neighborhood · ID · Local PCA · Tangent"]
 G --> P["Distribution Geometry<br/>MMD · Wasserstein/Sinkhorn · Support · Coverage"]
 G --> D
 M --> D
 P --> D
 D --> L["Gradient/Update → Representation Dynamics → Regime"]
 L --> H["NTK / Hessian / Fisher / Jacobian"]
 H --> LL["필요할 때 Loss-Landscape<br/>Perturbation → Restart → Interpolation → Connectivity"]
 XA --> S["4. Steer / Improve"]
 R --> S
 D --> S
 LL --> S
 S --> ABL["Low-Fidelity Screening → Controlled Ablation"]
 ABL --> C["5. Causal Validation"]
 C --> MI["MI · Ablation · Direction Removal · Patching"]
 MI --> N["새 Baseline"]
 N --> B
```

> **1. 어디서 못하는가? → 2. 무엇을 잘못 표현했는가? → 3. 왜 그렇게
> 학습되었는가? → 4. 어디를 고칠 것인가? → 5. 정말 그것이
> 원인이었는가?**

2와 3은 고정 순서가 아니다. 일반적인 성능 문제에서는 표현 상태를 먼저
보고 필요할 때 동역학으로 추적한다. loss 발산·진동·NaN처럼 학습 자체가
비정상이라면 3번으로 바로 들어간다.

------------------------------------------------------------------------

# 1. 현상 Localization --- 어디서 못하는가?

baseline과 평가 조건을 고정하고 먼저 본다.

-   train/validation loss와 task metric
-   class·domain·scale·난이도별 slice
-   FP/FN과 대표 failure sample
-   seed별 변동, confidence와 calibration
-   데이터 양·step·모델 크기·해상도 증가에 대한 반응
-   복합 pipeline의 component oracle gap

``` text
특정 class/domain/scale만 나쁨 → 데이터·표현·분포
train 자체가 안 좋음 → 학습동역학
train은 좋은데 validation만 나쁨 → 일반화·분포 차이·shortcut
입력 조건 변화에 급격히 무너짐 → 잘못된 입력 의존성
```

XAI는 초기 localization에 사용한다.

``` text
failure sample
→ occlusion / perturbation
→ counterfactual
→ attribution / Grad-CAM
→ 필요하면 concept-level 분석
```

------------------------------------------------------------------------

# 2. Representation Diagnosis --- 무엇을 어떻게 표현했는가?

## 2.1 Global Representation Geometry

### Spectrum / Effective Rank

activation covariance·SVD spectrum과 effective rank로 collapse, 불필요한
방향, layer별 차원 변화를 본다.

### 분리 구조

class 내·간 거리, margin, centroid, neighborhood purity로 **정답을
구별하기 좋은 구조인가**를 본다.

### Probe

``` text
probe도 낮음 → representation 자체의 정보 부족
probe는 높지만 출력은 낮음 → head/readout/후속 경로 문제 후보
```

### CKA

baseline↔개선 모델, pretrained↔fine-tuned, checkpoint↔checkpoint를
비교해 **변화가 시작되는 layer를 localization**한다.

## 2.2 Local / Neural-Manifold Geometry

Global geometry로 설명되지 않는 국소 실패가 있을 때만 내려간다.

``` text
k-NN neighborhood
→ local covariance / local PCA
→ intrinsic dimension 또는 LID
→ tangent / local subspace
→ 필요할 때 manifold radius · dimension · capacity · center correlation
```

여기서 manifold는 엄밀한 리만다양체를 먼저 가정한다는 뜻이 아니라
**고차원 activation 점구름의 국소 저차원 구조**를 뜻한다. explicit
geodesic, curvature, TDA는 이 구조 자체가 연구 질문일 때만 추가한다.

## 2.3 Distribution Geometry --- 분포 전체가 어떻게 다른가?

### 진입 조건

-   source↔target, train↔validation domain gap
-   OOD에서 domain 전체가 무너짐
-   augmentation/synthetic data가 support를 넓혔는지 확인
-   생성모델의 mode/coverage 손실
-   class geometry는 괜찮은데 domain별 성능 차이가 지속

표현 사상을 $z_\theta:\mathcal X\to\mathcal Z$, 입력분포를 $P_X$라 하면
표현분포는 pushforward $(z_\theta)_\#P_X$이다. 목적은 평균 feature가
아니라 **sample 집단 전체의 분포 차이**를 보는 것이다.

### 비용 순서

``` text
slice별 통계 / centroid / covariance
→ nearest-neighbor · support overlap
→ MMD
→ Wasserstein / Sinkhorn
→ density / coverage · class-conditional distribution
```

**MMD**: kernel feature space에서 두 sample 집단의 분포 차이를 본다.
source-target gap이 layer를 지나며 줄어드는지, augmentation이 target
쪽으로 이동시키는지 확인한다.

**Wasserstein / Sinkhorn**: 분포 질량을 옮기는 비용을 본다. 더 비싸므로
분포 이동 구조 자체가 질문일 때 사용한다.

**Support / Density / Coverage**: 특히 생성모델·synthetic data·OOD에서
quality와 coverage를 분리한다.

``` text
전체 distance↓ + class-conditional distance↓ + target metric↑
→ 분포 정렬 가설 강화

전체 distance↓ + 특정 class 악화
→ 평균적 정렬이 중요한 구조를 지웠을 가능성

coverage↑ + quality↓
→ support 확장과 품질 trade-off
```

분포 metric은 인과적 증거가 아니다. 이후 data-mixture ablation, source
addition/removal, counterfactual로 연결한다.

------------------------------------------------------------------------

# 3. Learning Dynamics --- 왜 그렇게 학습되었는가?

## 3.1 값싼 기본 동역학

-   gradient norm / layerwise gradient norm
-   gradient cosine
-   update norm / update-to-weight ratio
-   activation 평균·분산·sparsity
-   clipping·overflow

``` text
gradient≈0 → gradient flow / saturation
gradient는 있으나 update≈0 → optimizer scaling / clipping / LR
특정 layer에 몰림 → layerwise imbalance
activation 붕괴 → 표현 붕괴의 생성 원인 후보
gradient 방향 반전 반복 → 진동·고곡률·큰 LR 후보
```

## 3.2 Representation Dynamics

checkpoint별 effective rank, spectrum, margin, probe, CKA/representation
drift를 추적한다. 필요하면 distribution gap도 시간축으로 본다.

``` text
학습 초반 정상
→ 특정 시점 rank 감소
→ margin 붕괴
→ validation 정체
```

처럼 최종 상태를 시간 순서의 가설로 바꾼다.

## 3.3 Regime Diagnosis

**Lazy/kernel-like**: parameter가 움직여도 feature가 거의 변하지
않는다.\
**Rich/feature-learning**: 학습 중 representation 자체가 재구성된다.

조사 후보: - feature drift - parameter displacement - feature/gradient
correlation - 관련 spectrum - empirical NTK와 kernel alignment 변화

NTK는 기본 모니터링이 아니라 **feature-learning regime 자체가 질문일
때** 사용한다.

## 3.4 Curvature / Jacobian

1차 동역학으로 설명되지 않을 때만 올라간다.

**Hessian/Fisher**: plateau/saddle 후보, 고곡률 방향, optimizer/LR
안정성, gradient-곡률 정렬.\
**Jacobian**: 입력 방향이 표현·출력으로 어떻게 전달되고 증폭·감쇠되는지,
robustness와 gradient flow를 본다.

``` text
gradient/update
→ representation dynamics
→ regime
→ NTK / Hessian / Fisher / Jacobian
```

## 3.5 Loss-Landscape 정밀분석

### 진입 조건

-   plateau 장기 정체
-   작은 gradient에서 saddle/minimum 구분 필요
-   optimizer의 saddle 탈출 주장 검증
-   seed/checkpoint가 같은 basin인지 조사
-   weight averaging/model merging 판단
-   Hessian만으로 주변 지형 설명이 부족

### ① Directional Perturbation

checkpoint $\theta_*$와 정규화 방향 $v$에 대해

$$
\varepsilon\mapsto L(\theta_*+\varepsilon v)
$$

를 측정한다.

방향: random, gradient, optimizer update, Hessian 최대/최소 고유벡터,
최근 trajectory PCA.

### ② Restart / Escape

같은 checkpoint에서 LR, batch size, momentum/optimizer, 작은 parameter
noise, seed, 일부 layer 재초기화를 하나씩 바꾼다.

``` text
작은 perturbation에도 반복 탈출 → 안정된 local minimum 가능성 약화
큰 noise가 있어야 이동 → barrier가 있는 basin 후보
train loss만 개선·validation 악화 → 더 낮은 train minimum ≠ 더 좋은 함수
```

### ③ Checkpoint Interpolation

$$
\theta(\alpha)=(1-\alpha)\theta_A+\alpha\theta_B,\qquad \alpha\in[0,1].
$$

Barrier를

$$
B(\theta_A,\theta_B)
=
\max_{\alpha}L(\theta(\alpha))
-
\max\{L(\theta_A),L(\theta_B)\}
$$

로 요약할 수 있다.

### ④ Mode Connectivity / Basin

직선에 barrier가 있어도 분리된 basin이라고 단정하지 않는다. 곡선
low-loss path를 찾고 neuron permutation alignment, BatchNorm 통계,
train/validation loss, output disagreement를 함께 통제한다.

``` text
Hessian
→ directional perturbation
→ restart / escape
→ interpolation
→ mode connectivity / basin
```

------------------------------------------------------------------------

# 4. Steer / Improve --- 어디를 어떻게 고칠 것인가?

방법을 먼저 고르지 않고 **진단된 병목에 직접 대응하는 방법만** 후보로
올린다.

## 입력·데이터·분포

-   shortcut → counterfactual augmentation
-   domain support 부족 → 데이터 보강·mixture 조정
-   distribution gap → source/target sampling·alignment
-   coverage 부족 → rare mode 보강·synthetic data 재설계
-   label noise → label 정제

## 표현

-   정보 없음 → backbone / architecture / objective / pretraining
-   분리 구조 불량 → representation objective / loss / margin
-   probe는 좋고 head 실패 → head/readout
-   representation drift → fine-tuning 범위 / adapter / normalization
-   local manifold 붕괴 → 해당 slice용 data/objective/invariance

## 학습동역학·Landscape

-   gradient flow 실패 → initialization / residual / normalization
-   update 부족 → LR / optimizer / parameterization
-   고곡률 진동 → LR / optimizer / clipping / schedule
-   feature learning 부족 → scaling / LR / unfreeze
-   saddle 탈출 실패 → optimizer/LR/noise matched comparison
-   basin 연결성이 핵심 → averaging/merging 전 connectivity 확인

Low-Fidelity Screening은 short run, 축소 데이터, 핵심 slice, 작은 HP
sweep, 특정 stage, 1\~2 seed로 방향성만 확인하고 통과한 후보만
full-budget 실험으로 올린다.

------------------------------------------------------------------------

# 5. Causal Validation --- 정말 그것이 원인이었는가?

``` text
관찰적
CKA / spectrum / rank / distribution distance / attribution
↓
판별적
probe / perturbation / counterfactual / directional test
↓
개입적
component ablation / direction removal /
activation replacement / patching / causal tracing
```

XAI는 초기 localization, MI는 좁혀진 내부 요소의 **기능적 필요성**
검증에 사용한다.

최소 통제: - Baseline - Baseline + 개선법 - 핵심 요소 제거 -
parameter/FLOPs matched control - strength sweep - 여러 seed - 전체
metric + 목표 slice - 학습·추론 비용

------------------------------------------------------------------------

# 6. 상황별 진입점

``` text
A. Loss 발산/NaN
Localization → gradient/update/activation → 필요시 Hessian/Jacobian
→ 필요시 landscape → steer → 검증

B. Train 좋고 validation 나쁨
split/leakage/slice → XAI/counterfactual
→ representation → 필요시 distribution geometry
→ representation dynamics → steer → 검증

C. 특정 class/domain 실패
slice localization → margin/probe/neighborhood
→ local manifold 또는 distribution geometry
→ 필요시 dynamics → steer → intervention

D. 새 optimizer/scaling 검증
matched baseline → gradient/update → representation dynamics → regime
→ NTK/Hessian → 필요시 perturbation/restart → matched ablation

E. Fine-tuning 후 능력 소실
regression slice → probe+CKA → checkpoint drift
→ steer → activation intervention

F. Domain/OOD/생성 coverage 문제
slice → support/NN → MMD
→ 필요시 Wasserstein/Sinkhorn · density/coverage
→ data/mixture steer → ablation

G. Plateau/saddle/basin 연구
trajectory → Hessian → directional perturbation
→ restart → interpolation → mode connectivity
```

------------------------------------------------------------------------

# 7. 최종 원칙

``` text
넓고 싼 관찰
→ 문제 위치를 좁힘
→ 목적에 맞는 분석 가지 선택
→ 그 가지 안에서도 싼 것부터
→ 개입 위치 결정
→ 작은 screening
→ 정식 ablation
→ 필요할 때만 MI
```

이번 확장에서 추가된 두 가지는 독립된 최상위 단계가 아니다.

-   **Distribution Geometry**는 Representation Diagnosis 안에서, 개별
    feature가 아니라 **sample 집단 전체의 분포 구조**가 문제일 때
    진입한다.
-   **Loss-Landscape 정밀분석**은 Learning Dynamics의 Hessian 뒤에서,
    **정체·탈출·basin 자체가 질문일 때만** 진입한다.

따라서 프레임워크의 중심 구조는 그대로 유지하면서,
domain/OOD/generative와 optimizer/loss-landscape 연구에서 빠져 있던 정밀
branch만 보강한다.
