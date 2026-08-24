# 딥러닝 수학적 검증 기법 정리

## 0. 목적

이 문서는 딥러닝 연구에서 성능 향상을 단순 점수로만 보고 끝내지 않고,

- 무엇이 바뀌었는지
- 학습 과정이 어떻게 달라졌는지
- 표현과 함수가 어떻게 달라졌는지
- 데이터와 분포의 구조가 어떻게 달라졌는지
- 관찰된 차이가 실제 계산에 필요한지

를 수학적·실험적으로 검증하기 위한 기법들을 정리한다.

---

# 1. 수학적 검증 기법의 7개 축

## 1.1 학습 동역학

모델 파라미터가 학습 중 어떤 방향과 크기로 움직이는지 본다.

주요 기법:

- gradient norm
- layerwise gradient norm
- gradient cosine similarity
- gradient conflict rate
- gradient variance
- gradient signal-to-noise ratio
- update-to-weight ratio
- parameter displacement
- checkpoint 간 parameter trajectory
- optimizer step statistics

두 손실의 gradient를 비교할 필요가 있을 때 다음과 같이 둔다.

$$
g_1(\theta)=\nabla_\theta L_1(\theta),
\qquad
g_2(\theta)=\nabla_\theta L_2(\theta).
$$

두 gradient의 방향 정렬 정도는

$$
\operatorname{cos}(g_1,g_2)
=
\frac{\langle g_1,g_2\rangle}
{\|g_1\|\,\|g_2\|}
$$

로 측정할 수 있다.

해석:

- 양수: 두 목적이 대체로 협력한다.
- 0 부근: 서로 거의 독립적인 방향이다.
- 음수: 한 목적을 줄이는 update가 다른 목적을 악화시킬 가능성이 있다.

주요 활용 분야:

- optimizer 설계
- loss 설계
- multitask learning
- multimodal learning
- continual learning
- RL policy optimization
- initialization과 scaling

---

## 1.2 스펙트럼 분석

행렬이나 선형 연산자의 고유값·특이값 분포를 통해 학습과 표현의 구조를 본다.

주요 대상:

- Hessian
- Fisher information matrix
- input-output Jacobian
- activation covariance
- Gram matrix
- empirical NTK
- weight matrix
- feature matrix

주요 기법:

- 최대 고유값
- trace
- spectral density
- singular-value spectrum
- condition number
- effective rank
- low-rank structure
- eigenspace overlap

주요 질문:

- loss surface의 곡률은 얼마나 큰가?
- 특정 방향만 학습을 지배하는가?
- gradient가 일부 고곡률 방향과 정렬되는가?
- 표현이 소수 차원으로 붕괴했는가?
- Jacobian이 입력 신호와 gradient를 안정적으로 전달하는가?
- weight 또는 activation이 저랭크 구조를 갖는가?

주요 활용 분야:

- optimizer
- initialization
- scaling
- normalization
- architecture
- compression
- pruning
- robustness
- representation learning

---

## 1.3 표현 비교

서로 다른 모델·레이어·학습 시점의 표현이 얼마나 유사한지 본다.

주요 기법:

- CKA
- RSA
- SVCCA
- PWCCA
- Procrustes analysis
- subspace overlap
- principal-angle analysis

주요 질문:

- 두 모델이 비슷한 내부 표현을 학습했는가?
- 특정 변경의 영향이 어느 레이어부터 나타나는가?
- fine-tuning 이후 초기 표현이 얼마나 유지되었는가?
- teacher와 student가 비슷한 표현을 사용하는가?
- CNN과 ViT가 같은 정보를 비슷한 단계에서 표현하는가?

주의:

- 표현이 비슷하다고 해서 기능적으로 완전히 같다는 뜻은 아니다.
- 표현이 다르다고 해서 한쪽이 더 좋다는 뜻도 아니다.
- CKA만으로 실제 정보의 존재나 사용 여부를 확정할 수 없다.

따라서 보통 probe, 출력 비교, ablation과 함께 사용한다.

---

## 1.4 정보 추출과 probe

표현 안에 특정 정보가 얼마나 쉽게 읽히는지 본다.

주요 기법:

- linear probe
- logistic probe
- $k$-NN probe
- concept probe
- layerwise probe
- probing classifier
- auxiliary decoder

주요 질문:

- 클래스 정보가 어느 레이어에서 선형 분리되는가?
- 위치, 자세, 속성, 문법, 의미 정보가 표현에 존재하는가?
- fine-tuning 전후 어떤 정보가 생기거나 사라졌는가?
- self-supervised representation이 downstream task에 유용한가?

주의:

- probe가 성공했다는 것은 정보가 읽힐 수 있다는 뜻이다.
- 본 모델이 실제 추론 과정에서 그 정보를 사용한다는 뜻은 아니다.
- probe가 지나치게 강하면 표현에 없는 구조까지 학습할 수 있다.

따라서 probe는 보통 개입 실험과 함께 해석한다.

---

## 1.5 기하·거리·분포 분석

데이터, 표현, 출력, 정책, 생성 분포 사이의 차이를 거리와 divergence로 측정한다.

주요 기법:

### 벡터·점 사이 거리

- Euclidean distance
- cosine distance
- Mahalanobis distance
- geodesic distance
- class margin
- nearest-neighbor distance

### 분포 사이 차이

- KL divergence
- JS divergence
- MMD
- Wasserstein distance
- Sinkhorn divergence
- total variation
- density ratio
- support overlap
- precision/recall for distributions
- density/coverage

주요 질문:

- source와 target domain이 얼마나 다른가?
- 클래스 조건부 표현분포가 잘 분리되는가?
- 생성 분포가 실제 데이터 분포를 얼마나 덮는가?
- 데이터 증강이 support를 유효하게 확장했는가?
- OOD 입력은 학습 분포에서 얼마나 멀리 떨어졌는가?
- RL 정책 변화가 상태-행동 방문분포를 얼마나 바꾸는가?

---

## 1.6 함수공간 분석

파라미터 자체보다 모델이 나타내는 함수의 차이를 본다.

모델을

$$
f_\theta:\mathcal X\to\mathcal Y
$$

로 보고, 두 모델 $f$와 $g$의 출력 차이를 데이터분포 $P_X$ 위에서

$$
d_{P_X}(f,g)
=
\left(
\mathbb E_{X\sim P_X}
\left[
\|f(X)-g(X)\|^2
\right]
\right)^{1/2}
$$

처럼 측정할 수 있다.

주요 기법:

- output distance
- logit distance
- prediction disagreement
- calibration difference
- decision-boundary comparison
- mode connectivity
- loss interpolation
- empirical NTK
- kernel-target alignment
- function-space ensemble diversity

주요 질문:

- 서로 다른 파라미터가 실제로 같은 함수를 나타내는가?
- 두 optimizer가 다른 함수적 해를 선택하는가?
- fine-tuning이 출력 함수 전체를 얼마나 바꾸는가?
- pruning 이후 기능이 얼마나 보존되는가?
- 앙상블 구성원이 실제로 서로 다른 오류를 내는가?

---

## 1.7 XAI·MI와 개입적 검증

모델이 무엇을 보고 어떤 내부 요소를 실제로 사용하는지 확인한다.

### 상관적 설명

- saliency
- Integrated Gradients
- Grad-CAM
- attention visualization
- concept activation vectors
- attribution map
- feature visualization

### 개입적·인과적 검증

- channel ablation
- head ablation
- layer ablation
- neuron ablation
- occlusion test
- counterfactual input
- activation replacement
- activation patching
- causal tracing
- representation-direction removal
- concept erasure

주요 질문:

- 특정 영역이나 개념이 예측에 실제로 영향을 주는가?
- 특정 채널·헤드·레이어가 필요한가?
- shortcut이나 spurious feature에 의존하는가?
- 내부 표현의 특정 방향을 제거하면 기능이 사라지는가?
- 다른 입력에서 얻은 activation을 주입하면 출력이 복구되는가?

주의:

- attribution과 attention map은 인과적 증거가 아니다.
- 개입 결과도 개입 방식과 metric에 따라 달라질 수 있다.
- XAI·MI는 오류분해와 ablation을 보완하는 도구로 쓰는 편이 안전하다.

---

# 2. 9개 연구 분야별 활용 기법

## 2.1 표현학습·아키텍처·파인튜닝

가장 많은 종류의 수학적 분석법을 자연스럽게 사용할 수 있는 분야다.

핵심 기법:

- CKA, RSA, SVCCA, PWCCA
- linear probe, $k$-NN probe
- activation covariance spectrum
- singular-value spectrum
- effective rank
- intrinsic dimension
- 클래스 내·간 거리
- margin
- alignment와 uniformity
- representation drift
- 출력 disagreement
- concept probe
- XAI
- activation intervention

대표 검증 흐름:

$$
\text{성능 변화}
\to
\text{CKA로 변화 위치 탐색}
\to
\text{probe로 정보 확인}
\to
\text{스펙트럼으로 rank와 collapse 확인}
\to
\text{개입으로 실제 사용 여부 확인}.
$$

적합한 연구 질문:

- CNN과 ViT는 어떤 표현 구조를 만드는가?
- self-supervised objective가 어떤 정보를 보존하는가?
- fine-tuning은 어느 레이어를 가장 많이 바꾸는가?
- 특정 모듈이 표현의 분리도나 rank를 개선하는가?

---

## 2.2 Loss·정규화항·멀티태스크 목적함수 설계

학습 동역학과 표현기하를 동시에 보기 좋은 분야다.

핵심 기법:

- sample/class/task별 gradient norm
- gradient cosine
- gradient conflict rate
- gradient variance
- Hessian spectrum
- sharpness
- margin distribution
- class-wise covariance
- effective rank
- 클래스 내 응집도
- 클래스 간 분리도
- CKA
- probe
- calibration
- robustness
- Pareto front

대표 검증 흐름:

$$
\text{loss 변경}
\to
\text{gradient 구조 변화}
\to
\text{학습 궤적 변화}
\to
\text{표현기하 변화}
\to
\text{최종 오류 변화}.
$$

적합한 연구 질문:

- class imbalance loss가 minority class gradient를 회복하는가?
- contrastive loss가 표현공간의 분리도와 uniformity를 개선하는가?
- multitask loss가 task 간 gradient conflict를 줄이는가?
- regularization이 rank collapse나 overconfidence를 완화하는가?

---

## 2.3 옵티마이저·초기화·스케일링·정규화 모듈

파라미터와 gradient 공간을 가장 직접적으로 분석하는 분야다.

핵심 기법:

- gradient norm
- gradient noise
- gradient heavy-tail 분석
- update-to-weight ratio
- layerwise learning speed
- parameter displacement
- trajectory analysis
- Hessian spectrum
- gradient-eigenspace alignment
- sharpness
- Jacobian singular values
- dynamical isometry
- Fisher geometry
- loss interpolation
- mode connectivity
- empirical NTK
- checkpoint CKA

대표 검증 흐름:

$$
\text{update rule 변경}
\to
\text{gradient/update 통계}
\to
\text{곡률과 스펙트럼}
\to
\text{학습 궤적}
\to
\text{최종 함수와 표현 비교}.
$$

적합한 연구 질문:

- optimizer가 고곡률 방향의 step을 안정화하는가?
- initialization이 gradient flow를 보존하는가?
- normalization이 layer별 update imbalance를 줄이는가?
- width/depth scaling에서 update scale이 안정적으로 유지되는가?

---

## 2.4 Domain adaptation·OOD·데이터 증강·데이터 중심 학습

데이터공간과 분포공간을 가장 직접적으로 다루는 분야다.

핵심 기법:

- MMD
- Wasserstein distance
- Sinkhorn divergence
- KL, JS divergence
- density ratio
- support overlap
- 클래스 조건부 분포 거리
- Mahalanobis distance
- nearest-neighbor distance
- feature covariance
- subspace distance
- intrinsic dimension
- CKA
- probe
- calibration
- uncertainty
- shortcut attribution

입력 표현 사상

$$
z_\theta:\mathcal X\to\mathcal Z
$$

이 있을 때, 입력분포 $P_X$는 표현공간의 분포

$$
(z_\theta)_{\sharp}P_X
$$

로 이동한다. 이를 이용해 source와 target이 표현공간에서 가까워지는지 본다.

대표 검증 흐름:

$$
\text{입력분포 차이}
\to
\text{표현분포 차이}
\to
\text{클래스 조건부 정렬}
\to
\text{오류 변화}.
$$

적합한 연구 질문:

- augmentation이 실제 운영분포의 support를 넓히는가?
- source와 target이 전체적으로만 아니라 클래스별로도 정렬되는가?
- OOD 오류가 특정 표현 방향이나 density 부족과 관련되는가?
- synthetic data가 실제 데이터의 빈 영역을 채우는가?

---

## 2.5 Continual learning·멀티모달·전이학습

태스크·시점·모달리티 사이의 간섭과 표현 변화를 분석하기 좋다.

핵심 기법:

- task별 gradient conflict
- modality별 gradient norm
- CKA
- subspace overlap
- representation drift
- layerwise probe
- Fisher information
- parameter drift
- function drift
- forgetting matrix
- modality별 effective rank
- cross-modal alignment distance
- ablation
- activation patching

대표 질문:

- 이전 정보가 표현에서 사라졌는가?
- 정보는 남아 있지만 readout만 망가졌는가?
- 특정 모달리티가 gradient를 지배하는가?
- 두 모달리티가 같은 의미 부분공간에 정렬되는가?
- replay나 regularization이 실제로 representation drift를 줄이는가?

---

## 2.6 생성모델·Diffusion·Flow·VAE·GAN

분포 거리와 latent geometry가 중심이고, 동역학과 MI는 보조 축이다.

핵심 기법:

- FID
- KID
- CMMD
- precision/recall for generative models
- density/coverage
- MMD
- Wasserstein distance
- support overlap
- latent covariance spectrum
- effective rank
- intrinsic dimension
- mode collapse analysis
- encoder/decoder Jacobian
- score norm
- score-field consistency
- latent probe
- concept direction
- activation steering
- attention analysis
- activation patching

대표 검증 흐름:

$$
\text{전체 분포 거리}
+
\text{품질과 coverage 분해}
+
\text{조건별 성능}
+
\text{latent 구조}
+
\text{sampling 동역학}.
$$

적합한 연구 질문:

- 생성모델이 실제 분포의 mode를 충분히 덮는가?
- sampling step을 줄였을 때 어떤 mode부터 사라지는가?
- latent space가 일부 방향으로 붕괴하는가?
- guidance가 품질과 다양성을 어떻게 교환하는가?
- score 또는 velocity field가 데이터 구조를 적절히 반영하는가?

---

## 2.7 강건성·적대적 학습·안전성

입력공간의 변화가 표현과 출력으로 어떻게 증폭되는지 본다.

핵심 기법:

- input-output Jacobian norm
- local Lipschitz estimate
- spectral norm
- decision margin
- adversarial gradient alignment
- Hessian
- decision-boundary curvature
- clean/adversarial CKA
- neighborhood stability
- attribution stability
- counterfactual intervention

대표 검증 흐름:

$$
\text{입력 섭동}
\to
\text{표현 변화}
\to
\text{출력 변화}
\to
\text{실패 원인 개입}.
$$

적합한 연구 질문:

- 어떤 입력 방향에 가장 민감한가?
- robust training이 표현공간의 이웃 구조를 안정화하는가?
- Lipschitz 또는 spectral regularization이 실제 오류를 줄이는가?
- 적대적 예제가 특정 shortcut을 이용하는가?

---

## 2.8 압축·프루닝·지식증류·저랭크화

스펙트럼, 함수 유사성, 개입 실험을 활용하기 좋다.

핵심 기법:

- weight singular-value spectrum
- activation singular-value spectrum
- effective rank
- Hessian/Fisher importance
- teacher-student CKA
- output KL
- prediction agreement
- function-space distance
- linear probe
- channel/head/layer ablation
- mode connectivity
- activation patching

대표 질문:

- 어떤 부분공간과 기능이 압축 뒤에도 보존되는가?
- teacher와 student가 같은 표현을 배우는가?
- pruning이 단순히 redundancy만 제거하는가?
- low-rank approximation이 중요한 방향을 유지하는가?
- 정확도는 유지되지만 특정 기능이 사라지지는 않았는가?

---

## 2.9 강화학습·정책학습

지도학습보다 분석법의 표준화는 덜 되었지만 여러 공간을 연결할 수 있다.

핵심 기법:

- policy-gradient norm
- policy-gradient variance
- actor-critic gradient alignment
- Fisher information
- trust-region geometry
- policy KL
- policy Wasserstein distance
- occupancy-measure distance
- state-representation CKA
- state/action probe
- value-function spectrum
- policy Jacobian
- policy disagreement
- neuron/head ablation
- trajectory-level intervention

정책 $\pi$가 만드는 상태-행동 방문분포를

$$
d^\pi(s,a)
$$

라고 두면, 정책 업데이트는 단순 파라미터 변화가 아니라 방문분포 자체를 바꾼다.

대표 질문:

- 정책이 바뀌면서 어떤 상태를 더 방문하게 되었는가?
- policy KL은 작지만 occupancy measure는 크게 변하지 않았는가?
- actor와 critic의 gradient가 충돌하는가?
- state representation이 행동 결정에 필요한 정보를 보존하는가?
- 특정 내부 회로가 특정 행동 모드에 인과적으로 필요한가?

---

# 3. 분야의 범용성 순위

아래 순위는 한 분야에서 여러 수학적 분석법을 자연스럽게 쓸 수 있고, 다른 분야에도 경험이 재사용되는 정도를 기준으로 한다.

## 1위. 표현학습·아키텍처·파인튜닝

활용 가능한 기법:

- CKA
- probe
- 스펙트럼
- 거리
- effective rank
- intrinsic dimension
- XAI
- MI
- 함수 비교
- ablation

장점:

- 표현공간 분석의 거의 모든 기법을 적용할 수 있다.
- CNN, ViT, Transformer, multimodal, SSL 등으로 확장하기 쉽다.
- 작은 모델에서도 실험 가능하다.

---

## 2위. Loss·정규화항·멀티태스크 목적함수

활용 가능한 기법:

- gradient 동역학
- Hessian
- margin
- 표현기하
- CKA
- probe
- calibration
- Pareto geometry

장점:

- 목적함수 변화에서 최종 표현 변화까지 연결하기 좋다.
- 성능 향상의 원인 설명이 비교적 명확하다.
- 데이터 불균형, metric learning, multitask 등 실무 문제와 연결하기 쉽다.

---

## 3위. 옵티마이저·초기화·스케일링

활용 가능한 기법:

- gradient/update 통계
- Hessian/Fisher/Jacobian spectrum
- trajectory
- mode connectivity
- NTK
- 함수 비교

장점:

- 파라미터와 gradient 공간을 가장 직접적으로 다룬다.
- 학습 안정성 연구에 강하다.
- 수학적 깊이를 보여주기 좋다.

한계:

- 계산 비용이 커질 수 있다.
- 작은 성능 차이를 해석하기가 어렵다.
- XAI와 probe는 보조적이다.

---

## 4위. Domain adaptation·OOD·데이터 증강·데이터 중심 학습

활용 가능한 기법:

- MMD
- Wasserstein
- Sinkhorn
- support overlap
- density
- 클래스 조건부 거리
- intrinsic dimension
- CKA
- probe
- calibration

장점:

- 데이터공간과 표현분포 분석을 직접 연결한다.
- 운영환경의 distribution shift와 자연스럽게 연결된다.
- 비전, 음성, NLP 모두에서 활용 가능하다.

---

## 5위. Continual learning·멀티모달·전이학습

활용 가능한 기법:

- gradient conflict
- CKA
- representation drift
- Fisher
- subspace overlap
- probe
- ablation
- activation patching

장점:

- 간섭, 망각, 정렬을 여러 공간에서 분석할 수 있다.
- 모달리티·태스크·시간축을 함께 다룬다.

---

## 6위. 생성모델

활용 가능한 기법:

- 분포 거리
- quality/coverage 분해
- latent spectrum
- intrinsic dimension
- Jacobian
- score/velocity field
- concept intervention

장점:

- 확률분포와 측도공간 관점이 가장 직접적이다.
- diffusion, flow, VAE, GAN을 비교하기 좋다.

한계:

- 평가가 비싸고 불안정할 수 있다.
- 작은 모델에서는 생성 품질과 metric 신뢰도가 제한될 수 있다.

---

## 7위. 강건성·적대적 학습

활용 가능한 기법:

- Jacobian
- Lipschitz
- spectral norm
- margin
- curvature
- CKA
- attribution stability
- counterfactual intervention

장점:

- 입력공간과 함수공간의 민감도를 직접 연결한다.
- 안전성과 실무 리스크에 연결하기 쉽다.

---

## 8위. 압축·프루닝·증류

활용 가능한 기법:

- 스펙트럼
- Hessian/Fisher
- CKA
- 함수 거리
- ablation
- patching

장점:

- 계산량, 메모리, 지연시간과 수학적 구조를 함께 분석할 수 있다.
- 온디바이스·경량화 프로젝트와 잘 맞는다.

---

## 9위. 강화학습·정책학습

활용 가능한 기법:

- policy-gradient 동역학
- Fisher geometry
- policy distance
- occupancy measure
- CKA
- probe
- trajectory intervention

장점:

- 동역학, 분포, 표현, 제어를 모두 연결할 수 있다.

한계:

- 환경과 데이터가 정책에 따라 변해 변인통제가 어렵다.
- 동일 조건 반복과 통계적 검증 비용이 크다.
- 작은 실험에서도 seed variance가 클 수 있다.

---

# 4. 기법 자체의 범용성 순위

## 4.1 1군: 거의 모든 딥러닝 연구에 적용 가능

### 1. Gradient norm·cosine·variance

활용:

- optimizer
- loss
- multitask
- RL
- continual learning
- scaling

장점:

- 구현이 간단하다.
- 학습 이상을 빠르게 찾는다.
- 변경 원인을 update 수준에서 볼 수 있다.

---

### 2. 공분산·SVD·effective rank

활용:

- 표현학습
- 압축
- 생성모델
- collapse 분석
- architecture 비교

장점:

- 표현의 차원과 집중도를 정량화할 수 있다.
- 작은 모델에서도 계산 가능하다.
- 데이터, activation, weight 모두에 적용할 수 있다.

---

### 3. CKA 또는 RSA

활용:

- 레이어 비교
- 모델 비교
- fine-tuning drift
- teacher-student 비교
- architecture 비교

장점:

- 모델 크기나 뉴런 순서가 달라도 비교 가능하다.
- 학습 시점별 표현 변화를 보기 좋다.

주의:

- 기능적 동일성이나 인과성을 보장하지 않는다.

---

### 4. Linear probe와 $k$-NN probe

활용:

- representation learning
- SSL
- transfer learning
- continual learning
- multimodal learning

장점:

- 표현에 어떤 정보가 있는지 쉽게 확인할 수 있다.
- 레이어별 정보 흐름을 볼 수 있다.

주의:

- 정보가 존재한다는 것과 실제로 사용된다는 것은 다르다.

---

### 5. 거리·margin·클래스 내/간 분산

활용:

- classification
- metric learning
- contrastive learning
- OOD
- long-tail
- representation geometry

장점:

- 결과를 직관적으로 해석하기 쉽다.
- 클래스별 failure와 연결하기 좋다.

---

### 6. Controlled ablation

활용:

- 거의 모든 연구 분야

장점:

- 특정 구성요소의 기여를 직접 검증한다.
- 복잡한 수학적 분석의 최소 기준점이 된다.

주의:

- 한 번에 하나의 요소만 바꾸는 통제가 필요하다.
- seed와 비용을 함께 기록해야 한다.

---

## 4.2 2군: 가설이 있을 때 강력한 기법

### 7. Hessian·Fisher·Jacobian spectrum

적합한 가설:

- 학습 불안정성
- 곡률
- gradient flow
- robustness
- pruning sensitivity
- initialization

한계:

- 계산비용이 크다.
- 근사법에 따라 결과가 달라질 수 있다.

---

### 8. MMD·Wasserstein·Sinkhorn

적합한 가설:

- domain shift
- 생성분포
- 데이터 증강
- 표현분포 정렬
- OOD

한계:

- feature space와 kernel/cost 선택에 민감하다.
- 전체 거리만으로 클래스 조건부 오류를 설명하기 어렵다.

---

### 9. 함수 출력 거리·mode connectivity

적합한 가설:

- optimizer별 해 비교
- fine-tuning drift
- pruning
- ensemble diversity
- architecture 비교

한계:

- 어떤 입력분포에서 측정했는지에 따라 결론이 달라진다.

---

### 10. XAI attribution과 concept direction

적합한 가설:

- shortcut
- 특정 영역·개념 의존
- 오류 sample 해석
- 데이터 편향

한계:

- 인과적 증거가 아니다.
- 시각화 방법마다 결과가 달라질 수 있다.

---

### 11. Activation patching·causal tracing

적합한 가설:

- 특정 내부 계산의 필요성
- 정보 흐름
- circuit 단위 설명
- 기능 복구 실험

한계:

- 개입 방식과 metric 설계가 어렵다.
- 모델이 작아도 실험 조합 수가 커질 수 있다.

---

## 4.3 3군: 전문 연구에 가까운 기법

### 12. Empirical NTK와 kernel spectrum

적합한 연구:

- lazy training
- width scaling
- function-space dynamics
- kernel alignment

한계:

- 대규모 모델에서 계산비용이 크다.
- 실제 feature learning을 충분히 설명하지 못할 수 있다.

---

### 13. Intrinsic dimension과 TDA

적합한 연구:

- manifold structure
- representation complexity
- topology
- data geometry

한계:

- 추정기와 hyperparameter에 민감하다.
- 실무 성능과 직접 연결하기 어렵다.

---

### 14. Fisher–Rao·Wasserstein geometry

적합한 연구:

- natural gradient
- distribution dynamics
- generative modeling
- information geometry

한계:

- 구현과 해석 난도가 높다.
- 일반 응용 논문의 검증 도구로는 과할 수 있다.

---

### 15. 함수족 복잡도와 generalization bound

적합한 연구:

- statistical learning theory
- approximation theory
- capacity
- norm-based bounds

한계:

- 실제 모델의 성능 차이를 정밀하게 설명하지 못하는 경우가 많다.
- 실험 검증보다는 이론 분석 성격이 강하다.

---

# 5. 범용 검증 스택

## 5.1 전체 순서

$$
\boxed{
\text{성능·오류분해}
\to
\text{학습동역학}
\to
\text{스펙트럼·rank}
\to
\text{CKA·probe}
\to
\text{거리·분포 구조}
\to
\text{ablation·인과 개입}
}
$$

---

## 5.2 1단계: 성능과 오류분해

먼저 확인할 것:

- train/validation/test 성능
- 평균 metric
- class별 metric
- slice별 metric
- false positive
- false negative
- calibration
- seed별 분산
- 비용과 latency

목적:

- 실제로 개선이 존재하는지 확인한다.
- 개선이 어느 실패군에서 발생했는지 찾는다.
- 이후 수학적 분석의 가설을 정한다.

---

## 5.3 2단계: Gradient와 update 동역학

확인할 것:

- gradient norm
- layerwise gradient norm
- gradient cosine
- gradient variance
- update-to-weight ratio
- parameter displacement
- 학습 시점별 변화

목적:

- 변경이 실제 update에 어떤 영향을 주었는지 확인한다.
- 특정 layer나 class가 학습을 지배하는지 본다.
- instability, vanishing, explosion, conflict를 찾는다.

---

## 5.4 3단계: 스펙트럼과 rank

확인할 것:

- activation covariance spectrum
- weight singular values
- effective rank
- Hessian top eigenvalue
- Jacobian singular values
- condition number

목적:

- 표현 붕괴를 확인한다.
- 학습의 곡률과 conditioning을 본다.
- 저랭크 구조와 redundancy를 찾는다.

---

## 5.5 4단계: CKA와 probe

확인할 것:

- 레이어별 CKA
- checkpoint 간 CKA
- 모델 간 CKA
- linear probe
- $k$-NN probe
- concept probe

목적:

- 변화가 발생한 레이어를 찾는다.
- 표현에 어떤 정보가 추가·삭제되었는지 본다.
- 파라미터 변화가 실제 표현 변화로 이어졌는지 확인한다.

---

## 5.6 5단계: 거리와 분포 구조

확인할 것:

- class margin
- 클래스 내·간 거리
- Mahalanobis distance
- MMD
- Wasserstein
- support overlap
- density/coverage
- 클래스 조건부 거리

목적:

- 데이터·표현·생성분포의 구조를 정량화한다.
- domain shift와 OOD를 확인한다.
- 평균적 정렬이 아니라 클래스별 구조까지 본다.

---

## 5.7 6단계: Ablation과 인과 개입

확인할 것:

- component ablation
- channel/head/layer removal
- occlusion
- counterfactual input
- representation-direction removal
- activation patching
- causal tracing

목적:

- 상관적 증거를 넘어 실제 필요성을 확인한다.
- 특정 표현이나 내부 회로가 기능에 필요한지 검증한다.
- 성능 변화의 원인을 더 강하게 주장할 수 있게 한다.

---

# 6. 해석 시 증거의 강도

수학적 분석 결과는 다음과 같이 증거 강도가 다르다.

## 6.1 관찰적 증거

예:

- gradient norm이 줄었다.
- CKA가 높아졌다.
- effective rank가 증가했다.
- MMD가 감소했다.
- Grad-CAM이 객체 영역을 강조했다.

의미:

- 두 현상이 함께 나타났다는 증거다.
- 성능 향상의 원인임을 직접 보장하지 않는다.

---

## 6.2 예측적 증거

예:

- gradient conflict가 높은 조건에서 성능이 낮았다.
- 낮은 rank가 나타난 class에서 recall이 낮았다.
- 큰 distribution distance가 있는 domain에서 오류가 증가했다.

의미:

- 분석량이 실패를 예측하거나 설명한다.
- 여전히 인과성은 확정되지 않는다.

---

## 6.3 개입적 증거

예:

- 특정 채널을 제거하자 기능이 사라졌다.
- 특정 activation을 복구하자 정답 출력이 회복됐다.
- 특정 표현 방향을 제거하자 class 성능이 감소했다.
- 하나의 loss 항만 제거하자 gradient conflict와 성능이 동시에 악화됐다.

의미:

- 해당 요소가 실제 계산에 필요하다는 더 강한 증거다.
- 단, 개입 자체가 다른 부작용을 만들지 않았는지 통제해야 한다.

---

# 7. 실험 설계의 최소 원칙

수학적 분석법을 쓰더라도 다음 조건이 없으면 검증력이 약하다.

- baseline을 고정한다.
- 한 번에 하나의 요소만 바꾼다.
- seed를 여러 개 사용한다.
- 평균과 분산을 함께 보고한다.
- class·domain·난이도별 slice를 본다.
- 분석 metric과 task metric을 함께 본다.
- 계산 비용과 latency를 함께 기록한다.
- 분석이 가설을 사후적으로 꾸민 것이 아닌지 구분한다.
- 상관적 분석 뒤에는 가능한 범위에서 ablation이나 intervention을 붙인다.

---

# 8. 실무·프로젝트용 권장 조합

## 8.1 Loss 또는 optimizer 개선 프로젝트

추천 조합:

1. 성능·오류분해
2. gradient norm·cosine·variance
3. update-to-weight ratio
4. Hessian top eigenvalue 또는 sharpness
5. activation covariance spectrum
6. CKA
7. linear probe
8. loss 항 ablation

---

## 8.2 표현학습 또는 architecture 비교 프로젝트

추천 조합:

1. downstream 성능
2. layerwise linear probe
3. CKA
4. PCA/SVD와 effective rank
5. 클래스 내·간 거리
6. intrinsic dimension
7. concept probe
8. channel/head/layer ablation

---

## 8.3 데이터 증강·domain shift 프로젝트

추천 조합:

1. domain별 오류분해
2. 입력분포 거리
3. 표현분포 MMD 또는 Wasserstein
4. 클래스 조건부 거리
5. support overlap
6. CKA
7. calibration
8. shortcut attribution
9. augmentation ablation

---

## 8.4 생성모델 프로젝트

추천 조합:

1. FID/KID 또는 task-specific metric
2. precision/recall 또는 density/coverage
3. 조건별 오류분해
4. latent covariance spectrum
5. effective rank
6. support overlap
7. sampling step별 품질·latency
8. concept intervention 또는 latent probe

---

## 8.5 압축·온디바이스 프로젝트

추천 조합:

1. 정확도·latency·memory·전력
2. weight/activation SVD
3. effective rank
4. Hessian/Fisher sensitivity
5. teacher-student CKA
6. output KL과 disagreement
7. channel/head ablation
8. 실제 hardware benchmark

---

# 9. 핵심 요약

가장 범용적인 분석 도구는 다음과 같다.

$$
\boxed{
\text{gradient}
+
\text{SVD/rank}
+
\text{CKA}
+
\text{probe}
+
\text{거리/분포}
+
\text{ablation}
}
$$

분야별 중심축은 다음처럼 정리할 수 있다.

| 분야 | 중심 분석축 |
|---|---|
| 표현학습·architecture | 표현 비교, probe, spectrum, XAI/MI |
| loss 설계 | gradient, curvature, margin, 표현기하 |
| optimizer·scaling | 동역학, Hessian/Fisher/Jacobian, trajectory |
| domain adaptation·OOD | 데이터·표현분포 거리, support, density |
| continual·multimodal | drift, conflict, alignment, ablation |
| 생성모델 | 분포 거리, coverage, latent geometry, field dynamics |
| 강건성 | Jacobian, Lipschitz, margin, counterfactual |
| 압축·증류 | spectrum, function distance, CKA, ablation |
| 강화학습 | policy geometry, occupancy measure, trajectory intervention |

최종적으로 가장 안정적인 검증 논리는 다음이다.

$$
\boxed{
\text{성능이 달라졌다}
\to
\text{학습 과정이 달라졌다}
\to
\text{표현 또는 함수가 달라졌다}
\to
\text{분포와 오류 구조가 달라졌다}
\to
\text{개입해도 그 원인이 재현된다}
}
$$

---

# 10. 방향과 Jacobian 관점에서 본 7개 검증축의 공통 구조

딥러닝의 많은 validation 기법은 서로 다른 이름을 쓰지만, 수학적으로는 다음 공통 질문으로 정리할 수 있다.

> 고차원 공간 전체를 직접 이해하기 어려울 때, 어떤 **방향** 또는 **부분공간**이 중요한지 찾고, 그 방향으로 움직였을 때 무엇이 변하는가를 측정한다.

여기서 중요한 것은 모든 기법이 같은 공간을 보는 것이 아니라는 점이다.

| 기법 | 주로 보는 공간 | 방향의 의미 |
|---|---|---|
| Gradient norm·cosine | 파라미터 공간 | loss를 가장 빠르게 변화시키는 방향 |
| Gradient conflict | 파라미터 공간 | 서로 다른 task/loss가 요구하는 update 방향의 정렬 또는 충돌 |
| Hessian spectrum | 파라미터 공간 | loss surface가 특히 많이 또는 적게 휘는 방향 |
| Parameter trajectory | 파라미터 공간 | 실제 학습이 이동한 방향 |
| SVD·covariance spectrum | 표현공간 | activation 또는 데이터 변화가 집중된 주요 방향 |
| Effective rank | 표현공간 | 실질적으로 사용되는 방향의 수 |
| Linear probe | 표현공간 | label 또는 속성을 선형적으로 읽을 수 있는 방향 |
| CKA·subspace overlap | 표현공간 | 두 모델 또는 두 layer의 주요 부분공간 관계 |
| Margin | 표현공간·출력공간 | decision boundary와의 방향별 거리 |
| Jacobian | 입력·표현·출력 사이 | 한 공간의 방향 변화가 다음 공간에서 어떤 방향으로 전달되는가 |
| XAI·MI intervention | 표현공간 | 특정 의미 방향·부분공간을 건드렸을 때 기능이 어떻게 변하는가 |

## 10.1 Gradient: 학습이 어느 방향으로 가려 하는가

현재 parameter를 $\theta$라고 하면 loss $L$의 gradient는

$$
\nabla_\theta L(\theta)
$$

이다.

이 벡터는 파라미터 공간에서 loss가 가장 빠르게 증가하는 방향을 나타낸다. 따라서 gradient descent는 그 반대 방향으로 움직인다.

Gradient 분석에서 방향을 보는 대표적인 양은 cosine similarity다. 두 gradient $g_1,g_2$가 있을 때

$$
\operatorname{cos}(g_1,g_2)
=
\frac{\langle g_1,g_2\rangle}{\|g_1\|\,\|g_2\|}
$$

를 사용한다.

이 값은 다음 질문에 답한다.

- 두 loss가 비슷한 방향으로 parameter를 움직이려 하는가?
- 서로 다른 task가 같은 update 방향을 요구하는가?
- 연속 step의 gradient가 일관된가, 아니면 방향이 계속 뒤집히는가?

즉 gradient validation은 단순히 크기만 보는 것이 아니라 **학습 방향의 정렬과 충돌**을 보는 분석이다.

---

## 10.2 Hessian: 방향마다 loss surface가 어떻게 다른가

Gradient가 현재 위치에서의 1차 변화 방향을 알려준다면 Hessian은 방향별 곡률을 알려준다.

Loss의 Hessian을

$$
H(\theta)=\nabla_\theta^2L(\theta)
$$

라고 하자.

어떤 정규화된 방향 $v$로 아주 조금 움직일 때 그 방향의 2차 곡률은

$$
v^\top H(\theta)v
$$

로 측정된다.

특히 eigenvector $v_i$와 eigenvalue $\lambda_i$가

$$
H(\theta)v_i=\lambda_i v_i
$$

를 만족하면, $v_i$는 loss surface의 특별한 방향이고 $\lambda_i$는 그 방향의 곡률이다.

따라서 다음처럼 해석할 수 있다.

- 큰 $\lambda_i>0$: 매우 가파른 방향
- $\lambda_i\approx0$: 거의 평평한 방향
- $\lambda_i<0$: 내려갈 수 있는 saddle 방향

이 때문에 plateau, saddle, sharpness, optimizer 안정성 분석은 본질적으로 **파라미터 공간의 방향별 지형 분석**이다.

---

## 10.3 SVD와 covariance spectrum: 표현이 어느 방향에 집중되는가

어떤 layer의 표현을

$$
h(x)\in\mathbb R^d
$$

라고 하자.

여러 sample의 표현을 모으면 고차원 점구름이 만들어진다. 이때 모든 $d$개 방향이 똑같이 사용되는 것은 아니다.

중심화된 feature matrix를 $X$라고 하면 covariance나 SVD를 통해 주요 방향을 찾을 수 있다.

예를 들어

$$
X=U\Sigma V^\top
$$

에서 $V$의 column들은 표현공간에서 주요 변화 방향을 나타내고, singular value는 각 방향의 중요도를 나타낸다.

따라서 다음 기법들은 모두 같은 구조와 연결된다.

- singular-value spectrum: 어떤 방향이 얼마나 강한가
- effective rank: 실제로 몇 개 방향을 쓰는가
- collapse 분석: 지나치게 적은 방향만 사용하는가
- low-rank approximation: 중요한 방향만 남길 수 있는가

즉 spectrum 분석은 **표현공간에서 중요한 방향들의 분포를 측정하는 방법**이다.

---

## 10.4 Probe와 CKA: 읽을 수 있는 방향과 공유되는 부분공간

Linear probe는 표현 $h(x)$에서 label $y$를 읽기 위해 선형 사상

$$
q(h)=Wh+b
$$

를 학습한다.

따라서 probe가 잘 작동한다는 것은 표현공간 안에 label과 연결되는 **선형적으로 읽을 수 있는 방향 또는 부분공간**이 존재한다는 뜻이다.

CKA와 subspace overlap은 조금 다른 질문을 한다.

> 두 모델이나 두 layer가 같은 좌표를 쓰는가가 아니라, sample 관계를 담는 중요한 부분공간이 얼마나 비슷한가?

따라서 probe와 CKA는 각각 다음을 본다.

- probe: 어떤 정보가 어느 방향으로 읽히는가
- CKA/subspace analysis: 두 표현의 중요한 방향 집합이 얼마나 유사한가

이 관점에서 SVD, probe, CKA, TCAV는 서로 독립적인 기법이라기보다 **표현공간의 주요 방향과 부분공간을 서로 다른 방식으로 조사하는 방법**으로 볼 수 있다.

---

## 10.5 Jacobian: 서로 다른 공간의 방향을 연결하는 핵심 도구

방향 관점을 여러 validation 축 사이에서 연결할 때 가장 중요한 개념 중 하나가 Jacobian이다.

입력 $x$가 중간표현 $h(x)$로 가고, 다시 출력 $f(h)$로 간다고 하자.

$$
x
\xmapsto{\,h\,}
h(x)
\xmapsto{\,f\,}
y
$$

입력공간에서 작은 방향 $v_x$로 움직였을 때 표현공간에서의 1차 변화 방향은

$$
v_h
=
J_h(x)v_x
$$

이다.

여기서 $J_h(x)$는 입력 $x$에서 표현사상 $h$의 Jacobian이다.

다시 이 표현 변화가 출력에 미치는 1차 변화는

$$
v_y
=
J_f(h)v_h
$$

로 전달된다.

따라서 전체적으로는

$$
\boxed{
\text{입력의 방향}
\rightarrow
\text{표현의 방향}
\rightarrow
\text{출력의 방향}
}
$$

이라는 구조가 생긴다.

이 관점은 여러 기법을 자연스럽게 연결한다.

- saliency·gradient attribution: 출력이 입력의 어느 방향에 민감한가
- TCAV: 출력 gradient와 concept direction이 얼마나 정렬되는가
- Jacobian spectrum: 어떤 입력·표현 방향이 증폭되거나 감쇠되는가
- robustness: 작은 입력 방향 변화가 출력에서 얼마나 커지는가
- dynamical isometry: layer를 지날 때 방향과 크기가 안정적으로 전달되는가
- activation intervention: 특정 표현 방향 또는 부분공간을 바꿨을 때 출력이 어떻게 달라지는가

즉 Jacobian은 단순한 미분 행렬이 아니라 **서로 다른 표현공간 사이에서 방향이 어떻게 전달되는지를 나타내는 국소 선형사상**으로 이해할 수 있다.

---

## 10.6 XAI·MI: 의미 있는 방향을 찾고 실제로 개입한다

XAI·MI도 같은 방향 관점의 연장선으로 볼 수 있다.

### XAI

주로 묻는 질문은 다음과 같다.

> 어떤 입력 또는 표현 방향이 현재 출력과 강하게 연결되어 있는가?

예:

- saliency: 입력 방향 민감도
- Grad-CAM: 특정 출력과 연결되는 feature 방향을 공간 위치로 투영
- TCAV: 사람이 정의한 concept direction과 출력 gradient의 정렬

### MI

MI는 한 단계 더 나아간다.

> 그 방향이나 부분공간을 실제로 제거·교체·주입하면 기능이 변하는가?

예:

- direction removal
- concept erasure
- activation patching
- activation replacement
- causal tracing

따라서 XAI와 MI의 관계를 방향 관점에서 정리하면

$$
\boxed{
\text{XAI}
:
\text{중요한 방향을 찾는다}
}
$$

$$
\boxed{
\text{MI}
:
\text{그 방향·부분공간에 개입해 기능적 필요성을 확인한다}
}
$$

로 볼 수 있다.

---

## 10.7 7개 검증축을 방향 관점으로 다시 압축

7개 축 가운데 방향과 직접 연결되는 부분은 다음처럼 요약할 수 있다.

$$
\boxed{
\begin{array}{ll}
\text{Gradient} & \text{어느 방향으로 학습하려 하는가}\\
\text{Hessian} & \text{어느 방향으로 지형이 휘는가}\\
\text{SVD/Spectrum} & \text{어느 방향에 표현이 집중되는가}\\
\text{Probe} & \text{어느 방향으로 정보를 읽을 수 있는가}\\
\text{CKA/Subspace} & \text{중요 방향들의 공간이 서로 비슷한가}\\
\text{Jacobian} & \text{방향이 다음 공간으로 어떻게 전달되는가}\\
\text{XAI/MI} & \text{어느 의미 방향이 실제 출력과 기능에 쓰이는가}
\end{array}
}
$$

이를 더 큰 흐름으로 쓰면

$$
\boxed{
\text{중요한 방향을 찾는다}
\rightarrow
\text{방향별 크기·곡률·정보를 측정한다}
\rightarrow
\text{공간 사이에서 어떻게 전달되는지 본다}
\rightarrow
\text{필요하면 그 방향에 직접 개입한다}
}
$$

라고 정리할 수 있다.

---

## 10.8 방향 관점으로 완전히 환원되지 않는 축

방향은 강력한 공통 언어이지만 validation 전체의 유일한 원리는 아니다.

다음 기법은 방향보다 분포 전체나 집단 수준 구조가 더 중요하다.

- MMD, Wasserstein, Sinkhorn: 분포 전체의 차이
- support overlap, density/coverage: 분포의 support와 coverage
- error analysis: 실패 sample 집단과 slice 구조
- calibration: confidence와 실제 정답 빈도의 관계
- FID/KID: 생성분포와 실제분포의 전체적 차이

따라서 가장 정확한 정리는 다음과 같다.

> 딥러닝 validation 전체가 방향 하나로 환원되는 것은 아니다. 그러나 파라미터·표현·함수처럼 고차원 공간을 직접 분석할 때는 **중요한 방향과 부분공간을 찾고, 그 방향이 어떻게 전달되고 기능에 사용되는지를 보는 것**이 반복해서 등장하는 핵심 전략이다.

이 관점에서 Jacobian은 여러 공간의 방향을 연결하고, XAI·MI는 그 방향에 의미와 인과적 기능을 연결하는 역할을 한다.
