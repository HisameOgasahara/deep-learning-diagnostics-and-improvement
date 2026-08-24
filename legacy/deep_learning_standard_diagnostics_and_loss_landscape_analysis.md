# 딥러닝 표준 진단법과 Plateau·Saddle·Local Minimum 분석

## 0. 문서 목적

이 문서는 딥러닝 모델의 내부 상태를 실제로 측정하고 검증할 때 자주 쓰이는 방법을 다음 세 범주로 정리한다.

1. **표준 진단**: 일반적인 학습 디버깅과 표현 분석에 널리 사용할 수 있는 방법
2. **연구 표준 진단**: 최적화·표현 연구에서 반복적으로 사용되지만 계산비용이나 해석 난도가 더 높은 방법
3. **Plateau·saddle·local minimum·basin 분석**: 학습이 정체되거나 서로 다른 해의 관계를 조사할 때 사용하는 단계별 절차

핵심 원칙은 다음과 같다.

> 하나의 그림이나 수치로 모델 상태를 판정하지 않고,  
> 학습 궤적 → 국소 곡률 → 방향별 교란 → 재시작 → checkpoint 연결성을 순서대로 확인한다.

---

# 1. 표준 진단

## 1.1 Gradient 통계

### 왜 필요한가

Loss가 줄지 않거나 발산했을 때 가장 먼저 알고 싶은 것은 다음이다.

- gradient가 실제로 흐르는가?
- 특정 layer에서만 사라지거나 폭발하는가?
- update 방향이 계속 뒤집히는가?
- mini-batch에 따라 방향이 지나치게 흔들리는가?

현재 학습 step을 $t$라고 하고, 그때의 전체 gradient를

$$
g_t=\nabla_\theta L(\theta_t)
$$

라고 둔다.

여기서 $\theta_t$는 step $t$에서의 전체 model parameter이고, $L$은 학습 loss다.

### 실제 측정량

#### Gradient norm

$$
\|g_t\|
$$

해석:

- 갑자기 매우 커짐: gradient explosion, 과도한 learning rate, 수치 불안정 가능성
- 거의 0으로 유지됨: gradient vanishing, saturation, plateau 가능성
- 심하게 출렁임: batch noise, 불안정한 loss, 큰 learning rate 가능성

#### Layerwise gradient norm

Layer $\ell$의 parameter를 $\theta_t^{(\ell)}$라고 하면

$$
g_t^{(\ell)}
=
\nabla_{\theta^{(\ell)}}L(\theta_t)
$$

을 계산하고 $\|g_t^{(\ell)}\|$을 기록한다.

주요 목적:

- 앞쪽 layer에서 gradient가 사라지는지 확인
- 특정 head나 block에 gradient가 몰리는지 확인
- freeze·unfreeze 범위 결정
- optimizer나 normalization 변경 효과 비교

#### 연속 step gradient 방향

두 gradient의 방향이 얼마나 비슷한지 확인하기 위해

$$
\operatorname{cos}(g_t,g_{t-1})
=
\frac{
\langle g_t,g_{t-1}\rangle
}{
\|g_t\|\,\|g_{t-1}\|
}
$$

을 사용한다.

해석:

- $1$에 가까움: 비슷한 방향으로 일관되게 이동
- $0$ 근처: 방향 연관성이 약함
- 음수: update 방향이 자주 반전되며 진동할 가능성

#### Update-to-weight ratio

실제 parameter update를

$$
\Delta\theta_t
=
\theta_{t+1}-\theta_t
$$

라고 두고

$$
r_t
=
\frac{
\|\Delta\theta_t\|
}{
\|\theta_t\|+\varepsilon
}
$$

를 측정한다.

이 비율이 필요한 이유는 gradient가 커도 optimizer의 scaling, clipping, momentum 때문에 실제 이동량은 작을 수 있기 때문이다.

### 주요 사용 상황

- Loss가 NaN 또는 Inf가 됨
- Loss가 처음부터 거의 줄지 않음
- 특정 layer만 학습되지 않음
- optimizer 비교
- learning rate·batch size 조정
- gradient clipping 필요성 판단

---

## 1.2 Activation 통계

### 왜 필요한가

Gradient가 정상이어도 중간 activation이 죽거나 한 값으로 몰리면 표현을 제대로 만들 수 없다.

Layer $\ell$의 activation을

$$
h^{(\ell)}(x)
$$

라고 두며, 이는 입력 $x$를 layer $\ell$까지 통과시켰을 때 얻는 표현이다.

### 실제 측정량

- 평균
- 분산
- 최소값·최대값
- 0인 값의 비율
- saturation 비율
- channel별 histogram
- train mode와 eval mode의 분포 차이

### 주요 해석

| 관측 | 가능한 의미 |
|---|---|
| ReLU 출력 대부분이 0 | Dying ReLU 또는 지나친 음수 편향 |
| activation 분산이 layer를 거치며 급감 | 정보와 gradient가 소실될 가능성 |
| 분산이 급증 | 수치 불안정 또는 normalization 문제 |
| 모든 sample의 표현이 비슷해짐 | representation collapse 가능성 |
| train/eval 분포가 크게 다름 | BatchNorm, dropout, preprocessing 불일치 가능성 |

### 주요 사용 상황

- 학습은 진행되는데 성능이 거의 오르지 않음
- 특정 layer 이후 표현이 무너짐
- normalization·activation function 비교
- quantization 전후 내부 범위 확인
- train과 inference 결과 불일치 조사

---

## 1.3 Covariance spectrum과 Effective rank

### 왜 필요한가

Layer 출력이 512차원이라고 해도 실제 데이터가 512개 방향을 모두 사용하는 것은 아니다. 표현이 몇 개 방향에 집중되어 있는지 보려면 activation covariance의 고유값 또는 feature matrix의 singular value를 측정한다.

같은 layer에서 $n$개 sample의 중심화된 표현을 행렬

$$
X\in\mathbb R^{n\times d}
$$

로 모은다.

여기서 $n$은 sample 수이고, $d$는 activation dimension이다.

공분산 행렬은

$$
C
=
\frac{1}{n}X^\top X
$$

이다.

### 실제 측정량

- 고유값 분포
- 누적 설명분산
- 상위 고유값 비중
- singular-value spectrum
- condition number
- effective rank

고유값을 $\lambda_1,\ldots,\lambda_d$라고 하고

$$
p_i
=
\frac{\lambda_i}{\sum_j\lambda_j}
$$

로 정규화하면 entropy 기반 effective rank를

$$
r_{\mathrm{eff}}
=
\exp\left(
-\sum_i p_i\log p_i
\right)
$$

처럼 정의할 수 있다.

이 수치가 필요한 이유는 단순히 dimension 수가 아니라 **실제로 사용되는 방향 수**를 요약하기 위해서다.

### 주요 해석

| 관측 | 가능한 의미 |
|---|---|
| 소수 고유값에 대부분 집중 | 표현이 저랭크 구조에 집중 |
| effective rank가 급격히 감소 | collapse 또는 지나친 압축 가능성 |
| rank가 지나치게 높고 작은 고유값이 많음 | noise·중복 자유도 가능성 |
| 개선 모델에서 핵심 slice의 rank가 회복 | 더 다양한 특징을 사용하게 되었을 가능성 |

### 주요 사용 상황

- representation collapse 확인
- embedding dimension 축소 가능성 판단
- pruning·compression 후보 탐색
- architecture 또는 loss 변경 전후 표현 구조 비교
- layer별 정보 압축 양상 확인

---

## 1.4 Linear probe

### 왜 필요한가

중간 표현에 특정 정보가 존재하는지 확인하려면, 그 표현을 고정하고 간단한 선형 모델만 학습한다.

Layer $\ell$의 고정된 표현을 $h^{(\ell)}(x)$라고 할 때 probe는

$$
q_\phi:
h^{(\ell)}(x)\mapsto y
$$

형태의 선형 분류기 또는 선형 회귀기다.

여기서 $\phi$만 학습하고 원래 model parameter는 고정한다.

### 무엇을 확인하는가

- 클래스 정보가 어느 layer부터 선형적으로 읽히는가?
- pretrained representation이 downstream task에 유용한가?
- fine-tuning 전후 어떤 정보가 생기거나 사라졌는가?
- backbone에는 정보가 있지만 head가 활용하지 못하는가?

### 주요 사용 상황

- backbone과 head 중 병목 위치 구분
- self-supervised representation 평가
- layer별 정보 흐름 비교
- teacher와 student 비교
- fine-tuning 범위 결정

### 해석상의 주의

Linear probe 성능이 높다는 것은 **정보가 선형적으로 읽힐 수 있다**는 뜻이다.  
원래 모델이 실제 예측 과정에서 그 정보를 사용한다는 뜻은 아니다.

따라서 필요하면 ablation, occlusion, activation intervention을 추가한다.

---

# 2. 연구 표준 진단

## 2.1 CKA

### 왜 필요한가

서로 다른 두 layer나 두 모델은 neuron 수와 neuron 순서가 다를 수 있어 좌표별 직접 비교가 어렵다. CKA는 같은 입력 sample들 사이의 관계가 두 표현에서 얼마나 비슷한지를 비교한다.

두 activation matrix를

$$
X\in\mathbb R^{n\times d_x},
\qquad
Y\in\mathbb R^{n\times d_y}
$$

라고 한다.

두 표현의 sample 관계는 Gram matrix

$$
K=XX^\top,
\qquad
G=YY^\top
$$

로 나타낼 수 있다.

Linear CKA는 중심화된 두 Gram matrix가 얼마나 정렬되는지를 측정한다.

### 주요 사용 상황

- fine-tuning 전후 표현 변화
- 서로 다른 seed로 학습한 모델 비교
- CNN과 ViT layer 대응
- teacher와 student 표현 비교
- continual learning에서 representation drift 측정
- architecture 변경 영향이 시작되는 layer 탐색

### 결과 해석

- 높은 CKA: sample 관계가 비슷한 표현
- 낮은 CKA: 내부 표현 구조가 크게 다름

### 한계

- CKA가 높아도 판단 원인과 기능이 같다고 보장하지 않는다.
- CKA가 낮다고 한쪽이 더 나쁘다는 뜻도 아니다.
- probe, 출력 차이, intervention과 함께 해석하는 편이 안전하다.

---

## 2.2 Hessian 상위 고유값과 Spectrum

### 왜 필요한가

Gradient가 현재의 기울기를 알려준다면 Hessian은 주변 지형의 휘어짐을 알려준다.

Loss의 Hessian은

$$
H(\theta)
=
\nabla_\theta^2L(\theta)
$$

이다.

Parameter가 매우 많으므로 $H$ 전체를 만들지 않고, 임의의 vector $v$에 대한

$$
H(\theta)v
$$

인 Hessian-vector product를 계산해 power iteration이나 Lanczos 방법을 적용한다.

### 실제 측정량

- 최대 고유값 $\lambda_{\max}$
- 최소 고유값 $\lambda_{\min}$
- 큰 고유값 몇 개
- 음의 고유값 존재 여부
- 0 근처 고유값의 비율
- spectral density

### 주요 해석

| 관측 | 가능한 의미 |
|---|---|
| 큰 양의 고유값 | 특정 방향의 계곡벽이 매우 가파름 |
| 많은 0 근처 고유값 | 넓은 flat direction 또는 parameter symmetry |
| 음의 고유값 | 내려갈 수 있는 방향이 남은 saddle 가능성 |
| 고유값 범위가 매우 넓음 | 방향별 scale 차이가 큰 ill-conditioned valley |
| $\lambda_{\max}$가 learning rate에 비해 큼 | overshooting·진동 가능성 |

### 주요 사용 상황

- learning rate 안정성 분석
- plateau와 saddle 구분
- optimizer 비교
- sharpness 연구
- pruning·quantization 민감도
- edge-of-stability 분석

### 한계

Raw Hessian과 sharpness는 parameterization에 민감하다. 같은 함수를 나타내더라도 weight rescaling에 따라 값이 달라질 수 있다.

따라서 다음 조건을 통제해야 한다.

- 동일 architecture
- 동일 normalization
- 동일 loss definition
- 동일 데이터와 batch
- 가능한 경우 layer/filter normalization
- 함수공간 출력 비교 병행

---

# 3. Plateau·Saddle·Local Minimum의 구분

현재 checkpoint를 $\theta_\ast$라고 하자.

## 3.1 Plateau

Plateau는 주변의 많은 방향에서 loss 변화가 작아 학습이 느려지는 영역이다.

대표적 신호:

$$
\|\nabla L(\theta_\ast)\|
\approx 0
$$

이면서 Hessian의 많은 고유값이 0 근처에 있다.

그러나 plateau는 minimum과 같은 뜻이 아니다. 일부 특수 방향으로는 여전히 내려갈 수 있다.

---

## 3.2 Saddle point

Saddle은 어떤 방향으로는 loss가 증가하고 다른 방향으로는 감소하는 지점이다.

대표적 신호:

$$
\|\nabla L(\theta_\ast)\|
\approx 0,
\qquad
\lambda_{\min}\bigl(H(\theta_\ast)\bigr)<0.
$$

음의 고유값에 대응하는 eigenvector 방향으로 작은 이동을 주었을 때 loss가 감소할 수 있다.

---

## 3.3 Local minimum

Local minimum은 충분히 가까운 모든 방향에서 loss가 더 낮아지지 않는 지점이다.

이상적인 2차 조건은

$$
\nabla L(\theta_\ast)=0,
\qquad
H(\theta_\ast)\succeq0
$$

이다.

하지만 실제 신경망에서는 다음 이유로 엄밀한 판정이 어렵다.

- parameter 수가 매우 많음
- 0에 가까운 고유값이 많음
- neuron permutation과 rescaling symmetry가 존재
- 모든 방향을 조사할 수 없음
- mini-batch loss와 전체 데이터 loss가 다를 수 있음

따라서 실제 보고에서는 보통 다음처럼 제한적으로 표현한다.

> 조사한 gradient, Hessian 고유방향, 무작위 perturbation 범위에서는 추가적인 하강 방향을 발견하지 못했다.

---

## 3.4 Basin

Basin을 다루려면 먼저 허용할 loss 수준 $c$를 정한다.

$$
S_c
=
\{\theta\mid L(\theta)\leq c\}.
$$

이 집합이 필요한 이유는 “같은 basin”을 단순 parameter 거리 대신 **낮은 loss를 유지하며 이동할 수 있는 연결성**으로 해석하기 위해서다.

두 checkpoint가 $S_c$의 같은 연결성분 안에 있다면, 둘 사이를 잇는 저손실 경로가 존재한다.

---

# 4. Plateau·Saddle·Local Minimum을 실제로 분석하는 순서

## 4.1 1단계: 학습 궤적의 요약 통계

전체 학습경로를 HJE나 Fokker–Planck 방정식으로 직접 풀기보다, checkpoint와 step마다 관측량을 기록한다.

### 기록할 항목

- train loss와 validation loss
- gradient norm
- layerwise gradient norm
- update norm
- update-to-weight ratio
- 연속 gradient cosine
- batch별 gradient variance
- parameter displacement
- clipping 비율
- optimizer momentum 또는 adaptive scale

### 정체 상황별 1차 해석

| 관측 | 우선 의심할 것 |
|---|---|
| Loss 정체 + gradient도 작음 | plateau, saturation, stationary region |
| Loss 정체 + gradient는 큼 | 방향 진동, optimizer scaling, clipping |
| Gradient 방향이 자주 반전 | 좁은 계곡에서 좌우 진동 |
| Batch마다 gradient가 크게 다름 | gradient noise가 큰 상태 |
| 특정 layer update가 거의 0 | gradient flow 또는 optimizer scaling 문제 |
| Train loss는 감소하지만 validation은 악화 | 일반화 문제이지 지형 정체만의 문제는 아님 |

---

## 4.2 2단계: Hessian으로 국소 지형 검사

현재 checkpoint에서 다음을 추정한다.

1. $\lambda_{\max}$
2. $\lambda_{\min}$
3. 상위 양의 고유값
4. 가능한 범위에서 spectral density
5. 0 근처 고유값 비율

### 판정 예시

#### Plateau 후보

$$
\|\nabla L\|\approx0
$$

이며 대부분의 측정 고유값이 0 근처다.

#### Saddle 후보

$$
\|\nabla L\|\approx0
$$

이지만 $\lambda_{\min}<0$이다.

#### 좁은 계곡 후보

$\lambda_{\max}$는 매우 크고, 다른 방향은 완만하다. 연속 gradient cosine이 음수로 자주 바뀌는 현상이 함께 나타날 수 있다.

#### Local minimum 후보

Gradient가 작고 조사한 범위에서 음의 곡률이 발견되지 않는다.

단, 이는 엄밀한 증명이 아니라 **측정 범위 내 판정**이다.

---

## 4.3 3단계: 방향별로 주변을 직접 찔러보기

방향 $v$를 정하고

$$
\varepsilon
\longmapsto
L(\theta_\ast+\varepsilon v)
$$

를 여러 $\varepsilon$에서 계산한다.

### 사용할 방향

- 무작위 정규화 방향
- 현재 gradient 방향
- 실제 optimizer update 방향
- Hessian 최대 고유벡터
- Hessian 최소 고유벡터
- 최근 trajectory의 PCA 방향

### 왜 여러 방향이 필요한가

고차원 공간에서 내려갈 수 있는 방향이 소수라면 무작위 방향만으로는 발견하기 어렵다.

예를 들어 무작위 방향에서는 모두 loss가 증가하지만 최소 고유벡터 방향에서는 감소한다면, 그 지점은 minimum처럼 보이는 saddle일 수 있다.

### 기록할 항목

- 방향 종류
- perturbation 크기 $\varepsilon$
- train loss 변화
- validation loss 변화
- 출력 disagreement
- accuracy 또는 task metric 변화

---

## 4.4 4단계: 재시작과 탈출 실험

같은 checkpoint에서 다음 조건을 하나씩 바꾸어 다시 학습한다.

- learning rate를 일시적으로 증가
- learning rate를 감소
- batch size를 감소해 noise 증가
- momentum 변경
- optimizer 변경
- 작은 parameter noise 추가
- 여러 seed로 재시작
- 일부 layer만 재초기화

### 관측할 항목

- 더 낮은 loss에 도달하는가?
- 같은 영역으로 되돌아오는가?
- 탈출까지 걸린 step 수
- 어떤 방향으로 이동하는가?
- validation 성능도 함께 좋아지는가?
- 조건을 원래대로 돌리면 효과가 유지되는가?

### 해석

- 작은 perturbation에도 반복적으로 더 낮은 loss로 이동  
  → 정체가 안정된 minimum이 아닐 가능성
- noise를 크게 주어야만 이동  
  → barrier가 있는 basin일 가능성
- 여러 재시작이 비슷한 함수와 성능으로 복귀  
  → 넓은 안정 영역 또는 강한 implicit bias 가능성
- train loss는 낮아졌지만 validation이 악화  
  → 더 낮은 training minimum이 반드시 더 좋은 해는 아님

---

## 4.5 5단계: Checkpoint 사이의 연결성과 Basin 분석

두 checkpoint를 $\theta_A,\theta_B$라고 한다.

### 직선 보간

$$
\theta(\alpha)
=
(1-\alpha)\theta_A+\alpha\theta_B,
\qquad
0\leq\alpha\leq1.
$$

각 $\alpha$에서 loss를 측정한다.

Barrier height는 예를 들어

$$
B(\theta_A,\theta_B)
=
\max_{\alpha\in[0,1]}
L\bigl(\theta(\alpha)\bigr)
-
\max\{L(\theta_A),L(\theta_B)\}
$$

로 요약할 수 있다.

### 해석

- $B$가 작음: 직선으로도 호환되는 저손실 영역
- $B$가 큼: 직선경로에는 장벽이 있음

### 곡선 경로

직선에 장벽이 있어도 실제 low-loss region이 분리됐다고 단정할 수 없다. 중간 제어점을 가진 곡선 경로를 최적화해 low-loss path를 찾는다.

### 반드시 확인할 것

- neuron permutation alignment
- BatchNorm 통계 재계산
- train loss와 validation loss 모두 측정
- interpolation 전후 출력 disagreement
- 서로 다른 seed와 checkpoint 반복

### 결론의 한계

낮은 loss 경로를 찾으면 연결성에 대한 강한 실험적 증거가 된다.  
반대로 경로를 찾지 못했다고 해서 분리된 basin이라는 수학적 증명이 되는 것은 아니다.

---

# 5. 상황별 통합 진단표

## 5.1 Loss가 평평해 보이고 거의 줄지 않는다

### 우선 측정

1. Gradient norm
2. Layerwise gradient norm
3. Update-to-weight ratio
4. Activation variance
5. Hessian의 $\lambda_{\min}$과 0 근처 spectrum
6. Hessian 최소 고유벡터 방향 perturbation

### 구분

- Gradient와 update 모두 작고 음의 곡률 없음  
  → plateau 또는 local-minimum 후보
- Gradient는 작지만 음의 곡률 존재  
  → saddle 후보
- Gradient는 큰데 update가 작음  
  → optimizer scaling·clipping 문제
- Activation도 붕괴  
  → 단순 지형 문제보다 표현 붕괴 가능성

---

## 5.2 Loss가 크게 진동한다

### 우선 측정

1. Gradient norm
2. 연속 gradient cosine
3. Update norm
4. Hessian 최대 고유값
5. Batch별 gradient variance

### 해석

- $\lambda_{\max}$가 크고 gradient cosine이 반복적으로 음수  
  → 좁고 가파른 계곡에서 overshooting 가능성
- Batch variance가 매우 큼  
  → stochastic noise가 진동을 지배할 가능성
- 특정 layer만 update가 큼  
  → layerwise learning-rate imbalance 가능성

---

## 5.3 서로 다른 seed의 모델을 합치고 싶다

### 우선 측정

1. 출력 disagreement
2. CKA
3. Parameter permutation alignment
4. Linear interpolation loss
5. 곡선 mode connectivity
6. Weight averaging 뒤 validation 성능

### 해석

- CKA와 출력이 비슷하고 직선 barrier도 낮음  
  → 단순 평균이 성공할 가능성이 비교적 높음
- 출력은 비슷하지만 barrier가 큼  
  → permutation 또는 parameter symmetry 확인
- 곡선으로만 연결됨  
  → 같은 넓은 저손실 연결영역일 수 있지만 단순 평균은 실패할 수 있음

---

## 5.4 새 optimizer가 plateau를 더 잘 탈출한다고 주장하고 싶다

### 최소 검증 세트

1. 동일한 초기 checkpoint
2. 동일 데이터 순서 또는 반복 seed
3. Gradient norm과 update norm
4. Hessian 최소 고유값
5. 탈출까지 걸린 step 수
6. 도달한 최종 train·validation 성능
7. Parameter displacement
8. 재시작 반복
9. Learning rate·batch size matched control

### 필요한 결론

단순히 더 빨리 loss가 줄었다는 것만으로는 부족하다.

- 실제로 saddle 방향을 더 잘 이용했는가?
- 단순히 step size가 더 컸던 것은 아닌가?
- validation 성능도 좋아졌는가?
- 계산비용을 맞춘 뒤에도 차이가 남는가?

를 함께 확인해야 한다.

---

# 6. 권장 검증 스택

## 6.1 기본 스택

일반적인 YOLO·MobileNet·소형 Transformer 실험에서는 다음부터 시작한다.

1. 성능과 seed 반복
2. Train·validation curve
3. Gradient norm과 layerwise gradient
4. Update-to-weight ratio
5. Activation 평균·분산·0 비율
6. Covariance spectrum과 effective rank
7. Linear probe

---

## 6.2 연구 표준 확장

최적화나 표현 변화가 핵심 질문일 때 추가한다.

8. CKA
9. Hessian 최대·최소 고유값
10. Hessian top spectrum
11. 방향별 perturbation
12. Checkpoint interpolation
13. Mode connectivity

---

## 6.3 Plateau·basin 정밀 분석

정체 구간 자체가 연구 질문일 때 추가한다.

14. 최근 trajectory PCA
15. 여러 크기의 noise 재시작
16. 탈출시간 분포
17. Batch size와 learning rate 변화 실험
18. 같은 checkpoint에서 optimizer matched comparison
19. Permutation-aligned basin 연결성
20. 함수공간 출력 거리

---

# 7. 증거 강도

## 7.1 관찰적 증거

예:

- gradient norm이 작다.
- Hessian에 0 근처 고유값이 많다.
- CKA가 높다.
- interpolation barrier가 낮다.

이는 현상을 측정한 것이지만 원인을 확정하지는 않는다.

---

## 7.2 판별적 증거

예:

- 최소 Hessian 고유벡터 방향으로 perturbation했을 때만 loss가 감소한다.
- learning rate를 줄이자 gradient 반전과 loss 진동이 동시에 감소한다.
- permutation alignment 뒤 interpolation barrier가 사라진다.

이는 경쟁 가설을 일부 구분한다.

---

## 7.3 개입적 증거

예:

- 특정 방향이나 layer를 차단하자 탈출이 사라진다.
- 동일 checkpoint에서 optimizer만 바꾸었을 때 탈출시간이 반복적으로 감소한다.
- 특정 activation을 복구하자 representation rank와 task 성능이 함께 회복된다.

이는 해당 요소가 실제 동작에 필요하다는 더 강한 근거다.

---

# 8. 최소 실험 기록 항목

Plateau·saddle·basin 분석 결과를 재현하려면 다음을 남긴다.

- dataset version과 split
- code commit
- random seed
- model architecture
- checkpoint
- optimizer state
- scheduler state
- batch 순서
- learning rate
- batch size
- mixed precision 설정
- gradient clipping 설정
- Hessian 계산에 사용한 데이터
- Lanczos·power iteration 설정
- perturbation 방향과 크기
- interpolation 점 개수
- BatchNorm 통계 처리 방식
- train·validation metric
- 계산시간과 GPU memory

---

# 9. 핵심 요약

좁은 의미의 표준 진단은 다음이다.

$$
\boxed{
\text{Gradient 통계}
+
\text{Activation 통계}
+
\text{Covariance spectrum}
+
\text{Linear probe}
}
$$

최적화·표현 연구에서 연구 표준에 가까운 방법은 다음이다.

$$
\boxed{
\text{CKA}
+
\text{Hessian 상위 고유값과 spectrum}
}
$$

Plateau·saddle·local minimum·basin은 다음 순서로 분석한다.

$$
\boxed{
\text{학습 궤적 통계}
\rightarrow
\text{Hessian 국소기하}
\rightarrow
\text{방향별 perturbation}
\rightarrow
\text{재시작·탈출 실험}
\rightarrow
\text{checkpoint 연결성}
}
$$

핵심은 하나의 2D loss 그림으로 지형을 단정하지 않는 것이다.

- Gradient는 현재 움직임을 본다.
- Activation과 spectrum은 표현 상태를 본다.
- Hessian은 현재 위치 주변의 곡률을 본다.
- Perturbation은 실제 하강 방향이 존재하는지 시험한다.
- 재시작은 정체 영역에서 탈출 가능한지를 본다.
- Mode connectivity는 서로 다른 해 사이의 저손실 연결성을 본다.

이 관측들을 함께 사용해야 plateau, saddle, local minimum, basin을 비교적 설득력 있게 구분할 수 있다.
