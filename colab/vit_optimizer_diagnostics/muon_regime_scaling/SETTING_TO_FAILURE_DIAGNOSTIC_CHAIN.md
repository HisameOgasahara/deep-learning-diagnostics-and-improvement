# Muon 실험: 학습 설정 차이가 실패를 만드는 진단 연쇄

두 세팅을 그냥 “레짐이 다르다”로 뭉뚱그리지 말고, **설정 차이 → 실제 내부 상태 차이 → accuracy 차이**로 연결하면 이래.

과거 성공/복원 Muon과 실패 Muon에서 모델 구조와 Muon 하이퍼파라미터는 같고, 바뀐 건 주로 학습 데이터 규모·배치·seed·그에 따른 step 수였어.

| | 성공/복원 | 실패 |
|---|---:|---:|
| train 데이터 | 40k | 10k |
| batch | 512 | 256 |
| seed | 7 | 42 |
| epoch | 50 | 50 |
| 총 update | ≈3950 | ≈1950 |
| train acc | 0.885~0.923 | **0.443** |
| val acc | 0.746~0.769 | **0.467** |

그런데 `02`가 중요한 반례를 줬어. 성공 세팅에서 **약 1027 update만 해도 val 0.652**, 약 2054 update에서는 0.728이었어. 실패 세팅은 약 1950 update인데 0.467이었고.

따라서

```text
“step 수가 절반이라 실패했다”는 설명은 틀림
```

이야. 같은 정도의 step에서도 **40k/512/seed7 쪽의 동역학이 훨씬 낫다.**

그럼 실제로 무엇이 달라졌냐면, 먼저 **feature가 task에 맞게 조직되는 방식**이 달라졌어.

실패 → 복원에서

```text
CKA_init: 0.313 → 0.168
```

```text
linear probe: 0.4565 → 0.7558
```

였어. 복원된 probe 0.7558은 과거 성공 결과 0.758과 거의 같아.

즉 실패 Muon은 파라미터나 표현이 전혀 안 움직인 게 아니라, **초기 표현에서 충분히 task-specific한 방향으로 재구성되지 못했어.**

그 다음 단계가 더 선명해. 클래스 구조가

```text
NC1: 4.01 → 1.62
```

```text
kNN purity: 0.295 → 0.686
```

```text
margin: -0.179 → +2.55
```

로 바뀌었어. 과거 성공은 대략 `NC1=1.32`, `kNN=0.707`, `margin=+2.89`였고.

그러니까 두 학습 세팅의 핵심 내부 차이는 이렇게 말할 수 있어.

```text
실패 세팅에서는 클래스별 feature manifold가 제대로 untangle되지 않았다
```

같은 클래스들이 충분히 뭉치지 않고, 다른 클래스와 충분히 떨어지지 않으니까 마지막에 decision boundary margin까지 음수가 된 거야.

그리고 그 representation 실패가 **함수의 민감도 구조**에서도 나타났어.

실패:

```text
||J||_2 = 59.9,    r_J = 1.82
```

복원:

```text
||J||_2 = 10.9,    r_J = 4.32
```

이었어.

이건 실패한 모델이 입력의 **몇 개 방향에 지나치게 강하게 반응하는 함수**가 됐다는 뜻이야. 성공 세팅에서는 민감도가 훨씬 작고 여러 방향으로 퍼져 있어.

parameter-space에서도 정확히 같은 이상 징후가 나와.

실패:

```text
lambda_min(H) ≈ -10,353
lambda_max(H) ≈ 16,827
```

복원:

```text
-112, +192
```

였어.

즉 실패 세팅에서는 Muon이 **극단적으로 휘어진 saddle-like 영역**으로 들어갔고, 성공 세팅은 `O(10^2)` curvature 영역에 있었어.

그래서 관측된 흐름은 이거야.

```text
학습 설정 변경
    ↓
Muon이 경험하는 stochastic gradient field 변화
    ↓
feature는 움직이지만 class-separable하게 조직되지 않음
    ↓
probe↓, NC1↑, kNN↓, margin<0
    ↓
함수 민감도가 소수 방향에 집중
(||J||_2↑, r_J↓)
    ↓
극단적인 Hessian curvature 영역
    ↓
train accuracy 자체가 44%에서 정체
```

이 연결은 현재 데이터로 **상당히 강하게 지지**돼.

중요한 반대 증거도 있어. Tangent target alignment는 실패 0.861, 복원 0.889로 별 차이가 없었어. 즉 실패 모델도 “어느 방향으로 target을 맞춰야 하는지”에 해당하는 tangent 구조를 완전히 잃은 건 아니야.

그래서 문제는

```text
target 방향을 못 찾음
```

보다는

```text
그 방향을 안정적인 feature geometry로 만들어내는 과정이 실패
```

쪽이야.

그리고 “설정 중 무엇이 물리적으로 이런 차이를 만들었느냐”는 한 단계 더 구체적으로 생각할 수 있어.

`N=40k → 10k`는 한 epoch에서 보는 **서로 다른 데이터 다양성**을 1/4로 줄여. `B=512 → 256`은 한 step의 gradient가 추정하는 empirical gradient의 표본 수도 절반으로 줄여서 gradient fluctuation 구조를 바꿔. 그러므로 실패 세팅에서는 같은 Muon이라도 매 step 받는 행렬 gradient의 방향·스펙트럼이 성공 세팅과 달라질 수 있어.

특히 Muon은 일반 SGD처럼 gradient magnitude를 그대로 따르기보다 matrix gradient를 Newton–Schulz로 orthogonalize해서 update를 만들잖아. 그래서 **gradient covariance/singular-direction 구조가 달라지면 update geometry도 달라질 가능성**이 있어.

다만 이 마지막 문장은 아직 **기전 가설**이야. 우리가 이번 실험에서 직접 측정한 건 그 결과인 representation/Jacobian/Hessian이고, `N`과 `B` 중 어느 것이 gradient spectrum을 어떻게 바꿨는지는 아직 직접 분리 측정하지 않았어.

그래서 현재 증거를 세 단계로 나누면:

## 거의 확인됨

```text
설정 변경 → 완전히 다른 학습 상태
```

그리고 그 상태 차이는

```text
representation organization
+ Jacobian geometry
+ Hessian geometry
```

에서 동시에 확인됨.

## 강하게 지지됨

실패의 직접적인 내부 병목은 **class-separable feature learning 실패**다.

## 아직 가설

40k→10k와 512→256이 Muon의 **gradient spectrum/noise geometry를 바꾸고, orthogonalized update 방향을 나쁜 regime으로 밀었다.**

사실 다음 공부거리로는 여기가 제일 좋아. 지금까지는

```text
원인 후보
→ gradient/update dynamics
→ representation
→ function/landscape
→ accuracy
```

에서 가운데 **representation 이후는 매우 잘 잡았는데**, `N,B → gradient/update dynamics` 부분의 직접 증거가 아직 약해.

다음 실험에서 **gradient covariance spectrum, Muon orthogonalization 전/후 singular spectrum, update–gradient cosine, layer별 update-to-weight**를 성공/실패 세팅에서 비교하면 이 연결고리까지 닫을 수 있어.
