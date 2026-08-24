# 딥러닝 성능 병목 특정 및 성능 향상 워크플로

## 1. 전체 구조

이 워크플로의 출발점은 파라미터 공간이나 표현공간 같은 추상적 원인이 아니라, **baseline에서 실제로 관측되는 성능 상황**이다.

$$
\boxed{
\text{성능 상황 지도}
\rightarrow
\text{병목 위치 특정}
\rightarrow
\text{경쟁 원인가설}
\rightarrow
\text{값싼 판별 실험}
\rightarrow
\text{개선법 선별}
\rightarrow
\text{정식 ablation}
\rightarrow
\text{XAI/MI 기전 검증}
}
$$

핵심 목표는 단순히 실패 사례를 찾는 것이 아니라 다음을 밝히는 것이다.

> 현재 baseline에서 가장 큰 잔여 개선량(headroom)이 어디에 있으며, 어떤 개입이 그 병목에 실제로 맞는가?

---

# 2. 단계별 워크플로

## 0단계. 비교 가능한 baseline 고정

성능 개선을 주장하려면 먼저 비교 기준이 흔들리지 않아야 한다.

고정할 항목:

- 데이터 split과 평가 코드
- 학습 및 추론 compute budget
- checkpoint 선택 규칙
- 입력 해상도와 augmentation
- optimizer, scheduler, batch size
- seed와 반복 횟수
- 핵심 품질 metric
- latency, FLOPs, parameter 수, memory 등 비용 metric

### 산출물: Baseline Card

- 전체 성능
- 조건별 성능
- train/validation 학습곡선
- 학습 및 추론 비용
- seed별 변동
- 사용한 config와 checkpoint 규칙

---

## 1단계. Baseline의 성능 지형 작성

이 단계에서는 아직 내부 원인을 단정하지 않는다.  
먼저 **어느 방향에 점수가 남아 있는지**를 본다.

## 1.1 학습 진행 상태

확인할 질문:

- train과 validation 성능이 아직 모두 개선 중인가?
- train은 포화됐지만 validation만 뒤처지는가?
- epoch를 늘렸을 때 실제 개선이 계속되는가?
- seed마다 최종 성능 차이가 큰가?
- 학습 후반에 성능이 오히려 퇴화하는가?

대략적으로 다음 headroom을 구분할 수 있다.

- 학습예산 또는 최적화
- 일반화
- 학습 불안정성
- objective와 평가 metric의 불일치

---

## 1.2 태스크 고유 축별 성능

평균 점수만 보지 않고, 실제 태스크 구조에 맞는 축으로 성능을 분해한다.

### 객체 탐지 예시

- 객체 크기별 $AP_S, AP_M, AP_L$
- class별 AP
- classification과 localization
- 가림 정도
- 조명
- 해상도
- 객체 밀도
- confidence 구간별 성능

### 분류 예시

- class별 성능
- 난이도별 성능
- confidence별 정확도
- 배경, 스타일, source별 성능
- 이미지 품질별 성능

목표는 오류를 열거하는 것이 아니라 다음을 구하는 것이다.

> 어느 조건의 성능을 올리면 최종 leaderboard 점수가 가장 크게 움직이는가?

---

## 1.3 자원 증가에 대한 반응

다음 축을 2~4점 정도만 바꾸어 marginal gain을 본다.

- 데이터 양
- 모델 크기
- 입력 해상도
- 학습 step
- augmentation 강도
- loss weight

예시:

$$
25\% \rightarrow 50\% \rightarrow 75\% \rightarrow 100\%
$$

해석 예시:

- 데이터 증가에 계속 반응  
  → 데이터·일반화 쪽 headroom
- 해상도에만 크게 반응  
  → 공간 정보 또는 feature resolution 병목 후보
- 모델 크기에만 반응  
  → capacity 병목 후보
- 모든 축에서 포화  
  → objective, label 구조, architecture 선택 자체를 재검토할 가능성

---

## 1.4 Component Oracle Gap

복합 모델이나 pipeline에서는 각 구성요소를 정답으로 교체했을 때 점수가 얼마나 오르는지 본다.

$$
\Delta_j
=
S(\text{component } j \text{만 oracle})
-
S(\text{baseline})
$$

예시:

- 예측 class를 GT class로 교체
- 예측 box를 GT box로 교체
- tracking association을 GT로 교체
- retrieval 결과를 GT 문서로 교체
- predicted state를 GT state로 교체

$\Delta_j$가 큰 구성요소일수록 우선적인 개선 대상이다.

### 1단계 결과 예시

```text
1순위 병목: 작은 객체 localization
2순위 병목: 저조도 조건 일반화
보류: 대형 객체 classification
```

---

## 2단계. 병목마다 경쟁 원인가설 구성

하나의 관측된 병목에는 여러 원인이 가능하다.

예를 들어 작은 객체 성능이 낮다면:

```text
H1. 입력 또는 초기 downsampling에서 정보가 소실된다.
H2. 작은 객체 feature는 존재하지만 detection head가 활용하지 못한다.
H3. 큰 객체가 loss와 gradient를 지배한다.
H4. 배경이나 texture를 shortcut으로 사용한다.
H5. 작은 객체 담당 경로가 충분히 업데이트되지 않는다.
```

좋은 가설은 서로 다른 관측 결과를 예측해야 한다.

| 가설 | 예측되는 신호 |
|---|---|
| 초기 정보 소실 | 해상도·stride 변경에 즉시 반응 |
| Head 병목 | backbone feature probe는 좋지만 최종 출력만 낮음 |
| Loss 불균형 | 큰 객체 sample의 gradient 기여가 과도함 |
| Shortcut | 배경 교체 시 출력이 크게 흔들림 |
| 학습동역학 | 특정 stage의 gradient·update가 지속적으로 작음 |

이 단계의 핵심은 다음 형식이다.

$$
\boxed{
\text{가설}
\Rightarrow
\text{예측 신호}
\Rightarrow
\text{판별 실험}
}
$$

---

## 3단계. 값싼 판별 실험

전체 학습을 여러 번 돌리기 전에, 가설을 빠르게 탈락시키는 실험을 수행한다.

## 3.1 국소 개입

한 번에 하나의 요소만 바꾼다.

예시:

- 입력 해상도 한 단계 증가
- 특정 stride 제거
- 특정 feature level 추가 또는 제거
- loss weight만 변경
- backbone freeze/unfreeze
- head만 교체
- 특정 augmentation만 on/off

목표는 최고 점수를 얻는 것이 아니라:

> 이 병목이 해당 축에 실제로 반응하는가?

를 확인하는 것이다.

---

## 3.2 Oracle과 Counterfactual

### Oracle

구성요소별 최대 개선 가능량을 본다.

- GT box
- GT class
- GT retrieval
- GT association

### Counterfactual

정답 의미는 유지하면서 의심되는 nuisance만 바꾼다.

- 객체는 유지하고 배경 교체
- 위치만 이동
- texture만 제거
- blur만 변경
- 조명만 변경
- 색상만 변경
- 문장 표현만 paraphrase

출력이 크게 변하면 shortcut 또는 잘못된 invariance 후보가 된다.

---

## 3.3 XAI: 입력과 개념 의존성 진단

XAI는 다음 질문에 사용한다.

> 모델이 어느 입력 영역과 사람이 정의한 개념에 의존하는가?

대표 도구:

- Occlusion test
- Perturbation test
- Counterfactual test
- Grad-CAM
- Saliency map
- TCAV/CAV
- scale/layer별 attribution

### XAI의 역할

- 작은 객체 영역에 반응이 존재하는가?
- 객체보다 배경을 더 참고하는가?
- 특정 색, 질감, 위치에 과민한가?
- 어느 FPN level이 어떤 객체 크기에 기여하는가?

주의할 점:

- saliency 그림만으로 인과성을 주장하지 않는다.
- 가능하면 occlusion, counterfactual, randomization sanity check와 함께 사용한다.
- XAI는 개선법 후보를 만드는 탐색 도구로 사용한다.

---

## 3.4 저비용 내부 진단

MI로 바로 올라가기 전에 다음을 먼저 본다.

- layerwise linear probe
- 특정 feature level 제거
- backbone/head 분리 평가
- layer별 gradient norm
- layer별 update-to-weight ratio
- loss 항별 gradient 기여도

예를 들어 layer $\ell$의 update-to-weight ratio는 다음처럼 본다.

$$
r_\ell(t)
=
\frac{
\lVert \Delta\theta_\ell(t)\rVert
}{
\lVert\theta_\ell(t)\rVert+\varepsilon
}
$$

---

## 3.5 MI: 내부 경로의 인과성 진단

다음 상황에서 MI로 올라간다.

> 병목 위치와 입력 의존성 후보는 좁혀졌지만, 어떤 내부 구성요소가 실제 성능을 만드는지 확인해야 한다.

대표 도구:

- layer ablation
- channel ablation
- head ablation
- block ablation
- activation patching
- activation replacement
- causal tracing
- path intervention

### MI의 역할

- 어떤 내부 경로가 개선분에 필수적인가?
- 그 경로를 차단하면 개선분이 사라지는가?
- 개선 모델의 activation을 baseline에 patch하면 출력이 회복되는가?
- 특정 구성요소가 단순히 상관된 것이 아니라 인과적으로 필요한가?

### 3단계 결과 예시

```text
H1 지지: 해상도와 stride 변경에 선택적으로 반응
H2 약화: backbone probe도 낮음
H3 부분 지지: 작은 객체 gradient 비중이 낮음
H4 기각: 배경 counterfactual 영향 작음
```

---

## 4단계. 병목과 맞는 개선법 후보 선정

진단된 병목과 직접 대응하는 방법만 후보로 올린다.

| 관측 병목 | 개선법 후보 |
|---|---|
| 초기 공간정보 손실 | stride 조정, 고해상도 입력, P2/FPN |
| receptive field 부족 | dilation, larger kernel, context module |
| loss·sample 기여 불균형 | reweighting, focal 계열, sampling |
| 일반화 headroom | augmentation, regularization, pretraining |
| optimization 정체 | optimizer, LR schedule, normalization |
| shortcut 의존 | counterfactual augmentation, invariance constraint |
| head 병목 | head 구조, objective, assignment 개선 |

중요한 원칙:

> 방법을 먼저 고르고 효과를 찾지 않는다.  
> 진단된 병목과 일대일로 연결되는 방법군만 후보로 올린다.

---

## 5단계. Low-Fidelity Screening

유망 후보를 full-budget 실험으로 올리기 전에 낮은 비용으로 선별한다.

사용할 수 있는 방법:

- 전체 epoch의 일부만 학습
- 병목 slice를 보존한 축소 데이터셋
- 작은 hyperparameter sweep
- 한 stage만 변경
- 짧은 freeze/unfreeze 실험
- 한두 seed로 방향성 확인

이 단계에서 확인할 것:

- 병목 metric이 실제로 반응하는가?
- 전체 성능이 심하게 악화되지 않는가?
- 학습이 불안정하지 않은가?
- 추가 비용이 허용 가능한가?
- 예상한 내부 신호가 움직이는가?

주의:

- short-run 순위가 final-run 순위와 같다고 가정하지 않는다.
- screening은 탈락용이고 최종 증명용이 아니다.

---

## 6단계. 정식 통제 실험

최소한 다음 실험 구조를 갖춘다.

| 실험 | 의미 |
|---|---|
| Baseline | 기준 |
| Baseline + 제안법 | 전체 효과 |
| 핵심 요소 제거 | 어떤 요소가 실제 기여했는가 |
| Parameter/FLOPs matched control | 단순 용량 증가 효과 제거 |
| 강도 sweep | 우연한 한 설정인지 확인 |
| 여러 seed | 개선 폭이 실행 변동보다 큰지 확인 |
| 핵심 slice + 전체 metric | 병목 개선과 부작용 확인 |
| 비용 지표 | 실용적인 trade-off 확인 |

### 두 요소의 상호작용

두 요소 $A,B$를 함께 사용할 경우 $2\times2$ ablation을 수행한다.

|  | $B$ 없음 | $B$ 있음 |
|---|---:|---:|
| $A$ 없음 | Baseline | $B$ |
| $A$ 있음 | $A$ | $A+B$ |

이를 통해 다음을 구분한다.

- 단순 합산 효과
- 상호작용 효과
- 한 요소가 다른 요소가 있을 때만 작동하는 경우

---

## 7단계. XAI·MI 사후 기전 검증

점수가 올랐다는 사실만으로 개선 원인이 확인된 것은 아니다.

## 7.1 Architecture 개선

확인할 항목:

- 병목 layer 전후 linear probe
- feature rank 또는 spectrum
- baseline과 개선 모델의 CKA
- 특정 feature level 또는 경로 ablation

## 7.2 Loss·Sampling 개선

확인할 항목:

- sample군별 gradient 기여도
- loss 항 간 gradient cosine
- margin과 confidence
- hard/easy sample별 개선량
- 목표 slice의 gradient budget 변화

## 7.3 XAI 사후 검증

- 목표 객체 영역 attribution 증가
- 불필요한 배경 의존 감소
- counterfactual 안정성 증가
- TCAV 개념 민감도 변화
- 중요 영역 occlusion 시 출력 변화 증가

## 7.4 MI 사후 검증

- 새 경로를 ablate하면 개선분이 사라지는가?
- 개선 activation을 baseline에 patch하면 출력이 회복되는가?
- matched placebo에서는 같은 현상이 나타나지 않는가?
- 핵심 경로가 필요할 뿐 아니라 충분한가?

---

## 8단계. 새 baseline으로 갱신하고 반복

개선 성공 기준:

$$
\boxed{
\begin{aligned}
&\text{① 목표 병목 metric이 개선되고}\\
&\text{② 전체 metric도 유지 또는 개선되며}\\
&\text{③ 예측했던 내부·입력 의존성이 바뀌고}\\
&\text{④ 비용과 seed 통제 후에도 효과가 남는다.}
\end{aligned}
}
$$

조건을 만족하면 개선 모델을 새 baseline으로 삼고 성능 지형을 다시 그린다.

이전 병목이 줄어들면 다른 병목이 새롭게 지배적이 될 수 있다.

---

# 3. 종합 흐름도

```text
[0] Baseline과 평가 프로토콜 고정
                ↓
[1] 성능 지형 작성
    학습곡선 / slice / scaling / oracle gap
                ↓
[2] 가장 큰 잔여 headroom 선택
                ↓
[3] 경쟁 원인가설 2~4개 설정
                ↓
[4] 값싼 판별
    국소 개입 / oracle / counterfactual
    XAI / probe / gradient
                ↓
[5] 필요할 때만 MI로 내부 경로 특정
                ↓
[6] 병목에 맞는 개선법 후보 선정
                ↓
[7] short-run·subset으로 후보 screening
                ↓
[8] full-budget matched-control experiment
                ↓
[9] component·strength·interaction ablation
                ↓
[10] XAI/MI로 개선 기전 사후 검증
                ↓
[11] 새 baseline 갱신 후 반복
```

---

# 4. 포트폴리오 구성

## 4.1 Baseline Characterization

보여줄 것:

- 전체 metric
- 태스크 고유 slice
- 학습곡선
- scaling 반응
- oracle gap
- 비용 지표

핵심 질문:

> 어디에 가장 큰 headroom이 남아 있는가?

---

## 4.2 Bottleneck Localization

예시:

> 작은 객체 성능 저하 중 classification보다 localization oracle gap이 더 컸고, 해상도 증가와 초기 stride 변경에 선택적으로 반응했다.

즉 병목의 위치를 정량적으로 특정한다.

---

## 4.3 Competing Hypotheses

가능한 원인을 여러 개 제시하고 각 가설이 예측하는 신호를 적는다.

예시:

- 초기 공간정보 소실
- detection head 활용 실패
- loss imbalance
- shortcut
- optimizer update 부족

---

## 4.4 Discriminating Diagnostics

가설을 서로 탈락시키는 실험:

- local intervention
- oracle
- counterfactual
- XAI
- linear probe
- gradient/update 분석
- 필요할 때 MI

---

## 4.5 Method Selection

진단된 병목과 직접 대응하는 개선법을 선택한다.

예시:

```text
병목: 초기 해상도 손실
선택: P2 feature 추가 + 초기 stride 조정
제외: 단순 backbone 확대
이유: parameter-matched control보다 scale-specific 반응이 컸음
```

---

## 4.6 Controlled Ablation

포함할 것:

- baseline
- 제안법
- 핵심 구성요소 제거
- matched control
- strength sweep
- interaction test
- 여러 seed
- 비용 비교

---

## 4.7 Mechanism Validation

성능 상승뿐 아니라 다음을 확인한다.

- 예상한 feature 경로가 실제로 활성화됐는가?
- 목표 slice에서만 선택적으로 개선됐는가?
- XAI에서 입력 의존성이 의도한 방향으로 변했는가?
- MI에서 해당 내부 경로가 인과적으로 필요한가?

---

# 5. 최종 포트폴리오 서사

좋지 않은 서사:

> 모델의 실패 사례를 분석하고 여러 기법을 적용해 점수를 올렸다.

좋은 서사:

> Baseline의 성능 지형을 slice, scaling, oracle gap으로 분해해 가장 큰 잔여 headroom을 특정했다. 여러 경쟁 원인가설을 국소 개입, counterfactual, XAI, probe, gradient 분석으로 판별했고, 병목에 직접 대응하는 개선법만 low-fidelity screening으로 선별했다. 이후 matched-control ablation과 seed 반복으로 개선 효과를 검증하고, XAI·MI를 이용해 예상한 내부 기전 변화까지 확인했다.

---

# 6. 핵심 요약

$$
\boxed{
\text{관측}
\rightarrow
\text{headroom}
\rightarrow
\text{병목}
\rightarrow
\text{경쟁 가설}
\rightarrow
\text{값싼 판별}
\rightarrow
\text{방법 선택}
\rightarrow
\text{정식 ablation}
\rightarrow
\text{XAI/MI 검증}
\rightarrow
\text{새 baseline}
}
$$

이 워크플로의 핵심은 다음 세 가지다.

1. 추상적인 내부 원인보다 관측 가능한 성능 상황에서 출발한다.
2. 최신 기법을 먼저 적용하지 않고, 병목과 맞는 방법만 후보로 올린다.
3. 점수 상승뿐 아니라 matched control과 XAI/MI를 통해 개선 기전까지 검증한다.
