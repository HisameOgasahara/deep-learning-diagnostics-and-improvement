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

## 3. 계산 구조 관찰 — 의미 있을 때

- profiler로 operator graph / CUDA kernel / memory 이동 등을 확인.
- 해당 방법의 장점이 계산효율과 직접 연결된다면 중요도가 올라감.
- 필요하면 작은 CUDA/Triton 구현까지 내려가서 **왜 빨라지는지** 직접 비교.
- 반대로 kernel 차이가 핵심이 아니면 생략.

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
  - 실패하는 조건에서 어떤 개량을 생각할 수 있는가?
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

이 원칙이면 Architecture, Loss, Training Recipe 모두 같은 철학으로 갈 수 있고, `inference_in_generation_and_control`만 sampling/trajectory/rollout 특성 때문에 관찰 대상이 조금 달라지는 정도로 둔다.
