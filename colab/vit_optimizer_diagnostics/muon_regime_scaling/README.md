# Muon regime restoration & scaling study

이 실험은 같은 SmallViT와 같은 Muon 구현을 사용했는데도 과거에는 validation accuracy가 약 0.77까지 올라간 반면, 현재 실행에서는 약 0.47에 머문 현상을 진단량으로 추적하기 위해 만든다.

## 왜 이 실험을 하는가

현재 실패 run은 train accuracy 자체가 약 0.44에 머물렀다. 따라서 우선 일반화 문제가 아니라 optimization / feature learning 단계의 실패로 본다.

현재 진단에서는 다음이 동시에 관측됐다.

- penultimate representation은 초기값에서 움직였지만 CKA-to-init가 과거 성공 run보다 높다.
- linear probe와 kNN purity가 낮다.
- NC1은 높고 mean margin은 음수다.
- Jacobian spectral norm은 매우 크고 participation rank는 낮다.
- Hessian의 최소/최대 Ritz가 모두 극단적으로 커져 매우 anisotropic한 local geometry를 보인다.

반면 Muon 구현과 주요 optimizer HP 자체는 유지되었다. 큰 차이는 학습 regime이다.

| regime | seed | train N | batch | epochs | approximate updates |
|---|---:|---:|---:|---:|---:|
| old success | 7 | 40,000 | 512 | 50 | ~3,900 |
| current failure | 42 | 10,000 | 256 | 50 | ~1,950 |

따라서 질서변수처럼 쓰는 진단량 자체를 버리는 것이 아니라, `m = m(N, B, update budget, seed | model, optimizer)`처럼 regime에 조건부인 관측량으로 해석한다.

## 파일

- `01_restore_old_muon_regime.ipynb`
  - 현재 진단 코드는 그대로 사용
  - seed / data split / batch / epoch를 old successful regime으로 복원
  - Muon 하나만 학습
  - accuracy뿐 아니라 dynamics → representation → class geometry → Jacobian → tangent kernel → Hessian → trajectory가 함께 정상화되는지 확인

- `02_muon_regime_scaling.ipynb`
  - 복원 여부를 확인한 뒤 한 번에 하나의 제어변수를 바꾸는 scaling 실험
  - `regime_endpoints`
  - `data_scale_fixed_updates`
  - `update_budget_at_40k`

- `muon_regime_runner.py`
  - 두 notebook이 공유하는 학습 및 진단 실행기
  - 메인 폴더의 기존 진단 함수를 그대로 호출한다.

- `REFERENCE.md`
  - 과거 성공 run과 현재 실패 run의 출처, commit, Google Drive reference, 핵심 수치

- `reference_results.csv`
  - notebook에서 바로 불러와 비교하기 위한 대조군 표

## 진단 연쇄

이 실험의 목적은 최종 accuracy 하나를 원인으로 취급하지 않고 아래 연쇄를 검증하는 것이다.

```text
regime control variables
    N / batch / update budget / seed
            ↓
optimizer dynamics
    gradient / update / noise / displacement
            ↓
feature movement
    CKA / effective rank
            ↓
task organization
    probe / kNN / NC1 / margin / manifold geometry
            ↓
function sensitivity
    Jacobian spectrum / tangent kernel
            ↓
local loss geometry
    Hessian Ritz / relative sharpness
            ↓
final train & validation performance
```

## 실험 순서

1. 먼저 `01_restore_old_muon_regime.ipynb`만 실행한다.
2. old regime으로 돌아갔을 때 accuracy와 내부 진단량이 함께 회복되는지 확인한다.
3. 회복되면 `02_muon_regime_scaling.ipynb`에서 `update_budget_at_40k`를 먼저 실행한다. N/B/seed가 고정되어 해석이 가장 단순하다.
4. 그 다음 `data_scale_fixed_updates`로 N 효과를 본다.
5. `regime_endpoints`는 old/current 전체 조건 차이를 재현하는 sanity check로 사용한다.

## 해석상 주의

한 번의 old-regime 복원만으로 N, batch, seed, update budget 중 어느 것이 원인인지 분리할 수 없다. 복원 실험은 먼저 'regime이 원인 후보인가'를 확인하는 단계이고, 그 다음 scaling 실험이 원인 분리를 담당한다.

또 `empirical_dichotomy_capacity`는 현재 run에서 floor에 가까웠으므로 그대로 기록하되 핵심 판정량으로 사용하지 않는다. 대신 radius / participation dimension / center-axis / axis-axis geometry를 함께 본다.
