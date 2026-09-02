# Muon regime 대조군 자료

이 폴더의 실험은 두 개의 이미 관측된 Muon 상태를 대조군으로 삼는다.

## 1. 과거 성공 Muon

- 과거 실행 노트북 커밋: `c779521402c2395d8196299f466c7bde75e7064c`
- 당시 결과 보관 Google Drive: https://drive.google.com/drive/folders/14k-Psa4vaXeAUdyxqpL-KKcsU4frTsLT?usp=sharing
- 학습 설정: seed 7, CIFAR-10 train 40,000, validation 5,000, batch 512, 50 epochs
- 대략적 optimizer update 수: 약 3,900

기존 실행 결과에서 기록해 둔 핵심 값:

| quantity | old successful Muon |
|---|---:|
| train accuracy | 0.9234 |
| validation accuracy | 0.7686 |
| validation loss | 약 0.800 |
| penultimate CKA to init | 약 0.130 |
| penultimate linear probe | 약 0.758 |
| NC1 | 약 1.319 |
| kNN purity | 약 0.707 |
| mean margin | 약 +2.894 |
| Hessian min Ritz | 약 -311 |
| Hessian max Ritz | 약 +304 |

이 값들은 `reference_results.csv`에도 기입한다. 새로 추가된 Jacobian, tangent kernel, relative sharpness 등은 당시 노트북에서 측정하지 않았으므로 빈 값으로 남긴다.

## 2. 현재 실패 Muon

현재 메인 노트북의 완주 결과를 대조군으로 둔다.

- seed 42
- train 10,000
- validation: CIFAR-10 test에서 2,000
- batch 256
- 50 epochs
- 대략적 optimizer update 수: 약 1,950

핵심 값:

| quantity | current failed Muon |
|---|---:|
| train accuracy | 0.4431 |
| validation accuracy | 0.4665 |
| validation loss | 1.490599 |
| penultimate CKA to init | 0.312605 |
| penultimate linear probe | 0.4565 |
| NC1 | 4.012914 |
| kNN purity | 0.2945 |
| mean margin | -0.179084 |
| Hessian min Ritz | -10353.101294 |
| Hessian max Ritz | 16827.431911 |
| Jacobian spectral norm | 59.892714 |
| Jacobian participation rank | 1.824303 |
| tangent-target alignment | 0.861102 |
| relative sharpness | 263.673537 |

## 3. 이 reference를 사용하는 방법

`01_restore_old_muon_regime.ipynb`는 **현재 진단 코드를 유지한 채 old successful Muon의 학습 regime만 복원**한다.

만약 accuracy와 함께 probe / kNN / NC1 / margin / Jacobian / Hessian이 old successful 쪽으로 되돌아가면, 현재 실패가 단순한 optimizer 구현 오류보다 regime 변화와 연결되어 있다는 증거가 강해진다.

그 다음 `02_muon_regime_scaling.ipynb`에서 데이터 수와 update budget을 한 축씩 바꾸어 어떤 외부 제어변수가 진단량의 상태 전환을 일으키는지 본다.
