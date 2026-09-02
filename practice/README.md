# PyTorch Micro-Practice Notebooks

이 폴더는 작은 텐서와 바닐라 PyTorch를 사용해 현대 딥러닝 구조와 학습 알고리즘을 미시적으로 분해해 보는 실습 모음이다.

각 노트북은 기본적으로 다음 순서를 따른다.

1. 바닐라 PyTorch 코드로 즉시 동작 확인
2. `torch.profiler`로 ATen/CUDA 연산 확인
3. PyTorch API가 수학을 숨기는 경우에만 작은 텐서로 explicit 계산 추가
4. 노트북 마지막에 기법 번호와 논문/모델 출처를 `References and provenance`로 정리

구성은 01~03 기초 노트북, 04~19 구조·알고리즘 그룹, 20~22 강화학습 그룹으로 이루어진다.
