# Anima Attention Diagnostics V2

이번 버전은 이전 V2에서 여러 축을 한꺼번에 합쳐 신호가 사라지는 문제를 피하기 위해 **V1 parity baseline을 먼저 복원**하고, token / block / denoising call을 분리해서 보도록 바꿨다.

## 핵심

- `batch_mode = v1_parity`가 기본값이다.
- 이 모드는 기존 `Temp/anima_inference_diagnostics_node.py`와 동일하게 selected token attention을 계산한 뒤 **batch 평균 + head 평균**을 한다.
- 각 token map은 block/call별로 따로 저장하며, PNG는 V1과 동일하게 map별 1–99 percentile normalization을 사용한다.
- `arona -> [27,28,29]`라면 세 token을 먼저 각각 저장한다. word map은 그 다음에 세 token raw map을 평균해서 별도로 만든다.
- 따라서 token 단계에서 캐릭터 신호가 있었는지, 어느 block/denoising call에서 사라지는지 직접 확인할 수 있다.

## 노드

### Anima Text Encode + Token Map V2
기존 positive text encode를 대신한다.

출력:
- `conditioning` -> 기존 positive conditioning 위치
- `token_map` -> `Anima Attention Diagnostics V2.token_map`
- `token_map` -> 필요하면 `Anima Token Map Viewer V2.token_map`

### Anima Attention Diagnostics V2
`MODEL -> Diagnostics -> sampler` 순서로 연결한다.

권장 첫 테스트:
- `attention_words = arona`
- `selected_blocks = 0,6,12,18,24,27`
- `batch_mode = v1_parity`
- `snapshot_every_n_calls = 1`

`conditional_only`는 V1 parity 확인이 끝난 뒤 비교용으로만 사용한다.

### Anima Attention Overlay V2
생성 이미지와 `diagnostic_directory`를 받는다.

`view_mode`:
- `subtokens_global`: `arona`의 27/28/29를 각각 block/call 전체 평균해서 **3장의 overlay**로 출력. 우선 이것부터 확인.
- `single_token_map`: 지정한 `block`, `call_index`의 token별 map을 그대로 overlay. 어느 timestep/block에서 캐릭터를 잡는지 찾을 때 사용.
- `word_global`: subtoken을 먼저 word로 평균한 뒤 전체 block/call을 평균. 마지막 비교용이며 처음부터 이 결과만 믿지 않는다.

## 저장 구조

- `token_raw/`: V1 parity raw token attention
- `token_maps/`: V1 방식으로 각 map을 1–99 percentile normalize한 PNG
- `word_raw/`, `word_maps/`: subtoken을 평균한 word별 map
- `records.jsonl`: token / block / call / sigma 기록

## Postprocess

`postprocess_anima_diagnostics_v2.py`를 실행하면:
- `token_gifs/blockXX_tokenYYY.gif`: token × block별 denoising trajectory
- `token_global/tokenYYY_mean_relative.png`: 각 subtoken의 전체 평균
- `word_gifs/`: word × block trajectory
- `word_global/`: 최종 word aggregate

`arona`라면 먼저 `token_gifs`에서 token 27, 28, 29를 block별로 확인하는 것이 핵심이다.

## Colab 설치

```python
node_dir=pathlib.Path(COMFY_DIR)/"custom_nodes/anima_inference_diagnostics_v2"
node_dir.mkdir(parents=True,exist_ok=True)
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/anima_inference_diagnostics_node_v2.py -O {node_dir}/__init__.py
```

후처리:

```python
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/postprocess_anima_diagnostics_v2.py -O /content/postprocess_anima_diagnostics_v2.py
%run /content/postprocess_anima_diagnostics_v2.py
```
