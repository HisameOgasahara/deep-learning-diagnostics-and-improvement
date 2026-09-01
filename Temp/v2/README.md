# Anima Attention Diagnostics V2

ComfyUI에서 추가 학습/ablation/backward 없이 Anima(Cosmos Predict2)의 text cross-attention을 기록하고, 단어별 DAAM 스타일 heatmap/overlay를 만드는 custom node다.

## 현재 V2 핵심

1. `attention_words`에는 토큰 번호가 아니라 `arona`, `blue hair`처럼 **단어를 직접 입력**한다.
2. `Anima Text Encode + Token Map V2`가 Anima의 T5 target token stream을 기록하고 word -> cross-attention key index를 자동 매핑한다.
3. `Anima Token Map Viewer V2`로 전체 prompt의 index / token id / decoded token을 따로 확인할 수 있다.
4. CFG batch에서는 conditional branch만 attribution 계산에 사용한다.
5. **중간 map에는 percentile/min-max normalization을 하지 않는다.** subtoken -> head -> block/timestep(call) 순으로 raw attention을 먼저 aggregate한다.
6. DAAM처럼 보기 위한 `relative_final` 시각화는 **최종 word map이 완성된 뒤 딱 한 번만** 1-99 percentile normalization한다.
7. 정량 비교용 `absolute` visualization도 그대로 남긴다.
8. `concentration_topk`는 실험/디버그 옵션으로 남기되 기본값은 `all`이다. 집중도가 높다는 이유만으로 의미적으로 맞는 head라고 가정하지 않는다.
9. raw NPZ에는 subtoken별 map도 저장해서 `arona -> [27,28,29]` 같은 경우 각 subtoken을 사후 검증할 수 있다.

## 노드

### Anima Text Encode + Token Map V2
입력: `clip`, `text`
출력:
- `conditioning`: 기존 positive conditioning으로 사용
- `token_map`: 아래 Viewer/Diagnostics에 연결
- `mapping_text`: 문자열 출력

### Anima Token Map Viewer V2
`token_map`을 연결하면 전체 prompt token mapping을 UI에서 확인한다.

### Anima Attention Diagnostics V2

```text
MODEL ------------------------------------> Anima Attention Diagnostics V2.model
Anima Text Encode + Token Map V2.token_map -> Anima Attention Diagnostics V2.token_map
Anima Attention Diagnostics V2.model ------> sampler.model
```

권장 시작값:
- `attention_words`: `arona`
- `selected_blocks`: `0,6,12,18,24,27`
- `selected_heads`: `all`
- `head_selection_mode`: `all`
- `snapshot_every_n_calls`: `1`
- `ratio_vmax`: `6.0` (`absolute` 시각화에만 사용)

### Anima Attention Overlay V2

```text
VAE Decode.IMAGE --------------------------> Anima Attention Overlay V2.images
Diagnostics.diagnostic_directory ----------> Anima Attention Overlay V2.diagnostic_directory
```

권장값:
- `attention_words`: Diagnostics와 동일
- `aggregation`: `mean`
- `visualization`: `relative_final`
- `relative_low_percentile`: `1`
- `relative_high_percentile`: `99`
- `alpha`: `0.5`

`relative_final`은 **모든 raw map을 먼저 합친 뒤 마지막에 한 번만 정규화**한다. `absolute`는 `attention_ratio = token_probability * text_key_count` 공통 scale을 그대로 보여준다.

## 출력 폴더

기본: `/content/anima_diagnostics_v2/<session>/`

- `raw/*.npz`: raw word/head/subtoken attention
- `attention_maps/absolute_*.png`: call/block별 절대 scale preview
- `head_maps/`: head별 절대 scale preview
- `token_map.txt`, `token_map.json`: prompt-token mapping
- postprocess 실행 후:
  - `word-*_mean_absolute.png`
  - `word-*_mean_relative.png`
  - `word-*_tokenNNN_relative.png` (subtoken debug)

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

노드 파일을 다시 받은 뒤에는 ComfyUI를 재시작해야 새 기본값/입력이 반영된다.
