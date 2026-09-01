# Anima Attention Diagnostics V2

ComfyUI에서 **추가 학습/ablation/backward 없이** Anima(Cosmos Predict2)의 projected cross-attention Q/K를 기록하는 custom node다.

## V1에서 바뀐 점

1. **head 평균을 먼저 하지 않는다.** token × block × head × denoising call × spatial map을 보존한다.
2. 각 heatmap을 1–99 percentile로 따로 늘리지 않는다. `attention_ratio = token_probability × text_key_count`를 사용해 **모든 map에 같은 절대 scale**을 적용한다. `1.0`은 해당 query가 text key들을 균등하게 본 경우다.
3. `selected_heads=all`이면 모든 head를 사용할 수 있고, `head_selection_mode=concentration_topk`이면 spatial entropy가 낮은(localized) head 상위 `top_k_heads`만 aggregate한다.
4. 선택된 head별 PNG, raw NPZ, block/call별 aggregate PNG를 모두 남긴다.
5. postprocess는 token별로 기록된 block/call map을 mean/median으로 합쳐 최종 비교용 heatmap을 만든다.

> `concentration_topk`는 추가 학습 없이 가능한 휴리스틱이다. 위치가 맞다는 정답을 쓰는 것이 아니라 **공간적으로 더 집중된 head**를 고르는 것이므로, 최종 해석 전에 head PNG를 직접 확인하는 것을 권장한다.

## 파일

- `anima_inference_diagnostics_node_v2.py`: ComfyUI custom node
- `postprocess_anima_diagnostics_v2.py`: 최신 session의 token별 aggregate 생성

## ComfyUI 기본 설정

- Node: `Anima Attention Diagnostics V2`
- `selected_blocks`: `0,6,12,18,24,27`
- `text_token_indices`: 보고 싶은 text key index
- `selected_heads`: `all`
- `head_selection_mode`: `concentration_topk`
- `top_k_heads`: `4`
- `snapshot_every_n_calls`: `1`
- `ratio_vmax`: `6.0`

결과는 기본적으로 `/content/anima_diagnostics_v2/<session>/` 아래에 저장된다.

## Colab 설치

기존 노트북 1번 셀에서 custom node 다운로드 URL만 V2로 바꾸면 된다.

```python
node_dir=pathlib.Path(COMFY_DIR)/"custom_nodes/anima_inference_diagnostics_v2"
node_dir.mkdir(parents=True,exist_ok=True)
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/anima_inference_diagnostics_node_v2.py -O {node_dir}/__init__.py
```

기존 노트북 4번 postprocess 셀도 다음처럼 바꾼다.

```python
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/postprocess_anima_diagnostics_v2.py -O /content/postprocess_anima_diagnostics_v2.py
%run /content/postprocess_anima_diagnostics_v2.py
```
