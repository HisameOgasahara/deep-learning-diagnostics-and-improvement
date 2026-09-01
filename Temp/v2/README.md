# Anima Attention Diagnostics V2

이번 버전은 `nisaruj/comfyui-daam`의 **역할 분리와 aggregation 순서**를 최대한 그대로 따라가고, attention hook만 Anima/Cosmos Predict2에 맞게 바꾼 버전이다.

핵심 흐름:

```text
prompt + Anima tokenizer
        ↓
T5 token map
        ↓
샘플링 중 모든 실제 prompt token의 cross-attention 수집
        ↓
head 합산
        ↓
각 denoising call 안에서 layer/block 평균
        ↓
call/timestep 전체 합산 → token별 global heatmap
        ↓
선택한 word에 속한 subtoken global maps 평균
        ↓
최종 이미지 크기로 bicubic resize
        ↓
마지막에 한 번 min-max normalize
        ↓
overlay
```

이 순서는 `comfyui-daam`의 `BaseAttentionPatcher._up_sample_attn` + `GlobalHeatMap.compute_global_heat_map` + `compute_word_heat_map`의 의미를 Anima에 맞게 옮긴 것이다.

## 이전 V2와 달라진 점

- `attention_words`를 **Diagnostics에서 받지 않는다.** 샘플링 때 실제 prompt token 전체를 수집한다.
- 따라서 한 번 생성한 뒤 Overlay에서 `arona`, `blue hair`, `building`, `sky` 등 여러 단어를 바꿔가며 분석할 수 있다.
- `concentration_topk`, 특정 subtoken 수동 선택, selected block 휴리스틱을 기본 DAAM 경로에서 제거했다.
- positive attribution은 ComfyUI의 `cond_or_uncond`를 사용해 conditional branch만 수집한다.
- head는 원본 comfyui-daam처럼 합산한다.
- 각 denoising call 안에서 동일 resolution의 여러 cross-attention layer를 평균한다.
- 최종 token global map은 timestep/resolution 결과를 합산한다.
- word map은 **global token map을 먼저 만든 뒤** 해당 단어의 subtoken들을 평균한다.
- normalize는 최종 word map을 이미지 크기로 resize한 뒤 한 번만 min-max로 한다.

## 노드 연결

```text
CLIPLoader
   │
   ▼
Anima Text Encode + Token Map V2
   ├─ conditioning ──────────────→ positive conditioning
   └─ token_map ───────┬────────→ Anima Attention Diagnostics V2
                       ├────────→ Anima Attention Overlay V2
                       └────────→ Anima Token Map Viewer V2 (선택)

MODEL ───────────────────────────→ Anima Attention Diagnostics V2
                                    │
                                    └─ model ─→ sampler

VAE Decode IMAGE ─────────────────→ Anima Attention Overlay V2.images
Diagnostics.diagnostic_directory ─→ Anima Attention Overlay V2.diagnostic_directory
```

## Anima Text Encode + Token Map V2

기존 positive `CLIPTextEncode` 대신 사용한다.

출력:
- `conditioning`: 기존 positive conditioning 위치로 연결
- `token_map`: Diagnostics와 Overlay에 둘 다 연결
- `mapping_text`: 필요하면 문자열 출력 노드로 확인

## Anima Token Map Viewer V2

디버깅용이다. 프롬프트 전체의

```text
[index] token id, decoded T5 token
```

매핑을 표시한다.

## Anima Attention Diagnostics V2

역할은 원래 DAAM의 `KSamplerDAAM`에 해당한다. 단어를 여기서 고르지 않고, 실제 prompt token 전체의 attention을 수집한다.

권장 기본값:
- `snapshot_every_n_calls = 1`
- `max_map_side = 64`
- `colormap = turbo`

결과 폴더:
- `timestep_raw/`: layer 평균 후의 call/timestep별 raw token map
- `timestep_maps/`: 위 raw map을 개별 확인하기 위한 debug PNG
- `global/`: Overlay에서 만든 최종 word raw/heatmap
- `records.jsonl`: call, batch, token, layer count, sigma 기록

## Anima Attention Overlay V2

역할은 원래 DAAM의 `DAAMAnalyzer`에 해당한다.

입력 `attention_words`에 예를 들어:

```text
arona, blue hair, building, sky
```

처럼 comma-separated 단어/구를 넣는다.

각 word에 대해 자동으로:
1. 현재 Anima T5 token span을 찾고
2. 각 token의 global heatmap을 만든 뒤
3. subtoken global maps를 평균하고
4. 이미지 크기로 resize하고
5. 마지막 한 번 min-max normalize해서
6. overlay와 heatmap을 출력한다.

즉 사용자가 `token 27/28/29 중 뭘 골라야 하나`를 결정하는 구조가 아니다.

출력:
- `overlay_images`: word별 생성 이미지 + heatmap
- `heatmap_images`: word별 heatmap만
- `debug_info`: `word -> token indices` 확인용

## Postprocess

`postprocess_anima_diagnostics_v2.py`는 수집된 timestep token map에서 token별 global raw map을 만들어 `global_tokens/`에 저장한다.

이 파일은 검증/디버깅용이며 Overlay 실행에 필수는 아니다.

## Colab 설치

기존 V2 URL은 그대로다.

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

코드 갱신 후에는 ComfyUI를 재시작하고, 입력 스키마가 바뀌었으므로 기존 Diagnostics/Overlay 노드는 삭제 후 다시 추가하는 것을 권장한다.
