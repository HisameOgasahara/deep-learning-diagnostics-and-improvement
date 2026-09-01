# Anima Attention Diagnostics V2

ComfyUI에서 **추가 학습/ablation/backward 없이** Anima(Cosmos Predict2)의 projected cross-attention Q/K를 기록하고, 원래 DAAM처럼 **프롬프트 단어를 직접 지정해 생성 이미지 위에 heatmap을 overlay**하는 custom node 세트다.

## 핵심 변경점

1. **head 평균을 먼저 하지 않는다.** token × block × head × denoising call × spatial map을 보존한다.
2. 각 heatmap을 1–99 percentile로 따로 늘리지 않는다. `attention_ratio = token_probability × text_key_count`를 사용해 모든 map에 같은 절대 scale을 적용한다. `1.0`은 해당 query가 text key들을 균등하게 본 경우다.
3. `selected_heads=all` + `head_selection_mode=concentration_topk`이면 spatial entropy가 낮은(localized) head 상위 `top_k_heads`를 골라 aggregate한다.
4. ComfyUI의 `cond_or_uncond`를 보고 conditional branch만 attribution에 사용한다.
5. 사용자는 token index를 직접 넣지 않는다. `attention_words = girl, blue hair`처럼 문자열을 넣으면 Anima T5 target token 위치를 자동 매핑한다.
6. `Anima Token Map Viewer V2`로 프롬프트 전체의 `cross-attention key index / token id / decoded token` 매핑을 따로 확인할 수 있다.
7. `Anima Attention Overlay V2`가 생성 IMAGE와 누적 attention을 받아 **DAAM 스타일 overlay IMAGE**를 바로 출력한다.

> `concentration_topk`는 정답 위치를 쓰지 않는 추가학습 없는 휴리스틱이다. 최종 해석 전에 `head_maps/`의 head별 PNG도 확인하는 것을 권장한다.

## 노드

### 1. Anima Text Encode + Token Map V2
입력:
- `clip`
- `text`: 실제 positive prompt

출력:
- `conditioning`: 기존 positive conditioning 대신 sampler 쪽으로 연결
- `token_map`: Diagnostics / Token Map Viewer로 연결
- `mapping_text`: 필요하면 문자열 출력 노드에 연결

### 2. Anima Token Map Viewer V2
`token_map`을 연결하면 예를 들어 다음처럼 확인할 수 있다.

```text
prompt: a girl with blue hair
T5 target tokens -> Anima cross-attention key indices:
[000] id=... token='▁a'
[001] id=... token='▁girl'
...
```

동일한 정보는 diagnostic session의 `token_map.txt`, `token_map.json`에도 저장된다.

### 3. Anima Attention Diagnostics V2
MODEL 선에 끼운다.

기본 설정:
- `attention_words`: `girl` 또는 `girl, blue hair, ribbon`
- `selected_blocks`: `0,6,12,18,24,27`
- `selected_heads`: `all`
- `head_selection_mode`: `concentration_topk`
- `top_k_heads`: `4`
- `snapshot_every_n_calls`: `1`
- `ratio_vmax`: `6.0`

출력:
- `model`: sampler로 연결
- `diagnostic_directory`: Overlay V2로 연결
- `selected_word_mapping`: 디버깅용 `word -> token indices`

결과는 기본적으로 `/content/anima_diagnostics_v2/<session>/`에 저장된다.

### 4. Anima Attention Overlay V2
원래 comfyui-daam의 Analyzer처럼 최종 생성 이미지와 attention을 결합한다.

입력:
- `images`: VAE Decode에서 나온 최종 IMAGE
- `diagnostic_directory`: `Anima Attention Diagnostics V2` 출력
- `attention_words`: Diagnostics에 넣은 것과 같은 단어/구문
- `alpha`: heatmap 혼합 비율, 기본 0.5
- `aggregation`: block/call 누적을 `mean` 또는 `median`
- `caption`: overlay 하단에 단어 표시

출력:
- `overlay_images`: **생성 이미지 + heatmap overlay**
- `heatmap_images`: heatmap만 색상 이미지로 출력

여러 단어를 넣으면 `batch × word` 순서로 IMAGE batch가 나온다.

## 권장 연결 구조

```text
CLIPLoader
   │
   └──> Anima Text Encode + Token Map V2
           ├── conditioning ───────────────> sampler positive
           ├── token_map ──> Token Map Viewer V2
           └── token_map ──> Attention Diagnostics V2

MODEL ─────────────────────> Attention Diagnostics V2 ── model ──> sampler
                                      │
                                      └─ diagnostic_directory ──────────────┐
                                                                            │
sampler ── latent ──> VAE Decode ── IMAGE ───────────────────────────────┐  │
                                                                         ▼  ▼
                                                           Anima Attention Overlay V2
                                                              ├─ overlay_images
                                                              └─ heatmap_images
```

## Colab 설치

기존 노트북 1번 셀:

```python
node_dir=pathlib.Path(COMFY_DIR)/"custom_nodes/anima_inference_diagnostics_v2"
node_dir.mkdir(parents=True,exist_ok=True)
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/anima_inference_diagnostics_node_v2.py -O {node_dir}/__init__.py
```

수정 후에는 ComfyUI 프로세스를 재시작해야 새 노드가 로드된다.

후처리 셀:

```python
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/postprocess_anima_diagnostics_v2.py -O /content/postprocess_anima_diagnostics_v2.py
%run /content/postprocess_anima_diagnostics_v2.py
```

후처리는 필수는 아니다. Overlay V2는 `records.jsonl + raw/*.npz`를 직접 읽어 block/call 누적을 수행하므로 생성 직후 ComfyUI에서 바로 overlay를 만들 수 있다.
