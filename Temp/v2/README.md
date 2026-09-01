# Anima Attention Diagnostics V2

ComfyUI에서 **추가 학습/ablation/backward 없이** Anima의 projected cross-attention을 단어 기준으로 기록한다.

## 핵심 변경

- 사용자가 token index를 직접 입력하지 않는다.
- `Anima Text Encode + Token Map V2`가 기존 `CLIPTextEncode` 역할을 하면서 Anima의 **T5 target token 위치 → cross-attention key index** 매핑을 같이 만든다.
- `Anima Attention Diagnostics V2`에는 `girl, blue hair`처럼 보고 싶은 **단어/구문 문자열**을 입력한다.
- 한 단어가 여러 T5 subtoken이면 해당 token heatmap을 평균해 word heatmap으로 만든다. 이는 comfyui-daam의 word heatmap 방식과 같은 방향이다.
- `Anima Token Map Viewer V2`에 `token_map` 선을 연결하면 프롬프트 전체의 `[index] token id / decoded token` 매핑을 별도로 확인할 수 있다.
- CFG batch에서는 ComfyUI의 `cond_or_uncond` 정보를 이용해 conditional branch만 heatmap 계산에 사용한다.
- head 평균 전에 head별 map을 보존하고, `concentration_topk`에서는 공간적으로 집중된 head만 선택한다.
- map마다 percentile 재정규화를 하지 않고 `attention_ratio = token_probability * text_key_count` 공통 scale을 사용한다. `1.0`은 text key 전체에 균등한 attention이다.

## 왜 T5 token index인가

ComfyUI의 Anima text encoder는 Qwen3와 T5를 함께 tokenize한다. Qwen3 hidden state는 LLMAdapter의 source이고, **T5 token IDs가 LLMAdapter의 target positions**가 된다. LLMAdapter 출력이 최종 Anima DiT의 text context가 되므로 DiT cross-attention의 key 위치는 T5 target token 위치와 대응한다. 길이가 512보다 짧으면 Anima가 model-side에서 512까지 zero padding한다.

## 노드 연결

기존 positive prompt의 `CLIPTextEncode`를 다음처럼 바꾼다.

```text
CLIPLoader.CLIP
     │
     ├───────────────┐
     ▼               │
Anima Text Encode + Token Map V2
     │ conditioning  │
     ├──────────────> sampler positive
     │
     ├ token_map ────────────────┐
     │                            ▼
     │                  Anima Attention Diagnostics V2
     │                            ▲
     │                            │ MODEL
     │                    diffusion model
     │
     └ token_map ──> Anima Token Map Viewer V2
                       └─ 프롬프트 전체 index/token 매핑 표시

Anima Attention Diagnostics V2.MODEL
     └──────────────> sampler MODEL
```

`Anima Attention Diagnostics V2`의 `attention_words`에는 예를 들어:

```text
girl, blue hair, red ribbon
```

처럼 comma-separated 문자열을 넣는다. index는 자동으로 찾는다.

## 권장 첫 설정

- `attention_words`: 확인하고 싶은 단어 1~3개
- `selected_blocks`: `0,6,12,18,24,27`
- `selected_heads`: `all`
- `head_selection_mode`: `concentration_topk`
- `top_k_heads`: `4`
- `snapshot_every_n_calls`: `1`
- `ratio_vmax`: `6.0`
- `save_head_pngs`: `true`

결과는 `/content/anima_diagnostics_v2/<session>/`에 저장된다.

주요 파일:

- `token_map.txt`: 프롬프트 전체 T5 token index 매핑
- `token_map.json`: 같은 정보의 machine-readable 버전
- `attention_maps/`: word × block × denoising call aggregate
- `head_maps/`: 선택된 개별 head map
- `raw/`: raw probability/ratio/selected head/token indices NPZ
- `records.jsonl`: 기록 메타데이터
- postprocess 후 `word-<word>_mean_ratio.png`, `word-<word>_median_ratio.png`

## Colab 설치

기존 노트북 1번 셀:

```python
node_dir=pathlib.Path(COMFY_DIR)/"custom_nodes/anima_inference_diagnostics_v2"
node_dir.mkdir(parents=True,exist_ok=True)
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/anima_inference_diagnostics_node_v2.py -O {node_dir}/__init__.py
```

후처리 셀:

```python
!wget -q https://raw.githubusercontent.com/HisameOgasahara/deep-learning-diagnostics-and-improvement/main/Temp/v2/postprocess_anima_diagnostics_v2.py -O /content/postprocess_anima_diagnostics_v2.py
%run /content/postprocess_anima_diagnostics_v2.py
```

노드 파일을 갱신한 뒤에는 **ComfyUI 프로세스를 재시작**해야 새 class가 로드된다.
