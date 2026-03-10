---
title: "Session [2026-03-10 16:00]: PaddleOCRVL _IMAGE_LABELS 실측 정정"
tags:
  - session-summary
  - branch:master
  - paddleocr
  - image-labels
  - base64
type: session-summary
created: 2026-03-10T07:00:00Z
contextual_description: "PaddleOCRVL 파이프라인의 블록별 이미지(base64) 할당 로직 실측 검증 및 코드 수정"
keywords:
  - _IMAGE_LABELS
  - _BASE64_LABELS
  - worker_structured.py
  - output_formatter.py
  - paddleocr_vl/pipeline.py
  - vis_image_labels
---

## Session Handoff: PaddleOCRVL _IMAGE_LABELS 실측 정정

### 배경
worker_structured.py와 output_formatter.py에서 `_BASE64_LABELS`(9종)가 선언되어 있었으나:
1. 실제 코드에서 사용되지 않았고 (worker_structured.py)
2. PaddleOCRVL 파이프라인의 실제 이미지 할당과 불일치

### 핵심 발견

**PaddleOCRVL은 두 개의 서로 다른 파이프라인을 가짐:**

| 파이프라인 | 파일 | 이미지 할당 대상 |
|---|---|---|
| layout_parsing (비VL) | `paddlex/.../layout_parsing/pipeline_v2.py` | table, formula, chart, seal, image, figure |
| **paddleocr_vl (VL)** | `paddlex/.../paddleocr_vl/pipeline.py` | **image, header_image, footer_image, seal, chart** |

우리 코드는 VL 파이프라인을 사용하므로 후자가 정확.

**VL 파이프라인 이미지 할당 로직 (pipeline.py:274-278, 476):**
```python
vis_image_labels = IMAGE_LABELS + ["seal"]
# IMAGE_LABELS = ["image", "header_image", "footer_image"]
if not use_chart_recognition:
    vis_image_labels += ["chart"]

# Line 476: 이 조건으로 block.image 할당
if block_label in vis_image_labels and block_img is not None:
    block_info.image = {"path": img_path, "img": Image.fromarray(block_img)}
```

### 수정 내용

**커밋: `bfa723b`**

1. `worker_structured.py`:
   - `_BASE64_LABELS`(9종) → `_IMAGE_LABELS`(5종: image, header_image, footer_image, seal, chart)
   - 기존: `if img_data is not None:` → 변경: `if str(label) in _IMAGE_LABELS:`
   - 해당 라벨이면 base64_encoding 필드 반드시 포함 (null 허용)

2. `output_formatter.py`:
   - `_BASE64_LABELS` → `_IMAGE_LABELS` 동일 5종
   - 해당 라벨에만 base64_encoding 필드 추가

3. `docs/paddleocr-vl-image-labels-reference.md` — 코드 스니펫 포함 상세 레퍼런스

### 실측 검증 결과

| PDF | 라벨 | 총 블록 | base64 포함 | 결과 |
|---|---|---|---|---|
| table_image.pdf | chart | 1 | 1 | OK |
| table_image.pdf | image | 1 | 1 | OK |
| table_image.pdf | table | 7 | - | no field (정상) |
| graph_rich.pdf | image | 17 | 17 | OK |

### 유의사항 (후속 작업 시)
- `table`, `formula`, `figure`는 VL 파이프라인에서 이미지 미할당 — HTML/text content만 사용
- `chart`는 `use_chart_recognition=True`면 인식 파이프라인으로 처리되어 이미지 미할당 가능
- `_get_img_obj` fallback이 `{"path": ..., "img": None}`을 반환할 수 있음 → null 처리 필수
- layout_parsing/pipeline_v2.py의 로직과 혼동하지 말 것

### 파일 변경 목록
- `isolated_backends/paddleocr/worker_structured.py` — _IMAGE_LABELS 정정, 라벨 기반 필터링
- `isolated_backends/paddleocr/output_formatter.py` — _IMAGE_LABELS 정정, 라벨 기반 필터링
- `docs/paddleocr-vl-image-labels-reference.md` — 상세 레퍼런스 (신규)
