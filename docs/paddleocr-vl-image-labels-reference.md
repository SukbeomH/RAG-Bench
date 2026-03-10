# PaddleOCRVL 파이프라인 — 블록별 이미지(base64) 할당 레퍼런스

> 작성일: 2026-03-10
> 목적: PaddleOCRVL 파이프라인에서 어떤 블록 라벨에 PIL.Image (bbox crop)가 할당되는지 정리.
> worker_structured.py / output_formatter.py 수정 시 참고.

---

## 핵심 요약

PaddleOCRVL은 `layout_parsing/pipeline_v2.py`가 **아닌** 별도의 `paddleocr_vl/pipeline.py`를 사용한다.
두 파이프라인의 이미지 할당 로직이 다르므로 주의.

### _IMAGE_LABELS (VL 파이프라인 기준, 실측 확인)

```python
# PaddleOCRVL 파이프라인이 bbox crop PIL.Image를 할당하는 label
_IMAGE_LABELS = frozenset({"image", "header_image", "footer_image", "seal", "chart"})
```

---

## 상세 분석

### 1. 이미지 할당 코드 (paddleocr_vl/pipeline.py)

```python
# Line 274: vis_image_labels 정의
vis_image_labels = IMAGE_LABELS + ["seal"]
# IMAGE_LABELS = ["image", "header_image", "footer_image"]  (Line 51)
# → vis_image_labels = ["image", "header_image", "footer_image", "seal"]

# Line 276-278: chart는 use_chart_recognition=False일 때만 추가
if not use_chart_recognition:
    image_labels += ["chart"]
    vis_image_labels += ["chart"]

# Line 476: 이 조건으로 block.image 할당 여부 결정
if block_label in vis_image_labels and block_img is not None:
    block_info.image = {
        "path": img_path,
        "img": Image.fromarray(block_img),
    }
```

### 2. _get_img_obj fallback (Line 976-985)

```python
def _get_img_obj(block, model_settings):
    if block.get("image", None):
        return block["image"]
    # image/seal은 항상, chart는 조건부
    if block["block_label"] in ("image", "seal") or (
        block["block_label"] == "chart"
        and not model_settings.get("use_chart_recognition", False)
    ):
        path = construct_img_path(block["block_label"], block["block_bbox"])
        return {"path": path, "img": None}  # img=None인 fallback 주의!
    return None
```

---

## 라벨별 이미지 할당 정리

| 라벨 | `.image` 할당 | 조건 | base64 가능 |
|---|---|---|---|
| `image` | dict(path + PIL.Image) | 항상 | O |
| `header_image` | dict(path + PIL.Image) | 항상 | O |
| `footer_image` | dict(path + PIL.Image) | 항상 | O |
| `seal` | dict(path + PIL.Image) | 항상 | O |
| `chart` | dict(path + PIL.Image) | `use_chart_recognition=False` | O (기본값) |
| **`table`** | **None** | - | **X** |
| **`formula`** | **None** | - | **X** |
| **`figure`** | **None** (VL의 IMAGE_LABELS에 없음) | - | **X** |
| `text` 등 기타 | None | - | X |

---

## layout_parsing/pipeline_v2.py와의 차이 (혼동 주의)

```python
# layout_parsing/pipeline_v2.py (Line 756-766) — 비VL 파이프라인
# 여기서는 table, formula도 이미지가 할당됨!
if label in ["seal", "table", "formula", "chart"] + BLOCK_LABEL_MAP["image_labels"]:
    # BLOCK_LABEL_MAP["image_labels"] = ["image", "figure", "seal"]
    block.image = {"path": img_path, "img": img}
```

| | pipeline_v2 (비VL) | paddleocr_vl (VL) |
|---|---|---|
| table | O | **X** |
| formula | O | **X** |
| figure | O | **X** (IMAGE_LABELS에 없음) |
| header_image | X | **O** |
| footer_image | X | **O** |

---

## 실측 검증 결과

### table_image.pdf (5페이지, table/chart/image 혼합)

```
label                     total  has_b64  null_b64  no_field
--------------------------------------------------------------
chart                         1        1         0         0   _IMAGE
image                         1        1         0         0   _IMAGE
table                         7        0         0         7   (no field)
text                         30        0         0        30
paragraph_title              11        0         0        11
...
```

### graph_rich.pdf (10페이지, 그래프/이미지 풍부)

```
label                     total  has_b64  null_b64  no_field
--------------------------------------------------------------
image                        17       17         0         0   _IMAGE
text                         18        0         0        18
doc_title                     3        0         0         3
...
```

- `_IMAGE_LABELS`에 해당하는 블록: **100% base64 포함**
- 비해당 블록: `base64_encoding` 필드 자체 없음 (JSON 크기 최적화)

---

## worker_structured.py 코드 패턴

```python
# _IMAGE_LABELS에 해당하면 반드시 base64_encoding 필드 포함 (null 허용)
if str(label) in _IMAGE_LABELS:
    pil_img = None
    if img_data is not None:
        if isinstance(img_data, dict):
            pil_img = img_data.get("img")
        elif hasattr(img_data, "save"):
            pil_img = img_data
    if pil_img is not None:
        try:
            result["base64_encoding"] = _pil_to_base64(pil_img)
        except Exception:
            result["base64_encoding"] = None
    else:
        result["base64_encoding"] = None
```

## output_formatter.py 코드 패턴

```python
# _IMAGE_LABELS에 해당하면 반드시 base64_encoding 필드 포함
if label in _IMAGE_LABELS:
    b64 = _get_block_field(block, "base64_encoding")
    if not b64:
        b64 = _extract_base64(block)
    element["base64_encoding"] = b64 if b64 else None
```

---

## 유의사항

1. **PaddleOCRVL ≠ layout_parsing**: 두 파이프라인의 이미지 할당 대상이 다름
2. **chart의 조건부 할당**: `use_chart_recognition=True`면 chart는 인식 파이프라인으로 처리되어 이미지 미할당 가능
3. **_get_img_obj fallback**: `img=None`인 dict를 반환할 수 있음 → base64 변환 시 null 처리 필수
4. **table/formula는 VL 파이프라인에서 이미지 미할당**: HTML content로만 표현됨
5. **figure 라벨**: VL 파이프라인의 IMAGE_LABELS에 없음 (layout_parsing에만 있음)
