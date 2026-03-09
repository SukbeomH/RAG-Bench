# Upstage Document Parse API — 출력 구조 명세

> 공식 문서 + 실제 출력(`output_structure/*.json`) 기반 크로스 레퍼런스.
> PaddleOCR 네이티브 파이프라인 출력 표준화를 위한 타겟 스키마.

---

## 1. API 개요

| 항목 | 값 |
|---|---|
| Endpoint (Sync) | `POST https://api.upstage.ai/v1/document-digitization` |
| Endpoint (Async) | `POST https://api.upstage.ai/v1/document-digitization/async` |
| 최대 페이지 | Sync 100 / Async 1,000 |
| 최대 파일 크기 | 50 MB |
| 지원 포맷 | JPEG, PNG, BMP, PDF, TIFF, HEIC, DOCX, PPTX, XLSX, HWP, HWPX |
| 처리 모드 | `standard`, `enhanced`, `auto` |

---

## 2. 요청 파라미터 (Request)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `model` | string | Y | — | `document-parse` 또는 `document-parse-nightly` |
| `document` | file | Y | — | 처리할 문서 파일 |
| `mode` | string | N | `standard` | `standard` / `enhanced` / `auto` |
| `output_formats` | JSON array | N | `["html"]` | `["text"]`, `["html"]`, `["markdown"]` 조합 |
| `ocr` | string | N | `auto` | `auto`(이미지만 OCR) / `force`(항상 OCR) |
| `coordinates` | boolean | N | `true` | 바운딩 박스 좌표 포함 여부 |
| `chart_recognition` | boolean | N | `true` | 차트→표 변환 (enhanced 모드에서 항상 활성) |
| `merge_multipage_tables` | boolean | N | `false` | 다중 페이지 테이블 병합 (enhanced: 최대 20p) |
| `base64_encoding` | JSON array | N | — | base64 이미지 반환 대상 카테고리 (예: `["table","figure"]`) |

---

## 3. 응답 구조 (Response) — 최상위

```json
{
  "api": "2.0",
  "model": "document-parse-251217",
  "ocr": false,
  "content": { ... },
  "elements": [ ... ],
  "usage": { ... }
}
```

| 필드 | 타입 | 공식 문서 | 실제 출력 | 설명 |
|---|---|---|---|---|
| `api` / `apiVersion` | string | `apiVersion: "1.1"` | `api: "2.0"` | API 버전. 키 이름과 값이 버전에 따라 다름 |
| `model` | string | O | `"document-parse-251217"` | 사용된 모델 식별자 |
| `ocr` | boolean | 요청 파라미터로만 기재 | `false` | OCR 사용 여부 (응답에도 포함됨) |
| `content` | object | O | O | 전체 문서 결합 텍스트 (3가지 포맷) |
| `elements` | array | O | O | 개별 레이아웃 요소 배열 |
| `usage` | object | O | O (확장 필드 포함) | 페이지 사용량 정보 |

---

## 4. `content` — 전체 문서 결합 텍스트

```json
"content": {
  "html": "",
  "markdown": "2월호\n\n![image](/image/placeholder)...",
  "text": "2월호\n![image](/image/placeholder)..."
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `html` | string | HTML 형식 전체 문서. `output_formats`에 `"html"` 포함 시 채워짐 |
| `markdown` | string | Markdown 형식 전체 문서. 마크다운 문법(#, -, \|, !) 포함 |
| `text` | string | 순수 텍스트. 마크다운 문법 제거된 버전 |

> **참고**: `output_formats` 요청 파라미터에 포함된 포맷만 채워짐. 미요청 포맷은 빈 문자열.

---

## 5. `elements[]` — 개별 레이아웃 요소 (핵심)

### 5.1 공통 필드

```json
{
  "id": 15,
  "category": "table",
  "page": 3,
  "content": {
    "html": "",
    "markdown": "| col1 | col2 |\n| --- | --- |...",
    "text": "col1 col2..."
  },
  "coordinates": [
    {"x": 0.1044, "y": 0.1061},
    {"x": 0.8956, "y": 0.1061},
    {"x": 0.8956, "y": 0.5839},
    {"x": 0.1044, "y": 0.5839}
  ]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | int | 요소 고유 ID (문서 전체에서 유니크, 0부터 순번) |
| `category` | string | 요소 유형 (아래 5.2 참조) |
| `page` | int | 소속 페이지 번호 (1-based) |
| `content` | object | `{html, markdown, text}` — 해당 요소의 텍스트 (3가지 포맷) |
| `coordinates` | array[4] | 정규화 바운딩 박스 좌표 (아래 5.3 참조) |

### 5.2 `category` — 요소 유형 전체 목록

| Category | 공식 문서 | 실제 출력 확인 | 설명 |
|---|---|---|---|
| `paragraph` | O | O (47개) | 본문 텍스트 |
| `heading1` | O | O (30개) | 1단계 제목 (`# ` 접두사) |
| `heading2` | O | — | 2단계 제목 (문서에 따라 출현) |
| `list` | O | O (26개) | 리스트 항목 (`- ` 접두사) |
| `table` | O | O (9개) | 표. `\| col \| col \|` 마크다운 + base64 이미지 |
| `figure` | O | O (7개) | 그림/사진. `![image](...)` + base64 이미지 |
| `chart` | O | — | 차트/그래프. `chart_recognition=true`면 표로 변환 |
| `header` | O | O (3개) | 페이지 상단 반복 텍스트 (문서 제목 등) |
| `footer` | O | O (20개) | 페이지 하단 (페이지 번호 등) |
| `caption` | O | O (1개) | 표/그림 캡션 |
| `equation` | O | — | 수식 (LaTeX 형식) |
| `index` | O | — | 목차/색인 |
| `footnote` | O | — | 각주 |

> **실제 출력에서 확인된 카테고리**: `paragraph`, `heading1`, `list`, `table`, `figure`, `header`, `footer`, `caption` (8종)

### 5.3 `coordinates` — 바운딩 박스 좌표

```
좌상(0) ──── 우상(1)
  │              │
  │   element    │
  │              │
좌하(3) ──── 우하(2)
```

| 인덱스 | 위치 | 예시 |
|---|---|---|
| `[0]` | 좌상 (top-left) | `{"x": 0.167, "y": 0.109}` |
| `[1]` | 우상 (top-right) | `{"x": 0.308, "y": 0.109}` |
| `[2]` | 우하 (bottom-right) | `{"x": 0.308, "y": 0.128}` |
| `[3]` | 좌하 (bottom-left) | `{"x": 0.167, "y": 0.128}` |

- **값 범위**: `0.0` ~ `1.0` (페이지 크기 대비 정규화)
- **픽셀 변환**: `pixel_x = x * page_width_px`, `pixel_y = y * page_height_px`
- `coordinates` 요청 파라미터 `false`로 설정 시 응답에서 제외

### 5.4 `base64_encoding` — 이미지 데이터 (조건부)

`table`과 `figure`(+ `chart`) 카테고리에서 추가로 포함되는 필드:

```json
{
  "category": "table",
  "base64_encoding": "/9j/2wCEAAIBAQEBAQIBAQE...",
  ...
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `base64_encoding` | string | JPEG base64 인코딩 이미지. 해당 요소 영역을 잘라낸 이미지 |

- 요청 시 `base64_encoding: ["table", "figure"]`로 대상 카테고리 지정
- 실제 출력에서 `table` 평균 ~100KB, `figure` 평균 ~100KB (base64 문자열 기준)
- 모든 카테고리에 지정 가능: `table`, `figure`, `chart`, `heading1`, `header`, `footer`, `caption`, `paragraph`, `equation`, `list`, `index`, `footnote`

---

## 6. `usage` — 사용량 정보

```json
"usage": {
  "pages": 10,
  "standard": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

| 필드 | 타입 | 공식 문서 | 실제 출력 | 설명 |
|---|---|---|---|---|
| `pages` | int | O | O | 처리된 총 페이지 수 |
| `standard` | array[int] | — | O | `standard` 모드로 처리된 페이지 번호 목록 |

> **참고**: `mode: "auto"` 사용 시 `standard`/`enhanced` 배열이 각각 해당 모드로 처리된 페이지를 나타낼 수 있음.

---

## 7. 공식 문서 vs 실제 출력 차이점

| 항목 | 공식 문서 | 실제 출력 (`output_structure/`) | 비고 |
|---|---|---|---|
| 버전 키 | `apiVersion: "1.1"` | `api: "2.0"` | v2에서 키 이름 변경 |
| `ocr` 응답 필드 | 요청 파라미터로만 기재 | `ocr: false` 포함 | v2에서 응답에 추가 |
| `usage.standard` | 미기재 | `[1,2,...,10]` 배열 포함 | 모드별 처리 페이지 추적 |
| `heading2` | 카테고리 목록에 포함 | 미출현 | 문서 내용에 따라 다름 |
| `chart` | 별도 카테고리 | 미출현 | chart_recognition으로 table 변환 |
| `equation` | 카테고리 목록에 포함 | 미출현 | 수식 없는 문서 |
| `index`, `footnote` | 카테고리 목록에 포함 | 미출현 | 해당 요소 없는 문서 |

---

## 8. 페이지별 요소 분포 예시 (SPRi_AI_Brief 0-9페이지)

| 페이지 | paragraph | heading1 | list | table | figure | header | footer | caption |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | — | — | — | 1 | — | — | — |
| 2 | 2 | 1 | 4 | — | 4 | — | 1 | — |
| 3 | 1 | 6 | 1 | 4 | 1 | — | 1 | — |
| 4 | 8 | — | — | — | 1 | — | — | — |
| 5 | 9 | 4 | 4 | — | — | — | 5 | — |
| 6 | 4 | 3 | 3 | 1 | — | 1 | 1 | — |
| 7 | 7 | 6 | 3 | — | — | — | 5 | — |
| 8 | 2 | 5 | 4 | 1 | — | 1 | 1 | — |
| 9 | 9 | 3 | 3 | 1 | — | — | 5 | — |
| 10 | 4 | 2 | 4 | 2 | — | 1 | 1 | 1 |
| **합계** | **47** | **30** | **26** | **9** | **7** | **3** | **20** | **1** |

---

## 9. PaddleOCR 네이티브 파이프라인 매핑 포인트

현재 PaddleOCR 파이프라인 출력(`{page_num: markdown}`)을 이 스키마로 변환하려면:

| Upstage 필드 | PaddleOCR 소스 | 매핑 방법 |
|---|---|---|
| `api` | — | 고정값 `"2.0"` |
| `model` | — | `"paddleocr-vl"` 등 식별자 |
| `ocr` | — | `true` (항상 OCR 수행) |
| `content.markdown` | 전체 페이지 결합 | 기존 bridge.py 출력 그대로 |
| `content.text` | markdown에서 문법 제거 | 정규식 후처리 |
| `content.html` | — | 필요 시 markdown→html 변환 |
| `elements[].id` | — | 순번 자동 부여 |
| `elements[].category` | `block_label` | `text→paragraph`, `table→table`, `image→figure` 등 매핑 |
| `elements[].page` | `page_index + 1` | 0-based → 1-based 변환 |
| `elements[].content` | `block_content` | 블록 텍스트를 3가지 포맷으로 |
| `elements[].coordinates` | PaddleOCR bbox | 정규화 좌표 변환 필요 |
| `elements[].base64_encoding` | 페이지 이미지 crop | table/figure 영역 이미지 추출 |
| `usage.pages` | 전체 페이지 수 | `len(results)` |

---

## Sources

- [Upstage Document Parse — 공식 문서](https://console.upstage.ai/docs/capabilities/document-digitization/document-parsing)
- [Upstage API Reference (for Agents)](https://console.upstage.ai/api/docs/for-agents/raw)
- [LangChain Upstage Document Parse Parser](https://python.langchain.com/api_reference/upstage/document_parse_parsers/langchain_upstage.document_parse_parsers.UpstageDocumentParseParser.html)
