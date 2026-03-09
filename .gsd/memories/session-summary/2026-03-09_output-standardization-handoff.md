# 세션 핸드오프: PaddleOCR 출력 표준화 (2026-03-09)

## 완료된 작업

### 1. Upstage Document Parse API v2 출력 명세 분석
- `output_structure/` 실측 JSON + 공식 문서 크로스 레퍼런스
- 명세 문서: `docs/upstage-document-parse-output-spec.md`

### 2. PaddleOCR → Upstage v2 호환 JSON 변환기
- **`isolated_backends/paddleocr/output_formatter.py`**
  - HTML-first 전략: table HTML 원본 보존 → markdown → text 파생
  - 원본 label 보존 (`category = label`, Upstage 카테고리로 매핑 안 함)
  - 좌표 정규화: `[x1,y1,x2,y2]` 픽셀 → `[{x,y}×4]` (0~1)
  - base64 패스스루: worker에서 전달된 base64_encoding 그대로 전달

### 3. 구조화 Worker
- **`isolated_backends/paddleocr/worker_structured.py`**
  - 블록 단위 JSON 출력 (label, content, bbox, base64_encoding)
  - `block.image`가 메모리에 있으면 무조건 base64로 저장
  - table은 image=None (HTML만 제공), chart/image/figure에만 PIL 이미지 존재

### 4. E2E 테스트
- **`isolated_backends/paddleocr/run_e2e_test.py`**
  - SPRi AI Brief 29페이지 PDF, fitz로 10페이지 분할 처리
  - 결과: `output_structure/paddleocr_SPRi_AI_Brief_260210_*.json`

### 5. 벤치마크 비교 인프라
- **`isolated_backends/paddleocr/run_bench_pdfs.sh`** — PaddleOCR 벤치마크 11개 PDF 일괄 처리
- **`isolated_backends/paddleocr/run_upstage_bench.py`** — Upstage API 벤치마크 JSON 수집
- 결과 (gitignored, 로컬):
  - `data/benchmark_pdfs/paddleocr_upstage/` — PaddleOCR→Upstage 변환 (7개)
  - `data/benchmark_pdfs/upstage_raw/` — Upstage API raw (7개)
  - Upstage API 400 에러로 4개 PDF 실패 (원본DPI/200dpi) → 성공한 7개끼리만 비교 가능

## 미완료 / 후속 작업

### 1. 벤치마크 정량 비교 (미실행)
- 7개 공통 PDF에 대해 PaddleOCR vs Upstage JSON 구조 비교 가능
- 사용자가 "JSON 형태로만 나오면 됨, 최종 평가까진 불필요"로 지시 → JSON 생성만 완료

### 2. Bridge 통합 (미착수)
- `output_formatter.py`를 기존 `isolated_backends/paddleocr/bridge.py`에 통합
- 현재는 독립 모듈, bridge에서 import해서 사용 가능

### 3. table base64 이미지 (보류)
- PaddleOCR 네이티브 파이프라인은 table에 image=None (OTSL→HTML만 수행)
- 필요 시 PyMuPDF로 페이지 이미지에서 bbox 영역 크롭 가능하나, "기본 지원 형태만 진행" 지시로 보류

## 커밋 이력
- `f868961` feat: PaddleOCR 출력 Upstage v2 호환 JSON 표준화 + 벤치마크 비교 인프라
- `68a82c2` chore: GSD 세션 메모리 + CURRENT.md 업데이트
