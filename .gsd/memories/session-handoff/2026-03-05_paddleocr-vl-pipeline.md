---
title: "세션 핸드오프 — PaddleOCR-VL 네이티브 파이프라인 전환 + 품질 분석"
tags:
  - session-handoff
  - branch:master
type: session-handoff
created: 2026-03-05T09:00:00Z
---

# 세션 핸드오프 — 2026-03-05 17시

## 완료된 작업

### 1. PaddleOCR-VL 저품질 근본 원인 분석
- **VLM 직접호출(openai_compat.py)** vs **네이티브 파이프라인(PaddleOCRVL)** 비교
- VLM 직접호출 문제점:
  - 반복 생성: 표 페이지에서 "코기 코넷" 412회, "메타의 딜레마" 384회
  - LOC 토큰: 0~968개 (실행마다 변동)
  - NED: 0.42~0.43 (text_only), 0.43 (table_native)
- 네이티브 파이프라인(standalone 검증):
  - 레이아웃 분석 → 영역별 VLM → markdownify 조합
  - NED: **0.83** (text_only), **0.77** (table_native) — pymupdf/docling 능가
  - 반복 없음, LOC 0개

### 2. paddleocr-vl 백엔드 네이티브 파이프라인 전환 (커밋: c38443c)
- `paddleocr_vl.py`: 새 백엔드 (subprocess bridge 패턴)
- `__init__.py`: lazy loading 대상 변경 (openai_compat → paddleocr_vl)
- `openai_compat.py`: paddleocr-vl 등록 제거, 상세 영문 프롬프트 추가 (deepseek-ocr2용)
- `worker.py`: `max_new_tokens=4096` 추가
- 테스트: 172개 통과

### 3. 페이지별 GT 추가
- `data/benchmark_pdfs/gt/pages/`: text_only/table_native/graph_rich × 5p = 15개
- 페이지 구분 기준: "숫자 | AI 현황 보고서 Mirae Asset Securities Research"

### 4. 분석 인사이트
- **프롬프트 영향**: 영문 상세 프롬프트로 변경 시 LOC 토큰 제거됨, 그러나 반복생성은 해결 안 됨
- **max_tokens 영향**: mlx-vlm 기본값=256, 1024/2048/4096 모두 VLM 직접호출에서는 반복 발생
- **결정적 차이**: VLM 직접호출 vs 네이티브 파이프라인 — 영역 분리가 반복 억제의 핵심
- **mlx-vlm 결정성**: temperature=0.0에서도 같은 결과, 반복 발생 페이지는 항상 반복

## 다음 세션 작업 (우선순위순)

### 1. 네이티브 파이프라인으로 전체 벤치마크 재실행
- `--preset ocr`으로 paddleocr-vl × 11 PDF 실행
- PaddleOCR venv 환경(`paddleocr_standalone/PaddleOCR/`)이 필요
- `isolated_backends/paddleocr/bridge.py`의 `PADDLEOCR_DIR` 경로 확인 필요

### 2. 벤치마크 커버리지 확충
- upstage: 0/11 → 전체 (rate limit: --delay 10)
- openai: 2/11 → 나머지 9개
- deepseek-ocr2: 0/11 (GPU 필요)

### 3. 최종 보고서 재생성
- 네이티브 파이프라인 결과 포함
- `--report-only --results-dir` 사용

## 현재 상태
- **Branch**: master
- **최신 커밋**: c38443c (pushed)
- **테스트**: 172개 통과
- **서버**: mlx-vlm http://localhost:8111 실행 중
