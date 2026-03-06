---
title: "세션 핸드오프 — 정규화 분리 리팩토링 + 재평가 + paddleocr-vl 원인 분석"
tags:
  - session-handoff
  - branch:master
type: session-handoff
created: 2026-03-05T07:00:00Z
---

# 세션 핸드오프 — 2026-03-05 16시

## 완료된 작업

### 1. 정규화 분리 리팩토링 (커밋: cb0e9d9)
- `runner.py`: `_normalize_and_eval()` 분리, `reeval_spec()` / `reeval_dir()` 추가, `--reeval-only` CLI
- `pdf_bench_graph.py`: `normalize_pdfs` 노드 삽입 + `_should_parse()` 조건부 라우팅
- `pdf_bench_state.py`: `skip_parse`, `normalize_results` 필드 추가
- `pdf_bench.py`: `normalize_pdfs` 노드 (skip_parse=True → reeval_dir 호출)
- 테스트: reeval 3개 + graph 4개 업데이트, 전체 155개 통과

### 2. 정규화 재적용 실행 완료
- `bench_results/20260305-1528` 26개 결과에 reeval 실행
- 주요 정규화 효과:
  - pymupdf text_only: +0.086 (bold_in_headers)
  - pymupdf table_native: +0.046
  - paddleocr-vl: +0.013~0.019 (vlm_location_tokens)
  - openai text_only: +0.003 (code_block_wrapper, blockquote_markers)

### 3. paddleocr-vl NED 저품질 원인 분석 완료
- **근본 원인: VLM hallucination (반복 생성)**
  - text_only: "코기 코넷" 412회 반복 (출력의 15.7%)
  - table_native: "메타의 딜레마" 384회 반복 (22.0%)
  - table_image: 숫자 시퀀스 반복 생성
- 표(table) 파싱 시 셀 내용을 무한 반복하는 VLM 특성
- 정규화 후에도 NED 0.40~0.43 (LOC 토큰은 이미 제거됨)

### 4. README.md 전면 갱신
- 모노레포 구조 반영, 오래된 레거시(rag_bench/ 단일 패키지) 내용 제거
- LangGraph 3개 파이프라인 흐름도 추가
- PDF 파서 벤치마크 섹션 추가

### 5. 메모리 업데이트
- MEMORY.md: skip_parse 라우팅, reeval 기능, paddleocr-vl hallucination 기록
- rag-pipeline.md: PDF Bench 그래프 구조 업데이트

## 다음 세션 작업 (우선순위순)

### 1. normalize.py에 반복 시퀀스 제거 규칙 추가
- 패턴: 2~3단어 조합이 4회 이상 연속 반복 → 1회로 축소
- 예: `(코기\s+코넷\s+){4,}` → `코기 코넷`
- 주의: 정상적인 표 데이터(숫자 시퀀스 등)와 구분 필요
- 구현 후 `--reeval-only`로 paddleocr-vl 재평가 → NED 향상 확인

### 2. 벤치마크 커버리지 확충
- **upstage**: 0/11 PDF → 전체 실행 필요 (rate limit 주의: --delay 10)
- **openai**: 2/11 PDF → 나머지 9개 실행 (비용: PDF당 100-220초)
- **paddleocr-vl**: 4/11 PDF → 나머지 7개 실행 (mlx-vlm 서버 필요)
- **deepseek-ocr2**: 0/11 PDF → GPU 환경 필요

### 3. 보고서 재생성
- 정규화 + 반복제거 적용 후 최종 보고서 생성
- `--report-only --results-dir` 사용

## 현재 상태
- **Branch**: master
- **최신 커밋**: cb0e9d9 (pushed)
- **미커밋 변경**: .gsd/CURRENT.md, servers/mlx_vlm/run.sh 정도
- **벤치마크 결과**: bench_results/20260305-1528 (26개, 정규화 재적용 완료)
- **테스트**: 155개 통과
