# 세션 핸드오프: OCR 서버 독립 실행 인프라 + PDF 벤치마크

**날짜**: 2026-03-05
**브랜치**: master

## 완료된 작업

### 1. OCR 서버 독립 uv 프로젝트화
- `servers/paddleocr_vl/` — pyproject.toml (paddlepaddle+paddlex[ocr]) + run.sh (SSL 인증서 설정 포함)
- `servers/mlx_vlm/` — pyproject.toml (mlx-vlm) + run.sh (macOS Apple Silicon 전용)
- `servers/deepseek_ocr2/` — pyproject.toml + run.sh (CUDA 전용)
- `servers/got_ocr2/` — pyproject.toml + run.sh

### 2. openai_compat.py 수정 (pdf-parsers)
- **시스템 프롬프트**: 긴 SYSTEM_PROMPT(1843자) → 짧은 `_OPENSOURCE_SYSTEM_PROMPT` ("PDF를 마크다운으로 변환하세요.")
  - 원인: mlx-vlm이 긴 시스템 프롬프트에서 Internal Server Error 발생
- **max_tokens**: `max_tokens=4096` 추가
  - 원인: mlx-vlm 기본값 256 → 한국어 OCR 출력이 잘림

### 3. PaddleOCR-VL mlx-vlm 벤치마크 실행 완료
- 유효 결과: `bench_results/20260305-0825/` (max_tokens=4096 적용)
- 무효 결과 (max_tokens=256 문제): 20260304-1708, 20260304-1718, 20260304-1724
- **NED 결과 요약** (1.0 = 완벽):
  - text_only: 0.421 | table_image: 0.457 | graph_rich: 0.149
  - 평균 ~0.29 (K8s PPStructureV3 0.7594 대비 낮음 — 0.9B 경량 VLM 한계)

### 4. 기존 벤치마크 결과
- `bench_results/20260304-1642/` — pymupdf + docling × 11 PDF (22건 중 21건)
- `bench_results/20260304-1642-fix/` — docling-graph_rich_image_72dpi 누락 1건 보완

## 미완료 작업

### PPStructureV3 로컬 벤치마크
- 서버 설치 완료 (`uv sync` 성공), 모델 캐시도 있음 (~/.paddlex/official_models/)
- **서버 시작은 성공하나** Connection error로 벤치마크 미실행
- 서버 시작 명령: `cd servers/paddleocr_vl && bash run.sh` → port 8000
- 벤치마크 실행: `OPENSOURCE_VLM_ENDPOINT=http://localhost:8000/v1 OPENSOURCE_VLM_MODEL=paddleocr-vl-1.5 uv run python -m autorag_pdf_eval.runner --preset ocr --output ./bench_results`

### 통합 보고서 생성
```bash
uv run python -m autorag_pdf_eval.runner --report-only \
  --results-dir bench_results/20260304-1642,bench_results/20260304-1642-fix,bench_results/20260305-0825
```
- PPStructureV3 결과 추가 후 최종 보고서 생성 필요

### 서버 자동 시작/종료 파이프라인 통합
- 사용자 요청: runner가 서버 프로세스를 자동 시작/종료 (서버 방식 유지)
- 미착수 상태

## 핵심 발견 사항

1. **mlx-vlm vs PPStructureV3**: 같은 "PaddleOCR-VL" 이름이지만 완전히 다른 방식
   - mlx-vlm: VLM 단일 모델 (이미지→텍스트 직접 생성)
   - PPStructureV3: 16개 서브모델 파이프라인 (Layout Detection → OCR → Table → Formula)
2. **NED 척도**: `1 - edit_distance/max_len` → 1.0이 완벽 일치, 0.0이 완전히 다름
3. **paddlepaddle uv 설치**: PaddlePaddle PyPI 인덱스에서 다른 패키지를 찾으려 해서 `index-strategy = "unsafe-best-match"` 필요

## 실행 중인 프로세스
- mlx-vlm 서버 (PID 14743): port 8111 — 화요일부터 계속 실행 중
- 벤치마크에 불필요하면 `kill 14742 14743`으로 종료 가능
