---
title: "pdf_parser 모듈 구축 및 Hybrid Backend 구현"
tags:
  - execution
  - pdf-parser
  - hybrid-backend
  - rag
  - pdf-to-markdown
type: execution-summary
created: 2026-02-25T00:00:00Z
contextual_description: "pdf_to_md.ipynb 분석 기반 pdf_parser 모듈 신규 구축, MinerU 방식 Hybrid backend 패턴 적용 및 실 PDF 테스트 완료"
keywords:
  - pdf_parser
  - hybrid_backend
  - smart_router
  - PyMuPDF4LLM
  - Docling
  - Gemini VLM
  - MinerU
  - 페이지별 라우팅
  - OmniDocBench
related:
  - 2026-02-25_pdf-parser-research-notes
---

## pdf_parser 모듈 구축 및 Hybrid Backend 구현

### 배경
`pdf_to_md.ipynb` 노트북을 분석해 RAG 파이프라인용 PDF 파서를 별도 모듈로 제작.
단순/중간/복잡 3-tier 분류 + MinerU 방식 페이지 단위 Hybrid backend 구현.

### 생성 파일 구조
```
pdf_parser/
├── category1_simple.py    # PyMuPDF4LLM (디지털 텍스트 PDF)
├── category2_medium.py    # Docling (스캔/표/멀티컬럼)
├── category3_complex.py   # Google Gemini VLM (차트/다이어그램)
├── quality_checker.py     # 단어수/헤더/표/수식 품질 검사
├── smart_router.py        # 문서 단위 OR 페이지 단위 라우팅
├── hybrid_backend.py      # 페이지 단위 Hybrid backend (신규)
├── requirements.txt
└── research_notes.md      # 리서치 내용 문서화
```

### Hybrid Backend 핵심 로직
- `classify_page()`: chars < 50 OR images ≥ 3 → VLM, 그 외 → Rule-based
- `ConversionReport`: 페이지별 백엔드 선택 이력 + 오류 추적
- VLM 클라이언트는 실제 필요 시점에만 초기화 (API 연결 최소화)

### 실 테스트 결과 (SPRi AI Brief 30페이지)
- Rule-based 처리: 25페이지 (83%)
- VLM 필요: 5페이지 (17%, chars < 50인 표지/섹션 구분 이미지 페이지)
- 출력: 84.3 KB, 9,058 단어
- 한국어 텍스트 및 표 정상 추출 확인

### smart_router.py 변경
- 최상단 import → 실제 사용 시점 lazy import (docling 미설치 환경 대응)
- `--mode document | hybrid` 파라미터 추가
- argparse CLI 인터페이스 추가

### 리서치 주요 발견
- OmniDocBench (CVPR 2025): 업계 표준 벤치마크
- 속도: MinerU (0.21s/p) > Docling (0.49s/p) > Marker (0.86s/p)
- 신규 VLM 파서: dots.ocr-1.5 (2026.02), NVIDIA Nemotron-Parse 1.1 (2025.11), Dolphin v2 (ByteDance)
