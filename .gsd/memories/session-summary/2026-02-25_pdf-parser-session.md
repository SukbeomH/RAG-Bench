---
title: "pdf_parser 모듈 구축 세션 요약"
tags:
  - session
  - pdf-parser
  - rag
  - hybrid-backend
type: session-summary
created: 2026-02-25T00:00:00Z
contextual_description: "pdf_to_md.ipynb 분석 → pdf_parser 모듈 신규 구축 → PDF 파싱 리서치 → Hybrid backend 구현 → 실 PDF 테스트"
keywords:
  - pdf_parser
  - hybrid_backend
  - PyMuPDF4LLM
  - Docling
  - Gemini VLM
  - OmniDocBench
  - MinerU
  - dots.ocr
related:
  - 2026-02-25_pdf-parser-hybrid-backend
---

## pdf_parser 모듈 구축 세션 요약

### 세션 흐름

1. **노트북 분석**: `pdf_to_md.ipynb` — 3-tier PDF 분류 방식 파악
   - Simple → PyMuPDF4LLM / Medium → Docling / Complex → VLM

2. **pdf_parser 모듈 구축**: 6개 파일 신규 생성
   - 카테고리별 파서, 품질 검사, 문서 단위 라우터

3. **리서치**: 2025~2026 최신 PDF 파싱 동향
   - OmniDocBench 벤치마크 기준 정립
   - dots.ocr-1.5, NVIDIA Nemotron-Parse 1.1, Dolphin v2 등 신규 프로젝트 확인
   - MinerU 2.0+ Hybrid backend 패턴 발견
   - `research_notes.md` 문서 저장

4. **Hybrid Backend 구현**: `hybrid_backend.py` 신규 생성
   - 페이지 단위 백엔드 선택 (MinerU 방식)
   - smart_router.py에 `--mode hybrid` 통합

5. **테스트**: SPRi AI Brief (한국어, 30p)
   - 25p Rule-based + 5p VLM 분류 정확
   - lazy import로 docling 미설치 환경 대응

### 다음 세션 시 참고
- `GEMINI_API_KEY` 설정 후 VLM 페이지 실제 변환 테스트 가능
- MinerU 또는 dots.ocr-1.5를 category2 대체재로 추가 고려 가능
- OmniDocBench 지표 방식으로 quality_checker.py 고도화 가능 (표/수식/읽기순서 분리 평가)
