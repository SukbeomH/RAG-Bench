# PDF-Extract-Kit / MinerU 도입 검토 보고서

> 작성일: 2026-03-03
> 목적: 현재 PDF 파서 벤치마크 시스템(pdf_parser/)의 백엔드 대체/보완 가능성 평가
> 대상: [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) + [MinerU](https://github.com/opendatalab/MinerU)

---

## 1. Executive Summary

| 항목 | 현재 구성 | MinerU (Pipeline) | MinerU 2.5 (VLM) |
|------|----------|-------------------|-------------------|
| **정확도** | NED 0.66~0.74 (pymupdf/docling) | 추정 NED 0.80+ | OmniDocBench 90.67, TEDS 88.22 |
| **표 인식** | TEDS 0.49~0.53 (pymupdf/docling) | TableMaster 기반 | TEDS 88.22 (SOTA) |
| **GPU 필요** | ❌ (CPU-only K8s) | ❌ (CPU 가능, 느림) | ✅ (최소 8GB VRAM) |
| **라이선스** | MIT/Apache 혼합 | **AGPL-3.0** ⚠️ | **AGPL-3.0** ⚠️ |
| **한국어** | ✅ (PaddleOCR, Ollama) | ✅ (109개 언어 OCR) | ✅ (PaddleOCR 기반) |
| **설치 복잡도** | 백엔드별 독립 | 중간 (~20GB 모델) | 높음 (GPU + 모델) |

### 결론

1. **MinerU Pipeline 백엔드**: 현재 `mineru` 백엔드로 이미 spec.py에 예약됨. CPU-only 환경에서 실행 가능하며, **docling 대체 후보**로 적합. 단, AGPL-3.0 라이선스 확인 필요.
2. **MinerU 2.5 VLM**: SOTA 성능이나 **GPU 필수**(최소 8GB VRAM). 현재 K8s CPU-only 환경과 비호환. GPU 노드 확보 시 최우선 도입 대상.
3. **PDF-Extract-Kit 직접 사용**: MinerU가 PDF-Extract-Kit의 상위 래퍼이므로, MinerU를 통해 간접 사용하는 것이 실용적.
4. **즉시 도입 권장**: MinerU Pipeline 백엔드를 Phase 2에서 구현, 기존 9종 백엔드와 동일 벤치마크 비교.

---

## 2. PDF-Extract-Kit 개요

[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) (OpenDataLab)은 PDF에서 고품질 콘텐츠를 추출하는 모듈식 툴킷.

### 2-1. 핵심 모듈

| 모듈 | 모델 | 역할 |
|------|------|------|
| 레이아웃 감지 | DocLayout-YOLO, LayoutLMv3 | 텍스트/표/이미지/수식 영역 식별 |
| 수식 감지 | YOLOv8 (fine-tuned) | 인라인/블록 수식 위치 |
| 수식 인식 | UniMERNet | 수식 → LaTeX 변환 |
| OCR | PaddleOCR | 텍스트 추출 (109개 언어) |
| 표 인식 | TableMaster, StructEqTable | 표 → HTML/Markdown/LaTeX |

### 2-2. MinerU와의 관계

```
PDF-Extract-Kit (모듈 라이브러리)
    └── MinerU (end-to-end 파이프라인)
         ├── Pipeline 백엔드 (규칙 기반 + 모듈 조합)
         ├── Hybrid 백엔드 (Pipeline + VLM 혼합)
         └── VLM 백엔드 (MinerU 2.5, 단일 모델)
```

PDF-Extract-Kit는 개별 모듈을 제공하고, MinerU는 이를 조합한 end-to-end PDF→Markdown 파이프라인.

---

## 3. MinerU 상세 분석

### 3-1. 백엔드 아키텍처 비교

| 백엔드 | 정확도 | CPU 가능 | 최소 VRAM | 속도 (A100) | 특징 |
|--------|:------:|:--------:|:---------:|:-----------:|------|
| **Pipeline** | 82+ | ✅ | 6GB (GPU시) | ~1 p/s | 규칙 기반 + 5개 모듈 조합 |
| **Hybrid-auto** | 90+ | ❌ | 10GB | ~1.5 p/s | Pipeline + VLM 혼합 |
| **VLM-auto** (2.5) | 90+ | ❌ | 8GB | 2.12 p/s | 단일 1.2B VLM |
| **\*-http-client** | 90+ | ✅ | 불필요 | 네트워크 의존 | 원격 추론 서버 호출 |

### 3-2. MinerU 2.5 벤치마크 성능 (SOTA)

**OmniDocBench** (종합 PDF 파싱 벤치마크):

| 모델 | 파라미터 | 종합 점수 | 텍스트 ED | 수식 CDM | 표 TEDS | 속도 (p/s) |
|------|:--------:|:---------:|:---------:|:--------:|:-------:|:----------:|
| **MinerU 2.5** | **1.2B** | **90.67** | **0.047** | **88.46** | **88.22** | **2.12** |
| dots.ocr | 3B | 88.41 | 0.052 | 86.30 | 85.61 | 0.28 |
| MonkeyOCR-pro | 3.7B | 88.85 | 0.057 | 86.12 | 86.93 | 0.47 |
| Gemini-2.5 Pro | — | ~85 | — | — | — | API |
| GPT-4o | — | ~83 | — | — | — | API |

**Ocean-OCR** (순수 OCR):

| 언어 | Edit Distance | F1 Score |
|------|:-------------:|:--------:|
| 영문 | 0.033 | 0.945 |
| 중문 | 0.082 | 0.965 |

### 3-3. 하드웨어 요구사항

| 항목 | Pipeline (CPU) | Pipeline (GPU) | VLM 2.5 |
|------|:--------------:|:--------------:|:-------:|
| RAM | 16GB+ (32GB 권장) | 16GB+ | 16GB+ |
| VRAM | 불필요 | 6GB+ | 8GB+ |
| 디스크 | 20GB+ (모델 포함) | 20GB+ | 20GB+ |
| Python | 3.10~3.13 | 3.10~3.13 | 3.10~3.13 |
| 속도 | 느림 (~0.3 p/s) | 중간 (~1 p/s) | 빠름 (2.12 p/s) |

### 3-4. 설치

```bash
# 기본 설치 (GPU)
pip install "mineru[all]"

# CPU-only
pip install "mineru[all-cpu]"

# Docker
# docker/ 디렉토리 제공, 공식 이미지는 별도 확인 필요
```

---

## 4. 현재 시스템과의 비교

### 4-1. 백엔드 대응 매핑

| 현재 백엔드 | 역할 | MinerU 대체 | 비고 |
|-------------|------|:-----------:|------|
| `pymupdf` | 빠른 텍스트 추출 | Pipeline (텍스트 레이어) | MinerU도 PyMuPDF 내부 사용 |
| `docling` | OCR + 레이아웃 | **Pipeline** | 직접 대체 후보 |
| `openai` | GPT-4o VLM | VLM-http-client | OpenAI API 호출 |
| `upstage` | Document Parse API | 대체 불가 | Upstage 고유 기능 |
| `granite-vision` | K8s OCR | Pipeline (PaddleOCR) | OCR 엔진 차이 |
| `got-ocr2` | K8s OCR | Pipeline or VLM | 더 높은 정확도 기대 |
| `paddleocr-vl` | K8s OCR | Pipeline (동일 OCR) | PaddleOCR 기반으로 동일 |

### 4-2. 성능 비교 (추정)

| 지표 | pymupdf | docling | MinerU Pipeline | MinerU 2.5 |
|------|:-------:|:-------:|:---------------:|:----------:|
| Text NED | 0.66 | 0.74 | **0.80+** (추정) | **0.95+** |
| Table TEDS | 0.49 | 0.53 | **0.70+** (추정) | **0.88** |
| 속도 (페이지) | 0.4s | 10s | 3~5s (CPU) | 0.5s (GPU) |
| GPU 필요 | ❌ | ❌ | ❌ | ✅ |

> MinerU Pipeline 수치는 공식 벤치마크 미공개, OmniDocBench 논문의 "Pipeline=82+" 기준 추정.

### 4-3. 아키텍처 비교

**현재 시스템**:
```
pdf_parser/
├── category1_simple.py    → pymupdf (텍스트 전용)
├── category2_medium.py    → docling (OCR + 레이아웃)
├── category3_*.py         → VLM/OCR (페이지별 이미지 → API)
├── hybrid_backend.py      → 페이지별 text/VLM 라우팅
└── smart_router.py        → 문서별 라우팅
```

**MinerU 아키텍처**:
```
MinerU Pipeline:
1. PyMuPDF로 PDF 메타데이터/텍스트 추출
2. DocLayout-YOLO로 레이아웃 감지
3. PaddleOCR로 텍스트 인식
4. TableMaster/StructEqTable로 표 인식
5. UniMERNet으로 수식 인식
6. 읽기 순서 결정 → Markdown/JSON 출력
```

**핵심 차이**: 현재 시스템은 백엔드별 독립 파이프라인 (모듈 교체 가능), MinerU는 통합 파이프라인 (높은 정확도, 모듈 교체 어려움).

---

## 5. 도입 시나리오 분석

### 시나리오 A: MinerU Pipeline을 10번째 백엔드로 추가 (권장)

**작업량**: 소 (1~2일)
- `pdf_parser/backends/mineru.py` 구현 (spec.py에 이미 `"mineru"` 예약됨)
- `pip install "mineru[all-cpu]"` 의존성 추가
- 기존 벤치마크 프레임워크에서 동일 평가 (NED, TEDS)

**장점**:
- 기존 9종과 동일 기준 비교 가능
- CPU-only K8s 환경 호환
- docling 대비 표/수식 인식 개선 기대

**단점**:
- 모델 다운로드 ~10GB (PVC 용량 확인 필요)
- CPU 환경에서 속도 느림 (3~5s/page)
- AGPL-3.0 라이선스

### 시나리오 B: MinerU 2.5 VLM 백엔드 추가

**작업량**: 중 (3~5일)
- GPU 노드 확보 필요 (EKS에 GPU 인스턴스 추가)
- MinerU 2.5 서버 Deployment 작성
- http-client 방식으로 기존 category3_opensource.py 패턴 활용

**장점**:
- SOTA 성능 (TEDS 88.22, 현재 최고 대비 +35%p)
- 단일 모델로 텍스트/표/수식 통합 처리

**단점**:
- GPU 인스턴스 비용 (g5.xlarge ~$1/h)
- AGPL-3.0 라이선스
- K8s 클러스터 GPU 노드 미보유

### 시나리오 C: 현재 시스템의 hybrid_backend를 MinerU로 교체

**작업량**: 대 (1~2주)
- smart_router.py의 라우팅 로직을 MinerU Pipeline으로 교체
- 기존 평가 체계 유지하되 내부 엔진 변경
- Phase 2 벤치마크에서 검증

**장점**: 통합 파이프라인으로 일관된 결과
**단점**: 기존 백엔드 유연성 상실, 대규모 리팩토링

---

## 6. 라이선스 리스크

| 항목 | 현재 시스템 | MinerU / PDF-Extract-Kit |
|------|-----------|------------------------|
| 라이선스 | MIT (pymupdf4llm), MIT (docling) | **AGPL-3.0** |
| 상용 사용 | ✅ 제한 없음 | ⚠️ 소스 공개 의무 |
| 내부 도구 | ✅ | ✅ (배포하지 않으면 무관) |
| SaaS 제공 | ✅ | ❌ (AGPL: 네트워크 사용도 배포) |

**판단**: 벤치마크 도구(내부 평가 전용)로는 AGPL 문제 없음. 단, **서비스 파이프라인에 직접 통합 시 AGPL 전파** 주의.

---

## 7. 권장 사항

### 즉시 실행 (Phase 2 벤치마크)

1. **MinerU Pipeline 백엔드 구현** → `spec.py`의 `"mineru"` 활성화
   - `pip install "mineru[all-cpu]"` (K8s 워커 이미지에 추가)
   - `convert_pdf()` 래퍼 작성 (Markdown 출력 → 기존 NED/TEDS 평가)
   - 기존 11 PDF × MinerU = 11 벤치마크 Job 추가

2. **docling과 직접 비교** → 동일 데이터셋, 동일 평가 지표
   - 예상: 표 인식(TEDS) 개선, 텍스트(NED) 동등 이상

### 중기 검토 (GPU 확보 시)

3. **MinerU 2.5 VLM 서버 배포** → K8s GPU 노드에 Deployment
   - http-client 방식으로 기존 `category3_opensource.py` 패턴 재사용
   - TEDS 88+ 달성 시 모든 OCR 백엔드(granite-vision, got-ocr2, paddleocr-vl) 대체 가능

### 장기 검토

4. **서비스 파이프라인 통합** 시 AGPL 라이선스 법률 검토
5. **MinerU 2.5의 한국어 성능** 별도 평가 (현재 벤치마크는 영문/중문 중심)

---

## 참고 자료

- [PDF-Extract-Kit GitHub](https://github.com/opendatalab/PDF-Extract-Kit)
- [PDF-Extract-Kit 문서](https://pdf-extract-kit.readthedocs.io/en/latest/)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [MinerU 공식 문서](https://opendatalab.github.io/MinerU/)
- [MinerU2.5 논문 (arXiv:2509.22186)](https://arxiv.org/html/2509.22186v1)
- [MinerU2.5 성능 분석 (Neurohive)](https://neurohive.io/en/state-of-the-art/mineru2-5-open-source-1-2b-model-for-pdf-parsing-outperforms-gemini-2-5-pro-on-benchmarks/)
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
- [PDF 파싱 도구 비교 연구 (arXiv:2410.09871)](https://arxiv.org/html/2410.09871v1)
- [Air-gapped RAG용 PDF 파서 평가](https://dev.to/ashokan/from-pdfs-to-markdown-evaluating-document-parsers-for-air-gapped-rag-systems-58eh)
- [PDF Parsing for LLM Input (Nicolas' Notebook)](https://nbrosse.github.io/posts/pdf-parsing/pdf-parsing.html)
