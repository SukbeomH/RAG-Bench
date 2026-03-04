# RAG Parser 종합 벤치마크 보고서

> 생성일: 2026-03-04 | Phase 4 완료 기준 | 6 백엔드 × 11 PDF | 55 페이지

---

## 1. 용어 사전 (Glossary)

| 약어 | 정의 |
|---|---|
| **NED** | Normalized Edit Distance — `1 - (Levenshtein 거리 / max(길이))`, 0~1 범위, 높을수록 텍스트 일치도 높음 |
| **TEDS** | Tree Edit Distance-based Similarity — 마크다운 셀 단위 NED + 표 개수 패널티, 표 구조 정확도 지표 |
| **DPI** | Dots Per Inch — 이미지 해상도 단위. 높을수록 선명하나 파일 크기 증가 |
| **VLM** | Vision-Language Model — 이미지와 텍스트를 동시에 이해하는 멀티모달 AI 모델 |
| **OCR** | Optical Character Recognition — 이미지에서 텍스트를 인식·추출하는 기술 |
| **RAG** | Retrieval-Augmented Generation — 검색 기반 증강 생성. 외부 문서를 검색하여 LLM 응답에 활용 |
| **GT** | Ground Truth — 평가 기준이 되는 정답 데이터 |
| **SOTA** | State of the Art — 특정 시점 최고 성능 |
| **MoE** | Mixture of Experts — 전문가 혼합 아키텍처. 전체 파라미터 중 일부만 활성화하여 효율성 확보 |
| **AGPL** | GNU Affero General Public License — 네트워크 서비스에도 소스 공개 의무가 적용되는 강력한 오픈소스 라이선스 |
| **OmniDocBench** | CVPR 2025 발표 PDF 파싱 벤치마크. 1,355 페이지, 9 문서 유형, 4 레이아웃, 3 언어 |

---

## 2. 프로젝트 개요

### 목적

RAG 시스템에서 **문서 파싱 품질은 검색·생성 정확도의 상한선**을 결정한다. 본 프로젝트는 다양한 한국어 PDF 유형에 대해 **최적의 파싱 백엔드와 라우팅 전략**을 정량적으로 도출하는 것을 목표로 한다.

### 핵심 질문 4가지

1. 어떤 백엔드가 어떤 문서 유형에 가장 정확한가?
2. DPI 저하가 OCR 정확도에 미치는 영향은?
3. 로컬 파서가 API 파서를 대체할 수 있는가?
4. MinerU Pipeline 추가 시 순위 변동은? (Phase 5 예정)

### 벤치마크 규모

| 항목 | 값 |
|---|---|
| 백엔드 수 | 6종 (로컬 3 + API 3) |
| PDF 문서 수 | 11개 (텍스트 1 + 표 5 + 그래프 5) |
| 총 페이지 수 | 55페이지 |
| 라우팅 모드 | direct (백엔드 순수 비교) |
| 평가 지표 | Text NED, Table TEDS |

### 데이터셋 구성

| 유형 | PDF 파일 | 설명 |
|---|---|---|
| 📄 텍스트형 | `text_only` | 순수 텍스트 문서 |
| 📊 표형 | `table_native` | 네이티브 PDF 표 |
| | `table_image` | 이미지화된 표 (기본 DPI) |
| | `table_image_72dpi` | 이미지 표 (72 DPI) |
| | `table_image_150dpi` | 이미지 표 (150 DPI) |
| | `table_image_200dpi` | 이미지 표 (200 DPI) |
| 📈 그래프형 | `graph_rich` | 그래프+텍스트 혼재 (네이티브) |
| | `graph_rich_image` | 이미지화된 그래프 (기본 DPI) |
| | `graph_rich_image_72dpi` | 이미지 그래프 (72 DPI) |
| | `graph_rich_image_150dpi` | 이미지 그래프 (150 DPI) |
| | `graph_rich_image_200dpi` | 이미지 그래프 (200 DPI) |

---

## 3. 백엔드 상세 소개

### 3-1. PyMuPDF (Category 1 — 규칙 기반)

| 항목 | 내용 |
|---|---|
| **개발사** | Artifex Software |
| **모델 크기** | N/A (규칙 기반, AI 모델 없음) |
| **실행 환경** | 💻 로컬 (CPU only) |
| **핵심 라이브러리** | `pymupdf4llm` |
| **라이선스** | AGPL-3.0 |
| **기술 특성** | PDF 내부 텍스트 스트림 직접 추출. 이미지/스캔 문서 처리 불가 |
| **강점** | 극도로 빠름 (~2s/page), 의존성 최소, 네이티브 텍스트 정확도 높음 |
| **약점** | 이미지 기반 PDF 완전 불가 (NED = 0), 표 구조 인식 제한적 |
| **적합 용도** | 텍스트 선택 가능한 단순 PDF |

### 3-2. Docling (Category 2 — OCR 파이프라인)

| 항목 | 내용 |
|---|---|
| **개발사** | IBM Research |
| **모델 크기** | N/A (내부 복합 모델 파이프라인) |
| **실행 환경** | 💻 로컬 (CPU, K8s 환경) |
| **핵심 라이브러리** | `docling` (transformers 의존) |
| **라이선스** | MIT |
| **기술 특성** | 레이아웃 분석 + OCR + 표 구조 인식 통합 파이프라인 |
| **강점** | 기업 환경 적합, air-gapped 배포 가능, 표 구조 인식 |
| **약점** | `transformers>=4.49` 호환 문제로 로컬 재현 불가, 이미지 PDF NED 저조 |
| **적합 용도** | 중간 복잡도 문서, 기업 내부 배포 |

### 3-3. OpenAI GPT-4o (Category 3 — API VLM)

| 항목 | 내용 |
|---|---|
| **개발사** | OpenAI |
| **모델 크기** | ~200B (추정, 비공개) |
| **실행 환경** | 🌐 API |
| **핵심 라이브러리** | `openai` SDK |
| **라이선스** | 상용 API (종량제) |
| **기술 특성** | 페이지별 base64 인코딩 → GPT-4o VLM에 Markdown 변환 요청 |
| **강점** | 범용 문서 이해력, 그래프형 2위 |
| **약점** | 느림 (~3-5min/page), 표 TEDS 0.2859 저조, 150dpi 이상치 (NED 0.26) |
| **적합 용도** | 복잡한 시각 요소가 포함된 문서 (비용 허용 시) |

### 3-4. Upstage (Category 3 — API)

| 항목 | 내용 |
|---|---|
| **개발사** | Upstage AI |
| **모델 크기** | 비공개 |
| **실행 환경** | 🌐 API |
| **핵심 라이브러리** | `requests` + `fitz` (30MB 초과 시 페이지 분할) |
| **라이선스** | 상용 API (Document Parse API) |
| **기술 특성** | 전용 문서 파싱 API. 텍스트+표+레이아웃 동시 추출 |
| **강점** | 텍스트형·표형 NED 1위, 표 TEDS 1위 (0.6316), 전 DPI 구간 안정적 |
| **약점** | API 비용, 외부 의존성, 30MB 초과 파일 처리 시 페이지 분할 필요 |
| **적합 용도** | 텍스트 중심·표 중심 문서, 높은 정확도 요구 시 |

### 3-5. Upstage Enhanced (Category 3 — API VLM 정밀 모드)

| 항목 | 내용 |
|---|---|
| **개발사** | Upstage AI |
| **모델 크기** | 비공개 |
| **실행 환경** | 🌐 API |
| **핵심 라이브러리** | `requests` + `fitz` |
| **라이선스** | 상용 API |
| **기술 특성** | Upstage Document Parse의 VLM 정밀(enhanced) 모드 |
| **강점** | 텍스트형에서 Upstage 기본과 근접한 성능 (NED 0.8430) |
| **약점** | ⚠️ **그래프형 NED 0.1118로 급락** — 기본 Upstage 대비 대폭 저하. 그래프 문서에 사용 금지 |
| **적합 용도** | 텍스트/표형에만 제한적 사용 권장 |

### 3-6. PaddleOCR-VL (Category 3 — 로컬 VLM)

| 항목 | 내용 |
|---|---|
| **개발사** | Baidu (PaddlePaddle) |
| **모델 크기** | **0.9B** (공개) |
| **실행 환경** | 💻 로컬 (K8s CPU 서비스) |
| **핵심 라이브러리** | `paddleocr` (PPStructureV3) |
| **라이선스** | Apache-2.0 |
| **기술 특성** | PaddleOCR-VL-1.5, OmniDocBench v1.5에서 94.5% SOTA 달성. OpenAI-compatible API 제공 |
| **강점** | **종합 NED 1위 (0.7594)**, 그래프형 압도적 1위, 비용 0원, 데이터 보안 완전 확보 |
| **약점** | CPU 기준 느림 (~1.5min/page), 메모리 8Gi+ 필요, 표 TEDS는 Upstage 대비 열위 |
| **적합 용도** | 보안·비용 우선 환경, 그래프/차트 포함 문서 |

### 3-7. MinerU Pipeline (Phase 5 예정)

| 항목 | 내용 |
|---|---|
| **개발사** | OpenDataLab (Shanghai AI Lab) |
| **모델 크기** | 복합 (레이아웃 + OCR + 표 인식 파이프라인) |
| **실행 환경** | 💻 로컬 (CPU 가능) |
| **핵심 라이브러리** | `magic-pdf` |
| **라이선스** | AGPL-3.0 |
| **기술 특성** | PDF-Extract-Kit 기반 end-to-end 파이프라인. Pipeline/Hybrid/VLM 3가지 백엔드 모드 |
| **예상 성능** | NED 0.80+ / TEDS 0.70+ (OmniDocBench 기준 추정) |
| **상태** | ⬚ `backends/mineru.py` 미구현, Phase 5 대기 |

---

## 4. 종합 순위표 (전체 평균)

> NED: 텍스트 일치도 (0~1, 높을수록 좋음) | TEDS: 표 구조 정확도 (0~1, 높을수록 좋음)
>
> †K8s Phase1 결과 (run-id: 20260227-1118), 로컬 재현 불가 (docling: transformers 호환 문제)

| 순위 | 백엔드 | 유형 | 전체 평균 NED | 전체 평균 TEDS | 비고 |
|:---:|---|:---:|:---:|:---:|---|
| **1** | **PaddleOCR-VL** | 💻 로컬 | **0.7594** | 0.5586 | 그래프 압도적 1위, 종합 NED 1위 |
| **2** | **Upstage** | 🌐 API | 0.6937 | **0.6185** | 텍스트·표 NED 1위, 표 TEDS 1위 |
| **3** | **OpenAI GPT-4o** | 🌐 API | 0.5890 | 0.2859 | 표 TEDS 저조 |
| **4** | **PyMuPDF** | 💻 로컬 | 0.5463 | 0.4500 | K8s 결과† |
| **5** | **Docling** | 💻 로컬 | 0.4954 | 0.4400 | K8s 결과†, 로컬 재현 불가 |
| **6** | **Upstage Enhanced** | 🌐 API | 0.4855 | 0.5157 | 그래프형 NED 0.11로 폭락 |

---

## 5. 점수 / 모델 크기 효율 분석

파라미터가 공개된 모델에 대해 **NED ÷ 파라미터수 (10억 단위)** 비율을 계산하여 모델 크기 대비 효율성을 비교한다.

| 백엔드 | 파라미터 | 전체 NED | NED / B params | 효율 순위 |
|---|:---:|:---:|:---:|:---:|
| **PaddleOCR-VL** | 0.9B | 0.7594 | **0.8438** | 🥇 1 |
| **OpenAI GPT-4o** | ~200B (추정) | 0.5890 | 0.0029 | — (참고) |
| **PyMuPDF** | N/A (규칙) | 0.5463 | — | 모델 없음 |
| **Docling** | N/A (복합) | 0.4954 | — | 내부 모델 복합 |
| **Upstage** | 비공개 | 0.6937 | — | 비공개 |
| **Upstage Enhanced** | 비공개 | 0.4855 | — | 비공개 |

**효율성 분석 요약:**

- **PaddleOCR-VL (0.9B)** 은 1B 미만의 초경량 모델로 종합 NED 1위를 달성. 파라미터 대비 효율이 압도적
- OpenAI GPT-4o는 ~200B 추정 대비 NED 0.5890으로, 파라미터 효율 관점에서는 저조
- API 모델(Upstage, Upstage Enhanced)은 파라미터 비공개로 효율 비교 불가
- 규칙 기반(PyMuPDF)과 파이프라인(Docling)은 단일 AI 모델이 아니므로 비교 대상에서 제외

---

## 6. 문서 유형별 전체 순위

### 6-1. 📄 텍스트형

| 순위 | 백엔드 | NED | TEDS |
|:---:|---|:---:|:---:|
| 1 | **Upstage** ★ | 0.8493 | 0.5530 |
| 2 | Upstage Enhanced | 0.8430 | 0.5501 |
| 3 | PaddleOCR-VL | 0.8253 | 0.5510 |
| 4 | Docling | 0.7431 | 0.5348 |
| 5 | OpenAI GPT-4o | 0.7192 | 0.4288 |
| 6 | PyMuPDF | 0.6577 | 0.4928 |

> 상위 3개(Upstage, Upstage Enhanced, PaddleOCR-VL) 간 NED 차이 2.4%p 이내 — 실질적 동등 구간

### 6-2. 📊 표형 (native + image × 4 DPI 평균)

| 순위 | 백엔드 | NED | TEDS |
|:---:|---|:---:|:---:|
| 1 | **Upstage** ★ | 0.8068 | **0.6316** |
| 2 | Upstage Enhanced | 0.7877 | 0.5088 |
| 3 | PaddleOCR-VL | 0.7798 | 0.5601 |
| 4 | PyMuPDF | 0.6269 | 0.4072 |
| 5 | OpenAI GPT-4o | 0.5927 | 0.2573 |
| 6 | Docling | 0.4848 | 0.3926 |

> Upstage가 NED·TEDS 모두 1위. PaddleOCR-VL은 NED 기준 Upstage와 2.7%p 차이로 근접

### 6-3. 📈 그래프형 (graph_rich + image × 4 DPI 평균)

> 그래프형 GT에 마크다운 표 없음 → TEDS 전 백엔드 N/A (설계 의도)

| 순위 | 백엔드 | NED |
|:---:|---|:---:|
| 1 | **PaddleOCR-VL** ★ | 0.7259 |
| 2 | OpenAI GPT-4o | 0.5592 |
| 3 | Upstage | 0.5494 |
| 4 | Docling | 0.3821 |
| 5 | PyMuPDF | 0.3543 |
| 6 | Upstage Enhanced ⚠️ | 0.1118 |

> PaddleOCR-VL이 2위 대비 +16.7%p로 압도적 1위. Upstage Enhanced는 0.1118로 사용 금지 수준

---

## 7. 세부 결과표 (11 PDF × 6 백엔드)

> NED 기준 정렬 | TEDS N/A = 해당 PDF에 표 없음 (graph_rich 계열)
> `—` = 해당 백엔드에서 처리 불가 (PyMuPDF: 이미지 PDF 미지원, Docling: DPI 변형 미지원)

### 7-1. 📄 텍스트형

| PDF | PyMuPDF NED | PyMuPDF TEDS | Docling NED | Docling TEDS | OpenAI NED | OpenAI TEDS | Upstage NED | Upstage TEDS | Enh. NED | Enh. TEDS | PaddleOCR NED | PaddleOCR TEDS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| text_only | 0.6577 | 0.4928 | 0.7431 | 0.5348 | 0.7192 | 0.4288 | **0.8493** | 0.5530 | 0.8430 | 0.5501 | 0.8253 | 0.5510 |

### 7-2. 📊 표형

| PDF | PyMuPDF NED | Docling NED | OpenAI NED | Upstage NED | Enh. NED | PaddleOCR NED | Upstage TEDS | Enh. TEDS | PaddleOCR TEDS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| table_native | 0.6269 | 0.6995 | 0.6941 | **0.8098** | 0.7825 | 0.7735 | **0.6423** | 0.5181 | 0.5010 |
| table_image | — | 0.2700 | 0.7010 | **0.8080** | 0.7946 | 0.7766 | **0.6437** | 0.5095 | 0.5787 |
| table_image (72dpi) | — | — | 0.6352 | **0.7990** | 0.7833 | 0.7745 | **0.5849** | 0.4911 | 0.5585 |
| table_image (150dpi) | — | — | 0.2569 | **0.8087** | 0.7820 | 0.7980 | **0.6437** | 0.5151 | 0.5838 |
| table_image (200dpi) | — | — | 0.6764 | **0.8087** | 0.7962 | 0.7762 | **0.6435** | 0.5101 | 0.5785 |

### 7-3. 📈 그래프형 (NED)

| PDF | PyMuPDF | Docling | OpenAI | Upstage | Enh. | PaddleOCR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| graph_rich | 0.3543 | 0.6242 | 0.5647 | 0.6478 | 0.1141 ⚠️ | **0.7280** |
| graph_rich_image | — | 0.1400 | 0.5886 | 0.5329 | 0.1120 | **0.7281** |
| graph_rich_image (72dpi) | — | — | 0.4867 | 0.5294 | 0.1011 | **0.7096** |
| graph_rich_image (150dpi) | — | — | 0.5696 | 0.4955 | 0.1147 | **0.7285** |
| graph_rich_image (200dpi) | — | — | 0.5863 | 0.5415 | 0.1169 | **0.7355** |

---

## 8. DPI 영향 분석

### 8-1. 표형 — DPI별 NED 변화

| DPI | Upstage | Enh. | PaddleOCR | OpenAI |
|:---:|:---:|:---:|:---:|:---:|
| 기본 (image) | 0.8080 | 0.7946 | 0.7766 | 0.7010 |
| 72 | 0.7990 | 0.7833 | 0.7745 | 0.6352 |
| 150 | 0.8087 | 0.7820 | 0.7980 | **0.2569** ⚠️ |
| 200 | 0.8087 | 0.7962 | 0.7762 | 0.6764 |

**분석:**
- **Upstage / PaddleOCR-VL**: DPI 변화에 거의 무관 (±1~2%p 이내). 매우 안정적
- **Upstage Enhanced**: 소폭 변동 있으나 안정적 (±1.3%p)
- **OpenAI GPT-4o**: 150dpi에서 NED 0.2569로 **이상치 발생** — DPI 민감도 높음, 신뢰도 낮음

### 8-2. 그래프형 — DPI별 NED 변화

| DPI | PaddleOCR | Upstage | OpenAI | Enh. |
|:---:|:---:|:---:|:---:|:---:|
| 기본 (image) | 0.7281 | 0.5329 | 0.5886 | 0.1120 |
| 72 | 0.7096 | 0.5294 | 0.4867 | 0.1011 |
| 150 | 0.7285 | 0.4955 | 0.5696 | 0.1147 |
| 200 | 0.7355 | 0.5415 | 0.5863 | 0.1169 |

**분석:**
- **PaddleOCR-VL**: 전 DPI 구간 0.71~0.74로 극도로 안정적 (±2.6%p)
- **Upstage**: 150dpi에서 소폭 하락 (0.4955), 그 외 안정적
- **OpenAI GPT-4o**: 72dpi에서 0.4867로 하락, 100dpi 이상에서 회복
- **Upstage Enhanced**: 전 DPI 구간 0.10~0.12로 일관되게 **저조** — DPI 무관하게 그래프 처리 불가

---

## 9. 모델 선택 근거

### 9-1. 문서 유형별 추천

| 문서 유형 | 1순위 | 2순위 | 선택 근거 |
|---|---|---|---|
| 📄 **텍스트형** | **Upstage** | PaddleOCR-VL (보안 우선) | NED 0.8493, 상위 3종 2.4%p 이내 동등 구간 |
| 📊 **표형** | **Upstage** | PaddleOCR-VL (보안 우선) | NED 0.8068, TEDS 0.6316 — NED·TEDS 모두 1위 |
| 📈 **그래프형** | **PaddleOCR-VL** | OpenAI GPT-4o | NED 0.7259 — 2위 대비 +16.7%p 압도적 |

### 9-2. 관점별 분석

#### 비용 관점

| 백엔드 | 비용 | 평가 |
|---|---|---|
| PyMuPDF | 무료 | 가장 경제적이나 이미지 문서 불가 |
| Docling | 무료 | 로컬 실행 문제 해결 시 좋은 선택 |
| PaddleOCR-VL | 무료 | **비용 대비 성능 최적** — 종합 1위이면서 무료 |
| Upstage | 종량제 API | 텍스트·표 최고 정확도, 비용 발생 |
| Upstage Enhanced | 종량제 API | 그래프형 성능 문제로 비용 대비 효용 낮음 |
| OpenAI GPT-4o | 종량제 API | 가장 느리고 비싼데, 성능도 중위권 |

#### 보안 관점

| 구분 | 백엔드 | 데이터 유출 위험 |
|---|---|---|
| 🔒 안전 | PyMuPDF, Docling, PaddleOCR-VL | 외부 전송 없음. 사내 배포 가능 |
| ⚠️ 주의 | Upstage, Upstage Enhanced, OpenAI | 문서 데이터가 외부 API로 전송됨 |

#### 속도 관점

> ⚠️ 속도는 환경(네트워크/서버 부하)에 따라 편차가 큼. 순위 판단에 미반영.

| 백엔드 | 텍스트형 | 표형 (avg) | 그래프형 (avg) |
|---|:---:|:---:|:---:|
| PyMuPDF | ~2s | ~2s | ~2s |
| Docling | ~45s | ~65s | ~60s |
| Upstage | <1s | 17s | 17s |
| Upstage Enhanced | 18s | 24s | 23s |
| PaddleOCR-VL | 1.8min | 1.7min | 58s |
| OpenAI GPT-4o | 3.2min | 4.5min | 1.6min |

### 9-3. 종합 추천

| 시나리오 | 추천 백엔드 | 근거 |
|---|---|---|
| **보안·비용 우선** | PaddleOCR-VL | 종합 NED 1위, 무료, 로컬 실행, 전 유형 안정적 |
| **표 정확도 최우선** | Upstage | 표 TEDS 0.6316으로 유일한 0.6+ 달성 |
| **속도 최우선** | PyMuPDF (텍스트) + Upstage (표/그래프) | 라우팅 조합으로 속도-정확도 균형 |
| **그래프·차트 문서** | PaddleOCR-VL | 그래프형 NED 0.7259 — 2위 대비 +16.7%p |
| **혼합 문서 (Hybrid 모드)** | PyMuPDF (텍스트 페이지) + PaddleOCR-VL (비주얼 페이지) | 비용 0원 + 전 유형 커버 |

---

## 10. 속도 참고

> ⚠️ 속도는 환경 의존적 (네트워크 지연, 서버 부하, K8s 노드 상태). 순위 판단에는 미반영하며 참고용으로만 제공한다.

| 백엔드 | 텍스트형 | 표형 (avg) | 그래프형 (avg) | 비고 |
|---|:---:|:---:|:---:|---|
| PyMuPDF | 2s | 2s | 2s | K8s 결과† |
| Docling | 45s | 1.1min | 1.0min | K8s 결과† |
| OpenAI GPT-4o | 3.2min | 4.5min | 1.6min | API 레이턴시 포함 |
| Upstage | <1s | 17s | 17s | API, 30MB 미만 |
| Upstage Enhanced | 18s | 24s | 23s | API |
| PaddleOCR-VL | 1.8min | 1.7min | 58s | K8s CPU 서비스 |

---

## 11. 데이터 출처 / 제약사항

### 데이터 출처

| 백엔드 | Run ID | PDF 수 | 상태 |
|---|---|:---:|---|
| PyMuPDF | `20260227-1118 (K8s)` | 5† | K8s 결과 (부분) |
| Docling | `20260227-1118 (K8s)` | 5† | K8s 결과 (부분), 로컬 재현 불가 |
| OpenAI GPT-4o | `vlm-20260303-1550` | 11 | 완료 |
| Upstage | `upstage-20260303-1635` | 11 | 완료 |
| Upstage Enhanced | `upstage-20260303-1635` | 11 | 완료 |
| PaddleOCR-VL | `paddleocr-20260303-1541` | 11 | 완료 |

### 제약사항

1. **PyMuPDF/Docling 데이터 부분성**: K8s Phase1 결과로 11 PDF 중 5개만 실행됨. 이미지 기반 PDF(DPI 변형 포함)는 PyMuPDF가 처리 불가하여 `—` 표시
2. **Docling 로컬 재현 불가**: `transformers>=4.49.0` 호환 문제로 로컬 실행이 차단됨. K8s 환경에서만 결과 수집
3. **OpenAI 150dpi 이상치**: `table_image_150dpi`에서 NED 0.2569 기록 — 다른 DPI 대비 비정상적 저하. 원인 미확인
4. **Upstage Enhanced 그래프형 급락**: 전 DPI 구간에서 NED 0.10~0.12로 일관되게 저조. 기본 Upstage (0.55) 대비 약 80% 하락
5. **속도 데이터의 환경 의존성**: K8s CPU 환경(management 노드 m7i/m8i.2xlarge) 기준. GPU 환경에서는 PaddleOCR-VL, Docling 속도가 대폭 개선될 수 있음
6. **Phase 5 미반영**: MinerU Pipeline 백엔드가 추가되면 순위 변동 가능 (예상 NED 0.80+, TEDS 0.70+)
7. **파라미터 비공개 모델**: Upstage, Upstage Enhanced는 모델 크기 비공개로 효율 비교 불가. OpenAI GPT-4o는 ~200B 추정치 사용
