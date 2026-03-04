# PDF 파싱 벤치마크 실행 계획

> 작성일: 2026-02-26

---

## 1. 목표

동일한 한국어 PDF 세트에 대해 **파서(백엔드) × 라우팅 모드 × DPI** 조합별 파싱 품질·속도를 정량 비교하여, 문서 유형별 최적 파싱 전략을 도출한다.

### 핵심 질문

1. 어떤 백엔드가 어떤 문서 유형에서 가장 정확한가?
2. DPI 저하가 OCR 정확도에 미치는 영향은?
3. hybrid 라우팅이 document-level 대비 품질/속도 트레이드오프는?
4. 로컬 파서(MinerU, PaddleOCR)가 API 파서(Gemini)를 대체할 수 있는가?

---

## 2. 벤치마크 데이터

### 2.1 생성 완료 (pdf_parser/benchmark_pdfs/)

| 카테고리 | 파일 | 페이지 | 용도 |
|---------|------|--------|------|
| **A. 텍스트 전용** | `text_only.pdf` | 5p | 순수 텍스트 추출 성능 |
| **B. 표 (원본)** | `table_native.pdf` | 5p | 텍스트 기반 표 → **Ground Truth** |
| **C. 표 (래스터)** | `table_image.pdf` / `_200dpi` / `_150dpi` / `_72dpi` | 5p × 4 | DPI별 OCR 표 인식 |
| **E. 그래프 (원본)** | `graph_rich.pdf` | 5p | 차트/다이어그램 → **Ground Truth** |
| **E. 그래프 (래스터)** | `graph_rich_image.pdf` / `_200dpi` / `_150dpi` / `_72dpi` | 5p × 4 | DPI별 이미지 인식 |

**총 11개 PDF, 55페이지**

### 2.2 Ground Truth 전략

- `table_native.pdf` → PyMuPDF 텍스트 추출 결과를 GT로 사용
- `graph_rich.pdf` → 수동 검수 후 GT markdown 확정
- `text_only.pdf` → PyMuPDF 추출 결과를 GT로 사용

---

## 3. 벤치마크 매트릭스

### 3.1 백엔드 (행)

| 백엔드 | 코드 | 의존성 | GPU | 상태 |
|--------|------|--------|-----|------|
| **PyMuPDF4LLM** | `category1_simple.py` | pymupdf4llm | 불필요 | 구현 완료 |
| **Docling** | `category2_medium.py` | docling, tesseract-ocr-kor | 불필요 | 구현 완료 |
| **Gemini VLM** | `category3_complex.py` | google-genai | 불필요 (API) | 구현 완료 |
| **MinerU** | `backends/mineru.py` | magic-pdf | 권장 | **신규 구현** |
| **PaddleOCR** | `backends/paddleocr.py` | paddleocr>=3.0 | 권장 | **신규 구현** |

### 3.2 라우팅 모드 (열)

| 모드 | 설명 |
|------|------|
| **document** | 첫 페이지 분석 → 전체 문서에 단일 백엔드 적용 |
| **hybrid** | 페이지별 분석 → text/vlm 자동 라우팅 |
| **direct** | 라우팅 없이 지정 백엔드를 직접 적용 (**벤치마크 전용**) |

### 3.3 전체 조합

```
Phase 1 (기존 백엔드, direct 모드):
  3 backends × 11 PDFs = 33 Jobs

Phase 2 (신규 백엔드 추가 후):
  5 backends × 11 PDFs = 55 Jobs

Phase 3 (라우팅 모드 비교):
  3 modes × 3 backends × 3 PDF 카테고리 = 27 Jobs
```

---

## 4. 평가 지표

| 지표 | 측정 대상 | 계산 방식 | GT 필요 |
|------|-----------|-----------|---------|
| **Text NED** | 텍스트 정확도 | `1 - (edit_distance / max_len)` | Yes |
| **Table TEDS** | 표 구조 정확도 | Tree Edit Distance-based Similarity | Yes |
| **Speed** | 처리 속도 | sec/page | No |
| **Word Count** | 추출량 | 단어 수 | No |
| **Structure** | 구조 보존 | headers, tables, formulas 유무 | No |

### 평가 모듈 설계 (benchmark/evaluator.py)

```python
@dataclass
class PageScore:
    page: int
    text_ned: float      # 0.0 ~ 1.0 (1.0 = 완벽)
    table_teds: float    # 0.0 ~ 1.0
    speed_s: float       # 초
    word_count: int

@dataclass
class BenchResult:
    backend: str
    pdf_name: str
    mode: str
    pages: list[PageScore]
    avg_text_ned: float
    avg_table_teds: float
    avg_speed: float
    total_time_s: float
```

---

## 5. K8s 아키텍처

### 5.1 기존 인프라 재사용

| 컴포넌트 | 위치 | 재사용 여부 |
|----------|------|------------|
| Namespace `pdf-parser` | `k8s/manifests/namespace.yaml` | 그대로 사용 |
| PVC `pdf-storage` (EFS RWX 20Gi) | `k8s/manifests/pvc.yaml` | 그대로 사용 |
| Job Template | `k8s/manifests/job-template.yaml` | **확장** (BACKEND, EVAL_MODE 추가) |
| Orchestrator | `k8s/orchestrator.py` | **확장** (벤치마크 모드 추가) |
| Dockerfile | `k8s/Dockerfile` | **확장** (MinerU, PaddleOCR 추가) |
| Secret `parser-secrets` | K8s Secret | 그대로 (GEMINI_API_KEY) |

### 5.2 Job 설계

```
단일 Phase (prep/bench 분리 불필요 — 데이터가 작음)

Orchestrator (로컬)
  │
  ├─ 1. PVC 업로드: benchmark_pdfs/ + ground_truth/
  │     /data/bench/pdfs/*.pdf
  │     /data/bench/gt/*.md
  │
  ├─ 2. 조합별 Job 생성 (병렬)
  │     pdf-bench-{backend}-{pdf_category}-{run_id}
  │     예: pdf-bench-docling-tableimage300-20260227-1000
  │
  ├─ 3. Poll 대기 (15s 간격, DONE 감시)
  │
  └─ 4. 결과 수집 → 로컬 병합 → 리포트 생성
```

### 5.3 Worker 환경변수 (확장)

| 환경변수 | 값 | 설명 |
|---------|-----|------|
| `BACKEND` | pymupdf / docling / gemini / mineru / paddleocr | 사용할 파서 |
| `PDF_PATH` | `/data/bench/pdfs/table_image.pdf` | 대상 PDF |
| `GT_PATH` | `/data/bench/gt/table_native.md` | Ground Truth 경로 (선택) |
| `OUTPUT_DIR` | `/data/bench/results/{backend}/{pdf_name}` | 결과 저장 |
| `BATCH_ID` | `20260227-1000` | 실행 ID |
| `PARSE_MODE` | direct / document / hybrid | 라우팅 모드 |
| `GEMINI_API_KEY` | Secret 주입 | Gemini 전용 |

### 5.4 PVC 디렉토리 구조

```
/data/bench/
├── pdfs/                          # 벤치마크 PDF (업로드)
│   ├── text_only.pdf
│   ├── table_native.pdf
│   ├── table_image.pdf
│   ├── table_image_200dpi.pdf
│   ├── table_image_150dpi.pdf
│   ├── table_image_72dpi.pdf
│   ├── graph_rich.pdf
│   ├── graph_rich_image.pdf
│   ├── graph_rich_image_200dpi.pdf
│   ├── graph_rich_image_150dpi.pdf
│   └── graph_rich_image_72dpi.pdf
├── gt/                            # Ground Truth markdown (업로드)
│   ├── text_only.md
│   ├── table_native.md
│   └── graph_rich.md
└── results/                       # 실행 결과 (워커 기록)
    └── {run_id}/
        └── {backend}-{pdf_name}/
            ├── output.md          # 파싱 결과
            ├── metrics.json       # 평가 결과
            └── DONE               # 완료 신호
```

### 5.5 Job 이름 규칙

```
pdf-bench-{backend}-{pdf_category}-{run_id}

예시:
  pdf-bench-pymupdf-textonly-20260227-1000        (27자)
  pdf-bench-docling-tableimage300-20260227-1000    (44자)
  pdf-bench-paddleocr-graphrich72-20260227-1000    (45자)

최대 길이: 63자 이내 ✓
```

### 5.6 결과 JSON 스키마 (metrics.json)

```json
{
  "backend": "docling",
  "pdf_name": "table_image.pdf",
  "mode": "direct",
  "run_id": "20260227-1000",
  "pages": [
    {
      "page": 1,
      "text_ned": 0.92,
      "table_teds": 0.85,
      "speed_s": 2.3,
      "word_count": 450
    }
  ],
  "summary": {
    "avg_text_ned": 0.91,
    "avg_table_teds": 0.83,
    "avg_speed_s": 2.1,
    "total_time_s": 10.5,
    "total_words": 2100
  },
  "timestamp": "2026-02-27 10:15:30"
}
```

---

## 6. 구현 Phase 계획

### Phase 0: Ground Truth 준비

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P0-1 | text_only.pdf를 PyMuPDF로 추출 → GT markdown 생성 | `gt/text_only.md` | 없음 |
| P0-2 | table_native.pdf를 PyMuPDF로 추출 → GT markdown 생성 | `gt/table_native.md` | 없음 |
| P0-3 | graph_rich.pdf를 Gemini VLM으로 추출 → 수동 검수 → GT | `gt/graph_rich.md` | 없음 |

**검증**: GT 파일이 원본 PDF와 1:1 대응하는지 페이지 수 확인

---

### Phase 1: 평가 모듈 + 벤치마크 러너

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P1-1 | `benchmark/evaluator.py` — Text NED, Table TEDS 계산 | evaluator.py | P0 |
| P1-2 | `benchmark/spec.py` — BenchSpec dataclass + PRESETS | spec.py | 없음 |
| P1-3 | `benchmark/runner.py` — 로컬 벤치마크 실행 루프 | runner.py | P1-1, P1-2 |
| P1-4 | 로컬 테스트: 기존 3 백엔드 × text_only.pdf 실행 | 로컬 결과 | P1-3 |

**검증**: `python -m benchmark.runner --preset quick` 로컬 실행 성공

---

### Phase 2: 백엔드 통합 인터페이스 + 신규 백엔드

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P2-1 | `backends/__init__.py` — 통합 인터페이스 정의 | BaseBackend ABC | 없음 |
| P2-2 | 기존 category*.py → backends/ 래퍼 생성 (기존 코드 유지) | backends/pymupdf.py 등 | P2-1 |
| P2-3 | `backends/mineru.py` — MinerU 백엔드 구현 | mineru.py | P2-1 |
| P2-4 | `backends/paddleocr.py` — PaddleOCR 백엔드 구현 | paddleocr.py | P2-1 |
| P2-5 | 로컬 테스트: 5 백엔드 × text_only.pdf | 결과 비교 | P2-2~4 |

**통합 인터페이스:**
```python
class BaseBackend(ABC):
    name: str

    @abstractmethod
    def convert_pdf(self, pdf_path: Path, output_path: Path) -> str:
        """PDF → Markdown 변환. 반환: markdown 문자열."""

    @abstractmethod
    def convert_page(self, page: fitz.Page, page_num: int) -> str:
        """단일 페이지 변환 (hybrid 모드용)."""
```

**검증**: 5개 백엔드 모두 동일 PDF에 대해 .md 출력 생성

---

### Phase 3: K8s 벤치마크 워커 + Dockerfile 확장

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P3-1 | `k8s/bench_entrypoint.py` — 벤치마크 전용 워커 | bench_entrypoint.py | P1-3, P2-2 |
| P3-2 | `k8s/manifests/bench-job-template.yaml` — 벤치마크 Job 템플릿 | bench-job-template.yaml | 없음 |
| P3-3 | Dockerfile 확장 — MinerU, PaddleOCR 의존성 추가 | Dockerfile.bench | P2-3, P2-4 |
| P3-4 | Docker 이미지 빌드 + Harbor push | 이미지 태그 | P3-1~3 |

**bench_entrypoint.py 흐름:**
```
1. ENV 읽기: BACKEND, PDF_PATH, GT_PATH, OUTPUT_DIR
2. 백엔드 초기화 (backends/{BACKEND}.py)
3. PDF 파싱 실행 + 시간 측정
4. GT 존재 시 → evaluator로 NED/TEDS 계산
5. metrics.json 원자적 기록
6. DONE 파일 생성
```

**검증**: `--dry-run`으로 Job YAML 생성 → 수동 검토

---

### Phase 4: 오케스트레이터 확장 + K8s 실행

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P4-1 | orchestrator.py에 `--bench` 모드 추가 | 오케스트레이터 확장 | P3-2 |
| P4-2 | PVC에 벤치마크 데이터 + GT 업로드 | PVC 데이터 | P0, P4-1 |
| P4-3 | Phase 1 실행: 기존 3 백엔드 × 11 PDF (33 Jobs) | K8s 결과 | P4-2 |
| P4-4 | Phase 2 실행: 신규 2 백엔드 추가 (22 Jobs 추가) | K8s 결과 | P2-3~4 |

**오케스트레이터 CLI:**
```bash
# 벤치마크 모드
python orchestrator.py --bench \
  --image harbor.../pdf-parser/bench-worker:v1 \
  --preset standard \
  --output ./bench_results

# 특정 조합만
python orchestrator.py --bench \
  --backend docling \
  --pdf table_image.pdf \
  --output ./bench_results
```

**검증**: 33 Jobs 중 succeeded ≥ 30 (90% 이상 성공)

---

### Phase 5: 결과 수집 + 리포트

| ID | 작업 | 산출물 | 의존성 |
|----|------|--------|--------|
| P5-1 | `benchmark/report.py` — 결과 병합 + 비교 테이블 생성 | report.py | P1-1 |
| P5-2 | 결과 수집 (collector pod → kubectl cp) | 로컬 결과 | P4-3 |
| P5-3 | 리포트 생성: 백엔드별 × 문서유형별 NED/TEDS/Speed 히트맵 | report.html / report.csv | P5-1~2 |

**리포트 형식:**
```
=== 텍스트 정확도 (Text NED) ===
                text_only  table_300  table_200  table_150  table_72  graph_300  graph_72
pymupdf         0.98       0.95       -          -          -         0.45       -
docling         0.96       0.92       0.90       0.85       0.60      0.70       0.55
gemini          0.97       0.94       0.93       0.91       0.82      0.90       0.78
mineru          0.97       0.93       0.91       0.87       0.65      0.80       0.60
paddleocr       0.95       0.91       0.89       0.84       0.58      0.75       0.55

=== 처리 속도 (sec/page) ===
                text_only  table_300  ...
pymupdf         0.02       0.03
docling         0.45       0.52
gemini          3.50       4.20
mineru          0.21       0.25
paddleocr       0.35       0.40
```

---

## 7. 의존성 그래프

```
P0 (GT 준비)
  │
  ├──→ P1 (평가 모듈 + 러너) ──→ P3 (K8s 워커) ──→ P4 (K8s 실행)
  │                                    │                    │
  └──→ P2 (백엔드 통합) ──────────────┘                    │
                                                            │
                                                      P5 (리포트)
```

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| MinerU/PaddleOCR CPU-only 속도 저하 | 벤치마크 시간 증가 | `--timeout 7200` 확대, ai-platform 노드 활용 검토 |
| Gemini API 과금 | 비용 | graph_rich 5p × 4 DPI = 20 calls 정도, 소규모 |
| PVC EFS 지연 (NFS) | DONE 감지 지연 | poll 간격 15s로 충분, 필요시 30s |
| Docling 모델 다운로드 | 첫 실행 느림 | model-cache PVC 마운트 또는 이미지에 포함 |
| 63자 Job 이름 초과 | Job 생성 실패 | `_safe_label()` 적용 + 길이 검증 |

---

## 9. 클러스터 리소스 계획

| 항목 | 값 |
|------|-----|
| Namespace | `pdf-parser` |
| 동시 Job 수 | 최대 6 (management 노드 2대 × CPU 3 여유) |
| CPU request/Job | 1 core |
| Memory request/Job | 4Gi (Docling/MinerU), 2Gi (PyMuPDF/Gemini) |
| PVC | `pdf-storage` 20Gi (기존) |
| 예상 총 실행시간 | Phase 1: ~2시간 (33 Jobs, 6 병렬) |
| 이미지 레지스트리 | Harbor (기존) |

---

## 10. 성공 기준

- [ ] 5개 백엔드 × 11개 PDF 조합 중 90% 이상 성공 (≥50/55)
- [ ] Text NED, Speed 지표가 모든 성공 조합에 대해 산출됨
- [ ] 표 포함 PDF에 대해 Table TEDS 지표 산출 (최소 table_native 시리즈)
- [ ] 백엔드별 × 문서유형별 비교 리포트 생성
- [ ] DPI 저하에 따른 정확도 변화 그래프 도출
