# PDF Parser K8s 배포

PDF → Markdown 변환을 K8s Job으로 병렬 실행.

## 아키텍처

```
로컬 PDF 폴더
     │ kubectl cp (upload Pod)
     ▼
┌─ PVC (pdf-storage, EFS RWX) ──────────────────────┐
│  /data/input/{batch_id}/*.pdf                      │
│  /data/output/{batch_id}/*.md + result.json + DONE │
└────────────────────────────────────────────────────┘
     │                           ▲
     ▼                           │
┌─ Job: pdf-parse-{batch_id} ──────────────┐
│  worker_entrypoint.py                     │
│  ├─ PDF 분류 (simple/medium/complex)     │
│  ├─ Category 1: PyMuPDF4LLM (text)      │
│  ├─ Category 2: Docling (OCR+table)     │
│  ├─ Category 3: Gemini VLM (visual)     │
│  └─ Hybrid: 페이지별 자동 선택           │
│  → Markdown + quality metrics JSON 저장  │
└──────────────────────────────────────────┘
```

## 빠른 시작

```bash
# 1. 인프라 생성
kubectl apply -f pdf_parser/k8s/manifests/namespace.yaml
kubectl apply -f pdf_parser/k8s/manifests/pvc.yaml

# 2. Secret 생성 (VLM 모드 사용 시)
kubectl create secret generic parser-secrets \
    -n pdf-parser \
    --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY"

# 3. 이미지 빌드
docker buildx build --platform linux/amd64 --push \
    -t $REGISTRY/pdf-parser/worker:latest \
    -f pdf_parser/k8s/Dockerfile .

# 4. 실행 (로컬 PDF 업로드 → 변환 → 결과 수집)
python pdf_parser/k8s/orchestrator.py \
    --image $REGISTRY/pdf-parser/worker:latest \
    --upload ./my_pdfs

# 5. dry-run
python pdf_parser/k8s/orchestrator.py \
    --image $IMAGE --batch-id test --dry-run
```

## 파일 구조

```
pdf_parser/k8s/
├── Dockerfile                 # 멀티스테이지 워커 이미지 (tesseract OCR 포함)
├── requirements-worker.txt    # 워커 의존성 (pymupdf4llm, docling, google-genai)
├── worker_entrypoint.py       # K8s Job 엔트리포인트
├── orchestrator.py            # Job 생성/모니터링/수집 오케스트레이터
├── README.md                  # 이 문서
└── manifests/
    ├── namespace.yaml         # pdf-parser 네임스페이스
    ├── pvc.yaml               # PDF 저장소 PVC (EFS RWX)
    └── job-template.yaml      # Job 템플릿
```

## 리소스 설정

| 모드 | CPU req/lim | MEM req/lim | 비고 |
|------|-------------|-------------|------|
| hybrid (기본) | 1/2 | 4Gi/8Gi | Docling OCR + VLM API |
| document | 1/1 | 2Gi/4Gi | 단일 백엔드만 사용 |

## 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `INPUT_DIR` | PDF 입력 경로 | `/data/input/{batch_id}` |
| `OUTPUT_DIR` | Markdown 출력 경로 | `/data/output/{batch_id}` |
| `PARSE_MODE` | `hybrid` / `document` | `hybrid` |
| `GEMINI_API_KEY` | Gemini VLM API 키 | Secret에서 주입 |
| `BATCH_ID` | 배치 식별자 | 오케스트레이터가 주입 |
| `FILE_PATTERN` | 처리 파일 패턴 | `*.pdf` |

## TODO

- [ ] `--split N` 배치 분할 시 PDF 파일 분배 로직
- [ ] Harbor ImagePullSecret 공유 또는 신규 생성
- [ ] Docling 모델 캐시 PVC (첫 실행 시 다운로드 방지)
- [ ] 비동기 페이지 병렬 처리 (hybrid 모드 내)
- [ ] 결과 Markdown을 RAG 벤치마크 파이프라인에 직접 전달하는 연동
