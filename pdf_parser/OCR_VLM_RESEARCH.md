# OCR VLM 벤치마크 — 구현 방법 리서치

> 작성일: 2026-02-26
> 환경: K8s CPU (m7i/m8i.2xlarge, 8C/32G), `rag-bench-test` 네임스페이스
> 요구사항: Together AI 사용 안 함, K8s 로컬 실행, GPU 없음

---

## 1. 대상 모델 확정

| 백엔드 키 | 모델 | 파라미터 | HuggingFace ID | 서빙 방식 |
|---|---|---|---|---|
| `granite-vision` | Granite Vision 3.3 2B | ~2B | `ibm-granite/granite-vision-3.3-2b` | Ollama (`ibm/granite3.3-vision:2b`) |
| `got-ocr2` | GOT-OCR2.0 | 580M | `stepfun-ai/GOT-OCR-2.0-hf` | FastAPI (HF transformers 4.49+) |
| `paddleocr-vl` | PaddleOCR-VL-1.5 | 0.9B | `PaddlePaddle/PaddleOCR-VL-1.5` | FastAPI (paddleocr 패키지) |

> **DeepSeek-OCR-2 제외**: bf16 6.7GB 전용 (Q4 미출시), 메모리 부담 과다로 제외

---

## 2. 모델별 CPU 실행 특성

### 2.1 PaddleOCR-VL-1.5 (0.9B)

- **추론 방법 A** (권장): `paddleocr` pip 패키지의 `PPStructureV3` 파이프라인
  ```python
  from paddleocr import PPStructureV3
  pipeline = PPStructureV3(device="cpu")
  output = pipeline.predict(input="page.png")
  markdown = pipeline.concatenate_markdown_pages([r.markdown for r in output])
  ```
- **추론 방법 B**: HuggingFace `AutoModelForImageTextToText` (**transformers >= 5.0 필요** → 기존 환경 충돌)
- **CPU 공식 지원**: 공식 문서상 미지원 표기, `device="cpu"` 로 동작 확인됨
- **CPU 추론 속도**: Intel 8350C 기준 ~3.74s/페이지
- **메모리**: ~3–4GB (FP32)
- **설치**:
  ```bash
  pip install paddlepaddle          # CPU 버전 (paddlepaddle-gpu 아님)
  pip install "paddleocr[doc-parser]"
  ```
- **⚠️ 핵심 제약**: transformers v5 방법은 기존 GOT-OCR2(v4.45+) 환경과 충돌 → **별도 Docker 이미지 필수**

### 2.2 GOT-OCR2.0 (580M)

- **추론 방법**: HuggingFace transformers 직접 로드 (Ollama **미지원**)
  ```python
  from transformers import AutoProcessor, AutoModelForImageTextToText
  model = AutoModelForImageTextToText.from_pretrained(
      "stepfun-ai/GOT-OCR-2.0-hf",
      torch_dtype=torch.float32,   # CPU: float32
      device_map="cpu",
  ).eval()
  ```
- **CPU 공식 지원**: ✓ (HF에 CPU 전용 버전 별도 존재: `RufusRubin777/GOT-OCR2_0_CPU`)
- **메모리**: ~2–2.5GB (FP32)
- **transformers 버전**: >= 4.49 (v4 범위, 기존 환경 호환 ✓)
- **서빙**: FastAPI OpenAI-compatible 래퍼로 HTTP 엔드포인트 제공
- **마크다운 출력**: `ocr_type='format'`

### 2.3 Granite Vision 3.3 2B (~2B)

- **추론 방법**: Ollama `ibm/granite3.3-vision:2b`
  ```bash
  ollama pull ibm/granite3.3-vision:2b
  ```
  → OpenAI-compatible API
- **Ollama 모델 크기**: Q8_0 ~2.7GB + mmproj ~0.89GB = **총 ~3.6GB**
- **CPU 메모리**: ~4–5GB
- **transformers 버전**: >= 4.49 (v4 범위 ✓)
- **특이사항**: Docling 공식 Vision 백엔드, OCRBench 7B 미만 2위

---

## 3. K8s 아키텍처 설계

### 3.1 구조

```
rag-bench-test 네임스페이스
│
├── Deployment: ollama-server
│   ├── 모델: ibm/granite3.3-vision:2b  (~3.6GB)
│   ├── 모델: deepseek-ocr:3b           (6.7GB, 순차 로드)
│   ├── Service: ollama-server:11434    → http://ollama-server:11434/v1
│   └── PVC: ollama-model-cache (30Gi EFS RWX)
│
├── Deployment: got-ocr2-server
│   ├── FastAPI OpenAI-compatible 서버 (transformers 4.49+)
│   ├── HF 모델: stepfun-ai/GOT-OCR-2.0-hf (~2.5GB)
│   ├── Service: got-ocr2-server:8000   → http://got-ocr2-server:8000/v1
│   └── PVC: model-cache (기존 50Gi 공유)
│
├── Deployment: paddleocr-vl-server
│   ├── FastAPI OpenAI-compatible 서버 (transformers 5.0 격리 이미지)
│   ├── HF 모델: PaddlePaddle/PaddleOCR-VL-1.5 (~3GB)
│   ├── Service: paddleocr-vl-server:8000 → http://paddleocr-vl-server:8000/v1
│   └── PVC: model-cache (기존 50Gi 공유)
│
└── K8s Jobs: pdf-parser-bench-*
    └── category3_opensource.py → 각 서비스 OpenAI API 호출
```

### 3.2 메모리 계획 (32GB 노드)

```
OS + K8s 시스템:          ~2.0 GB
Ollama (Granite, 상시):   ~4.5 GB  (3.6GB + 오버헤드)
GOT-OCR2 서버 (상시):     ~3.0 GB
PaddleOCR-VL 서버 (상시): ~4.0 GB
워커 Job × 5 (병렬):      ~2.5 GB
────────────────────────────────
합계:                     ~16.0 GB ← 32GB 노드에서 여유 ✓
```

**핵심 Ollama 설정**:
```yaml
- name: OLLAMA_MAX_LOADED_MODELS
  value: "1"      # CPU에서 모델 1개만 메모리 유지
- name: OLLAMA_KEEP_ALIVE
  value: "30m"    # Job 간 재로드 방지
- name: OLLAMA_NUM_PARALLEL
  value: "1"      # CPU 순차 처리
```

---

## 4. MODEL_PROFILES 설계

```python
# category3_opensource.py
MODEL_PROFILES: dict[str, tuple[str, str]] = {
    # Ollama 서빙 (K8s 내부)
    "granite-vision": ("ibm/granite3.3-vision:2b",  "http://ollama-server:11434/v1"),
    # FastAPI 서빙 (별도 Deployment)
    "got-ocr2":       ("got-ocr2",                  "http://got-ocr2-server:8000/v1"),
    "paddleocr-vl":   ("paddleocr-vl-1.5",          "http://paddleocr-vl-server:8000/v1"),
}
```

---

## 5. 필요 신규 파일

| 파일 경로 | 역할 |
|---|---|
| `k8s/got_ocr2_server.py` | GOT-OCR2 FastAPI OpenAI-compatible 서버 |
| `k8s/paddleocr_vl_server.py` | PaddleOCR-VL FastAPI 서버 (transformers 5.0) |
| `k8s/Dockerfile.got-ocr2` | GOT-OCR2 서버 이미지 (transformers 4.49+) |
| `k8s/Dockerfile.paddleocr-vl` | PaddleOCR-VL 서버 이미지 (transformers 5.0 격리) |
| `k8s/manifests/ollama-deployment.yaml` | Ollama K8s Deployment + Service + PVC |
| `k8s/manifests/got-ocr2-deployment.yaml` | GOT-OCR2 K8s Deployment + Service |
| `k8s/manifests/paddleocr-vl-deployment.yaml` | PaddleOCR-VL K8s Deployment + Service |

## 수정 파일

| 파일 경로 | 변경 내용 |
|---|---|
| `pdf_parser/category3_opensource.py` | MODEL_PROFILES 교체 (4개 신규 백엔드) |
| `pdf_parser/benchmark/spec.py` | Backend Literal + OPENSOURCE_BACKENDS + `ocr` 프리셋 |
| `pdf_parser/benchmark/runner.py` | 4개 백엔드 디스패처 추가 |
| `k8s/pdf_parser_entrypoint.py` | 4개 백엔드 디스패처 추가 |
| `k8s/pdf_parser_orchestrator.py` | 백엔드/프리셋 선택지 추가 |
| `k8s/manifests/pdf-parser-job-template.yaml` | OLLAMA_ENDPOINT 환경변수 추가 |

---

## 6. PaddleOCR-VL transformers 버전 충돌 분석

| 컴포넌트 | transformers 버전 |
|---|---|
| GOT-OCR2 | >= 4.45 |
| Granite Vision | >= 4.49 |
| PaddleOCR-VL-1.5 | **>= 5.0.0** (충돌) |
| 기존 워커 환경 | 4.x |

**해결**: PaddleOCR-VL은 전용 Docker 이미지로 완전 격리. 기존 워커 이미지는 수정 없이 HTTP만 호출.

---

## 7. 배포 순서

```bash
# 1. Ollama PVC + Deployment
kubectl apply -f k8s/manifests/ollama-deployment.yaml
kubectl -n rag-bench-test rollout status deployment/ollama-server

# 2. GOT-OCR2 이미지 빌드 (K8s 빌더)
docker buildx build --platform linux/amd64 \
  -t $HARBOR_REGISTRY/rag-bench-test/got-ocr2:latest \
  -f k8s/Dockerfile.got-ocr2 --push .
kubectl apply -f k8s/manifests/got-ocr2-deployment.yaml

# 3. PaddleOCR-VL 이미지 빌드 (K8s 빌더)
docker buildx build --platform linux/amd64 \
  -t $HARBOR_REGISTRY/rag-bench-test/paddleocr-vl:latest \
  -f k8s/Dockerfile.paddleocr-vl --push .
kubectl apply -f k8s/manifests/paddleocr-vl-deployment.yaml

# 4. 서비스 확인
kubectl -n rag-bench-test get pods,svc
kubectl -n rag-bench-test run test --rm -it --image=curlimages/curl -- \
  curl -s http://ollama-server:11434/api/tags
kubectl -n rag-bench-test run test --rm -it --image=curlimages/curl -- \
  curl -s http://got-ocr2-server:8000/health

# 5. 벤치마크 실행
python k8s/pdf_parser_orchestrator.py \
  --image $HARBOR_REGISTRY/rag-bench-test/pdf-parser:latest \
  --preset ocr
```

---

## 8. 미결 사항 / 리스크

| 항목 | 상태 | 비고 |
|---|---|---|
| PaddleOCR-VL CPU 공식 지원 | 비공식 | paddleocr 패키지 `device="cpu"` 동작 확인됨 |
| GOT-OCR2 Ollama | 미지원 | FastAPI 래퍼로 해결 |
| model-cache PVC 추가 용량 | 기존 50Gi 공유 | GOT-OCR2(~1.5GB). Granite는 ollama-model-cache(10Gi 신규) |
| 32GB 노드 메모리 | 여유 | 3개 서버 합계 ~16GB |

---

## 참고 소스

- [PP-StructureV3 Usage](http://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/PP-StructureV3.html)
- [PaddleOCR-VL-1.5 HF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- [GOT-OCR2 transformers docs](https://huggingface.co/docs/transformers/en/model_doc/got_ocr2)
- [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- [ibm/granite3.3-vision:2b Ollama](https://ollama.com/ibm/granite3.3-vision:2b)
- [deepseek-ocr Ollama](https://ollama.com/library/deepseek-ocr)
- [ibm-granite/granite-vision-3.3-2b-GGUF](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-GGUF)
- [Ollama K8s 배포 가이드](https://collabnix.com/running-ollama-on-kubernetes/)
- [FastAPI GOT-OCR2 래퍼 예시](https://github.com/iammuhammadnoumankhan/FastAPI-GOT-OCR-2-Transformers)
