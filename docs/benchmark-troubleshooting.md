# 벤치마크 실행 트러블슈팅 가이드

## 발견된 문제 및 원인 (2026-03-05)

### 1. API 키 미전달 (openai, upstage)

**증상**: NED ≈ 0.07, 모든 PDF에서 동일한 단어 수(360), 또는 401 Unauthorized

**원인**: `.env` 파일에서 `source .env`로 변수를 로드하면 현재 쉘에만 설정되고, `uv run`이 생성하는 subprocess에는 전달되지 않음.

```bash
# 잘못된 방법 — subprocess에 변수 미전달
source .env
uv run python -m autorag_pdf_eval.runner ...

# 올바른 방법 — set -a로 자동 export
set -a && source .env && set +a
uv run python -m autorag_pdf_eval.runner ...
```

**영향 받는 환경변수**: `OPENAI_API_KEY`, `UPSTAGE_API_KEY`, `HF_TOKEN`

**예방 조치**:
- 벤치마크 실행 스크립트에 `set -a && source .env && set +a` 패턴 사용
- runner.py에서 `.env` 자동 로드 기능 추가 고려 (`python-dotenv`)

---

### 2. mlx-vlm 모델 경로 불일치 (paddleocr-vl)

**증상**: `Failed to load model: 404 Client Error`

**원인**: `openai_compat.py`의 `MODEL_PROFILES`에서 모델명을 `paddleocr-vl-1.5`로 전송하지만, mlx-vlm 서버는 HuggingFace 전체 경로(`PaddlePaddle/PaddleOCR-VL-1.5`)를 요구.

```python
# openai_compat.py
MODEL_PROFILES = {
    "paddleocr-vl": ("paddleocr-vl-1.5", ...),  # 짧은 이름
}
# mlx-vlm 서버는 HF 전체 경로 필요: "PaddlePaddle/PaddleOCR-VL-1.5"
```

**해결**: `OPENSOURCE_VLM_MODEL` 환경변수로 오버라이드:
```bash
export OPENSOURCE_VLM_MODEL=PaddlePaddle/PaddleOCR-VL-1.5
```

**근본 해결**: `MODEL_PROFILES`의 기본 모델명을 HF 전체 경로로 변경하거나, K8s/로컬 자동 감지 로직 추가.

---

### 3. mlx-vlm 엔드포인트 경로 (paddleocr-vl)

**증상**: K8s 서버(`paddleocr-vl-server:8000`)에 연결 시도, 로컬에서 실행 불가

**원인**: `openai_compat.py`의 기본 엔드포인트가 K8s 서비스 주소로 하드코딩.

```python
MODEL_PROFILES = {
    "paddleocr-vl": (..., "http://paddleocr-vl-server:8000/v1"),
}
```

**해결**: `OPENSOURCE_VLM_ENDPOINT` 환경변수로 오버라이드:
```bash
export OPENSOURCE_VLM_ENDPOINT=http://localhost:8111
```

**주의**: mlx-vlm은 `/v1` 접두사를 사용하지 않음. OpenAI SDK의 `base_url`에 `http://localhost:8111`을 설정하면 `http://localhost:8111/chat/completions`로 요청.

---

### 4. mlx-vlm 서버 HF_TOKEN 미설정

**증상**: `Failed to load model: 401 Client Error`

**원인**: mlx-vlm 서버가 모델을 동적으로 로드할 때 HuggingFace 인증이 필요한 경우, 서버 프로세스에 `HF_TOKEN`이 설정되어 있어야 함.

**해결**: 서버 시작 시 `HF_TOKEN` export:
```bash
export HF_TOKEN=hf_...
uv run python -m mlx_vlm.server --port 8111
```

---

### 5. mlx-vlm 서버 장시간 가동 시 불안정

**증상**: `/health`는 정상이지만 모든 요청에서 `Internal Server Error`

**원인**: mlx-vlm 서버가 수일간 가동 후 메모리/상태 문제로 불안정해짐.

**해결**: 서버 재시작 (`kill` + 재실행)

---

## 벤치마크 실행 체크리스트

1. **환경변수 export 확인**: `set -a && source .env && set +a`
2. **mlx-vlm 서버 상태 확인**: `curl http://localhost:8111/health`
3. **mlx-vlm 모델 로드 테스트**:
   ```bash
   curl -X POST http://localhost:8111/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"PaddlePaddle/PaddleOCR-VL-1.5","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
   ```
4. **paddleocr-vl 필수 환경변수**:
   ```bash
   export OPENSOURCE_VLM_ENDPOINT=http://localhost:8111
   export OPENSOURCE_VLM_MODEL=PaddlePaddle/PaddleOCR-VL-1.5
   export HF_TOKEN=hf_...
   ```
5. **mlx-vlm 서버 시작 위치**: `servers/mlx_vlm/` 디렉토리에서 실행
6. **결과 검증**: NED > 0.5, 단어 수 > 500 (text_only.pdf 기준)
