"""
PaddleOCR-VL-1.5 OpenAI-compatible FastAPI 서버.

paddleocr 패키지의 PPStructureV3 파이프라인 사용:
  - transformers 버전 충돌 없음 (paddleocr 자체 처리)
  - PP-StructureV3: Layout + OCR + Table + Formula → Markdown
  - CPU 지원, device="cpu"

모델: PaddleOCR-VL-1.5 (0.9B, ~3-4GB RAM)

/v1/chat/completions — base64 이미지 수신 → Markdown 반환
/v1/models           — 모델 목록
/health              — 헬스체크 (로딩 중: 503, 준비 완료: 200)
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

_pipeline = None
_loading_error: str | None = None


async def _load_pipeline_async() -> None:
    """PPStructureV3를 스레드풀에서 비동기 로드. lifespan yield 후 background task로 실행."""
    global _pipeline, _loading_error

    def _do_load() -> None:
        global _pipeline
        print("[PaddleOCR-VL] PPStructureV3 초기화 (device=cpu)...")
        from paddleocr import PPStructureV3
        _pipeline = PPStructureV3(device="cpu")
        print("[PaddleOCR-VL] 준비 완료")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_load)
    except Exception as exc:
        _loading_error = str(exc)
        print(f"[PaddleOCR-VL] 로딩 오류: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 즉시 yield → uvicorn이 바로 HTTP 서비스 시작
    # 파이프라인은 background task로 비동기 로드 → readiness probe가 200 반환 시 트래픽 유입
    asyncio.create_task(_load_pipeline_async())
    print("[PaddleOCR-VL] 서버 시작 (파이프라인 백그라운드 로딩 중...)")
    yield
    global _pipeline
    del _pipeline


app = FastAPI(title="PaddleOCR-VL API", lifespan=lifespan)


# ── OpenAI-compatible 스키마 ──────────────────────────────────────────────────

class ImageURL(BaseModel):
    url: str
    detail: str = "auto"

class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]

class ChatRequest(BaseModel):
    model: str = "paddleocr-vl-1.5"
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.0


def _b64_to_pil(data_url: str) -> Image.Image:
    b64 = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _extract_image(messages: list[Message]) -> Image.Image | None:
    for msg in messages:
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, list):
            for part in content:
                if part.type == "image_url" and part.image_url:
                    return _b64_to_pil(part.image_url.url)
    return None


def _concat_markdown(pipeline, markdown_list: list) -> str:
    """버전 호환: concatenate_markdown_pages 여러 경로 시도."""
    # 1) 직접 호출 (최신 버전)
    try:
        return pipeline.concatenate_markdown_pages(markdown_list)
    except AttributeError:
        pass

    # 2) paddlex_pipeline 경유 (일부 버전)
    try:
        return pipeline.paddlex_pipeline.concatenate_markdown_pages(markdown_list)
    except (AttributeError, Exception):
        pass

    # 3) 수동 연결 (fallback)
    parts = []
    for item in markdown_list:
        if isinstance(item, dict):
            parts.append(item.get("markdown_texts", str(item)))
        elif isinstance(item, str):
            parts.append(item)
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _loading_error:
        return JSONResponse(status_code=500, content={"error": f"파이프라인 로딩 실패: {_loading_error}"})
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"error": "파이프라인 로딩 중, 잠시 후 재시도하세요."})

    image = _extract_image(req.messages)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "이미지가 messages에 없습니다."})

    # predict()는 CPU-집약 동기 작업 → 스레드 풀에서 실행하여 이벤트 루프 해방
    # (블로킹 시 liveness probe HTTP 요청에 응답 불가 → 재시작 루프)
    def _do_inference() -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            results = list(_pipeline.predict(input=tmp_path))
            markdown_list = [r.markdown for r in results if hasattr(r, "markdown") and r.markdown]

            if not markdown_list:
                # markdown 속성 없을 경우 fallback
                return "\n\n".join(str(r) for r in results if r)

            return _concat_markdown(_pipeline, markdown_list)
        finally:
            os.unlink(tmp_path)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _do_inference)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return {
        "id": f"chatcmpl-paddle-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "paddleocr-vl-1.5", "object": "model", "owned_by": "paddlepaddle"}],
    }


@app.get("/health")
async def health():
    if _loading_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "model": "PaddleOCR-VL-1.5", "error": _loading_error},
        )
    loaded = _pipeline is not None
    if loaded:
        return {"status": "ok", "model": "PaddleOCR-VL-1.5"}
    # 503: readiness probe가 이 Pod을 Service에서 제외
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "model": "PaddleOCR-VL-1.5"},
    )
