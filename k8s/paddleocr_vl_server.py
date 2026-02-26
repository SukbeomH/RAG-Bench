"""
PaddleOCR-VL-1.5 OpenAI-compatible FastAPI 서버.

paddleocr 패키지의 PPStructureV3 파이프라인 사용:
  - transformers 버전 충돌 없음 (paddleocr 자체 처리)
  - PP-StructureV3: Layout + OCR + Table + Formula → Markdown
  - CPU 지원, device="cpu"

모델: PaddleOCR-VL-1.5 (0.9B, ~3-4GB RAM)

/v1/chat/completions — base64 이미지 수신 → Markdown 반환
/v1/models           — 모델 목록
/health              — 헬스체크
"""
from __future__ import annotations

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

# paddleocr는 import 시 모델 자동 로드하지 않음
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("[PaddleOCR-VL] PPStructureV3 초기화 (device=cpu)...")
    from paddleocr import PPStructureV3
    _pipeline = PPStructureV3(device="cpu")
    print("[PaddleOCR-VL] 준비 완료")
    yield
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


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"error": "파이프라인 초기화 중"})

    image = _extract_image(req.messages)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "이미지가 messages에 없습니다."})

    # PNG로 임시 저장 후 파이프라인 호출 (PPStructureV3은 파일 경로 또는 ndarray 수신)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        results = _pipeline.predict(input=tmp_path)
        markdown = _pipeline.concatenate_markdown_pages(
            [r.markdown for r in results if hasattr(r, "markdown") and r.markdown]
        )
        if not markdown:
            markdown = "\n\n".join(
                str(r) for r in results if r
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(tmp_path)

    return {
        "id": f"chatcmpl-paddle-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": markdown},
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
    loaded = _pipeline is not None
    return {"status": "ok" if loaded else "loading", "model": "PaddleOCR-VL-1.5"}
