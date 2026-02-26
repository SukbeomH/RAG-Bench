"""
GOT-OCR2.0 OpenAI-compatible FastAPI 서버.

transformers >= 4.49 (v4 범위, 기존 벤치마크 환경 호환)
모델: stepfun-ai/GOT-OCR-2.0-hf (580M, FP32 CPU ~2.5GB)

/v1/chat/completions — base64 이미지 수신 → OCR 마크다운 반환
/v1/models           — 모델 목록
/health              — 헬스체크
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = os.environ.get("GOT_MODEL_ID", "stepfun-ai/GOT-OCR-2.0-hf")
HF_HOME  = os.environ.get("HF_HOME", "/models")
DEVICE   = "cpu"

_model: AutoModelForImageTextToText | None = None
_processor: AutoProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor
    print(f"[GOT-OCR2] 모델 로딩: {MODEL_ID}")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_HOME)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        cache_dir=HF_HOME,
    ).eval()
    print("[GOT-OCR2] 준비 완료")
    yield
    del _model, _processor


app = FastAPI(title="GOT-OCR2 API", lifespan=lifespan)


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
    model: str = "got-ocr2"
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.0


def _b64_to_pil(data_url: str) -> Image.Image:
    """data:image/...;base64,<data> → PIL Image."""
    b64 = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _extract_image_and_prompt(messages: list[Message]) -> tuple[Image.Image | None, str]:
    image = None
    prompt = "Convert this document page to clean, structured markdown."
    for msg in messages:
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            prompt = msg.content
        else:
            for part in msg.content:
                if part.type == "image_url" and part.image_url:
                    image = _b64_to_pil(part.image_url.url)
                elif part.type == "text" and part.text:
                    prompt = part.text
    return image, prompt


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _model is None or _processor is None:
        return JSONResponse(status_code=503, content={"error": "모델 로딩 중"})

    image, prompt = _extract_image_and_prompt(req.messages)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "이미지가 messages에 없습니다."})

    # GOT-OCR2 HF native API: 이미지 + 텍스트 프롬프트
    inputs = _processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
        )

    # input 토큰 제거 후 생성 텍스트만 디코드
    gen_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    result = _processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return {
        "id": f"chatcmpl-got-{int(time.time())}",
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
        "data": [{"id": "got-ocr2", "object": "model", "owned_by": "stepfun-ai"}],
    }


@app.get("/health")
async def health():
    loaded = _model is not None
    return {"status": "ok" if loaded else "loading", "model": MODEL_ID}
