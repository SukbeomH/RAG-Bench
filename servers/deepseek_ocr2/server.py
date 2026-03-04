"""
DeepSeek-OCR-2 OpenAI-compatible FastAPI 서버.

모델: deepseek-ai/DeepSeek-OCR-2
  - transformers AutoModel + trust_remote_code
  - flash_attention_2 필수 → GPU(CUDA) 필수
  - bfloat16 추론

입력: OpenAI /v1/chat/completions (base64 이미지)
출력: Markdown 텍스트

/v1/chat/completions — base64 이미지 수신 → Markdown 반환
/v1/models           — 모델 목록
/health              — 헬스체크 (로딩 중: 503, 준비 완료: 200)

환경변수:
  MODEL_NAME   모델 경로 (기본: deepseek-ai/DeepSeek-OCR-2)
  HF_HOME      HuggingFace 캐시 경로 (PVC 마운트 시 /models/hf 권장)
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
DOC_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

_model = None
_tokenizer = None
_loading_error: str | None = None


async def _load_model_async() -> None:
    """GPU 모델 로드를 스레드풀에서 비동기 실행. lifespan yield 후 background task."""
    global _model, _tokenizer, _loading_error

    def _do_load() -> None:
        global _model, _tokenizer
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"[DeepSeek-OCR-2] 모델 로딩: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        _model = _model.eval().cuda().to(torch.bfloat16)
        print("[DeepSeek-OCR-2] 준비 완료")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_load)
    except Exception as exc:
        _loading_error = str(exc)
        print(f"[DeepSeek-OCR-2] 로딩 오류: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_model_async())
    print("[DeepSeek-OCR-2] 서버 시작 (모델 백그라운드 로딩 중...)")
    yield
    global _model, _tokenizer
    del _model, _tokenizer


app = FastAPI(title="DeepSeek-OCR-2 API", lifespan=lifespan)


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
    model: str = "deepseek-ocr2"
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
    if _loading_error:
        return JSONResponse(
            status_code=500,
            content={"error": f"모델 로딩 실패: {_loading_error}"},
        )
    if _model is None or _tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "모델 로딩 중, 잠시 후 재시도하세요."},
        )

    image = _extract_image(req.messages)
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "이미지가 messages에 없습니다."},
        )

    def _do_inference() -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.png"
            out_path = Path(tmpdir) / "output"  # 확장자 없이 — 모델이 .md 추가
            image.save(str(img_path), format="PNG")

            res = _model.infer(
                _tokenizer,
                prompt=DOC_PROMPT,
                image_file=str(img_path),
                output_path=str(out_path),
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )

            # 반환값이 문자열이면 그대로 사용
            if isinstance(res, str) and res.strip():
                return res

            # save_results=True로 저장된 파일 읽기 (output.md 또는 output)
            for candidate in [out_path.with_suffix(".md"), out_path]:
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8")

            # fallback: res를 문자열로 변환
            return str(res) if res else "<!-- 변환 결과 없음 -->"

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _do_inference)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return {
        "id": f"chatcmpl-dsocr2-{int(time.time())}",
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
        "data": [{"id": "deepseek-ocr2", "object": "model", "owned_by": "deepseek-ai"}],
    }


@app.get("/health")
async def health():
    if _loading_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "model": MODEL_NAME, "error": _loading_error},
        )
    loaded = _model is not None
    if loaded:
        return {"status": "ok", "model": MODEL_NAME}
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "model": MODEL_NAME},
    )
