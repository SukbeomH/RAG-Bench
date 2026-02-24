"""
문서 샘플링 전략 (sampler).

카테고리별 샘플링 비율에 따라 문서의 대표 텍스트를 추출한다.
전체 문서를 처리하는 대신 핵심 부분만 벤치마크에 활용하여 비용과 시간을 절감한다.

외부 인터페이스:
  sample_document(path, doc_type, *, max_chars) -> str
  sample_text(text, doc_type, *, max_chars) -> str
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from rag_bench.document_types.types import DocType, DOC_TYPE_METADATA


# ---------------------------------------------------------------------------
# 기본 파라미터
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CHARS = 50_000   # 샘플링 후 최대 문자 수 (LLM 입력 제한)
_DEFAULT_CHUNK_SIZE = 1_000   # 청크 샘플링 단위 (characters)
_SEED = 42                    # 재현 가능한 샘플링


# ---------------------------------------------------------------------------
# 텍스트 샘플링
# ---------------------------------------------------------------------------

def sample_text(
    text: str,
    doc_type: DocType,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    seed: int = _SEED,
) -> str:
    """카테고리 샘플링 비율에 따라 텍스트에서 대표 구간을 추출한다.

    전략:
    - ratio=1.0: 전체 텍스트 (최대 max_chars까지)
    - ratio<1.0: 앞 50% 고정 + 나머지 50% 랜덤 샘플링
      (앞부분은 주로 목차/개요로 중요도가 높음)

    Args:
        text: 샘플링 대상 텍스트.
        doc_type: 문서 종류 (샘플링 비율 결정).
        max_chars: 반환 최대 문자 수.
        seed: 랜덤 시드.

    Returns:
        샘플링된 텍스트.
    """
    meta = DOC_TYPE_METADATA[doc_type]
    ratio = meta["sampling_ratio"]

    if not text:
        return ""

    # ratio=1.0이거나 텍스트가 짧으면 전체 반환
    target_chars = int(len(text) * ratio)
    if ratio >= 1.0 or target_chars >= len(text):
        return text[:max_chars]

    # 앞 50%는 고정 포함 (개요/목차가 중요)
    fixed_chars = target_chars // 2
    fixed_part = text[:fixed_chars]

    # 나머지 50%는 랜덤 청크 샘플링
    remaining_text = text[fixed_chars:]
    random_target = target_chars - fixed_chars

    chunks = [
        remaining_text[i: i + _DEFAULT_CHUNK_SIZE]
        for i in range(0, len(remaining_text), _DEFAULT_CHUNK_SIZE)
    ]

    rng = random.Random(seed)
    n_chunks = max(1, random_target // _DEFAULT_CHUNK_SIZE)
    n_chunks = min(n_chunks, len(chunks))
    sampled_chunks = rng.sample(chunks, n_chunks)
    sampled_chunks.sort(key=lambda c: remaining_text.index(c))  # 원래 순서 보존

    sampled_part = "".join(sampled_chunks)

    result = fixed_part + "\n\n[...샘플링된 구간...]\n\n" + sampled_part
    return result[:max_chars]


def sample_document(
    path: "str | Path",
    doc_type: DocType,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    seed: int = _SEED,
    encoding: str = "utf-8",
) -> str:
    """파일을 읽고 카테고리 샘플링 전략을 적용하여 대표 텍스트를 반환한다.

    PDF/DOCX/HTML은 multi_parser.parse_document()로 먼저 파싱한 후 사용한다.
    이 함수는 텍스트 파일(.txt, .md)에 직접 사용 가능하다.

    Args:
        path: 텍스트 파일 경로.
        doc_type: 문서 종류.
        max_chars: 반환 최대 문자 수.
        seed: 랜덤 시드.
        encoding: 파일 인코딩.

    Returns:
        샘플링된 텍스트.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: 지원하지 않는 파일 포맷일 때.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".txt", ".md", ".text"):
        raise ValueError(
            f"sample_document()는 텍스트 파일(.txt, .md)만 직접 지원합니다. "
            f"다른 포맷은 multi_parser.parse_document()로 먼저 파싱하세요. "
            f"받은 확장자: '{suffix}'"
        )

    text = path.read_text(encoding=encoding, errors="ignore")
    return sample_text(text, doc_type, max_chars=max_chars, seed=seed)


def get_sampling_info(doc_type: DocType, text_length: int) -> dict:
    """샘플링 파라미터 정보 반환 (로깅/디버깅용)."""
    meta = DOC_TYPE_METADATA[doc_type]
    ratio = meta["sampling_ratio"]
    target = int(text_length * ratio)
    return {
        "doc_type": doc_type.value,
        "sampling_ratio": ratio,
        "original_chars": text_length,
        "target_chars": min(target, _DEFAULT_MAX_CHARS),
        "strategy": "전체" if ratio >= 1.0 else f"앞50%고정+랜덤50% (비율 {ratio:.0%})",
    }
