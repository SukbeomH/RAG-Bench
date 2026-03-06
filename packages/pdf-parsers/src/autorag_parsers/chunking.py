"""텍스트 청킹 + 출처 메타데이터 상속.

PDF 파서 결과(ConversionResult)를 검색에 적합한 크기의 청크로 분할하고,
각 청크에 출처 정보(문서ID, 페이지, 위치)를 자동 부여한다.

청킹 알고리즘:
  1. 페이지 텍스트를 separator(기본 ``\\n\\n``)로 단락 분할
  2. 단락을 순서대로 누적하다가 chunk_size 초과 시 청크 확정
  3. chunk_overlap > 0이면 이전 청크 끝부분을 다음 청크 시작에 복사

bbox 계산 (_compute_bbox_for_chunk):
  - 청크에 포함된 단어들을 PageResult.bbox_data에서 매칭
  - 매칭된 단어들의 좌표를 합쳐 (min_x0, min_y0, max_x1, max_y1) 반환
  - bbox_data가 없거나 매칭 단어가 없으면 None

사용 예::

    from autorag_parsers import get_parser, chunk_document, ChunkConfig

    parser = get_parser("upstage")
    result = parser.convert("document.pdf")

    # 기본 설정 (512자, 64자 오버랩)
    chunks = chunk_document(result)

    # 커스텀 설정
    chunks = chunk_document(result, config=ChunkConfig(
        chunk_size=1000, chunk_overlap=100, separator="\\n\\n"
    ))

    for chunk in chunks:
        print(chunk.chunk_id, chunk.page_number, len(chunk.chunk_text))
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.provenance import ChunkProvenance


@dataclass
class ChunkConfig:
    """텍스트 청킹 설정.

    Attributes:
        chunk_size: 청크 최대 문자 수 (기본 512).
        chunk_overlap: 이전 청크 끝부분에서 다음 청크로 복사할 문자 수 (기본 64).
            연속된 청크 간 문맥 유지를 위해 사용. 0이면 오버랩 없음.
        separator: 단락 분할 구분자 (기본 빈 줄 ``\\n\\n``).
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    separator: str = "\n\n"


def _compute_bbox_for_chunk(
    chunk_text: str, bbox_data: list[dict] | None
) -> tuple[float, float, float, float] | None:
    """청크 텍스트에 해당하는 바운딩 박스를 word-level bbox에서 계산.

    bbox_data 각 항목 형식: {"x0": float, "y0": float, "x1": float, "y1": float, "text": str}
    좌표는 페이지 기준 절대좌표(pt 단위). 매칭된 단어들의 외접 사각형을 반환.
    """
    if not bbox_data:
        return None

    # Find words that appear in chunk
    chunk_words = set(re.findall(r"\S+", chunk_text.lower()))
    matching = [
        b for b in bbox_data if b.get("text", "").lower().strip() in chunk_words
    ]

    if not matching:
        return None

    x0 = min(b["x0"] for b in matching)
    y0 = min(b["y0"] for b in matching)
    x1 = max(b["x1"] for b in matching)
    y1 = max(b["y1"] for b in matching)
    return (x0, y0, x1, y1)


def _doc_id_from_path(source_path: str) -> str:
    return hashlib.sha256(source_path.encode()).hexdigest()[:12]


def chunk_page(
    page: PageResult,
    source_path: str,
    doc_id: str | None = None,
    config: ChunkConfig | None = None,
) -> list[ChunkProvenance]:
    """단일 페이지를 청크로 분할하고 출처 메타데이터를 부여.

    Args:
        page: 파서가 생성한 페이지 결과 (markdown, bbox_data 포함).
        source_path: 원본 PDF 파일 경로 (출처 추적용).
        doc_id: 문서 고유 ID. None이면 source_path의 SHA-256 해시 앞 12자.
        config: 청킹 설정. None이면 기본값 (512자, 64자 오버랩).

    Returns:
        ChunkProvenance 목록. 각 청크에 chunk_id(``p{page}_c{idx}``),
        page_number, bbox(있으면), backend 정보가 포함됨.
    """
    cfg = config or ChunkConfig()
    did = doc_id or _doc_id_from_path(source_path)

    text = page.markdown
    if not text.strip():
        return []

    # Split by separator, then merge into chunks
    paragraphs = text.split(cfg.separator)
    chunks: list[ChunkProvenance] = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current) + len(para) + len(cfg.separator) > cfg.chunk_size and current:
            bbox = _compute_bbox_for_chunk(current, page.bbox_data)
            chunks.append(
                ChunkProvenance(
                    doc_id=did,
                    source_path=source_path,
                    page_number=page.page_num,
                    chunk_id=f"p{page.page_num}_c{chunk_idx}",
                    chunk_text=current.strip(),
                    bbox=bbox,
                    backend=page.backend,
                )
            )
            # Overlap: keep tail of current chunk
            if cfg.chunk_overlap > 0:
                current = current[-cfg.chunk_overlap :] + cfg.separator + para
            else:
                current = para
            chunk_idx += 1
        else:
            current = current + cfg.separator + para if current else para

    # Flush remaining
    if current.strip():
        bbox = _compute_bbox_for_chunk(current, page.bbox_data)
        chunks.append(
            ChunkProvenance(
                doc_id=did,
                source_path=source_path,
                page_number=page.page_num,
                chunk_id=f"p{page.page_num}_c{chunk_idx}",
                chunk_text=current.strip(),
                bbox=bbox,
                backend=page.backend,
            )
        )

    return chunks


def chunk_document(
    result: ConversionResult,
    config: ChunkConfig | None = None,
) -> list[ChunkProvenance]:
    """문서 전체를 청크로 분할. 모든 페이지를 순회하며 chunk_page() 호출.

    Args:
        result: PDF 파서의 변환 결과 (ConversionResult).
        config: 청킹 설정. None이면 기본값.

    Returns:
        전체 페이지의 ChunkProvenance 목록 (페이지 순서 유지).
    """
    doc_id = _doc_id_from_path(result.pdf_path)
    all_chunks: list[ChunkProvenance] = []

    for page in result.pages:
        all_chunks.extend(
            chunk_page(page, result.pdf_path, doc_id=doc_id, config=config)
        )

    return all_chunks
