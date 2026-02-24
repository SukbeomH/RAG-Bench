"""
datasets — HuggingFace 표준 데이터셋 로더.

Phase 2에서 hf_loader.py를 이 패키지에 추가한다.
6개 HuggingFace 데이터셋을 BeIR 포맷(corpus, queries, qrels)으로 로드하는
통합 인터페이스를 제공한다.

데이터셋 카테고리:
  GENERAL  : miracl/miracl (ko), Ko-StrategyQA, facebook/belebele, mteb/mrtidy
  LEGAL    : yjoonjang/markers_bm (law 서브셋)
  BUSINESS : yjoonjang/markers_bm (finance+public+commerce)
  MEDICAL  : xhluca/publichealth-qa (korean)
"""

from rag_bench.datasets.hf_loader import (
    HFDatasetLoader,
    BeirDataset,
    beir_to_parent_child_chunks,
)

__all__ = [
    "HFDatasetLoader",
    "BeirDataset",
    "beir_to_parent_child_chunks",
]
