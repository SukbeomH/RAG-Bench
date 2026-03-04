"""
datasets — HuggingFace 표준 데이터셋 로더.

6개 HuggingFace 데이터셋을 BeIR 포맷(corpus, queries, qrels)으로 로드하는
통합 인터페이스를 제공한다.
"""

from autorag_retrieval.datasets.hf_loader import (
    HFDatasetLoader,
    BeirDataset,
    beir_to_parent_child_chunks,
)

__all__ = [
    "HFDatasetLoader",
    "BeirDataset",
    "beir_to_parent_child_chunks",
]
