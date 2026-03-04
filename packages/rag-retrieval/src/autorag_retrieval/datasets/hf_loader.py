"""
HuggingFace 데이터셋 로더 (hf_loader).

6개 HuggingFace 표준 데이터셋을 BeIR 포맷(corpus, queries, qrels)으로 로드하고
벤치마크에 사용 가능한 Parent-Child 청크로 변환한다.

데이터셋 카테고리 매핑:
  GENERAL  → miracl/miracl (ko), taeminlee/Ko-StrategyQA,
              facebook/belebele, mteb/mrtidy
  LEGAL    → yjoonjang/markers_bm (law 서브셋)
  BUSINESS → yjoonjang/markers_bm (finance+public+commerce)
  MEDICAL  → xhluca/publichealth-qa (korean)
  TECHNICAL → sionic-ai/nanobeir-ko (NanoSCIDOCS — 과학/기술 문서 검색)

BeIR 포맷:
  corpus  : {doc_id: {"title": str, "text": str}}
  queries : {query_id: str}
  qrels   : {query_id: {doc_id: relevance_score}}
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from autorag_retrieval.document_types.types import DocType


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------


@dataclass
class BeirDataset:
    """BeIR 포맷 데이터셋."""

    corpus: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # doc_id → {"title","text"}
    queries: Dict[str, str] = field(default_factory=dict)  # query_id → query_text
    qrels: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # query_id → {doc_id: score}
    doc_type: DocType = DocType.GENERAL
    source_name: str = ""

    @property
    def n_docs(self) -> int:
        return len(self.corpus)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    def to_langchain_docs(self) -> List[Document]:
        """corpus를 LangChain Document 목록으로 변환."""
        docs = []
        for doc_id, entry in self.corpus.items():
            text = entry.get("text", "") or entry.get("title", "")
            title = entry.get("title", "")
            if title and text and not text.startswith(title):
                content = f"{title}\n\n{text}"
            else:
                content = text
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "source": self.source_name,
                        "doc_type": self.doc_type.value,
                    },
                )
            )
        return docs

    def get_qa_pairs(self) -> List[Dict[str, str]]:
        """(query, ground_truth) 쌍 목록 반환."""
        pairs = []
        for qid, query in self.queries.items():
            rel_docs = self.qrels.get(qid, {})
            # 관련 문서 중 점수가 가장 높은 것을 ground_truth로 사용
            if rel_docs:
                top_did = max(rel_docs, key=lambda d: rel_docs[d])
                gt_entry = self.corpus.get(top_did, {})
                gt = gt_entry.get("text", "") or gt_entry.get("title", "")
            else:
                gt = ""
            pairs.append({"question": query, "ground_truth": gt, "query_id": qid})
        return pairs


# ---------------------------------------------------------------------------
# 데이터셋별 로더 함수
# ---------------------------------------------------------------------------


def _sample_corpus(corpus: Dict, max_size: int, seed: int = 42) -> Dict:
    """코퍼스가 max_size를 초과하면 랜덤 샘플링한다."""
    if len(corpus) <= max_size:
        return corpus
    rng = random.Random(seed)
    sampled_keys = rng.sample(list(corpus.keys()), max_size)
    return {k: corpus[k] for k in sampled_keys}


def _make_id(text: str, prefix: str = "doc") -> str:
    """텍스트 해시 기반 고유 ID 생성."""
    h = hashlib.md5(
        text.encode("utf-8", errors="ignore"), usedforsecurity=False
    ).hexdigest()[:12]
    return f"{prefix}_{h}"


def _load_klue_mrc(max_corpus: int = 50_000, max_queries: int = 500) -> BeirDataset:
    """klue/klue (mrc) 로드 — GENERAL 카테고리용 한국어 MRC 데이터셋.

    miracl/miracl-corpus는 커스텀 스크립트 방식으로 더 이상 지원되지 않아
    klue/klue mrc 데이터셋으로 대체한다.
    corpus: context(뉴스 본문), queries: question, qrels: context ↔ question 매핑.
    """
    from datasets import load_dataset

    print("  [KLUE-MRC] 로드 중 (streaming)...")
    ds = load_dataset("klue/klue", "mrc", streaming=True)

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    count = 0
    for row in ds["train"]:
        if max_queries > 0 and count >= max_queries:
            break
        if row.get("is_impossible", False):
            continue
        context = row.get("context", "")
        question = row.get("question", "")
        if not context or not question:
            continue

        did = _make_id(context)
        qid = f"klue_{row.get('guid', _make_id(question, 'q'))}"

        if did not in corpus:
            corpus[did] = {"title": row.get("title", ""), "text": context}

        queries[qid] = question
        qrels[qid] = {did: 1}
        count += 1

    corpus = _sample_corpus(corpus, max_corpus)
    print(f"  [KLUE-MRC] 코퍼스: {len(corpus):,}개 | 쿼리: {len(queries):,}개")
    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.GENERAL,
        source_name="klue-mrc",
    )


def _load_ko_strategyqa(max_queries: int = 500) -> BeirDataset:
    """taeminlee/Ko-StrategyQA 로드."""
    from datasets import load_dataset

    print("  [Ko-StrategyQA] 로드 중...")
    ds = load_dataset("taeminlee/Ko-StrategyQA", split="train", trust_remote_code=True)
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    count = 0
    for row in ds:
        if count >= max_queries:
            break
        qid = _make_id(str(row.get("qid", row.get("question", count))), "q")
        q = row.get("question", "")
        if not q:
            continue
        queries[qid] = q

        # facts/decomposition을 context로 사용
        facts = row.get("facts", []) or []
        if isinstance(facts, str):
            facts = [facts]
        context_text = "\n".join(str(f) for f in facts[:5])
        if context_text:
            did = _make_id(context_text)
            corpus[did] = {"title": q, "text": context_text}
            qrels[qid] = {did: 1}
        count += 1

    print(f"  [Ko-StrategyQA] 쿼리: {len(queries):,}개")
    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.GENERAL,
        source_name="ko-strategyqa",
    )


def _load_belebele_ko() -> BeirDataset:
    """facebook/belebele (kor_Hang) 로드."""
    from datasets import load_dataset

    print("  [Belebele-ko] 로드 중...")
    ds = load_dataset(
        "facebook/belebele", "kor_Hang", split="test", trust_remote_code=True
    )
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for i, row in enumerate(ds):
        passage = row.get("flores_passage", "") or ""
        question = row.get("question", "") or ""
        if not passage or not question:
            continue
        did = _make_id(passage, "p")
        qid = f"belebele_q{i}"
        corpus[did] = {"title": "", "text": passage}
        queries[qid] = question
        qrels[qid] = {did: 1}

    print(f"  [Belebele-ko] 지문: {len(corpus):,}개, 쿼리: {len(queries):,}개")
    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.GENERAL,
        source_name="belebele-ko",
    )


def _load_mrtidy_ko(max_corpus: int = 50_000, max_queries: int = 200) -> BeirDataset:
    """mteb/mrtidy (korean) 로드."""
    from datasets import load_dataset

    print("  [MrTiDy-ko] 코퍼스 로드 중...")
    try:
        corpus_ds = load_dataset(
            "mteb/mrtidy", "korean-corpus", split="train", trust_remote_code=True
        )
        corpus: Dict[str, Dict[str, str]] = {}
        for row in corpus_ds:
            did = str(row.get("_id", _make_id(row.get("text", ""))))
            corpus[did] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
        corpus = _sample_corpus(corpus, max_corpus)
    except Exception as e:
        print(f"  [MrTiDy-ko] 코퍼스 로드 실패: {e}. 쿼리셋 방식으로 폴백...")
        corpus = {}

    print(f"  [MrTiDy-ko] 코퍼스: {len(corpus):,}개")

    print("  [MrTiDy-ko] 쿼리 로드 중...")
    try:
        queries_ds = load_dataset(
            "mteb/mrtidy", "korean", split="test", trust_remote_code=True
        )
        queries: Dict[str, str] = {}
        qrels: Dict[str, Dict[str, int]] = {}

        for i, row in enumerate(queries_ds):
            if i >= max_queries:
                break
            qid = str(row.get("query_id", f"mrtidy_q{i}"))
            q = row.get("query", "")
            if not q:
                continue
            queries[qid] = q

            corpus_ids = row.get("corpus_ids", []) or []
            qrels[qid] = {}
            for did in corpus_ids:
                did_str = str(did)
                if did_str in corpus:
                    qrels[qid][did_str] = 1

    except Exception as e:
        print(f"  [MrTiDy-ko] 쿼리 로드 실패: {e}")
        queries, qrels = {}, {}

    print(f"  [MrTiDy-ko] 쿼리: {len(queries):,}개")
    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.GENERAL,
        source_name="mrtidy-ko",
    )


def _load_markers_bm(subset: str, max_queries: int = 0) -> BeirDataset:
    """yjoonjang/markers_bm 로드 (law / finance / public / commerce).

    subset: "law" | "finance" | "public" | "commerce" | "finance+public+commerce"
    max_queries: 0이면 전체 로드, 양수이면 해당 수만큼 샘플링.

    데이터셋 구조 (streaming=True 필수 — CAS 다운로드 오류 우회):
      corpus  config → "corpus" split : _id, text, title  (_id 형식: "<category> - <file> - <page>")
      queries config → "queries" split: _id, text         (_id 형식: "<index>_<category>")
      default config → "test"   split : query-id, corpus-id, score  (qrels)
    """
    from datasets import load_dataset

    subsets = [s.strip() for s in subset.split("+")]
    doc_type = DocType.LEGAL if "law" in subsets else DocType.BUSINESS

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    # --- corpus: _id 형식 "<category> - <file> - <page>" 기준 필터링 ---
    print(f"  [markers_bm] corpus 로드 중 (subsets: {subsets})...")
    try:
        corpus_ds = load_dataset("yjoonjang/markers_bm", "corpus", streaming=True)
        for row in corpus_ds["corpus"]:
            did = str(row.get("_id", ""))
            if not did:
                continue
            if any(did.startswith(sub + " - ") for sub in subsets):
                corpus[did] = {
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
        print(f"  [markers_bm] corpus: {len(corpus):,}개")
    except Exception as e:
        print(f"  [markers_bm] corpus 로드 실패: {e}")

    # --- queries: _id 형식 "<index>_<category>" 기준 필터링 ---
    print("  [markers_bm] queries 로드 중...")
    try:
        queries_ds = load_dataset("yjoonjang/markers_bm", "queries", streaming=True)
        for row in queries_ds["queries"]:
            if max_queries > 0 and len(queries) >= max_queries:
                break
            qid = str(row.get("_id", ""))
            if not qid:
                continue
            if any(qid.endswith("_" + sub) for sub in subsets):
                queries[qid] = row.get("text", "")
        print(f"  [markers_bm] queries: {len(queries):,}개")
    except Exception as e:
        print(f"  [markers_bm] queries 로드 실패: {e}")

    # --- qrels: 로드된 query/corpus 집합 내 항목만 수집 ---
    print("  [markers_bm] qrels 로드 중...")
    try:
        qrels_ds = load_dataset("yjoonjang/markers_bm", "default", streaming=True)
        for row in qrels_ds["test"]:
            qid = str(row.get("query-id", ""))
            did = str(row.get("corpus-id", ""))
            if qid in queries and did in corpus:
                qrels.setdefault(qid, {})[did] = int(row.get("score", 1))
        print(f"  [markers_bm] qrels: {len(qrels):,}개 queries")
    except Exception as e:
        print(f"  [markers_bm] qrels 로드 실패: {e}")

    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=doc_type,
        source_name=f"markers_bm-{subset}",
    )


def _load_nanobeir_ko_scidocs(max_queries: int = 0) -> BeirDataset:
    """sionic-ai/nanobeir-ko (NanoSCIDOCS) 로드 — TECHNICAL 카테고리용.

    과학/기술 문서 검색 벤치마크. BeIR NanoSCIDOCS의 한국어 번역 버전.
    3개 config(corpus/queries/qrels) × NanoSCIDOCS split으로 구성.
    max_queries: 0이면 전체 로드(50개), 양수이면 해당 수만큼 샘플링.
    """
    from datasets import load_dataset

    print("  [NanoBEIR-ko/SCIDOCS] 로드 중 (streaming)...")
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    try:
        # --- corpus: config='corpus', split='NanoSCIDOCS' ---
        corpus_ds = load_dataset(
            "sionic-ai/nanobeir-ko",
            "corpus",
            split="NanoSCIDOCS",
            streaming=True,
        )
        for row in corpus_ds:
            did = str(row.get("_id", ""))
            if did:
                corpus[did] = {
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
        print(f"  [NanoBEIR-ko/SCIDOCS] corpus: {len(corpus):,}개")

        # --- queries: config='queries', split='NanoSCIDOCS' ---
        queries_ds = load_dataset(
            "sionic-ai/nanobeir-ko",
            "queries",
            split="NanoSCIDOCS",
            streaming=True,
        )
        count = 0
        for row in queries_ds:
            if max_queries > 0 and count >= max_queries:
                break
            qid = str(row.get("_id", ""))
            text = row.get("text", "")
            if qid and text:
                queries[qid] = text
                count += 1
        print(f"  [NanoBEIR-ko/SCIDOCS] queries: {len(queries):,}개")

        # --- qrels: config='qrels', split='NanoSCIDOCS' ---
        qrels_ds = load_dataset(
            "sionic-ai/nanobeir-ko",
            "qrels",
            split="NanoSCIDOCS",
            streaming=True,
        )
        for row in qrels_ds:
            qid = str(row.get("query-id", ""))
            did = str(row.get("corpus-id", ""))
            score = int(row.get("score", 1))
            if qid in queries and did in corpus:
                qrels.setdefault(qid, {})[did] = score
        print(f"  [NanoBEIR-ko/SCIDOCS] qrels: {len(qrels):,}개 queries")

    except Exception as e:
        print(f"  [NanoBEIR-ko/SCIDOCS] 로드 실패: {e}")

    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.TECHNICAL,
        source_name="nanobeir-ko-scidocs",
    )


def _load_publichealth_qa_ko(max_queries: int = 0) -> BeirDataset:
    """xhluca/publichealth-qa (korean) 로드.

    max_queries: 0이면 전체 로드, 양수이면 해당 수만큼 샘플링.
    """
    from datasets import load_dataset

    print("  [publichealth-qa/ko] 로드 중...")
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    try:
        ds = load_dataset("xhluca/publichealth-qa", "korean", trust_remote_code=True)
        splits = list(ds.keys())
        split = "train" if "train" in splits else splits[0]

        for i, row in enumerate(ds[split]):
            if max_queries > 0 and len(queries) >= max_queries:
                break
            question = row.get("question", "") or row.get("query", "")
            answer = row.get("answer", "") or row.get("passage", "")
            if not question:
                continue
            did = _make_id(answer or question, "ph")
            qid = f"ph_q{i}"
            corpus[did] = {"title": "", "text": answer}
            queries[qid] = question
            qrels[qid] = {did: 1}

    except Exception as e:
        print(f"  [publichealth-qa/ko] 로드 실패: {e}")

    print(f"  [publichealth-qa/ko] 쿼리: {len(queries):,}개")
    return BeirDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        doc_type=DocType.MEDICAL,
        source_name="publichealth-qa-ko",
    )


# ---------------------------------------------------------------------------
# 통합 로더 클래스
# ---------------------------------------------------------------------------


class HFDatasetLoader:
    """DocType → HuggingFace 데이터셋 통합 로더."""

    # 카테고리별 로더 함수 매핑
    _PRIMARY: Dict[DocType, str] = {
        DocType.GENERAL: "miracl",
        DocType.LEGAL: "markers_bm_law",
        DocType.BUSINESS: "markers_bm_biz",
        DocType.MEDICAL: "publichealth",
        DocType.TECHNICAL: "nanobeir_scidocs",
    }

    def __init__(
        self,
        max_corpus: int = 50_000,
        max_queries: int = 500,
        seed: int = 42,
    ):
        self.max_corpus = max_corpus
        self.max_queries = max_queries
        self.seed = seed

    def load(self, doc_type: DocType) -> BeirDataset:
        """지정 카테고리의 HuggingFace 데이터셋을 로드한다.

        Args:
            doc_type: 로드할 문서 종류.

        Returns:
            BeirDataset (corpus + queries + qrels).

        Raises:
            ValueError: HF 데이터셋이 없는 카테고리(TECHNICAL)이면 에러.
        """
        if doc_type == DocType.TECHNICAL:
            return _load_nanobeir_ko_scidocs(max_queries=self.max_queries)

        if doc_type == DocType.GENERAL:
            return _load_klue_mrc(
                max_corpus=self.max_corpus, max_queries=self.max_queries
            )
        elif doc_type == DocType.LEGAL:
            return _load_markers_bm("law", max_queries=self.max_queries)
        elif doc_type == DocType.BUSINESS:
            return _load_markers_bm(
                "finance+public+commerce", max_queries=self.max_queries
            )
        elif doc_type == DocType.MEDICAL:
            return _load_publichealth_qa_ko(max_queries=self.max_queries)
        else:
            raise ValueError(f"지원하지 않는 DocType: {doc_type}")

    def load_secondary(self, doc_type: DocType) -> List[BeirDataset]:
        """GENERAL 카테고리의 보조 데이터셋을 로드한다.

        Returns:
            보조 데이터셋 목록 (Ko-StrategyQA, Belebele, MrTiDy).
        """
        if doc_type != DocType.GENERAL:
            return []

        results = []
        loaders = [
            (
                "Ko-StrategyQA",
                lambda: _load_ko_strategyqa(max_queries=self.max_queries),
            ),
            ("Belebele-ko", lambda: _load_belebele_ko()),
            (
                "MrTiDy-ko",
                lambda: _load_mrtidy_ko(
                    max_corpus=self.max_corpus, max_queries=min(self.max_queries, 200)
                ),
            ),
        ]
        for name, fn in loaders:
            try:
                print(f"\n  보조 데이터셋 로드: {name}")
                ds = fn()
                results.append(ds)
            except Exception as e:
                print(f"  [경고] {name} 로드 실패: {e}")
        return results

    def save_cache(self, dataset: BeirDataset, cache_dir: Path) -> None:
        """데이터셋을 JSON으로 캐시한다 (재실행 시 재다운로드 방지)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{dataset.source_name}.json"
        data = {
            "doc_type": dataset.doc_type.value,
            "source_name": dataset.source_name,
            "corpus": dataset.corpus,
            "queries": dataset.queries,
            "qrels": dataset.qrels,
        }
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"  [캐시 저장] {cache_file} ({dataset.n_docs:,}docs / {dataset.n_queries:,}queries)"
        )

    @staticmethod
    def load_cache(cache_dir: Path, source_name: str) -> Optional[BeirDataset]:
        """캐시된 데이터셋을 로드한다."""
        cache_file = cache_dir / f"{source_name}.json"
        if not cache_file.exists():
            return None
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return BeirDataset(
            corpus=data["corpus"],
            queries=data["queries"],
            qrels=data["qrels"],
            doc_type=DocType(data["doc_type"]),
            source_name=data["source_name"],
        )


# ---------------------------------------------------------------------------
# BeIR → Parent-Child 청크 변환
# ---------------------------------------------------------------------------


def beir_to_parent_child_chunks(
    dataset: BeirDataset,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """BeIR corpus를 Parent-Child 청크로 변환한다.

    HuggingFace 데이터셋의 각 corpus 문서를 parent로 사용하고,
    RecursiveCharacterTextSplitter로 child 청크를 생성한다.

    Returns:
        (parent_pairs, child_chunks)
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_pairs: List[Tuple[str, Document]] = []
    child_chunks: List[Document] = []

    for doc_id, entry in dataset.corpus.items():
        text = entry.get("text", "")
        title = entry.get("title", "")
        if title and text:
            full_text = f"{title}\n\n{text}"
        elif title:
            full_text = title
        else:
            full_text = text

        if not full_text.strip():
            continue

        parent_doc = Document(
            page_content=full_text,
            metadata={
                "parent_id": doc_id,
                "source": dataset.source_name,
                "doc_type": dataset.doc_type.value,
            },
        )
        parent_pairs.append((doc_id, parent_doc))

        children = splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = doc_id
        child_chunks.extend(children)

    return parent_pairs, child_chunks
