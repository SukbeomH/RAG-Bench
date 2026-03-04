---
title: "벤치마크 효율화 + API 전략 확장 + HTML 보고서 (5-Phase GSD Plan)"
tags:
  - execution
  - summary
  - pdf-sampling
  - qa-limiting
  - openai-embed
  - upstage-embed
  - graphrag-removal
  - html-report
type: execution-summary
created: 2026-02-20T00:00:00+09:00
contextual_description: "5단계 GSD 플랜 전체 완료: PDF 페이지 샘플링(10%, 상한 5p), QA 수 제한(페이지당 2개), OpenAI/Upstage 임베딩 전략 추가, GraphRAG 완전 제거(73→72 콤보), HTML 보고서 자동 생성."
keywords:
  - PDF 샘플링
  - QA 제한
  - OpenAIEmbedStrategy
  - UpstageEmbedStrategy
  - GraphRAG 제거
  - HTML 보고서
  - matplotlib base64
  - solar-embedding-1-large
  - text-embedding-3-small
related:
  - 2026-02-20_graphrag-removal
  - 2026-02-20_rag-bench-service-modularization-cookbook
---

## 벤치마크 효율화 + API 전략 확장 + HTML 보고서 (5-Phase GSD Plan)

### 실행 내용
5개 Phase로 구성된 GSD 플랜을 순서 (Phase 1 → 4 → 2 → 3 → 5)에 따라 전체 완료.

---

### Phase 1: PDF 페이지 샘플링 + QA 수 제한

#### 변경 파일
| 파일 | 변경 |
|------|------|
| `rag_bench/indexing/pdf_converter.py` | 샘플링 파라미터 추가 |
| `rag_bench/scripts/generate_qa.py` | QA 수 상한 로직 추가 |

#### `pdf_converter.py` 상세
- `pdf_to_markdown()`, `pdfs_to_markdowns()` 모두에 추가된 파라미터:
  - `sample_pages: bool = False`
  - `page_sample_ratio: float = 0.1` (10%)
  - `max_sample_pages: int = 5` (상한 5페이지)
- `fitz.open()`으로 총 페이지 수 파악
- `random.sample(range(total_pages), sample_count)` 로 무작위 샘플링
- `pymupdf4llm.to_markdown(str(_pdf_path), pages=sampled_pages)` 로 선택 페이지만 변환

#### `generate_qa.py` 상세
- 추가 인수: `--max_qa_per_page` (기본값 2), `--sample_pages` (store_true)
- 신규 함수 `_compute_effective_num_qa(args, parent_pairs)`:
  - 청크 메타데이터에서 소스별 페이지 수 추정
  - `effective_num_qa = min(args.num_qa, sampled_page_count * args.max_qa_per_page)`
- `_sample_parents()` 및 `_generate_qa_pairs()` 에 `effective_num_qa` 전달

---

### Phase 4: GraphRAG 완전 제거

→ 별도 메모리: `2026-02-20_graphrag-removal.md` 참조

**요약**: 5개 파일 + 노트북에서 모든 GraphRAG 관련 코드 제거. 총 콤보 수 73 → 72개.

---

### Phase 2: OpenAI Embedding 전략

#### 신규 파일
**`rag_bench/strategies/openai_embed.py`**
```python
class OpenAIEmbedStrategy(BaseRAGStrategy):
    def __init__(self, model="text-embedding-3-small", qdrant_path=None, collection_name="openai_embed", k=3)
    # name: f"OpenAI({self.model})"
    # index(): langchain_openai.OpenAIEmbeddings + QdrantVectorStore.from_documents()
    # retrieve(): vectorstore.similarity_search(query, k=k)
    # cleanup(): vectorstore=None, qdrant_path 디렉토리 삭제
```
- **환경변수**: `OPENAI_API_KEY`
- **의존성**: `langchain-openai`, `langchain-qdrant` (기존 설치)

---

### Phase 3: Upstage Solar Embedding 전략

#### 신규 파일
**`rag_bench/strategies/upstage_embed.py`**
```python
class UpstageEmbedStrategy(BaseRAGStrategy):
    def __init__(self, model="solar-embedding-1-large-passage", query_model="solar-embedding-1-large-query", ...)
    # name: f"Upstage({self.model.split('-')[-1]})"  → "Upstage(large)"
    # index(): UpstageEmbeddings(model=passage_model)
    # retrieve(): query_embeddings.embed_query() → similarity_search_by_vector()

class _UpstageRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strategy: UpstageEmbedStrategy
    k: int = 3
```
- **핵심**: 문서 임베딩과 쿼리 임베딩에 별도 모델 사용 (passage vs query)
- **환경변수**: `UPSTAGE_API_KEY`
- **의존성**: `langchain-upstage>=0.3` (pyproject.toml에 추가)

#### 노트북 변경
- Cell 1.5 추가: Upstage API 키 Colab Secrets에서 로드
  ```python
  from google.colab import userdata
  os.environ["UPSTAGE_API_KEY"] = userdata.get("UPSTAGE_API_KEY", "")
  ```

---

### Phase 5: HTML 벤치마크 보고서

#### 신규 파일
**`rag_bench/scripts/generate_html_report.py`**
- `generate_html_report(latency_df, ragas_df, output_path, session_id, run_record)` 메인 함수
- 차트 함수: `_build_latency_chart()`, `_build_ragas_heatmap()`, `_build_scatter_chart()`, `_build_radar_chart()`
- 모든 차트: `matplotlib` → `io.BytesIO()` → `base64` → HTML 인라인 PNG
- Bootstrap 5 CDN, 순수 f-string 템플릿 (외부 의존성 없음)
- 보고서 섹션: 헤더, 요약 카드, 실행 환경, 레이턴시 순위, RAGAS 히트맵, 레이더, 산점도, 전략 설명, Top-3 추천
- 생성 파일 크기: ~193KB

#### `colab_runner.py` 변경
- `export_results()` 내부에 HTML 보고서 자동 생성 코드 추가:
  ```python
  from rag_bench.scripts.generate_html_report import generate_html_report
  html_path = output_dir / "report.html"
  generate_html_report(latency_df, ragas_df, str(html_path), session_id=self.session_id, run_record=run_record)
  ```

#### 노트북 변경
- 마지막 셀에 HTML 보고서 링크 추가:
  ```python
  from IPython.display import HTML, display
  display(HTML(f'<a href="{html_path}" target="_blank">📊 HTML 보고서 열기</a>'))
  ```

---

### pyproject.toml 변경
- 제거: `"lightrag-hku>=1.0"`, `"nest-asyncio>=1.6"`
- 추가: `"langchain-upstage>=0.3"`

---

### strategies/__init__.py 최종 상태
```python
from rag_bench.strategies.dense_sparse import DenseSparseStrategy
from rag_bench.strategies.colbert import ColBERTStrategy
from rag_bench.strategies.colbert_rerank import ColBERTRerankStrategy
from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy
from rag_bench.strategies.flashrank_rerank import FlashRankRerankStrategy
from rag_bench.strategies.openai_embed import OpenAIEmbedStrategy
from rag_bench.strategies.upstage_embed import UpstageEmbedStrategy
# GraphRAGStrategy 제거됨
```

---

### 해결된 주요 이슈
1. **시스템 Python3 미설치 패키지**: `.venv/bin/python3` 사용으로 검증
2. **NotebookEdit 셀 ID 없음**: Python `json` 조작으로 직접 편집
3. **pyproject.toml 중복 항목**: 두 번째 Edit로 수정

---

### 현재 전략 구현 상태
| 전략 | 상태 |
|------|------|
| DenseSparseStrategy | 완료 (3 dense × 6 retrieval = 18가지 조합) |
| ColBERTStrategy | 완료 |
| ColBERTRerankStrategy | 완료 |
| ContextualRetrievalStrategy | 완료 |
| FlashRankRerankStrategy | 완료 |
| OpenAIEmbedStrategy | 완료 (text-embedding-3-small) |
| UpstageEmbedStrategy | 완료 (solar-embedding-1-large) |
| GraphRAGStrategy | 제거됨 |
| **총 콤보** | **72개** |
