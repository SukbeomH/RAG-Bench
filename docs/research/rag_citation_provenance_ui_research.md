# RAG Citation / Provenance Tracking UI - 오픈소스 리서치

> 작성일: 2026-03-04
> 목적: RAG 답변의 출처 문서 추적, 문서 내 정확한 위치(페이지/단락/청크) 표시, 프론트엔드 시각화 구현을 위한 오픈소스 생태계 조사

---

## 1. Executive Summary

RAG 답변에 대한 citation/provenance 추적은 (1) 인덱싱 시 공간 메타데이터 보존, (2) 검색/생성 시 citation anchor 전달, (3) UI에서 bounding box 기반 하이라이트 렌더링의 3단계로 구성된다. 현재 가장 완성도 높은 오픈소스 솔루션은 **Kotaemon**(PDF 뷰어 내 하이라이트)과 **RAGFlow**(grounded citation UI)이며, PDF 하이라이트 전용 컴포넌트로는 **react-pdf-highlighter-extended**와 **RAG Document Viewer**가 실전 활용에 적합하다.

**권장 아키텍처**: Tensorlake 패턴(citation anchor + spatial metadata) 백엔드 + react-pdf-highlighter-extended 또는 RAG Document Viewer 프론트엔드

---

## 2. 완성형 오픈소스 RAG 플랫폼 (Citation 내장)

### 2-1. Kotaemon (Cinnamon)

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/Cinnamon/kotaemon |
| Stars | 20k+ |
| 라이선스 | Apache 2.0 |
| Tech Stack | Python 3.10+, Gradio UI, ChromaDB/LanceDB/Milvus, PDF.js 뷰어 |
| Citation 방식 | 답변 내 citation 번호 + 관련도 점수 표시, 클릭 시 PDF 뷰어에서 해당 위치 하이라이트 |
| 문서 포맷 | PDF, HTML, MHTML, XLSX, DOCX |
| 특징 | 멀티유저, GraphRAG 통합, Hybrid retrieval(full-text + vector) + reranking |

**장점**: 즉시 사용 가능한 citation UI 완성체. PDF.js 기반 인브라우저 뷰어에서 citation 하이라이트 직접 확인 가능.
**단점**: Gradio 기반이라 커스텀 UI 자유도 제한. React/Next.js 프로젝트에 컴포넌트 단위 통합 불가.

**적합도**: 한국어 PDF 문서 RAG에 바로 적용 가능. 독립 서비스로 운영하거나 PoC용으로 최적.

### 2-2. RAGFlow (InfiniFlow)

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/infiniflow/ragflow |
| Stars | 40k+ |
| 라이선스 | Apache 2.0 |
| Tech Stack | Python/FastAPI 백엔드, React 프론트엔드, Elasticsearch/Infinity, MinIO, MySQL, Redis |
| Citation 방식 | `##i$$` 패턴의 citation flag를 답변에 삽입, 클릭 시 원문 chunk 미리보기 |
| 문서 포맷 | Word, PPT, Excel, TXT, 이미지, 스캔본, 웹페이지 |
| 특징 | deepdoc 모듈로 deep document understanding, template 기반 chunking |

**Citation 구현 상세**:
- 전처리 시 chunk에 공간 정보(page, bbox) 임베딩
- 검색된 chunk의 citation ID를 답변에 `##1$$`, `##2$$` 형태로 삽입
- UI에서 citation 클릭 시 원문 chunk 내용 + 출처 문서 표시

**장점**: React 프론트엔드로 커스터마이징 가능. 대규모 엔터프라이즈 기능 포함.
**단점**: 전체 플랫폼 배포 필요 (Docker Compose). citation 컴포넌트만 분리 추출 어려움.

### 2-3. Onyx (formerly Danswer)

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/onyx-dot-app/onyx |
| 라이선스 | MIT (Community Edition) |
| Tech Stack | Python 백엔드, React 프론트엔드 |
| Citation 방식 | 답변에 인용문(quotes) + 출처 참조(references) 포함 |
| 특징 | 40+ 데이터 소스 커넥터, Web Search, MCP 지원 |

**적합도**: 엔터프라이즈 검색 + RAG. PDF 특화 citation 하이라이트보다는 문서 수준 출처 표시에 강점.

### 2-4. Open WebUI

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/open-webui/open-webui |
| Citation 방식 | RAG 답변에 citation 번호 표시, 클릭 시 출처 문서명 + 관련 텍스트 표시 |
| 제한사항 | Knowledge collection 전체 조회 시 첫 번째 소스만 인용되는 버그 존재 |

**적합도**: Ollama 기반 로컬 LLM + RAG에 적합하나, 세밀한 페이지/bbox 수준 citation 미지원.

---

## 3. PDF 하이라이트 / 문서 뷰어 컴포넌트

### 3-1. RAG Document Viewer (Preprocess.co) -- 강력 추천

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/preprocess-co/rag-document-viewer |
| 라이선스 | GPL v3 (pdf2htmlEX 의존) |
| Tech Stack | Python (서버사이드 HTML 번들 생성), iframe 기반 프론트엔드 |
| 지원 포맷 | PDF, DOCX, PPTX, XLSX, ODS, ODT, ODP |

**Bounding Box API**:
```python
boxes = [
  [{"page": 1, "top": 0.02, "left": 0.1, "height": 0.1, "width": 0.5}],
  [{"page": 2, "top": 0.5, "left": 0.2, "height": 0.2, "width": 0.6}]
]
RAG_DV(file_path="doc.pdf", store_path="/viewers/doc", chunks=boxes)
```

**핵심 기능**:
- 좌표는 페이지 크기 대비 정규화값 (0.0~1.0)
- Chunk Navigator: 이전/다음 chunk 간 네비게이션
- Scrollbar Navigator: 스크롤바에 chunk 위치 표시
- URL 파라미터 제어: `goto_chunk=N`, `goto_page=N`, `chunks=[0,2,3]`
- 다중 bounding box per chunk 지원 (다단 레이아웃)

**장점**: RAG에 특화된 설계. bbox 좌표만 전달하면 자동 하이라이트 + 네비게이션. 높은 렌더링 충실도.
**단점**: GPL 라이선스(pdf2htmlEX). Python 서버사이드 렌더링 필요. React 컴포넌트가 아닌 iframe 방식.

### 3-2. react-pdf-highlighter-extended -- 강력 추천

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/DanielArnould/react-pdf-highlighter-extended |
| 라이선스 | MIT |
| Tech Stack | React, TypeScript, PDF.js |

**Highlight 데이터 구조**:
```typescript
interface Highlight {
  id: string;
  position: {
    boundingRect: BoundingRect;
    rects: Array<Rect>;
  };
  content?: {
    text?: string;
    image?: string;
  };
}
```

**핵심 기능**:
- `viewportToScaled()`: 뷰포트 좌표 -> 플랫폼 독립 좌표 변환 (서버 저장용)
- `screenshot()`: bounding rectangle 이미지 캡처
- TextHighlight + AreaHighlight 컴포넌트
- MonitoredHighlightContainer: hover popup 지원
- 좌표가 viewport 독립적 -> 서버 저장/복원 가능

**장점**: MIT 라이선스. React/Next.js 프로젝트에 직접 통합. 프로그래밍 방식으로 하이라이트 추가 가능.
**단점**: PDF 전용. 하이라이트 렌더링만 담당하며 RAG 파이프라인 연동은 직접 구현 필요.

### 3-3. react-pdf-highlighter (Original)

| 항목 | 내용 |
|---|---|
| GitHub | https://github.com/agentcooper/react-pdf-highlighter |
| npm 다운로드 | 12+ dependent packages |
| 특징 | 원조 프로젝트. 텍스트/영역 하이라이트. PDF.js 기반. |

### 3-4. react-pdf-highlighter-plus

| 항목 | 내용 |
|---|---|
| URL | https://react-pdf-highlighter-plus-demo.vercel.app/ |
| 특징 | 하이라이트, 노트, 도형, 이미지 삽입, PDF 내보내기 지원 |

### 3-5. React PDF Viewer (highlight plugin)

| 항목 | 내용 |
|---|---|
| URL | https://react-pdf-viewer.dev/plugins/highlight/ |
| 특징 | 플러그인 방식으로 기존 PDF 뷰어에 하이라이트 추가 |

---

## 4. Citation-Aware RAG 백엔드 패턴

### 4-1. Tensorlake 패턴 (가장 체계적) -- 강력 추천

출처: https://www.tensorlake.ai/blog/rag-citations

**3단계 아키텍처**:

#### Stage 1: 인덱싱 시 공간 메타데이터 보존

문서 파싱 시 각 요소에서 content, page_number, bounding_box 추출. Chunk 텍스트에 경량 citation anchor 삽입:

```
원본 텍스트: "CDSMOTE reduces class imbalance..."
anchor 삽입: "<c>2.1</c> CDSMOTE reduces class imbalance..."
```

anchor 형식: `<c>[page_num].[reading_order]</c>`

메타데이터에 공간 정보 별도 저장:
```json
{
  "citations": {
    "2.1": {"page": 23, "bbox": {"x1": 12, "y1": 15, "x2": 149, "y2": 328}},
    "2.2": {"page": 23, "bbox": {"x1": 12, "y1": 35, "x2": 360, "y2": 400}}
  }
}
```

스토리지 오버헤드: 약 10-15% 추가.

#### Stage 2: LLM 응답 생성 시 citation 전달

프롬프트에서 LLM에게 anchor를 문장에 포함하지 말되, citation 배열로 반환하도록 지시:

```
Answer using the context below.
Ignore the <c>...</c> tags in your writing,
but list them in a "citations" array.
```

LLM 응답:
```json
{
  "answer": "CDSMOTE reduces class imbalance by clustering...",
  "citations": ["2.1"]
}
```

#### Stage 3: UI 렌더링

citation ID -> spatial metadata 조회 -> 페이지 번호 + bounding box로 해석 -> PDF 뷰어에서 해당 영역 하이라이트 또는 딥링크 생성.

### 4-2. Tensorlake Structured Extraction with Citations

```python
StructuredExtractionOptions(
    schema_name="BankStatement",
    json_schema=BankStatement,
    provide_citations=True
)
```

출력 형식:
```json
{
  "amount": "50.00",
  "amount_citation": [{
    "page_number": 1,
    "x1": 515, "x2": 585,
    "y1": 447, "y2": 482
  }]
}
```

### 4-3. LangChain Citation 패턴

출처: https://python.langchain.com/docs/how_to/qa_citations/

**방식 1**: XML 소스 태그로 문서 포맷팅
```python
# 검색된 문서에 source_id 부여
formatted_docs = [
    f"<source id='{i}'>{doc.page_content}</source>"
    for i, doc in enumerate(retrieved_docs)
]
```

**방식 2**: Pydantic 구조화 출력
```python
class Citation(BaseModel):
    source_id: int
    quote: str

class CitedAnswer(BaseModel):
    answer: str
    citations: List[Citation]
```

### 4-4. Haystack (deepset) Citation 패턴

- `AnswerBuilder` 컴포넌트의 `reference_pattern` 파라미터로 LLM 답변에서 reference 추출
- 프롬프트에서 LLM에게 `[doc_N]` 형식으로 출처 표기하도록 지시
- 추출된 reference와 원본 Document 객체 매핑

### 4-5. txtai Citation 파이프라인

출처: https://github.com/neuml/rag

- RAG 파이프라인에 citation 생성 로직 내장
- 답변과 컨텍스트 비교로 가장 관련된 reference 자동 결정
- Python 기반, 가벼운 설치

---

## 5. 학술 연구

### 5-1. VISA: Visual Source Attribution (ACL 2025)

| 항목 | 내용 |
|---|---|
| 논문 | https://arxiv.org/abs/2412.14457 |
| 저자 | Xueguang Ma, Shengyao Zhuang 등 |
| 학회 | ACL 2025 |

**핵심 기여**: VLM(Vision-Language Model)을 활용하여 검색된 문서 스크린샷에서 답변 근거 영역을 bounding box로 직접 표시. 기존 문서 수준 참조의 한계를 극복.

**데이터셋**: Wiki-VISA (Wikipedia), Paper-VISA (PubLayNet/의학 도메인)

**시사점**: PDF 파싱 없이 문서 이미지에서 직접 visual attribution 가능. 한국어 문서에도 VLM 기반 접근 적용 가능성.

### 5-2. CiteFix (2025)

| 항목 | 내용 |
|---|---|
| 논문 | https://arxiv.org/html/2504.15629v2 |
| 주제 | 후처리 기반 citation 보정 |

RAG 생성 후 citation 정확도를 후처리로 개선하는 접근.

---

## 6. 시각화 / 디버깅 도구

### 6-1. RAGxplorer
- Streamlit 앱, 문서 chunk의 임베딩을 2D 산점도로 시각화
- 쿼리와 chunk 간 유사도 시각적 확인

### 6-2. Chunk Visualizer (Hugging Face)
- https://huggingface.co/spaces/m-ric/chunk_visualizer
- chunking 전략별 결과 실시간 비교

### 6-3. RAG Visualizer
- https://github.com/gzguevara/rag-visualizer
- RAG 파이프라인 전체 흐름 시각화

---

## 7. 아키텍처 권장안

### 7-1. 권장 조합 (한국어 PDF 문서 RAG)

```
[Document Ingestion]
  PDF -> PyMuPDF/Docling/MinerU (텍스트 + bbox 추출)
       -> Tensorlake 패턴: citation anchor 삽입 + spatial metadata 저장

[Vector Store]
  Chunk text (with <c> anchors) + metadata (page, bbox) -> ChromaDB/Milvus

[Retrieval + Generation]
  LangChain/Haystack + structured output (citations array)

[Frontend]
  Option A: react-pdf-highlighter-extended (React/Next.js 프로젝트)
  Option B: RAG Document Viewer (iframe 기반, 다양한 포맷)
  Option C: Kotaemon (독립 서비스, 즉시 사용)
```

### 7-2. 구현 우선순위

| 순위 | 컴포넌트 | 권장 도구 | 이유 |
|---|---|---|---|
| 1 | PoC / 빠른 검증 | **Kotaemon** | 설치 후 즉시 citation + PDF 하이라이트 사용 가능 |
| 2 | 백엔드 citation 파이프라인 | **Tensorlake 패턴** + LangChain | 가장 체계적인 spatial metadata 보존 방식 |
| 3 | PDF 하이라이트 프론트엔드 | **react-pdf-highlighter-extended** | MIT, React 네이티브, 프로그래밍 방식 하이라이트 |
| 4 | 문서 뷰어 (다양한 포맷) | **RAG Document Viewer** | bbox 기반 자동 하이라이트 + chunk 네비게이션 |
| 5 | 전체 플랫폼 | **RAGFlow** | React UI + citation 내장, 대규모 배포 |

### 7-3. 한국어 문서 특수 고려사항

1. **OCR/파싱**: 한국어 PDF의 bbox 추출 시 PyMuPDF의 `page.get_text("dict")` 또는 Docling의 레이아웃 분석 활용
2. **Chunking**: 한국어 문장 분리 시 kss(Korean Sentence Splitter) 사용하되 bbox 매핑 유지
3. **Citation anchor**: `<c>` 태그가 한국어 텍스트 임베딩에 미치는 영향 최소화 필요 (임베딩 시 strip)
4. **폰트 렌더링**: react-pdf-highlighter는 PDF.js 기반이므로 한국어 폰트 렌더링 정상 지원

---

## 8. 데이터 흐름 상세

```
PDF 문서
  |
  v
[1. Document Parser] (PyMuPDF / Docling / MinerU)
  |-- 텍스트 블록 추출 (content + page + bbox)
  |-- 표/이미지 추출 (별도 처리)
  v
[2. Chunker with Citation Anchors]
  |-- chunk 텍스트에 <c>page.order</c> anchor 삽입
  |-- chunk metadata에 citations map 저장:
  |     {"2.1": {"page": 2, "bbox": [x1,y1,x2,y2]}}
  v
[3. Vector Store] (ChromaDB / Milvus)
  |-- embedding: anchor 제거 후 임베딩
  |-- metadata: citations map 포함 저장
  v
[4. Retrieval]
  |-- 쿼리 -> top-k chunks 검색
  |-- chunks에 anchor 포함된 텍스트 + citations metadata 반환
  v
[5. LLM Generation]
  |-- 프롬프트: "anchor 태그 무시하되 citations 배열로 반환"
  |-- 응답: {"answer": "...", "citations": ["2.1", "3.4"]}
  v
[6. Citation Resolution]
  |-- citation ID -> metadata에서 page + bbox 조회
  |-- 프론트엔드에 전달: {answer, sources: [{page, bbox, text}]}
  v
[7. Frontend Rendering]
  |-- 답변 텍스트에 citation 번호 인라인 표시
  |-- 클릭 시 PDF 뷰어에서 해당 페이지로 이동 + bbox 하이라이트
```

---

## 9. 참고 리소스

| 리소스 | URL | 설명 |
|---|---|---|
| Tensorlake Citation Blog | https://www.tensorlake.ai/blog/rag-citations | Citation-aware RAG 구현 가이드 (가장 상세) |
| Tensorlake Provable RAG | https://dev.to/tensorlake/make-rag-provable-page-bbox-citations-for-all-extracted-data-4ipc | 필드별 citation + bbox |
| LangChain Citations | https://python.langchain.com/docs/how_to/qa_citations/ | LangChain citation 공식 가이드 |
| Kotaemon | https://github.com/Cinnamon/kotaemon | 즉시 사용 가능한 citation UI |
| RAGFlow | https://github.com/infiniflow/ragflow | React 기반 grounded citation |
| RAG Document Viewer | https://github.com/preprocess-co/rag-document-viewer | bbox 기반 PDF/Office 뷰어 |
| react-pdf-highlighter-extended | https://github.com/DanielArnould/react-pdf-highlighter-extended | React PDF 하이라이트 컴포넌트 |
| VISA Paper | https://arxiv.org/abs/2412.14457 | Visual Source Attribution (ACL 2025) |
| Vercel AI SDK RAG | https://github.com/vercel/ai-sdk-rag-starter | Next.js RAG 스타터 |
| Onyx (Danswer) | https://github.com/onyx-dot-app/onyx | 엔터프라이즈 RAG + citation |
| Open WebUI | https://docs.openwebui.com/tutorials/tips/rag-tutorial/ | 로컬 LLM RAG |
| Haystack Citations | https://github.com/deepset-ai/haystack/discussions/8286 | Haystack citation 논의 |
| txtai RAG | https://github.com/neuml/rag | 경량 citation 파이프라인 |
| CiteFix Paper | https://arxiv.org/html/2504.15629v2 | Citation 후처리 보정 |
