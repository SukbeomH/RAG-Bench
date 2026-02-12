# RAG 데이터셋 제작 방법론 비교: Ragas vs AutoRAG (Marker-Inc-Korea)

본 문서는 RAG 시스템 평가 및 최적화를 위한 데이터셋 제작 기능을 제공하는 두 주요 프레임워크, **Ragas**와 **AutoRAG (Marker-Inc-Korea)**의 방법론을 분석하고 비교합니다.

> **현황 참고**: rag_bench 프로젝트는 초기에 AutoRAG(Marker-Inc-Korea)를 활용했으나, langchain-core 버전 충돌 등 호환성 문제로 제거하고 자체 `generate_qa.py` 스크립트로 전환했습니다. 아래 분석은 방법론 비교를 위한 리서치 기록입니다.

## 1. Ragas: 진화적 합성 데이터 생성 (Evolutionary Generation)

Ragas는 단순한 질문 생성을 넘어, **복잡하고 다양한 유형의 질문**을 생성하여 RAG 시스템을 극한까지 테스트하는 데 초점을 맞춥니다.

### 1.1 핵심 철학
*   **복잡성 지향:** LLM이 생성하는 질문이 단순해지는 경향을 극복하기 위해, 질문을 "진화(Evol)"시켜 난이도를 높입니다.
*   **다양성 확보:** 추론(Reasoning), 조건부(Conditioning), 멀티 컨텍스트(Multi-Context) 등 다양한 유형의 질문을 생성합니다.

### 1.2 생성 프로세스 (TestsetGenerator)
1.  **Document Ingestion:** 문서를 로드하고 청킹합니다.
2.  **Knowledge Graph Creation:** 문서 내 엔티티와 관계를 파악하여 지식 그래프를 구성, 질문 생성의 기초로 삼습니다.
3.  **Question Evolution:**
    *   **Seed Question:** 기본적인 질문을 생성합니다.
    *   **Evolution:** Seed 질문을 더 복잡하게 변형합니다 (예: "A는 무엇인가?" -> "A와 B의 차이점은 무엇이며, C에 미치는 영향은?").
4.  **Synthesizer:**
    *   `SingleHop`: 단일 문서 기반 질문.
    *   `MultiHop`: 여러 문서에 걸친 정보를 종합해야 하는 질문.
5.  **Output:** `user_input` (질문), `reference_contexts` (근거 문서), `reference` (정답)이 포함된 데이터셋 생성.

### 1.3 장점 및 활용
*   **강건성 테스트:** 어려운 질문에 대한 RAG의 대처 능력을 평가하기 좋음.
*   **심층 평가:** 단순 검색 실패뿐만 아니라 추론 실패 등을 분석 가능.

---

## 2. AutoRAG: 최적화 파이프라인 통합 생성 (Optimization-Centric)

AutoRAG는 **RAG 파이프라인의 최적화(Optimization)**를 위해, 원본 문서(Raw Data)부터 최종 QA 데이터셋까지 이어지는 **End-to-End 데이터 파이프라인**을 제공합니다.

### 2.1 핵심 철학
*   **실용적 최적화:** 다양한 청킹 전략과 검색기(Retriever)를 비교 평가하기 위해, **Corpus(검색 대상)**와 **QA(평가 문제)**를 명확히 분리하고 매핑합니다.
*   **자동화:** PDF 등 Raw 파일에서부터 파싱, 청킹, QA 생성을 자동화된 흐름으로 처리합니다.

### 2.2 생성 프로세스
1.  **Parsing:** Raw 문서(PDF 등)를 텍스트로 변환합니다.
2.  **Chunking:**
    *   다양한 청킹 전략(Token, Recursive 등)을 적용하여 **Corpus Dataset**을 생성합니다.
    *   이 Corpus ID(`doc_id`)는 QA 데이터셋의 정답 근거로 사용됩니다.
3.  **QA Creation:**
    *   청크 단위로 LLM이 질문과 모범 답안을 생성합니다.
    *   생성된 질문이 해당 청크의 내용만으로 답변 가능한지 검증하는 필터링 과정을 거칩니다.
4.  **Output:**
    *   `corpus.parquet`: `doc_id`, `contents`, `metadata`.
    *   `qa.parquet`: `qid`, `retrieval_gt` (정답 `doc_id` 리스트), `query`, `generation_gt` (모범 답안).

### 2.3 장점 및 활용
*   **아키텍처 비교 용이:** 청킹 전략이 바뀌어도 Corpus ID 매핑을 통해 일관된 평가 가능.
*   **Retriever 최적화:** `retrieval_gt`가 명확하여 검색 정확도(Recall/NDCG) 측정에 최적화됨.

---

## 3. 비교 및 추천

| 특징 | Ragas | AutoRAG |
| :--- | :--- | :--- |
| **주요 목적** | **System Robustness** (시스템 강건성 평가) | **Pipeline Optimization** (최적 조합 탐색) |
| **질문 스타일** | 추론, 멀티홉 등 **복잡하고 논리적인 질문** | RAG가 검색해야 할 **정보 중심의 질문** |
| **데이터 구조** | Question - Context - Answer | Corpus(DB) - QA(Query+GT) **분리 구조** |
| **강점** | "우리 RAG가 얼마나 똑똑한가?" 평가 시 유리 | "어떤 청킹/검색기가 성능이 좋은가?" 실험 시 유리 |

### 결론: 벤치마크 적용 제안

본 `rag_bench` 프로젝트는 **RAG 전략 간의 성능 차이를 비교(Benchmarking)**하는 것이 목표입니다. 현재는 자체 `generate_qa.py` 스크립트로 Parent-Child 청킹 기반 QA 데이터셋을 생성하며, GPT-4o-mini를 사용한 LLM 기반 QA 자동 생성 + 해시 캐싱 방식을 채택하고 있습니다.

향후 질문의 **난이도와 다양성**을 확보하기 위해, **Ragas의 진화적 생성 프롬프트 전략**을 차용하여 현재 파이프라인에 통합하는 방안을 검토 중입니다.
