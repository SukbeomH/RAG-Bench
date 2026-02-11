# RAG 아키텍처 비교를 위한 통제된 벤치마크 방법론

본 문서는 RAG(Retrieval-Augmented Generation) 아키텍처 간의 성능을 공정하고 정밀하게 비교하기 위해, **외부 요인을 최소화하고 변수를 통제하는 실험 설계 가이드라인**을 제시합니다.

## 1. 핵심 원칙: "Ceteris Paribus" (다른 모든 조건은 동일하게)

RAG 아키텍처(예: Naive RAG vs Advanced RAG vs GraphRAG)를 비교할 때, 아키텍처의 고유한 특성 외의 모든 변수는 **완벽하게 동일하게 유지**해야 합니다.

### 1.1 통제해야 할 외부 변수 (Controlled Variables)

다음 요소들은 모든 아키텍처 실험군에서 **동일한 값**으로 고정되어야 합니다.

| 변수 | 통제 방안 | 비고 |
| :--- | :--- | :--- |
| **LLM (Generator)** | 동일 모델, 동일 버전 고정 (예: `gpt-4o-2024-05-13`) | Temperature=0.0 설정 필수. |
| **Embedding Model** | 동일 임베딩 모델 사용 (예: `text-embedding-3-small`) | 차원 수 등 파라미터 일치. |
| **Chunking Strategy** | 동일한 청킹 전략 및 사이즈 (예: 500 tokens, 50 overlap) | GraphRAG 등 구조적 차이로 불가피한 경우 제외. |
| **Test Dataset** | 동일한 질문-정답 쌍 (Golden Dataset) | 평가 시마다 데이터가 변하지 않도록 고정. |
| **Evaluation Metric** | 동일한 평가 도구 및 버전 (예: `ragas` v0.1.7) | LLM-as-a-Judge 사용 시 Judge 모델도 고정. |
| **Prompt Template** | 가능한 동일한 시스템 프롬프트 사용 | 아키텍처별 필수 프롬프트 차이는 허용하되 최소화. |

---

## 2. 평가 방법론: 컴포넌트 단위 분리 (Component-wise Evaluation)

RAG 전체(End-to-End) 성능만 비교하면 원인을 파악하기 어렵습니다. 검색과 생성을 분리하여 평가함으로써 아키텍처의 영향을 정확히 측정합니다.

### 2.1 검색(Retrieval) 성능 고립 평가
*   **목적:** 아키텍처가 "얼마나 관련성 높은 정보를 가져오는가"만 측정. LLM의 생성 능력 배제.
*   **평가 지표:**
    *   **Recall@K:** 정답 문서를 얼마나 포함했는가?
    *   **NDCG@K:** 정답 문서가 상위에 랭크되었는가?
    *   **Context Precision:** 불필요한 정보(Noise)가 얼마나 적은가?

### 2.2 생성(Generation) 성능 고립 평가
*   **목적:** 동일한 문맥(Context)이 주어졌을 때 아키텍처별로 정보 처리/답변 능력이 다른지 측정.
*   **통제:** 검색 결과를 아키텍처가 가져온 것이 아니라, **Golden Context(정답 문서)**를 강제로 주입하여 평가할 수 있음 (아키텍처의 *검색* 능력이 아닌 *활용* 능력 비교 시).
*   **순수 아키텍처 비교 시:** 검색된 Context를 그대로 사용하여 **Context Relevancy**와 **Faithfulness**를 본다.

---

## 3. 실험 설계 시나리오

아키텍처의 효용성을 증명하기 위한 비교 실험 설계 예시입니다.

### 시나리오 A: 검색 아키텍처 비교 (Vector vs Hybrid vs Graph)
*   **고정:** 청킹(Token-based), 임베딩(OpenAI), LLM(GPT-4o), 질문셋.
*   **변수 (Independent Variable):**
    1.  **Baseline:** Dense Vector Search (Cosine Similarity).
    2.  **Exp 1:** Hybrid Search (BM25 + Vector + Reciprocal Rank Fusion).
    3.  **Exp 2:** GraphRAG (Entity-Relationship Network traversal).
*   **측정:** Retrieval Metrics (Recall@10, NDCG@10) 집중 측정.

### 시나리오 B: 리랭킹(Reranking) 효과 검증
*   **고정:** 1차 검색기(BM25), 청킹, 임베딩.
*   **변수:**
    1.  **Baseline:** No Reranker (Top-K 그대로 사용).
    2.  **Exp 1:** ColBERT Reranker (`pylate`).
    3.  **Exp 2:** Cross-Encoder Reranker (`BGE-Reranker`).
*   **측정:** 1차 검색 결과 대비 최종 Top-5의 NDCG 변화량.

---

## 4. 데이터셋 구성 가이드 (Gold Standard)

외부 요인을 줄이려면 데이터셋 자체가 편향되지 않아야 합니다.

1.  **Ground Truth 보유:** 단순 질문만 있는 것이 아니라, **"참조해야 할 정답 문서 ID"**가 매핑되어 있어야 검색 성능을 정확히 측정 가능 (`retrieval_gt`).
2.  **다양한 난이도:** 단답형, 멀티홉(Multi-hop), 요약형 질문이 섞여 있어야 특정 아키텍처(예: GraphRAG)의 장단점이 드러남.
3.  **Synthetic Data 활용 주의:** LLM이 생성한 합성 데이터는 특정 모델의 편향을 가질 수 있으므로, 사람의 검수(Human-in-the-loop)를 거친 Golden Set을 권장.

## 5. 결론: "외부 요인 제로"를 위한 체크리스트

실험 시작 전 다음 항목을 체크하세요.

- [ ] **Temperature=0**로 설정되었는가?
- [ ] 평가에 사용하는 **LLM Judge 모델**이 실험군/대조군 모두 동일한가?
- [ ] **캐싱(Caching)**이 비활성화되었거나, 동일 조건(Cold/Warm start)인가?
- [ ] 네트워크 레이턴시 등 인프라 요인이 결과(속도 측정 시)에 영향을 주지 않는 환경인가?

이 가이드를 준수함으로써, 우리는 "운(Luck)"이나 "모델 빨(Model Capability)"이 아닌, **순수한 RAG 아키텍처의 구조적 우위**를 입증할 수 있습니다.
