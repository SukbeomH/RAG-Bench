# RAGAS Testset Generation v2 리서치 — QA 데이터셋 고도화 전략

> **목적**: 현재 `generate_qa.py`의 단순 GPT-4o-mini QA 생성(20개)을 RAGAS의 Knowledge Graph 기반 진화적 생성으로 교체하여, 50~100개 QA를 난이도/유형 다양성을 확보하여 생성한다.

## 1. 현재 구현 분석 (`generate_qa.py`)

### 현행 방식
```
문서(*.md) → Parent-Child 청킹 → Parent 균등 샘플링 → GPT-4o-mini 1:1 QA 생성
```

### 한계점
| 항목 | 현재 | 문제 |
|------|------|------|
| 질문 유형 | 단일 유형 (사실 기반) | 추론/멀티홉/조건부 질문 부재 |
| 난이도 | 균일 (쉬움) | LLM이 생성하는 질문은 기본적으로 단순해지는 경향 |
| 컨텍스트 활용 | 1 Parent → 1 QA | 문서 간 관계를 활용하는 질문 불가 |
| 다양성 | 랜덤 샘플링 의존 | 키워드/헤드라인 기반 체계적 커버리지 부족 |
| 페르소나 | 없음 | 사용자 유형별 질문 스타일 반영 불가 |
| 생성량 | 20개 (수동 확장 어려움) | 통계적 유의성 부족 |

---

## 2. RAGAS Testset Generation 아키텍처 (v0.4+)

### 2.1 전체 파이프라인

```
Documents
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1: Knowledge Graph 구축          │
│                                         │
│  Extractors ──→ Nodes                   │
│    ├── HeadlinesExtractor (LLM)         │
│    ├── KeyphrasesExtractor (LLM)        │
│    └── NERExtractor (LLM)               │
│                                         │
│  Splitters ──→ Chunks                   │
│    └── HeadlineSplitter(max_tokens)     │
│                                         │
│  RelationshipBuilders ──→ Edges         │
│    └── JaccardSimilarityBuilder         │
│                                         │
│  → KnowledgeGraph (저장/재사용 가능)     │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Phase 2: Testset 생성                  │
│                                         │
│  Personas × QuerySynthesizers           │
│    ├── SingleHopSpecificQuerySynthesizer│
│    ├── MultiHopAbstractQuerySynthesizer │
│    └── MultiHopSpecificQuerySynthesizer │
│                                         │
│  Scenarios → Samples                    │
│    (user_input, reference_contexts,     │
│     reference, synthesizer_name)        │
└─────────────────────────────────────────┘
```

### 2.2 핵심 컴포넌트

#### Knowledge Graph
- 문서를 Node로 변환, Extractor로 속성 추출, RelationshipBuilder로 엣지 생성
- **저장/재사용 가능**: `kg.save("knowledge_graph.json")` / `KnowledgeGraph.load()`
- 한 번 구축하면 다양한 testset 반복 생성 가능

#### Query Types (쿼리 유형)

| 유형 | 설명 | 예시 |
|------|------|------|
| **SingleHop-Specific** | 단일 문서에서 사실 기반 검색 | "엔비디아 NVLink Fusion의 스케일업 대역폭은?" |
| **SingleHop-Abstract** | 단일 문서의 해석/요약 | "AI DC의 전력 자급 전략은 어떤 방향인가?" |
| **MultiHop-Specific** | 여러 문서의 사실 종합 | "NVLink와 AI DC 전력 관계는?" |
| **MultiHop-Abstract** | 여러 문서의 개념 통합 | "AI 인프라 발전이 산업에 미치는 영향은?" |

#### Query Distribution (기본값)

```python
from ragas.testset.synthesizers import default_query_distribution

# 기본 분포:
# SingleHopSpecificQuerySynthesizer: 0.5 (50%)
# MultiHopAbstractQuerySynthesizer:  0.25 (25%)
# MultiHopSpecificQuerySynthesizer:  0.25 (25%)
```

#### Persona (페르소나)

```python
from ragas.testset.persona import Persona, generate_personas_from_kg

# 수동 정의
persona = Persona(
    name="AI 인프라 엔지니어",
    role_description="AI 데이터센터 인프라를 설계하고 운영하는 엔지니어. "
                     "GPU, 네트워크, 전력 등 기술 세부사항에 관심."
)

# 자동 생성 (Knowledge Graph 기반)
personas = generate_personas_from_kg(kg=kg, llm=llm, num_personas=5)
```

### 2.3 v0.1.x → v0.4+ 변화

| 항목 | v0.1.x (Legacy) | v0.4+ (현재) |
|------|-----------------|-------------|
| 패러다임 | Evolution (simple→reasoning→multi_context) | Knowledge Graph + Synthesizer |
| 질문 분류 | `simple`, `reasoning`, `multi_context`, `conditional` | `SingleHop(Specific/Abstract)`, `MultiHop(Specific/Abstract)` |
| 배포 설정 | `{simple: 0.5, reasoning: 0.3, multi_context: 0.2}` | `[(Synthesizer, weight), ...]` |
| 비평가 LLM | 별도 `critic_llm` 필요 | 단일 LLM 사용 |
| 그래프 | 암묵적 | 명시적 KnowledgeGraph (저장/재사용) |
| 페르소나 | 없음 | `Persona` + 자동 생성 지원 |

---

## 3. 비영어(한국어) 지원

RAGAS는 `adapt_prompts()` 메서드로 비영어 언어를 지원한다:

```python
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

# 한국어 프롬프트 적응
for synthesizer, weight in query_distribution:
    prompts = await synthesizer.adapt_prompts("korean", llm=generator_llm)
    synthesizer.set_prompts(**prompts)
```

---

## 4. 구현 전략: `generate_qa.py` 개편안

### 4.1 제안 구조

```python
# rag_bench/scripts/generate_qa.py (개편)

from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.persona import Persona

# Phase 1: Knowledge Graph 구축
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
    ))

transforms = [
    HeadlinesExtractor(llm=generator_llm, max_num=20),
    HeadlineSplitter(max_tokens=1500),
    KeyphrasesExtractor(llm=generator_llm),
]
apply_transforms(kg, transforms)
kg.save(str(BENCH_DATA_DIR / "knowledge_graph.json"))

# Phase 2: 페르소나 정의
personas = [
    Persona(name="AI 인프라 엔지니어",
            role_description="AI DC 인프라 설계/운영. GPU, 네트워크, 전력 기술에 관심."),
    Persona(name="경영 전략가",
            role_description="AI 산업 동향과 시장 전략에 관심. 투자/비용 관점으로 질문."),
    Persona(name="정책 연구원",
            role_description="AI 규제, 데이터 주권, 에너지 정책 등 거시적 이슈에 관심."),
]

# Phase 3: Query Distribution 설정
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="headlines"), 0.35),
    (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="keyphrases"), 0.15),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]

# 한국어 적응
for synth, _ in query_distribution:
    prompts = await synth.adapt_prompts("korean", llm=llm)
    synth.set_prompts(**prompts)

# Phase 4: 생성
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)
testset = generator.generate(testset_size=100, query_distribution=query_distribution)
```

### 4.2 제안 질문 분포 (100개 기준)

| 유형 | 비율 | 개수 | 난이도 | 특징 |
|------|:----:|:----:|:------:|------|
| SingleHop-Specific (headlines) | 35% | 35 | 쉬움 | 섹션 제목 기반 사실 질문 |
| SingleHop-Specific (keyphrases) | 15% | 15 | 중간 | 키워드 기반 개념 질문 |
| MultiHop-Specific | 25% | 25 | 어려움 | 여러 문서 사실 종합 |
| MultiHop-Abstract | 25% | 25 | 매우 어려움 | 여러 문서 개념 통합 |

### 4.3 출력 포맷 변경

```json
{
  "docs_hash": "a815b521473ca3bc",
  "num_qa": 100,
  "generation_method": "ragas_v2",
  "query_distribution": {
    "single_hop_specific_headlines": 0.35,
    "single_hop_specific_keyphrases": 0.15,
    "multi_hop_specific": 0.25,
    "multi_hop_abstract": 0.25
  },
  "qa_pairs": [
    {
      "question": "...",
      "ground_truth": "...",
      "reference_contexts": ["ctx1", "ctx2"],
      "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
      "query_type": "single_hop_specific",
      "parent_id": null,
      "source": "ragas_kg"
    }
  ]
}
```

### 4.4 CLI 인터페이스

```bash
# 기본 (100개, RAGAS v2 방식)
python -m rag_bench.scripts.generate_qa --num_qa 100 --method ragas

# 레거시 호환 (기존 방식)
python -m rag_bench.scripts.generate_qa --num_qa 20 --method legacy

# Knowledge Graph만 구축 (재사용 가능)
python -m rag_bench.scripts.generate_qa --build-kg-only

# 기존 KG 재사용
python -m rag_bench.scripts.generate_qa --num_qa 100 --method ragas --reuse-kg
```

---

## 5. 비용 추정

### Knowledge Graph 구축 (1회)
| 작업 | 토큰 추정 | 비용 (gpt-4o-mini) |
|------|----------|-------------------|
| HeadlinesExtractor (33 parents) | ~50K input, ~5K output | ~$0.01 |
| KeyphrasesExtractor (33 parents) | ~50K input, ~5K output | ~$0.01 |
| 합계 | ~100K input, ~10K output | **~$0.02** |

### Testset 생성 (100개 QA)
| 작업 | 토큰 추정 | 비용 (gpt-4o-mini) |
|------|----------|-------------------|
| 시나리오 생성 (100개) | ~200K input, ~30K output | ~$0.05 |
| 한국어 적응 프롬프트 | ~10K input, ~5K output | ~$0.01 |
| 합계 | ~210K input, ~35K output | **~$0.06** |

**총 예상 비용: ~$0.08** (매우 저렴)

---

## 6. 의존성 변경

```toml
# pyproject.toml 추가
ragas = ">=0.4"  # 이미 설치되어 있을 가능성 높음
```

현재 `rag_bench/evaluation.py`에서 RAGAS를 이미 사용 중이므로 추가 의존성은 없을 가능성이 높다.

---

## 7. 마이그레이션 전략

### 단계적 전환
1. **Phase A**: `generate_qa.py`에 `--method ragas` 옵션 추가. 기존 `--method legacy`는 유지.
2. **Phase B**: 100개 QA 생성 후, 기존 20개 QA와 벤치마크 결과 비교.
3. **Phase C**: RAGAS 방식이 검증되면 기본값을 `ragas`로 변경.

### 호환성
- 기존 `qa_dataset.json` 포맷에 `synthesizer_name`, `query_type` 필드만 추가. 기존 필드(`question`, `ground_truth`)는 동일.
- `run_all_combos.py`는 `queries`와 `ground_truths` 리스트만 사용하므로 하위 호환성 유지됨.

---

## 8. 참고 자료

- [RAGAS Testset Generation for RAG (공식 문서)](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/)
- [RAGAS Query Types in RAG (개념)](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)
- [RAGAS Non-English Testset Generation](https://docs.ragas.io/en/stable/howtos/customizations/testgenerator/_language_adaptation/)
- [RAGAS Single-Hop Testset Application](https://docs.ragas.io/en/stable/howtos/applications/singlehop_testset_gen/)
- [RAGAS Persona Generation](https://docs.ragas.io/en/stable/howtos/customizations/testgenerator/_persona_generator/)
- [RAGAS v0.1.x Evolutionary Generation (Legacy)](https://docs.ragas.io/en/v0.1.21/concepts/testset_generation.html)
- [RAGAS Generate API Reference](https://docs.ragas.io/en/stable/references/generate/)
