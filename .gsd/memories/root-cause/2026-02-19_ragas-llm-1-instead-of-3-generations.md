---
title: "RAGAS 'LLM returned 1 generations instead of requested 3' 경고 원인 분석"
tags:
  - debug
  - root-cause
  - ragas
  - openai
  - langchain
  - pass2
type: root-cause
created: "2026-02-19T00:00:00+09:00"
contextual_description: "RAGAS ResponseRelevancy가 strictness=3으로 LangChainLLMWrapper에 n=3을 요청하지만, ChatOpenAI가 n 파라미터를 OpenAI API에 전달하지 않아 1개만 반환됨"
keywords:
  - ragas
  - pydantic_prompt
  - generate_multiple
  - strictness
  - LangchainLLMWrapper
  - ChatOpenAI
  - n parameter
  - answer_relevance
  - ResponseRelevancy
related:
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
---

## RAGAS 'LLM returned 1 generations instead of requested 3' 경고 원인 분석

### 증상

Pass 2 RAGAS 평가 실행 중 아래 경고가 출력됨:
```
WARNING:ragas.prompt.pydantic_prompt:LLM returned 1 generations instead of requested 3.
Proceeding with 1 generations.
```
평가는 중단되지 않고 진행되지만 `answer_relevance` 점수의 신뢰도가 저하됨.

---

## 원인 분석 (3-Layer)

### Layer 1: RAGAS `ResponseRelevancy.strictness = 3`

**파일**: `.venv/lib/python3.12/site-packages/ragas/metrics/_answer_relevance.py`

```python
# 기본값
strictness: int = 3

# 평가 시 n=strictness(=3)으로 generate_multiple 호출
responses = await self.question_generation.generate_multiple(
    data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
)
```

`ResponseRelevancy`는 답변 관련성을 측정하기 위해 LLM에게 동일한 답변으로부터 **3개의 서로 다른 질문**을 생성하도록 요청한다 (다양성 확보 목적).

---

### Layer 2: RAGAS의 LangChain LLM 처리 방식

**파일**: `.venv/lib/python3.12/site-packages/ragas/prompt/pydantic_prompt.py`

```python
if is_langchain_llm(llm):
    # "LangChain doesn't support n parameter directly, so we batch multiple prompts"
    langchain_llm = t.cast(BaseLanguageModel, llm)
    prompts = [prompt_value for _ in range(n)]   # 동일한 프롬프트를 n개 배열로
    resp = await langchain_llm.agenerate_prompt(
        prompts,          # [prompt, prompt, prompt]
        stop=stop,
        callbacks=prompt_cb,
    )
```

RAGAS는 LangChain LLM에 `n` 파라미터를 직접 전달하는 대신,
**동일한 프롬프트를 n개 배열로 만들어 배치 처리**하는 우회 방식을 사용한다.

---

### Layer 3: ChatOpenAI의 `n` 파라미터 기본값

**파일**: `.venv/lib/python3.12/site-packages/langchain_openai/chat_models/base.py`

```python
n: int | None = None
"""Number of chat completions to generate for each prompt."""
```

`ChatOpenAI`는 `n=None` (=1) 이 기본값이다.
RAGAS가 배치로 `[prompt, prompt, prompt]`를 보내도,
LangChain은 내부적으로 각 프롬프트를 **독립된 API 호출로 처리**한다.

---

## 근본 원인: 캐시 dedup 충돌

**핵심 메커니즘**:

```
RAGAS 요청:   [prompt, prompt, prompt]  (동일한 프롬프트 3개 배열)
                       ↓
LangChain agenerate_prompt() 배치 처리
                       ↓
LangChain 캐시 레이어: 동일한 입력 → 캐시 히트 → 1개 결과만 반환
                       ↓
resp.generations = [[gen1]]   ← 3개가 아닌 1개
                       ↓
RAGAS 검증:
  actual_n = len(flattened_generations) = 1
  if actual_n < n(=3):
      WARNING: LLM returned 1 generations instead of requested 3
```

LangChain의 캐싱 레이어(`InMemoryCache` 또는 Redis)가 활성화되어 있거나,
동일한 입력에 대해 API 수준 dedup이 발생하면 프롬프트 3개 중 1개만 실제 호출되고 나머지는 캐시에서 반환된다.
캐시에서 반환된 결과는 RAGAS의 generations 카운트에서 누락될 수 있다.

**추가 원인**: OpenAI `gpt-4o-mini` 등 최신 모델은 Chat Completions API의 `n` 파라미터를 지원하지만, o1·o3 계열 reasoning 모델은 `n > 1`을 **지원하지 않는다**. 모델 설정이 reasoning 모델로 바뀌면 이 경고가 항상 발생한다.

---

## OpenAI API `n` 파라미터 스펙 요약

| 항목 | 내용 |
|------|------|
| 파라미터명 | `n` |
| 역할 | 각 입력 메시지에 대해 생성할 completion 개수 |
| 기본값 | `1` |
| gpt-4o / gpt-4o-mini | **지원** |
| o1 / o3 (reasoning) | **미지원** (`n=1`만 허용) |
| 비용 | 생성된 총 토큰 수 기준 과금 (prompt는 1회만 전송) |
| LangChain ChatOpenAI | `n: int | None = None` — 명시적으로 설정해야 함 |

---

## 해결 방법

### 방법 1: `strictness=1`로 낮추기 (권장, 비용 절감)

```python
from ragas.metrics import ResponseRelevancy

answer_relevancy = ResponseRelevancy()
answer_relevancy.strictness = 1  # 기본값 3 → 1
```

생성 다양성이 다소 낮아지나 경고 없이 동작하며 비용도 1/3.

### 방법 2: `ChatOpenAI(n=3)` 명시 설정

**파일**: `rag_bench/evaluation/evaluator.py`

```python
llm = ChatOpenAI(
    model=self.llm_model,
    http_client=http_client,
    http_async_client=async_client,
    n=3,  # RAGAS strictness 기본값과 맞춤
)
```

OpenAI API 레벨에서 단일 요청으로 3개 completions를 받아 RAGAS 기대값 충족.
단, 비용은 약 3배 (생성 토큰 3배).

### 방법 3: LangChain 캐시 비활성화

```python
from langchain.globals import set_llm_cache
set_llm_cache(None)  # 캐시 비활성화
```

캐시 dedup으로 인한 누락 방지. 근본 해결책은 아님.

---

## 적용 파일

| 파일 | 수정 위치 |
|------|---------|
| `rag_bench/evaluation/evaluator.py` | `ChatOpenAI` 초기화 (line ~143) |
| `rag_bench_colab/colab_runner.py` | Pass 2 RAGAS 평가 설정 |

---

## 핵심 교훈

- RAGAS `ResponseRelevancy.strictness`는 LLM에게 요청할 생성 횟수이며 기본값 3
- RAGAS는 LangChain LLM에 `n`을 직접 전달 못하고 동일 프롬프트 배열로 우회
- LangChain 캐싱 + OpenAI API dedup 조합에서 3개 중 1개만 반환될 수 있음
- o1/o3 reasoning 모델은 `n > 1` 자체를 미지원 — 모델 선택 시 주의
- 경고는 soft warning이므로 평가는 계속되지만 `answer_relevance` 점수 신뢰도 저하
