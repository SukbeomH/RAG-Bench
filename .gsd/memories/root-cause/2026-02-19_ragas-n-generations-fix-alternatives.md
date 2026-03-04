---
title: "RAGAS n generations 경고 — 정밀 원인 분석 + 해결 대안 비교"
tags:
  - debug
  - root-cause
  - ragas
  - openai
  - langchain
  - fix-alternatives
type: root-cause
created: "2026-02-19T00:00:00+09:00"
contextual_description: "LangchainLLMWrapper가 deprecated되었고 n 파라미터 동적 할당이 ChatOpenAI Pydantic 모델에 실제로 반영되지 않아 경고 발생. 대안: llm_factory 전환(권장), strictness=1, ChatOpenAI(n=3) 명시"
keywords:
  - LangchainLLMWrapper
  - deprecated
  - llm_factory
  - bypass_n
  - strictness
  - ChatOpenAI
  - n parameter
  - generate_multiple
  - pydantic_prompt
related:
  - 2026-02-19_ragas-llm-1-instead-of-3-generations
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
---

## RAGAS n generations 경고 — 정밀 원인 분석 + 해결 대안 비교

---

## 정밀 원인 분석

### 1. `LangchainLLMWrapper`는 deprecated

**파일**: `.venv/.../ragas/llms/base.py`

```python
class LangchainLLMWrapper(BaseRagasLLM):
    """
    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LangchainLLMWrapper is deprecated and will be removed in a future version.
    #     Use llm_factory instead:
    #     from openai import OpenAI
    #     from ragas.llms import llm_factory
    #     client = OpenAI(api_key="...")
    #     llm = llm_factory("gpt-4o-mini", client=client)
    """
    def __init__(self, langchain_llm, ..., bypass_n: bool = False):
        import warnings
        warnings.warn("LangchainLLMWrapper is deprecated...", DeprecationWarning, ...)
```

현재 `evaluator.py`가 `LangchainLLMWrapper`를 사용하는 것 자체가 구식 방식.

---

### 2. `agenerate_text`의 n 처리 — Pydantic 동적 할당 문제

`LangchainLLMWrapper.agenerate_text` 내부:

```python
async def agenerate_text(self, prompt, n=1, ...):
    # n > 1이고 bypass_n=False이면: langchain_llm.n을 동적으로 설정 후 단일 프롬프트 전송
    if hasattr(self.langchain_llm, "n") and not self.bypass_n:
        self.langchain_llm.n = n          # ← ChatOpenAI.n = 3 동적 설정
        result = await self.langchain_llm.agenerate_prompt(
            prompts=[prompt],             # ← 단일 프롬프트
            ...
        )
    else:
        result = await self.langchain_llm.agenerate_prompt(
            prompts=[prompt] * n,         # ← n개 배열 방식
            ...
        )
        generations = [[g[0] for g in result.generations]]
        result.generations = generations  # ← n개처럼 보이도록 reshape
```

**문제**: `ChatOpenAI`는 Pydantic BaseModel이다.
- LangChain v0.3+ / Pydantic v2에서 `model.field = value` 동적 할당이 실제 API 요청의 payload 빌드 시점에 반영되지 않을 수 있다
- 결과적으로 OpenAI API는 `n=1` (초기화 기본값)로 호출됨 → 1개만 반환

**결과 처리** (`pydantic_prompt.py`):
```python
# BaseRagasLLM 경로
available_generations = len(resp.generations[0])  # = 1 (n=3 미반영)
actual_n = min(3, 1) = 1
# → WARNING: LLM returned 1 generations instead of requested 3
```

---

### 3. `is_multiple_completion_supported` — ChatOpenAI는 지원 목록에 있음

```python
MULTIPLE_COMPLETION_SUPPORTED = [
    ChatOpenAI,       # ← 포함됨
    AzureChatOpenAI,
    ChatVertexAI,
    ...
]

def is_multiple_completion_supported(llm) -> bool:
    return isinstance(llm, tuple(MULTIPLE_COMPLETION_SUPPORTED))
```

`ChatOpenAI`는 지원 목록에 있으므로 RAGAS는 `n` 설정을 시도하지만, 동적 할당이 실제로 반영되지 않아 실패.

---

## 해결 대안 비교

### 대안 A: `llm_factory` 전환 (RAGAS 공식 권장)

**원리**: RAGAS가 OpenAI SDK를 직접 제어 → `n` 파라미터를 API payload에 올바르게 주입

**코드 변경** (`rag_bench/evaluation/evaluator.py`):

```python
# 변경 전
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
import httpx

http_client = httpx.Client(verify=False)
async_client = httpx.AsyncClient(verify=False)

llm = ChatOpenAI(
    model=self.llm_model,
    http_client=http_client,
    http_async_client=async_client,
)
self._evaluator_llm = LangchainLLMWrapper(llm)
```

```python
# 변경 후
from openai import AsyncOpenAI
from ragas.llms import llm_factory
import httpx

http_client = httpx.AsyncClient(verify=False)

openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=http_client,
)
self._evaluator_llm = llm_factory(
    model=self.llm_model,
    client=openai_client,
)
```

| 항목 | 평가 |
|------|------|
| 경고 해결 | ✅ 완전 해결 |
| strictness=3 유지 | ✅ |
| deprecated 경고 해결 | ✅ LangchainLLMWrapper 제거 |
| 코드 변경 범위 | evaluator.py `_ensure_initialized` 1곳 |
| SSL verify=False 유지 | ✅ AsyncOpenAI http_client로 전달 |
| 비용 | n=3 API 호출 → 비용 증가 없음 (단일 요청에 n=3) |

---

### 대안 B: `strictness=1` 설정

**원리**: 생성 요청 수를 1로 줄여 경고 조건 자체를 제거

**코드 변경** (`rag_bench/evaluation/metrics.py` 또는 `evaluator.py`):

```python
from ragas.metrics import ResponseRelevancy, AnswerRelevancy

# create_metrics() 내부 또는 evaluator 초기화 시
for metric in instances:
    if hasattr(metric, "strictness"):
        metric.strictness = 1
```

| 항목 | 평가 |
|------|------|
| 경고 해결 | ✅ 완전 해결 |
| strictness=3 유지 | ❌ 1로 낮춤 |
| answer_relevancy 신뢰도 | ⚠️ 소폭 감소 (단일 비교점) |
| deprecated 경고 해결 | ❌ LangchainLLMWrapper 그대로 |
| 코드 변경 범위 | metrics.py 메트릭 생성 후 1줄 추가 |
| 비용 | ↓ API 호출 1/3으로 감소 |

> **RAGAS 공식 문서**: strictness 권장 범위는 3~5. 1로 낮추면 단일 질문 생성으로 평가 신뢰도 저하 가능.

---

### 대안 C: `ChatOpenAI(n=3)` 초기화 명시

**원리**: Pydantic 모델 초기화 시 n=3을 고정하여 동적 할당 문제 우회

```python
llm = ChatOpenAI(
    model=self.llm_model,
    n=3,                      # ← 초기화 시 고정
    http_client=http_client,
    http_async_client=async_client,
)
self._evaluator_llm = LangchainLLMWrapper(llm)
```

| 항목 | 평가 |
|------|------|
| 경고 해결 | ⚠️ 부분 해결 (동적 할당이 초기값을 덮어쓰면 무효) |
| strictness=3 유지 | ✅ |
| deprecated 경고 해결 | ❌ LangchainLLMWrapper 그대로 |
| 코드 변경 범위 | 1줄 (n=3 추가) |
| 신뢰도 | ⚠️ 근본 원인이 Pydantic 할당 문제라면 여전히 실패 가능 |

---

### 대안 D: `LangchainLLMWrapper(llm, bypass_n=True)`

**원리**: n 동적 할당 시도를 건너뛰고 프롬프트 배열 방식으로 전환

```python
self._evaluator_llm = LangchainLLMWrapper(llm, bypass_n=True)
```

내부 동작:
```python
# bypass_n=True → 프롬프트 배열 방식 사용
result = await langchain_llm.agenerate_prompt(
    prompts=[prompt] * n,     # 3개 별도 API 호출
    ...
)
generations = [[g[0] for g in result.generations]]
result.generations = generations  # [[gen1, gen2, gen3]] 형태로 reshape
```

| 항목 | 평가 |
|------|------|
| 경고 해결 | ✅ 완전 해결 |
| strictness=3 유지 | ✅ (3번 별도 호출) |
| deprecated 경고 해결 | ❌ LangchainLLMWrapper 그대로 |
| 코드 변경 범위 | 1줄 (bypass_n=True 추가) |
| 비용 | 동일 (3번 API 호출이지만 토큰 수 동일) |
| 응답 시간 | ⚠️ 단일 n=3 요청 대비 소폭 느림 (3번 직렬 호출) |

---

## 권장 적용 순서

```
대안 A (llm_factory) — 가장 권장
  → RAGAS 최신 방식, deprecated 해결, n 파라미터 완전 지원
  → evaluator.py _ensure_initialized 수정 필요

대안 D (bypass_n=True) — 빠른 임시 수정
  → 1줄 변경으로 경고 즉시 제거
  → deprecated 경고는 여전히 남음

대안 B (strictness=1) — 비용 절감 우선 시
  → 빠른 수정, 평가 신뢰도 소폭 감소

대안 C (ChatOpenAI(n=3)) — 비권장
  → Pydantic 동적 할당 문제가 근본 원인이면 해결 안 됨
```

---

## 적용 파일

| 파일 | 수정 내용 |
|------|---------|
| `rag_bench/evaluation/evaluator.py` | `_ensure_initialized`: llm_factory 전환 (대안 A) 또는 bypass_n=True (대안 D) |
| `rag_bench/evaluation/metrics.py` | `create_metrics`: strictness 설정 (대안 B 선택 시) |
