---
title: "세션 인수인계: torch.set_default_device 버그 수정 + 세션 마무리"
type: session-handoff
date: 2026-02-20
tags: [bug-fix, torch, mps, config, dropout, no_grad]
---

## 세션 요약

이전 세션(benchmark-ragas-fix-parallel)의 미커밋 항목 커밋/푸시 완료 후,
이전 벤치마크 실행 로그에서 발견된 `torch.set_default_device("cpu")` 충돌 버그를 수정.

---

## 완료된 작업

### 1. 이전 세션 미커밋 항목 커밋/푸시

| 커밋 | 내용 |
|------|------|
| `d934d59` | `KoreanBM25Encoder` vocab thread-safety + `docs/memory_oom_analysis.md` 신규 |

### 2. `torch.set_default_device` → `mps.is_available` 패치 교체

**파일**: `rag_bench/config.py` (`setup_ssl_bypass()`)

**문제**:
```
TypeError: dropout() received an invalid combination of arguments
TypeError: set_grad_enabled() received an invalid combination of arguments
```
- `torch.set_default_device("cpu")`가 전역 `__torch_function__` 훅(`DeviceContext`) 등록
- `torch.no_grad().__exit__` → `torch.set_grad_enabled(bool)` 호출 시 훅이 가로채 TypeError 발생
- XLM-RoBERTa 기반 SPLADE 인덱싱 시 재현됨

**수정**:
```python
# 변경 전
torch.set_default_device("cpu")  # 전역 훅 등록 → dropout/no_grad 충돌

# 변경 후
torch.backends.mps.is_available = lambda: False  # 전역 훅 없이 MPS 비활성화
```

- 모든 라이브러리(SentenceTransformer 등)가 MPS 없다고 인식 → CPU 자동 사용
- `torch.no_grad()`, `dropout()`, `set_grad_enabled()` 정상 동작 확인

**커밋**: `0fbecbb`

---

## 현재 브랜치 상태

```
0fbecbb fix(config): torch.set_default_device → mps.is_available 패치로 교체
d934d59 fix(dense): KoreanBM25Encoder vocab thread-safety + OOM 분석 문서 추가
9946a38 feat(runner): 전략 병렬 실행 지원 — parallel_strategies / --pass1-workers
6d75351 fix(eval): LLM 초기화 llm_factory → LangchainLLMWrapper, 기본 모델 gpt-4o-mini
```

모두 `origin/master`에 푸시 완료.

---

## 미완료 작업 / 다음 세션

1. **RAGAS 수정 검증**: `--preset standard` 재실행 후 weighted 점수가 실제 값인지 확인
   - 수정 내용: `llm_factory()` → `LangchainLLMWrapper(ChatOpenAI(...))`, `gpt-4o-mini`
2. **병렬 실행 검증**: `--pass1-workers 4` 속도 비교
3. **SPLADE 싱글톤**: `IndexCacheManager`에 `_splade_cache` 추가 (High 우선순위)
4. **인덱스 재사용 문제**: 재실행 시에도 재인덱싱 발생하는 원인 미확인

---

## 파일 변경 목록 (이번 세션)

| 파일 | 변경 |
|------|------|
| `rag_bench/config.py` | `torch.set_default_device` → `mps.is_available = lambda: False` |
