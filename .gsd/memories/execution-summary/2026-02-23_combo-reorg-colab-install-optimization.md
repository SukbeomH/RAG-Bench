---
title: "벤치마크 조합 재구성 + Colab 설치 최적화"
tags:
  - execution
  - summary
  - refactor
  - colab
  - benchmark
type: execution-summary
created: 2026-02-23T00:00:00+09:00
contextual_description: "minilm/fastembed_bm25 제거, openai-large/upstage 복원으로 60개 조합(5×2×6) 재편. Colab Cell 1.2 uv + flash-attn wheel 캐시 최적화. README 전면 업데이트."
keywords:
  - combo-reorg
  - minilm-removed
  - fastembed_bm25-removed
  - openai-large
  - upstage
  - 60-combos
  - uv-package-manager
  - flash-attn-wheel-cache
  - colab-optimization
related:
  - 2026-02-20_viz-phase2-complete
---

## 벤치마크 조합 재구성 + Colab 설치 최적화

### 세션 날짜: 2026-02-23

---

## 변경 내용 요약

### 1. 벤치마크 조합 재구성 (72개 → 60개)

**제거:**
- `minilm` (all-MiniLM-L6-v2, 384d) — Dense 레지스트리에서 제거
- `fastembed_bm25` (FastEmbed BM25) — Sparse 레지스트리에서 제거
- `openai-small` (text-embedding-3-small) — 처음부터 PRESETS에 없었음

**복원:**
- `openai-large` (text-embedding-3-large, 3072d) — `_ALL_DENSE_MODELS`에 추가
- `upstage` (solar-embedding-1-query, 4096d) — `_ALL_DENSE_MODELS`에 추가

**결과 조합 구조:**
```
Dense 5종: kosimcse | e5 | bge-m3 | openai-large | upstage
Sparse 2종: korean_bm25 | splade
Mode  6종: hybrid / +contextual / +colbert / +colbert+ctx / +flashrank / +flashrank+ctx
총합: 5 × 2 × 6 = 60개
```

**프리셋:**
- `quick`: bge-m3 × korean_bm25 × (hybrid, flashrank) = **2개**
- `standard`: 5 × 2 × (hybrid, flashrank) = **20개**
- `full`: 5 × 2 × 6 = **60개**

### 2. 수정 파일 (코어)

| 파일 | 변경 내용 |
|------|-----------|
| `rag_bench/strategies/dense_sparse.py` | DENSE_MODELS/DENSE_DIMS에서 minilm 제거, openai-large/upstage 복원; SPARSE_TYPES fastembed_bm25 제거; `_init_dense()` HF/OpenAI/Upstage 분기 복원; 기본 sparse `"korean_bm25"` |
| `rag_bench/combo/spec.py` | `_HF_DENSE_MODELS`=[kosimcse,e5,bge-m3]; `_ALL_DENSE_MODELS`=HF+유료; quick/standard/full 프리셋 조합 수 반영 |
| `rag_bench/scripts/run_all_combos.py` | docstring 36개→60개 업데이트 |

### 3. Colab 동기화

| 파일 | 변경 내용 |
|------|-----------|
| `rag_bench_colab/colab_config.py` | get_qdrant_path() docstring 예시 fastembed_bm25→korean_bm25 |
| `rag_bench_colab/colab_runner.py` | `_strategy_name_from_spec` 예시 minilm→bge-m3 |
| `rag_bench_colab/rag_benchmark.ipynb` | Cell 0 126-Combo→60-Combo; Cell 1.2 uv+flash-attn wheel 교체; 중복 cell-3 삭제 (39→38셀) |

### 4. Colab Cell 1.2 설치 최적화

**적용 내용:**
1. **uv 패키지 매니저** (pip 대비 ~8배 빠름)
   - `pip install uv` → `uv pip install -r requirements_colab.txt`
   - `UV_CACHE_DIR=Drive/colab_cache/uv` 로 Drive 캐시
   - pip fallback 포함

2. **flash-attn 사전 빌드 wheel 캐시**
   - CUDA/Python/torch/cxx11abi 버전 조합으로 wheel 파일명 동적 구성
   - 설치 우선순위: ① Drive 캐시 → ② GitHub wheel 다운로드 → ③ 소스 빌드
   - 다운로드 완료 후 Drive에 저장하여 재시작 시 ~5초로 재사용

3. **Drive 마운트 이동**: Cell 1.3 → Cell 1.2 상단으로 통합
   - uv 캐시 + flash-attn wheel 캐시를 `_colab_cache` 단일 변수로 공용

### 5. README 전면 업데이트

| 파일 | 주요 변경 |
|------|-----------|
| `README.md` | 72→60개, 3-Layer 다이어그램 재작성, Dense/Sparse 모델 표, Colab 프리셋, 레거시 모드 제거 |
| `rag_bench/README.md` | 60개 조합, 프리셋 표(quick=2,standard=20,full=60), 인덱스 캐싱 12→10회, 레이어별 기여도 예시 |
| `rag_bench_colab/README.md` | 60개 조합, Cell 1.2 uv+flash-attn 최적화 내용 추가, 프리셋 표 |

---

## 커밋 목록

```
605a1b2 docs: 모델 조합 재구성 — 72개 → 60개 (5×2×6) + Dense 5종 / Sparse 2종 반영
feat(colab): uv 패키지 매니저 통합 + Drive 캐시 일원화
feat(colab): flash-attn wheel 캐시 최적화 + 중복 Cell 1.2 제거
refactor(colab): minilm·fastembed_bm25 제거 + openai-large·upstage 반영
refactor(combo): minilm·fastembed_bm25 제거 + openai-large·upstage 복원
```

---

## 병렬 작업 판단

`parallel_strategies` (전략 병렬화)는 Colab T4 GPU에서 **적용 불가**로 결론:
- HF 모델 동시 로딩 시 OOM 위험
- 체크포인트 순차 구조와 충돌
- `parallel_queries`는 이미 적용됨 (쿼리 레벨 병렬화)
