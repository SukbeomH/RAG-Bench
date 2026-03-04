---
title: "FlashRank /tmp 캐시 재부팅 삭제 버그"
tags:
  - rag_bench
  - flashrank
  - bug
  - root-cause
type: root-cause
created: 2026-02-24T00:00:00Z
contextual_description: "FlashRank 모델이 /tmp에 저장되어 재부팅 시 삭제 → 20개 전략 실패. MODELS_DIR/flashrank로 영속화 해결."
keywords:
  - FlashRank
  - /tmp
  - 캐시
  - 재부팅
  - ONNX
  - 모델영속화
  - cache_dir
  - MODELS_DIR
---

## FlashRank /tmp 캐시 재부팅 삭제 버그

## 증상
rag_bench full preset 실행 시 FlashRank 관련 20개 전략 전부 실패:
```
✗ ONNXRuntimeError: Load model from /tmp/ms-marco-MultiBERT-L-12/flashrank-MultiBERT-L12_Q.onnx failed. File doesn't exist
```

## 근본 원인
`flashrank` 라이브러리 `Config.py`에 `default_cache_dir = "/tmp"` 하드코딩.
macOS `/tmp`는 재부팅 시 자동 삭제. 모델 98.7MB가 사라짐.

## 영향 범위
full preset: 5 dense × 2 sparse × 1 flashrank × 2 contextual = **20개 전략 전부 실패**.
재부팅할 때마다 반복 발생.

## 수정 (`rag_bench/combo/cache.py`)
`get_flashrank_ranker()`에서 `cache_dir` 명시:
```python
flashrank_cache_dir = MODELS_DIR / "flashrank"  # rag_bench/_models/flashrank/
flashrank_cache_dir.mkdir(parents=True, exist_ok=True)
Ranker(model_name=..., max_length=..., cache_dir=str(flashrank_cache_dir))
```
모델 `rag_bench/_models/flashrank/ms-marco-MultiBERT-L-12/` 에 영속 저장 완료 (98.7MB).

## 커밋
`7c08f31` fix+feat: FlashRank 모델 영속화 + 레이어별 실행 필터 추가
