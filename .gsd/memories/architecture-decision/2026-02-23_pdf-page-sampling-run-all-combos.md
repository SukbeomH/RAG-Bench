---
title: "PDF 페이지 샘플링 기능 — run_all_combos.py --sample-pages / --regenerate-qa"
tags:
  - architecture-decision
  - benchmark
  - pdf-sampling
  - run_all_combos
  - qa-generation
type: architecture-decision
created: 2026-02-23T12:00:00+09:00
contextual_description: "Colab 노트북처럼 PDF 페이지 일부만 샘플링하여 인덱싱 + QA 생성 후 벤치마크하는 기능을 run_all_combos.py에 추가. --sample-pages로 docs/*.pdf → rag_bench/docs/*.md 교체, --regenerate-qa로 QA 재생성."
keywords:
  - sample-pages
  - page-sample-ratio
  - max-sample-pages
  - regenerate-qa
  - pdfs_to_markdowns
  - RAGAS KG
  - colab
  - rag_benchmark.ipynb
related:
  - 2026-02-23_benchmark-data-contamination-scope-change
  - 2026-02-23_dense-filter-append-results-timing-tools
---

## PDF 페이지 샘플링 기능 도입 (2026-02-23)

### 배경
- Colab 노트북(`rag_benchmark.ipynb`)은 `prepare_qa(sample_pages=True)`로 PDF 일부 페이지만 사용
- 로컬 `run_all_combos.py`는 전체 문서(763 chunks)를 항상 인덱싱
- QA와 인덱스 코퍼스가 동일해야 의미 있는 평가 → 샘플링 기능 추가

### 신규 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--sample-pages` | False | PDF 페이지 샘플링 후 rag_bench/docs/*.md 교체 |
| `--page-sample-ratio` | 0.1 | 샘플링 비율 (10%) |
| `--max-sample-pages` | 5 | 최대 샘플 페이지 수 |
| `--max-qa-per-page` | 2 | parent 청크당 QA 수 (총 QA = 청크 수 × 이 값) |
| `--regenerate-qa` | False | 기존 qa_dataset.json 무시, 현재 문서에서 QA 재생성 |

### 실행 흐름

```
Step 0 (--sample-pages 시):
  docs/*.pdf → pdfs_to_markdowns(sample_pages=True)
            → rag_bench/docs/*.md 교체

Step 1 (--regenerate-qa 또는 --sample-pages 시):
  rag_bench/docs/*.md → create_parent_child_chunks()
                      → _compute_effective_num_qa()
                      → _generate_qa_ragas() (RAGAS KG 빌드)
                      → qa_dataset.json 저장

Step 2~7: 기존과 동일 (샘플된 docs 기준으로 인덱싱)
```

### 사용 예시
```bash
# 전체: PDF 샘플링 + QA 재생성 + 전체 벤치마크
python -m rag_bench.scripts.run_all_combos \
  --preset full --k 3 --top_n 10 \
  --sample-pages --page-sample-ratio 0.1 --max-sample-pages 5

# docs/*.md 이미 샘플링된 경우: QA만 재생성
python -m rag_bench.scripts.run_all_combos \
  --preset full --k 3 --top_n 10 \
  --regenerate-qa
```

### 수정 파일
- `rag_bench/scripts/run_all_combos.py`: Step 0/1 로직 + 5개 argparse 옵션 추가
- `rag_bench/scripts/run_all_combos.py`: `DOCS_DIR`, `pdfs_to_markdowns` import 추가

### 샘플링 결과 예시 (2026-02-23)
- AI 현황 보고서 (222p) → 5p 샘플 → 26 child chunks (구: 763)
- SPRi AI Brief (30p) → 3p 샘플
- QA 목표: parent 2개 × 2 = 4개

### 주의
- `--sample-pages`는 `rag_bench/docs/*.md`를 덮어씀 (기존 전체 내용 소실)
- RAGAS KG 생성에 `rapidfuzz` 패키지 필요 (`uv add rapidfuzz`)
- 코퍼스 변경 시 반드시 Qdrant 인덱스/캐시 삭제 후 실행할 것
