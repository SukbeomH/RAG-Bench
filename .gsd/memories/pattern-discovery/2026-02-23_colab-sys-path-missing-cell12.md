---
title: "Colab Cell 1.2 sys.path 미설정 — colab_config import 실패"
tags:
  - pattern
  - learning
  - colab
  - import
  - sys-path
type: pattern-discovery
created: 2026-02-23T10:00:00+09:00
contextual_description: "Colab 노트북 Cell 1.3에서 from colab_config import init_colab 실패. Cell 1.2에 sys.path 미설정이 원인. /content/RAG-Bench와 /content/RAG-Bench/rag_bench_colab 양쪽 추가 필요."
keywords:
  - colab
  - sys.path
  - ModuleNotFoundError
  - colab_config
  - rag_benchmark.ipynb
  - Cell 1.2
  - import
related:
  - 2026-02-23_combo-reorg-colab-install-optimization
---

## Colab Cell 1.2 sys.path 미설정 — colab_config import 실패

### 증상
```
ModuleNotFoundError: No module named 'colab_config'
```
Cell 1.3에서 `from colab_config import init_colab` 실패.

### 원인
`rag_benchmark.ipynb`의 Cell 1.2가 패키지를 설치하지만 Python path에 프로젝트 경로를 추가하지 않음.

- `colab_config.py` 위치: `/content/RAG-Bench/rag_bench_colab/colab_config.py`
- Colab 기본 working dir: `/content` (프로젝트 루트 미포함)

### 수정 (Cell 1.2 상단에 추가)
```python
from pathlib import Path
_project_root = Path("/content/RAG-Bench")
for _p in [str(_project_root), str(_project_root / "rag_bench_colab")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

### 왜 두 경로 모두 필요한가
| 경로 | 용도 |
|------|------|
| `/content/RAG-Bench` | `rag_bench` 패키지 (pip install 없이) |
| `/content/RAG-Bench/rag_bench_colab` | `colab_config`, `colab_runner`, `colab_visualizer` 직접 import |

### 수정 파일
- `rag_bench_colab/rag_benchmark.ipynb` — Cell 1.2 상단에 sys.path 설정 블록 추가
