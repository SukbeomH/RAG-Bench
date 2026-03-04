# GPU/MPS OOM 사전 파악 분석 노트

> 작성일: 2026-02-20
> 환경: Apple Silicon (MPS) — 현재는 CPU 강제 모드로 OOM 없음
> 목적: CUDA 환경 이식 또는 병렬 실행 확장 시 참고용

---

## 1. 현재 메모리 구조

### 컴포넌트별 디바이스 및 메모리

| 컴포넌트 | 디바이스 | 예상 메모리 |
|---------|---------|------------|
| Dense — KoSimCSE (768d) | **CPU 강제** | ~440MB |
| Dense — E5-Large (1024d) | **CPU 강제** | ~1.7GB |
| Dense — BGE-M3 (1024d) | **CPU 강제** | ~2.3GB |
| Dense — MiniLM (384d) | **CPU 강제** | ~132MB |
| Sparse — SPLADE | **CUDA 자동** | ~400MB |
| Sparse — Korean BM25 | RAM | ~50MB |
| Sparse — FastEmbed BM25 | RAM | ~5MB |
| Reranker — ColBERT (fp16) | **CUDA 자동** | ~280MB |
| Reranker — FlashRank (ONNX) | CPU only | 4~150MB |

### 최악 시나리오 (E5 + SPLADE + ColBERT)

```
CPU RAM : 1.7GB (E5)
CUDA    : ~680MB (SPLADE 400 + ColBERT 280)
```

→ VRAM 1GB 이상이면 안전, 4GB+ 권장

---

## 2. 현재 OOM 방지 메커니즘

### 2-1. MPS CPU 강제 (config.py)

```python
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.set_default_device("cpu")
torch.mps.empty_cache()
```

Apple Silicon에서 MPS 대신 CPU를 강제 사용하여 OOM 원천 차단.

### 2-2. 전략 생성 후 GC (run_all_combos.py)

```python
def _release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

전략 생성 완료 후 호출하여 중간 텐서/캐시 해제.

### 2-3. 모델 싱글톤 (IndexCacheManager)

- ColBERT 모델: `_colbert_model` — 1회 로드 후 12개 전략 공유
- FlashRank Ranker: `_flashrank_ranker` — 1회 로드 후 12개 전략 공유
- Dense/Sparse 모델: `share_embeddings()` — 동일 조합 재사용

### 2-4. SPLADE 배치 처리 (dense_sparse.py)

```python
for i in range(0, len(texts), batch_size):  # batch_size=32
    batch_vecs = self._compute_vectors_batch(batch)
```

전체를 한 번에 GPU에 올리지 않고 배치 단위로 처리.

---

## 3. 발견된 위험 요소

### 3-1. SPLADE 싱글톤 미적용 ⚠️

```python
# dense_sparse.py (현재)
elif sparse_type == "splade":
    self._sparse_embeddings = SpladeEncoder()  # 매번 새 인스턴스 생성
```

SPLADE는 ColBERT/FlashRank와 달리 싱글톤이 없어 전략마다 GPU에 새로 로드됨.

**병렬 실행 시 위험:**
```
--pass1-workers 4 + SPLADE 조합 4개 동시 실행
→ 400MB × 4 = 1.6GB CUDA 동시 점유
→ 2GB VRAM 환경에서 OOM
```

**권장 수정 방향:**
```python
# IndexCacheManager에 SPLADE 싱글톤 추가
_splade_cache: dict = field(default_factory=dict)

def get_splade_encoder(self, model_name: str) -> SpladeEncoder:
    if model_name not in self._splade_cache:
        self._splade_cache[model_name] = SpladeEncoder(model_name)
    return self._splade_cache[model_name]
```

### 3-2. ColBERT MPS OOM 주석 (colbert.py)

```python
def _detect_device(self) -> str:
    """CUDA → CPU 자동 감지 (MPS는 OOM 위험으로 사용하지 않음)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

MPS는 의도적으로 제외됨. CUDA 없는 환경에서는 자동으로 CPU 사용.

### 3-3. 병렬 실행 시 Dense 모델 RAM 압박 ⚠️

Dense 모델은 CPU에 로드되므로 CUDA OOM은 없지만 RAM 압박 가능:

```
--pass1-workers 4 + BGE-M3 조합 4개 동시 실행 (이상적으로는 캐시 공유)
실제로는 ThreadPool이므로 같은 프로세스 → 캐시 공유 → 문제없음
```

ThreadPoolExecutor 기반 병렬화이므로 메모리는 공유됨. 문제 없음.

---

## 4. 실행 전 메모리 체크 코드 (미구현, 참고용)

```python
def check_memory_before_run(preset: str, parallel_workers: int = 0):
    """실행 전 GPU/RAM 여유 메모리 확인."""
    import torch, psutil

    # RAM 체크
    free_ram_gb = psutil.virtual_memory().available / 1024**3
    required_ram = {"quick": 1.0, "standard": 3.0, "full": 5.0}.get(preset, 3.0)
    if free_ram_gb < required_ram:
        print(f"⚠️ 여유 RAM {free_ram_gb:.1f}GB < 권장 {required_ram}GB")

    # CUDA 체크
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info()[0] / 1024**3
        splade_count = parallel_workers if parallel_workers > 0 else 1
        required_vram = 0.4 * splade_count + 0.28  # SPLADE × N + ColBERT
        if free_vram_gb < required_vram:
            print(f"⚠️ 여유 VRAM {free_vram_gb:.1f}GB < 필요 {required_vram:.1f}GB")
            print(f"   SPLADE {splade_count}개 병렬 → --pass1-workers 줄이거나 SPLADE 조합 제외")

    # MPS 체크
    elif torch.backends.mps.is_available():
        print("MPS 감지 → CPU 강제 모드 (OOM 없음)")
```

---

## 5. 위험도 요약

| 위험 | 발생 조건 | 현재 상태 | 우선순위 |
|------|---------|---------|---------|
| MPS OOM | Apple Silicon + MPS 활성화 | ✅ CPU 강제로 방지 | 완료 |
| CUDA OOM — SPLADE 병렬 | `--pass1-workers N` + SPLADE N개 동시 | ⚠️ 싱글톤 없음 | **High** |
| CUDA OOM — ColBERT | ColBERT preset 사용 시 | ✅ standard에 없음 | Low |
| RAM OOM — Dense 병렬 | ThreadPool이므로 메모리 공유됨 | ✅ 문제없음 | 완료 |
| fp16 미사용 | 모든 모델 float32 로드 | ⚠️ 50% 절감 가능 | Low |

---

## 6. 권장 실행 구성

```bash
# 경량 (RAM < 2GB, GPU 불필요)
python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only

# 표준 (RAM ~3GB, GPU 680MB)
python -m rag_bench.scripts.run_all_combos --preset standard

# 표준 + 병렬 (RAM ~3GB, GPU: SPLADE 싱글톤 적용 후 안전)
python -m rag_bench.scripts.run_all_combos --preset standard --pass1-workers 4
```

---

## 7. 향후 개선 과제

- [ ] SPLADE 싱글톤 추가 (`IndexCacheManager._splade_cache`)
- [ ] 실행 전 preflight 메모리 체크 함수 구현 (`config.py` 또는 `verify_env.py`)
- [ ] fp16 양자화 옵션 추가 (HuggingFaceEmbeddings `model_kwargs={"torch_dtype": torch.float16}`)
- [ ] `--pass1-workers` 상한 자동 계산 (VRAM / 400MB)
