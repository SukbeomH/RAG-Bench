## Thread Safety
- PyLate ColBERT `encode()` + `rank.rerank()` is NOT thread-safe — use module-level `threading.Lock()` — colbert_rerank.py
- FlashRank ONNX inference IS thread-safe — no lock needed — flashrank_rerank.py
- DenseSparse BM25 already has `_vocab_lock` for vocab writes — dense_sparse.py:49

## Benchmark Execution
- Restart with cached Qdrant indices: Pass 1 ~3-5min (vs 12h from scratch)
- `--pass1-workers N` parallel + ColBERT → tensor crash without Lock
- `--pass2-workers N` + `RunConfig(max_workers=16)` → Pass 2 ~4-5× speedup
- Pass 2 progress invisible without `[N/total] ETA` logging — now fixed in runner.py

## Contextual Retrieval Optimization
- `pre_enriched` BM25 fit must use enriched text in BOTH new-index AND cache-hit paths — cache.py
- `ContextualRetrievalStrategy(base_strategy=None)` allowed after Optional type annotation — contextual_retrieval.py
- Step 2.5 in run_all_combos.py: `enrich_only()` once → all contextual strategies reuse — run_all_combos.py:273

## Worktree
- Optimization work in `.claude/worktrees/ctx-retrieval-optimization` branch `worktree-ctx-retrieval-optimization`
- All ctx-retrieval changes committed, need merge to master after benchmark validates
