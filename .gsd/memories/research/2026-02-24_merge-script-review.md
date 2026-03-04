# merge_service_results.py 설계 정합성 검토 보고서

- **작성일:** 2026-02-24
- **대상 파일:** `rag_bench/scripts/merge_service_results.py`
- **커밋 기준:** `3057e16` (fix: RAGAS 지표 누락 시 병합 즉시 중단)
- **검토 범위:** 초기 작성(534bdd1) → 버그 수정(534bdd1) → 누락 차단(3057e16) 전 과정

---

## 1. 제작 의도 vs 구현 현황

| 의도 | 구현 현황 | 판정 |
|------|-----------|------|
| 여러 run_dir의 결과를 통합 | 구현됨, 단 충돌 정책 불일치 존재 | ⚠️ 부분 |
| 카테고리 × 전략 RAGAS 히트맵 | 구현됨 | ✅ |
| 레이턴시 비교 표 | 구현됨, 단 RAGAS 테이블과 전략 집합 불일치 가능 | ⚠️ 부분 |
| 지표 누락 시 병합 중단 | 구현됨 | ✅ |
| 데이터 오염 방지 | 일부 경로에서 여전히 발생 가능 | ⚠️ 부분 |

**종합 달성도: 72%**

---

## 2. 수정 방향 적합성 평가

### ✅ 올바르게 수정된 항목

**`_short_strategy` 충돌 방지 (reranker prefix 추가)**
- full 프리셋(colbert+flashrank 혼용) 병합 시 동일 DS 조합이 같은 key로 합쳐지는 버그를 구조적으로 차단.
- `"ColBERT Rerank (DS(bge-m3+bm25))"` → `"colbert|bge-m3+bm25"` 형식으로 리랭커를 prefix화.

**`validate_and_collect_ragas` — 지표 누락 시 중단**
- 검증 범위가 `ragas_targets`(선택된 카테고리)에만 적용되어 `--categories` 옵션과 올바르게 연동됨.
- 오류를 전부 수집한 뒤 한 번에 출력하는 방식으로 사용자 친화적.

**오류 행 레이턴시 제외**
- `load_latency`에서 `error` 컬럼 비어 있지 않은 행 필터링. 정상 동작.

---

### ⚠️ 수정 방향은 맞으나 실행이 불완전한 항목

#### ① RAGAS-레이턴시 충돌 정책 불일치 (데이터 오염 잔존)

`all_results` 충돌 시 n_qa 기준으로 교체하지만, `all_latency` 충돌 시 무조건 first-wins:

```python
# result: n_qa 더 큰 쪽 사용 (standard_run/general, n_qa=20)
if new_n > existing_n:
    all_results[category] = data

# latency: 무조건 첫 번째 유지 (service_run/general 레이턴시 그대로)
if cat not in all_latency:
    all_latency[cat] = rows
```

**결과:** `general`의 RAGAS는 standard_run 기준, 레이턴시는 service_run 기준이 된다.
같은 카테고리의 두 지표가 서로 다른 실행을 가리키는 **교차 오염** 발생.

#### ② `aggregate_latency`의 파싱 실패 시 0.0 폴백 잔존

```python
except (ValueError, TypeError):
    lat = 0.0  # ← 여전히 남아 있음
```

`load_latency`에서 error 행은 걸러지지만, `latency_ms` 필드 자체가 빈 문자열이거나
비정상 값일 경우 0ms로 평균에 포함됨.

#### ③ NaN이 `is None` 검증을 통과함

RAGAS는 일부 케이스에서 `float('nan')` 반환. `entry.get(m) is None`은 `nan`을 통과시킴:

```python
import math
math.isnan(float('nan'))  # True — 현재 검증이 잡지 못함
```

`nan`은 `float(entry[m])`에서 정상 변환되고, CSV 출력 시 `"nan"`으로 기록되어 데이터 오염.

---

### ❌ 수정 이후 새로 발생한 문제

#### ④ `dir_to_category` 딕셔너리 컴프리헨션 버그 위험

```python
# 현재 (문제 있음): _ 는 관례상 "미사용" 의미지만 여기선 category key
dir_to_category = {dir_name: data["category"] if "category" in data else dir_name
                   for _, (data, dir_name) in loaded.items()}

# 올바른 형태
dir_to_category = {dir_name: category for category, (data, dir_name) in loaded.items()}
```

추가로, `load_run_dir`에서 이미 `data.get("category", cat_dir.name)`으로 확정한
category와 다른 로직이 중복 적용될 가능성 있음.

#### ⑤ HTML 레이턴시 테이블이 RAGAS 테이블과 전략 집합이 다를 수 있음

`latency_agg`는 `all_latency`에서 독립적으로 집계되므로,
RAGAS 검증을 통과하지 않은 전략도 레이턴시 테이블에 등장할 수 있음.
두 테이블이 서로 다른 전략 목록을 가지면 보고서 독자가 혼동.

#### ⑥ `_short_strategy` pipe 문자가 표시 이름으로 그대로 노출

```
colbert|bge-m3+korean_bm25  ← HTML 히트맵과 CSV에 그대로 출력
```

내부 집계 key로는 유효하지만, 보고서 가독성이 저하됨.
집계 key와 display name을 분리해야 함:

```python
def _strategy_key(name):   ...  # "colbert|bge-m3+korean_bm25" — 집계용
def _strategy_label(name): ...  # "ColBERT / BGE-M3 + BM25"   — 표시용
```

---

## 3. 미수정 잔존 문제

| 항목 | 위치 | 영향 |
|------|------|------|
| `import re` 함수 내부 매 호출마다 | `_short_strategy` L47 | 경미 (성능) |
| `save_ragas_csv`의 `if val is not None` 방어코드 | L222–224 | 검증 통과 후 dead code — 혼란 |
| `--run_dirs` 상대경로 기준 불명확 | `main` L460 | CWD 의존 |
| HTML 전략 이름 이스케이프 미처리 | `ragas_table` L334 | XSS 잠재 위험 |
| 전략 정렬이 알파벳순 (성능/모델 기준 아님) | `strategies = sorted(...)` L295 | 가독성 |
| `validate_and_collect_ragas` 함수명 — "collect" 미수행 | L117 | 오해 유발 |
| L506 print문 — 검증 게이트 이후 중복 | `main` L506 | dead print |

---

## 4. 권고 수정 사항 (우선순위순)

### P1 — 즉시 수정 필요

```python
# 1. RAGAS-latency 충돌 정책 통일: result 선택과 연동
#    → all_results 선택 시 해당 run_dir의 latency도 함께 선택

# 2. NaN 검증 추가 (validate_and_collect_ragas)
import math
missing = [
    m for m in RAGAS_METRICS
    if entry.get(m) is None
    or (isinstance(entry.get(m), float) and math.isnan(entry.get(m)))
]

# 3. aggregate_latency 파싱 실패 시 skip
except (ValueError, TypeError):
    continue  # 0.0 대신 해당 row 제외
```

### P2 — 보고서 품질

```python
# 4. key와 display name 분리
def _strategy_key(name: str) -> str:
    """집계 key: "colbert|bge-m3+korean_bm25" """
    ...

def _strategy_label(key: str) -> str:
    """표시용: "ColBERT / BGE-M3 + BM25" """
    return key.replace("|", " / ").replace("+", " + ")

# 5. HTML latency 테이블을 ragas_agg 전략 집합으로 제한
for strat in sorted(ragas_agg.keys()):  # latency_agg 독립 순회 대신
```

### P3 — 코드 정리

```python
# 6. dir_to_category 단순화
dir_to_category = {dir_name: category for category, (data, dir_name) in loaded.items()}

# 7. save_ragas_csv의 dead code 제거 (validate 통과 후 None 불가)
row[col] = f"{scores[m]:.4f}"  # is not None 체크 불필요

# 8. import re를 모듈 최상단으로 이동

# 9. 함수명 변경: validate_and_collect_ragas → validate_ragas_completeness
```

---

## 5. 관련 커밋 이력

| 커밋 | 내용 |
|------|------|
| `534bdd1` | feat: 병합 스크립트 최초 작성 + 데이터 오염 방어 6종 |
| `3057e16` | fix: RAGAS 지표 누락 시 병합 즉시 중단 |
