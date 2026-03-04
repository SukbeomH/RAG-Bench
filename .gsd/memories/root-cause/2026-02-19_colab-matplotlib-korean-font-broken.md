---
title: "Colab matplotlib 한글 깨짐 — 원인 분석 + 해결 방법 비교"
tags:
  - debug
  - root-cause
  - colab
  - matplotlib
  - font
  - korean
type: root-cause
created: "2026-02-19T00:00:00+09:00"
contextual_description: "colab_visualizer.py에 폰트 설정이 전혀 없고 Colab 기본 환경에 한글 폰트가 없어 깨짐 발생. koreanize-matplotlib 설치 + init_colab에 폰트 초기화 추가로 해결"
keywords:
  - matplotlib
  - 한글 깨짐
  - NanumGothic
  - koreanize-matplotlib
  - rcParams
  - colab_visualizer
  - colab_config
related:
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
---

## Colab matplotlib 한글 깨짐 — 원인 분석 + 해결 방법 비교

---

## 원인

### 1. `colab_visualizer.py` — 폰트 설정 전무

현재 파일 전체에 matplotlib 폰트 설정이 없다.
- `plt.rcParams` 설정 없음
- `fontproperties` 사용 없음
- `font.family` / `font.sans-serif` 설정 없음

### 2. 한글 텍스트 위치 (colab_visualizer.py)

| 위치 | 내용 |
|------|------|
| line ~521 | 테이블 컬럼 레이블 `["항목", "값"]` |
| line ~559–567 | `_PHASE_LABELS` 딕셔너리 (8개 항목 전부 한글) |
| line ~613 | x축 레이블 `"소요 시간 (s)"` |
| line ~720–721 | 파이 차트 레이블 `"인덱싱 (Contextual)"`, `"전체"` |

### 3. Colab 기본 환경에 한글 폰트 없음

Google Colab Ubuntu 컨테이너에는 한글 폰트가 기본 미탑재.
matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프를 포함하지 않아
해당 문자를 `□`(두부, tofu) 또는 `?`로 렌더링.

---

## 해결 방법 비교

### 방법 A: `koreanize-matplotlib` 패키지 (권장)

**원리**: NanumGothic 폰트를 패키지 내 번들로 포함, import 시 자동 rcParams 설정

```bash
pip install koreanize-matplotlib
```

```python
import koreanize_matplotlib  # 이 한 줄로 완료
```

내부 동작:
1. 번들 NanumGothic 폰트를 `font_manager`에 등록
2. `matplotlib.font_manager._rebuild()` 폰트 캐시 재빌드
3. `rcParams['font.family'] = 'NanumGothic'`
4. `rcParams['axes.unicode_minus'] = False`

| 항목 | 평가 |
|------|------|
| 런타임 재시작 필요 | ❌ 불필요 |
| apt-get 설치 필요 | ❌ 불필요 |
| 적용 범위 | matplotlib + seaborn (rcParams 전역) |
| plotly | ❌ 별도 처리 필요 (plotly는 rcParams 미사용) |
| 유지보수 | pip 버전 관리로 단순 |

---

### 방법 B: apt-get fonts-nanum (전통적)

```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
# → 런타임 재시작 필요
```

재시작 후:
```python
import matplotlib as mpl
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = mpl.font_manager.FontProperties(fname=path).get_name()
plt.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
```

| 항목 | 평가 |
|------|------|
| 런타임 재시작 필요 | ✅ 필요 |
| 추가 패키지 없음 | ✅ |
| 노트북 흐름 단절 | ⚠️ 재시작으로 인한 셀 재실행 필요 |

---

### 방법 C: colab_visualizer.py 상단 직접 설정

```python
# colab_visualizer.py 상단
import matplotlib as _mpl
_mpl.rcParams['font.family'] = 'NanumGothic'
_mpl.rcParams['axes.unicode_minus'] = False
```

| 항목 | 평가 |
|------|------|
| 런타임 재시작 필요 | ❌ 불필요 |
| 폰트 미설치 시 | ⚠️ fallback 폰트 사용 (한글 여전히 깨짐) |
| 방법 A와 병행 필요 | ✅ 방법 A 설치 후 보조로 활용 가능 |

---

## 권장 적용 방법

### Step 1: `requirements_colab.txt`에 패키지 추가

```
koreanize-matplotlib
```

### Step 2: `colab_config.py` `init_colab()` 에 폰트 초기화 추가

```python
def _setup_korean_font() -> None:
    """matplotlib 한글 폰트 설정 (koreanize-matplotlib)."""
    try:
        import koreanize_matplotlib  # noqa: F401
        print("[Font] 한글 폰트 설정 완료 (NanumGothic)")
    except ImportError:
        print("[Warning] koreanize-matplotlib 미설치. 한글 그래프가 깨질 수 있습니다.")
```

`init_colab()` 내 `patch_dense_device()` 호출 이후에 `_setup_korean_font()` 추가.

### Step 3: plotly 그래프 한글 처리 (별도)

plotly는 rcParams를 사용하지 않으므로 개별 설정 필요:
```python
import plotly.graph_objects as go

fig.update_layout(
    font=dict(family="NanumGothic, sans-serif"),
)
```

---

## 적용 파일

| 파일 | 수정 내용 |
|------|---------|
| `rag_bench_colab/requirements_colab.txt` | `koreanize-matplotlib` 추가 |
| `rag_bench_colab/colab_config.py` | `_setup_korean_font()` 추가 + `init_colab()` 에서 호출 |
| `rag_bench_colab/colab_visualizer.py` | plotly 레이아웃에 `font=dict(family="NanumGothic")` 추가 (선택) |
