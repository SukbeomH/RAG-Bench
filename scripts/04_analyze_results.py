"""
04. 벤치마크 결과 분석 스크립트

AutoRAG 벤치마크 결과를 파싱하여 임베딩 모델별 성능을 비교 분석한다.

비교 축:
1. Dense 임베딩: KoSimCSE vs E5 vs BGE-M3 vs MiniLM vs OpenAI
2. Sparse 검색: ko_kiwi vs ko_okt vs space (BM25 토크나이저)
3. 검색 방식: Dense only vs BM25 only vs Hybrid RRF
4. Reranker 효과: pass vs FlashRank
5. 청킹 크기: 500 vs 1024 토큰
"""
import json
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "autorag_benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"


def find_latest_trial():
    """가장 최근 trial 디렉토리 찾기"""
    trial_dirs = sorted(RESULTS_DIR.glob("[0-9]*"))
    if not trial_dirs:
        print("[ERROR] 벤치마크 결과가 없습니다.")
        print("먼저 scripts/03_run_benchmark.py를 실행해주세요.")
        sys.exit(1)
    return trial_dirs[-1]


def load_trial_info(trial_dir: Path):
    """Trial 메타 정보 로드"""
    trial_json = trial_dir / "trial.json"
    if trial_json.exists():
        with open(trial_json) as f:
            return json.load(f)
    return {}


def analyze_retrieval(trial_dir: Path):
    """Retrieval 노드 결과 분석"""
    print("\n" + "=" * 60)
    print("1. Retrieval 성능 비교")
    print("=" * 60)

    # retrieval summary 찾기
    retrieval_dirs = list(trial_dir.glob("**/retrieval"))
    if not retrieval_dirs:
        print("[WARN] retrieval 결과를 찾을 수 없습니다.")
        return None

    retrieval_dir = retrieval_dirs[0]
    summary_path = retrieval_dir / "summary.csv"

    if not summary_path.exists():
        # 개별 결과 파일에서 수집
        result_files = list(retrieval_dir.glob("*.csv"))
        if not result_files:
            print("[WARN] retrieval CSV 파일을 찾을 수 없습니다.")
            return None
        summary_path = result_files[0]

    df = pd.read_csv(summary_path)
    print(f"\n  결과 파일: {summary_path}")
    print(f"  총 실험 수: {len(df)}")

    # 메트릭 컬럼 식별
    metric_cols = [
        c for c in df.columns if any(
            m in c for m in ["f1", "recall", "precision", "mrr", "ndcg", "map"]
        )
    ]
    if metric_cols:
        print(f"\n  평가 메트릭: {metric_cols}")
        print(f"\n{df[['module_type'] + metric_cols].to_string(index=False) if 'module_type' in df.columns else df[metric_cols].to_string(index=False)}")

    return df


def analyze_reranker(trial_dir: Path):
    """Reranker 노드 결과 분석"""
    print("\n" + "=" * 60)
    print("2. Reranker 효과 비교")
    print("=" * 60)

    reranker_dirs = list(trial_dir.glob("**/passage_reranker"))
    if not reranker_dirs:
        reranker_dirs = list(trial_dir.glob("**/reranker"))

    if not reranker_dirs:
        print("[WARN] reranker 결과를 찾을 수 없습니다.")
        return None

    reranker_dir = reranker_dirs[0]
    result_files = list(reranker_dir.glob("*.csv"))
    if not result_files:
        print("[WARN] reranker CSV 파일을 찾을 수 없습니다.")
        return None

    df = pd.read_csv(result_files[0])
    print(f"\n  결과 파일: {result_files[0]}")

    metric_cols = [
        c for c in df.columns if any(m in c for m in ["f1", "recall", "precision"])
    ]
    if metric_cols:
        print(f"\n{df[metric_cols].to_string(index=False)}")

    return df


def analyze_generator(trial_dir: Path):
    """Generator 노드 결과 분석"""
    print("\n" + "=" * 60)
    print("3. Generator 성능 비교")
    print("=" * 60)

    gen_dirs = list(trial_dir.glob("**/generator"))
    if not gen_dirs:
        print("[WARN] generator 결과를 찾을 수 없습니다.")
        return None

    gen_dir = gen_dirs[0]
    result_files = list(gen_dir.glob("*.csv"))
    if not result_files:
        print("[WARN] generator CSV 파일을 찾을 수 없습니다.")
        return None

    df = pd.read_csv(result_files[0])
    print(f"\n  결과 파일: {result_files[0]}")

    metric_cols = [
        c for c in df.columns if any(
            m in c for m in ["bleu", "meteor", "rouge", "sem_score"]
        )
    ]
    if metric_cols:
        print(f"\n{df[metric_cols].to_string(index=False)}")

    return df


def print_overall_summary(trial_dir: Path):
    """전체 요약 출력"""
    print("\n" + "=" * 60)
    print("4. 전체 요약 (Best Pipeline)")
    print("=" * 60)

    summary_path = trial_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"\n{df.to_string(index=False)}")
    else:
        print("[WARN] 전체 summary.csv를 찾을 수 없습니다.")

    # Best config 확인
    best_config = BENCHMARK_DIR / "best_pipeline.yaml"
    if best_config.exists():
        print(f"\n  최적 파이프라인 설정: {best_config}")
        with open(best_config) as f:
            print(f.read())


def print_comparison_table():
    """비교 테이블 출력"""
    print("\n" + "=" * 60)
    print("5. 비교 축 요약")
    print("=" * 60)

    comparisons = [
        ("Dense 임베딩", "KoSimCSE vs E5 vs BGE-M3 vs MiniLM vs OpenAI"),
        ("Sparse 검색", "ko_kiwi vs ko_okt vs space (BM25 토크나이저)"),
        ("검색 방식", "Dense only vs BM25 only vs Hybrid RRF"),
        ("Reranker 효과", "pass (없음) vs FlashRank"),
        ("생성 온도", "temperature 0 vs 0.3"),
    ]

    print(f"\n{'비교 축':<20} {'비교 대상'}")
    print("-" * 60)
    for axis, targets in comparisons:
        print(f"{axis:<20} {targets}")


if __name__ == "__main__":
    print("AutoRAG 벤치마크 결과 분석")
    print("=" * 60)

    # 최신 trial 찾기
    trial_dir = find_latest_trial()
    print(f"분석 대상: {trial_dir}")

    # Trial 메타 정보
    trial_info = load_trial_info(trial_dir)
    if trial_info:
        print(f"Trial 정보: {json.dumps(trial_info, indent=2, ensure_ascii=False)}")

    # 각 노드별 분석
    analyze_retrieval(trial_dir)
    analyze_reranker(trial_dir)
    analyze_generator(trial_dir)

    # 전체 요약
    print_overall_summary(trial_dir)

    # 비교 테이블
    print_comparison_table()

    print("\n" + "=" * 60)
    print("[TIP] 대시보드로 시각화:")
    print(f"  autorag dashboard --trial_dir {trial_dir}")
    print("=" * 60)
