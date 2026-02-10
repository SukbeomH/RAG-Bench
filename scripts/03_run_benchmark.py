"""
03. AutoRAG 벤치마크 실행 스크립트

benchmark_config.yaml 설정으로 전체 벤치마크를 실행하고,
결과를 분석하여 최적의 RAG 파이프라인을 도출한다.

필수 조건:
- Qdrant Docker 컨테이너 실행 중 (docker run -p 6333:6333 qdrant/qdrant)
- OPENAI_API_KEY 환경변수 설정
- qa.parquet, corpus.parquet 생성 완료
"""
import os
import subprocess
import sys
from pathlib import Path

# SSL 인증서 설정 (선택 - 사설 CA 사용 시 SSL_CERT_FILE 환경변수 설정)
CERT_PATH = os.getenv("SSL_CERT_FILE")
if CERT_PATH and Path(CERT_PATH).exists():
    os.environ["REQUESTS_CA_BUNDLE"] = CERT_PATH
    os.environ["CURL_CA_BUNDLE"] = CERT_PATH

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "autorag_benchmark"
CONFIG_DIR = BENCHMARK_DIR / "config"
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"



def check_prerequisites():
    """실행 전 필수 조건 확인"""
    errors = []

    # OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    # 데이터 파일
    if not (DATA_DIR / "qa.parquet").exists():
        errors.append(f"qa.parquet 파일이 없습니다: {DATA_DIR / 'qa.parquet'}")
    if not (DATA_DIR / "corpus.parquet").exists():
        errors.append(f"corpus.parquet 파일이 없습니다: {DATA_DIR / 'corpus.parquet'}")

    # 설정 파일
    if not (CONFIG_DIR / "benchmark_config.yaml").exists():
        errors.append(f"benchmark_config.yaml 파일이 없습니다.")

    if errors:
        print("[ERROR] 필수 조건이 충족되지 않았습니다:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Qdrant Docker 확인 (경고만)
    try:
        import requests

        resp = requests.get("http://localhost:6333/healthz", timeout=3)
        if resp.status_code == 200:
            print("[OK] Qdrant Docker 컨테이너 실행 중")
        else:
            print("[WARN] Qdrant 응답 이상. Docker 컨테이너를 확인해주세요.")
    except Exception:
        print("[WARN] Qdrant에 연결할 수 없습니다.")
        print("  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print("  위 명령으로 Qdrant를 시작해주세요.")


def run_validation():
    """설정 파일 검증"""
    print("\n" + "=" * 60)
    print("Phase 1: 설정 검증 (autorag validate)")
    print("=" * 60)

    autorag_bin = os.path.join(os.path.dirname(sys.executable), "autorag")
    cmd = [
        autorag_bin, "validate",
        "--config", str(CONFIG_DIR / "benchmark_config.yaml"),
        "--qa_data_path", str(DATA_DIR / "qa.parquet"),
        "--corpus_data_path", str(DATA_DIR / "corpus.parquet"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] 설정 검증 경고:")
        print(result.stderr[:500] if result.stderr else "")
        # 검증 실패해도 계속 진행 (일부 버전에서 validate 미지원)
        return True

    print("[SUCCESS] 설정 검증 통과")
    print(result.stdout)
    return True


def run_benchmark():
    """벤치마크 실행"""
    print("\n" + "=" * 60)
    print("Phase 2: 벤치마크 실행 (autorag evaluate)")
    print("=" * 60)

    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path=str(DATA_DIR / "qa.parquet"),
        corpus_data_path=str(DATA_DIR / "corpus.parquet"),
        project_dir=str(RESULTS_DIR),
    )
    evaluator.start_trial(str(CONFIG_DIR / "benchmark_config.yaml"))

    print("[SUCCESS] 벤치마크 실행 완료")
    print(f"결과 디렉토리: {RESULTS_DIR}")


def extract_best_config():
    """최적 설정 추출"""
    print("\n" + "=" * 60)
    print("Phase 3: 최적 설정 추출")
    print("=" * 60)

    # 가장 최근 trial 디렉토리 찾기
    trial_dirs = sorted(RESULTS_DIR.glob("[0-9]*"))
    if not trial_dirs:
        print("[ERROR] trial 디렉토리가 없습니다.")
        return

    latest_trial = trial_dirs[-1]
    output_path = BENCHMARK_DIR / "best_pipeline.yaml"

    autorag_bin = os.path.join(os.path.dirname(sys.executable), "autorag")
    cmd = [
        autorag_bin, "extract_best_config",
        "--trial_path", str(latest_trial),
        "--output_path", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] 최적 설정 추출: {output_path}")
    else:
        print(f"[WARN] 최적 설정 추출 실패: {result.stderr}")


def show_summary():
    """결과 요약 출력"""
    import pandas as pd

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    trial_dirs = sorted(RESULTS_DIR.glob("[0-9]*"))
    if not trial_dirs:
        return

    latest_trial = trial_dirs[-1]
    summary_path = latest_trial / "summary.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"\n{df.to_string(index=False)}")
    else:
        print("[WARN] summary.csv를 찾을 수 없습니다.")

    print(f"\n[TIP] 대시보드 실행:")
    print(f"  autorag dashboard --trial_dir {latest_trial}")


if __name__ == "__main__":
    print("AutoRAG 벤치마크 실행")
    print("=" * 60)

    # 사전 조건 확인
    check_prerequisites()

    # 설정 검증
    if not run_validation():
        print("\n설정을 수정한 후 다시 시도해주세요.")
        sys.exit(1)

    # 벤치마크 실행
    run_benchmark()

    # 최적 설정 추출
    extract_best_config()

    # 결과 요약
    show_summary()

    print("\n" + "=" * 60)
    print("다음 단계: python scripts/04_analyze_results.py")
    print("=" * 60)
