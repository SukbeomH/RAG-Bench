"""
rag_bench 패키지 End-to-End 검증 스크립트.

Step 1: 패키지 import 검증
Step 2: Parent-Child 청킹 검증 (합성 Markdown 데이터)
Step 3: DenseSparseStrategy Combo 4 (MiniLM + BM25) 인덱싱 + 검색
Step 4: BenchmarkRunner 실행 + compare()
Step 5: LangGraph 에이전트 빌드 검증
"""

import os
import sys
import shutil
import traceback
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# .env 로드
from dotenv import load_dotenv

load_dotenv()

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"
results = []


def step(name):
    """단계 시작 표시."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def record(name, success, detail=""):
    status = PASS if success else FAIL
    results.append((name, status, detail))
    print(f"  → {status} {detail}")


# =========================================================================
# Step 1: 패키지 Import 검증
# =========================================================================
step("Step 1: 패키지 Import 검증")

try:
    from rag_bench import BaseRAGStrategy, BenchmarkRunner
    from rag_bench.strategies import (
        DenseSparseStrategy,
        ColBERTStrategy,
        GraphRAGStrategy,
    )
    from rag_bench.indexing import create_parent_child_chunks, pdfs_to_markdowns
    from rag_bench.config import (
        setup_ssl_bypass,
        ensure_dirs,
        MARKDOWN_DIR,
        PARENT_STORE_PATH,
    )
    from rag_bench.cli import RAGChat
    from rag_bench.graph import build_agent_graph
    from rag_bench.graph.state import State, AgentState, QueryAnalysis
    from rag_bench.graph.prompts import (
        get_conversation_summary_prompt,
        get_query_analysis_prompt,
        get_rag_agent_prompt,
        get_aggregation_prompt,
    )

    record("패키지 Import", True, "모든 모듈 import 성공")
except Exception as e:
    record("패키지 Import", False, str(e))
    traceback.print_exc()
    sys.exit(1)


# =========================================================================
# Step 2: Parent-Child 청킹 검증
# =========================================================================
step("Step 2: Parent-Child 청킹 검증 (합성 Markdown)")

TEST_MARKDOWN_DIR = str(PROJECT_ROOT / "markdown")
TEST_PARENT_STORE = str(PROJECT_ROOT / "parent_store")

try:
    # 합성 Markdown 파일 생성
    md_dir = Path(TEST_MARKDOWN_DIR)
    md_dir.mkdir(parents=True, exist_ok=True)

    sample_md_1 = """# 쿠버네티스 개요

## Pod란 무엇인가

Pod는 쿠버네티스에서 배포할 수 있는 가장 작은 컴퓨팅 단위입니다.
Pod는 하나 이상의 컨테이너 그룹으로, 스토리지와 네트워크 리소스를 공유하며
해당 컨테이너를 실행하는 방법에 대한 명세를 가지고 있습니다.
Pod의 콘텐츠는 항상 함께 배치되고, 함께 스케줄링되며, 공유 컨텍스트에서 실행됩니다.
Pod는 애플리케이션별 "논리적 호스트"를 모델링합니다.
즉, 비교적 밀접하게 결합된 하나 이상의 애플리케이션 컨테이너를 포함합니다.

쿠버네티스 클러스터의 Pod는 두 가지 주요 방식으로 사용됩니다:
1. 단일 컨테이너를 실행하는 Pod - "Pod 당 하나의 컨테이너" 모델
2. 함께 작동해야 하는 여러 컨테이너를 실행하는 Pod - 사이드카 패턴

각 Pod는 주어진 애플리케이션의 단일 인스턴스를 실행하기 위한 것입니다.
애플리케이션을 수평으로 확장하려면 여러 Pod를 사용해야 합니다.
이를 쿠버네티스에서는 복제(replication)라고 합니다.

## Service와 네트워크

쿠버네티스의 Service는 Pod 집합에서 실행중인 애플리케이션을 네트워크 서비스로
노출하는 추상화 방법입니다. 쿠버네티스는 Pod에게 고유한 IP 주소와
Pod 집합에 대한 단일 DNS 이름을 부여하며, 이를 통해 로드밸런싱이 가능합니다.

ClusterIP, NodePort, LoadBalancer, ExternalName 등의 Service 유형이 있습니다.
"""

    sample_md_2 = """# Docker 기초

## 컨테이너와 가상머신 차이

Docker 컨테이너는 가상머신(VM)과는 다른 가상화 방식을 사용합니다.
VM은 하이퍼바이저를 통해 전체 운영체제를 가상화하지만,
컨테이너는 호스트 OS의 커널을 공유하면서 프로세스를 격리합니다.

이로 인해 컨테이너는 다음과 같은 장점이 있습니다:
- 더 빠른 시작 시간 (초 단위 vs 분 단위)
- 더 적은 리소스 사용 (MB 단위 vs GB 단위)
- 더 높은 밀도 (단일 호스트에서 더 많은 인스턴스 실행)
- 이식성 (어디서나 동일하게 실행)

## Dockerfile

Dockerfile은 Docker 이미지를 빌드하기 위한 텍스트 파일입니다.
FROM, RUN, COPY, CMD 등의 명령어를 사용하여 이미지를 구성합니다.
멀티스테이지 빌드를 사용하면 최종 이미지 크기를 최소화할 수 있습니다.
"""

    (md_dir / "kubernetes_guide.md").write_text(sample_md_1, encoding="utf-8")
    (md_dir / "docker_basics.md").write_text(sample_md_2, encoding="utf-8")
    print(f"  합성 Markdown 파일 2개 생성: {md_dir}/")

    # 청킹 실행
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=TEST_MARKDOWN_DIR,
        parent_store_path=TEST_PARENT_STORE,
        min_parent_size=200,  # 테스트용 완화
        max_parent_size=5000,
        child_chunk_size=300,
        child_chunk_overlap=50,
    )

    print(f"  Parents: {len(parent_pairs)}, Children: {len(child_chunks)}")

    # parent_store JSON 파일 확인
    json_files = list(Path(TEST_PARENT_STORE).glob("*.json"))
    print(f"  Parent JSON files: {len(json_files)}")

    ok = len(parent_pairs) >= 1 and len(child_chunks) >= 1 and len(json_files) >= 1
    record(
        "Parent-Child 청킹",
        ok,
        f"parents={len(parent_pairs)}, children={len(child_chunks)}, json_files={len(json_files)}",
    )

except Exception as e:
    record("Parent-Child 청킹", False, str(e))
    traceback.print_exc()

# =========================================================================
# Step 3: DenseSparseStrategy 인덱싱 + 검색
# =========================================================================
step("Step 3: DenseSparseStrategy Combo 4 (MiniLM + BM25) 인덱싱 + 검색")

try:
    # SSL 우회 설정 (회사 네트워크 환경)
    setup_ssl_bypass()

    strategy = DenseSparseStrategy(combo_id=4, qdrant_path="qdrant_db_verify_test")
    print(f"  전략: {strategy.name}")
    print(f"  설명: {strategy.description}")

    # 인덱싱
    print("\n  --- Indexing ---")
    strategy.index(child_chunks)
    print(f"  is_ready: {strategy.is_ready}")

    # 검색 테스트
    print("\n  --- Retrieval ---")
    test_queries = [
        "쿠버네티스 Pod란?",
        "Docker와 VM의 차이",
        "Service 종류",
    ]

    retrieval_ok = True
    for q in test_queries:
        docs = strategy.retrieve(q, k=3)
        print(f"  '{q}' → {len(docs)} results")
        if docs:
            print(f"    첫 결과 (100자): {docs[0].page_content[:100]}...")
        else:
            retrieval_ok = False

    record(
        "DenseSparse 인덱싱+검색",
        strategy.is_ready and retrieval_ok,
        f"is_ready={strategy.is_ready}, 3개 쿼리 검색 완료",
    )

except Exception as e:
    record("DenseSparse 인덱싱+검색", False, str(e))
    traceback.print_exc()

# =========================================================================
# Step 4: BenchmarkRunner 실행
# =========================================================================
step("Step 4: BenchmarkRunner 실행")

try:
    runner = BenchmarkRunner(
        strategies=[strategy],
        queries=["쿠버네티스 Pod란?", "Docker와 VM의 차이"],
        k=3,
    )
    run_results = runner.run()
    runner.compare()

    # DataFrame 테스트
    df = runner.to_dataframe()
    if df is not None:
        print(f"\n  DataFrame shape: {df.shape}")
        print(df.to_string(index=False))

    ok = len(run_results) > 0
    record("BenchmarkRunner", ok, f"전략 {len(run_results)}개 실행 완료")

except Exception as e:
    record("BenchmarkRunner", False, str(e))
    traceback.print_exc()

# =========================================================================
# Step 5: LangGraph 에이전트 빌드 검증
# =========================================================================
step("Step 5: LangGraph 에이전트 빌드 검증")

try:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-proj-xxx"):
        record("LangGraph 에이전트", False, "OPENAI_API_KEY 미설정")
    else:
        graph = build_agent_graph(strategy)
        print(f"  그래프 객체: {type(graph)}")
        record(
            "LangGraph 에이전트 빌드",
            True,
            f"그래프 컴파일 성공 ({type(graph).__name__})",
        )
except Exception as e:
    record("LangGraph 에이전트 빌드", False, str(e))
    traceback.print_exc()


# =========================================================================
# Cleanup
# =========================================================================
step("Cleanup")

try:
    strategy.cleanup()
    # 테스트용 Qdrant DB 삭제
    qdrant_test_path = PROJECT_ROOT / "qdrant_db_verify_test"
    if qdrant_test_path.exists():
        shutil.rmtree(qdrant_test_path)
        print(f"  Removed: {qdrant_test_path}")
    # 합성 Markdown 정리
    for f in (md_dir / "kubernetes_guide.md", md_dir / "docker_basics.md"):
        if f.exists():
            f.unlink()
            print(f"  Removed: {f.name}")
    print("  Cleanup 완료")
except Exception as e:
    print(f"  ⚠️ Cleanup 중 오류 (무시): {e}")


# =========================================================================
# 최종 결과 요약
# =========================================================================
print(f"\n{'=' * 60}")
print("  검증 결과 요약")
print(f"{'=' * 60}")
for name, status, detail in results:
    print(f"  {status} {name}: {detail}")

total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")

if failed > 0:
    sys.exit(1)
else:
    print("\n  🎉 모든 검증 통과!")
    sys.exit(0)
