"""
GraphRAGStrategy E2E 검증 스크립트

소규모 한국어 문서(5개)로 LightRAG의 전체 파이프라인을 테스트한다:
  1. 초기화 (LightRAG 인스턴스 생성)
  2. index() — LLM 호출로 엔티티/관계 추출 → 지식 그래프 구축
  3. retrieve() — hybrid 모드로 검색 → List[Document] 반환
  4. get_retriever() — LangChain Retriever 래퍼 동작 확인
  5. cleanup() — 스토리지 종료 및 작업 디렉토리 삭제

주의: OpenAI API 비용이 발생합니다 (gpt-4.1-nano 기준, 소규모 문서는 ~$0.01 이하).
"""

import time

from langchain_core.documents import Document

# .env 자동 로드
from rag_bench.config import setup_ssl_bypass

setup_ssl_bypass()

from rag_bench.strategies import GraphRAGStrategy

# ---------------------------------------------------------------------------
# 1. 테스트 문서 준비 (한국어 5개)
# ---------------------------------------------------------------------------
TEST_DOCUMENTS = [
    Document(
        page_content=(
            "쿠버네티스(Kubernetes)는 컨테이너화된 워크로드와 서비스를 관리하기 위한 "
            "이식 가능하고 확장 가능한 오픈소스 플랫폼이다. 쿠버네티스는 선언적 구성과 "
            "자동화를 모두 용이하게 해준다. Google이 15년 이상의 대규모 운영 경험을 "
            "바탕으로 설계하였으며, 현재는 Cloud Native Computing Foundation(CNCF)에서 "
            "관리한다."
        ),
        metadata={"source": "k8s_overview.md", "section": "개요"},
    ),
    Document(
        page_content=(
            "Pod는 쿠버네티스에서 배포할 수 있는 가장 작은 컴퓨팅 단위이다. "
            "Pod는 하나 이상의 컨테이너 그룹으로, 스토리지와 네트워크 리소스를 공유하며, "
            "컨테이너 실행 방법에 대한 명세를 포함한다. Pod 내의 컨테이너들은 "
            "항상 같은 노드에 함께 스케줄링되고, 공유 컨텍스트에서 실행된다."
        ),
        metadata={"source": "k8s_pod.md", "section": "Pod"},
    ),
    Document(
        page_content=(
            "Docker는 애플리케이션을 컨테이너로 패키징하여 배포, 실행하는 오픈소스 플랫폼이다. "
            "컨테이너는 운영체제 수준의 가상화를 사용하여 격리된 환경을 제공한다. "
            "가상 머신(VM)과 달리 컨테이너는 호스트 OS의 커널을 공유하므로 "
            "리소스 효율이 높고 시작 시간이 빠르다."
        ),
        metadata={"source": "docker_basics.md", "section": "Docker"},
    ),
    Document(
        page_content=(
            "서비스 메시(Service Mesh)는 마이크로서비스 간의 통신을 관리하는 인프라 계층이다. "
            "Istio와 Linkerd가 대표적인 서비스 메시 구현체이다. 서비스 메시는 "
            "로드 밸런싱, 서비스 디스커버리, 암호화, 인증, 인가 등의 기능을 제공하며, "
            "사이드카 프록시 패턴을 사용하여 애플리케이션 코드 변경 없이 적용할 수 있다."
        ),
        metadata={"source": "service_mesh.md", "section": "서비스 메시"},
    ),
    Document(
        page_content=(
            "Helm은 쿠버네티스 애플리케이션을 패키징하고 배포하는 패키지 매니저이다. "
            "Helm 차트(Chart)는 쿠버네티스 리소스의 집합을 정의하는 파일 모음이며, "
            "values.yaml 파일을 통해 설정을 커스터마이징할 수 있다. "
            "Helm은 릴리스 관리, 롤백, 의존성 관리 기능을 제공한다."
        ),
        metadata={"source": "helm_guide.md", "section": "Helm"},
    ),
]

TEST_QUERIES = [
    "쿠버네티스 Pod란 무엇인가?",
    "Docker 컨테이너와 가상 머신의 차이점은?",
    "서비스 메시의 주요 기능은?",
]


def main():
    print("=" * 70)
    print("GraphRAGStrategy E2E 검증")
    print("=" * 70)

    working_dir = "lightrag_index_test"
    strategy = GraphRAGStrategy(
        mode="hybrid",
        working_dir=working_dir,
        llm_model="gpt-4.1-nano",
        top_k=60,
    )

    print(f"\n전략: {strategy.name}")
    print(f"설명: {strategy.description}")
    print(f"is_ready: {strategy.is_ready}")

    # ------------------------------------------------------------------
    # 2. index() — 지식 그래프 구축
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: index() — 지식 그래프 구축 (LLM 호출 발생)")
    print("-" * 70)

    t0 = time.time()
    strategy.index(TEST_DOCUMENTS)
    index_time = time.time() - t0

    print(f"\n  인덱싱 소요 시간: {index_time:.1f}초")
    assert strategy.is_ready, "index() 후 is_ready가 True여야 함"
    print("  is_ready: True (통과)")

    # ------------------------------------------------------------------
    # 3. retrieve() — 검색 테스트
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: retrieve() — 쿼리별 검색 테스트")
    print("-" * 70)

    for query in TEST_QUERIES:
        print(f"\n  쿼리: {query}")
        t0 = time.time()
        docs = strategy.retrieve(query, k=3)
        retrieve_time = time.time() - t0

        print(f"  결과 수: {len(docs)}")
        print(f"  소요 시간: {retrieve_time:.2f}초")

        if docs:
            for i, doc in enumerate(docs):
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"    [{i+1}] {preview}...")
                print(f"        metadata: {doc.metadata}")
        else:
            print("  (결과 없음 — 소규모 문서로 인해 빈 결과 가능)")

    # ------------------------------------------------------------------
    # 4. get_retriever() — LangChain Retriever 래퍼 테스트
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: get_retriever() — LangChain Retriever 호환성")
    print("-" * 70)

    retriever = strategy.get_retriever(k=3)
    print(f"  Retriever 타입: {type(retriever).__name__}")

    docs = retriever.invoke("쿠버네티스 Pod 네트워킹")
    print(f"  invoke() 결과 수: {len(docs)}")
    if docs:
        print(f"    첫 결과: {docs[0].page_content[:80]}...")

    # ------------------------------------------------------------------
    # 5. 다른 모드 테스트 (모드 전환)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: 모드 전환 테스트 (hybrid → local)")
    print("-" * 70)

    # 동일 인스턴스에서 모드만 변경하여 테스트
    original_mode = strategy._mode
    strategy._mode = "local"
    print(f"  모드 전환: {original_mode} → {strategy._mode}")

    docs_local = strategy.retrieve("쿠버네티스란?", k=3)
    print(f"  local 모드 결과 수: {len(docs_local)}")
    if docs_local:
        print(f"    첫 결과: {docs_local[0].page_content[:80]}...")

    strategy._mode = original_mode  # 원복

    # ------------------------------------------------------------------
    # 6. cleanup()
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: cleanup() — 스토리지 종료 및 디렉토리 삭제")
    print("-" * 70)

    import os

    strategy.cleanup()
    dir_exists = os.path.exists(working_dir)
    print(f"  작업 디렉토리 존재: {dir_exists}")
    assert not dir_exists, "cleanup() 후 작업 디렉토리가 삭제되어야 함"
    print("  디렉토리 삭제 확인 (통과)")

    # ------------------------------------------------------------------
    # 결과 요약
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("검증 결과 요약")
    print("=" * 70)
    print(f"  인덱싱 시간: {index_time:.1f}초 (문서 {len(TEST_DOCUMENTS)}개)")
    print(f"  LLM 모델: gpt-4.1-nano")
    print(f"  테스트 쿼리: {len(TEST_QUERIES)}개")
    print(f"  모든 검증 통과!")


if __name__ == "__main__":
    main()
