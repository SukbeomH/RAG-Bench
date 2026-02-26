"""
K8s 벤치마크 오케스트레이터 — 2-Phase 병렬 실행.

Phase 1: 카테고리별 데이터 준비 (4개 병렬)
  → HF 데이터 + 청킹 + Contextual enrichment
Phase 2: (카테고리 × 전략 조합)별 벤치마크 (최대 24개 병렬)
  → 인덱스 빌드 + Pass1(레이턴시) + Pass2(RAGAS)
Phase 3: 결과 수집 + 병합 리포트

사용법:
  source .env  # HARBOR_REGISTRY 로드

  # 전체 실행
  python k8s/orchestrator.py --image $HARBOR_REGISTRY/rag-bench-test/worker:latest

  # 특정 카테고리만
  python k8s/orchestrator.py --image $HARBOR_REGISTRY/rag-bench-test/worker:latest --categories general,legal

  # dry-run
  python k8s/orchestrator.py --image $HARBOR_REGISTRY/rag-bench-test/worker:latest --dry-run

  # Phase 2만 (Phase 1 이미 완료)
  python k8s/orchestrator.py --image $HARBOR_REGISTRY/rag-bench-test/worker:latest --skip-prep --run-id 20260225-1430

  # 결과만 수집
  python k8s/orchestrator.py --image $HARBOR_REGISTRY/rag-bench-test/worker:latest --collect-only --run-id 20260225-1430
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

NAMESPACE = "rag-bench-test"
MANIFESTS_DIR = Path(__file__).parent / "manifests"
PREP_TEMPLATE = MANIFESTS_DIR / "prep-job-template.yaml"
BENCH_TEMPLATE = MANIFESTS_DIR / "bench-job-template.yaml"

DEFAULT_CATEGORIES = ["general", "legal", "business", "medical"]
POLL_INTERVAL_S = 30

# service 프리셋 기준 전략 조합
SERVICE_COMBOS = [
    {
        "dense": "kosimcse",
        "sparse": "korean_bm25",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
    {
        "dense": "kosimcse",
        "sparse": "splade",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
    {
        "dense": "e5",
        "sparse": "korean_bm25",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
    {
        "dense": "e5",
        "sparse": "splade",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
    {
        "dense": "bge-m3",
        "sparse": "korean_bm25",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
    {
        "dense": "bge-m3",
        "sparse": "splade",
        "reranker": "colbert",
        "llm_support": "contextual",
    },
]

FULL_COMBOS = []
for d in ["kosimcse", "e5", "bge-m3", "openai-large", "upstage"]:
    for s in ["korean_bm25", "splade"]:
        for r in ["colbert", "flashrank"]:
            FULL_COMBOS.append(
                {"dense": d, "sparse": s, "reranker": r, "llm_support": "contextual"}
            )

PRESET_COMBOS = {
    "service": SERVICE_COMBOS,
    "full": FULL_COMBOS,
}


# ---------------------------------------------------------------------------
# kubectl 래퍼
# ---------------------------------------------------------------------------


def kubectl(*args: str, capture: bool = True, check: bool = True) -> str:
    cmd = ["kubectl"] + list(args)
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else ""


def kubectl_apply(path: str) -> str:
    return kubectl("apply", "-f", path)


def _safe_label(combo: Dict) -> str:
    """K8s Job 이름에 사용 가능한 조합 라벨."""
    raw = f"{combo['dense']}-{combo['sparse']}-{combo['reranker']}"
    return raw.replace("_", "").replace("/", "-").lower()


# ---------------------------------------------------------------------------
# Job 렌더링
# ---------------------------------------------------------------------------


def _render_template(template_path: Path, replacements: Dict[str, str]) -> str:
    content = template_path.read_text(encoding="utf-8")
    for k, v in replacements.items():
        content = content.replace(k, v)
    return content


def _apply_yaml(yaml_content: str, dry_run: bool) -> Optional[str]:
    """YAML을 적용하거나 dry-run 출력."""
    if dry_run:
        print(yaml_content)
        print("---")
        return None
    tmp = f"/tmp/k8s-bench-{os.getpid()}-{time.time_ns()}.yaml"
    Path(tmp).write_text(yaml_content, encoding="utf-8")
    kubectl_apply(tmp)
    return tmp


# ---------------------------------------------------------------------------
# Phase 1: Prep Jobs
# ---------------------------------------------------------------------------


def create_prep_jobs(
    categories: List[str],
    run_id: str,
    image: str,
    args,
    dry_run: bool = False,
) -> Dict[str, str]:
    """카테고리별 prep Job 생성. {category: job_name}"""
    jobs = {}
    for cat in categories:
        job_name = f"prep-{cat}-{run_id}"
        yaml = _render_template(
            PREP_TEMPLATE,
            {
                "${CATEGORY}": cat,
                "${RUN_ID}": run_id,
                "${IMAGE}": image,
                "${MAX_CORPUS}": str(args.max_corpus),
                "${MAX_QUERIES}": str(args.max_queries),
                "${CONTEXTUAL_LLM}": args.contextual_llm,
                "${CPU_REQUEST}": args.prep_cpu_request,
                "${CPU_LIMIT}": args.prep_cpu_limit,
                "${MEMORY_REQUEST}": args.prep_memory_request,
                "${MEMORY_LIMIT}": args.prep_memory_limit,
            },
        )
        _apply_yaml(yaml, dry_run)
        if not dry_run:
            print(f"    prep Job 생성: {job_name}")
        jobs[cat] = job_name
    return jobs


# ---------------------------------------------------------------------------
# Phase 2: Bench Jobs
# ---------------------------------------------------------------------------


def create_bench_jobs(
    categories: List[str],
    combos: List[Dict],
    run_id: str,
    image: str,
    args,
    dry_run: bool = False,
) -> Dict[str, str]:
    """(카테고리 × 조합)별 bench Job 생성. {key: job_name}"""
    jobs = {}
    for cat in categories:
        for combo in combos:
            label = _safe_label(combo)
            job_name = f"bench-{cat}-{label}-{run_id}"
            # K8s Job 이름은 63자 제한
            if len(job_name) > 63:
                job_name = job_name[:63].rstrip("-")

            yaml = _render_template(
                BENCH_TEMPLATE,
                {
                    "${CATEGORY}": cat,
                    "${RUN_ID}": run_id,
                    "${IMAGE}": image,
                    "${COMBO_DENSE}": combo["dense"],
                    "${COMBO_SPARSE}": combo["sparse"],
                    "${COMBO_RERANKER}": combo["reranker"],
                    "${COMBO_LLM_SUPPORT}": combo["llm_support"],
                    "${COMBO_LABEL}": label,
                    "${CPU_REQUEST}": args.bench_cpu_request,
                    "${CPU_LIMIT}": args.bench_cpu_limit,
                    "${MEMORY_REQUEST}": args.bench_memory_request,
                    "${MEMORY_LIMIT}": args.bench_memory_limit,
                },
            )
            _apply_yaml(yaml, dry_run)
            key = f"{cat}/{combo['dense']}+{combo['sparse']}+{combo['reranker']}"
            if not dry_run:
                print(f"    bench Job 생성: {job_name}")
            jobs[key] = job_name
    return jobs


# ---------------------------------------------------------------------------
# 모니터링
# ---------------------------------------------------------------------------


def get_job_status(job_name: str) -> Dict:
    try:
        output = kubectl("get", "job", job_name, "-n", NAMESPACE, "-o", "json")
        status = json.loads(output).get("status", {})
        return {
            "succeeded": status.get("succeeded", 0),
            "failed": status.get("failed", 0),
            "active": status.get("active", 0),
        }
    except subprocess.CalledProcessError:
        return {"succeeded": 0, "failed": 0, "active": 0}


def get_job_logs(job_name: str, tail: int = 30) -> str:
    try:
        return kubectl(
            "logs",
            f"job/{job_name}",
            "-n",
            NAMESPACE,
            "-c",
            "worker",
            f"--tail={tail}",
            check=False,
        )
    except Exception:
        return "(로그 조회 불가)"


def wait_for_jobs(
    jobs: Dict[str, str],
    phase_name: str,
    timeout_s: int,
) -> Dict[str, str]:
    """{key: "succeeded"|"failed"|"timeout"}"""
    results = {}
    pending = dict(jobs)
    start = time.time()
    total = len(jobs)

    print(f"\n  [{phase_name}] {total}개 Job 대기 (timeout: {timeout_s}s)")

    while pending and (time.time() - start) < timeout_s:
        done_keys = []
        for key, job_name in pending.items():
            st = get_job_status(job_name)
            if st["succeeded"] > 0:
                results[key] = "succeeded"
                done_keys.append(key)
            elif st["failed"] > 0:
                results[key] = "failed"
                done_keys.append(key)

        for k in done_keys:
            status = results[k]
            elapsed = time.time() - start
            icon = "OK" if status == "succeeded" else "FAIL"
            print(f"    [{icon}] {k} ({elapsed:.0f}s)")
            if status == "failed":
                print(f"         {get_job_logs(pending[k], tail=10)}")
            del pending[k]

        if pending:
            elapsed = time.time() - start
            done = total - len(pending)
            print(f"    [{elapsed:.0f}s] {done}/{total} 완료, {len(pending)} 진행 중")
            time.sleep(POLL_INTERVAL_S)

    for key in pending:
        results[key] = "timeout"
        print(f"    [TIMEOUT] {key}")

    return results


# ---------------------------------------------------------------------------
# 결과 수집
# ---------------------------------------------------------------------------


def collect_results(
    categories: List[str],
    run_id: str,
    local_output: Path,
) -> Dict[str, Path]:
    """PVC에서 결과를 로컬로 복사."""
    local_output.mkdir(parents=True, exist_ok=True)
    collected = {}

    collector_pod = f"collector-{run_id}"
    collector_yaml = f"""\
apiVersion: v1
kind: Pod
metadata:
  name: {collector_pod}
  namespace: {NAMESPACE}
spec:
  restartPolicy: Never
  containers:
    - name: collector
      image: busybox:1.36
      command: ["sleep", "600"]
      volumeMounts:
        - name: results
          mountPath: /results
  volumes:
    - name: results
      persistentVolumeClaim:
        claimName: bench-results
"""
    tmp = f"/tmp/collector-{run_id}.yaml"
    Path(tmp).write_text(collector_yaml, encoding="utf-8")

    try:
        kubectl_apply(tmp)
        print(f"\n  수집 Pod 생성: {collector_pod}")
        kubectl(
            "wait",
            "--for=condition=Ready",
            f"pod/{collector_pod}",
            "-n",
            NAMESPACE,
            "--timeout=60s",
        )

        for cat in categories:
            local_cat = local_output / cat
            local_cat.mkdir(parents=True, exist_ok=True)
            try:
                kubectl(
                    "cp",
                    f"{NAMESPACE}/{collector_pod}:/results/{cat}/",
                    str(local_cat),
                    "-c",
                    "collector",
                )

                # DONE 파일이 있는 조합만 유효 (원자적 쓰기 보장)
                found = list(local_cat.rglob("DONE"))
                # prepared/DONE은 Phase 1 결과이므로 제외
                combo_done = [f for f in found if "prepared" not in str(f)]
                if combo_done:
                    print(f"    [{cat}] {len(combo_done)}개 조합 결과 수집")
                    collected[cat] = local_cat
                else:
                    print(f"    [{cat}] 완료된 조합 없음")
            except Exception as e:
                print(f"    [{cat}] 수집 실패: {e}")
    finally:
        kubectl(
            "delete",
            "pod",
            collector_pod,
            "-n",
            NAMESPACE,
            "--ignore-not-found",
            check=False,
        )

    return collected


# ---------------------------------------------------------------------------
# 병합
# ---------------------------------------------------------------------------


def merge_results(
    local_output: Path,
    categories: List[str],
    expected_combos: int,
    bench_results: Dict[str, str],
) -> None:
    """수집된 (카테고리/조합) 결과를 merge_service_results 형식으로 재구성 후 병합."""
    print(f"\n  결과 재구성 + 병합...")

    # 부분 실패 감지
    for cat in categories:
        cat_dir = local_output / cat
        if not cat_dir.exists():
            print(f"    WARN [{cat}] 결과 디렉토리 없음")
            continue

        # DONE 파일이 있는 조합만 유효한 결과
        combo_dirs = [
            d
            for d in cat_dir.iterdir()
            if d.is_dir() and d.name != "prepared" and (d / "DONE").exists()
        ]

        combo_results = []
        for combo_dir in combo_dirs:
            result_file = combo_dir / "result.json"
            if result_file.exists():
                data = json.loads(result_file.read_text(encoding="utf-8"))
                combo_results.append(data)

        found = len(combo_results)
        if found < expected_combos:
            # 실패한 조합 식별
            failed_keys = [
                k
                for k, v in bench_results.items()
                if k.startswith(f"{cat}/") and v != "succeeded"
            ]
            print(
                f"    WARN [{cat}] {found}/{expected_combos} 조합만 성공"
                f" — 누락: {[k.split('/', 1)[1] for k in failed_keys]}"
            )

        if not combo_results:
            print(f"    [{cat}] 유효한 결과 없음 — 건너뜀")
            continue

        # 카테고리 통합 result.json 생성
        merged = {
            "category": cat,
            "n_qa": combo_results[0].get("n_qa", 0),
            "n_combos_expected": expected_combos,
            "n_combos_succeeded": found,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ragas": [],
        }
        for cr in combo_results:
            if "ragas" in cr:
                merged["ragas"].extend(cr["ragas"])

        (cat_dir / "result.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"    [{cat}] {found}/{expected_combos} 조합 → 통합 result.json")

    # 기존 merge 스크립트 실행
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_bench.scripts.merge_service_results",
                "--run_dirs",
                str(local_output),
                "--output",
                str(local_output / "merged_report.html"),
            ],
            check=True,
        )
        print(f"  병합 리포트: {local_output / 'merged_report.html'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  병합 스크립트 실행 실패. 수동 실행:")
        print(
            f"    python -m rag_bench.scripts.merge_service_results --run_dirs {local_output}"
        )


# ---------------------------------------------------------------------------
# 인프라 셋업
# ---------------------------------------------------------------------------


def setup_infrastructure() -> None:
    print("  인프라 셋업...")
    for f in ["namespace.yaml", "results-pvc.yaml", "model-cache-pvc.yaml"]:
        path = MANIFESTS_DIR / f
        if path.exists():
            kubectl_apply(str(path))
            print(f"    적용: {f}")

    # API Key Secret (OPENAI + UPSTAGE)
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    upstage_key = os.environ.get("UPSTAGE_API_KEY", "")

    try:
        kubectl("get", "secret", "bench-secrets", "-n", NAMESPACE)
        # 기존 Secret 존재 → Upstage 키 추가 (있으면)
        if upstage_key:
            import base64

            encoded = base64.b64encode(upstage_key.encode()).decode()
            kubectl(
                "patch",
                "secret",
                "bench-secrets",
                "-n",
                NAMESPACE,
                "-p",
                json.dumps({"data": {"UPSTAGE_API_KEY": encoded}}),
            )
            print("    패치: bench-secrets에 UPSTAGE_API_KEY 추가")
    except subprocess.CalledProcessError:
        if not openai_key:
            print("    ERROR: OPENAI_API_KEY 환경변수 필수")
            sys.exit(1)
        literals = [f"--from-literal=OPENAI_API_KEY={openai_key}"]
        if upstage_key:
            literals.append(f"--from-literal=UPSTAGE_API_KEY={upstage_key}")
        kubectl(
            "create",
            "secret",
            "generic",
            "bench-secrets",
            "-n",
            NAMESPACE,
            *literals,
        )
        print("    생성: bench-secrets Secret")


def _verify_prep_data(categories: List[str], run_id: str) -> None:
    """Phase 1 결과가 PVC에서 읽을 수 있는지 검증 Pod로 확인."""
    verifier_pod = f"prep-verify-{run_id}"
    # 모든 카테고리의 DONE 파일 존재 여부를 한 번에 확인
    checks = " && ".join(f"test -f /results/{cat}/prepared/DONE" for cat in categories)
    verifier_yaml = f"""\
apiVersion: v1
kind: Pod
metadata:
  name: {verifier_pod}
  namespace: {NAMESPACE}
spec:
  restartPolicy: Never
  containers:
    - name: verify
      image: busybox:1.36
      command: ["sh", "-c", "sleep 5 && {checks} && echo VERIFIED"]
      volumeMounts:
        - name: results
          mountPath: /results
  volumes:
    - name: results
      persistentVolumeClaim:
        claimName: bench-results
"""
    tmp = f"/tmp/prep-verify-{run_id}.yaml"
    Path(tmp).write_text(verifier_yaml, encoding="utf-8")

    print("\n  Phase 1 데이터 가시성 검증 중...")
    try:
        kubectl_apply(tmp)
        # Pod 완료 대기 (최대 120초)
        kubectl(
            "wait",
            "--for=condition=Ready",
            f"pod/{verifier_pod}",
            "-n",
            NAMESPACE,
            "--timeout=30s",
        )
        # 실행 결과 확인
        for attempt in range(12):
            time.sleep(10)
            try:
                status = json.loads(
                    kubectl(
                        "get",
                        "pod",
                        verifier_pod,
                        "-n",
                        NAMESPACE,
                        "-o",
                        "jsonpath={.status.phase}",
                    )
                )
            except (json.JSONDecodeError, subprocess.CalledProcessError):
                status = kubectl(
                    "get",
                    "pod",
                    verifier_pod,
                    "-n",
                    NAMESPACE,
                    "-o",
                    "jsonpath={.status.phase}",
                )
            if status == "Succeeded":
                print("    PVC 가시성 확인 완료")
                break
            if status == "Failed":
                logs = kubectl("logs", verifier_pod, "-n", NAMESPACE, check=False)
                print(f"    PVC 검증 실패 — Phase 1 데이터 미발견\n    {logs}")
                sys.exit(1)
        else:
            print("    WARN: PVC 검증 타임아웃 — 계속 진행")
    except subprocess.CalledProcessError as e:
        print(f"    WARN: 검증 Pod 생성 실패 ({e}) — 계속 진행")
    finally:
        kubectl(
            "delete",
            "pod",
            verifier_pod,
            "-n",
            NAMESPACE,
            "--ignore-not-found",
            check=False,
        )


def cleanup_run(run_id: str) -> None:
    print(f"\n  Run {run_id} 정리...")
    kubectl(
        "delete",
        "jobs",
        "-n",
        NAMESPACE,
        "-l",
        f"run-id={run_id}",
        "--ignore-not-found",
        check=False,
    )
    print("  완료")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K8s 벤치마크 오케스트레이터 — 2-Phase 병렬 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--image", required=True, help="워커 컨테이너 이미지")
    p.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="카테고리 (쉼표 구분)",
    )
    p.add_argument(
        "--preset",
        default="service",
        choices=["service", "full"],
        help="전략 프리셋 (기본: service)",
    )
    p.add_argument(
        "--dense",
        default=None,
        help="Dense 모델 필터 (쉼표 구분, 예: openai-large,upstage)",
    )
    p.add_argument(
        "--rerankers",
        default=None,
        help="리랭커 필터 (쉼표 구분, 예: colbert)",
    )
    p.add_argument("--run-id", default=None, help="실행 ID (기본: 타임스탬프)")
    p.add_argument("--output", default=None, help="로컬 결과 경로")
    p.add_argument(
        "--contextual-llm",
        default="gpt-4o-mini",
        help="Contextual Retrieval LLM (기본: gpt-4o-mini)",
    )
    p.add_argument("--max-corpus", type=int, default=10_000)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--timeout", type=int, default=7200, help="Phase별 타임아웃 (초)")

    # Phase 1 리소스
    p.add_argument("--prep-cpu-request", default="1")
    p.add_argument("--prep-cpu-limit", default="1")
    p.add_argument("--prep-memory-request", default="4Gi")
    p.add_argument("--prep-memory-limit", default="8Gi")

    # Phase 2 리소스 (Dense 모델 로딩 + ColBERT + BM25 → CPU 2, 메모리 8Gi 필요)
    p.add_argument("--bench-cpu-request", default="1")
    p.add_argument("--bench-cpu-limit", default="2")
    p.add_argument("--bench-memory-request", default="4Gi")
    p.add_argument("--bench-memory-limit", default="8Gi")

    # 모드
    p.add_argument("--dry-run", action="store_true", help="YAML만 출력")
    p.add_argument("--skip-prep", action="store_true", help="Phase 1 건너뜀")
    p.add_argument("--collect-only", action="store_true", help="결과만 수집")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="완료 후 정리")
    p.add_argument("--skip-setup", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    combos = PRESET_COMBOS[args.preset]
    if args.dense:
        dense_filter = [d.strip() for d in args.dense.split(",")]
        combos = [c for c in combos if c["dense"] in dense_filter]
    if args.rerankers:
        reranker_filter = [r.strip() for r in args.rerankers.split(",")]
        combos = [c for c in combos if c["reranker"] in reranker_filter]
    if not combos:
        print("  ERROR: 필터 조건에 매칭되는 조합 없음")
        sys.exit(1)
    local_output = Path(args.output) if args.output else Path(f"./k8s_results/{run_id}")
    total_jobs = len(categories) * len(combos)

    print(f"\n{'=' * 60}")
    print(" K8s Benchmark Orchestrator — 2-Phase")
    print(f"{'=' * 60}")
    print(f"  Run ID       : {run_id}")
    print(f"  Image        : {args.image}")
    print(f"  Categories   : {categories}")
    print(f"  Preset       : {args.preset} ({len(combos)} combos)")
    print(
        f"  Total Jobs   : Phase1={len(categories)} + Phase2={total_jobs} = {len(categories) + total_jobs}"
    )
    print(f"  Output       : {local_output}")
    print()

    # ── collect-only ──────────────────────────────────────────
    if args.collect_only:
        collected = collect_results(categories, run_id, local_output)
        if collected and not args.no_merge:
            merge_results(local_output, categories, len(combos), {})
        return

    # ── dry-run ───────────────────────────────────────────────
    if args.dry_run:
        print("\n--- Phase 1: Prep Jobs ---\n")
        create_prep_jobs(categories, run_id, args.image, args, dry_run=True)
        print("\n--- Phase 2: Bench Jobs ---\n")
        create_bench_jobs(categories, combos, run_id, args.image, args, dry_run=True)
        return

    # ── 인프라 ────────────────────────────────────────────────
    if not args.skip_setup:
        setup_infrastructure()

    # ── Phase 1: Prep ─────────────────────────────────────────
    if not args.skip_prep:
        print(f"\n{'─' * 60}")
        print(f"  Phase 1: Prep ({len(categories)} Jobs)")
        print(f"{'─' * 60}")

        prep_jobs = create_prep_jobs(categories, run_id, args.image, args)
        prep_results = wait_for_jobs(prep_jobs, "Phase 1", args.timeout)

        failed_cats = [k for k, v in prep_results.items() if v != "succeeded"]
        if failed_cats:
            print(f"\n  Phase 1 실패: {failed_cats}")
            # 실패한 카테고리는 Phase 2에서 제외
            categories = [c for c in categories if c not in failed_cats]
            if not categories:
                print("  모든 prep 실패. 종료.")
                sys.exit(1)

    # ── Phase 1→2 PVC 동기화 ─────────────────────────────────
    # ReadWriteMany PVC(NFS/CephFS)는 쓰기 후 다른 노드에서 즉시 보이지 않을 수 있음.
    # Phase 1 완료 후 검증 Pod로 데이터 가시성을 확인한다.
    if not args.skip_prep:
        _verify_prep_data(categories, run_id)

    # ── Phase 2: Bench ────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(
        f"  Phase 2: Bench ({len(categories)} × {len(combos)} = {len(categories) * len(combos)} Jobs)"
    )
    print(f"{'─' * 60}")

    bench_jobs = create_bench_jobs(categories, combos, run_id, args.image, args)
    bench_results = wait_for_jobs(bench_jobs, "Phase 2", args.timeout)

    # ── 결과 수집 + 병합 ─────────────────────────────────────
    succeeded = [k.split("/")[0] for k, v in bench_results.items() if v == "succeeded"]
    succeeded_cats = sorted(set(succeeded))

    if succeeded_cats:
        collected = collect_results(succeeded_cats, run_id, local_output)
        if collected and not args.no_merge:
            merge_results(local_output, succeeded_cats, len(combos), bench_results)

    # ── 정리 ──────────────────────────────────────────────────
    if args.cleanup:
        cleanup_run(run_id)

    # ── 요약 ──────────────────────────────────────────────────
    n_ok = sum(1 for v in bench_results.values() if v == "succeeded")
    n_fail = sum(1 for v in bench_results.values() if v != "succeeded")

    print(f"\n{'=' * 60}")
    print(f" 완료")
    print(f"{'=' * 60}")
    print(f"  Run ID   : {run_id}")
    print(f"  성공     : {n_ok}/{len(bench_results)} Jobs")
    if n_fail:
        print(f"  실패     : {n_fail} Jobs")
        for k, v in bench_results.items():
            if v != "succeeded":
                print(f"    [{v}] {k}")
    print(f"  결과     : {local_output}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
