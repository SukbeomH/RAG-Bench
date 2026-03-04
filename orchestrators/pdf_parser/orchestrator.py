"""
PDF Parser K8s 벤치마크 오케스트레이터.

백엔드 × PDF 조합별 K8s Job을 rag-bench-test 네임스페이스에 병렬 생성하고,
완료 후 결과를 PVC에서 수집하여 요약 리포트를 출력한다.

사용법:
  # 환경변수 설정
  source .env
  export HARBOR_REGISTRY=...

  # 전체 (기존 3 백엔드 × 11 PDF)
  python k8s/pdf_parser_orchestrator.py \\
      --image $HARBOR_REGISTRY/rag-bench-test/pdf-parser:latest \\
      --preset phase1

  # VLM 비교 (openai + upstage × 11 PDF)
  python k8s/pdf_parser_orchestrator.py \\
      --image $HARBOR_REGISTRY/rag-bench-test/pdf-parser:latest \\
      --preset vlm

  # 단일 조합
  python k8s/pdf_parser_orchestrator.py \\
      --image $HARBOR_REGISTRY/rag-bench-test/pdf-parser:latest \\
      --backend upstage --pdf text_only.pdf

  # dry-run
  python k8s/pdf_parser_orchestrator.py \\
      --image $HARBOR_REGISTRY/rag-bench-test/pdf-parser:latest \\
      --preset quick --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 상수 ─────────────────────────────────────────────────────────────────────

NAMESPACE = "rag-bench-test"
MANIFESTS_DIR = Path(__file__).parent / "manifests"
JOB_TEMPLATE = MANIFESTS_DIR / "pdf-parser-job-template.yaml"

POLL_INTERVAL_S = 20  # Job 상태 폴링 간격 (초)
MAX_PARALLEL = 10  # 동시 실행 Job 상도

from autorag_pdf_eval.spec import BenchSpec, get_preset, GT_MAP


# ── kubectl 래퍼 ──────────────────────────────────────────────────────────────


def kubectl(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "-n", NAMESPACE] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def job_exists(job_name: str) -> bool:
    r = kubectl(["get", "job", job_name], check=False)
    return r.returncode == 0


def apply_manifest(yaml_text: str, dry_run: bool = False) -> None:
    cmd = ["kubectl", "apply", "-f", "-"]
    if dry_run:
        cmd += ["--dry-run=client"]
    subprocess.run(cmd, input=yaml_text, text=True, check=True)


def delete_job(job_name: str) -> None:
    kubectl(["delete", "job", job_name, "--ignore-not-found"], check=False)


# ── 매니페스트 렌더링 ─────────────────────────────────────────────────────────


def _k8s_safe(s: str) -> str:
    """K8s 이름 안전 문자열: 소문자, 숫자, 하이픈만, 최대 63자."""
    import re

    s = s.lower().replace("_", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s[:63]


def render_job(
    spec: BenchSpec,
    image: str,
    run_id: str,
) -> tuple[str, str]:
    """
    Job 매니페스트 YAML 문자열과 Job 이름 반환.
    """
    pdf_stem = spec.pdf_name.replace(".pdf", "")
    raw_name = f"pdf-{spec.backend}-{pdf_stem}-{spec.mode}"
    job_name = _k8s_safe(raw_name) + f"-{run_id}"
    job_name = job_name[:62]  # run_id 포함 63자 초과 방지

    template = JOB_TEMPLATE.read_text(encoding="utf-8")
    yaml_text = (
        template.replace("${JOB_NAME}", _k8s_safe(raw_name))
        .replace("${RUN_ID}", run_id)
        .replace("${BACKEND}", spec.backend)
        .replace("${PDF_FILE}", spec.pdf_name)
        .replace("${PARSE_MODE}", spec.mode)
        .replace("${IMAGE}", image)
    )
    return yaml_text, job_name


# ── Job 상태 조회 ─────────────────────────────────────────────────────────────


def get_job_status(job_name: str) -> str:
    """'running' | 'succeeded' | 'failed' | 'unknown'"""
    r = kubectl(["get", "job", job_name, "-o", "json"], check=False)
    if r.returncode != 0:
        return "unknown"
    info = json.loads(r.stdout)
    conds = info.get("status", {}).get("conditions", [])
    for c in conds:
        if c["type"] == "Complete" and c["status"] == "True":
            return "succeeded"
        if c["type"] == "Failed" and c["status"] == "True":
            return "failed"
    return "running"


# ── 결과 수집 ─────────────────────────────────────────────────────────────────


def collect_results(run_id: str, results_base: Path) -> list[dict]:
    """
    PVC 마운트 경로(results_base)에서 metrics.json을 수집.

    실제 K8s 환경에서는 kubectl cp 또는 PVC 직접 마운트 필요.
    로컬 테스트: --results-dir로 로컬 경로 지정.
    """
    run_dir = results_base / run_id
    if not run_dir.exists():
        print(f"[WARN] 결과 디렉토리 없음: {run_dir}")
        return []

    results = []
    for metrics_file in run_dir.rglob("metrics.json"):
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            results.append(data)
        except Exception as e:
            print(f"[WARN] {metrics_file} 파싱 오류: {e}")
    return results


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 75}")
    print("결과 요약")
    print(f"{'=' * 75}")
    print(f"{'백엔드':<12} {'PDF':<35} {'NED':>6} {'TEDS':>6} {'속도':>8} {'단어':>6}")
    print("-" * 75)
    for r in results:
        s = r.get("summary", {})
        ned = (
            f"{s['avg_text_ned']:.3f}" if s.get("avg_text_ned") is not None else "  N/A"
        )
        teds = (
            f"{s['avg_table_teds']:.3f}"
            if s.get("avg_table_teds") is not None
            else "  N/A"
        )
        spd = (
            f"{s['avg_speed_s']:.2f}s" if s.get("avg_speed_s") is not None else "  N/A"
        )
        w = str(s.get("total_words", 0))
        err = r.get("error") or ""
        tag = f"  ERROR: {err[:30]}" if err else ""
        print(
            f"{r['backend']:<12} {r['pdf_name']:<35} {ned:>6} {teds:>6} {spd:>8} {w:>6}{tag}"
        )
    print(f"{'=' * 75}\n")


# ── 메인 루프 ─────────────────────────────────────────────────────────────────


def wait_for_job(job_name: str) -> str:
    """Job이 완료(succeeded/failed)될 때까지 블로킹. 최종 상태 반환.

    Job이 TTL로 삭제된 경우(unknown 3회 연속) succeeded로 간주.
    """
    unknown_streak = 0
    while True:
        status = get_job_status(job_name)
        if status in ("succeeded", "failed"):
            return status
        if status == "unknown":
            unknown_streak += 1
            if unknown_streak >= 3:
                print(f"  [WARN] {job_name} 미존재 (TTL 삭제 추정) → succeeded 처리")
                return "succeeded"
        else:
            unknown_streak = 0
        time.sleep(POLL_INTERVAL_S)


def run_jobs(
    specs: list[BenchSpec],
    image: str,
    run_id: str,
    dry_run: bool = False,
    results_dir: Optional[Path] = None,
    max_parallel: int = MAX_PARALLEL,
) -> None:
    total = len(specs)
    print(f"\n{'=' * 75}")
    print("PDF Parser K8s 벤치마크 시작")
    print(f"  조합 수  : {total}")
    print(f"  Run ID   : {run_id}")
    print(f"  네임스페이스: {NAMESPACE}")
    print(f"  이미지   : {image}")
    print(f"  동시 실행 : {max_parallel}")
    if dry_run:
        print("  [DRY-RUN 모드]")
    print(f"{'=' * 75}\n")

    job_names: list[str] = []
    succeeded, failed = [], []

    if max_parallel == 1:
        # 순차 실행: 하나 완료 후 다음 생성
        for idx, spec in enumerate(specs, 1):
            yaml_text, job_name = render_job(spec, image, run_id)

            if job_exists(job_name):
                print(f"[{idx:>3}/{total}] SKIP (이미 존재) — {job_name}")
                job_names.append(job_name)
                # 이미 존재하는 Job은 완료 대기
                if not dry_run:
                    status = wait_for_job(job_name)
                    (succeeded if status == "succeeded" else failed).append(job_name)
                    print(
                        f"  [{('OK' if status == 'succeeded' else 'FAIL')}] {job_name}"
                    )
                continue

            print(f"[{idx:>3}/{total}] CREATE — {job_name}")
            try:
                apply_manifest(yaml_text, dry_run=dry_run)
                job_names.append(job_name)
            except subprocess.CalledProcessError as e:
                print(f"  오류: {(e.stderr or e.stdout or '')[:200]}", file=sys.stderr)
                continue

            if not dry_run:
                print(f"  대기 중... ({job_name})")
                status = wait_for_job(job_name)
                (succeeded if status == "succeeded" else failed).append(job_name)
                print(
                    f"  [{('OK' if status == 'succeeded' else 'FAIL')}] {job_name}  "
                    f"(성공 {len(succeeded)}, 실패 {len(failed)} / {idx})"
                )
    else:
        # 병렬 실행: 전체 Job 일괄 생성 후 완료 대기
        for idx, spec in enumerate(specs, 1):
            yaml_text, job_name = render_job(spec, image, run_id)

            if job_exists(job_name):
                print(f"[{idx:>3}/{total}] SKIP (이미 존재) — {job_name}")
                job_names.append(job_name)
                continue

            print(f"[{idx:>3}/{total}] CREATE — {job_name}")
            try:
                apply_manifest(yaml_text, dry_run=dry_run)
                job_names.append(job_name)
            except subprocess.CalledProcessError as e:
                print(f"  오류: {e.stderr[:200]}", file=sys.stderr)

            if idx % max_parallel == 0 and not dry_run:
                print(f"  ({idx}개 생성 완료, 잠시 대기...)")
                time.sleep(2)

        if dry_run:
            print(
                f"\n[DRY-RUN] {len(job_names)}개 Job 매니페스트 생성 완료 (실제 적용 안됨)"
            )
            return

        # Job 완료 대기
        print(f"\n총 {len(job_names)}개 Job 모니터링 중... (폴링 {POLL_INTERVAL_S}s)\n")
        pending = set(job_names)

        while pending:
            still_running = set()
            for job_name in pending:
                status = get_job_status(job_name)
                if status == "succeeded":
                    succeeded.append(job_name)
                    print(f"  [OK]    {job_name}")
                elif status == "failed":
                    failed.append(job_name)
                    print(f"  [FAIL]  {job_name}")
                else:
                    still_running.add(job_name)

            pending = still_running
            if pending:
                print(
                    f"  대기 중 {len(pending)}개 / 완료 {len(succeeded) + len(failed)}개 "
                    f"(성공 {len(succeeded)}, 실패 {len(failed)})"
                )
                time.sleep(POLL_INTERVAL_S)

    if dry_run:
        print(
            f"\n[DRY-RUN] {len(job_names)}개 Job 매니페스트 생성 완료 (실제 적용 안됨)"
        )
        return

    print(f"\n완료: 성공 {len(succeeded)} / 실패 {len(failed)} / 전체 {len(job_names)}")

    # 결과 수집 (로컬 PVC 마운트 경로 지정 시)
    if results_dir:
        results = collect_results(run_id, results_dir)
        if results:
            print_summary(results)
        else:
            print("[INFO] 결과 파일을 찾을 수 없습니다. PVC를 직접 확인하세요.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF Parser K8s 벤치마크 오케스트레이터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="컨테이너 이미지 URI")
    parser.add_argument(
        "--preset",
        default=None,
        choices=[
            "quick",
            "phase1",
            "phase2",
            "vlm",
            "ocr",
            "vlm-all",
            "tables",
            "graphs",
        ],
        help="사전 정의된 조합 집합",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=[
            "pymupdf",
            "docling",
            "openai",
            "openai-4.1",
            "upstage",
            "upstage-enhanced",
            "granite-vision",
            "got-ocr2",
            "paddleocr-vl",
            "mineru",
        ],
        help="단일 백엔드 (--pdf와 함께)",
    )
    parser.add_argument(
        "--pdf", default=None, help="단일 PDF 파일명 (--backend와 함께)"
    )
    parser.add_argument(
        "--mode", default="direct", choices=["direct", "document", "hybrid"]
    )
    parser.add_argument("--run-id", default=None, help="실행 ID (기본: 타임스탬프)")
    parser.add_argument(
        "--dry-run", action="store_true", help="실제 Job 생성 없이 매니페스트만 검증"
    )
    parser.add_argument(
        "--results-dir", default=None, help="PVC 결과 디렉토리 로컬 경로 (결과 수집용)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL,
        help=f"동시 실행 Job 최대 수 (기본: {MAX_PARALLEL}, 1=순차 실행)",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="프리셋에서 실행할 백엔드 필터 (쉼표 구분, 예: granite-vision,got-ocr2)",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M")
    results_dir = Path(args.results_dir) if args.results_dir else None

    # 스펙 결정
    if args.preset:
        specs = get_preset(args.preset)
        if args.backends:
            filter_set = set(b.strip() for b in args.backends.split(","))
            specs = [s for s in specs if s.backend in filter_set]
            if not specs:
                parser.error(f"--backends 필터 결과 스펙 없음: {args.backends}")
    elif args.backend and args.pdf:
        specs = [
            BenchSpec(
                backend=args.backend,
                pdf_name=args.pdf,
                mode=args.mode,
                gt_name=GT_MAP.get(args.pdf),
            )
        ]
    else:
        parser.error("--preset 또는 --backend + --pdf 를 지정하세요.")

    run_jobs(
        specs,
        args.image,
        run_id,
        dry_run=args.dry_run,
        results_dir=results_dir,
        max_parallel=args.max_parallel,
    )


if __name__ == "__main__":
    main()
