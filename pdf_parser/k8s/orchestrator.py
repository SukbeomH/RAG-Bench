#!/usr/bin/env python3
"""
PDF Parser K8s Orchestrator

PDF 파일을 배치 단위로 분할하여 K8s Job으로 병렬 변환.

사용법:
    # 단일 배치 (PVC에 PDF 업로드 후)
    python pdf_parser/k8s/orchestrator.py \
        --image $IMAGE --batch-id batch01

    # 로컬 PDF 폴더 → PVC 업로드 → Job 실행
    python pdf_parser/k8s/orchestrator.py \
        --image $IMAGE --upload ./my_pdfs

    # 대량 PDF를 N개 배치로 분할 병렬 처리
    python pdf_parser/k8s/orchestrator.py \
        --image $IMAGE --upload ./my_pdfs --split 4

    # dry-run (YAML만 출력)
    python pdf_parser/k8s/orchestrator.py \
        --image $IMAGE --batch-id test --dry-run
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

NAMESPACE = "pdf-parser"
MANIFESTS_DIR = Path(__file__).parent / "manifests"
JOB_TEMPLATE = MANIFESTS_DIR / "job-template.yaml"

POLL_INTERVAL_S = 15


# ---------------------------------------------------------------------------
# kubectl 래퍼
# ---------------------------------------------------------------------------

def kubectl(*args: str, capture: bool = True, check: bool = True) -> str:
    cmd = ["kubectl"] + list(args)
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else ""


def _render_template(replacements: dict) -> str:
    content = JOB_TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, str(value))
    return content


def _apply_yaml(yaml_str: str, dry_run: bool = False):
    if dry_run:
        print(yaml_str)
        print("---")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_str)
        f.flush()
        kubectl("apply", "-f", f.name)
    Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PVC 업로드
# ---------------------------------------------------------------------------

def upload_to_pvc(local_dir: str, batch_id: str) -> int:
    """로컬 PDF 폴더를 PVC의 input/{batch_id}/에 업로드.

    busybox Pod를 임시 생성하여 kubectl cp로 복사.
    Returns: 업로드된 PDF 파일 수.
    """
    local_path = Path(local_dir)
    pdf_files = sorted(local_path.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: PDF 파일 없음: {local_dir}")
        sys.exit(1)

    print(f"\n  PVC 업로드: {len(pdf_files)} PDF → /data/input/{batch_id}/")

    # 임시 Pod 생성
    pod_name = f"upload-{batch_id}"
    pod_yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: {NAMESPACE}
spec:
  restartPolicy: Never
  containers:
    - name: uploader
      image: busybox
      command: ["sleep", "300"]
      volumeMounts:
        - name: pdf-storage
          mountPath: /data
  volumes:
    - name: pdf-storage
      persistentVolumeClaim:
        claimName: pdf-storage
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pod_yaml)
        f.flush()
        kubectl("apply", "-f", f.name)
    Path(f.name).unlink(missing_ok=True)

    # Pod Ready 대기
    print("    Pod 대기 중...")
    kubectl("wait", "pod", pod_name, "-n", NAMESPACE,
            "--for=condition=Ready", "--timeout=60s")

    # 디렉토리 생성
    kubectl("exec", pod_name, "-n", NAMESPACE, "--",
            "mkdir", "-p", f"/data/input/{batch_id}")

    # PDF 복사
    for pdf in pdf_files:
        kubectl("cp", str(pdf),
                f"{NAMESPACE}/{pod_name}:/data/input/{batch_id}/{pdf.name}")
        print(f"    ↑ {pdf.name}")

    # Pod 정리
    kubectl("delete", "pod", pod_name, "-n", NAMESPACE,
            "--ignore-not-found", check=False)

    return len(pdf_files)


# ---------------------------------------------------------------------------
# Job 생성
# ---------------------------------------------------------------------------

def create_job(batch_id: str, args, dry_run: bool = False) -> str:
    """PDF 파서 Job 생성."""
    job_name = f"pdf-parse-{batch_id}"

    yaml = _render_template({
        "${BATCH_ID}": batch_id,
        "${IMAGE}": args.image,
        "${PARSE_MODE}": args.mode,
        "${FILE_PATTERN}": args.file_pattern,
        "${CPU_REQUEST}": args.cpu_request,
        "${CPU_LIMIT}": args.cpu_limit,
        "${MEMORY_REQUEST}": args.memory_request,
        "${MEMORY_LIMIT}": args.memory_limit,
    })
    _apply_yaml(yaml, dry_run)

    if not dry_run:
        print(f"    Job 생성: {job_name}")
    return job_name


# ---------------------------------------------------------------------------
# 모니터링
# ---------------------------------------------------------------------------

def wait_for_jobs(job_names: list, timeout: int = 3600) -> dict:
    """Job 완료 대기. {job_name: status}."""
    pending = set(job_names)
    results = {}
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        for job_name in list(pending):
            try:
                status = kubectl(
                    "get", "job", job_name, "-n", NAMESPACE,
                    "-o", "jsonpath={.status.conditions[0].type}",
                    check=False,
                )
            except Exception:
                status = ""

            if status == "Complete":
                results[job_name] = "succeeded"
                pending.discard(job_name)
                print(f"    {job_name}: 완료")
            elif status == "Failed":
                results[job_name] = "failed"
                pending.discard(job_name)
                print(f"    {job_name}: 실패")

        if pending:
            time.sleep(POLL_INTERVAL_S)

    for job_name in pending:
        results[job_name] = "timeout"
        print(f"    {job_name}: 타임아웃")

    return results


# ---------------------------------------------------------------------------
# 결과 수집
# ---------------------------------------------------------------------------

def collect_results(batch_id: str, local_output: Path):
    """PVC에서 변환 결과를 로컬로 복사."""
    local_output.mkdir(parents=True, exist_ok=True)

    pod_name = f"collect-{batch_id}"
    pod_yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: {NAMESPACE}
spec:
  restartPolicy: Never
  containers:
    - name: collector
      image: busybox
      command: ["sleep", "120"]
      volumeMounts:
        - name: pdf-storage
          mountPath: /data
  volumes:
    - name: pdf-storage
      persistentVolumeClaim:
        claimName: pdf-storage
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pod_yaml)
        f.flush()
        kubectl("apply", "-f", f.name)
    Path(f.name).unlink(missing_ok=True)

    kubectl("wait", "pod", pod_name, "-n", NAMESPACE,
            "--for=condition=Ready", "--timeout=60s")

    # 결과 복사
    print(f"\n  결과 수집: /data/output/{batch_id}/ → {local_output}/")
    try:
        kubectl("cp",
                f"{NAMESPACE}/{pod_name}:/data/output/{batch_id}/.",
                str(local_output))
    except subprocess.CalledProcessError as e:
        print(f"    WARN: 수집 실패 — {e}")

    kubectl("delete", "pod", pod_name, "-n", NAMESPACE,
            "--ignore-not-found", check=False)

    # 결과 출력
    md_files = sorted(local_output.glob("*.md"))
    json_files = sorted(local_output.glob("*.json"))
    print(f"    Markdown: {len(md_files)}개, JSON: {len(json_files)}개")

    for jf in json_files:
        try:
            report = json.loads(jf.read_text())
            print(f"    {jf.name}: {report.get('success', 0)} 성공, "
                  f"{report.get('failed', 0)} 실패")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 인프라 설정
# ---------------------------------------------------------------------------

def setup_infrastructure():
    """네임스페이스 + PVC 생성."""
    print("\n  인프라 설정...")
    kubectl("apply", "-f", str(MANIFESTS_DIR / "namespace.yaml"))
    kubectl("apply", "-f", str(MANIFESTS_DIR / "pvc.yaml"))
    print("    네임스페이스 + PVC 생성 완료")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PDF Parser K8s Orchestrator"
    )
    p.add_argument("--image", required=True, help="워커 컨테이너 이미지")
    p.add_argument("--batch-id", help="배치 ID (자동 생성 가능)")
    p.add_argument("--upload", help="로컬 PDF 폴더 → PVC 업로드 후 실행")
    p.add_argument("--split", type=int, default=1,
                   help="PDF를 N개 배치로 분할 병렬 처리 (기본: 1)")
    p.add_argument("--mode", choices=["hybrid", "document"],
                   default="hybrid", help="파싱 모드 (기본: hybrid)")
    p.add_argument("--file-pattern", default="*.pdf")
    p.add_argument("--output", help="로컬 결과 출력 경로")
    p.add_argument("--timeout", type=int, default=3600)

    # 리소스
    p.add_argument("--cpu-request", default="1")
    p.add_argument("--cpu-limit", default="2")
    p.add_argument("--memory-request", default="4Gi")
    p.add_argument("--memory-limit", default="8Gi")

    # 모드
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--collect-only", action="store_true",
                   help="결과만 수집 (Job 실행 안 함)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    batch_id = args.batch_id or time.strftime("%Y%m%d-%H%M")
    local_output = Path(args.output) if args.output else Path(f"./parse_results/{batch_id}")

    print(f"\n{'=' * 60}")
    print(f" PDF Parser K8s Orchestrator")
    print(f"{'=' * 60}")
    print(f"  Batch ID   : {batch_id}")
    print(f"  Image      : {args.image}")
    print(f"  Mode       : {args.mode}")
    print(f"  Split      : {args.split}")
    print(f"  Output     : {local_output}")
    print()

    # ── collect-only ──
    if args.collect_only:
        collect_results(batch_id, local_output)
        return

    # ── dry-run ──
    if args.dry_run:
        print("--- Job Template ---\n")
        create_job(batch_id, args, dry_run=True)
        return

    # ── 인프라 ──
    if not args.skip_setup:
        setup_infrastructure()

    # ── 업로드 ──
    if args.upload:
        pdf_count = upload_to_pvc(args.upload, batch_id)
        print(f"    {pdf_count}개 PDF 업로드 완료")

    # ── Job 생성 ──
    if args.split <= 1:
        job_names = [create_job(batch_id, args)]
    else:
        # 분할 배치: batch_id-01, batch_id-02, ...
        # 실제 파일 분할은 upload 시 하위 폴더로 나누는 방식 (TODO)
        job_names = []
        for i in range(1, args.split + 1):
            sub_id = f"{batch_id}-{i:02d}"
            job_names.append(create_job(sub_id, args))

    # ── 대기 ──
    print(f"\n  Job 완료 대기 ({len(job_names)}개)...")
    results = wait_for_jobs(job_names, args.timeout)

    # ── 결과 수집 ──
    n_ok = sum(1 for v in results.values() if v == "succeeded")
    if n_ok > 0:
        collect_results(batch_id, local_output)

    # ── 요약 ──
    n_fail = sum(1 for v in results.values() if v != "succeeded")
    print(f"\n{'=' * 60}")
    print(f"  완료: {n_ok} 성공, {n_fail} 실패")
    print(f"  결과: {local_output}")
    print(f"{'=' * 60}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
