"""
벤치마크 수행 이력 추적 모듈.

각 벤치마크 실행의 전략, 모델별 소요 시간, 토큰 사용량, 플랫폼 정보를 기록한다.
"""

import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 플랫폼 정보 수집
# ---------------------------------------------------------------------------


def collect_platform_info() -> Dict[str, Any]:
    """현재 실행 플랫폼 정보를 수집한다."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }

    # CPU 코어 수
    info["cpu_count_logical"] = os.cpu_count()
    try:
        info["cpu_count_physical"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        info["cpu_count_physical"] = info["cpu_count_logical"]

    # 메모리 (psutil 없이도 작동)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        # macOS: sysctl
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    info["ram_total_gb"] = round(int(result.stdout.strip()) / (1024 ** 3), 1)
            except Exception:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            info["ram_total_gb"] = round(kb / (1024 ** 2), 1)
                        elif line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            info["ram_available_gb"] = round(kb / (1024 ** 2), 1)
            except Exception:
                pass

    # GPU 정보
    info["gpu"] = _detect_gpu()

    # macOS Apple Silicon 칩 감지
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["apple_chip"] = result.stdout.strip()
        except Exception:
            pass

    # Git 커밋 해시
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    return info


def _detect_gpu() -> Optional[str]:
    """사용 가능한 GPU 정보를 반환한다."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# 토큰 사용량 데이터 구조
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """LLM API 토큰 사용량."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_model: str = ""
    num_calls: int = 0

    def merge(self, other: "TokenUsage"):
        """다른 TokenUsage를 병합한다."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.total_cost_usd += other.total_cost_usd
        self.num_calls += other.num_calls


@contextmanager
def track_openai_tokens():
    """LangChain OpenAI 콜백으로 토큰 사용량을 추적하는 컨텍스트 매니저.

    Yields:
        TokenUsage: 컨텍스트 종료 시 사용량이 채워진 객체.

    Usage:
        with track_openai_tokens() as usage:
            llm.invoke(prompt)
        print(usage.total_tokens)
    """
    usage = TokenUsage()
    try:
        from langchain_community.callbacks import get_openai_callback
        with get_openai_callback() as cb:
            yield usage
        usage.prompt_tokens = cb.prompt_tokens
        usage.completion_tokens = cb.completion_tokens
        usage.total_tokens = cb.total_tokens
        usage.total_cost_usd = round(cb.total_cost, 6)
        usage.num_calls = cb.successful_requests
    except ImportError:
        # langchain_community 없으면 추적 없이 진행
        yield usage


# ---------------------------------------------------------------------------
# 수행 이력 데이터 구조
# ---------------------------------------------------------------------------


@dataclass
class StrategyTiming:
    """개별 전략의 빌드 + 쿼리 타이밍."""
    label: str
    dense_model: Optional[str] = None
    sparse_model: Optional[str] = None
    reranker: Optional[str] = None
    llm_support: Optional[str] = None
    retrieval_mode: Optional[str] = None

    build_time_s: float = 0.0
    build_success: bool = True
    build_error: Optional[str] = None

    # 인덱싱 토큰 (contextual retrieval 등 LLM 사용 시)
    indexing_tokens: Optional[Dict[str, Any]] = None

    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_queries: int = 0
    error_count: int = 0

    ragas_scores: Optional[Dict[str, float]] = None


@dataclass
class PhaseTime:
    """개별 단계의 소요 시간 및 토큰."""
    phase: str
    duration_s: float = 0.0
    tokens: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRunRecord:
    """단일 벤치마크 실행의 전체 기록."""
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    preset: str = ""
    k: int = 3
    top_n: Optional[int] = None
    pass1_only: bool = False
    layers: bool = False
    num_combos: int = 0
    num_queries: int = 0
    num_docs: int = 0

    platform_info: Dict[str, Any] = field(default_factory=dict)
    strategy_timings: List[Dict[str, Any]] = field(default_factory=list)

    # 단계별 소요 시간
    phase_times: List[Dict[str, Any]] = field(default_factory=list)

    # 토큰 사용량 총계
    token_usage_total: Dict[str, Any] = field(default_factory=dict)

    # 요약 통계
    total_build_time_s: float = 0.0
    total_query_time_s: float = 0.0
    successful_strategies: int = 0
    failed_strategies: int = 0


# ---------------------------------------------------------------------------
# RunTracker 클래스
# ---------------------------------------------------------------------------


class RunTracker:
    """벤치마크 수행 이력을 추적하고 기록한다."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir / "run_history"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._record = BenchmarkRunRecord()
        self._record.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._record.started_at = datetime.now(timezone.utc).isoformat()
        self._record.platform_info = collect_platform_info()
        self._start_time = time.time()

        self._timings: List[StrategyTiming] = []
        self._phases: List[PhaseTime] = []
        self._token_total = TokenUsage()
        self._current_build_start: Optional[float] = None

    def set_config(
        self,
        preset: str,
        k: int,
        top_n: Optional[int],
        pass1_only: bool,
        layers: bool,
        num_combos: int,
        num_queries: int,
        num_docs: int,
    ):
        """벤치마크 설정을 기록한다."""
        self._record.preset = preset
        self._record.k = k
        self._record.top_n = top_n
        self._record.pass1_only = pass1_only
        self._record.layers = layers
        self._record.num_combos = num_combos
        self._record.num_queries = num_queries
        self._record.num_docs = num_docs

    # ------------------------------------------------------------------
    # 단계 타이밍
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str):
        """단계(phase) 소요 시간을 자동 측정하는 컨텍스트 매니저.

        Usage:
            with tracker.phase("chunking"):
                ...  # 청킹 로직
        """
        pt = PhaseTime(phase=name)
        t0 = time.time()
        yield pt
        pt.duration_s = round(time.time() - t0, 2)
        self._phases.append(pt)
        print(f"  [RunTracker] {name}: {pt.duration_s}s")

    def record_phase(self, name: str, duration_s: float,
                     tokens: Optional[TokenUsage] = None):
        """단계 소요 시간을 수동 기록한다."""
        pt = PhaseTime(phase=name, duration_s=round(duration_s, 2))
        if tokens:
            pt.tokens = asdict(tokens)
            self._token_total.merge(tokens)
        self._phases.append(pt)

    # ------------------------------------------------------------------
    # 토큰 추적
    # ------------------------------------------------------------------

    def add_tokens(self, usage: TokenUsage, phase: str = ""):
        """토큰 사용량을 총계에 추가한다."""
        self._token_total.merge(usage)

    # ------------------------------------------------------------------
    # 전략 빌드 타이밍
    # ------------------------------------------------------------------

    def start_build(self, label: str, dense: Optional[str] = None, sparse: Optional[str] = None,
                    reranker: Optional[str] = None, llm_support: Optional[str] = None,
                    retrieval_mode: Optional[str] = None) -> StrategyTiming:
        """전략 빌드 시작을 기록한다."""
        timing = StrategyTiming(
            label=label,
            dense_model=dense,
            sparse_model=sparse,
            reranker=reranker,
            llm_support=llm_support,
            retrieval_mode=retrieval_mode,
        )
        self._current_build_start = time.time()
        self._timings.append(timing)
        return timing

    def end_build(self, timing: StrategyTiming, success: bool, error: Optional[str] = None,
                  tokens: Optional[TokenUsage] = None):
        """전략 빌드 완료를 기록한다."""
        if self._current_build_start is not None:
            timing.build_time_s = round(time.time() - self._current_build_start, 2)
        timing.build_success = success
        timing.build_error = error
        if tokens and tokens.total_tokens > 0:
            timing.indexing_tokens = asdict(tokens)
            self._token_total.merge(tokens)
        self._current_build_start = None

    # ------------------------------------------------------------------
    # 쿼리 / RAGAS
    # ------------------------------------------------------------------

    def record_query_stats(self, timing: StrategyTiming, latencies_ms: List[float],
                           error_count: int = 0):
        """쿼리 레이턴시 통계를 기록한다."""
        if not latencies_ms:
            return
        sorted_lats = sorted(latencies_ms)
        n = len(sorted_lats)
        timing.total_queries = n + error_count
        timing.error_count = error_count
        timing.avg_latency_ms = round(sum(sorted_lats) / n, 1)
        timing.min_latency_ms = round(sorted_lats[0], 1)
        timing.max_latency_ms = round(sorted_lats[-1], 1)
        timing.p50_latency_ms = round(sorted_lats[n // 2], 1)
        timing.p95_latency_ms = round(sorted_lats[int(n * 0.95)], 1)

    def record_ragas(self, timing: StrategyTiming, scores: Dict[str, float]):
        """RAGAS 점수를 기록한다."""
        timing.ragas_scores = scores

    def record_ragas_tokens(self, tokens: TokenUsage):
        """RAGAS 평가에서 소비된 토큰을 기록한다."""
        self.record_phase("ragas_evaluation_tokens", 0.0, tokens=tokens)

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    def find_timing(self, label: str) -> Optional[StrategyTiming]:
        """라벨로 타이밍을 검색한다."""
        for t in self._timings:
            if t.label == label:
                return t
        return None

    # ------------------------------------------------------------------
    # 공개 스냅샷
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """현재까지의 기록을 dict로 반환한다 (시각화/외부 연동용).

        finalize() 호출 전에도 사용 가능하며, 이 시점까지의 데이터를 스냅샷으로 반환한다.
        """
        rec = self._record
        rec.strategy_timings = [asdict(t) for t in self._timings]
        rec.phase_times = [asdict(p) for p in self._phases]
        rec.token_usage_total = asdict(self._token_total)
        return asdict(rec)

    # ------------------------------------------------------------------
    # 마무리
    # ------------------------------------------------------------------

    def finalize(self) -> Path:
        """기록을 마무리하고 JSON 파일로 저장한다."""
        self._record.finished_at = datetime.now(timezone.utc).isoformat()
        self._record.duration_s = round(time.time() - self._start_time, 1)

        # 타이밍 데이터 집계
        self._record.strategy_timings = [asdict(t) for t in self._timings]
        self._record.successful_strategies = sum(1 for t in self._timings if t.build_success)
        self._record.failed_strategies = sum(1 for t in self._timings if not t.build_success)
        self._record.total_build_time_s = round(
            sum(t.build_time_s for t in self._timings), 1
        )
        self._record.total_query_time_s = round(
            sum(t.avg_latency_ms * t.total_queries / 1000 for t in self._timings
                if t.build_success), 1
        )

        # 단계별 시간
        self._record.phase_times = [asdict(p) for p in self._phases]

        # 토큰 사용량 총계
        self._record.token_usage_total = asdict(self._token_total)

        # JSON 저장
        record_dict = asdict(self._record)
        filename = f"run_{self._record.run_id}.json"
        filepath = self._output_dir / filename
        filepath.write_text(
            json.dumps(record_dict, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # 최신 실행 심링크 갱신
        latest = self._output_dir / "latest.json"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(filename)

        # 요약 출력
        tt = self._token_total
        print(f"\n  [RunTracker] 수행 이력 저장: {filepath}")
        print(f"  [RunTracker] 총 소요: {self._record.duration_s}s, "
              f"성공: {self._record.successful_strategies}, "
              f"실패: {self._record.failed_strategies}")
        if tt.total_tokens > 0:
            print(f"  [RunTracker] 토큰 총계: {tt.total_tokens:,} "
                  f"(prompt: {tt.prompt_tokens:,}, "
                  f"completion: {tt.completion_tokens:,}, "
                  f"calls: {tt.num_calls}, "
                  f"cost: ${tt.total_cost_usd:.4f})")

        # 단계별 시간 + 비중 요약
        if self._phases:
            total_s = self._record.duration_s or 1
            print(f"  [RunTracker] 단계별 소요 (총 {total_s}s):")
            for p in self._phases:
                if p.duration_s <= 0:
                    continue
                pct = p.duration_s / total_s * 100
                token_info = ""
                if p.tokens and p.tokens.get("total_tokens", 0) > 0:
                    token_info = f" | {p.tokens['total_tokens']:,} tokens"
                print(f"    {p.phase}: {p.duration_s}s ({pct:.1f}%){token_info}")

        return filepath
