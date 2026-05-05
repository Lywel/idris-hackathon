"""Latency + GPU memory tracking + profilers around SSeRiouSS inference.

CUDA-only. Wraps the pure inference functions from `sseriouss.py`.

Two profilers are exposed:
  - `profile_torch`: PyTorch's built-in profiler (CPU + CUDA via CUPTI).
    Returns a printable top-ops table.
  - `nvtx_range` / `cuda_profiler_session`: NVTX annotations + cudaProfilerApi
    toggles so the interesting region can be captured by Nsight Systems
    (`nsys profile --capture-range=cudaProfilerApi`).
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.cuda.nvtx as nvtx

from sseriouss import SAMPLE_RATE, WINDOW_SAMPLES, chunk_waveform


# --- helpers ------------------------------------------------------------

@contextmanager
def cuda_sync(device):
    torch.cuda.synchronize(device)
    yield
    torch.cuda.synchronize(device)


@contextmanager
def nvtx_range(name: str):
    """NVTX scope visible in Nsight Systems timelines."""
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


@contextmanager
def cuda_profiler_session():
    """Toggle cudaProfilerStart/Stop. Pair with `nsys --capture-range=cudaProfilerApi`."""
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.cudart().cudaProfilerStop()


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _time_forward(model, x, device) -> float:
    with cuda_sync(device):
        t0 = time.perf_counter()
        model(x)
    return (time.perf_counter() - t0) * 1000.0


def _gpu_snapshot(device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "peak_alloc": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "current_alloc": torch.cuda.memory_allocated(device),
        "device_props": {"name": props.name, "total_memory": props.total_memory},
    }


def _reset_gpu(device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


# --- result containers --------------------------------------------------

@dataclass
class LatencyStats:
    timings_ms: List[float]

    @property
    def mean(self): return statistics.mean(self.timings_ms)
    @property
    def median(self): return statistics.median(self.timings_ms)
    @property
    def stdev(self): return statistics.stdev(self.timings_ms) if len(self.timings_ms) > 1 else 0.0
    @property
    def p95(self):
        s = sorted(self.timings_ms)
        return s[max(0, int(0.95 * len(s)) - 1)]
    @property
    def min(self): return min(self.timings_ms)
    @property
    def max(self): return max(self.timings_ms)


@dataclass
class FullFileResult:
    duration_sec: float
    num_chunks: int
    total_seconds: float
    per_batch_ms: List[float]
    output_shape: tuple
    peak_alloc: Optional[int] = None
    peak_reserved: Optional[int] = None
    current_alloc: Optional[int] = None
    device_props: Optional[dict] = None

    @property
    def rtf(self): return self.total_seconds / self.duration_sec
    @property
    def xrt(self): return self.duration_sec / self.total_seconds
    @property
    def chunks_per_sec(self): return self.num_chunks / self.total_seconds


# --- benchmarks ---------------------------------------------------------

def benchmark_single_window(model, device, warmup=3, runs=50) -> LatencyStats:
    dummy = torch.zeros(1, 1, WINDOW_SAMPLES, device=device)
    with torch.inference_mode():
        with nvtx_range("warmup"):
            for _ in range(warmup):
                model(dummy)
        with nvtx_range("single-window"):
            timings = [_time_forward(model, dummy, device) for _ in range(runs)]
    return LatencyStats(timings)


def benchmark_full_file(model, waveform, device, batch_size=32, step_seconds=2.5) -> FullFileResult:
    duration_sec = waveform.shape[1] / SAMPLE_RATE
    chunks = chunk_waveform(waveform, int(step_seconds * SAMPLE_RATE))

    _reset_gpu(device)
    per_batch_ms: List[float] = []
    pieces: List[torch.Tensor] = []

    with cuda_sync(device):
        t_start = time.perf_counter()

    with torch.inference_mode(), nvtx_range(f"full-file:bs={batch_size}"):
        for i, batch in enumerate(chunks.split(batch_size)):
            batch = batch.to(device, non_blocking=True)
            with nvtx_range(f"batch[{i}]"), cuda_sync(device):
                t0 = time.perf_counter()
                scores = model(batch)
            per_batch_ms.append((time.perf_counter() - t0) * 1000.0)
            pieces.append(scores.cpu())

    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - t_start

    full_scores = torch.cat(pieces, dim=0)
    result = FullFileResult(
        duration_sec=duration_sec,
        num_chunks=chunks.shape[0],
        total_seconds=total_seconds,
        per_batch_ms=per_batch_ms,
        output_shape=tuple(full_scores.shape),
    )
    for k, v in _gpu_snapshot(device).items():
        setattr(result, k, v)
    return result


# --- profilers ----------------------------------------------------------

def profile_torch(model, device, row_limit=15) -> str:
    """PyTorch built-in profiler (CPU + CUDA via CUPTI). Returns top-ops table."""
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    dummy = torch.zeros(1, 1, WINDOW_SAMPLES, device=device)
    with torch.inference_mode(), torch.profiler.profile(activities=activities) as prof:
        with nvtx_range("torch-profile"):
            model(dummy)
        torch.cuda.synchronize(device)
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit)
