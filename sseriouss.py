"""SSeRiouSS model + audio + benchmark helpers."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch
import torchaudio

from pyannote.audio.models.segmentation.SSeRiouSS import SSeRiouSS
from pyannote.audio.core.task import Specifications, Problem, Resolution


SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)


# --- audio --------------------------------------------------------------

def load_audio(path: str) -> torch.Tensor:
    """Load file -> mono 16kHz tensor of shape (1, samples)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def chunk_waveform(waveform: torch.Tensor, step_samples: int) -> torch.Tensor:
    """Slice (1, N) into (num_chunks, 1, WINDOW_SAMPLES), zero-padding the tail."""
    total = waveform.shape[1]
    if total < WINDOW_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, WINDOW_SAMPLES - total))
        total = WINDOW_SAMPLES

    num_chunks = max(1, 1 + (total - WINDOW_SAMPLES + step_samples - 1) // step_samples)
    needed = (num_chunks - 1) * step_samples + WINDOW_SAMPLES
    if needed > total:
        waveform = torch.nn.functional.pad(waveform, (0, needed - total))

    chunks = waveform.unfold(dimension=1, size=WINDOW_SAMPLES, step=step_samples)
    return chunks.permute(1, 0, 2).contiguous()


# --- model --------------------------------------------------------------

def build_model() -> SSeRiouSS:
    """SSeRiouSS with WavLM_BASE backbone + 3-speaker frame-level head."""
    model = SSeRiouSS(wav2vec="WAVLM_BASE", wav2vec_frozen=True)
    model.specifications = Specifications(
        problem=Problem.MULTI_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=WINDOW_SECONDS,
        classes=["speaker#1", "speaker#2", "speaker#3"],
    )
    model.build()
    return model.eval()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --- benchmark helpers --------------------------------------------------

@contextmanager
def cuda_sync(device: torch.device):
    """Synchronise CUDA before/after a block; no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        for _ in range(warmup):
            model(dummy)
        timings = [_time_forward(model, dummy, device) for _ in range(runs)]
    return LatencyStats(timings)


def _gpu_memory_snapshot(device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "peak_alloc": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "current_alloc": torch.cuda.memory_allocated(device),
        "device_props": {"name": props.name, "total_memory": props.total_memory},
    }


def run_full_file(model, waveform, device, batch_size=32, step_seconds=2.5) -> FullFileResult:
    duration_sec = waveform.shape[1] / SAMPLE_RATE
    chunks = chunk_waveform(waveform, int(step_seconds * SAMPLE_RATE))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    per_batch_ms: List[float] = []
    all_scores: List[torch.Tensor] = []

    with torch.inference_mode(), cuda_sync(device):
        t_start = time.perf_counter()

    with torch.inference_mode():
        for i in range(0, chunks.shape[0], batch_size):
            batch = chunks[i : i + batch_size].to(device, non_blocking=True)
            with cuda_sync(device):
                tb0 = time.perf_counter()
                scores = model(batch)
            per_batch_ms.append((time.perf_counter() - tb0) * 1000.0)
            all_scores.append(scores.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - t_start

    full_scores = torch.cat(all_scores, dim=0)
    result = FullFileResult(
        duration_sec=duration_sec,
        num_chunks=chunks.shape[0],
        total_seconds=total_seconds,
        per_batch_ms=per_batch_ms,
        output_shape=tuple(full_scores.shape),
    )
    if device.type == "cuda":
        for k, v in _gpu_memory_snapshot(device).items():
            setattr(result, k, v)
    return result


def profile_single_window(model, device, row_limit=15) -> str:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    dummy = torch.zeros(1, 1, WINDOW_SAMPLES, device=device)
    with torch.inference_mode(), torch.profiler.profile(activities=activities) as prof:
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
