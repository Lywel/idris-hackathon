"""
Minimal demo of pyannote.audio's SSeRiouSS segmentation model + benchmark.

Architecture: WavLM (wav2vec) -> LSTM -> Linear -> Classifier
Input:  waveform (batch, channel, samples) at 16kHz mono
Output: per-frame scores (batch, frames, classes)

SSeRiouSS is designed to operate on short windows (5s by default). To process
a long recording we slice it into overlapping windows and batch them through
the model -- this mirrors what `pyannote.audio.Inference` does internally.

NOTE: WavLM is loaded with pretrained torchaudio weights but the LSTM +
classifier head are randomly initialised. Output values are NOT meaningful;
this script measures inference cost only.

Usage:
    python demo_sseriouss.py <audio_file> [--device cuda|cpu|auto]
                                          [--runs N] [--warmup N]
                                          [--batch-size N] [--step SECONDS]
                                          [--profile]
"""

import argparse
import statistics
import time
from contextlib import contextmanager
from typing import Optional

import torch
import torchaudio

from pyannote.audio.models.segmentation.SSeRiouSS import SSeRiouSS
from pyannote.audio.core.task import Specifications, Problem, Resolution


SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Audio + model
# ---------------------------------------------------------------------------

def load_audio(path: str) -> torch.Tensor:
    """Load full audio, downmix to mono, resample to 16kHz. Returns (1, samples)."""
    waveform, sr = torchaudio.load(path)  # (channels, samples)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform  # (1, samples)


def chunk_waveform(
    waveform: torch.Tensor,
    window_samples: int,
    step_samples: int,
) -> torch.Tensor:
    """Slice (1, samples) into (num_chunks, 1, window_samples) with given step.

    The last partial chunk is zero-padded so we cover the whole file.
    """
    total = waveform.shape[1]

    # Pad so we have an integer number of windows covering everything.
    if total < window_samples:
        pad = window_samples - total
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        total = waveform.shape[1]

    # Number of chunks needed to cover the file given the step size.
    if total <= window_samples:
        num_chunks = 1
    else:
        num_chunks = 1 + (total - window_samples + step_samples - 1) // step_samples

    needed = (num_chunks - 1) * step_samples + window_samples
    if needed > total:
        waveform = torch.nn.functional.pad(waveform, (0, needed - total))

    # Build (num_chunks, 1, window_samples).
    chunks = waveform.unfold(dimension=1, size=window_samples, step=step_samples)
    # unfold gives (1, num_chunks, window_samples) -> reorder to (num_chunks, 1, window_samples)
    chunks = chunks.permute(1, 0, 2).contiguous()
    return chunks


def build_model() -> SSeRiouSS:
    model = SSeRiouSS(
        wav2vec="WAVLM_BASE",
        wav2vec_frozen=True,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )
    model.specifications = Specifications(
        problem=Problem.MULTI_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=WINDOW_SECONDS,
        classes=["speaker#1", "speaker#2", "speaker#3"],
        powerset_max_classes=None,
    )
    model.build()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

@contextmanager
def cuda_sync(device: torch.device):
    """Synchronise CUDA before and after a block, no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def benchmark_single_window(
    model: SSeRiouSS,
    device: torch.device,
    warmup: int,
    runs: int,
) -> None:
    """Per-window latency stats on a fixed 5s dummy window."""
    print(f"\n=== Single-window latency on {device} ===")
    print(f"  warmup runs : {warmup}")
    print(f"  timed runs  : {runs}")

    dummy = torch.zeros(1, 1, int(WINDOW_SECONDS * SAMPLE_RATE), device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms = []
    with torch.inference_mode():
        for _ in range(runs):
            with cuda_sync(device):
                t0 = time.perf_counter()
                _ = model(dummy)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    mean = statistics.mean(timings_ms)
    median = statistics.median(timings_ms)
    stdev = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    p95 = sorted(timings_ms)[max(0, int(0.95 * len(timings_ms)) - 1)]
    rtf = (mean / 1000.0) / WINDOW_SECONDS

    print(f"  input shape : (1, 1, {int(WINDOW_SECONDS*SAMPLE_RATE)})")
    print("  -- latency (ms / window) --")
    print(f"    mean   : {mean:8.2f}")
    print(f"    median : {median:8.2f}")
    print(f"    stdev  : {stdev:8.2f}")
    print(f"    p95    : {p95:8.2f}")
    print(f"    min    : {min(timings_ms):8.2f}")
    print(f"    max    : {max(timings_ms):8.2f}")
    print(f"    RTF    : {rtf:.4f}  ({1/rtf:.1f}x real-time)")


def benchmark_full_file(
    model: SSeRiouSS,
    waveform: torch.Tensor,
    device: torch.device,
    batch_size: int,
    step_seconds: float,
) -> None:
    """Process the entire waveform via sliding windows and report totals."""
    duration_sec = waveform.shape[1] / SAMPLE_RATE
    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    step_samples = int(step_seconds * SAMPLE_RATE)

    print(f"\n=== Full-file inference on {device} ===")
    print(f"  audio duration : {duration_sec:.1f} s ({duration_sec/60:.2f} min)")
    print(f"  window         : {WINDOW_SECONDS} s")
    print(f"  step           : {step_seconds} s "
          f"(overlap = {WINDOW_SECONDS - step_seconds:.2f} s)")
    print(f"  batch size     : {batch_size}")

    chunks = chunk_waveform(waveform, window_samples, step_samples)  # (N, 1, W)
    num_chunks = chunks.shape[0]
    print(f"  num chunks     : {num_chunks}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    all_scores = []
    per_batch_ms = []

    with torch.inference_mode():
        with cuda_sync(device):
            t_start = time.perf_counter()

        for i in range(0, num_chunks, batch_size):
            batch = chunks[i : i + batch_size].to(device, non_blocking=True)
            with cuda_sync(device):
                tb0 = time.perf_counter()
                scores = model(batch)  # (B, frames, classes)
            tb1 = time.perf_counter()
            per_batch_ms.append((tb1 - tb0) * 1000.0)
            all_scores.append(scores.cpu())

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_end = time.perf_counter()

    total_s = t_end - t_start
    rtf = total_s / duration_sec
    xrt = duration_sec / total_s

    full_scores = torch.cat(all_scores, dim=0)  # (N, frames, classes)

    print("\n  -- throughput --")
    print(f"    total wall time : {total_s:.2f} s")
    print(f"    RTF             : {rtf:.5f}  ({xrt:.1f}x real-time)")
    print(f"    chunks/sec      : {num_chunks/total_s:.1f}")

    if per_batch_ms:
        print("  -- per-batch latency (ms) --")
        print(f"    mean   : {statistics.mean(per_batch_ms):8.2f}")
        print(f"    median : {statistics.median(per_batch_ms):8.2f}")
        print(f"    min    : {min(per_batch_ms):8.2f}")
        print(f"    max    : {max(per_batch_ms):8.2f}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        cur = torch.cuda.memory_allocated(device)
        props = torch.cuda.get_device_properties(device)
        print("  -- gpu memory --")
        print(f"    current alloc : {fmt_bytes(cur)}")
        print(f"    peak alloc    : {fmt_bytes(peak)}")
        print(f"    peak reserved : {fmt_bytes(peak_reserved)}")
        print(f"    device        : {props.name} ({fmt_bytes(props.total_memory)})")
    else:
        print("  -- cpu --")
        print(f"    threads : {torch.get_num_threads()}")

    print(f"\n  stacked output shape : {tuple(full_scores.shape)}  "
          f"(num_chunks, frames_per_window, classes)")


def find_max_batch_size(
    model: SSeRiouSS,
    device: torch.device,
    start: int = 4,
    cap: int = 1024,
    safety: float = 0.9,
) -> int:
    """Probe the largest power-of-two batch size that fits in GPU memory.

    Doubles until OOM, then returns last_ok scaled by `safety` (rounded down
    to a power of two) to leave headroom for fragmentation. CPU returns start.
    """
    if device.type != "cuda":
        return start

    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    last_ok = 0
    bs = start
    print(f"\n=== Probing max batch size on {device} ===")
    while bs <= cap:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.zeros(bs, 1, window_samples, device=device)
            with torch.inference_mode():
                _ = model(dummy)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            print(f"  batch={bs:5d}  ok   peak={fmt_bytes(peak)} "
                  f"({100*peak/total:.1f}% of {fmt_bytes(total)})")
            last_ok = bs
            del dummy
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:5d}  OOM")
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={bs:5d}  OOM ({e.__class__.__name__})")
                torch.cuda.empty_cache()
                break
            raise

    if last_ok == 0:
        print("  fell back to batch=1")
        return 1

    # Stay one power of two below the OOM ceiling for safety.
    picked = last_ok
    print(f"  -> picked batch={picked} (safety margin {int((1-safety)*100)}%)")
    return picked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="device to run inference on (default: cuda)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="warmup runs (single-window bench)")
    parser.add_argument("--runs", type=int, default=50, help="timed runs (single-window bench)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for full-file inference (default 32). "
             "The conv feature extractor's GroupNorm spikes memory; "
             "use --auto-batch to find the largest safe value.",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="probe the largest power-of-two batch size that fits in GPU memory, "
             "then use it for full-file inference",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.5,
        help="sliding window step in seconds (default 2.5 -> 50%% overlap)",
    )
    parser.add_argument(
        "--no-full-file",
        action="store_true",
        help="skip full-file sliding-window inference",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run torch.profiler for one inference and print top ops",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[1/3] Loading audio: {args.audio}")
    waveform = load_audio(args.audio)  # (1, samples) on CPU
    duration = waveform.shape[1] / SAMPLE_RATE
    print(f"      waveform shape={tuple(waveform.shape)} sr={SAMPLE_RATE} "
          f"duration={duration:.1f}s ({duration/60:.2f} min)")

    print("[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...")
    t0 = time.perf_counter()
    model = build_model().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_ms = (time.perf_counter() - t0) * 1000.0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      total parameters : {n_params/1e6:.1f} M")
    print(f"      build + to(dev)  : {build_ms:.1f} ms")

    print("[3/3] Running benchmarks...")
    benchmark_single_window(model, device, args.warmup, args.runs)

    batch_size = args.batch_size
    if args.auto_batch:
        batch_size = find_max_batch_size(model, device)

    if not args.no_full_file:
        benchmark_full_file(model, waveform, device, batch_size, args.step)

    if args.profile:
        print("\n=== torch.profiler (single window) ===")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        dummy = torch.zeros(1, 1, int(WINDOW_SECONDS * SAMPLE_RATE), device=device)
        with torch.inference_mode():
            with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=15))

    print(
        "\nReminder: LSTM + classifier head are untrained, "
        "so output values are not meaningful."
    )


if __name__ == "__main__":
    main()
