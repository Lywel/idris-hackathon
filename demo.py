"""CLI demo + benchmark for SSeRiouSS. ML logic lives in `sseriouss.py`."""

import argparse
import statistics
import time

import torch

from sseriouss import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    benchmark_single_window,
    build_model,
    fmt_bytes,
    load_audio,
    profile_single_window,
    resolve_device,
    run_full_file,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--step", type=float, default=2.5,
                   help="sliding window step in seconds (default 2.5 -> 50%% overlap)")
    p.add_argument("--no-full-file", action="store_true")
    p.add_argument("--profile", action="store_true")
    return p.parse_args()


def report_single_window(stats, device, warmup, runs):
    rtf = (stats.mean / 1000.0) / WINDOW_SECONDS
    print(f"\n=== Single-window latency on {device} ===")
    print(f"  warmup={warmup}  runs={runs}  input=(1, 1, {WINDOW_SAMPLES})")
    print(f"  mean={stats.mean:.2f}ms  median={stats.median:.2f}ms  "
          f"stdev={stats.stdev:.2f}ms  p95={stats.p95:.2f}ms")
    print(f"  min={stats.min:.2f}ms  max={stats.max:.2f}ms  "
          f"RTF={rtf:.4f} ({1/rtf:.1f}x real-time)")


def report_throughput(result, batch_size, step):
    print(f"\n=== Full-file inference ===")
    print(f"  duration={result.duration_sec:.1f}s ({result.duration_sec/60:.2f} min)  "
          f"window={WINDOW_SECONDS}s  step={step}s")
    print(f"  batch_size={batch_size}  num_chunks={result.num_chunks}")
    print(f"  total={result.total_seconds:.2f}s  RTF={result.rtf:.5f} "
          f"({result.xrt:.1f}x real-time)  chunks/s={result.chunks_per_sec:.1f}")

    if result.per_batch_ms:
        b = result.per_batch_ms
        print(f"  per-batch ms: mean={statistics.mean(b):.2f}  "
              f"median={statistics.median(b):.2f}  min={min(b):.2f}  max={max(b):.2f}")


def report_memory(result, device):
    if device.type == "cuda" and result.device_props:
        print(f"  gpu: peak_alloc={fmt_bytes(result.peak_alloc)}  "
              f"peak_reserved={fmt_bytes(result.peak_reserved)}  "
              f"device={result.device_props['name']} "
              f"({fmt_bytes(result.device_props['total_memory'])})")
    else:
        print(f"  cpu threads: {torch.get_num_threads()}")
    print(f"  output shape: {result.output_shape}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"[1/3] Loading audio: {args.audio}")
    waveform = load_audio(args.audio)
    duration = waveform.shape[1] / SAMPLE_RATE
    print(f"      shape={tuple(waveform.shape)}  duration={duration:.1f}s "
          f"({duration/60:.2f} min)")

    print("[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...")
    t0 = time.perf_counter()
    model = build_model().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"      params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"build={1000*(time.perf_counter()-t0):.1f}ms")

    print("[3/3] Running benchmarks...")
    report_single_window(
        benchmark_single_window(model, device, args.warmup, args.runs),
        device, args.warmup, args.runs,
    )

    if not args.no_full_file:
        result = run_full_file(model, waveform, device, args.batch_size, args.step)
        report_throughput(result, args.batch_size, args.step)
        report_memory(result, device)

    if args.profile:
        print("\n=== torch.profiler (single window) ===")
        print(profile_single_window(model, device))

    print("\nReminder: LSTM + classifier head are untrained; values not meaningful.")


if __name__ == "__main__":
    main()
