"""CLI demo for SSeRiouSS. CUDA-only.

Inference: `sseriouss.py`. Tracking + profilers: `bench.py`.

Profiler modes:
  --profile-torch   PyTorch profiler (CPU + GPU kernels via CUPTI), prints top ops.
  --nsys            Wrap the inference region in cudaProfilerStart/Stop so that
                    `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`
                    only captures the part you care about.

NVTX ranges (`warmup`, `single-window`, `full-file:bs=N`, `batch[i]`) are always
emitted, so an `nsys profile uv run python demo.py ...` invocation will already
group kernels under named ranges in the timeline.
"""

import argparse
import statistics
import time

import torch

from sseriouss import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    build_model,
    load_audio,
    require_cuda,
)
from bench import (
    benchmark_full_file,
    benchmark_single_window,
    cuda_profiler_session,
    fmt_bytes,
    nvtx_range,
    profile_torch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--step", type=float, default=2.5,
                   help="sliding window step in seconds (default 2.5 -> 50%% overlap)")
    p.add_argument("--no-full-file", action="store_true")
    p.add_argument("--profile-torch", action="store_true",
                   help="run PyTorch profiler on a single window")
    p.add_argument("--chrome-trace", default=None,
                   help="path to also write a Chrome/Perfetto JSON trace "
                        "(implies --profile-torch)")
    p.add_argument("--tb-trace", default=None,
                   help="directory to write a TensorBoard-compatible trace "
                        "(implies --profile-torch). View with `tensorboard --logdir DIR` "
                        "after `pip install torch-tb-profiler`")
    p.add_argument("--nsys", action="store_true",
                   help="toggle cudaProfilerStart/Stop around inference for Nsight Systems")
    return p.parse_args()


def report_single_window(stats, warmup, runs):
    rtf = (stats.mean / 1000.0) / WINDOW_SECONDS
    print(f"\n=== Single-window latency ===")
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


def report_memory(result):
    print(f"  gpu: peak_alloc={fmt_bytes(result.peak_alloc)}  "
          f"peak_reserved={fmt_bytes(result.peak_reserved)}  "
          f"device={result.device_props['name']} "
          f"({fmt_bytes(result.device_props['total_memory'])})")
    print(f"  output shape: {result.output_shape}")


def main():
    args = parse_args()
    device = require_cuda()

    print(f"[1/3] Loading audio: {args.audio}")
    waveform = load_audio(args.audio)
    duration = waveform.shape[1] / SAMPLE_RATE
    print(f"      shape={tuple(waveform.shape)}  duration={duration:.1f}s "
          f"({duration/60:.2f} min)")

    print("[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...")
    t0 = time.perf_counter()
    with nvtx_range("build_model"):
        model = build_model().to(device)
        torch.cuda.synchronize(device)
    print(f"      params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"build={1000*(time.perf_counter()-t0):.1f}ms")

    print("[3/3] Running benchmarks...")
    capture = cuda_profiler_session() if args.nsys else nvtx_range("inference")
    with capture:
        report_single_window(
            benchmark_single_window(model, device, args.warmup, args.runs),
            args.warmup, args.runs,
        )
        if not args.no_full_file:
            result = benchmark_full_file(model, waveform, device, args.batch_size, args.step)
            report_throughput(result, args.batch_size, args.step)
            report_memory(result)

    if args.profile_torch or args.chrome_trace or args.tb_trace:
        print("\n=== torch.profiler (single window) ===")
        print(profile_torch(model, device,
                            chrome_trace=args.chrome_trace,
                            tb_dir=args.tb_trace))
        if args.chrome_trace:
            print(f"  chrome trace written to: {args.chrome_trace}")
            print(f"  open at https://ui.perfetto.dev/  or  chrome://tracing")
        if args.tb_trace:
            print(f"  tensorboard trace written under: {args.tb_trace}")
            print(f"  view with: tensorboard --logdir {args.tb_trace}")

    print("\nReminder: LSTM + classifier head are untrained; values not meaningful.")


if __name__ == "__main__":
    main()
