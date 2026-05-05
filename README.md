# SSeRiouSS demo (CUDA-only)

Minimal benchmark for `pyannote.audio`'s SSeRiouSS segmentation model.

## Setup

```
uv sync
```

## Run

```
uv run python demo.py path/to/audio.wav
```

## Options

```
--batch-size N         default: 32
--step SECONDS         sliding window step, default: 2.5
--runs N --warmup N    single-window latency bench, default: 50 / 3
--no-full-file         skip full-file pass
--profile-torch        print PyTorch profiler top ops (CPU + GPU kernels)
--nsys                 toggle cudaProfilerStart/Stop around inference for Nsight Systems
```

## Profiling

### PyTorch profiler (built-in, easiest)
```
uv run python demo.py audio.wav --profile-torch
```
Prints top ops sorted by `cuda_time_total`. Uses CUPTI under the hood, so kernel times are real.

### Nsight Systems (system-wide timeline)
NVTX ranges (`build_model`, `warmup`, `single-window`, `full-file:bs=N`, `batch[i]`) are always emitted.

Capture only the inference region (skip model build / WavLM download):
```
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o sseriouss_trace \
  uv run python demo.py audio.wav --nsys
```
Open `sseriouss_trace.nsys-rep` in the Nsight Systems GUI.

Or capture everything:
```
nsys profile -o sseriouss_trace --trace=cuda,nvtx uv run python demo.py audio.wav
```

### Nsight Compute (single-kernel deep dive)
After Nsight Systems pinpoints a hot kernel, profile it with:
```
ncu --target-processes all --set full \
    --nvtx --nvtx-include "single-window/" \
    -o kernel_report \
    uv run python demo.py audio.wav --runs 5 --no-full-file
```

Note: WavLM_BASE is pretrained, but the LSTM and classifier head are random — output values are not meaningful, only timings are.
