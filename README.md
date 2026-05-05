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

```
 uv run python demo.py B047.wav
[1/3] Loading audio: B047.wav
      shape=(1, 58010080)  duration=3625.6s (60.43 min)
[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...
      params=96.5M  build=673.3ms
[3/3] Running benchmarks...

=== Single-window latency ===
  warmup=3  runs=50  input=(1, 1, 80000)
  mean=26.92ms  median=23.79ms  stdev=11.84ms  p95=47.17ms
  min=15.43ms  max=60.17ms  RTF=0.0054 (185.7x real-time)

=== Full-file inference ===
  duration=3625.6s (60.43 min)  window=5.0s  step=2.5s
  batch_size=32  num_chunks=1451
  total=23.47s  RTF=0.00647 (154.5x real-time)  chunks/s=61.8
  per-batch ms: mean=509.30  median=514.82  min=170.85  max=556.47
  gpu: peak_alloc=2.3 GiB  peak_reserved=2.4 GiB  device=NVIDIA GeForce RTX 5060 Laptop GPU (7.5 GiB)
  output shape: (1451, 249, 3)

Reminder: LSTM + classifier head are untrained; values not meaningful.

~/Documents/idris-hackaton/demo main* 29s
❯ uv run python demo.py B047.wav --profile-torch
[1/3] Loading audio: B047.wav
      shape=(1, 58010080)  duration=3625.6s (60.43 min)
[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...
      params=96.5M  build=657.8ms
[3/3] Running benchmarks...

=== Single-window latency ===
  warmup=3  runs=50  input=(1, 1, 80000)
  mean=22.07ms  median=16.53ms  stdev=10.30ms  p95=46.24ms
  min=15.37ms  max=51.86ms  RTF=0.0044 (226.6x real-time)

=== Full-file inference ===
  duration=3625.6s (60.43 min)  window=5.0s  step=2.5s
  batch_size=32  num_chunks=1451
  total=23.01s  RTF=0.00635 (157.6x real-time)  chunks/s=63.1
  per-batch ms: mean=499.05  median=506.52  min=168.36  max=534.22
  gpu: peak_alloc=2.3 GiB  peak_reserved=2.4 GiB  device=NVIDIA GeForce RTX 5060 Laptop GPU (7.5 GiB)
  output shape: (1451, 249, 3)

=== torch.profiler (single window) ===
/home/maxime/Documents/idris-hackaton/demo/.venv/lib/python3.13/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           aten::linear         3.14%       1.127ms        28.96%      10.386ms     162.279us       0.000us         0.00%       7.168ms     111.995us            64
                                            aten::addmm        11.55%       4.142ms        15.66%       5.617ms     108.014us       7.075ms        38.83%       7.075ms     136.054us            52
                                           aten::conv1d         0.02%       7.422us         3.89%       1.394ms     174.225us       0.000us         0.00%       5.924ms     740.530us             8
                                      aten::convolution         0.02%       8.838us         3.87%       1.386ms     173.297us       0.000us         0.00%       5.924ms     740.530us             8
                                     aten::_convolution         0.13%      45.504us         3.84%       1.378ms     172.193us       0.000us         0.00%       5.924ms     740.530us             8
                                aten::cudnn_convolution         0.72%     257.666us         3.56%       1.277ms     159.574us       5.710ms        31.34%       5.914ms     739.194us             8
                                             aten::lstm         0.56%     200.027us         5.07%       1.819ms       1.819ms       0.000us         0.00%       3.054ms       3.054ms             1
                                       aten::_cudnn_rnn         2.02%     725.626us         4.39%       1.573ms       1.573ms       3.054ms        16.76%       3.054ms       3.054ms             1
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.018ms        16.56%       3.018ms     125.742us            24
void RNN_blockPersist_fp_LSTM<float, float, float, 1...         0.00%       0.000us         0.00%       0.000us       0.000us       2.946ms        16.17%       2.946ms     368.265us             8
void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...         0.00%       0.000us         0.00%       0.000us       0.000us       2.728ms        14.98%       2.728ms     170.523us            16
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       2.046ms        11.23%       2.046ms     157.415us            13
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us       1.979ms        10.86%       1.979ms     164.901us            12
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...         0.00%       0.000us         0.00%       0.000us       0.000us       1.004ms         5.51%       1.004ms       1.004ms             1
void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...         0.00%       0.000us         0.00%       0.000us       0.000us     919.439us         5.05%     919.439us     183.888us             5
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 35.863ms
Self CUDA time total: 18.219ms


Reminder: LSTM + classifier head are untrained; values not meaningful.
```
