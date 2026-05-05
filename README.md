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
NVTX ranges (`build_model`, `warmup`, `single-window`, `full-file:bs=N`, `batch[i]`) are always emitted by the Python code; Nsight Systems renders them as bars on the timeline.

Quick capture (helper script wraps the right `nsys` flags):
```
./nsys_profile.sh B047.wav --no-full-file --runs 10
```
Output: `sseriouss_trace.nsys-rep`.

Inspect on the CLI:
```
nsys stats --report nvtx_sum --report cuda_gpu_kern_sum sseriouss_trace.nsys-rep
```
- `nvtx_sum`: total time inside each NVTX range (`:warmup`, `:single-window`, `:full-file:bs=N`, `:batch[i]`).
- `cuda_gpu_kern_sum`: kernel-by-kernel time on the GPU (LSTM, GEMMs, conv, FlashAttention, ...).

Or open the GUI:
```
nsys-ui sseriouss_trace.nsys-rep
```
You'll see a CPU thread track + a CUDA stream track + an "NVTX" track with the named ranges. Zoom into one `batch[i]` range to see the exact kernel sequence.

#### Manual invocation (if you don't want the helper)
```
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o sseriouss_trace \
  uv run python demo.py B047.wav --nsys
```
The `--nsys` flag in `demo.py` calls `cudaProfilerStart/Stop` so only the inference region is captured (not the WavLM download, model build, or argparse).

To capture **everything** instead (no cudaProfilerApi gating):
```
nsys profile -o sseriouss_trace --trace=cuda,nvtx uv run python demo.py B047.wav
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
~/Documents/idris-hackaton/demo main*
❯  uv run python demo.py B047.wav --profile-torch --warmup 3
[1/3] Loading audio: B047.wav
      shape=(1, 58010080)  duration=3625.6s (60.43 min)
[2/3] Building SSeRiouSS (downloads WavLM_BASE on first run)...
      params=96.5M  build=635.5ms
[3/3] Running benchmarks...

=== Single-window latency ===
  warmup=3  runs=50  input=(1, 1, 80000)
  mean=16.68ms  median=16.67ms  stdev=0.59ms  p95=17.09ms
  min=15.73ms  max=19.65ms  RTF=0.0033 (299.8x real-time)

=== Full-file inference ===
  duration=3625.6s (60.43 min)  window=5.0s  step=2.5s
  batch_size=32  num_chunks=1451
  total=23.27s  RTF=0.00642 (155.8x real-time)  chunks/s=62.3
  per-batch ms: mean=504.81  median=509.94  min=162.86  max=613.19
  gpu: peak_alloc=2.3 GiB  peak_reserved=2.4 GiB  device=NVIDIA GeForce RTX 5060 Laptop GPU (7.5 GiB)
  output shape: (1451, 249, 3)

=== torch.profiler (single window) ===
-- top ops by GPU self time --
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                            aten::addmm         3.36%     510.132us         4.66%     706.564us      13.588us       6.969ms        39.00%       6.969ms     134.027us            52
                                aten::cudnn_convolution         1.42%     215.868us         7.05%       1.069ms     133.590us       5.316ms        29.75%       5.487ms     685.847us             8
                                       aten::_cudnn_rnn         0.54%      81.946us         1.02%     154.786us     154.786us       3.209ms        17.96%       3.209ms       3.209ms             1
void RNN_blockPersist_fp_LSTM<float, float, float, 1...         0.00%       0.000us         0.00%       0.000us       0.000us       3.111ms        17.41%       3.111ms     388.879us             8
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       2.978ms        16.67%       2.978ms     124.098us            24
void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...         0.00%       0.000us         0.00%       0.000us       0.000us       2.740ms        15.33%       2.740ms     171.228us            16
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       2.015ms        11.28%       2.015ms     154.995us            13
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us       1.933ms        10.82%       1.933ms     161.096us            12
void cudnn::engines_precompiled::nchwToNhwcKernel<fl...         0.00%       0.000us         0.00%       0.000us       0.000us     803.168us         4.49%     803.168us      18.254us            44
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...         0.00%       0.000us         0.00%       0.000us       0.000us     772.895us         4.33%     772.895us     772.895us             1
void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...         0.00%       0.000us         0.00%       0.000us       0.000us     772.894us         4.33%     772.894us     154.579us             5
                     aten::_efficient_attention_forward         0.42%      63.579us         0.97%     146.500us      12.208us     622.111us         3.48%     622.111us      51.843us            12
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us     622.111us         3.48%     622.111us      51.843us            12
                                             aten::gelu         0.47%      71.909us         0.79%     119.134us       5.957us     312.349us         1.75%     312.349us      15.617us            20
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     309.373us         1.73%     309.373us      16.283us            19
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 15.165ms
Self CUDA time total: 17.869ms

-- top ops by CPU self time (host work, includes async launch) --
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  cudaDeviceSynchronize        49.45%       7.500ms        49.45%       7.500ms       3.750ms       0.000us         0.00%       0.000us       0.000us             2
                                  cudaStreamSynchronize        12.40%       1.880ms        12.40%       1.880ms       1.880ms       0.000us         0.00%       0.000us       0.000us             1
                                       cudaLaunchKernel         5.10%     773.529us         5.10%     773.529us       2.366us       0.000us         0.00%       0.000us       0.000us           327
                                Activity Buffer Request         4.13%     626.656us         4.13%     626.656us     626.656us     171.072us         0.96%     171.072us     171.072us             1
                                            aten::addmm         3.36%     510.132us         4.66%     706.564us      13.588us       6.969ms        39.00%       6.969ms     134.027us            52
                                            aten::empty         1.62%     245.871us         1.62%     245.871us       1.518us       0.000us         0.00%       0.000us       0.000us           162
                                              aten::mul         1.56%     236.896us         2.08%     315.718us       8.308us     178.590us         1.00%     178.590us       4.700us            38
                                              aten::add         1.44%     219.033us         1.96%     297.977us       7.640us      66.752us         0.37%      66.752us       1.712us            39
                                aten::cudnn_convolution         1.42%     215.868us         7.05%       1.069ms     133.590us       5.316ms        29.75%       5.487ms     685.847us             8
                                              aten::sub         1.29%     195.357us         1.47%     223.160us      17.166us      11.616us         0.07%      11.616us       0.894us            13
                                            aten::copy_         1.24%     187.352us        14.09%       2.137ms      97.145us     130.496us         0.73%     130.496us       5.932us            22
                                            aten::fill_         0.85%     129.464us         1.09%     166.045us      10.378us      41.663us         0.23%      41.663us       2.604us            16
                                         cuLaunchKernel         0.85%     128.424us         0.85%     128.424us       2.140us       0.000us         0.00%       0.000us       0.000us            60
                                aten::native_layer_norm         0.82%     123.642us         2.24%     340.084us      13.080us      97.887us         0.55%     102.943us       3.959us            26
                                           aten::linear         0.81%     123.087us         8.29%       1.257ms      19.644us       0.000us         0.00%       7.062ms     110.347us            64
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 15.165ms
Self CUDA time total: 17.869ms


Reminder: LSTM + classifier head are untrained; values not meaningful.
```
