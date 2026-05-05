# SSeRiouSS demo

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
--device {cuda,cpu,auto}   default: cuda
--batch-size N             default: 32
--step SECONDS             sliding window step, default: 2.5
--runs N --warmup N        single-window latency bench, default: 50 / 3
--no-full-file             skip full-file pass
--profile                  print torch.profiler top ops
```

Note: WavLM_BASE is pretrained, but the LSTM and classifier head are random — output values are not meaningful, only timings are.
