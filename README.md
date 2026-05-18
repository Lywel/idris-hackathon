# bench

Minimal `pyannote/speaker-diarization-community-1` run with NVTX annotations
for Nsight Systems profiling.

Requires CUDA, an HF token (`huggingface-cli login`), and accepted terms for
the pyannote community models.

```sh
uv sync               # install
just                  # diarize ../demo/sample.wav, print turns
just nsys             # capture short + long traces (.nsys-rep)
just summary          # export sqlite + print self-time table + baseline.png
just ui short         # open trace in nsys-ui
```

## Files

- `bench.py` -- pyannote pipeline + NVTX hooks (~40 lines)
- `summary.py` -- sqlite -> self-time table + bar plot (~100 lines)
- `justfile` -- recipes
