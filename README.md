# bench

Minimal `pyannote/speaker-diarization-3.1` run with NVTX annotations for Nsight Systems profiling.

## Setup

```sh
# install just (https://github.com/casey/just)
cargo install just            # or: apt install just / brew install just

uv sync
```

Requires CUDA, an HF token (`huggingface-cli login`), and accepted terms for
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

## Run

```sh
just                # diarize ../demo/B047.wav, print turns
just nsys           # profile under nsys -> trace.nsys-rep, opens nsys-ui
```
