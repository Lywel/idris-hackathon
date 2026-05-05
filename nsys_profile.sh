#!/usr/bin/env bash
# Capture the inference region of demo.py with Nsight Systems.
#
# The script wraps `uv run python demo.py ... --nsys` so that
# `cudaProfilerStart/Stop` (toggled by --nsys) bracket the captured region:
# only the actual inference is recorded -- not the WavLM download, model
# build, or argparse.
#
# Output: sseriouss_trace.nsys-rep  (open in `nsys-ui` or upload to a host)
set -euo pipefail

AUDIO=${1:-B047.wav}
shift || true

OUT=${OUT:-sseriouss_trace}

nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "$OUT" \
    uv run python demo.py "$AUDIO" --nsys "$@"

echo
echo "Done. Open in GUI:    nsys-ui ${OUT}.nsys-rep"
echo "Or summarise in CLI:  nsys stats ${OUT}.nsys-rep"
