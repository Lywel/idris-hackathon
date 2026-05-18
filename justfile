default:
    uv run python bench.py

nsys: && ui
    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        -o trace \
        uv run python bench.py

ui:
    nsys-ui trace.nsys-rep &
