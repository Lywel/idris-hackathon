default:
    uv run python bench.py

# Capture both audio lengths under nsys
nsys: (capture "../demo/sample.wav" "short") (capture "../demo/B047.wav" "long")

capture audio tag:
    nsys profile \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --trace=cuda,nvtx,osrt --force-overwrite=true \
        --pytorch=autograd-shapes-nvtx \
        -o trace_{{tag}} \
        uv run python bench.py {{audio}}

# Export sqlite + run summary
summary:
    nsys export --type sqlite --force-overwrite=true -o trace_short.sqlite trace_short.nsys-rep
    nsys export --type sqlite --force-overwrite=true -o trace_long.sqlite trace_long.nsys-rep
    uv run python summary.py | tee summary.txt

ui tag="short":
    nsys-ui trace_{{tag}}.nsys-rep &
