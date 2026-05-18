"""Minimal pyannote diarization run with NVTX annotations for nsys."""
import functools, sys, torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import clustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from torch.nn.modules.module import (
    register_module_forward_hook, register_module_forward_pre_hook,
)

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "../demo/sample.wav"
assert torch.cuda.is_available(), "CUDA required"

# Push/pop NVTX around every nn.Module forward, named after the module class.
register_module_forward_pre_hook(lambda m, _: torch.cuda.nvtx.range_push(type(m).__name__))
register_module_forward_hook(lambda *_: torch.cuda.nvtx.range_pop())

def nvtx(fn, name):
    @functools.wraps(fn)
    def wrap(*a, **k):
        with torch.cuda.nvtx.range(name): return fn(*a, **k)
    return wrap

# Annotate pipeline phases + pure-CPU clustering (not caught by module hooks).
for name in ("get_segmentations", "get_embeddings", "reconstruct"):
    setattr(SpeakerDiarization, name, nvtx(getattr(SpeakerDiarization, name), name))
for cls in (clustering.VBxClustering, clustering.AgglomerativeClustering):
    cls.__call__ = nvtx(cls.__call__, cls.__name__)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
).to(torch.device("cuda"))

# Pre-warm `min_num_samples`: pyannote computes it via a 13-step binary search
# of dummy forwards on first access (inside get_embeddings). Touching it here
# keeps that one-time overhead out of the trace.
_ = pipeline._embedding.min_num_samples

with torch.cuda.profiler.profile(), torch.cuda.nvtx.range("diarize"):
    out = pipeline(AUDIO)

for turn, _, spk in out.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:7.2f}s - {turn.end:7.2f}s  {spk}")
