"""Minimal pyannote speaker-diarization run, NVTX-annotated for nsys."""
import functools
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import clustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

#AUDIO = "../demo/B047.wav"
AUDIO = "../demo/sample.wav"

assert torch.cuda.is_available(), "CUDA required"


def _push(m, _inp):
    torch.cuda.nvtx.range_push(type(m).__name__)


def _pop(*_):
    torch.cuda.nvtx.range_pop()


def nvtx_wrap(fn, name):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with torch.cuda.nvtx.range(name):
            return fn(*a, **kw)
    return wrapper


register_module_forward_pre_hook(_push)
register_module_forward_hook(_pop)

# Annotate clustering (CPU code, not caught by nn.Module hooks)
clustering.cluster_vbx = nvtx_wrap(clustering.cluster_vbx, "cluster_vbx")
for cls in (clustering.VBxClustering, clustering.AgglomerativeClustering):
    cls.__call__ = nvtx_wrap(cls.__call__, cls.__name__)

# Annotate top-level pipeline phases
for name in ("get_segmentations", "get_embeddings", "reconstruct"):
    setattr(SpeakerDiarization, name, nvtx_wrap(getattr(SpeakerDiarization, name), name))

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1").to(
    torch.device("cuda")
)

with torch.cuda.profiler.profile(), torch.cuda.nvtx.range("diarize"):
    diarization = pipeline(AUDIO)

for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:7.2f}s - {turn.end:7.2f}s  {speaker}")
