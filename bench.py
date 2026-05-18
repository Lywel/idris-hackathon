"""Minimal pyannote speaker-diarization run, NVTX-annotated for nsys."""
import torch
from pyannote.audio import Pipeline
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


register_module_forward_pre_hook(_push)
register_module_forward_hook(_pop)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1").to(
    torch.device("cuda")
)

with torch.cuda.profiler.profile(), torch.cuda.nvtx.range("diarize"):
    diarization = pipeline(AUDIO)

for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:7.2f}s - {turn.end:7.2f}s  {speaker}")
