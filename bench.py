"""Minimal pyannote diarization run with NVTX annotations for nsys."""
import functools, sys, torch
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import clustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from torch.nn.modules.module import (
    register_module_forward_hook, register_module_forward_pre_hook,
)

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "../demo/sample.wav"
assert torch.cuda.is_available(), "CUDA required"

# Push/pop NVTX around every nn.Module forward, named after the module class.
# NOTE: hooks MUST return None -- a non-None return value would replace the
# module's forward args/output. `range_push`/`range_pop` return ints, so we
# can't use lambdas here.
def _push(m, _): torch.cuda.nvtx.range_push(type(m).__name__)
def _pop(*_):    torch.cuda.nvtx.range_pop()
register_module_forward_pre_hook(_push)
register_module_forward_hook(_pop)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
).to(torch.device("cuda"))

# Pre-warm `min_num_samples`: pyannote computes it via a 13-step binary search
# of dummy forwards on first access (inside get_embeddings). Touching it here
# keeps that one-time overhead out of the trace.
_ = pipeline._embedding.min_num_samples


audio = Audio(sample_rate=16000, mono='downmix')
waveform, sample_rate = audio({"audio": AUDIO})

input = { "waveform": waveform, "sample_rate": sample_rate }

pipeline(input)

with torch.cuda.profiler.profile(), torch.cuda.nvtx.range("diarize"):
    out = pipeline(input)

for turn, _, spk in out.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:7.2f}s - {turn.end:7.2f}s  {spk}")
