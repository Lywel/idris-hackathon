"""Pure SSeRiouSS inference: audio -> model -> per-frame scores. CUDA-only."""

import torch
import torch.nn.functional as F
import torchaudio

from pyannote.audio.models.segmentation.SSeRiouSS import SSeRiouSS
from pyannote.audio.core.task import Specifications, Problem, Resolution


SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    return torch.device("cuda")


def load_audio(path):
    """Load file -> mono 16kHz tensor of shape (1, samples)."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
    return torchaudio.functional.resample(wav, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wav


def chunk_waveform(waveform, step_samples):
    """Slice (1, N) into (num_chunks, 1, WINDOW_SAMPLES), zero-padding the tail."""
    n = waveform.shape[1]
    needed = max(WINDOW_SAMPLES, ((n - 1) // step_samples) * step_samples + WINDOW_SAMPLES)
    waveform = F.pad(waveform, (0, max(0, needed - n)))
    return waveform.unfold(1, WINDOW_SAMPLES, step_samples).permute(1, 0, 2).contiguous()


def build_model():
    """SSeRiouSS with WavLM_BASE backbone + 3-speaker frame-level head."""
    model = SSeRiouSS(wav2vec="WAVLM_BASE", wav2vec_frozen=True)
    model.specifications = Specifications(
        problem=Problem.MULTI_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=WINDOW_SECONDS,
        classes=["speaker#1", "speaker#2", "speaker#3"],
    )
    model.build()
    return model.eval()


def infer_waveform(model, waveform, device, batch_size=32, step_seconds=2.5):
    """Sliding-window inference -> stacked scores (num_chunks, frames, classes)."""
    chunks = chunk_waveform(waveform, int(step_seconds * SAMPLE_RATE))
    with torch.inference_mode():
        pieces = [model(b.to(device, non_blocking=True)).cpu()
                  for b in chunks.split(batch_size)]
    return torch.cat(pieces, dim=0)
