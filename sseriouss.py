"""Pure SSeRiouSS inference: audio -> model -> per-frame scores.

CUDA-only. No timing, no memory tracking, no reporting.
"""

from __future__ import annotations

import torch
import torchaudio

from pyannote.audio.models.segmentation.SSeRiouSS import SSeRiouSS
from pyannote.audio.core.task import Specifications, Problem, Resolution


SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")
    return torch.device("cuda")


# --- audio --------------------------------------------------------------

def load_audio(path: str) -> torch.Tensor:
    """Load file -> mono 16kHz tensor of shape (1, samples)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def chunk_waveform(waveform: torch.Tensor, step_samples: int) -> torch.Tensor:
    """Slice (1, N) into (num_chunks, 1, WINDOW_SAMPLES), zero-padding the tail."""
    total = waveform.shape[1]
    if total < WINDOW_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, WINDOW_SAMPLES - total))
        total = WINDOW_SAMPLES

    num_chunks = max(1, 1 + (total - WINDOW_SAMPLES + step_samples - 1) // step_samples)
    needed = (num_chunks - 1) * step_samples + WINDOW_SAMPLES
    if needed > total:
        waveform = torch.nn.functional.pad(waveform, (0, needed - total))

    chunks = waveform.unfold(dimension=1, size=WINDOW_SAMPLES, step=step_samples)
    return chunks.permute(1, 0, 2).contiguous()


# --- model --------------------------------------------------------------

def build_model() -> SSeRiouSS:
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


# --- inference ----------------------------------------------------------

def infer_batches(model, chunks, device, batch_size=32):
    """Yield (batch_index, scores_cpu) for each batch run through the model."""
    with torch.inference_mode():
        for i in range(0, chunks.shape[0], batch_size):
            batch = chunks[i : i + batch_size].to(device, non_blocking=True)
            scores = model(batch)
            yield i, scores.cpu()


def infer_waveform(model, waveform, device, batch_size=32, step_seconds=2.5):
    """Full sliding-window inference. Returns stacked scores (N, frames, classes)."""
    chunks = chunk_waveform(waveform, int(step_seconds * SAMPLE_RATE))
    pieces = [s for _, s in infer_batches(model, chunks, device, batch_size)]
    return torch.cat(pieces, dim=0)
