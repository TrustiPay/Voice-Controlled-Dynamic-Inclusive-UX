from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_MODEL: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    return _MODEL


def transcribe_pcm16k(pcm_bytes: bytes) -> str:
    """
    Transcribe 16 kHz mono PCM s16le audio bytes into text.
    """
    if not pcm_bytes:
        return ""

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return ""

    model = _get_model()
    segments, _info = model.transcribe(audio, language="en", beam_size=1)
    transcript = "".join(segment.text for segment in segments).strip()
    return transcript
