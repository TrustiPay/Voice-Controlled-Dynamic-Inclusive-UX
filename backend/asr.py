from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
from faster_whisper import WhisperModel  # type: ignore

logger = logging.getLogger("trustipay.asr")

MODEL_DIR = Path(__file__).resolve().parent / "models" / "faster-whisper-small"


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """
    Load faster-whisper model once (small, CPU-friendly) from local snapshot.
    Uses int8 compute for speed on CPU.
    """
    model_path = str(MODEL_DIR)
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    logger.info("Loaded faster-whisper model from %s (int8, cpu)", model_path)
    return model


def transcribe_pcm16k(pcm_bytes: bytes, language: Optional[str] = "en") -> Tuple[str, Optional[str]]:
    """
    Transcribe a mono 16 kHz pcm_s16le buffer to text.
    Returns (transcript, detected_language).
    """
    if not pcm_bytes:
        return "", None

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    model = _get_model()
    lang = None if language in {None, "", "auto"} else language
    segments, _info = model.transcribe(
        audio,
        language=lang,
        beam_size=1,
        best_of=1,
    )
    detected_language = _info.language if lang is None else lang

    texts: List[str] = []
    for seg in segments:
        if seg.text:
            texts.append(seg.text.strip())
    transcript = " ".join(texts).strip()
    logger.info("ASR transcript: %s (lang=%s)", transcript, detected_language)
    return transcript, detected_language
