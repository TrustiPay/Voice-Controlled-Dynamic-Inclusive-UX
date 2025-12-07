from __future__ import annotations

from typing import Optional


def transcribe(audio_bytes: bytes, sample_rate: int = 16_000) -> Optional[str]:
    """
    Placeholder ASR stub. Returns None to indicate no transcription yet.
    Replace with real VAD + Whisper pipeline in milestone C.
    """
    if not audio_bytes:
        return None
    return "ASR stub: transcription not implemented yet."
