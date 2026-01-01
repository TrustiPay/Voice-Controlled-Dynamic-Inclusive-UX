from __future__ import annotations

import io
import logging
import math
import wave
from typing import Final

logger = logging.getLogger("trustipay.tts")

SAMPLE_RATE: Final[int] = 16_000


def synthesize_wav(text: str) -> bytes:
    """
    Generate a short WAV beep so we can validate binary streaming.
    Uses only the standard library; Piper will replace this later.
    """
    duration_seconds = max(1.0, min(3.0, 0.35 + len(text) / 18.0))
    amplitude = 0.25
    base_freq = 880.0
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)

        for i in range(int(duration_seconds * SAMPLE_RATE)):
            t = i / SAMPLE_RATE
            sample = int(amplitude * 32767 * math.sin(2 * math.pi * base_freq * t))
            wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))

    audio_bytes = buffer.getvalue()
    logger.debug("Generated %d bytes of placeholder audio", len(audio_bytes))
    return audio_bytes
