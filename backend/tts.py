from __future__ import annotations

import io
import math
import wave
from typing import Tuple

SAMPLE_RATE = 16_000


def synthesize(text: str, duration: float = 0.6) -> Tuple[bytes, int]:
    """
    Placeholder TTS that generates a short sine-wave tone.
    Returns WAV bytes and the sample rate.
    """
    frames = int(duration * SAMPLE_RATE)
    amplitude = 0.15
    frequency = 440.0  # A4

    data = bytearray()
    for i in range(frames):
        sample = amplitude * math.sin(2 * math.pi * frequency * (i / SAMPLE_RATE))
        # 16-bit signed PCM
        int_sample = int(sample * 32767)
        data.extend(int_sample.to_bytes(2, byteorder="little", signed=True))

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(bytes(data))
        return buffer.getvalue(), SAMPLE_RATE
