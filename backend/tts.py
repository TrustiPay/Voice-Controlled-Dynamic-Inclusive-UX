from __future__ import annotations

import io
import math
import wave

SAMPLE_RATE = 16_000


def synthesize_wav(text: str, duration: float = 0.6) -> bytes:
    """
    Placeholder TTS: generate a short sine-wave beep and return WAV bytes.
    """
    frames = int(duration * SAMPLE_RATE)
    amplitude = 0.15
    frequency = 880.0  # Higher beep to make it noticeable

    data = bytearray()
    for i in range(frames):
        sample = amplitude * math.sin(2 * math.pi * frequency * (i / SAMPLE_RATE))
        int_sample = int(sample * 32767)  # 16-bit signed PCM
        data.extend(int_sample.to_bytes(2, byteorder="little", signed=True))

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(bytes(data))
        return buffer.getvalue()
