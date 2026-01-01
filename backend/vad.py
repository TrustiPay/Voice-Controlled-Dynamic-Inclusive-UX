from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger("trustipay.vad")


class VADSegmenter:
    """
    Streaming VAD wrapper around Silero to yield utterance byte chunks.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 32,
        threshold: float = 0.6,
        max_silence_ms: int = 800,
        start_trigger_ms: int = 64,
        preroll_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * frame_ms / 1000)
        self.threshold = threshold
        self.max_silence_windows = max(1, int(max_silence_ms / frame_ms))
        self.start_trigger_windows = max(1, int(start_trigger_ms / frame_ms))
        self.preroll_max_bytes = int(preroll_ms * sample_rate / 1000) * 2

        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.to("cpu")
        self.model.eval()

        self._float_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._raw_buffer: bytearray = bytearray()
        self._preroll: deque[bytes] = deque()
        self._speech_active = False
        self._start_buildup = 0
        self._silence_windows = 0
        self._current_raw: bytearray = bytearray()

        logger.info(
            "Initialized VADSegmenter (frame_ms=%s, threshold=%.2f)",
            frame_ms,
            threshold,
        )

    def _append_preroll(self, raw_bytes: bytes) -> None:
        self._preroll.append(raw_bytes)
        while sum(len(chunk) for chunk in self._preroll) > self.preroll_max_bytes:
            self._preroll.popleft()

    def _consume_raw_window(self) -> Optional[bytes]:
        needed = self.window_size * 2
        if len(self._raw_buffer) < needed:
            return None
        window = bytes(self._raw_buffer[:needed])
        del self._raw_buffer[:needed]
        return window

    def accept_bytes(self, pcm_bytes: bytes) -> List[bytes]:
        """
        Feed pcm_s16le bytes, return a list of completed utterance byte buffers.
        """
        if not pcm_bytes:
            return []

        self._raw_buffer.extend(pcm_bytes)
        pcm_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self._float_buffer.size == 0:
            self._float_buffer = pcm_chunk
        else:
            self._float_buffer = np.concatenate([self._float_buffer, pcm_chunk])

        utterances: List[bytes] = []

        while self._float_buffer.shape[0] >= self.window_size:
            window_f = self._float_buffer[: self.window_size]
            self._float_buffer = self._float_buffer[self.window_size :]

            raw_window = self._consume_raw_window()
            if raw_window is None:
                break

            window_t = torch.from_numpy(window_f)
            with torch.no_grad():
                prob = float(self.model(window_t, self.sample_rate))

            if not self._speech_active:
                self._append_preroll(raw_window)
                if prob > self.threshold:
                    self._start_buildup += 1
                else:
                    self._start_buildup = 0

                if self._start_buildup >= self.start_trigger_windows:
                    self._speech_active = True
                    self._silence_windows = 0
                    self._current_raw = bytearray(b"".join(self._preroll))
                    self._current_raw.extend(raw_window)
                    self._preroll.clear()
            else:
                self._current_raw.extend(raw_window)
                if prob > self.threshold:
                    self._silence_windows = 0
                else:
                    self._silence_windows += 1
                    if self._silence_windows >= self.max_silence_windows:
                        utterances.append(bytes(self._current_raw))
                        self._speech_active = False
                        self._current_raw = bytearray()
                        self._start_buildup = 0
                        self._silence_windows = 0

        return utterances

    def flush(self) -> List[bytes]:
        """
        Emit any buffered speech on teardown.
        """
        if self._speech_active and self._current_raw:
            utterance = bytes(self._current_raw)
            self._speech_active = False
            self._current_raw = bytearray()
            self._start_buildup = 0
            self._silence_windows = 0
            return [utterance]
        return []
