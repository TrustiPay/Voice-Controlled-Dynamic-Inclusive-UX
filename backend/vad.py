from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    return _MODEL


@dataclass
class VadConfig:
    sample_rate: int = 16_000
    frame_size: int = 512  # 32 ms at 16 kHz
    start_threshold: float = 0.6
    end_threshold: float = 0.5
    start_trigger_frames: int = 3
    end_trigger_frames: int = 20  # ~640 ms
    pre_roll_ms: int = 300
    min_utterance_ms: int = 600


class SileroVAD:
    """
    Streaming VAD that yields utterance byte chunks using Silero probabilities.
    """

    def __init__(self, config: VadConfig | None = None):
        self.config = config or VadConfig()
        self.model = _load_model()
        self.device = torch.device("cpu")

        frame_ms = (self.config.frame_size / self.config.sample_rate) * 1000
        self.pre_roll_frames = int(self.config.pre_roll_ms / frame_ms)
        self.min_utterance_frames = int(self.config.min_utterance_ms / frame_ms)

        self.pre_frames: Deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self.current_frames: List[bytes] = []
        self.speech_run = 0
        self.silence_run = 0
        self.in_speech = False

    def _frame_prob(self, frame_float: np.ndarray) -> float:
        if frame_float.size == 0:
            return 0.0
        with torch.no_grad():
            tensor = torch.from_numpy(frame_float).to(self.device)
            prob = float(self.model(tensor, self.config.sample_rate).item())
        return prob

    def accept_bytes(self, pcm_bytes: bytes) -> List[bytes]:
        """
        Feed raw PCM (s16le, 16 kHz, mono) bytes. Returns a list of completed utterance byte buffers.
        """
        utterances: List[bytes] = []
        if not pcm_bytes:
            return utterances

        pcm_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        total_frames = int(np.ceil(len(pcm_samples) / self.config.frame_size))

        for idx in range(total_frames):
            start = idx * self.config.frame_size
            end = start + self.config.frame_size
            frame_pcm = pcm_samples[start:end]
            if frame_pcm.size < self.config.frame_size:
                frame_pcm = np.pad(frame_pcm, (0, self.config.frame_size - frame_pcm.size))

            frame_bytes = frame_pcm.tobytes()
            frame_float = frame_pcm.astype(np.float32) / 32768.0
            prob = self._frame_prob(frame_float)

            if not self.in_speech:
                self.pre_frames.append(frame_bytes)
                if prob >= self.config.start_threshold:
                    self.speech_run += 1
                    if self.speech_run >= self.config.start_trigger_frames:
                        self.in_speech = True
                        self.current_frames = list(self.pre_frames)
                        self.current_frames.append(frame_bytes)
                        self.silence_run = 0
                else:
                    self.speech_run = 0
                continue

            # In speech
            self.current_frames.append(frame_bytes)
            if prob < self.config.end_threshold:
                self.silence_run += 1
                if self.silence_run >= self.config.end_trigger_frames:
                    if len(self.current_frames) >= self.min_utterance_frames:
                        utterances.append(b"".join(self.current_frames))
                        logger.debug("Completed utterance with %s frames", len(self.current_frames))
                    self.in_speech = False
                    self.speech_run = 0
                    self.silence_run = 0
                    self.current_frames = []
                    self.pre_frames.clear()
            else:
                self.silence_run = 0

        return utterances
