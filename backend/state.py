from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

Steps = str


@dataclass
class SessionState:
    """
    Conversation/session state maintained per WebSocket connection.
    """

    user_id: str = "anon"
    user_name: str = "Guest"
    language: str = "en"

    step: Steps = "idle"
    intent: Optional[str] = None
    recipient_label: Optional[str] = None
    recipient_contact_id: Optional[str] = None
    amount_lkr: Optional[int] = None
    note: Optional[str] = None
    draft_summary: Optional[str] = None
    pending_transfer: Optional[Dict[str, Any]] = None

    awaiting_biometric: bool = False

    audio_config: Dict[str, Any] = field(default_factory=dict)
    bytes_received: int = 0

    def reset(self, user_id: str, user_name: str, language: str) -> None:
        """
        Reset the state for a new session while keeping per-connection counters clean.
        """
        self.user_id = user_id
        self.user_name = user_name
        self.language = language or "en"

        self.step = "idle"
        self.intent = None
        self.recipient_label = None
        self.recipient_contact_id = None
        self.amount_lkr = None
        self.note = None
        self.draft_summary = None
        self.pending_transfer = None

        self.awaiting_biometric = False

        self.audio_config = {}
        self.bytes_received = 0

    def patch(self, updates: Dict[str, Any]) -> None:
        """
        Apply a partial update to known fields only.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def record_audio_chunk(self, size: int) -> None:
        self.bytes_received += size
