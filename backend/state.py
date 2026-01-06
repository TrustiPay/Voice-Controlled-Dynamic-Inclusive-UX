from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConversationState:
    step: str = "idle"
    intent: Optional[str] = None
    recipient_label: Optional[str] = None
    recipient_contact_id: Optional[str] = None
    amount_lkr: Optional[int] = None
    note: Optional[str] = None
    pending_note: Optional[str] = None
    draft_summary: Optional[str] = None
    pending_transfer: Optional[Dict] = None
    user_name: str = "Chinthana"
    draft_id: Optional[str] = None
    awaiting_biometric: bool = False

    def reset(self) -> None:
        self.step = "idle"
        self.intent = None
        self.recipient_label = None
        self.recipient_contact_id = None
        self.amount_lkr = None
        self.note = None
        self.pending_note = None
        self.draft_summary = None
        self.pending_transfer = None
        self.draft_id = None
        self.awaiting_biometric = False

    def build_summary(self) -> str:
        amount = f"{self.amount_lkr} LKR" if self.amount_lkr is not None else "an amount"
        recipient = self.recipient_label or "the contact"
        summary = f"Send {amount} to {recipient}"
        if self.note:
            summary += f" with note '{self.note}'"
        return summary + "."
