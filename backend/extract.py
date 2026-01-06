from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from heuristics import (
    detect_transfer_intent,
    extract_amount,
    extract_note,
    extract_recipient_hint,
    is_affirmative,
)


@dataclass
class SlotResult:
    intent: str = "unknown"  # transfer|history|help|unknown
    recipient_label: Optional[str] = None
    amount_lkr: Optional[int] = None
    note: Optional[str] = None
    confirm: bool = False


def heuristic_extract(text: str) -> SlotResult:
    """
    Lightweight extraction with regex/keywords. Keeps labels as-heard.
    """
    intent = "history" if "history" in text.lower() else "unknown"

    recipient = extract_recipient_hint(text)
    amount = extract_amount(text)
    note = extract_note(text)
    confirm = is_affirmative(text)

    if detect_transfer_intent(text) or recipient or amount or note:
        intent = "transfer" if intent != "history" else "history"

    return SlotResult(
        intent=intent,
        recipient_label=recipient,
        amount_lkr=amount,
        note=note,
        confirm=confirm,
    )


def llm_extract(text: str) -> SlotResult:
    """
    Placeholder for an LLM extractor (Ollama). For now, reuse heuristics.
    """
    return heuristic_extract(text)


def extract_slots(text: str, mode: str = "heuristic") -> SlotResult:
    if mode == "llm":
        return llm_extract(text)
    return heuristic_extract(text)
