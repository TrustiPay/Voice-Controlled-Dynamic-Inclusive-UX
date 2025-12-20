from __future__ import annotations

import re
from typing import Optional, Tuple

INTENT_KEYWORDS = [
    "transfer",
    "send money",
    "send",
    "pay",
    "send rupees",
    "send to",
    "payment",
]


def detect_transfer_intent(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in INTENT_KEYWORDS)


AMOUNT_PATTERNS = [
    re.compile(r"(?:lkr|rs\.?|rupees?)\s*([0-9][0-9,]*)", re.IGNORECASE),
    re.compile(r"([0-9][0-9,]*)\s*(?:lkr|rs\.?|rupees?)", re.IGNORECASE),
    re.compile(r"\bsend(?:\s+\w+){0,2}\s+([0-9][0-9,]*)", re.IGNORECASE),
    re.compile(r"([0-9][0-9,]*)", re.IGNORECASE),
]


def extract_amount(text: str, min_amount: int = 1, max_amount: int = 1_000_000) -> Optional[int]:
    for pat in AMOUNT_PATTERNS:
        match = pat.search(text)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                value = int(raw)
                if min_amount <= value <= max_amount:
                    return value
            except ValueError:
                continue
    return None


NOTE_PATTERNS = [
    re.compile(r"add a note saying ['\"]?(.+)", re.IGNORECASE),
    re.compile(r"note(?: saying)?[:\s]+['\"]?(.+)", re.IGNORECASE),
    re.compile(r"with a note[:\s]+['\"]?(.+)", re.IGNORECASE),
    re.compile(r"message[:\s]+['\"]?(.+)", re.IGNORECASE),
]


def _clean_note(note: str) -> str:
    note = note.strip()
    if note.startswith(("'", '"')) and note.endswith(("'", '"')) and len(note) >= 2:
        note = note[1:-1]
    return note.rstrip(".!, ").strip()


def extract_note(text: str) -> Optional[str]:
    for pat in NOTE_PATTERNS:
        match = pat.search(text)
        if match:
            return _clean_note(match.group(1))
    # if user says "note thank you for the tickets"
    if "note" in text.lower():
        idx = text.lower().find("note")
        remainder = text[idx + len("note") :]
        if remainder.strip():
            return _clean_note(remainder)
    return None


RECIPIENT_PATTERN = re.compile(r"(?:to|for)\s+([a-zA-Z0-9 ]{2,})")


def extract_recipient(text: str) -> Optional[str]:
    match = RECIPIENT_PATTERN.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    # avoid generic terms
    if candidate.lower() in {"my friend", "friend", "someone"}:
        return None
    return candidate


def extract_amount_and_note(text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Convenience helper to parse both from a single utterance.
    """
    return extract_amount(text), extract_note(text)
