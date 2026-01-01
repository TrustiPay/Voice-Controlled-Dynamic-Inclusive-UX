from __future__ import annotations

import re
from typing import Optional

TRANSFER_KEYWORDS = [
    "transfer",
    "send",
    "send money",
    "send rupees",
    "pay",
    "send to",
]


def detect_transfer_intent(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in TRANSFER_KEYWORDS)


def extract_amount(text: str) -> Optional[int]:
    lower = text.lower()
    patterns = [
        r"(?:lkr|rs\.?|rupees?)\s*([0-9][0-9,]*)",
        r"([0-9][0-9,]*)\s*(?:lkr|rs\.?|rupees?)",
        r"(?:send|transfer)\D*([0-9][0-9,]*)",
        r"\b([0-9][0-9,]{1,})\b",
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            raw = m.group(1).replace(",", "").strip()
            try:
                amount = int(raw)
                if 1 <= amount <= 1_000_000:
                    return amount
            except ValueError:
                continue
    return None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1].strip()
    return value.strip()


def extract_note(text: str) -> Optional[str]:
    lower = text.lower()
    patterns = [
        r"add a note saying\s+(.+)",
        r"note(?: saying)?\s+(.+)",
        r"with a note\s+(.+)",
        r"message\s+(.+)",
        r"remark\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            captured = text[m.start(1) : m.end(1)]
            cleaned = _strip_quotes(captured).rstrip(".")
            return cleaned.strip()
    return None


def is_note_request_only(text: str) -> bool:
    """
    Detect utterances that ask to add a note but do not include the content.
    """
    if is_skip_note(text):
        return False
    if extract_note(text):
        return False
    lower = text.lower().strip()
    return bool(
        re.search(r"\badd( a)? note\b", lower)
        or lower in {"note", "note please", "add note"}
    )


def extract_recipient_hint(text: str) -> Optional[str]:
    lower = text.lower()
    m = re.search(r"(?:to|for)\s+([a-z0-9 ]{2,})", lower)
    if m:
        candidate = m.group(1).strip()
        # Avoid generic phrases
        if candidate not in {"my friend", "friend"}:
            return candidate
    return None


def is_skip_note(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in [
            "no note",
            "skip note",
            "no thanks",
            "no need",
            "skip",
            "no",
        ]
    )


def is_affirmative(text: str) -> bool:
    lower = text.lower().strip().rstrip(".!?")
    exact = {
        "yes",
        "yeah",
        "yep",
        "correct",
        "that is correct",
        "ok",
        "okay",
        "sure",
        "affirmative",
        "yes please",
        "please do",
        "sure please",
        "confirm",
        "confirmed",
        "use it",
        "use this",
        "looks good",
        "sounds good",
        "that's fine",
        "thats fine",
        "all good",
        "go ahead",
        "continue",
        "proceed",
    }
    if lower in exact:
        return True

    contains_phrases = [
        "use that",
        "use this note",
        "keep it",
        "keep that",
        "looks good",
        "sounds good",
        "go ahead",
        "proceed",
    ]
    return any(phrase in lower for phrase in contains_phrases)
