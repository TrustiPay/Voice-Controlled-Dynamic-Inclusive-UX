from __future__ import annotations

import re
from typing import Optional

ACTION_WORDS = ["send", "transfer", "pay"]
MONEY_WORDS = ["money", "rupee", "rupees", "lkr", "rs", "amount", "cash"]
RECIPIENT_CUES = ["to", "for", "recipient", "contact", "friend"]

ACTION_RE = re.compile(r"\b(send|transfer|pay)\b", re.I)
CURRENCY_RE = re.compile(r"\b(lkr|rs\.?|rupees?)\b", re.I)


def _norm(text: str) -> str:
    """Normalize spacing and casing for heuristic matching."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def detect_transfer_intent(text: str) -> bool:
    """
    Score intent using multiple signals to avoid false positives like "send me the history".
    Require at least two groups among action, money, or recipient cues.
    """
    norm = _norm(text)
    if not norm:
        return False

    if re.search(r"\b(send money|transfer money|make a transfer|pay\s+\w+)\b", norm):
        return True

    score = 0
    if ACTION_RE.search(norm):
        score += 1
    if CURRENCY_RE.search(norm) or any(word in norm for word in MONEY_WORDS):
        score += 1
    if re.search(r"\b(to|for|recipient|contact|friend)\b", norm):
        score += 1

    return score >= 2


def _looks_like_phone(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    if 9 <= len(digits) <= 12 and (digits.startswith("07") or digits.startswith("94")):
        return True
    return False


def _parse_amount_token(token: str) -> Optional[int]:
    cleaned = token.lower().replace(",", "").strip()
    match_k = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", cleaned)
    if match_k:
        value = float(match_k.group(1))
        return int(round(value * 1000))
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        if "." in cleaned:
            try:
                return int(round(float(cleaned)))
            except ValueError:
                return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _validate_amount(raw: str) -> Optional[int]:
    if _looks_like_phone(raw):
        return None
    amt = _parse_amount_token(raw)
    if amt is None:
        return None
    if 1 <= amt <= 1_000_000:
        return amt
    return None


def extract_amount(text: str, *, allow_loose: bool = False) -> Optional[int]:
    """
    Extract an amount only when it appears near currency/action context.
    If allow_loose=True, fall back to any isolated numeric/k token (for amount-specific turns).
    """
    norm = _norm(text)
    if not norm:
        return None

    # Prefer currency markers
    m = re.search(r"(?:lkr|rs\.?|rupees?)\s*([0-9][0-9,]*(?:\.\d+)?\s*k?)", norm)
    if not m:
        m = re.search(r"([0-9][0-9,]*(?:\.\d+)?\s*k?)\s*(?:lkr|rs\.?|rupees?)", norm)
    if m:
        amt = _validate_amount(m.group(1))
        if amt is not None:
            return amt

    # Then: number near an action verb within a short window
    m2 = re.search(r"\b(send|transfer|pay)\b.{0,24}?(\d[\d,]*(?:\.\d+)?\s*k?)\b", norm)
    if m2:
        amt = _validate_amount(m2.group(2))
        if amt is not None:
            return amt

    if allow_loose:
        fallback = re.search(r"\b(\d[\d,]*(?:\.\d+)?\s*k?)\b", norm)
        if fallback:
            amt = _validate_amount(fallback.group(1))
            if amt is not None:
                return amt

    return None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1].strip()
    return value.strip()


def extract_note(text: str) -> Optional[str]:
    """
    Prefer quoted notes; otherwise capture until a connector phrase to avoid greedy grabs.
    """
    if not text:
        return None

    # Quoted notes after note/message markers
    quoted = re.search(r"\b(note|message|remark|memo)\b[^\"']*(['\"])(.+?)\2", text, re.I)
    if quoted:
        return quoted.group(3).strip()

    # Unquoted: take the tail after a marker, but stop at connectors
    tail_match = re.search(r"\b(add (?:a )?note|note|message|remark|memo)\b[.:,-]?\s*(.+)$", text, re.I)
    if tail_match:
        tail = tail_match.group(2).strip()
        tail = re.split(
            r"\b(and then|then|also|after that|and send|and transfer|then send|then transfer)\b",
            tail,
            maxsplit=1,
            flags=re.I,
        )[0]
        tail = _strip_quotes(tail).strip().strip(".")
        return tail or None

    return None


def is_note_request_only(text: str) -> bool:
    """
    Detect utterances that ask to add a note but do not include the content.
    """
    if is_skip_note(text):
        return False
    if extract_note(text):
        return False
    lower = _norm(text)
    return bool(
        re.search(r"\badd( a)? note\b", lower)
        or lower in {"note", "note please", "add note"}
    )


def extract_recipient_hint(text: str) -> Optional[str]:
    lower = text.lower()
    # Strip punctuation that often ends the recipient fragment
    cleaned = re.sub(r"[,.!?]", " ", lower)
    # Capture text after "to|for" until a connector like "with"/"and"/"note"
    m = re.search(r"(?:to|for)\s+([a-z0-9 ']+)", cleaned)
    if m:
        candidate = m.group(1)
        candidate = re.split(r"\b(?:with|including|and|note)\b", candidate, maxsplit=1)[0].strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and candidate not in {"my friend", "friend"}:
            return candidate
    return None


def is_skip_note(text: str) -> bool:
    norm = _norm(text)
    if not norm:
        return False
    # Treat as skip only for short utterances to avoid catching "no, send 4000"
    if len(norm.split()) > 4:
        return False
    return bool(
        re.fullmatch(
            r"(no|nope|skip|skip note|skip the note|no note|no thanks|dont add a note|don't add a note|do not add a note)",
            norm,
        )
    )


def is_affirmative(text: str) -> bool:
    norm = _norm(text).rstrip(".!?")
    if not norm:
        return False
    if len(norm.split()) > 6:
        return False
    return bool(
        re.fullmatch(
            r"(yes|yeah|yep|ok|okay|sure|correct|confirm|confirmed|go ahead|proceed|continue|sounds good|looks good|that is correct|thats fine|that's fine|all good|sure please|please do)",
            norm,
        )
    )
