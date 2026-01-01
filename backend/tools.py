from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
CONTACTS_PATH = BASE_DIR / "data" / "contacts.json"
LEDGER_PATH = BASE_DIR / "data" / "ledger.json"


def load_contacts() -> List[Dict]:
    if not CONTACTS_PATH.exists():
        return []
    with CONTACTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9+ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def search_contact(query: str) -> Optional[Dict]:
    """
    Fuzzy-ish search: substring, token overlap, and prefix scoring.
    Keeps it deterministic without extra deps.
    """
    norm_query = _normalize(query)
    if not norm_query:
        return None
    query_tokens = set(norm_query.split())

    best: Optional[Dict] = None
    best_score = 0.0

    for c in load_contacts():
        label_raw = c.get("label") or ""
        label = _normalize(label_raw)
        label_tokens = set(label.split())

        score = 0.0
        if norm_query in label:
            score += 3.0
        if label.startswith(norm_query) or norm_query.startswith(label):
            score += 1.5

        token_overlap = len(query_tokens & label_tokens)
        score += token_overlap * 1.0

        # Light phone match support
        phone = _normalize(c.get("phone") or "")
        if phone and phone.endswith(norm_query):
            score += 1.0

        if score > best_score or (score == best_score and best and len(label) < len(_normalize(best.get("label") or ""))):
            best_score = score
            best = c

    return best if best_score >= 1.0 else None


def append_ledger_entry(entry: Dict) -> None:
    ledger: List[Dict] = []
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            ledger = json.load(f)
    ledger.append(entry)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def get_ledger() -> List[Dict]:
    if not LEDGER_PATH.exists():
        return []
    with LEDGER_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    items = list(items)
    items.reverse()
    return items


def make_transfer_entry(
    to_contact_id: str, to_label: str, amount_lkr: int, note: Optional[str]
) -> Dict:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": f"t_{uuid.uuid4().hex[:8]}",
        "type": "transfer",
        "to_contact_id": to_contact_id,
        "to_label": to_label,
        "amount_lkr": amount_lkr,
        "note": note,
        "timestamp": now,
        "title": f"Sent to {to_label}",
    }
