from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

DATA_DIR = Path(__file__).parent / "data"
CONTACTS_PATH = DATA_DIR / "contacts.json"
LEDGER_PATH = DATA_DIR / "ledger.json"

# In-memory draft store for the prototype.
_DRAFTS: Dict[str, Dict] = {}


def _read_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_contacts() -> List[Dict]:
    return _read_json(CONTACTS_PATH)


def search_contact(query: str) -> Optional[Dict]:
    if not query:
        return None
    query_lower = query.lower()
    for contact in load_contacts():
        if query_lower in contact.get("label", "").lower():
            return contact
    return None


def get_history() -> List[Dict]:
    return _read_json(LEDGER_PATH)


def prepare_transfer(contact_id: str, amount_lkr: int, note: Optional[str]) -> Dict:
    contacts = {c["id"]: c for c in load_contacts()}
    contact = contacts.get(contact_id)
    if not contact:
        raise ValueError("Unknown contact id")
    if amount_lkr <= 0:
        raise ValueError("Amount must be positive")

    draft_id = f"draft-{uuid4().hex[:8]}"
    draft = {
        "draft_id": draft_id,
        "contact_id": contact_id,
        "to_label": contact["label"],
        "amount_lkr": amount_lkr,
        "note": note,
    }
    _DRAFTS[draft_id] = draft
    summary = f"Pay {contact['label']} LKR {amount_lkr}" + (f" — {note}" if note else "")
    return {"draft_id": draft_id, "summary": summary}


def execute_transfer(draft_id: str) -> Dict:
    draft = _DRAFTS.get(draft_id)
    if not draft:
        raise ValueError("Draft not found")

    ledger = get_history()
    entry = {
        "id": f"t-{uuid4().hex[:8]}",
        "type": "transfer",
        "to_contact_id": draft["contact_id"],
        "to_label": draft["to_label"],
        "amount_lkr": draft["amount_lkr"],
        "note": draft.get("note"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ledger.append(entry)
    _write_json(LEDGER_PATH, ledger)
    _DRAFTS.pop(draft_id, None)
    return entry
