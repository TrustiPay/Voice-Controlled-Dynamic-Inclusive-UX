from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

DATA_DIR = Path(__file__).parent / "data"
CONTACTS_PATH = DATA_DIR / "contacts.json"
LEDGER_PATH = DATA_DIR / "ledger.json"

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
    query_norm = query.strip().lower()
    best: Optional[Dict] = None
    for contact in load_contacts():
        label = contact.get("label", "").lower()
        if query_norm in label:
            if best is None or len(label) < len(best.get("label", "")):
                best = contact
    return best


def get_ledger() -> List[Dict]:
    ledger = _read_json(LEDGER_PATH)
    return list(reversed(ledger))


def append_ledger_entry(to_contact_id: str, to_label: str, amount_lkr: int, note: Optional[str]) -> Dict:
    if amount_lkr <= 0:
        raise ValueError("Amount must be positive")
    entry = {
        "id": f"t_{uuid4().hex[:8]}",
        "type": "transfer",
        "to_contact_id": to_contact_id,
        "to_label": to_label,
        "amount_lkr": amount_lkr,
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": f"Sent to {to_label}",
    }
    ledger = _read_json(LEDGER_PATH)
    ledger.append(entry)
    _write_json(LEDGER_PATH, ledger)
    return entry
