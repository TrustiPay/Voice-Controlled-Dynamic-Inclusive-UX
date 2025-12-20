from __future__ import annotations

from __future__ import annotations

from typing import Any, Dict, List, Optional

from heuristics import (
    detect_transfer_intent,
    extract_amount,
    extract_amount_and_note,
    extract_note,
    extract_recipient,
)
from state import SessionState
from tools import append_ledger_entry, get_ledger, search_contact


def _nav(screen: str) -> Dict[str, Any]:
    return {"type": "NAVIGATE", "screen": screen}


def _set_field(field: str, value: Any) -> Dict[str, Any]:
    return {"type": "SET_FIELD", "field": field, "value": value}


def _show_confirm(summary: str) -> Dict[str, Any]:
    return {"type": "SHOW_CONFIRM", "summary": summary}


def _prompt_biometric() -> Dict[str, Any]:
    return {"type": "PROMPT_BIOMETRIC"}


def _show_toast(message: str) -> Dict[str, Any]:
    return {"type": "SHOW_TOAST", "message": message}


def _build_confirm_summary(state: SessionState) -> str:
    base = f"Send {state.amount_lkr} LKR to {state.recipient_label}"
    if state.note:
        base += f" with note '{state.note}'"
    return base + ". Approve with fingerprint to continue."


def handle_user_text(state: SessionState, text: str) -> Dict[str, Any]:
    """
    Deterministic dialogue manager; returns dict with 'say' and 'ui_actions'.
    """
    say = ""
    actions: List[Dict[str, Any]] = []
    lower = text.lower()

    if state.step == "idle":
        if detect_transfer_intent(lower):
            state.intent = "transfer"
            state.step = "collect_recipient"
            actions.append(_nav("transfer"))
            say = f"Sure {state.user_name}. What is your friend's name as saved in your contacts?"
        else:
            say = "I can help you send money. Say something like: Send 4000 rupees to Kevin at work."
        return {"say": say, "ui_actions": actions}

    if state.step == "collect_recipient":
        candidate = extract_recipient(text) or text
        contact = search_contact(candidate)
        if contact:
            state.recipient_label = contact["label"]
            state.recipient_contact_id = contact["id"]
            state.step = "collect_amount"
            actions.append(_set_field("recipient", contact["label"]))
            say = "Thanks. How much would you like to send?"
        else:
            say = "I couldn't find that contact. Please say the name exactly as saved."
        return {"say": say, "ui_actions": actions}

    if state.step == "collect_amount":
        amount, possible_note = extract_amount_and_note(text)
        if amount:
            state.amount_lkr = amount
            actions.append(_set_field("amount", amount))
            if possible_note:
                state.note = possible_note
                actions.append(_set_field("note", possible_note))
            state.step = "collect_note_optional"
            say = "Got it. Do you want to add a note?"
        else:
            say = "Please tell me the amount in rupees."
        return {"say": say, "ui_actions": actions}

    if state.step == "collect_note_optional":
        if any(kw in lower for kw in ["no note", "skip note", "no", "skip"]):
            state.note = None
        else:
            note = extract_note(text)
            if note:
                state.note = note
                actions.append(_set_field("note", note))
        # proceed to confirm once note decision is made (even if empty)
        if state.recipient_contact_id and state.amount_lkr:
            state.step = "awaiting_biometric"
            state.draft_summary = _build_confirm_summary(state)
            actions.extend(
                [
                    _nav("confirm"),
                    _show_confirm(state.draft_summary),
                    _prompt_biometric(),
                ]
            )
            say = state.draft_summary
        else:
            say = "I need both recipient and amount before confirming."
        return {"say": say, "ui_actions": actions}

    if state.step == "awaiting_biometric":
        say = "Please approve with fingerprint to continue."
        return {"say": say, "ui_actions": []}

    if state.step == "done":
        say = "Transfer completed. Would you like to do another one?"
        return {"say": say, "ui_actions": []}

    say = "I'm ready to help with your transfer."
    return {"say": say, "ui_actions": actions}


def handle_biometric_ok(state: SessionState) -> Dict[str, Any]:
    if state.step != "awaiting_biometric":
        return {"say": "No payment is awaiting approval.", "ui_actions": []}

    if not (state.recipient_contact_id and state.recipient_label and state.amount_lkr):
        return {"say": "Missing details, cannot proceed.", "ui_actions": []}

    entry = append_ledger_entry(
        to_contact_id=state.recipient_contact_id,
        to_label=state.recipient_label,
        amount_lkr=state.amount_lkr,
        note=state.note,
    )
    state.pending_transfer = entry
    state.step = "done"
    actions = [
        _nav("success"),
        _show_toast("Payment sent"),
        {"type": "SHOW_HISTORY", "items": get_ledger()},
    ]
    say = f"Done. I sent {entry['amount_lkr']} LKR to {entry['to_label']}."
    return {"say": say, "ui_actions": actions}
