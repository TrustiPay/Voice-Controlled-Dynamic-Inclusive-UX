from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from extract import SlotResult, extract_slots
from heuristics import extract_amount, is_affirmative, is_skip_note


class State(str, enum.Enum):
    IDLE = "IDLE"
    TRANSFER = "TRANSFER"
    HISTORY = "HISTORY"
    HELP = "HELP"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    # Transfer substates
    T_START = "T_START"
    T_COLLECT_RECIPIENT = "T_COLLECT_RECIPIENT"
    T_RESOLVE_CONTACT = "T_RESOLVE_CONTACT"
    T_COLLECT_AMOUNT = "T_COLLECT_AMOUNT"
    T_COLLECT_NOTE = "T_COLLECT_NOTE"
    T_CONFIRM = "T_CONFIRM"
    T_AWAIT_BIOMETRIC = "T_AWAIT_BIOMETRIC"
    T_EXECUTE = "T_EXECUTE"
    T_SUCCESS = "T_SUCCESS"
    T_CANCELLED = "T_CANCELLED"


class EventType(str, enum.Enum):
    START_SESSION = "EVT_START_SESSION"
    USER_TEXT = "EVT_USER_TEXT"
    TOOL_RESULT = "EVT_TOOL_RESULT"
    BIOMETRIC = "EVT_BIOMETRIC"
    CANCEL = "EVT_CANCEL"
    TIMEOUT = "EVT_TIMEOUT"


@dataclass
class Event:
    type: EventType
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    biometric_ok: Optional[bool] = None


@dataclass
class ActionBundle:
    say: str = ""
    ui_actions: List[Dict[str, Any]] = field(default_factory=list)
    tool_call: Optional[Dict[str, Any]] = None
    state_patch: Optional[Dict[str, Any]] = None
    next_state: State = State.IDLE


TEXT = {
    "ask_recipient": {
        "en": "Who should I send to?",
        "si": "මුදල් යවන්නේ කාහටද?",
    },
    "looking_up": {
        "en": "Got it. Let me find that contact.",
        "si": "හරි, මම ඒ සම්බන්ධතා නාමය සොයනවා.",
    },
    "contact_not_found": {
        "en": "I could not find that contact. Please try again.",
        "si": "සම්බන්ධතා නාමය සොයා ගත නොහැකියි. නැවත උත්හස කරන්න.",
    },
    "ask_amount": {
        "en": "How much should I send?",
        "si": "යවන මුදල කොපමණද?",
    },
    "ask_note": {
        "en": "Do you want to add a note?",
        "si": "සටහනක් එකතු කරන්න අවශ්‍යද?",
    },
    "confirm_send": {
        "en": "I can send {amount} LKR to {recipient}. Should I proceed?",
        "si": "රු. {amount} {recipient} වෙත යැවීමට සූදානම්. තොරතුරු නිවැරදිද?",
    },
    "waiting_biometric": {
        "en": "Please approve with fingerprint to continue.",
        "si": "කරුණාකර ඇඟිලි සලකුණ යොදා මුදල් යැවීම තහවුරු කරන්න.",
    },
    "prepare_transfer": {
        "en": "Preparing your transfer.",
        "si": "ඔබේ ගෙවීම සූදානම් කරමින් පවතී.",
    },
    "history_fetch": {
        "en": "Fetching your recent payments.",
        "si": "ඔබගේ මෑත ගෙවීම් ලබා ගනිමින් පවතී.",
    },
    "history_shown": {
        "en": "Here is your recent history.",
        "si": "ඔබගේ මෑත ඉතිහාසය මෙසේය.",
    },
    "cancelled": {
        "en": "Cancelled the transfer.",
        "si": "ගෙවීම අවලංගු කළා.",
    },
}

STATE_SCREENS: Dict[State, str] = {
    State.IDLE: "home",
    State.T_START: "transfer",
    State.T_COLLECT_RECIPIENT: "transfer",
    State.T_RESOLVE_CONTACT: "transfer",
    State.T_COLLECT_AMOUNT: "transfer",
    State.T_COLLECT_NOTE: "transfer",
    State.T_CONFIRM: "confirm",
    State.T_AWAIT_BIOMETRIC: "confirm",
    State.T_EXECUTE: "confirm",
    State.T_SUCCESS: "success",
    State.HISTORY: "history",
    State.T_CANCELLED: "home",
}

ALLOWED_ACTIONS: Dict[State, set] = {
    State.IDLE: {"NAVIGATE", "SHOW_TOAST"},
    State.T_COLLECT_RECIPIENT: {"NAVIGATE", "SET_FIELD", "SHOW_TOAST"},
    State.T_RESOLVE_CONTACT: {"NAVIGATE", "SET_FIELD", "SHOW_TOAST"},
    State.T_COLLECT_AMOUNT: {"NAVIGATE", "SET_FIELD", "SHOW_TOAST"},
    State.T_COLLECT_NOTE: {"NAVIGATE", "SET_FIELD", "SHOW_TOAST"},
    State.T_CONFIRM: {"NAVIGATE", "SET_FIELD", "SHOW_CONFIRM", "SHOW_TOAST"},
    State.T_AWAIT_BIOMETRIC: {"NAVIGATE", "PROMPT_BIOMETRIC", "SHOW_CONFIRM"},
    State.T_EXECUTE: {"NAVIGATE", "SHOW_TOAST", "SHOW_HISTORY"},
    State.T_SUCCESS: {"NAVIGATE", "SHOW_TOAST", "SHOW_HISTORY"},
    State.T_CANCELLED: {"NAVIGATE", "SHOW_TOAST"},
    State.HISTORY: {"NAVIGATE", "SHOW_HISTORY", "SHOW_TOAST"},
}


def _lang(pref: str, detected: Optional[str]) -> str:
    if pref == "auto":
        return (detected or "en").split("-")[0]
    return (pref or "en").split("-")[0]


def _say(key: str, language: str, **kwargs: Any) -> str:
    template = TEXT.get(key, {}).get(language) or TEXT.get(key, {}).get("en") or ""
    return template.format(**kwargs)


def _nav_for_state(state: State) -> Optional[Dict[str, Any]]:
    screen = STATE_SCREENS.get(state)
    if screen:
        return {"type": "NAVIGATE", "screen": screen}
    return None


def validate_ui_actions(state: State, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = ALLOWED_ACTIONS.get(state, set())
    filtered: List[Dict[str, Any]] = []
    for action in actions:
        if action.get("type") in allowed:
            filtered.append(action)
    return filtered


def _summary(conv) -> str:
    amount = f"{conv.amount_lkr} LKR" if conv.amount_lkr is not None else "an amount"
    recipient = conv.recipient_label or "the contact"
    summary = f"Send {amount} to {recipient}"
    if conv.note:
        summary += f" with note '{conv.note}'"
    return summary + "."


def handle_event(state: State, conv, event: Event, extract_mode: str = "heuristic") -> ActionBundle:
    lang = _lang(conv.language, conv.detected_language)
    bundle = ActionBundle(next_state=state, state_patch={"step": state.value})

    def ensure_nav(target_state: State) -> None:
        nav = _nav_for_state(target_state)
        if nav:
            bundle.ui_actions.append(nav)

    if event.type == EventType.START_SESSION:
        conv.intent = None
        conv.step = "idle"
        conv.recipient_label = None
        conv.recipient_contact_id = None
        conv.amount_lkr = None
        conv.note = None
        conv.pending_note = None
        bundle.state_patch = {"step": "idle"}
        bundle.next_state = State.IDLE
        return bundle

    # --- HISTORY path ---
    if state in {State.IDLE, State.HISTORY} and event.type == EventType.USER_TEXT:
        slots = extract_slots(event.text or "", mode=extract_mode)
        if slots.intent == "history":
            bundle.say = _say("history_fetch", lang)
            bundle.tool_call = {"name": "get_history", "args": {}}
            bundle.next_state = State.HISTORY
            ensure_nav(State.HISTORY)
            return bundle

    if state == State.HISTORY and event.type == EventType.TOOL_RESULT:
        if event.tool_name == "get_history":
            bundle.say = _say("history_shown", lang)
            bundle.ui_actions.append({"type": "SHOW_HISTORY", "items": event.tool_result or []})
            ensure_nav(State.HISTORY)
            bundle.next_state = State.HISTORY
            return bundle

    # --- TRANSFER flow ---
    if state == State.IDLE and event.type == EventType.USER_TEXT:
        slots = extract_slots(event.text or "", mode=extract_mode)
        if slots.intent != "transfer":
            bundle.say = _say("ask_recipient", lang)
            ensure_nav(State.T_COLLECT_RECIPIENT)
            bundle.next_state = State.T_COLLECT_RECIPIENT
            return bundle
        # seed slots if present
        conv.recipient_label = slots.recipient_label or conv.recipient_label
        conv.amount_lkr = slots.amount_lkr or conv.amount_lkr
        conv.note = slots.note or conv.note
        bundle.state_patch = {
            "recipient_label": conv.recipient_label,
            "amount_lkr": conv.amount_lkr,
            "note": conv.note,
            "intent": "transfer",
            "step": State.T_START.value,
        }
        bundle.next_state = State.T_START
        ensure_nav(State.T_COLLECT_RECIPIENT)
        bundle.say = _say("ask_recipient", lang) if not conv.recipient_label else _say("looking_up", lang)
        if conv.recipient_label:
            bundle.tool_call = {"name": "search_contact", "args": {"query": conv.recipient_label}}
            bundle.next_state = State.T_RESOLVE_CONTACT
        else:
            bundle.next_state = State.T_COLLECT_RECIPIENT
        return bundle

    if state == State.T_COLLECT_RECIPIENT and event.type == EventType.USER_TEXT:
        slots = extract_slots(event.text or "", mode=extract_mode)
        if slots.recipient_label:
            conv.recipient_label = slots.recipient_label
            bundle.state_patch = {
                "recipient_label": conv.recipient_label,
                "recipient_contact_id": None,
                "step": State.T_RESOLVE_CONTACT.value,
            }
            bundle.say = _say("looking_up", lang)
            bundle.tool_call = {"name": "search_contact", "args": {"query": conv.recipient_label}}
            ensure_nav(State.T_RESOLVE_CONTACT)
            bundle.next_state = State.T_RESOLVE_CONTACT
        else:
            bundle.say = _say("ask_recipient", lang)
            ensure_nav(State.T_COLLECT_RECIPIENT)
            bundle.next_state = State.T_COLLECT_RECIPIENT
        return bundle

    if state == State.T_RESOLVE_CONTACT and event.type == EventType.TOOL_RESULT:
        if event.tool_name == "search_contact":
            result = event.tool_result
            if not result:
                conv.recipient_contact_id = None
                bundle.state_patch = {"recipient_contact_id": None, "step": State.T_COLLECT_RECIPIENT.value}
                bundle.say = _say("contact_not_found", lang)
                ensure_nav(State.T_COLLECT_RECIPIENT)
                bundle.next_state = State.T_COLLECT_RECIPIENT
            else:
                conv.recipient_contact_id = result.get("id")
                conv.recipient_label = result.get("label") or conv.recipient_label
                bundle.state_patch = {
                    "recipient_contact_id": conv.recipient_contact_id,
                    "recipient_label": conv.recipient_label,
                    "step": State.T_COLLECT_AMOUNT.value,
                }
                bundle.ui_actions.extend(
                    [
                        {"type": "SET_FIELD", "field": "recipient", "value": conv.recipient_label},
                    ]
                )
                bundle.say = _say("ask_amount", lang)
                ensure_nav(State.T_COLLECT_AMOUNT)
                bundle.next_state = State.T_COLLECT_AMOUNT
        return bundle

    if state in {State.T_COLLECT_AMOUNT, State.T_COLLECT_RECIPIENT} and event.type == EventType.USER_TEXT:
        slots = extract_slots(event.text or "", mode=extract_mode)
        amt = slots.amount_lkr or extract_amount(event.text or "")
        if amt:
            conv.amount_lkr = int(amt)
            bundle.state_patch = {"amount_lkr": conv.amount_lkr, "step": State.T_COLLECT_NOTE.value}
            bundle.ui_actions.append({"type": "SET_FIELD", "field": "amount", "value": conv.amount_lkr})
            ensure_nav(State.T_COLLECT_NOTE)
            bundle.say = _say("ask_note", lang)
            bundle.next_state = State.T_COLLECT_NOTE
        else:
            bundle.say = _say("ask_amount", lang)
            ensure_nav(State.T_COLLECT_AMOUNT)
            bundle.next_state = State.T_COLLECT_AMOUNT
        return bundle

    if state == State.T_COLLECT_NOTE and event.type == EventType.USER_TEXT:
        text = event.text or ""
        if is_skip_note(text):
            conv.note = None
            conv.pending_note = "skipped"
        else:
            slots = extract_slots(text, mode=extract_mode)
            if slots.note:
                conv.note = slots.note
        bundle.state_patch = {"note": conv.note, "pending_note": conv.pending_note, "step": State.T_CONFIRM.value}
        ensure_nav(State.T_CONFIRM)
        summary = _summary(conv)
        bundle.ui_actions.append({"type": "SHOW_CONFIRM", "summary": summary})
        bundle.say = _say(
            "confirm_send",
            lang,
            amount=conv.amount_lkr if conv.amount_lkr is not None else "0",
            recipient=conv.recipient_label or "",
        )
        bundle.next_state = State.T_CONFIRM
        return bundle

    if state == State.T_CONFIRM and event.type == EventType.USER_TEXT:
        text = event.text or ""
        amt = extract_amount(text)
        if amt:
            conv.amount_lkr = int(amt)
            bundle.state_patch = {"amount_lkr": conv.amount_lkr}
        if is_affirmative(text):
            summary = _summary(conv)
            bundle.say = _say("prepare_transfer", lang)
            bundle.ui_actions.append({"type": "SHOW_CONFIRM", "summary": summary})
            bundle.tool_call = {
                "name": "prepare_transfer",
                "args": {
                    "contact_id": conv.recipient_contact_id,
                    "amount_lkr": conv.amount_lkr,
                    "note": conv.note,
                    "recipient_label": conv.recipient_label,
                },
            }
            ensure_nav(State.T_AWAIT_BIOMETRIC)
            bundle.next_state = State.T_AWAIT_BIOMETRIC
        else:
            summary = _summary(conv)
            bundle.ui_actions.append({"type": "SHOW_CONFIRM", "summary": summary})
            bundle.say = _say(
                "confirm_send",
                lang,
                amount=conv.amount_lkr if conv.amount_lkr is not None else "0",
                recipient=conv.recipient_label or "",
            )
            ensure_nav(State.T_CONFIRM)
            bundle.next_state = State.T_CONFIRM
        return bundle

    if state == State.T_AWAIT_BIOMETRIC and event.type == EventType.TOOL_RESULT:
        if event.tool_name == "prepare_transfer":
            bundle.say = _say("waiting_biometric", lang)
            bundle.ui_actions.append({"type": "PROMPT_BIOMETRIC"})
            ensure_nav(State.T_AWAIT_BIOMETRIC)
            bundle.next_state = State.T_AWAIT_BIOMETRIC
            return bundle

    if state == State.T_AWAIT_BIOMETRIC:
        if event.type == EventType.BIOMETRIC:
            if event.biometric_ok:
                bundle.say = _say("prepare_transfer", lang)
                bundle.tool_call = {"name": "execute_transfer", "args": {"draft_id": conv.draft_id}}
                ensure_nav(State.T_EXECUTE)
                bundle.next_state = State.T_EXECUTE
            else:
                bundle.say = _say("waiting_biometric", lang)
                ensure_nav(State.T_AWAIT_BIOMETRIC)
                bundle.next_state = State.T_AWAIT_BIOMETRIC
            return bundle
        if event.type == EventType.USER_TEXT:
            bundle.say = _say("waiting_biometric", lang)
            ensure_nav(State.T_AWAIT_BIOMETRIC)
            bundle.next_state = State.T_AWAIT_BIOMETRIC
            return bundle

    if state == State.T_EXECUTE and event.type == EventType.TOOL_RESULT:
        if event.tool_name == "execute_transfer":
            ensure_nav(State.T_SUCCESS)
            bundle.ui_actions.extend(
                [
                    {"type": "SHOW_TOAST", "message": "Payment sent"},
                    {"type": "SHOW_HISTORY", "items": event.tool_result.get("ledger", []) if event.tool_result else []},
                ]
            )
            bundle.say = event.tool_result.get("message", "") if event.tool_result else ""
            bundle.next_state = State.T_SUCCESS
        return bundle

    if event.type == EventType.CANCEL:
        bundle.say = _say("cancelled", lang)
        ensure_nav(State.T_CANCELLED)
        bundle.next_state = State.T_CANCELLED
        return bundle

    return bundle
