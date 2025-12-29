from __future__ import annotations

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from state import SessionState
from tools import append_ledger_entry, get_ledger, search_contact

logger = logging.getLogger(__name__)

ALLOWED_SCREENS = {"home", "transfer", "confirm", "success", "history"}
ALLOWED_FIELDS = {"recipient", "amount", "note"}
ALLOWED_ACTION_TYPES = {"NAVIGATE", "SET_FIELD", "SHOW_CONFIRM", "PROMPT_BIOMETRIC", "SHOW_TOAST", "SHOW_HISTORY"}
ALLOWED_STATE_PATCH = {"recipient_label", "recipient_contact_id", "amount_lkr", "note", "intent", "step", "draft_summary"}
ALLOWED_TOOLS = {"search_contact", "prepare_transfer", "execute_transfer", "get_history"}


class UIAction(BaseModel):
    type: str
    screen: Optional[str] = None
    field: Optional[str] = None
    value: Any = None
    summary: Optional[str] = None
    message: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "UIAction":
        if self.type not in ALLOWED_ACTION_TYPES:
            raise ValueError(f"Unsupported action type: {self.type}")
        if self.type == "NAVIGATE":
            if self.screen not in ALLOWED_SCREENS:
                raise ValueError("NAVIGATE requires valid screen")
        if self.type == "SET_FIELD":
            if self.field not in ALLOWED_FIELDS:
                raise ValueError("SET_FIELD requires valid field")
        if self.type == "SHOW_CONFIRM" and not self.summary:
            raise ValueError("SHOW_CONFIRM requires summary")
        if self.type == "SHOW_TOAST" and not self.message:
            raise ValueError("SHOW_TOAST requires message")
        if self.type == "SHOW_HISTORY" and not isinstance(self.items, list):
            raise ValueError("SHOW_HISTORY requires items list")
        return self


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def check_tool_name(cls, v: str) -> str:
        if v not in ALLOWED_TOOLS:
            raise ValueError(f"Unsupported tool: {v}")
        return v


class AgentDecision(BaseModel):
    say: str
    ui_actions: List[UIAction] = Field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    state_patch: Optional[Dict[str, Any]] = None
    end_session: Optional[bool] = False


def _state_summary(state: SessionState) -> Dict[str, Any]:
    return {
        "step": state.step,
        "intent": state.intent,
        "recipient_label": state.recipient_label,
        "recipient_contact_id": state.recipient_contact_id,
        "amount_lkr": state.amount_lkr,
        "note": state.note,
        "draft_summary": state.draft_summary,
    }


SYSTEM_PROMPT = """
You are TrustiPay's deterministic voice agent. Output ONLY JSON matching the schema:
{ "say": str, "ui_actions": [UIAction...], "tool_call": {...}|null, "state_patch": {...}|null, "end_session": bool|null }

Rules:
- JSON only, no markdown, no trailing text.
- Ask one question at a time.
- Use tools instead of guessing contacts or ledger.
- Never execute transfers without biometric approval. You may prompt biometric but the backend enforces it.
- Allowed actions: NAVIGATE, SET_FIELD(recipient|amount|note), SHOW_CONFIRM, PROMPT_BIOMETRIC, SHOW_TOAST, SHOW_HISTORY.
- If information is missing, ask for it in 'say'.
- Keep responses brief and polite.
- Do not invent contacts; use search_contact.
""".strip()


def _build_user_prompt(state: SessionState, user_text: Optional[str], last_tool_result: Optional[Dict[str, Any]], recent_messages: List[Dict[str, str]]) -> str:
    payload = {
        "user_name": state.user_name,
        "state": _state_summary(state),
        "recent_messages": recent_messages,
        "user_text": user_text or "",
        "last_tool_result": last_tool_result,
        "safety": {
            "biometric_required": True,
            "do_not_execute_transfer_directly": True,
        },
    }
    return json.dumps(payload)


def _repair_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class LlmAgent:
    def __init__(self, model: str = "mistral"):
        self.model_name = model
        self.llm = ChatOllama(model=model, temperature=0.1)

    def decide(
        self,
        state: SessionState,
        user_text: Optional[str],
        last_tool_result: Optional[Dict[str, Any]],
        recent_messages: List[Dict[str, str]],
    ) -> AgentDecision:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state, user_text, last_tool_result, recent_messages)},
        ]
        raw = self.llm.invoke(messages).content
        decision = self._parse_decision(raw)
        if decision:
            return decision
        logger.warning("Failed to parse agent output, falling back. Raw: %s", raw)
        return AgentDecision(say="Sorry, I did not get that. Please repeat.", ui_actions=[], tool_call=None, state_patch=None)

    def _parse_decision(self, text: str) -> Optional[AgentDecision]:
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = _repair_json(text)
        if parsed is None:
            return None
        try:
            return AgentDecision.model_validate(parsed)
        except ValidationError as exc:
            logger.warning("Decision validation failed: %s", exc)
            return None


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
        {"type": "NAVIGATE", "screen": "success"},
        {"type": "SHOW_TOAST", "message": "Payment sent"},
        {"type": "SHOW_HISTORY", "items": get_ledger()},
    ]
    say = f"Done. I sent {entry['amount_lkr']} LKR to {entry['to_label']}."
    return {"say": say, "ui_actions": actions}


def apply_state_patch(state: SessionState, patch: Optional[Dict[str, Any]]) -> None:
    if not patch:
        return
    for key, value in patch.items():
        if key in ALLOWED_STATE_PATCH and hasattr(state, key):
            setattr(state, key, value)


def execute_tool_call(state: SessionState, tool_call: ToolCall) -> Dict[str, Any]:
    """
    Execute a validated tool call with safety checks.
    """
    name = tool_call.name
    args = tool_call.args or {}

    if name == "search_contact":
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "empty_query"}
        contact = search_contact(query)
        if contact:
            state.recipient_label = contact["label"]
            state.recipient_contact_id = contact["id"]
            return {"found": True, "contact": contact}
        return {"found": False}

    if name == "get_history":
        return {"history": get_ledger()}

    if name == "prepare_transfer":
        if not state.recipient_contact_id or not state.amount_lkr:
            return {"error": "missing_slots"}
        summary = f"Send {state.amount_lkr} LKR to {state.recipient_label}"
        if state.note:
            summary += f" with note '{state.note}'"
        draft_id = f"draft-{uuid4().hex[:8]}"
        state.draft_summary = summary
        return {"draft_id": draft_id, "summary": summary}

    if name == "execute_transfer":
        return {"error": "biometric_required"}

    return {"error": "unsupported"}
