from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, model_validator

from state import ConversationState

logger = logging.getLogger("trustipay.agent")


class UIAction(BaseModel):
    type: Literal[
        "NAVIGATE",
        "SET_FIELD",
        "SHOW_CONFIRM",
        "PROMPT_BIOMETRIC",
        "SHOW_TOAST",
        "SHOW_HISTORY",
    ]
    screen: Optional[Literal["home", "transfer", "confirm", "success", "history"]] = None
    field: Optional[Literal["recipient", "amount", "note"]] = None
    value: Optional[Any] = None
    summary: Optional[str] = None
    message: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "UIAction":
        if self.type == "NAVIGATE" and not self.screen:
            raise ValueError("NAVIGATE requires screen")
        if self.type == "SET_FIELD" and not self.field:
            raise ValueError("SET_FIELD requires field")
        if self.type == "SHOW_CONFIRM" and not self.summary:
            raise ValueError("SHOW_CONFIRM requires summary")
        if self.type == "SHOW_TOAST" and not self.message:
            raise ValueError("SHOW_TOAST requires message")
        if self.type == "SHOW_HISTORY" and self.items is None:
            raise ValueError("SHOW_HISTORY requires items")
        return self


class ToolCall(BaseModel):
    name: Literal[
        "search_contact",
        "prepare_transfer",
        "get_history",
        "execute_transfer",
    ]
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    say: str = ""
    ui_actions: List[UIAction] = Field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    state_patch: Optional[Dict[str, Any]] = None
    end_session: Optional[bool] = None


class TrustiAgent:
    def __init__(self, model: str = "mistral") -> None:
        # format="json" nudges Ollama to emit strict JSON
        self.llm = ChatOllama(model=model, temperature=0.2, format="json")

    def _prompt(self, state: ConversationState, user_text: Optional[str], last_tool: Optional[Dict[str, Any]], history: List[Dict[str, str]]) -> List[Any]:
        system_rules = """
You are the TrustiPay voice assistant.
Respond ONLY with valid JSON matching this exact schema:
{"say":"string","ui_actions":[{"type":"NAVIGATE","screen":"home|transfer|confirm|success|history"} | {"type":"SET_FIELD","field":"recipient|amount|note","value":...} | {"type":"SHOW_CONFIRM","summary":...} | {"type":"PROMPT_BIOMETRIC"} | {"type":"SHOW_TOAST","message":...} | {"type":"SHOW_HISTORY","items":[...] }], "tool_call":{"name":"search_contact|prepare_transfer|get_history|execute_transfer","args":{...}} | null, "state_patch":{...} | null, "end_session":false}
No markdown, no extra text, no comments. Always include keys even if null.
Examples:
{"say":"Sure John, who should I send to?","ui_actions":[{"type":"NAVIGATE","screen":"transfer"}],"tool_call":null,"state_patch":{"step":"collect_recipient"},"end_session":false}
{"say":"Found Kevin at work. How much?","ui_actions":[{"type":"SET_FIELD","field":"recipient","value":"Kevin at work"}],"tool_call":{"name":"search_contact","args":{"query":"Kevin at work"}},"state_patch":{"recipient_label":"Kevin at work"},"end_session":false}
Allowed tools: search_contact(query), prepare_transfer(contact_id, amount_lkr, note), get_history(), execute_transfer(draft_id).
Safety: never execute transfer without biometric approval; do not invent contacts; ask for missing info one step at a time; require confirmation summary before biometric; keep replies concise.
"""
        ctx = f"""
User name: {state.user_name}
Current step: {state.step}
Intent: {state.intent}
Slots: recipient_label={state.recipient_label}, recipient_contact_id={state.recipient_contact_id}, amount_lkr={state.amount_lkr}, note={state.note}
Pending note: {state.pending_note}
Draft summary: {state.draft_summary}
"""
        msgs: List[Any] = [SystemMessage(content=system_rules + ctx)]
        for m in history[-8:]:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(SystemMessage(content=f"assistant: {m['content']}"))

        if last_tool is not None:
            msgs.append(SystemMessage(content=f"last_tool_result: {json.dumps(last_tool)}"))
        if user_text:
            msgs.append(HumanMessage(content=user_text))
        return msgs

    def decide(
        self,
        state: ConversationState,
        user_text: Optional[str],
        last_tool: Optional[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> AgentDecision:
        messages = self._prompt(state, user_text, last_tool, history)
        response = self.llm.invoke(messages)
        raw = response.content
        decision = self._parse_decision(raw)
        return decision

    def _parse_decision(self, raw: str) -> AgentDecision:
        def try_parse(text: str) -> Optional[AgentDecision]:
            try:
                data = json.loads(text)
                return AgentDecision.parse_obj(data)
            except (json.JSONDecodeError, ValidationError):
                return None

        parsed = try_parse(raw)
        if parsed:
            return parsed

        # Simple repair: extract first/last braces
        if "{" in raw and "}" in raw:
            candidate = raw[raw.find("{") : raw.rfind("}") + 1]
            parsed = try_parse(candidate)
            if parsed:
                return parsed

        logger.warning("Falling back due to unparseable agent output: %s", raw)
        return AgentDecision(
            say="Sorry, I didn't get that. Please say the recipient and amount, for example 'send 4000 to Kevin at work'.",
            ui_actions=[],
        )


agent = TrustiAgent()
