from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, model_validator

from heuristics import is_affirmative, is_skip_note
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


class ExtractedSlots(BaseModel):
    intent: Literal["transfer", "history", "unknown"] = "unknown"
    recipient_label: Optional[str] = None
    amount_lkr: Optional[int] = None
    note: Optional[str] = None
    confirm: bool = False

    @model_validator(mode="after")
    def normalize(self) -> "ExtractedSlots":
        # Keep the extractor schema predictable and drop unusable amounts
        if isinstance(self.amount_lkr, str):
            try:
                self.amount_lkr = int(self.amount_lkr.replace(",", "").strip())
            except Exception:
                self.amount_lkr = None
        if self.amount_lkr is not None and self.amount_lkr <= 0:
            self.amount_lkr = None
        if self.recipient_label:
            self.recipient_label = self.recipient_label.strip()
        if self.note is not None and isinstance(self.note, str):
            self.note = self.note.strip()
        return self


class TrustiAgent:
    def __init__(self, model: str = "mistral") -> None:
        # Stage 1: constrained extractor
        self.extractor = ChatOllama(
            model=model,
            temperature=0.1,
            top_p=0.2,
            repeat_penalty=1.1,
            format="json",
        )
        # Repair-only formatter to salvage near-miss JSON
        self.repair_llm = ChatOllama(
            model=model,
            temperature=0.0,
            top_p=0.2,
            repeat_penalty=1.15,
            format="json",
        )
        self.max_tool_loops = 2

    def _build_extractor_messages(
        self, state: ConversationState, user_text: str
    ) -> List[Any]:
        schema = '{"intent":"transfer|history|unknown","recipient_label":null|string,"amount_lkr":null|int,"note":null|string,"confirm":true|false}'
        context = (
            f"Known slots -> recipient_label: {state.recipient_label}, "
            f"contact_id: {state.recipient_contact_id}, amount_lkr: {state.amount_lkr}, "
            f"note: {state.note}, awaiting_biometric: {state.awaiting_biometric}."
        )
        system_prompt = f"""
You are TrustiPay intent extractor.
Return ONLY JSON. No markdown. No explanations. Do not wrap in code fences.
Schema: {schema}
Rules:
- Use integers for amount_lkr; strip commas; if missing set null.
- Never invent contact IDs; keep recipient_label exactly as heard.
- confirm=true only when the user explicitly approves or says yes.
- If unsure, keep fields null and set intent="unknown".
Example: {{"intent":"transfer","recipient_label":"Kevin at work","amount_lkr":4000,"note":null,"confirm":false}}
If the user asks for history, set intent="history".
Otherwise, stick to the schema and avoid extra keys or comments.
{context}
"""
        return [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_text),
        ]

    def _parse_extraction(self, raw: str) -> Optional[ExtractedSlots]:
        def try_parse(text: str) -> Optional[ExtractedSlots]:
            try:
                data = json.loads(text)
                return ExtractedSlots.parse_obj(data)
            except (json.JSONDecodeError, ValidationError):
                return None

        parsed = try_parse(raw)
        if parsed:
            return parsed

        if "{" in raw and "}" in raw:
            candidate = raw[raw.find("{") : raw.rfind("}") + 1]
            parsed = try_parse(candidate)
            if parsed:
                return parsed
        return None

    def _repair_extraction(self, raw: str) -> Optional[ExtractedSlots]:
        schema = '{"intent":"transfer|history|unknown","recipient_label":null|string,"amount_lkr":null|int,"note":null|string,"confirm":true|false}'
        messages = [
            SystemMessage(
                content=(
                    "You are a JSON formatter. Convert the input into valid JSON "
                    f"matching this schema: {schema}. Output only JSON."
                )
            ),
            HumanMessage(content=f"Fix this to match schema: {raw}"),
        ]
        repaired = self.repair_llm.invoke(messages).content
        return self._parse_extraction(repaired)

    def _extract_slots(
        self, state: ConversationState, user_text: str
    ) -> ExtractedSlots:
        messages = self._build_extractor_messages(state, user_text)
        response = self.extractor.invoke(messages)
        raw = response.content
        parsed = self._parse_extraction(raw)
        if parsed:
            return parsed
        repaired = self._repair_extraction(raw)
        if repaired:
            return repaired
        logger.warning("Falling back due to unparseable extractor output: %s", raw)
        return ExtractedSlots()

    def _respond_to_tool_result(
        self, state: ConversationState, last_tool: Dict[str, Any]
    ) -> AgentDecision:
        tool = last_tool.get("tool")
        state_patch: Dict[str, Any] = {}

        if tool == "search_contact":
            result = last_tool.get("result")
            if result:
                state.recipient_contact_id = result.get("id")
                state.recipient_label = result.get("label") or state.recipient_label
                state.step = "collect_amount"
                state_patch.update(
                    {
                        "recipient_contact_id": state.recipient_contact_id,
                        "recipient_label": state.recipient_label,
                        "step": state.step,
                    }
                )
                actions = [
                    UIAction(type="NAVIGATE", screen="transfer"),
                    UIAction(
                        type="SET_FIELD", field="recipient", value=state.recipient_label
                    ),
                ]
                say = f"Found {state.recipient_label}. How much should I send?"
            else:
                state.recipient_contact_id = None
                state.step = "collect_recipient"
                state_patch.update(
                    {"recipient_contact_id": None, "step": state.step}
                )
                actions = [
                    UIAction(type="SHOW_TOAST", message="Contact not found"),
                    UIAction(type="NAVIGATE", screen="transfer"),
                ]
                say = "I could not find that contact. Who should I send to?"
            return AgentDecision(
                say=say, ui_actions=actions, state_patch=state_patch or None
            )

        if tool == "get_history":
            items = last_tool.get("items") or []
            state.intent = "history"
            state.step = "history_shown"
            state_patch.update({"intent": "history", "step": state.step})
            actions = [UIAction(type="SHOW_HISTORY", items=items)]
            say = "Here is your recent history."
            return AgentDecision(
                say=say, ui_actions=actions, state_patch=state_patch or None
            )

        if tool == "prepare_transfer":
            if last_tool.get("error"):
                state.awaiting_biometric = False
                state.step = "collect_recipient"
                state_patch.update(
                    {"awaiting_biometric": False, "step": state.step}
                )
                say = "I need a recipient and amount before preparing the transfer."
                actions = [UIAction(type="NAVIGATE", screen="transfer")]
                return AgentDecision(
                    say=say, ui_actions=actions, state_patch=state_patch or None
                )
            summary = last_tool.get("summary") or state.build_summary()
            state.awaiting_biometric = True
            state.draft_summary = summary
            state.step = "awaiting_biometric"
            state_patch.update(
                {
                    "awaiting_biometric": True,
                    "draft_summary": summary,
                    "step": state.step,
                }
            )
            actions = [
                UIAction(type="SHOW_CONFIRM", summary=summary),
                UIAction(type="PROMPT_BIOMETRIC"),
            ]
            say = f"{summary} When you're ready, approve with fingerprint."
            return AgentDecision(
                say=say, ui_actions=actions, state_patch=state_patch or None
            )

        return AgentDecision(
            say="Sorry, I couldn't handle that request safely.",
            ui_actions=[],
            state_patch=state_patch or None,
        )

    def _plan_history(
        self, state: ConversationState, slots: Optional[ExtractedSlots]
    ) -> AgentDecision:
        state.intent = "history"
        state.step = "history_requested"
        state_patch = {"intent": "history", "step": state.step}
        return AgentDecision(
            say="Fetching your recent payments.",
            ui_actions=[],
            tool_call=ToolCall(name="get_history", args={}),
            state_patch=state_patch,
        )

    def _plan_transfer(
        self,
        state: ConversationState,
        slots: Optional[ExtractedSlots],
        user_text: Optional[str],
    ) -> AgentDecision:
        state_patch: Dict[str, Any] = {}
        actions: List[UIAction] = []
        user_confirmed = False

        if slots:
            if slots.intent == "transfer":
                state.intent = "transfer"
                state_patch["intent"] = "transfer"
            if slots.recipient_label:
                state.recipient_label = slots.recipient_label
                # Reset contact_id until search confirms
                state.recipient_contact_id = None
                state_patch.update(
                    {
                        "recipient_label": state.recipient_label,
                        "recipient_contact_id": None,
                    }
                )
            if slots.amount_lkr:
                state.amount_lkr = int(slots.amount_lkr)
                state_patch["amount_lkr"] = state.amount_lkr
            if slots.note is not None:
                state.note = slots.note or None
                state_patch["note"] = state.note
            if slots.confirm:
                user_confirmed = True

        if user_text and is_affirmative(user_text):
            user_confirmed = True
        if user_text and is_skip_note(user_text):
            state.note = None
            state.pending_note = "skipped"
            state_patch.update({"note": None, "pending_note": "skipped"})

        if state.intent is None and (slots is None or slots.intent == "unknown"):
            return AgentDecision(
                say="I can help send a payment or show your history. What would you like to do?",
                ui_actions=[UIAction(type="NAVIGATE", screen="home")],
                state_patch=state_patch or None,
            )

        if state.awaiting_biometric:
            return AgentDecision(
                say="Waiting for fingerprint approval to finish the transfer.",
                ui_actions=[UIAction(type="PROMPT_BIOMETRIC")],
                state_patch=state_patch or None,
            )

        if state.intent != "transfer":
            state.intent = "transfer"
            state_patch["intent"] = "transfer"

        if state.step == "idle":
            state.step = "collect_recipient"
            state_patch["step"] = state.step

        if not state.recipient_label:
            state.step = "collect_recipient"
            state_patch["step"] = state.step
            actions.append(UIAction(type="NAVIGATE", screen="transfer"))
            return AgentDecision(
                say="Who should I send to?",
                ui_actions=actions,
                state_patch=state_patch or None,
            )

        if not state.recipient_contact_id:
            state.step = "resolving_recipient"
            state_patch["step"] = state.step
            actions.extend(
                [
                    UIAction(type="NAVIGATE", screen="transfer"),
                    UIAction(
                        type="SET_FIELD",
                        field="recipient",
                        value=state.recipient_label,
                    ),
                ]
            )
            return AgentDecision(
                say=f"Looking up {state.recipient_label}.",
                ui_actions=actions,
                tool_call=ToolCall(
                    name="search_contact", args={"query": state.recipient_label}
                ),
                state_patch=state_patch or None,
            )

        actions.extend(
            [
                UIAction(type="NAVIGATE", screen="transfer"),
                UIAction(
                    type="SET_FIELD", field="recipient", value=state.recipient_label
                ),
            ]
        )

        if state.amount_lkr is None:
            state.step = "collect_amount"
            state_patch["step"] = state.step
            return AgentDecision(
                say="How much should I send?",
                ui_actions=actions,
                state_patch=state_patch or None,
            )

        actions.append(
            UIAction(type="SET_FIELD", field="amount", value=state.amount_lkr)
        )

        if state.note:
            actions.append(UIAction(type="SET_FIELD", field="note", value=state.note))
            state.pending_note = "captured"
            state_patch["pending_note"] = state.pending_note
        elif state.pending_note not in {"asked", "skipped"}:
            state.pending_note = "asked"
            state.step = "collect_note"
            state_patch.update({"pending_note": "asked", "step": state.step})
            return AgentDecision(
                say="Do you want to add a note?",
                ui_actions=actions,
                state_patch=state_patch or None,
            )

        summary = state.build_summary()
        actions.append(UIAction(type="SHOW_CONFIRM", summary=summary))
        state.step = "confirm"
        state_patch["step"] = state.step

        if user_confirmed:
            return AgentDecision(
                say="Preparing your transfer.",
                ui_actions=actions,
                tool_call=ToolCall(
                    name="prepare_transfer",
                    args={
                        "contact_id": state.recipient_contact_id,
                        "amount_lkr": state.amount_lkr,
                        "note": state.note,
                        "recipient_label": state.recipient_label,
                    },
                ),
                state_patch=state_patch or None,
            )

        return AgentDecision(
            say=f"I can send {state.amount_lkr} LKR to {state.recipient_label}. Should I proceed?",
            ui_actions=actions,
            state_patch=state_patch or None,
        )

    def decide(
        self,
        state: ConversationState,
        user_text: Optional[str],
        last_tool: Optional[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> AgentDecision:
        if last_tool:
            return self._respond_to_tool_result(state, last_tool)

        slots: Optional[ExtractedSlots] = None
        if user_text:
            slots = self._extract_slots(state, user_text)

        if slots and slots.intent == "history":
            return self._plan_history(state, slots)

        return self._plan_transfer(state, slots, user_text)


agent = TrustiAgent()
