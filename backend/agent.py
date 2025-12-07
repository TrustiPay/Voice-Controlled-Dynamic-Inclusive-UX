from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from state import SessionState


@dataclass
class AgentResult:
    say: str
    ui_actions: List[Dict[str, Any]]
    tool_call: Optional[Dict[str, Any]] = None
    state_patch: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "say": self.say,
            "ui_actions": self.ui_actions,
            "tool_call": self.tool_call,
            "state_patch": self.state_patch,
        }


def _greeting(state: SessionState) -> AgentResult:
    text = f"Hello {state.user_name}, I can help you make a transfer. Tell me who to pay and how much."
    actions = [{"type": "NAVIGATE", "screen": "home"}]
    return AgentResult(say=text, ui_actions=actions, tool_call=None, state_patch=None)


def run_agent(event: str, state: SessionState, user_text: Optional[str] = None) -> AgentResult:
    """
    Deterministic placeholder agent. Keeps the JSON structure stable for future LLM use.
    """
    if event == "start":
        return _greeting(state)

    if user_text:
        say = f"I heard: {user_text}. I'll soon guide you through recipient, amount, and note."
    else:
        say = "I'm listening for your instructions."

    return AgentResult(say=say, ui_actions=[], tool_call=None, state_patch=None)
