from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from starlette.websockets import WebSocketState  # type: ignore

from agent import AgentDecision, agent
from asr import transcribe_pcm16k
from state import ConversationState
from tools import append_ledger_entry, get_ledger, make_transfer_entry, search_contact
from tts import synthesize_wav
from vad import VADSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trustipay.backend")

app = FastAPI(title="TrustiPay Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, bool]:
    return {"ok": True}


@dataclass
class SessionState:
    audio_config: Optional[Dict[str, Any]] = None
    total_audio_bytes: int = 0
    last_stats_time: float = field(default_factory=time.time)
    vad: VADSegmenter = field(default_factory=VADSegmenter)
    utterance_count: int = 0
    conversation: ConversationState = field(default_factory=ConversationState)
    history: List[Dict[str, str]] = field(default_factory=list)
    last_tool_result: Optional[Dict[str, Any]] = None

    def seconds_estimate(self) -> float:
        cfg = self.audio_config or {}
        sample_rate = int(cfg.get("sample_rate", 16000) or 16000)
        channels = int(cfg.get("channels", 1) or 1)
        bytes_per_frame = 2 * max(1, channels)
        if sample_rate <= 0 or bytes_per_frame <= 0:
            return 0.0
        frames = self.total_audio_bytes / bytes_per_frame
        return frames / sample_rate


ALLOWED_PATCH_FIELDS = {
    "recipient_label",
    "recipient_contact_id",
    "amount_lkr",
    "note",
    "pending_note",
    "intent",
    "step",
    "draft_summary",
    "draft_id",
    "awaiting_biometric",
}


def _apply_state_patch(conv: ConversationState, patch: Optional[Dict[str, Any]]) -> None:
    if not patch:
        return
    for key, value in patch.items():
        if key in ALLOWED_PATCH_FIELDS:
            setattr(conv, key, value)


def _run_tool_call(tool_call: Dict[str, Any], session_state: SessionState) -> Dict[str, Any]:
    name = tool_call.get("name")
    args = tool_call.get("args") or {}
    conv = session_state.conversation
    if name == "search_contact":
        query = args.get("query") or args.get("label") or ""
        result = search_contact(query)
        return {"tool": "search_contact", "query": query, "result": result}
    if name == "get_history":
        return {"tool": "get_history", "items": get_ledger()}
    if name == "prepare_transfer":
        contact_id = conv.recipient_contact_id
        amount = conv.amount_lkr
        if args.get("recipient_label"):
            conv.recipient_label = args.get("recipient_label")
        if args.get("note") and not conv.note:
            conv.note = args.get("note")
        if not contact_id or amount is None:
            return {"tool": "prepare_transfer", "error": "missing contact or amount"}
        conv.amount_lkr = int(amount)
        conv.awaiting_biometric = True
        conv.draft_id = conv.draft_id or f"draft_{uuid.uuid4().hex[:8]}"
        conv.pending_transfer = {
            "to_contact_id": conv.recipient_contact_id,
            "to_label": conv.recipient_label,
            "amount_lkr": conv.amount_lkr,
            "note": conv.note,
        }
        conv.step = "awaiting_biometric"
        summary = conv.build_summary()
        conv.draft_summary = summary
        return {
            "tool": "prepare_transfer",
            "summary": summary,
            "draft_id": conv.draft_id,
        }
    if name == "execute_transfer":
        return {"tool": "execute_transfer", "error": "biometric_required"}
    return {"error": f"unknown_tool_{name}"}


async def _process_agent_decision(
    websocket: WebSocket,
    session_state: SessionState,
    decision: AgentDecision,
    loop_depth: int = 0,
) -> None:
    conv = session_state.conversation
    _apply_state_patch(conv, decision.state_patch)

    if decision.ui_actions:
        await websocket.send_json(
            {"type": "UI_ACTIONS", "actions": [a.dict(exclude_none=True) for a in decision.ui_actions]}
        )

    if decision.say:
        await websocket.send_json({"type": "AGENT_MESSAGE", "text": decision.say})
        audio_bytes = synthesize_wav(decision.say)
        await websocket.send_json(
            {"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(audio_bytes)}
        )
        await websocket.send_bytes(audio_bytes)
        session_state.history.append({"role": "assistant", "content": decision.say})

    if decision.tool_call and loop_depth < 2:
        result = _run_tool_call(decision.tool_call.dict(), session_state)
        session_state.last_tool_result = result
        follow_up = agent.decide(
            session_state.conversation, user_text=None, last_tool=result, history=session_state.history
        )
        await _process_agent_decision(websocket, session_state, follow_up, loop_depth + 1)
    else:
        session_state.last_tool_result = None


async def _handle_user_utterance(
    websocket: WebSocket, session_state: SessionState, transcript: str
) -> None:
    session_state.history.append({"role": "user", "content": transcript})
    decision = agent.decide(
        session_state.conversation,
        user_text=transcript,
        last_tool=session_state.last_tool_result,
        history=session_state.history,
    )
    await _process_agent_decision(websocket, session_state, decision)


async def _speak_and_act(
    websocket: WebSocket, text: str, actions: Optional[List[Dict[str, Any]]] = None
) -> None:
    if text:
        await websocket.send_json({"type": "AGENT_MESSAGE", "text": text})
        audio_bytes = synthesize_wav(text)
        await websocket.send_json(
            {"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(audio_bytes)}
        )
        await websocket.send_bytes(audio_bytes)
    if actions:
        await websocket.send_json({"type": "UI_ACTIONS", "actions": actions})


async def _execute_transfer(websocket: WebSocket, session_state: SessionState) -> None:
    conv = session_state.conversation
    if (
        conv.step != "awaiting_biometric"
        or not conv.awaiting_biometric
        or not conv.recipient_contact_id
        or not conv.recipient_label
        or conv.amount_lkr is None
    ):
        await _speak_and_act(
            websocket, "I cannot execute the transfer yet. Please complete the details."
        )
        return

    entry = make_transfer_entry(
        conv.recipient_contact_id, conv.recipient_label, conv.amount_lkr, conv.note
    )
    append_ledger_entry(entry)
    ledger_items = get_ledger()
    conv.step = "done"
    conv.pending_transfer = None
    conv.awaiting_biometric = False
    conv.draft_id = None
    conv.draft_summary = None

    actions = [
        {"type": "NAVIGATE", "screen": "success"},
        {"type": "SHOW_TOAST", "message": "Payment sent"},
        {"type": "SHOW_HISTORY", "items": ledger_items},
    ]
    message = f"Done. I sent {conv.amount_lkr} LKR to {conv.recipient_label}."
    await _speak_and_act(websocket, message, actions)


async def _handle_start_session(
    websocket: WebSocket, payload: Dict[str, Any], session_state: SessionState
) -> None:
    user = payload.get("user") or {}
    name = user.get("name") or "John"
    session_state.conversation.reset()
    session_state.conversation.user_name = name
    session_state.conversation.step = "idle"
    session_state.history.clear()
    session_state.last_tool_result = None
    greeting = f"Hello {name}, I can help you with that."
    logger.info("Sending greeting to %s", name)

    await websocket.send_json({"type": "AGENT_MESSAGE", "text": greeting})

    audio_bytes = synthesize_wav(greeting)
    await websocket.send_json(
        {"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(audio_bytes)}
    )
    await websocket.send_bytes(audio_bytes)


async def _handle_text_message(
    websocket: WebSocket, text: str, session_state: SessionState
) -> bool:
    """Returns True if the websocket should close."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Received non-JSON text frame: %s", text[:80])
        return False

    message_type = payload.get("type")
    if message_type == "START_SESSION":
        await _handle_start_session(websocket, payload, session_state)
    elif message_type == "AUDIO_CONFIG":
        session_state.audio_config = {
            "format": payload.get("format"),
            "sample_rate": payload.get("sample_rate"),
            "channels": payload.get("channels"),
        }
        logger.info("Received AUDIO_CONFIG: %s", session_state.audio_config)
    elif message_type == "BIOMETRIC_RESULT":
        ok = bool(payload.get("ok"))
        if ok:
            await _execute_transfer(websocket, session_state)
        else:
            await _speak_and_act(
                websocket, "Fingerprint verification failed. Please try again."
            )
    elif message_type == "STOP":
        logger.info("Received STOP from client")
        final_utts = session_state.vad.flush()
        for utt in final_utts:
            transcript = transcribe_pcm16k(utt)
            await websocket.send_json({"type": "ASR_FINAL", "text": transcript})
            await _handle_user_utterance(websocket, session_state, transcript)
        return True
    else:
        logger.info("Unhandled message type %s", message_type)

    return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_state = SessionState()
    closing = False
    logger.info("WebSocket connected from %s", websocket.client)
    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                logger.info("Client disconnected")
                break

            if "text" in message:
                should_close = await _handle_text_message(
                    websocket, message.get("text") or "", session_state
                )
                if should_close:
                    closing = True
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_json(
                                {"type": "END", "reason": "stopped"}
                            )
                        except RuntimeError:
                            pass
                    break
            elif "bytes" in message:
                payload = message.get("bytes") or b""
                session_state.total_audio_bytes += len(payload)
                utterances = session_state.vad.accept_bytes(payload)
                for utt in utterances:
                    session_state.utterance_count += 1
                    transcript = transcribe_pcm16k(utt)
                    await websocket.send_json(
                        {"type": "ASR_FINAL", "text": transcript}
                    )
                    await _handle_user_utterance(websocket, session_state, transcript)

                now = time.time()
                if now - session_state.last_stats_time >= 1.0:
                    session_state.last_stats_time = now
                    await websocket.send_json(
                        {
                            "type": "AUDIO_STATS",
                            "total_bytes": session_state.total_audio_bytes,
                            "seconds_estimate": round(
                                session_state.seconds_estimate(), 2
                            ),
                        }
                    )
            else:
                logger.info("Ignored websocket frame %s", message_type)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected unexpectedly")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED and not closing:
            try:
                await websocket.close()
            except RuntimeError:
                pass
