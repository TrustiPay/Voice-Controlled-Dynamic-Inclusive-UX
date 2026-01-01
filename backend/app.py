from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from starlette.websockets import WebSocketState # type: ignore

from asr import transcribe_pcm16k
from heuristics import (
    detect_transfer_intent,
    extract_amount,
    extract_note,
    extract_recipient_hint,
    is_affirmative,
    is_note_request_only,
    is_skip_note,
)
from state import ConversationState
from tools import (
    append_ledger_entry,
    get_ledger,
    make_transfer_entry,
    search_contact,
)
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


def _build_confirmation(state: ConversationState) -> Tuple[str, List[Dict[str, Any]]]:
    summary = state.build_summary()
    state.draft_summary = summary
    state.pending_transfer = {
        "to_contact_id": state.recipient_contact_id,
        "to_label": state.recipient_label,
        "amount_lkr": state.amount_lkr,
        "note": state.note,
    }
    state.step = "awaiting_biometric"
    actions = [
        {"type": "NAVIGATE", "screen": "confirm"},
        {"type": "SHOW_CONFIRM", "summary": summary},
        {"type": "PROMPT_BIOMETRIC"},
    ]
    say = f"Please confirm: {summary}"
    return say, actions


@dataclass
class SessionState:
    audio_config: Optional[Dict[str, Any]] = None
    total_audio_bytes: int = 0
    last_stats_time: float = field(default_factory=time.time)
    vad: VADSegmenter = field(default_factory=VADSegmenter)
    utterance_count: int = 0
    conversation: ConversationState = field(default_factory=ConversationState)

    def seconds_estimate(self) -> float:
        cfg = self.audio_config or {}
        sample_rate = int(cfg.get("sample_rate", 16000) or 16000)
        channels = int(cfg.get("channels", 1) or 1)
        bytes_per_frame = 2 * max(1, channels)
        if sample_rate <= 0 or bytes_per_frame <= 0:
            return 0.0
        frames = self.total_audio_bytes / bytes_per_frame
        return frames / sample_rate


def _handle_user_text(state: SessionState, text: str) -> Tuple[str, List[Dict[str, Any]]]:
    conv = state.conversation
    actions: List[Dict[str, Any]] = []
    cleaned = text.strip()
    if not cleaned:
        return "I didn't catch that. Could you repeat?", actions

    lower = cleaned.lower()

    if conv.step == "idle":
        if detect_transfer_intent(cleaned):
            conv.intent = "transfer"
            conv.step = "collect_recipient"
            actions.append({"type": "NAVIGATE", "screen": "transfer"})
            say = f"Sure {conv.user_name}. What is your friend's name as saved in your contacts?"
        else:
            say = "I can help you send money. Say something like: send 4000 rupees to Kevin at work."
        return say, actions

    if conv.step == "collect_recipient":
        label_hint = extract_recipient_hint(cleaned) or cleaned
        contact = search_contact(label_hint)
        if contact:
            conv.recipient_label = contact["label"]
            conv.recipient_contact_id = contact["id"]
            conv.step = "collect_amount"
            actions.append(
                {"type": "SET_FIELD", "field": "recipient", "value": conv.recipient_label}
            )
            say = "Thanks. How much would you like to send?"
        else:
            say = "I couldn't find that contact. Please say the name exactly as saved."
        return say, actions

    if conv.step == "collect_amount":
        amount = extract_amount(cleaned)
        note = extract_note(cleaned)
        if amount:
            conv.amount_lkr = amount
            actions.append({"type": "SET_FIELD", "field": "amount", "value": amount})
            if note:
                conv.pending_note = note
                actions.append({"type": "SET_FIELD", "field": "note", "value": note})
                conv.step = "confirm_note"
                say = f"I heard the note '{note}'. Should I use this note or change it?"
            else:
                if is_note_request_only(cleaned):
                    conv.pending_note = None
                    conv.step = "collect_note_value"
                    say = "Sure. What should the note say?"
                else:
                    conv.step = "collect_note_optional"
                    say = "Got it. Do you want to add a note?"
        else:
            say = "Please tell me the amount in rupees."
        return say, actions

    if conv.step == "collect_note_optional":
        if is_skip_note(lower):
            conv.note = None
            conv.pending_note = None
            say, confirm_actions = _build_confirmation(conv)
            actions.extend(confirm_actions)
            return say, actions
        if is_affirmative(cleaned) or is_note_request_only(cleaned):
            conv.pending_note = None
            conv.step = "collect_note_value"
            say = "Sure. What should the note say?"
        else:
            note = extract_note(cleaned) or cleaned
            note = note.strip()
            if note:
                conv.pending_note = note
                actions.append({"type": "SET_FIELD", "field": "note", "value": note})
                conv.step = "confirm_note"
                say = f"I heard the note '{note}'. Should I use this note or change it?"
            else:
                say = "Please tell me the note, or say no note."
        return say, actions

    if conv.step == "collect_note_value":
        if is_skip_note(lower):
            conv.note = None
            conv.pending_note = None
            say, confirm_actions = _build_confirmation(conv)
            actions.extend(confirm_actions)
            return say, actions
        note = extract_note(cleaned) or cleaned
        if note:
            conv.pending_note = note.strip()
            actions.append({"type": "SET_FIELD", "field": "note", "value": conv.pending_note})
            conv.step = "confirm_note"
            say = f"I captured the note '{conv.pending_note}'. Should I use this note or change it?"
        else:
            say = "I didn't catch the note. Please say it again, or say no note."
        return say, actions

    if conv.step == "confirm_note":
        if is_skip_note(lower):
            conv.note = None
            conv.pending_note = None
            say, confirm_actions = _build_confirmation(conv)
            actions.extend(confirm_actions)
            return say, actions
        if is_affirmative(cleaned):
            if conv.pending_note:
                conv.note = conv.pending_note
            conv.pending_note = None
            if conv.note:
                actions.append({"type": "SET_FIELD", "field": "note", "value": conv.note})
            say, confirm_actions = _build_confirmation(conv)
            actions.extend(confirm_actions)
            return say, actions

        # Treat this utterance as a replacement note
        new_note = extract_note(cleaned) or cleaned
        if new_note:
            conv.pending_note = new_note.strip()
            actions.append({"type": "SET_FIELD", "field": "note", "value": conv.pending_note})
            say = f"Updated the note to '{conv.pending_note}'. Should I use this note or change it?"
        else:
            say = "Please confirm the note, or say it again, or say no note."
        return say, actions

    if conv.step == "awaiting_biometric":
        return "Please approve with fingerprint to continue.", actions

    if conv.step == "done":
        return "This session is completed. Start a new session to continue.", actions

    return "I can help you send money. Say something like send 4000 rupees to Kevin at work.", actions


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
        or not conv.recipient_contact_id
        or not conv.recipient_label
        or not conv.amount_lkr
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
            say, actions = _handle_user_text(session_state, transcript)
            await _speak_and_act(websocket, say, actions)
        await websocket.close(code=1000)
        return True
    else:
        logger.info("Unhandled message type %s", message_type)

    return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_state = SessionState()
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
                    say, actions = _handle_user_text(session_state, transcript)
                    await _speak_and_act(websocket, say, actions)

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
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
