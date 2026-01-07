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

from asr import transcribe_pcm16k
from fsm import ActionBundle, Event, EventType, State, handle_event, validate_ui_actions
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
    language: str = "en"
    utterance_language: Optional[str] = None
    fsm_state: State = State.IDLE

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

LANG_TEXT = {
    "greeting": {
        "en": "Hello {name}, how can I help you today?",
        "si": "ආයුබෝවන් {name}, මම කෙසේ උදව් කරන්නද?",
    },
    "transfer_incomplete": {
        "en": "I cannot execute the transfer yet. Please complete the details.",
        "si": "මට තවම ගෙවීම කළ නොහැක. කරුණාකර විස්තර සම්පූර්ණ කරන්න.",
    },
    "fingerprint_fail": {
        "en": "Fingerprint verification failed. Please try again.",
        "si": "ඇඟිලි රේඛා සත්‍යාපනය අසාර්ථකයි. නැවත උත්සාහ කරන්න.",
    },
    "transfer_done": {
        "en": "Done. I sent {amount} LKR to {recipient}.",
        "si": "අවසන්. රු. {amount} {recipient}ට යවා දීලා.",
    },
}


def _apply_state_patch(conv: ConversationState, patch: Optional[Dict[str, Any]]) -> None:
    if not patch:
        return
    for key, value in patch.items():
        if key in ALLOWED_PATCH_FIELDS:
            setattr(conv, key, value)


def _tts_language(session_state: SessionState) -> str:
    if session_state.language == "auto":
        return (
            session_state.utterance_language
            or session_state.conversation.detected_language
            or "en"
        )
    return session_state.language or "en"


def _localized_text(key: str, session_state: SessionState, **kwargs: Any) -> str:
    lang = _tts_language(session_state).split("-")[0]
    template = LANG_TEXT.get(key, {}).get(lang) or LANG_TEXT.get(key, {}).get("en") or ""
    return template.format(**kwargs)


def _run_tool_call(tool_call: Dict[str, Any], session_state: SessionState) -> Dict[str, Any]:
    name = tool_call.get("name")
    args = tool_call.get("args") or {}
    conv = session_state.conversation
    if name == "search_contact":
        query = args.get("query") or args.get("label") or ""
        result = search_contact(query)
        return {"tool": "search_contact", "result": result}
    if name == "get_history":
        return {"tool": "get_history", "result": get_ledger()}
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
            "result": {"summary": summary, "draft_id": conv.draft_id},
        }
    if name == "execute_transfer":
        if (
            not conv.awaiting_biometric
            or not conv.recipient_contact_id
            or not conv.recipient_label
            or conv.amount_lkr is None
        ):
            return {"tool": "execute_transfer", "error": "biometric_required"}
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
        return {
            "tool": "execute_transfer",
            "result": {
                "entry": entry,
                "ledger": ledger_items,
                "message": f"Done. I sent {conv.amount_lkr} LKR to {conv.recipient_label}.",
            },
        }
    return {"tool": name or "unknown", "error": "unknown_tool"}


async def _send_tts(websocket: WebSocket, text: str, session_state: SessionState) -> None:
    audio_bytes = synthesize_wav(text, language=_tts_language(session_state))
    await websocket.send_json(
        {"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(audio_bytes)}
    )
    await websocket.send_bytes(audio_bytes)


async def _apply_bundle(websocket: WebSocket, session_state: SessionState, bundle: ActionBundle) -> None:
    conv = session_state.conversation
    _apply_state_patch(conv, bundle.state_patch)
    session_state.fsm_state = bundle.next_state
    conv.step = bundle.next_state.value
    if bundle.say:
        await websocket.send_json({"type": "AGENT_MESSAGE", "text": bundle.say})
        await _send_tts(websocket, bundle.say, session_state)
        session_state.history.append({"role": "assistant", "content": bundle.say})

    actions = validate_ui_actions(bundle.next_state, bundle.ui_actions or [])
    if actions:
        await websocket.send_json({"type": "UI_ACTIONS", "actions": actions})


async def _process_event(
    websocket: WebSocket, session_state: SessionState, event: Event, extract_mode: str = "heuristic"
) -> None:
    # User messages go into history
    if event.type == EventType.USER_TEXT and event.text:
        session_state.history.append({"role": "user", "content": event.text})

    loops = 0
    pending_event: Optional[Event] = event
    while pending_event and loops < 4:
        bundle = handle_event(session_state.fsm_state, session_state.conversation, pending_event, extract_mode)
        await _apply_bundle(websocket, session_state, bundle)

        pending_event = None
        if bundle.tool_call:
            tool_result = _run_tool_call(bundle.tool_call, session_state)
            session_state.last_tool_result = tool_result
            if tool_result.get("error"):
                message = (
                    _localized_text("transfer_incomplete", session_state)
                    if tool_result["error"] == "biometric_required"
                    else f"Sorry, there was a problem: {tool_result['error']}"
                )
                await websocket.send_json({"type": "AGENT_MESSAGE", "text": message})
                await _send_tts(websocket, message, session_state)
                break
            pending_event = Event(
                type=EventType.TOOL_RESULT,
                tool_name=tool_result.get("tool"),
                tool_result=tool_result.get("result"),
            )
        loops += 1


async def _handle_start_session(
    websocket: WebSocket, payload: Dict[str, Any], session_state: SessionState
) -> None:
    user = payload.get("user") or {}
    name = user.get("name") or "User"
    session_state.language = str(payload.get("language") or "en").lower()
    session_state.utterance_language = None
    session_state.conversation.reset()
    session_state.conversation.user_name = name
    session_state.conversation.language = session_state.language
    session_state.conversation.detected_language = None
    session_state.conversation.step = "idle"
    session_state.history.clear()
    session_state.last_tool_result = None
    session_state.fsm_state = State.IDLE
    start_event = Event(type=EventType.START_SESSION)
    await _process_event(websocket, session_state, start_event)
    greeting = _localized_text("greeting", session_state, name=name)
    logger.info("Sending greeting to %s", name)

    await websocket.send_json({"type": "AGENT_MESSAGE", "text": greeting})
    await _send_tts(websocket, greeting, session_state)


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
        await _process_event(websocket, session_state, Event(type=EventType.BIOMETRIC, biometric_ok=ok))
    elif message_type == "STOP":
        logger.info("Received STOP from client")
        final_utts = session_state.vad.flush()
        for utt in final_utts:
            transcript, detected_lang = transcribe_pcm16k(
                utt, language=session_state.language
            )
            if session_state.language == "auto" and detected_lang:
                session_state.utterance_language = detected_lang
                session_state.conversation.detected_language = detected_lang
            await websocket.send_json(
                {"type": "ASR_FINAL", "text": transcript, "language": detected_lang}
            )
            await _process_event(
                websocket,
                session_state,
                Event(type=EventType.USER_TEXT, text=transcript),
            )
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
                    transcript, detected_lang = transcribe_pcm16k(
                        utt, language=session_state.language
                    )
                    if session_state.language == "auto" and detected_lang:
                        session_state.utterance_language = detected_lang
                        session_state.conversation.detected_language = detected_lang
                    await websocket.send_json(
                        {"type": "ASR_FINAL", "text": transcript, "language": detected_lang}
                    )
                    await _process_event(
                        websocket,
                        session_state,
                        Event(type=EventType.USER_TEXT, text=transcript),
                    )

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
