from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agent import (
    AgentDecision,
    LlmAgent,
    ToolCall,
    apply_state_patch,
    execute_tool_call,
    handle_biometric_ok,
)
from asr import transcribe_pcm16k
from state import SessionState
from tts import synthesize_wav
from vad import SileroVAD


@dataclass
class WsSession:
    state: SessionState = field(default_factory=SessionState)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    total_audio_bytes: int = 0
    bytes_since_stats: int = 0
    last_stats_time: float = field(default_factory=time.monotonic)
    vad: SileroVAD = field(default_factory=SileroVAD)
    utterance_count: int = 0
    stats_interval_bytes: int = 32_000  # ~1 second at 16kHz mono 16-bit
    messages: list[Dict[str, str]] = field(default_factory=list)

    def record_bytes(self, size: int) -> None:
        self.total_audio_bytes += size
        self.bytes_since_stats += size

    def pop_stats(self, sample_rate: int) -> Dict[str, Any] | None:
        if self.bytes_since_stats < self.stats_interval_bytes:
            return None
        total_bytes = self.total_audio_bytes
        seconds_estimate = total_bytes / 2 / max(sample_rate, 1)
        self.bytes_since_stats = 0
        self.last_stats_time = time.monotonic()
        return {
            "type": "AUDIO_STATS",
            "total_bytes": total_bytes,
            "seconds_estimate": round(seconds_estimate, 2),
        }

    def add_message(self, role: str, content: str) -> None:
        if not content:
            return
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-8:]

    def recent_messages(self) -> list[Dict[str, str]]:
        return self.messages[-8:]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trustipay.app")

app = FastAPI(title="TrustiPay Prototype")
agent = LlmAgent()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


async def _send_greeting(websocket: WebSocket, user_name: str) -> None:
    text = f"Hello {user_name}, I can help you with that."
    await websocket.send_json({"type": "AGENT_MESSAGE", "text": text})

    wav_bytes = synthesize_wav(text)
    await websocket.send_json({"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(wav_bytes)})
    await websocket.send_bytes(wav_bytes)


async def _send_response(websocket: WebSocket, say: str, ui_actions: list[dict[str, Any]] | None = None) -> None:
    await websocket.send_json({"type": "AGENT_MESSAGE", "text": say})
    if ui_actions:
        await websocket.send_json({"type": "UI_ACTIONS", "actions": ui_actions})
    if say:
        wav_bytes = synthesize_wav(say)
        await websocket.send_json({"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(wav_bytes)})
        await websocket.send_bytes(wav_bytes)


def _serialize_actions(actions: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for action in actions:
        if hasattr(action, "model_dump"):
            serialized.append(action.model_dump(exclude_none=True))
        elif isinstance(action, dict):
            serialized.append({k: v for k, v in action.items() if v is not None})
    return serialized


def _apply_actions_to_state(state: SessionState, actions: list[dict[str, Any]]) -> None:
    for action in actions:
        if action.get("type") == "SET_FIELD":
            field = action.get("field")
            value = action.get("value")
            if field == "recipient":
                state.recipient_label = value
            elif field == "amount":
                state.amount_lkr = value
            elif field == "note":
                state.note = value
        if action.get("type") == "PROMPT_BIOMETRIC":
            state.step = "awaiting_biometric"


async def _handle_client_message(
    websocket: WebSocket,
    payload: Dict[str, Any],
    session: WsSession,
) -> bool:
    """
    Returns False if the loop should terminate.
    """
    msg_type = payload.get("type")
    if msg_type == "START_SESSION":
        user = payload.get("user") or {}
        user_name = user.get("name", "John")
        session.state.reset(user_id=user.get("id", "anon"), user_name=user_name, language=payload.get("language", "en"))
        await _send_greeting(websocket, user_name)
        return True

    if msg_type == "AUDIO_CONFIG":
        session.audio_config = payload
        logger.info("Audio config set: %s", payload)
        return True

    if msg_type == "BIOMETRIC_RESULT":
        if payload.get("ok"):
            response = handle_biometric_ok(session.state)
        logger.info("Biometric OK -> %s", response["ui_actions"])
        await _send_response(websocket, response["say"], response["ui_actions"])
    else:
        await _send_response(websocket, "Fingerprint verification failed. Please try again.", [])
    return True

    if msg_type == "STOP":
        await websocket.send_json({"type": "END", "reason": "stopped"})
        return False

    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = WsSession()
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Discarding non-JSON text frame: %s", message["text"])
                    continue

                keep_going = await _handle_client_message(websocket, payload, session)
                if not keep_going:
                    break
            elif message.get("bytes") is not None:
                chunk = message["bytes"] or b""
                chunk_len = len(chunk)
                session.record_bytes(chunk_len)

                sample_rate = (session.audio_config or {}).get("sample_rate", 16_000)
                stats_payload = session.pop_stats(sample_rate)
                if stats_payload:
                    await websocket.send_json(stats_payload)

                # VAD + ASR
                try:
                    utterances = session.vad.accept_bytes(chunk)
                except Exception as exc:  # safeguard against VAD errors
                    logger.exception("VAD processing failed: %s", exc)
                    utterances = []

                for utterance in utterances:
                    transcript = ""
                    try:
                        transcript = transcribe_pcm16k(utterance)
                    except Exception as exc:
                        logger.exception("ASR transcription failed: %s", exc)
                    if transcript:
                        session.utterance_count += 1
                        logger.info("ASR_FINAL: %s", transcript)
                        await websocket.send_json({"type": "ASR_FINAL", "text": transcript})
                        session.add_message("user", transcript)
                        await _run_decision_flow(websocket, session, user_text=transcript)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()
