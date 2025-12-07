from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agent import AgentResult, run_agent
from state import SessionState
from tts import synthesize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trustipay.app")

app = FastAPI(title="TrustiPay Prototype")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _send_tts(websocket: WebSocket, text: str) -> None:
    audio_bytes, sample_rate = synthesize(text)
    await websocket.send_json(
        {"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(audio_bytes), "sample_rate": sample_rate}
    )
    await websocket.send_bytes(audio_bytes)


async def _emit_agent_result(websocket: WebSocket, result: AgentResult, state: SessionState) -> None:
    await websocket.send_json({"type": "AGENT_MESSAGE", "text": result.say})
    if result.ui_actions:
        await websocket.send_json({"type": "UI_ACTIONS", "actions": result.ui_actions})
    if result.state_patch:
        state.patch(result.state_patch)
        logger.info("Applied state patch: %s", result.state_patch)
    if result.say:
        await _send_tts(websocket, result.say)


async def _handle_start_session(websocket: WebSocket, payload: Dict[str, Any], state: SessionState) -> None:
    user = payload.get("user") or {}
    state.reset(user_id=user.get("id", "anon"), user_name=user.get("name", "Guest"), language=payload.get("language", "en"))
    logger.info("Session started for user_id=%s user_name=%s", state.user_id, state.user_name)
    result = run_agent(event="start", state=state)
    await _emit_agent_result(websocket, result, state)


async def _handle_client_message(websocket: WebSocket, payload: Dict[str, Any], state: SessionState) -> bool:
    """
    Returns False if the loop should terminate.
    """
    msg_type = payload.get("type")
    if msg_type == "START_SESSION":
        await _handle_start_session(websocket, payload, state)
        return True

    if msg_type == "AUDIO_CONFIG":
        state.audio_config = payload
        logger.info("Audio config set: %s", payload)
        return True

    if msg_type == "BIOMETRIC_RESULT":
        state.awaiting_biometric = False
        await websocket.send_json({"type": "AGENT_MESSAGE", "text": "Biometric result received."})
        return True

    if msg_type == "STOP":
        await websocket.send_json({"type": "END", "reason": "stopped"})
        return False

    if msg_type == "PING":
        await websocket.send_json({"type": "PONG"})
        return True

    logger.warning("Unhandled message type: %s", msg_type)
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state = SessionState()
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

                keep_going = await _handle_client_message(websocket, payload, state)
                if not keep_going:
                    break
            elif message.get("bytes") is not None:
                chunk = message["bytes"] or b""
                state.record_audio_chunk(len(chunk))
                logger.debug("Received %s audio bytes (total %s)", len(chunk), state.bytes_received)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()
