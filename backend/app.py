from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from tts import synthesize_wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trustipay.app")

app = FastAPI(title="TrustiPay Prototype")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


async def _send_greeting(websocket: WebSocket, user_name: str) -> None:
    text = f"Hello {user_name}, I can help you with that."
    await websocket.send_json({"type": "AGENT_MESSAGE", "text": text})

    wav_bytes = synthesize_wav(text)
    await websocket.send_json({"type": "TTS_AUDIO", "mime": "audio/wav", "nbytes": len(wav_bytes)})
    await websocket.send_bytes(wav_bytes)


async def _handle_client_message(websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    """
    Returns False if the loop should terminate.
    """
    msg_type = payload.get("type")
    if msg_type == "START_SESSION":
        user = payload.get("user") or {}
        user_name = user.get("name", "John")
        await _send_greeting(websocket, user_name)
        return True

    if msg_type == "STOP":
        await websocket.send_json({"type": "END", "reason": "stopped"})
        return False

    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
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

                keep_going = await _handle_client_message(websocket, payload)
                if not keep_going:
                    break
            elif message.get("bytes") is not None:
                # Ignore binary frames for Milestone A (no mic streaming yet).
                logger.debug("Received %s binary bytes (ignored in Milestone A)", len(message["bytes"] or b""))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()
