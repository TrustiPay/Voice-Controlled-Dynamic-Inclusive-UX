from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import time

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


async def _handle_client_message(
    websocket: WebSocket,
    payload: Dict[str, Any],
    session: Dict[str, Any],
) -> bool:
    """
    Returns False if the loop should terminate.
    """
    msg_type = payload.get("type")
    if msg_type == "START_SESSION":
        user = payload.get("user") or {}
        user_name = user.get("name", "John")
        await _send_greeting(websocket, user_name)
        return True

    if msg_type == "AUDIO_CONFIG":
        session["audio_config"] = payload
        logger.info("Audio config set: %s", payload)
        return True

    if msg_type == "STOP":
        await websocket.send_json({"type": "END", "reason": "stopped"})
        return False

    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session: Dict[str, Any] = {
        "audio_config": None,
        "total_audio_bytes": 0,
        "bytes_since_stats": 0,
        "last_stats_time": time.monotonic(),
    }
    stats_interval_bytes = 32_000  # ~1 second at 16kHz mono 16-bit
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
                session["total_audio_bytes"] += chunk_len
                session["bytes_since_stats"] += chunk_len

                if session["bytes_since_stats"] >= stats_interval_bytes:
                    total_bytes = session["total_audio_bytes"]
                    sample_rate = (session.get("audio_config") or {}).get("sample_rate", 16_000)
                    seconds_estimate = total_bytes / 2 / max(sample_rate, 1)
                    await websocket.send_json(
                        {
                            "type": "AUDIO_STATS",
                            "total_bytes": total_bytes,
                            "seconds_estimate": round(seconds_estimate, 2),
                        }
                    )
                    session["bytes_since_stats"] = 0
                    session["last_stats_time"] = time.monotonic()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()
