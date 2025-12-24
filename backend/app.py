from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agent import handle_biometric_ok, handle_user_text
from asr import transcribe_pcm16k
from state import SessionState
from tts import synthesize_wav
from vad import SileroVAD

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


async def _send_response(websocket: WebSocket, say: str, ui_actions: list[dict[str, Any]] | None = None) -> None:
    await websocket.send_json({"type": "AGENT_MESSAGE", "text": say})
    if ui_actions:
        await websocket.send_json({"type": "UI_ACTIONS", "actions": ui_actions})
    if say:
        wav_bytes = synthesize_wav(say)
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
        session["state"].reset(user_id=user.get("id", "anon"), user_name=user_name, language=payload.get("language", "en"))
        await _send_greeting(websocket, user_name)
        return True

    if msg_type == "AUDIO_CONFIG":
        session["audio_config"] = payload
        logger.info("Audio config set: %s", payload)
        return True

    if msg_type == "BIOMETRIC_RESULT":
        if payload.get("ok"):
            response = handle_biometric_ok(session["state"])
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
    session: Dict[str, Any] = {
        "audio_config": None,
        "total_audio_bytes": 0,
        "bytes_since_stats": 0,
        "last_stats_time": time.monotonic(),
        "vad": SileroVAD(),
        "utterance_count": 0,
        "state": SessionState(),
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

                # VAD + ASR
                try:
                    utterances = session["vad"].accept_bytes(chunk)
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
                        session["utterance_count"] += 1
                        logger.info("ASR_FINAL: %s", transcript)
                        await websocket.send_json({"type": "ASR_FINAL", "text": transcript})
                        response = handle_user_text(session["state"], transcript)
                        logger.info(
                            "State step=%s recipient=%s amount=%s note=%s actions=%s",
                            session["state"].step,
                            session["state"].recipient_label,
                            session["state"].amount_lkr,
                            session["state"].note,
                            response["ui_actions"],
                        )
                        await _send_response(websocket, response["say"], response["ui_actions"])

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()
