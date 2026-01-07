from __future__ import annotations

import io
import logging
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Final, Optional

logger = logging.getLogger("trustipay.tts")

SAMPLE_RATE: Final[int] = 16_000
PIPER_MODEL_ENV: Final[str] = "PIPER_EN_VOICE"
PIPER_BIN_ENV: Final[str] = "PIPER_BIN"
SINHALA_MODEL_ENV: Final[str] = "SINHALA_TTS_MODEL"
SINHALA_SPEAKER_ENV: Final[str] = "SINHALA_TTS_SPEAKER"
PIPER_DEFAULT_DIR: Final[Path] = Path(__file__).resolve().parent / "models" / "piper-voices"
PIPER_BUNDLE_DIR: Final[Path] = Path(__file__).resolve().parent / "models" / "piper"
PIPER_BUNDLE_BIN: Final[Path] = PIPER_BUNDLE_DIR / "piper"

DEFAULT_EN_COQUI: Final[str] = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_SI_COQUI: Final[str] = "tts_models/si/si_lk/vits"

_coqui_cache: Dict[str, object] = {}


def _beep_fallback(text: str) -> bytes:
    duration_seconds = max(1.0, min(3.0, 0.35 + len(text) / 18.0))
    amplitude = 0.25
    base_freq = 520.0
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)

        for i in range(int(duration_seconds * SAMPLE_RATE)):
            t = i / SAMPLE_RATE
            sample = int(amplitude * 32767 * math.sin(2 * math.pi * base_freq * t))
            wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))

    return buffer.getvalue()


def _tts_with_coqui(model_name: str, text: str, speaker: Optional[str] = None) -> Optional[bytes]:
    try:
        tts = _coqui_cache.get(model_name)
        if tts is None:
            from TTS.api import TTS  # type: ignore

            tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
            _coqui_cache[model_name] = tts
            logger.info("Loaded Coqui TTS model: %s", model_name)

        wav = tts.tts(text, speaker=speaker) if speaker else tts.tts(text) # type: ignore
        import soundfile as sf  # type: ignore

        with io.BytesIO() as buf:
            sf.write(buf, wav, tts.synthesizer.output_sample_rate, format="WAV") # type: ignore
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Coqui TTS (%s) failed: %s", model_name, exc)
        return None


def _discover_piper_voice() -> Optional[Path]:
    """
    Find a usable Piper voice in the repo (backend/models/piper-voices).
    Prefers en_US-lessac-* if present, otherwise first .onnx file.
    """
    if not PIPER_DEFAULT_DIR.exists():
        return None

    candidates = sorted(PIPER_DEFAULT_DIR.glob("*.onnx"))
    if not candidates:
        return None

    for cand in candidates:
        if "en_US-lessac" in cand.name:
            return cand
    return candidates[0]


def _find_piper_bin() -> Optional[str]:
    """
    Locate the Piper binary:
    1) PIPER_BIN env
    2) bundled binary at backend/models/piper/piper
    3) fallback to PATH lookup ("piper")
    """
    env_bin = os.getenv(PIPER_BIN_ENV)
    if env_bin:
        return env_bin

    if PIPER_BUNDLE_BIN.exists() and os.access(PIPER_BUNDLE_BIN, os.X_OK):
        return str(PIPER_BUNDLE_BIN)

    path_bin = shutil.which("piper")
    return path_bin


def _tts_with_piper(text: str) -> Optional[bytes]:
    """
    Use local Piper CLI if available. Requires env PIPER_EN_VOICE to point to a voice file.
    """
    model_path = os.getenv(PIPER_MODEL_ENV)
    if not model_path:
        auto_voice = _discover_piper_voice()
        if auto_voice:
            model_path = str(auto_voice)
            logger.info("Using bundled Piper voice: %s", model_path)

    piper_bin = _find_piper_bin()

    if not model_path:
        logger.debug("Piper skipped: %s env not set", PIPER_MODEL_ENV)
        return None
    if not piper_bin:
        logger.warning("Piper binary not found (set %s or install on PATH)", PIPER_BIN_ENV)
        return None

    voice = Path(model_path)
    if not voice.exists():
        logger.warning("Piper voice not found: %s", voice)
        return None

    run_env = os.environ.copy()
    if PIPER_BUNDLE_DIR.exists():
        lib_path = run_env.get("LD_LIBRARY_PATH", "")
        parts = [str(PIPER_BUNDLE_DIR)]
        if lib_path:
            parts.append(lib_path)
        run_env["LD_LIBRARY_PATH"] = ":".join(parts)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_path = Path(tmp_wav.name)

    try:
        proc = subprocess.run(
            [piper_bin, "--model", str(voice), "--output_file", str(tmp_path)],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=run_env,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            logger.warning("Piper failed (code %s): %s", proc.returncode, stderr)
            return None
        return tmp_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Piper synthesis error: %s", exc)
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _cli_fallback(text: str, lang: str) -> Optional[bytes]:
    """
    Fallback using festival/text2wave or espeak-ng if installed.
    """
    lang = lang or "en"

    text2wave = shutil.which("text2wave")
    if text2wave:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                [text2wave, "-o", str(tmp_path)],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0 and tmp_path.exists():
                return tmp_path.read_bytes()
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            logger.warning("text2wave failed (code %s): %s", proc.returncode, stderr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("text2wave error: %s", exc)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    for espeak_bin in ("espeak-ng", "espeak"):
        bin_path = shutil.which(espeak_bin)
        if not bin_path:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                [bin_path, "-v", lang, "-w", str(tmp_path)],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0 and tmp_path.exists():
                return tmp_path.read_bytes()
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            logger.warning("%s failed (code %s): %s", espeak_bin, proc.returncode, stderr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s error: %s", espeak_bin, exc)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    return None


def synthesize_wav(text: str, language: str = "en") -> bytes:
    """
    Language-aware TTS with fallbacks:
    - English: Piper voice from env, then Coqui.
    - Sinhala: Coqui VITS (Hugging Face model).
    - Otherwise: CLI fallback or beep.
    """
    lang = (language or "en").split("-")[0].lower()

    if lang == "en":
        audio = _tts_with_piper(text)
        if audio:
            return audio
        audio = _tts_with_coqui(DEFAULT_EN_COQUI, text)
        if audio:
            return audio
        cli_audio = _cli_fallback(text, lang)
        if cli_audio:
            return cli_audio
    elif lang in {"si", "sin", "sinhala"}:
        model_name = os.getenv(SINHALA_MODEL_ENV, DEFAULT_SI_COQUI)
        speaker = os.getenv(SINHALA_SPEAKER_ENV)
        audio = _tts_with_coqui(model_name, text, speaker=speaker)
        if audio:
            return audio
        # Sinhala voices in festival/espeak are rare; fallback keeps lang flag anyway.
        cli_audio = _cli_fallback(text, lang)
        if cli_audio:
            return cli_audio
    else:
        cli_audio = _cli_fallback(text, lang)
        if cli_audio:
            return cli_audio

    # Final safety net.
    return _beep_fallback(text)
