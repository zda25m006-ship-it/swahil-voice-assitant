"""
Text-to-speech via gTTS. Produces an MP3 file path that Gradio can play.
Returns None when TTS is disabled or fails (the UI then just shows text).
"""
from __future__ import annotations

import tempfile

from .config import cfg


def speak(text: str, lang: str = "en"):
    text = (text or "").strip()
    if not cfg.ENABLE_TTS or not text:
        return None
    if lang not in {"en", "sw"}:
        lang = "en"
    text = text[: cfg.TTS_MAX_CHARS]
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang)
        path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        tts.save(path)
        return path
    except Exception:
        return None
