"""
Speech recognition with automatic language detection.

Uses faster-whisper (CTranslate2) which runs efficiently on CPU and, crucially
for a bilingual app, returns the *detected language* of the audio. That single
fact drives the whole translation flow: we know whether the speaker used English
or Swahili and can route accordingly.
"""
from __future__ import annotations

import threading
from typing import Tuple

from .config import cfg

_model = None
_lock = threading.Lock()


def _get_model():
    """Lazy, thread-safe singleton so the model loads once."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(
                    cfg.ASR_MODEL,
                    device=cfg.ASR_DEVICE,
                    compute_type=cfg.ASR_COMPUTE,
                )
    return _model


# Languages this assistant actively supports. Anything else is mapped to the
# nearest of these two so downstream logic stays simple.
SUPPORTED = {"en", "sw"}


def transcribe(audio_path: str) -> Tuple[str, str]:
    """
    Transcribe an audio file.

    Returns (text, language) where language is 'en' or 'sw'.
    If Whisper detects another language we keep the transcript but fall back to
    'en' so the pipeline never breaks.
    """
    if not audio_path:
        return "", "en"

    model = _get_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,          # trims silence -> cleaner transcripts
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    lang = (info.language or "en").lower()
    if lang not in SUPPORTED:
        # Whisper's detector occasionally returns a neighbour language for
        # short clips; clamp to the two we serve.
        lang = "sw" if lang in {"sw", "swh"} else "en"
    return text, lang
