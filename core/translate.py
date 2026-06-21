"""
Two-way English <-> Swahili translation.

This is the heart of the "language barrier" feature:
  * a tourist speaks English  -> we render it in Swahili for a local
  * a local replies in Swahili -> we render it in English for the tourist

Two providers:
  * google : deep-translator's GoogleTranslator. Very accurate for sw<->en,
             no API key, needs internet.
  * llm    : ask the configured LLM (e.g. local Ollama) to translate. Works
             fully offline; quality depends on the model.
"""
from __future__ import annotations

import re

from .config import cfg

# A few high-frequency Swahili function words. Used as a cheap, offline
# heuristic to guess language when we only have typed text (no ASR signal).
_SWAHILI_HINTS = {
    "nataka", "wapi", "ni", "kwa", "naweza", "kuona", "sehemu", "gani", "nzuri",
    "fukwe", "chakula", "cha", "usiku", "habari", "asante", "karibu", "tafadhali",
    "kutembelea", "wewe", "mimi", "hapa", "pale", "ngapi", "bei", "samaki",
    "twende", "njoo", "hujambo", "mambo", "poa", "sawa", "ndiyo", "hapana",
}


def detect_lang(text: str) -> str:
    """Best-effort language guess for typed text: returns 'en' or 'sw'."""
    if not text:
        return "en"
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in _SWAHILI_HINTS)
    return "sw" if hits >= 1 and hits / len(words) > 0.12 else "en"


def _google(text: str, src: str, tgt: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=src, target=tgt).translate(text)


def _llm(text: str, src: str, tgt: str) -> str:
    # Imported here to avoid a hard dependency cycle at import time.
    from . import llm
    names = {"en": "English", "sw": "Swahili"}
    prompt = (
        f"Translate the following text from {names[src]} to {names[tgt]}. "
        f"Return ONLY the translation, no quotes, no notes.\n\n{text}"
    )
    return llm.raw_complete(prompt).strip()


def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate `text` from `src` to `tgt` (both in {'en','sw'}).

    Falls back to the LLM provider if the primary provider errors, and finally
    returns the original text so the UI never shows an empty box.
    """
    text = (text or "").strip()
    if not text or src == tgt:
        return text

    provider = cfg.TRANSLATE_PROVIDER.lower()
    try:
        return _google(text, src, tgt) if provider == "google" else _llm(text, src, tgt)
    except Exception:
        try:
            return _llm(text, src, tgt) if provider == "google" else _google(text, src, tgt)
        except Exception:
            return text


def other(lang: str) -> str:
    """The opposite supported language."""
    return "sw" if lang == "en" else "en"
