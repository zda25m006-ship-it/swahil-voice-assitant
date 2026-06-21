"""
Central configuration for SautiSafari Advanced.

Every tunable lives here and is overridable through environment variables
(or a local .env file). Import `cfg` anywhere you need a setting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the working directory if present
except Exception:
    pass


def _b(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # ---- Speech recognition (faster-whisper) -------------------------------
    # tiny / base / small / medium  (small is a good CPU/accuracy balance)
    ASR_MODEL: str = os.getenv("ASR_MODEL", "small")
    ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu")          # cpu | cuda
    ASR_COMPUTE: str = os.getenv("ASR_COMPUTE", "int8")        # int8 (cpu) | float16 (gpu)

    # ---- Retrieval (RAG) ----------------------------------------------------
    EMBED_MODEL: str = os.getenv(
        "EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    RAG_THRESHOLD: float = float(os.getenv("RAG_THRESHOLD", "0.16"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # ---- Translation --------------------------------------------------------
    # google  -> deep-translator (accurate, needs internet, no key)
    # llm     -> use the configured LLM to translate (works offline w/ Ollama)
    TRANSLATE_PROVIDER: str = os.getenv("TRANSLATE_PROVIDER", "google")

    # ---- Language model -----------------------------------------------------
    # ollama | openai | none
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    # Generic OpenAI-compatible endpoint (OpenAI, Together, Groq, etc.)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ---- Text to speech -----------------------------------------------------
    ENABLE_TTS: bool = _b("ENABLE_TTS", "1")
    TTS_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "900"))

    # ---- Geo / maps ---------------------------------------------------------
    # Public demo services. For production self-host OSRM + a Nominatim mirror.
    NOMINATIM_URL: str = os.getenv(
        "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
    )
    OSRM_URL: str = os.getenv("OSRM_URL", "https://router.project-osrm.org")
    GEO_USER_AGENT: str = os.getenv("GEO_USER_AGENT", "SautiSafari/2.0 (tourism assistant)")
    # Map default centre: Stone Town, Zanzibar
    MAP_CENTER_LAT: float = float(os.getenv("MAP_CENTER_LAT", "-6.1659"))
    MAP_CENTER_LON: float = float(os.getenv("MAP_CENTER_LON", "39.2026"))

    # ---- Server -------------------------------------------------------------
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "7860"))


cfg = Config()
