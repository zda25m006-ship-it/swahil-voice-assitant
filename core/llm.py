"""
Grounded answer generation.

The LLM is only ever asked to answer FROM the retrieved context. The system
prompt forbids inventing facts, which keeps a small 3B model honest and on-topic
for tourism questions. Answers come back in the tourist's own language.

Providers:
  * ollama  -> local, offline (default): llama3.2:3b
  * openai  -> any OpenAI-compatible /chat/completions endpoint (set keys)
  * none    -> deterministic template answer built straight from the context
"""
from __future__ import annotations

import json
from typing import List, Dict, Any

import requests

from .config import cfg

LANG_NAME = {"en": "English", "sw": "Swahili"}

SYSTEM = (
    "You are SautiSafari, a friendly local guide for tourists in Zanzibar and "
    "Tanzania. Answer ONLY using the CONTEXT provided. If the context does not "
    "contain the answer, say you are not sure and suggest asking a local guide. "
    "Never invent place names, prices, or directions. Keep answers short, warm "
    "and practical (2-4 sentences). Reply ENTIRELY in {lang}."
)


def _format_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        loc = ""
        if h.get("lat") is not None and h.get("lon") is not None:
            loc = f" (coords: {h['lat']:.4f}, {h['lon']:.4f})"
        lines.append(f"- {h['name']}{loc}: {h.get('description','')}")
    return "\n".join(lines)


def raw_complete(prompt: str) -> str:
    """A bare completion with no system grounding (used by the translator)."""
    provider = cfg.LLM_PROVIDER.lower()
    if provider == "ollama":
        return _ollama(prompt)
    if provider == "openai":
        return _openai("You are a helpful assistant.", prompt)
    return prompt  # 'none' -> echo


def answer(query: str, hits: List[Dict[str, Any]], lang: str) -> str:
    """Produce a grounded tourism answer in `lang` from retrieval `hits`."""
    if not hits:
        if lang == "sw":
            return ("Samahani, sina uhakika kuhusu hilo. Tafadhali muulize "
                    "mwongozaji wa eneo kwa msaada zaidi.")
        return ("Sorry, I'm not sure about that one. A local guide can give you "
                "the most accurate help.")

    context = _format_context(hits)
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "none":
        return _template(hits, lang)

    sys = SYSTEM.format(lang=LANG_NAME.get(lang, "English"))
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    try:
        if provider == "ollama":
            return _ollama(f"{sys}\n\n{user}").strip()
        if provider == "openai":
            return _openai(sys, user).strip()
    except Exception:
        pass
    return _template(hits, lang)


def _template(hits: List[Dict[str, Any]], lang: str) -> str:
    """No-LLM fallback: a clean sentence built from the top hit."""
    top = hits[0]
    name = top["name"]
    desc = top.get("description", "").strip()
    if lang == "sw":
        return f"Ningependekeza {name}. {desc}".strip()
    return f"I'd suggest {name}. {desc}".strip()


def _ollama(prompt: str) -> str:
    resp = requests.post(
        cfg.OLLAMA_URL,
        json={"model": cfg.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _openai(system: str, user: str) -> str:
    resp = requests.post(
        f"{cfg.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.OPENAI_API_KEY}"},
        json={
            "model": cfg.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
