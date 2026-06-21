"""
SautiSafari Advanced — main application.

A FastAPI server that mounts:
  * a Gradio UI at "/"  with three tabs:
       1. Assistant  — ask a tourism question by voice or text, get a grounded,
                       spoken answer in your language + the other language, plus
                       a distance note from your live location.
       2. Translator — two-way English <-> Swahili voice/text translator that
                       breaks the language barrier with locals.
       3. Map        — live GPS map with search and turn-by-turn directions.
  * the live map page at "/static/map.html"

Run:  python app.py     (opens http://127.0.0.1:7860)
"""
from __future__ import annotations

import os
import re

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import asr, translate, rag, llm, tts, geo
from core.config import cfg

# ---------------------------------------------------------------------------
# Smalltalk: answer greetings naturally without firing the whole RAG pipeline.
# ---------------------------------------------------------------------------
_GREETING_RE = re.compile(
    r"\b(hi|hii|hello|hey|how are you|good morning|good evening|"
    r"habari|mambo|hujambo|shikamoo|salama|niaje)\b",
    re.IGNORECASE,
)
_GREETINGS = {
    "en": "Hello! I'm SautiSafari, your Zanzibar guide. Ask me about beaches, "
          "food, history or how to get somewhere.",
    "sw": "Habari! Mimi ni SautiSafari, mwongozaji wako wa Zanzibar. Niulize "
          "kuhusu fukwe, chakula, historia au jinsi ya kufika mahali.",
}


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text)) and len(text.split()) <= 6


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tab 1 — Assistant
# ---------------------------------------------------------------------------
def run_assistant(audio, text, lat, lon):
    if audio:
        query, lang = asr.transcribe(audio)
    else:
        query = (text or "").strip()
        lang = translate.detect_lang(query)

    if not query:
        return ("(no input detected)", "Please speak or type a question.",
                "", None, [], "")

    user_lat, user_lon = _to_float(lat), _to_float(lon)

    if _is_greeting(query):
        answer_text = _GREETINGS[lang]
        hits = []
    else:
        hits = rag.best_above_threshold(query) or []
        answer_text = llm.answer(query, hits, lang)
        note = geo.describe_location(hits, user_lat, user_lon, lang)
        if note:
            answer_text = f"{answer_text}\n\n{note}"

    # Bilingual mirror so a companion who speaks the other language follows along.
    other_lang = translate.other(lang)
    answer_other = translate.translate(answer_text, lang, other_lang)

    audio_out = tts.speak(answer_text, lang)

    evidence = [
        [h["name"], h.get("score", ""), h.get("lat", ""), h.get("lon", ""), h.get("source", "")]
        for h in hits
    ]

    transcript = f"[{lang}] {query}"
    top_place = hits[0]["name"] if hits else ""
    return transcript, answer_text, answer_other, audio_out, evidence, top_place


# ---------------------------------------------------------------------------
# Tab 2 — Translator
# ---------------------------------------------------------------------------
def run_translator(audio, text, direction):
    if audio:
        source_text, detected = asr.transcribe(audio)
    else:
        source_text = (text or "").strip()
        detected = translate.detect_lang(source_text)

    if not source_text:
        return "(no input detected)", "", None

    if direction == "Auto detect":
        src = detected
        tgt = translate.other(src)
    elif direction == "English → Swahili":
        src, tgt = "en", "sw"
    else:  # Swahili → English
        src, tgt = "sw", "en"

    translated = translate.translate(source_text, src, tgt)
    audio_out = tts.speak(translated, tgt)
    shown = f"[{src}] {source_text}"
    return shown, f"[{tgt}] {translated}", audio_out


# ---------------------------------------------------------------------------
# Tab 3 — Map (live directions live inside the iframe page)
# ---------------------------------------------------------------------------
def map_iframe(dest: str = "") -> str:
    qs = f"?dest={dest}" if dest else ""
    return (
        f'<iframe src="/static/map.html{qs}" allow="geolocation; gyroscope" '
        'style="width:100%;height:620px;border:0;border-radius:14px;'
        'box-shadow:0 2px 18px rgba(0,0,0,.12);"></iframe>'
    )


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    geo_js = """() => new Promise((resolve) => {
        if (!navigator.geolocation) { resolve(['', '']); return; }
        navigator.geolocation.getCurrentPosition(
            p => resolve([String(p.coords.latitude), String(p.coords.longitude)]),
            () => resolve(['', ''])
        );
    })"""

    with gr.Blocks(title="SautiSafari — Zanzibar Voice Guide") as demo:
        gr.Markdown(
            "# 🌍 SautiSafari\n"
            "**Voice guide & live wayfinder for Zanzibar.** Speak English or "
            "Kiswahili — get grounded answers, two-way translation, and "
            "turn-by-turn directions."
        )

        # ---- Assistant ----------------------------------------------------
        with gr.Tab("🎙️ Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    a_audio = gr.Audio(sources=["microphone", "upload"],
                                       type="filepath", label="Speak your question")
                    a_text = gr.Textbox(label="…or type it",
                                        placeholder="Where can I see giant tortoises?")
                    with gr.Row():
                        a_lat = gr.Textbox(label="My latitude", scale=1)
                        a_lon = gr.Textbox(label="My longitude", scale=1)
                    a_loc_btn = gr.Button("📍 Use my location", size="sm")
                    a_go = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    a_transcript = gr.Textbox(label="Understood as")
                    a_answer = gr.Textbox(label="Answer", lines=4)
                    a_answer2 = gr.Textbox(label="Also in the other language", lines=4)
                    a_audio_out = gr.Audio(label="Spoken answer", autoplay=True)
                    a_top = gr.Textbox(label="Top place (paste into Map tab)")
            a_evidence = gr.Dataframe(
                headers=["Place", "Score", "Lat", "Lon", "Source"],
                label="Evidence (retrieved places)", wrap=True,
            )
            a_loc_btn.click(fn=None, inputs=None, outputs=[a_lat, a_lon], js=geo_js)
            a_go.click(
                run_assistant,
                inputs=[a_audio, a_text, a_lat, a_lon],
                outputs=[a_transcript, a_answer, a_answer2, a_audio_out, a_evidence, a_top],
            )

        # ---- Translator ---------------------------------------------------
        with gr.Tab("🔁 Translator"):
            gr.Markdown(
                "Talk to locals. Speak/type in one language, get the other back "
                "as **text and speech**.")
            with gr.Row():
                with gr.Column():
                    t_audio = gr.Audio(sources=["microphone", "upload"],
                                       type="filepath", label="Speak")
                    t_text = gr.Textbox(label="…or type")
                    t_dir = gr.Radio(
                        ["Auto detect", "English → Swahili", "Swahili → English"],
                        value="Auto detect", label="Direction")
                    t_go = gr.Button("Translate", variant="primary")
                with gr.Column():
                    t_source = gr.Textbox(label="Heard / source")
                    t_out = gr.Textbox(label="Translation", lines=3)
                    t_audio_out = gr.Audio(label="Spoken translation", autoplay=True)
            gr.Examples(
                examples=[["How much is this?"], ["Where is the bus stop?"],
                          ["Asante sana"], ["Bei gani?"]],
                inputs=[t_text],
            )
            t_go.click(
                run_translator,
                inputs=[t_audio, t_text, t_dir],
                outputs=[t_source, t_out, t_audio_out],
            )

        # ---- Map ----------------------------------------------------------
        with gr.Tab("🗺️ Map & Directions"):
            gr.Markdown(
                "Tap **Locate me** on the map, search a place, then **Directions** "
                "for live distance and turn-by-turn routing.")
            with gr.Row():
                m_dest = gr.Textbox(label="Destination", scale=3,
                                    placeholder="Prison Island")
                m_go = gr.Button("Open in map", scale=1, variant="primary")
            m_html = gr.HTML(map_iframe())
            m_go.click(lambda d: map_iframe(d), inputs=[m_dest], outputs=[m_html])

        try:
            kb_count = rag.kb_size()
            search_status = rag.embedding_status()
        except Exception:
            kb_count = "?"
            search_status = "unavailable"
        gr.Markdown(
            f"<sub>KB places loaded: **{kb_count}** · LLM: "
            f"`{cfg.LLM_PROVIDER}` · Search: `{search_status}` · "
            f"ASR: `whisper-{cfg.ASR_MODEL}`</sub>")
    return demo


# ---------------------------------------------------------------------------
# FastAPI host (serves Gradio + the static map page)
# ---------------------------------------------------------------------------
def create_app():
    """Build FastAPI + Gradio app.  Called inside __main__ so heavy ML models
    (torch, sentence-transformers) load only when actually running, not on
    import."""
    server = FastAPI(title="SautiSafari")
    _static_dir = os.path.join(os.path.dirname(__file__), "static")
    server.mount("/static", StaticFiles(directory=_static_dir), name="static")
    server = gr.mount_gradio_app(server, build_ui(), path="/")
    return server


if __name__ == "__main__":
    import uvicorn
    print(f"\n  SautiSafari running → http://{cfg.HOST}:{cfg.PORT}\n")
    app = create_app()
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT)
