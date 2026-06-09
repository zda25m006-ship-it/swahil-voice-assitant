"""
SautiSafari v2: Swahili/English Voice Assistant for Zanzibar/Tanzania Tourists
----------------------------------------------------------------------------
Features:
1) Voice or text input in Kiswahili/English
2) Whisper ASR, loaded only when microphone/upload audio is used
3) RAG retrieval over local CSV knowledge bases
4) Small-talk/intent handling so greetings do not trigger tourism retrieval
5) Optional local LLM through Ollama for RAG answer generation
6) Spoken answer using gTTS

Run:
    python app.py

Optional local LLM:
    1) Install Ollama: https://ollama.com
    2) ollama pull llama3.2:3b
    3) set LLM_PROVIDER=ollama
    4) python app.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Models. Use whisper-base or whisper-tiny on slow CPU.
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Optional LLM provider. Default is no external LLM so the project runs today without keys.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower().strip()  # none | ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

ENABLE_TTS = os.getenv("ENABLE_TTS", "1") == "1"
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "900"))
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", "0.16"))
TOP_K = int(os.getenv("TOP_K", "5"))

SW_HINT_WORDS = {
    "habari", "hujambo", "sijambo", "mambo", "poa", "niko", "wapi", "karibu", "naenda",
    "nataka", "ni", "ipi", "gani", "bei", "safari", "chakula", "fukwe", "bahari", "tembelea",
    "tanzania", "zanzibar", "mji", "mkongwe", "msaada", "tafadhali", "asante", "ndiyo", "hapana",
}

TOURISM_SCOPE_WORDS = {
    "zanzibar", "tanzania", "stone", "town", "forodhani", "nungwi", "kendwa", "jozani", "paje", "jambiani",
    "serengeti", "kilimanjaro", "ngorongoro", "dar", "salaam", "dodoma", "beach", "hotel", "food", "restaurant",
    "history", "heritage", "museum", "safari", "wildlife", "airport", "ferry", "island", "spice", "farm", "tour",
    "fukwe", "chakula", "historia", "wanyama", "msitu", "kisiwa", "bahari", "mji", "mkongwe", "wapi", "tembelea",
}

SMALLTALK_PATTERNS = [
    r"^hi$", r"^hello$", r"^hey$", r"^how are you\??$", r"^how r u\??$", r"^how are u\??$",
    r"^habari\??$", r"^hujambo\??$", r"^mambo\??$", r"^asante$", r"^thank you$", r"^thanks$",
    r"^who are you\??$", r"^what is your name\??$", r"^jina lako ni nani\??$",
]

asr_pipe = None
embedder = None
kb_df: Optional[pd.DataFrame] = None
kb_embeddings = None


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_lang(text: str) -> str:
    text_l = (text or "").lower()
    words = set(re.sub(r"[^\w\s]", " ", text_l).split())
    if words & SW_HINT_WORDS:
        return "sw"
    # Very small heuristic: common Swahili fragments
    if any(fragment in text_l for fragment in ["nina", "nataka", "tafadhali", "wapi", "gani", "karibu"]):
        return "sw"
    return "en"


def get_smalltalk_answer(query: str, lang: str) -> Optional[str]:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    q = q.replace("!", "").replace(".", "")
    for pattern in SMALLTALK_PATTERNS:
        if re.match(pattern, q):
            if lang == "sw":
                if "asante" in q:
                    return "Karibu! Unaweza kuniuliza kuhusu maeneo ya kutembelea Zanzibar au Tanzania."
                if "nani" in q or "jina" in q:
                    return "Mimi ni SautiSafari, msaidizi wa sauti kwa watalii wa Zanzibar na Tanzania."
                return "Niko vizuri, asante! Unaweza kuniuliza kuhusu fukwe, historia, chakula, safari au maeneo ya Tanzania."
            else:
                if "thank" in q:
                    return "You are welcome! Ask me about places to visit in Zanzibar or Tanzania."
                if "who" in q or "name" in q:
                    return "I am SautiSafari, a tourist voice assistant for Zanzibar and Tanzania."
                return "I am good, thanks! Ask me about beaches, history, food, safaris, or places in Tanzania."
    return None


def looks_tourism_related(query: str) -> bool:
    words = set(re.sub(r"[^\w\s]", " ", (query or "").lower()).split())
    return bool(words & TOURISM_SCOPE_WORDS)


def standardize_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Convert different source CSVs into one RAG schema."""
    original_cols = list(df.columns)
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]

    def pick(*names, default=""):
        for n in names:
            n = norm_col(n)
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df))

    name = pick("name", "title", "attraction", "attraction_name", "site_name", "asciiname")
    lat = pick("latitude", "lat", "y", "center_lat")
    lon = pick("longitude", "lon", "lng", "x", "center_lon")
    category = pick("category", "tourism", "historic", "amenity", "feature_code", "type", "class", default="place")
    region = pick("region", "admin1", "district", "admin1_code", "location", default="Tanzania/Zanzibar")
    description_en = pick("description_en", "description", "desc", "about", "summary", default="")
    description_sw = pick("description_sw", "maelezo", default="")
    tips = pick("tips", "tourist_tip", "note", "notes", default="")
    keywords = pick("keywords", "tags", "raw_tags", "alternatenames", default="")

    out = pd.DataFrame({
        "name": name.map(clean_text),
        "category": category.map(clean_text).replace("", "place"),
        "region": region.map(clean_text).replace("", "Tanzania/Zanzibar"),
        "latitude": pd.to_numeric(lat, errors="coerce"),
        "longitude": pd.to_numeric(lon, errors="coerce"),
        "description_sw": description_sw.map(clean_text),
        "description_en": description_en.map(clean_text),
        "tips": tips.map(clean_text),
        "source": source_name,
        "keywords": keywords.map(clean_text),
    })

    # Build a useful description for raw OSM/GeoNames rows.
    mask_empty = out["description_en"].str.len() == 0
    out.loc[mask_empty, "description_en"] = (
        out.loc[mask_empty, "name"] + " is listed as a " + out.loc[mask_empty, "category"].astype(str) +
        " place in " + out.loc[mask_empty, "region"].astype(str) + "."
    )
    out.loc[out["description_sw"].str.len() == 0, "description_sw"] = out.loc[out["description_sw"].str.len() == 0, "description_en"]
    out.loc[out["tips"].str.len() == 0, "tips"] = "Verify opening hours, prices, transport, and local guidance before visiting."

    out = out[out["name"].str.len() > 1].copy()
    out["id"] = [f"{source_name}-{i}" for i in range(len(out))]
    return out


def load_knowledge_base() -> pd.DataFrame:
    """Load all available CSV knowledge sources from data/."""
    files = [
        DATA_DIR / "knowledge_base.csv",
        DATA_DIR / "undp_zanzibar_attractions.csv",
        DATA_DIR / "osm_zanzibar_tourism.csv",
        DATA_DIR / "osm_tanzania_tourism.csv",
        DATA_DIR / "geonames_tanzania.csv",
        DATA_DIR / "custom_knowledge_base.csv",
    ]

    frames = []
    for path in files:
        if path.exists():
            try:
                raw = pd.read_csv(path)
                frames.append(standardize_frame(raw, path.stem))
            except Exception as exc:
                print(f"[WARN] Could not load {path}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No CSV knowledge base files found in {DATA_DIR}")

    df = pd.concat(frames, ignore_index=True)
    df["name_norm"] = df["name"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    # De-duplicate conservatively but keep different coordinates when present.
    df["lat_round"] = df["latitude"].round(3)
    df["lon_round"] = df["longitude"].round(3)
    df = df.drop_duplicates(subset=["name_norm", "lat_round", "lon_round"], keep="first")
    df = df.drop(columns=["name_norm", "lat_round", "lon_round"], errors="ignore")

    for col in ["description_sw", "description_en", "tips", "keywords", "category", "region", "source"]:
        df[col] = df[col].fillna("").map(clean_text)

    df["retrieval_text"] = (
        df["name"] + " | " + df["category"] + " | " + df["region"] + " | " +
        df["description_sw"] + " | " + df["description_en"] + " | " + df["tips"] + " | " +
        df["keywords"] + " | source: " + df["source"]
    )
    return df.reset_index(drop=True)


def get_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def load_rag_models():
    global embedder, kb_df, kb_embeddings
    if kb_df is None:
        kb_df = load_knowledge_base()
        print(f"[INFO] Loaded {len(kb_df)} knowledge rows.")
    if embedder is None:
        print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
        embedder = SentenceTransformer(EMBED_MODEL)
        kb_embeddings = embedder.encode(kb_df["retrieval_text"].tolist(), normalize_embeddings=True, show_progress_bar=False)


def load_asr_model():
    global asr_pipe
    if asr_pipe is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"[INFO] Loading ASR model: {ASR_MODEL}")
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL,
            dtype=dtype,
            device=get_device(),
        )


def keyword_bonus(query: str, text: str) -> float:
    q_words = set(w for w in re.sub(r"[^\w\s]", " ", query.lower()).split() if len(w) > 2)
    if not q_words:
        return 0.0
    t_words = set(w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) > 2)
    overlap = len(q_words & t_words)
    return min(0.08, 0.015 * overlap)


def retrieve(query: str, top_k: int = TOP_K) -> List[Tuple[pd.Series, float]]:
    load_rag_models()
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(q_emb, kb_embeddings)[0]

    # Add a tiny keyword boost so exact place names beat generic semantic matches.
    adjusted = []
    for i, s in enumerate(scores):
        adjusted.append(float(s) + keyword_bonus(query, kb_df.iloc[i]["retrieval_text"]))
    adjusted = np.array(adjusted)
    idxs = np.argsort(adjusted)[::-1][:top_k]
    return [(kb_df.iloc[i], float(adjusted[i])) for i in idxs]


def results_to_table(results: List[Tuple[pd.Series, float]]) -> pd.DataFrame:
    rows = []
    for row, score in results:
        rows.append({
            "name": row["name"],
            "category": row["category"],
            "region": row["region"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "score": round(score, 3),
            "source": row["source"],
            "tips": row["tips"],
        })
    return pd.DataFrame(rows)


def build_context(results: List[Tuple[pd.Series, float]], max_items: int = 5) -> str:
    chunks = []
    for rank, (row, score) in enumerate(results[:max_items], 1):
        chunks.append(
            f"[{rank}] Name: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"Region: {row['region']}\n"
            f"Coordinates: {row['latitude']}, {row['longitude']}\n"
            f"Swahili description: {row['description_sw']}\n"
            f"English description: {row['description_en']}\n"
            f"Tips: {row['tips']}\n"
            f"Source: {row['source']}\n"
            f"Retrieval score: {score:.3f}"
        )
    return "\n\n".join(chunks)


def template_answer(query: str, lang: str, results: List[Tuple[pd.Series, float]], detail_level: str) -> str:
    if not results:
        return "I do not have enough information yet." if lang == "en" else "Sina taarifa za kutosha bado."

    best, best_score = results[0]
    if best_score < RAG_THRESHOLD:
        if lang == "sw":
            return (
                "Samahani, sina taarifa za kutosha kwenye hifadhidata kwa swali hilo. "
                "Kwa sasa naweza kusaidia kuhusu maeneo ya utalii, fukwe, historia, chakula, safari na jiografia ya Zanzibar/Tanzania."
            )
        return (
            "I do not have enough reliable information in my current database for that question. "
            "Right now I can help with tourist places, beaches, history, food, safaris, and geography in Zanzibar/Tanzania."
        )

    other_names = [r[0]["name"] for r in results[1:3]]
    detailed = detail_level == "Detailed"

    if lang == "sw":
        answer = (
            f"{best['name']} ni chaguo bora kutoka kwenye hifadhidata yangu. "
            f"Ipo/linahusiana na {best['region']} na aina yake ni {best['category']}. "
            f"{best['description_sw']} "
        )
        if detailed:
            answer += f"Ushauri: {best['tips']} Mahali: latitudo {best['latitude']}, longitudo {best['longitude']}. "
            if other_names:
                answer += "Maeneo mengine yanayofanana: " + ", ".join(other_names) + "."
        else:
            answer += f"Ushauri mfupi: {best['tips']}"
        return answer.strip()

    answer = (
        f"{best['name']} is the best match from my database. "
        f"It is in/related to {best['region']} and its category is {best['category']}. "
        f"{best['description_en']} "
    )
    if detailed:
        answer += f"Tourist tip: {best['tips']} Location: latitude {best['latitude']}, longitude {best['longitude']}. "
        if other_names:
            answer += "Other related places: " + ", ".join(other_names) + "."
    else:
        answer += f"Quick tip: {best['tips']}"
    return answer.strip()


def ollama_answer(query: str, lang: str, context: str, detail_level: str) -> Optional[str]:
    if LLM_PROVIDER != "ollama":
        return None
    target_lang = "Kiswahili" if lang == "sw" else "English"
    length_rule = "2-3 short sentences" if detail_level == "Short" else "4-6 clear sentences"
    prompt = f"""
You are SautiSafari, a careful tourist voice assistant for Zanzibar and Tanzania.
Use ONLY the provided RAG context. Do not invent prices, opening hours, safety claims, or transport details.
Answer in {target_lang}. Keep the answer to {length_rule}.
If the context is insufficient, say that the database does not have enough information and suggest asking about Zanzibar/Tanzania tourist places.

RAG context:
{context}

User question:
{query}

Answer:
""".strip()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = clean_text(data.get("response", ""))
        return answer or None
    except Exception as exc:
        print(f"[WARN] Ollama generation failed: {exc}")
        return None


def make_answer(query: str, lang: str = "auto", detail_level: str = "Short") -> Tuple[str, pd.DataFrame, str]:
    if not query or not query.strip():
        msg = "Please ask a question by voice or text."
        return msg, pd.DataFrame(), ""

    if lang == "auto":
        lang = detect_lang(query)

    smalltalk = get_smalltalk_answer(query, lang)
    if smalltalk:
        return smalltalk, pd.DataFrame(), "Intent: small talk / greeting. No RAG retrieval needed."

    if not looks_tourism_related(query) and len(query.split()) <= 5:
        if lang == "sw":
            msg = "Naweza kusaidia zaidi ukiuliza kuhusu maeneo ya utalii, chakula, fukwe, historia au safari Tanzania/Zanzibar."
        else:
            msg = "I can help best when you ask about tourist places, food, beaches, history, or safaris in Tanzania/Zanzibar."
        return msg, pd.DataFrame(), "Intent: out of tourism scope."

    results = retrieve(query, top_k=TOP_K)
    table = results_to_table(results)
    context = build_context(results)

    # Use LLM only when retrieval confidence is adequate; otherwise avoid hallucination.
    llm_text = None
    if results and results[0][1] >= RAG_THRESHOLD:
        llm_text = ollama_answer(query, lang, context, detail_level)

    answer = llm_text or template_answer(query, lang, results, detail_level)
    return answer, table, context


def transcribe_audio(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    load_asr_model()
    try:
        out = asr_pipe(
            audio_path,
            generate_kwargs={"language": "swahili", "task": "transcribe"},
        )
        return clean_text(out.get("text", ""))
    except Exception as exc:
        return f"ASR error: {exc}"


def text_to_speech(text: str, lang: str) -> Optional[str]:
    if not ENABLE_TTS or not text:
        return None
    try:
        safe_text = text[:TTS_MAX_CHARS]
        tts_lang = "sw" if lang == "sw" else "en"
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        gTTS(text=safe_text, lang=tts_lang).save(out_path)
        return out_path
    except Exception as exc:
        print(f"[WARN] TTS failed: {exc}")
        return None


def respond(audio_path, typed_text, answer_language, detail_level):
    if typed_text and typed_text.strip():
        user_text = typed_text.strip()
    else:
        user_text = transcribe_audio(audio_path)

    lang = "sw" if answer_language == "Swahili" else "en" if answer_language == "English" else detect_lang(user_text)
    answer, table, context = make_answer(user_text, lang=lang, detail_level=detail_level)
    audio_out = text_to_speech(answer, lang)
    return user_text, answer, table, context, audio_out


EXAMPLES = [
    [None, "How are you?", "English", "Short"],
    [None, "Nataka kutembelea fukwe nzuri Zanzibar", "Auto", "Short"],
    [None, "Where can I see history in Stone Town?", "English", "Detailed"],
    [None, "Ni sehemu gani nzuri kwa chakula cha usiku Zanzibar?", "Swahili", "Short"],
    [None, "Tell me about Jozani forest", "English", "Detailed"],
]

with gr.Blocks(title="SautiSafari: RAG Voice Assistant") as demo:
    gr.Markdown(
        """
        # SautiSafari: Swahili Voice Assistant for Zanzibar/Tanzania Tourists
        Ask by microphone or text in **Kiswahili or English**. This version uses **RAG retrieval** over local tourism/geographical datasets and optional **local LLM generation with Ollama**.
        """
    )
    with gr.Row():
        audio_in = gr.Audio(label="Speak here / Rekodi sauti", sources=["microphone", "upload"], type="filepath")
        with gr.Column():
            typed_text = gr.Textbox(label="Or type your question", placeholder="Mfano: Nifike wapi kuona historia Zanzibar?")
            answer_language = gr.Radio(["Auto", "Swahili", "English"], value="Auto", label="Answer language")
            detail_level = gr.Radio(["Short", "Detailed"], value="Short", label="Answer length")
            btn = gr.Button("Ask Assistant", variant="primary")

    transcribed = gr.Textbox(label="Recognized / typed question")
    answer = gr.Textbox(label="Assistant answer", lines=6)
    table = gr.Dataframe(label="Top retrieved places / RAG evidence")
    context_box = gr.Textbox(label="RAG context used", lines=10, visible=True)
    audio_out = gr.Audio(label="Spoken answer", type="filepath")
    gr.Examples(examples=EXAMPLES, inputs=[audio_in, typed_text, answer_language, detail_level])
    btn.click(
        respond,
        inputs=[audio_in, typed_text, answer_language, detail_level],
        outputs=[transcribed, answer, table, context_box, audio_out],
    )

if __name__ == "__main__":
    load_rag_models()
    print("[INFO] SautiSafari v2 is ready.")
    print(f"[INFO] LLM_PROVIDER={LLM_PROVIDER}. Set LLM_PROVIDER=ollama for local LLM generation.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=os.getenv("GRADIO_SHARE", "0") == "1")
