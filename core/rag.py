"""
Retrieval-Augmented Generation knowledge base.

Loads every CSV in the data directory, normalises wildly different schemas
(curated KB, OSM, GeoNames, UNDP) into a common record, embeds them with a
multilingual sentence transformer, and retrieves the most relevant places for
a query using cosine similarity plus a small keyword-overlap boost.

The loader is deliberately tolerant: it sniffs columns by name so you can drop
new CSVs into data/ without touching code.
"""
from __future__ import annotations

import glob
import os
import re
import threading
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .config import cfg

_model = None
_records: List[Dict[str, Any]] = []
_matrix: np.ndarray | None = None
_embedding_error: str | None = None
_lock = threading.Lock()

# Column-name candidates, in priority order, for each normalised field.
_NAME_COLS = ["name", "title", "place", "attraction", "feature", "label"]
_TEXT_COLS = ["description", "summary", "text", "details", "info", "about", "notes"]
_CAT_COLS = ["category", "type", "kind", "class", "tags", "fclass"]
_LAT_COLS = ["lat", "latitude", "y"]
_LON_COLS = ["lon", "lng", "long", "longitude", "x"]
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "where", "what",
    "which", "about", "near", "can", "see", "visit", "how", "kwa", "na",
    "ni", "ya", "za", "cha", "wapi", "gani", "kuhusu", "tafadhali",
}


def _pick(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        for col in row.index:
            if col.lower() == c and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()
    return ""


def _pick_float(row: pd.Series, candidates: List[str]):
    for c in candidates:
        for col in row.index:
            if col.lower() == c and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    continue
    return None


def _load_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pattern = os.path.join(cfg.DATA_DIR, "*.csv")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if fname.startswith("evaluation"):
            continue  # eval questions are not knowledge
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        source = fname.replace(".csv", "")
        for _, row in df.iterrows():
            name = _pick(row, _NAME_COLS)
            desc = _pick(row, _TEXT_COLS)
            cat = _pick(row, _CAT_COLS)
            if not name and not desc:
                # As a last resort concatenate every string cell.
                desc = " ".join(
                    str(v) for v in row.values if isinstance(v, str)
                ).strip()
            if not name and not desc:
                continue
            doc = " | ".join(p for p in [name, cat, desc] if p)
            records.append(
                {
                    "name": name or desc[:60],
                    "description": desc,
                    "category": cat,
                    "lat": _pick_float(row, _LAT_COLS),
                    "lon": _pick_float(row, _LON_COLS),
                    "source": source,
                    "doc": doc,
                }
            )
    return records


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(cfg.EMBED_MODEL)
    return _model


def _ensure_index():
    """Build the embedding matrix once (thread-safe)."""
    global _records, _matrix, _embedding_error
    if _matrix is not None:
        return
    with _lock:
        if _matrix is not None:
            return
        _records = _load_records()
        if not _records:
            _matrix = np.zeros((0, 384), dtype=np.float32)
            return
        try:
            model = _get_model()
            embs = model.encode(
                [r["doc"] for r in _records],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            _matrix = np.asarray(embs, dtype=np.float32)
            _embedding_error = None
        except Exception as exc:
            # PyTorch can fail to load on unsupported Python/Windows builds.
            # Keep the app useful by falling back to lexical retrieval.
            _matrix = np.zeros((0, 0), dtype=np.float32)
            _embedding_error = f"{type(exc).__name__}: {exc}"


def kb_size() -> int:
    _ensure_index()
    return len(_records)


def embedding_status() -> str:
    _ensure_index()
    return "lexical fallback" if _embedding_error else "embeddings"


def _keyword_boost(query: str, doc: str) -> float:
    q = set(w for w in query.lower().split() if len(w) > 2)
    d = set(doc.lower().split())
    if not q:
        return 0.0
    return 0.05 * (len(q & d) / len(q))


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-zA-Z][a-zA-Z']+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _lexical_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    q = _tokens(query)
    if not q:
        return []

    ranked = []
    for idx, rec in enumerate(_records):
        doc = rec["doc"]
        d = _tokens(doc)
        if not d:
            continue
        overlap = len(q & d)
        name_hits = len(q & _tokens(rec.get("name", "")))
        category_hits = len(q & _tokens(rec.get("category", "")))
        if overlap == 0 and name_hits == 0 and category_hits == 0:
            continue
        score = (overlap / len(q)) + (0.35 * name_hits) + (0.2 * category_hits)
        ranked.append((score, idx))

    ranked.sort(reverse=True)
    out = []
    for score, idx in ranked[:top_k]:
        rec = dict(_records[idx])
        rec["score"] = round(float(score), 4)
        out.append(rec)
    return out


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """Return the top-k records, each with a `score`, sorted high to low."""
    _ensure_index()
    if _matrix is None or len(_records) == 0 or not query.strip():
        return []

    top_k = top_k or cfg.TOP_K
    if _embedding_error:
        return _lexical_retrieve(query, top_k)

    model = _get_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = _matrix @ q_emb  # cosine, since everything is normalised

    results = []
    for idx, base in enumerate(sims):
        score = float(base) + _keyword_boost(query, _records[idx]["doc"])
        results.append((score, idx))
    results.sort(reverse=True)

    out = []
    for score, idx in results[:top_k]:
        rec = dict(_records[idx])
        rec["score"] = round(score, 4)
        out.append(rec)
    return out


def best_above_threshold(query: str):
    """Top result only if it clears RAG_THRESHOLD, else None (anti-hallucination)."""
    hits = retrieve(query, top_k=cfg.TOP_K)
    if hits and hits[0]["score"] >= cfg.RAG_THRESHOLD:
        return hits
    return None
