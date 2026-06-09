# SautiSafari v2 Project Notes

## Problem statement

Tourists in Zanzibar and Tanzania need quick information about beaches, heritage sites, food places, wildlife, transport gateways, and geographical locations. Many users may prefer speaking in Kiswahili or English. This project builds a Swahili/English voice assistant that recognizes speech, retrieves grounded information from local tourism/geographical datasets, and responds by text and speech.

## Final architecture

Voice/Text Input → Whisper ASR → Language/Intent Detection → RAG Retriever → Optional Local LLM → Answer + Evidence Table → gTTS Speech Output

## Why RAG is used

RAG is better than a plain chatbot because tourism facts should be grounded in a database. The assistant retrieves relevant rows from the local knowledge base and uses those rows as evidence. If retrieval confidence is low, it refuses to invent information.

## Datasets

1. Curated seed knowledge base: small manually prepared CSV for same-day demo.
2. UNDP GeoHub Zanzibar Tourism Attractions: main target dataset for Zanzibar attractions.
3. OpenStreetMap Overpass: tourist POIs, historic places, beaches, restaurants, ferry terminals.
4. GeoNames Tanzania: place names, coordinates, and administrative/geographic features.
5. Common Voice Swahili/FLEURS Swahili: optional ASR fine-tuning and evaluation.

## Evaluation plan

### RAG evaluation

Prepare 50–100 questions with expected top answer/place. Measure:

- Top-1 retrieval accuracy
- Top-3 retrieval accuracy
- Average retrieval confidence

### ASR evaluation

Record 20–50 Swahili tourist questions. Measure:

- Word Error Rate (WER)
- Character Error Rate (CER)

### Human answer quality

Ask 3–5 reviewers to score answers from 1–5:

- Correctness
- Helpfulness
- Language quality
- Grounding in retrieved evidence

## Current limitation

The default app does not invent detailed opening hours, ticket prices, or live transport times. Add live APIs or verified datasets if those answers are required.
