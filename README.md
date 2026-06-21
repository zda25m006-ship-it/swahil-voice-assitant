# 🌍 SautiSafari v2 — Swahili RAG Voice Assistant for Zanzibar/Tanzania Tourists

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Gradio-4.44%2B-orange?style=flat-square&logo=gradio" />
  <img src="https://img.shields.io/badge/Whisper-small-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Language-Swahili%20%7C%20English-purple?style=flat-square" />
</p>

> **SautiSafari** (Swahili: *sauti* = voice, *safari* = journey) is a bilingual Kiswahili/English voice assistant for tourists visiting Zanzibar and Tanzania. It combines OpenAI **Whisper** for Automatic Speech Recognition (ASR), a **Retrieval-Augmented Generation (RAG)** pipeline over curated local tourism datasets, optional **local LLM generation via Ollama**, and **gTTS** text-to-speech — all served through a Gradio web interface.

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Models Used](#-models-used)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Dataset Collection Scripts](#-dataset-collection-scripts)
- [Evaluation](#-evaluation)
- [Fine-Tuning Whisper (Optional)](#-fine-tuning-whisper-optional)
- [Research References](#-research-references)
- [License](#-license)

---

## 🎬 Demo

Launch the app and open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

**Try these example queries:**

| Query | Language | Expected response |
|---|---|---|
| `How are you?` | English | Friendly greeting (no retrieval) |
| `Where can I see history in Stone Town?` | English | Stone Town heritage info |
| `Nataka kutembelea fukwe nzuri Zanzibar` | Swahili | Nungwi / Kendwa beach info |
| `Tell me about Jozani forest` | English | Red colobus monkeys, mangroves |
| `Ni sehemu gani nzuri kwa chakula cha usiku?` | Swahili | Forodhani Gardens food market |
| `Where can I see giant tortoises near Stone Town?` | English | Prison Island |

---

## 🏗 Architecture

```
User Voice / Text Input
        │
        ▼
┌────────────────────┐
│  Whisper ASR (sw)  │  ← openai/whisper-small
└────────────────────┘
        │  transcribed text
        ▼
┌────────────────────────────────────────┐
│  Language & Intent Detection           │
│  (Swahili word heuristics + regex)    │
└────────────────────────────────────────┘
        │
   ┌────┴──────┐
   │ Smalltalk │  → Template greeting response
   └───────────┘
        │ tourism query
        ▼
┌──────────────────────────────────────────────┐
│  RAG Retriever                               │
│  • Sentence-Transformer embeddings           │
│    (paraphrase-multilingual-MiniLM-L12-v2)  │
│  • Cosine similarity + keyword boost         │
│  • Top-K results from merged CSV knowledgebase│
└──────────────────────────────────────────────┘
        │  retrieved context
        ▼
   ┌────┴────────────────────┐
   │ Optional Ollama LLM     │  ← llama3.2:3b (local, offline)
   │ (grounded, no halluc.)  │
   └─────────────────────────┘
        │  or template answer (default)
        ▼
┌────────────────────┐
│  gTTS TTS Output   │  → spoken MP3
└────────────────────┘
        │
        ▼
  Gradio Web UI
  (answer + evidence table + RAG context)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Bilingual ASR** | Whisper speech recognition for Kiswahili & English |
| 📚 **RAG Retrieval** | Multilingual sentence-transformer retrieval over local tourism CSVs |
| 🛡 **No hallucination** | Refuses to answer below confidence threshold (`RAG_THRESHOLD`) |
| 💬 **Smalltalk handling** | Greetings answered naturally without spurious retrieval |
| 🔊 **TTS Output** | gTTS generates spoken MP3 answers in Swahili or English |
| 🤖 **Optional LLM** | Ollama (llama3.2:3b) for richer grounded answer generation |
| 🗺 **Multi-dataset KB** | Merges UNDP GeoHub + OSM Overpass + GeoNames + curated CSV |
| 📊 **Evidence table** | Shows top retrieved places, scores, coordinates, and sources |
| 🧪 **Evaluation scripts** | RAG Top-1/Top-3 accuracy + ASR WER/CER |

---

## 🤖 Models Used

### 1. Automatic Speech Recognition — OpenAI Whisper

| Property | Detail |
|---|---|
| **Model** | `openai/whisper-small` (244M params) |
| **HuggingFace Hub** | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| **Task** | Multilingual ASR → transcribe (Swahili/English) |
| **CPU alternative** | `openai/whisper-base` or `openai/whisper-tiny` (set `ASR_MODEL` env var) |
| **Paper** | Radford et al., 2022 — *Robust Speech Recognition via Large-Scale Weak Supervision* |

### 2. Sentence Embeddings — paraphrase-multilingual-MiniLM-L12-v2

| Property | Detail |
|---|---|
| **Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **HuggingFace Hub** | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| **Task** | Dense retrieval embeddings for RAG (50+ languages incl. Swahili) |
| **Dimensions** | 384 |
| **Paper** | Reimers & Gurevych, 2019 — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* |

### 3. Optional LLM — Llama 3.2 3B via Ollama

| Property | Detail |
|---|---|
| **Model** | `llama3.2:3b` (Meta, via Ollama) |
| **Ollama page** | [ollama.com/library/llama3.2](https://ollama.com/library/llama3.2) |
| **Role** | RAG-grounded answer generation (context-only, no hallucination) |
| **Hardware** | ~4 GB RAM; no GPU required for 3B |

### 4. Text-to-Speech — Google gTTS

| Property | Detail |
|---|---|
| **Library** | `gTTS` (Google Text-to-Speech) |
| **Languages** | `sw` (Swahili) and `en` (English) |
| **Output** | MP3 streamed through Gradio Audio component |

---

## 📦 Datasets

### Seed Knowledge Base (included in repo)

| File | Description | Entries |
|---|---|---|
| `data/knowledge_base.csv` | Hand-curated tourist destinations, landmarks, and cities | 15 |
| `data/evaluation_questions.csv` | 12 evaluation questions with expected answers | 12 |

### External Datasets (fetched via scripts)

| Dataset | Source | Link | Script |
|---|---|---|---|
| **UNDP GeoHub — Zanzibar Tourism Attractions** | UNDP GeoHub | [geohub.data.undp.org/data/4ca2ead…](https://geohub.data.undp.org/data/4ca2ead25b5903e8e1c7897f8f3bae38) | `scripts/fetch_undp_zanzibar.py` |
| **OpenStreetMap — Zanzibar Tourism POIs** | OSM Overpass API | [overpass-api.de](https://overpass-api.de) | `scripts/build_osm_dataset.py --area zanzibar` |
| **OpenStreetMap — Tanzania Tourism POIs** | OSM Overpass API | [overpass-api.de](https://overpass-api.de) | `scripts/build_osm_dataset.py --area tanzania` |
| **GeoNames — Tanzania Places** | GeoNames | [download.geonames.org/export/dump/TZ.zip](https://download.geonames.org/export/dump/TZ.zip) | `scripts/fetch_geonames_tanzania.py` |

### ASR Fine-Tuning / Evaluation Datasets

| Dataset | Source | Link | Usage |
|---|---|---|---|
| **FLEURS Swahili** (`sw_ke`) | Google / HuggingFace | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | Whisper fine-tuning (optional) |
| **Mozilla Common Voice — Swahili** | Mozilla | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/sw/datasets) | ASR WER/CER evaluation |

---

## 📁 Project Structure

```
swahili_voice_assistant_zanzibar/
│
├── app.py                        # Main Gradio application (ASR + RAG + TTS)
│
├── data/
│   ├── knowledge_base.csv        # Seed curated knowledge base (15 entries)
│   └── evaluation_questions.csv  # RAG evaluation questions (12 entries)
│
├── scripts/
│   ├── build_osm_dataset.py      # Fetch tourism POIs from OpenStreetMap Overpass
│   ├── fetch_geonames_tanzania.py# Download Tanzania places from GeoNames dump
│   ├── fetch_undp_zanzibar.py    # Fetch UNDP GeoHub Zanzibar attractions
│   ├── run_data_builders.py      # Run OSM + GeoNames builders in one command
│   ├── evaluate_rag.py           # RAG Top-1/Top-3 retrieval accuracy evaluation
│   ├── evaluate_asr.py           # ASR WER/CER evaluation on a small audio set
│   └── train_whisper_swahili.py  # Optional Whisper fine-tuning on FLEURS Swahili
│
├── report/
│   └── project_notes.md          # Architecture decisions and evaluation plan
│
├── requirements.txt              # Core runtime dependencies
├── requirements-train.txt        # Extra deps for ASR training/evaluation
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── run_gitbash.sh                # Helper to activate venv & run (Git Bash)
└── run_windows.bat               # Helper to activate venv & run (Windows CMD)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- (Optional) CUDA GPU for faster Whisper inference

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sautisafari-v2.git
cd sautisafari-v2
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash) or Linux/macOS:**
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or
source .venv/bin/activate        # Linux/macOS
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Copy environment config

```bash
cp .env.example .env
# Edit .env to change models or enable Ollama
```

### 5. Run the app

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:7860](http://127.0.0.1:7860)**.

---

## ⚙️ Configuration

All settings are controlled via environment variables (copy `.env.example` → `.env`):

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL` | `openai/whisper-small` | Whisper model size (`whisper-tiny`, `whisper-base`, `whisper-small`) |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-Transformer model for RAG embeddings |
| `LLM_PROVIDER` | `none` | Set to `ollama` to enable local LLM generation |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `ENABLE_TTS` | `1` | Set to `0` to disable gTTS speech output |
| `TTS_MAX_CHARS` | `900` | Max characters sent to gTTS (avoids timeout) |
| `RAG_THRESHOLD` | `0.16` | Minimum cosine similarity to return a retrieval answer |
| `TOP_K` | `5` | Number of RAG results to retrieve per query |

### Enabling Ollama LLM

```bash
# 1. Install Ollama from https://ollama.com
# 2. Pull the model
ollama pull llama3.2:3b

# 3a. Linux/macOS / Git Bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
python app.py

# 3b. Windows PowerShell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2:3b"
python app.py
```

---

## 🗺 Dataset Collection Scripts

### Fetch all datasets at once (OSM + GeoNames)

```bash
python scripts/run_data_builders.py
```

### OpenStreetMap POIs

```bash
# Zanzibar
python scripts/build_osm_dataset.py --area zanzibar --out data/osm_zanzibar_tourism.csv

# Tanzania mainland
python scripts/build_osm_dataset.py --area tanzania --out data/osm_tanzania_tourism.csv
```

### GeoNames Tanzania

```bash
python scripts/fetch_geonames_tanzania.py --out data/geonames_tanzania.csv
# Default: keeps places with population ≥ 1000 + mountains/water/parks
python scripts/fetch_geonames_tanzania.py --out data/geonames_tanzania.csv --min_population 5000
```

### UNDP GeoHub Zanzibar Attractions

```bash
# Auto-discovery (tries known GeoHub API endpoints)
python scripts/fetch_undp_zanzibar.py --out data/undp_zanzibar_attractions.csv

# Manual URL (copy from the GeoHub dataset page)
python scripts/fetch_undp_zanzibar.py \
  --url "PASTE_GEOHUB_API_OR_FILE_URL" \
  --out data/undp_zanzibar_attractions.csv

# If API returns FlatGeobuf (.fgb), install optional GIS deps first:
pip install geopandas pyogrio
```

After adding new CSVs, restart `python app.py` to reload the knowledge base.

---

## 🧪 Evaluation

### RAG Retrieval Accuracy

```bash
python scripts/evaluate_rag.py
```

Evaluates Top-1 and Top-3 retrieval accuracy against `data/evaluation_questions.csv`.
Results saved to `report/rag_eval_results.csv`.

**Expected output:**
```
RAG Evaluation
==============
Questions: 12
Top-1 accuracy: 0.xxx
Top-3 accuracy: 0.xxx
Saved: report/rag_eval_results.csv
```

### ASR Word Error Rate / Character Error Rate

Prepare a manifest CSV:

```csv
audio_path,reference
eval_audio/q1.wav,Nataka kutembelea Stone Town
eval_audio/q2.wav,Ni wapi naweza kuona kima punju
```

Install evaluation dependencies and run:

```bash
pip install -r requirements-train.txt
python scripts/evaluate_asr.py --manifest data/asr_eval_manifest.csv
```

Results saved to `report/asr_eval_results.csv`.

### Recommended Metrics

| Metric | Target | Tool |
|---|---|---|
| ASR WER | < 30% | `jiwer` |
| ASR CER | < 15% | `jiwer` |
| RAG Top-1 | > 70% | `scripts/evaluate_rag.py` |
| RAG Top-3 | > 85% | `scripts/evaluate_rag.py` |
| Human answer quality | ≥ 3.5/5 | Manual rating (1–5 scale) |

---

## 🎓 Fine-Tuning Whisper (Optional)

Fine-tune `openai/whisper-small` on FLEURS Swahili for improved Swahili ASR:

```bash
pip install -r requirements-train.txt
python scripts/train_whisper_swahili.py
```

- Requires a **CUDA GPU** (recommended: ≥ 8 GB VRAM)
- Training data: **google/fleurs** (`sw_ke` config — Swahili Kenya)
- Output model saved to `whisper-small-swahili-fleurs/final/`
- After training, update `ASR_MODEL=whisper-small-swahili-fleurs/final` in `.env`

---

## 📚 Research References

### Speech Recognition

1. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).**
   *Robust Speech Recognition via Large-Scale Weak Supervision.*
   OpenAI. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
   → **Whisper ASR model** used in this project.

2. **Pratap, V., Tjandra, A., Shi, B., et al. (2023).**
   *Scaling Speech Technology to 1,000+ Languages.*
   Meta AI Research. [arXiv:2305.13516](https://arxiv.org/abs/2305.13516)
   → Background on multilingual speech systems and low-resource language support.

### Retrieval-Augmented Generation (RAG)

3. **Lewis, P., Perez, E., Piktus, A., et al. (2020).**
   *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
   Facebook AI Research. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
   → Foundational RAG architecture paper motivating this project's retrieval design.

4. **Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. (2020).**
   *REALM: Retrieval-Augmented Language Model Pre-Training.*
   Google Research. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
   → Related RAG pre-training methodology.

### Sentence Embeddings & Dense Retrieval

5. **Reimers, N., & Gurevych, I. (2019).**
   *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
   EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
   → **paraphrase-multilingual-MiniLM-L12-v2** embedding model.

6. **Reimers, N., & Gurevych, I. (2020).**
   *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.*
   EMNLP 2020. [arXiv:2004.09813](https://arxiv.org/abs/2004.09813)
   → Multilingual extension of SBERT enabling Swahili retrieval.

### Swahili NLP & Low-Resource Languages

7. **Adelani, D. I., Abbott, J., Neubig, G., et al. (2021).**
   *MasakhaNER: Named Entity Recognition for African Languages.*
   TACL 2021. [arXiv:2103.11811](https://arxiv.org/abs/2103.11811)
   → Context for African/Swahili NLP challenges.

8. **Ochieng', W., Ndungu, P., & Gitau, S. (2021).**
   *Benchmarking Swahili Speech Recognition.*
   Interspeech 2021.
   → Baseline WER/CER metrics for Swahili ASR systems.

### Voice Assistants & Tourism Applications

9. **Gao, J., Galley, M., & Li, L. (2019).**
   *Neural Approaches to Conversational AI.*
   ACL 2019 Tutorial. [arXiv:1809.08267](https://arxiv.org/abs/1809.08267)
   → Overview of dialogue system architectures motivating the assistant design.

10. **Koreeda, Y., & Manning, C. D. (2021).**
    *Constrained Language Models Yield Few-Shot Semantic Parsers.*
    EMNLP 2021. [arXiv:2104.08768](https://arxiv.org/abs/2104.08768)
    → Grounded generation and hallucination avoidance strategies.

### Datasets Referenced

11. **Ardila, R., Branson, M., Davis, K., et al. (2020).**
    *Common Voice: A Massively-Multilingual Speech Corpus.*
    LREC 2020. [arXiv:1912.06670](https://arxiv.org/abs/1912.06670)
    → Mozilla Common Voice Swahili corpus for ASR evaluation.

12. **Conneau, A., Ma, M., Khanuja, S., et al. (2023).**
    *FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech.*
    SLT 2022. [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)
    → **FLEURS Swahili** (`sw_ke`) used for optional Whisper fine-tuning.

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `No CSV knowledge base files found` | Run `python scripts/run_data_builders.py` first, or use only `data/knowledge_base.csv` |
| Slow ASR on CPU | Set `ASR_MODEL=openai/whisper-tiny` or `openai/whisper-base` in `.env` |
| `TTS failed: [Errno …]` | Disable TTS with `ENABLE_TTS=0` if offline; check network for gTTS |
| Ollama not connecting | Check `ollama serve` is running; verify `OLLAMA_URL` in `.env` |
| FlatGeobuf read error for UNDP | Run `pip install geopandas pyogrio` then re-run the fetch script |
| Low retrieval accuracy | Add more rows to `data/knowledge_base.csv` or lower `RAG_THRESHOLD` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the multilingual ASR model
- [Sentence Transformers](https://www.sbert.net/) for multilingual dense retrieval
- [Gradio](https://gradio.app/) for the rapid web interface
- [Ollama](https://ollama.com/) for offline local LLM inference
- [UNDP GeoHub](https://geohub.data.undp.org/) for open Zanzibar tourism data
- [OpenStreetMap](https://www.openstreetmap.org/) contributors for POI data
- [GeoNames](https://www.geonames.org/) for Tanzania place data
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) and [Google FLEURS](https://huggingface.co/datasets/google/fleurs) for Swahili speech data

---

<p align="center">Made with ❤️ for Zanzibar & Tanzania tourists</p>
