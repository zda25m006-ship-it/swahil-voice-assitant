# SautiSafari Advanced

SautiSafari Advanced is a bilingual voice guide and live wayfinder for tourists in Zanzibar. It helps a visitor ask questions in English or Kiswahili, get grounded tourism answers, translate conversations, find nearby places, and open a live map with directions.

Repository: https://github.com/zda25m006-ship-it/swahil-voice-assitant

## What This Version Adds

- Voice and text assistant for Zanzibar tourism questions.
- English to Swahili and Swahili to English translation.
- Spoken answers using text-to-speech.
- Local tourism knowledge base from CSV files in `data/`.
- LLM-grounded answers through Ollama, OpenAI-compatible APIs, or template fallback.
- Browser geolocation for the tourist's current position.
- Distance notes in assistant answers when latitude and longitude are available.
- Live Leaflet map in `static/map.html`.
- Zanzibar-biased place search through Nominatim.
- Turn-by-turn road directions through OSRM.
- Retrieval fallback: if PyTorch or `sentence-transformers` cannot load, the app keeps working with lexical CSV search.

## Main Features

### Assistant

Ask by microphone or by typing. The assistant detects English or Kiswahili, searches the local tourism knowledge base, answers from retrieved context, speaks the result, and shows the same answer in the other language.

If you click **Use my location**, the assistant can add a distance or route note for the top matching place.

Example questions:

```text
Where can I see giant tortoises?
What food should I try in Stone Town?
Ninaweza kuona fukwe nzuri wapi?
How do I get to Prison Island?
```

### Translator

Use the translator tab for tourist-local conversations:

- English to Swahili
- Swahili to English
- Auto detect
- Voice input or typed input
- Text and spoken output

### Map And Directions

The map tab provides:

- Locate me
- Search for places in Zanzibar
- Destination marker
- Driving route
- Distance and estimated travel time
- Turn-by-turn steps

The map uses public OpenStreetMap tiles, Nominatim search, and OSRM routing.

## How It Works

```text
microphone or typed text
        |
        +--> faster-whisper ASR for voice
        +--> typed-text language heuristic
        |
        v
Assistant tab:
  CSV knowledge base -> semantic retrieval or lexical fallback -> grounded LLM/template answer
  -> optional distance/route note -> translation mirror -> speech output

Translator tab:
  source text -> Google Translate or local LLM -> speech output

Map tab:
  browser GPS -> Nominatim search -> OSRM route -> Leaflet map
```

## Project Structure

```text
sautisafari_advanced/
├── app.py                 # FastAPI + Gradio application
├── core/
│   ├── asr.py             # faster-whisper speech recognition
│   ├── config.py          # environment-driven settings
│   ├── geo.py             # distance, geocoding, OSRM routing helpers
│   ├── llm.py             # grounded answers with Ollama/OpenAI/template fallback
│   ├── rag.py             # CSV retrieval with embedding and lexical fallback
│   ├── translate.py       # English/Swahili translation
│   └── tts.py             # gTTS speech output
├── data/
│   ├── README.txt
│   └── knowledge_base.csv
├── static/
│   └── map.html           # live GPS map and directions UI
├── .env.example           # configuration template
├── requirements.txt       # Python dependencies
├── run_windows.bat        # Windows launcher
├── run.sh                 # Linux/macOS/Git Bash launcher
└── README.md
```

## Requirements

- Python 3.11 or 3.12 recommended.
- Python 3.13 can run the app, but PyTorch may fail to load on some Windows setups. This project now falls back to lexical search when that happens.
- Ollama for local LLM answers.
- Internet access for Google translation, map tiles, Nominatim search, OSRM routing, and gTTS.

## Setup

### 1. Clone The Repository

```bash
git clone https://github.com/zda25m006-ship-it/swahil-voice-assitant.git
cd swahil-voice-assitant
```

### 2. Create A Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash, Linux, or macOS:

```bash
python -m venv .venv
source .venv/Scripts/activate
# On Linux/macOS, use this instead:
# source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The default settings are enough for local development.

### 5. Install And Start Ollama

Install Ollama from https://ollama.com, then run:

```bash
ollama pull llama3.2:3b
ollama serve
```

If `ollama serve` says port `11434` is already in use, Ollama is already running. Do not start a second copy.

To run without Ollama first, set this in `.env`:

```env
LLM_PROVIDER=none
```

### 6. Run The App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

Allow location access in the browser if you want live distance and directions.

## Quick Launch Scripts

Windows:

```bat
run_windows.bat
```

Linux/macOS/Git Bash:

```bash
./run.sh
```

## Configuration

Edit `.env` to tune the app:

```env
ASR_MODEL=small
ASR_DEVICE=cpu
ASR_COMPUTE=int8

DATA_DIR=data
RAG_THRESHOLD=0.16
TOP_K=5

TRANSLATE_PROVIDER=google

LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://localhost:11434/api/generate

ENABLE_TTS=1

NOMINATIM_URL=https://nominatim.openstreetmap.org/search
OSRM_URL=https://router.project-osrm.org
MAP_CENTER_LAT=-6.1659
MAP_CENTER_LON=39.2026
```

## Knowledge Base

Tourism data lives in `data/*.csv`. The loader accepts flexible column names, including:

- `name`, `title`, `place`, `attraction`
- `description`, `summary`, `text`, `details`
- `category`, `type`, `kind`
- `lat`, `latitude`
- `lon`, `lng`, `longitude`

Add more Zanzibar attractions, hotels, beaches, restaurants, cultural sites, or transport locations to improve answers.

## Troubleshooting

| Problem | Fix |
|---|---|
| `WinError 1114` or `c10.dll` from PyTorch | Use Python 3.11/3.12, or keep running with the built-in lexical fallback. |
| `ollama serve` says port `11434` is in use | Ollama is already running. Start only the app with `python app.py`. |
| Map is blank | Check internet access; map tiles load from OpenStreetMap. |
| Location button fails | Allow browser location permission. Use `localhost` or a trusted browser context. |
| Directions fail | OSRM routes roads only. Some island or boat routes may not be available. |
| Translation returns original text | Google translation may be blocked/offline; try `TRANSLATE_PROVIDER=llm`. |
| KB places loaded is `0` | Put CSV files inside `data/` and restart the app. |
| Gradio font/static 404 messages appear | Usually harmless Gradio asset noise; focus on Python traceback errors. |

## Deployment Notes

For phone testing on the same Wi-Fi, set:

```env
HOST=0.0.0.0
```

Then open:

```text
http://YOUR-COMPUTER-IP:7860
```

For production traffic, do not rely on public demo routing/geocoding services. Self-host OSRM and use a proper Nominatim setup or another geocoding provider. Also set `GEO_USER_AGENT` in `.env` to a real project/contact value.

## Recent Improvements

- Added geographic location support.
- Added live map and destination search.
- Added OSRM directions and route steps.
- Added distance notes in assistant answers.
- Added bilingual text and speech translation.
- Added greeting handling for short messages such as `hi` and `hii`.
- Added fallback retrieval when PyTorch or embeddings fail to load.

## License

Add your chosen license before publishing or submitting the project.
