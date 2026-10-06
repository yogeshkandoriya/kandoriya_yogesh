# Summary AI Bot 📄⚡

Upload a PDF and get an instant AI summary, key topics, and a chat that answers **only from your document** — with page citations.

![stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20FAISS%20%2B%20OpenRouter-8b5cf6)

## Quick start

```bash
cd bot
pip install -r requirements.txt
# 1. Create a .env file:
#    OPENROUTER_API_KEY=sk-or-v1-...     (free key from https://openrouter.ai)
#    OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
# 2. Run:
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
# 3. Open http://127.0.0.1:8000
```

## Features

| Feature | How |
|---|---|
| **Instant summaries** | Three lengths — quick bullets, overview, or a detailed structured report |
| **Chat with your PDF** | FAISS vector search + grounded LLM answers that cite pages |
| **Key topics** | Offline TF-IDF keyword cloud; click a keyword to ask about it |
| **Suggested questions** | AI-generated starter questions for every document |
| **Export** | Download the last summary as Markdown |
| **Document stats** | Pages, words and indexed chunks shown in the sidebar |
| **Model fallback** | If a free model is rate-limited, it automatically retries the next one |

## Architecture

```
PDF ──> PyMuPDF extraction ──> page-aware overlapping chunks
    ──> MiniLM embeddings ──> FAISS (cosine) index
    ──> hybrid retrieval on question ──> OpenRouter LLM
    ──> grounded answer + (p. N) citations
```

- **Backend** — FastAPI (`backend.py`), one endpoint per feature, background indexing, in-memory session cache, per-length summary caching.
- **AI engine** — `ai_engine.py`: OpenRouter chat completions with a primary model plus automatic fallbacks, rate-limit retry, and grounded prompts.
- **Retrieval** — `vector_store.py`: 180-word overlapping chunks, normalized embeddings, `IndexFlatIP` cosine search returning page numbers and scores.
- **Frontend** — vanilla JS + marked.js, dark glassmorphism theme, live upload progress, clickable source chips, no build step.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/upload` | Upload a PDF (multipart), starts background indexing |
| GET | `/status/{id}` | Processing status + page/word/chunk stats |
| POST | `/summary` | `{file_id, length: short\|medium\|detailed}` |
| POST | `/chat` | `{file_id, question}` → answer + source pages |
| GET | `/keywords/{id}` | Top TF-IDF keywords |
| GET | `/suggest/{id}` | AI-suggested questions |
| GET | `/export/{id}` | Last summary as `.md` |
| GET | `/health` | Config check |

## Notes

- Free OpenRouter tier: ~20 requests/min, 50/day. The engine falls back across models to absorb rate limits.
- Everything runs locally — documents are processed on your machine and only text excerpts are sent to the LLM.
- `.env` holds your API key and is git-ignored. Never commit it.
