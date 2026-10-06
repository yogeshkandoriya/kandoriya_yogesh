# Summary AI

**AI-powered PDF summarisation and document assistant.** Upload a PDF, get a
length-targeted summary, and ask questions that are answered **only** from your
document — with page-number citations and a built-in refusal when the answer
isn't there.

Built on [OpenRouter](https://openrouter.ai) free models, so it runs at **$0 cost**.

---

## ✨ Features

| | |
|---|---|
| **Grounded Q&A** | Every answer cites the page it came from. The model is instructed — and the output is verified — to refuse rather than guess. |
| **Length-targeted summaries** | Short / Medium / Detailed, scaled to the document's own word count, with a live coverage report. |
| **Hybrid retrieval** | Dense MiniLM embeddings *and* lexical matching, weighted per query type, so exact numbers and dates are never missed. |
| **Five output formats** | Bullet points, key findings, paragraph, structured, FAQ. |
| **Map-reduce summarisation** | Long documents are summarised section-by-section and merged, so quality holds at any length. |
| **Multi-format export** | Markdown, `.docx`, `.pptx`, `.pdf`, plain text. |
| **Slash commands** | `/summary`, `/keypoints`, `/explain`, `/find`, `/keywords`, `/faq`, `/compare`, `/help`. |
| **OCR for scans** | Tesseract fallback for image-only pages; degrades gracefully when unavailable. |
| **Resilient** | Automatic retry with backoff, plus failover across free models when one is rate-limited. |
| **Offline test suite** | 126 tests in ~5 s. No network, no model downloads. |

---

## 🚀 Quick start

```bash
git clone https://github.com/yogeshkandoriya/Summary-AI.git
cd Summary-AI

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

Edit `.env` and add your key from [openrouter.ai/keys](https://openrouter.ai/keys):

```ini
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Then run:

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. You can also paste a key directly into
the sidebar, so `.env` is optional.

---

## 🧠 How it works

```text
PDF upload
   ↓
PyMuPDF extraction ──► Tesseract OCR (auto, for scanned pages)
   ↓
Text cleaning ── running headers, page numbers, de-hyphenation
   ↓
Page-aware chunking ── 900 chars / 120 overlap, keeps page numbers
   ↓
MiniLM embeddings (all-MiniLM-L6-v2) → FAISS IndexFlatIP (cosine)
   ↓
Hybrid retrieval ── semantic 60/40 · lexical 65/35 for numeric & date queries
   │                 threshold → dedup → adjacent-chunk expansion
   ↓
OpenRouter LLM (free models)
   ↓
Summary · Cited answer · Refusal guard · Keywords · Export
```

### Why hybrid retrieval?

Pure vector search is poor at questions containing exact values —
*"what wattage PSU was recommended in 2023?"*. MiniLM blurs precise tokens, so
the chunk containing `650W` may not surface.

`retrieval/hybrid.py` classifies the query first:

| Query type | Semantic | Lexical |
|---|---|---|
| General | 60% | 40% |
| Numerical, date, definition, formula | 35% | 65% |

…then boosts any chunk that literally contains a number from the question.

### Anti-hallucination

Three independent layers:

1. **Prompt** — the model sees only retrieved excerpts and is told to reply with a fixed refusal string when evidence is missing.
2. **Retrieval gate** — if no chunk clears the relevance threshold, the pipeline returns the refusal without calling the model at all.
3. **Output check** — `is_refusal()` scans the reply for hedging language and overrides it with the standard refusal.

```
Q: What was the salary of the CEO in 2024?
A: I could not find sufficient information in the uploaded PDF to answer this question.
```

---

## 🏗️ Project structure

```
SummaryAI/
├── app.py                      # Streamlit UI (presentation only)
├── requirements.txt
├── .env.example
├── src/summaryai/
│   ├── config.py               # Settings, .env loading, model catalogue
│   ├── schemas.py              # Document, Chunk, Citation, SearchResult
│   ├── pipeline.py             # DocumentAssistant — the orchestrator
│   ├── ingest/
│   │   ├── extractor.py        # PyMuPDF + metadata
│   │   ├── ocr.py              # Tesseract (optional)
│   │   └── cleaner.py          # Headers, page numbers, de-hyphenation
│   ├── retrieval/
│   │   ├── chunker.py          # Page-aware chunking
│   │   ├── embeddings.py       # MiniLM + offline hash embedder
│   │   ├── vector_store.py     # FAISS wrapper
│   │   └── hybrid.py           # Query analysis + fusion + re-ranking
│   ├── generation/
│   │   ├── prompts.py          # Prompts + refusal guard
│   │   ├── summarizer.py       # Map-reduce, length targeting
│   │   └── qa.py               # Cited answers
│   ├── analysis/keywords.py    # TF-IDF, term search, comparison
│   └── export/exporters.py     # MD / DOCX / PPTX / PDF / TXT
└── tests/                      # 126 offline tests
```

The original project was a single 3,741-line `app.py`. This splits every
concern into a focused module; the UI only ever talks to `DocumentAssistant`.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

126 tests, ~5 seconds, **completely offline**. The `LLMEngine` protocol means
tests inject a `MockEngine` instead of a real model, and `HashEmbedder`
replaces MiniLM — so no network calls and no model downloads.

Covered: extraction · cleaning · OCR merging · chunking · embeddings · FAISS
search · query classification · lexical scoring · hybrid ranking · refusal
guard · summarisation · map-reduce merge · citations · keyword extraction ·
exports · HTTP retry and model failover · the orchestrator.

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required.** Your OpenRouter key. |
| `OPENROUTER_MODEL` | `qwen/qwen3.8-27b:free` | Any `:free` model. |
| `OPENROUTER_APP_URL` | `http://localhost:8501` | Dashboard attribution. |
| `OPENROUTER_APP_TITLE` | `Summary AI` | Dashboard attribution. |

**Free models included:** Qwen3.8 27B, Nemotron 3 Super 120B, Ling 3.0 Flash,
Nemotron 3.5 Lightning (1 M context), Gemma 4 31B, Apodex 1.1 Mini. Free
tiers are rate-limited, so the app rotates between them automatically.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. **Create app → Yup, I have an app** → select the repo, branch `main`, file `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```

OCR is unavailable on Community Cloud (no Tesseract binary); everything else
works. To enable it, add a `packages.txt` containing:

```
tesseract-ocr
tesseract-ocr-eng
```

---

## 📄 License

MIT — free for academic, personal and research use.
