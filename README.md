# Summary AI — Intelligent Document Workspace 📄⚡

> An intelligent, grounded document intelligence workspace. Upload any PDF to get multi-format AI summaries, grounded Q&A with exact page citations, offline topic graphs, interactive page reader, and live generation interruption controls.

![UI Theme](https://img.shields.io/badge/theme-royal--purple%20glassmorphism-8b5cf6)
![Backend](https://img.shields.io/badge/backend-FastAPI%20%2B%20FAISS%20%2B%20MiniLM-6366f1)
![LLM](https://img.shields.io/badge/LLM-OpenRouter%20%2B%20AutoFallback-ec4899)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Key Features

- **⚡ 5 AI Summary Modes**: Quick Bullets, Executive Overview, Structured Deep Dive, Key Insights & Data, and Essential FAQ.
- **💬 Grounded Q&A Chat with Citations**: Answers strictly derived from your PDF text chunks using FAISS cosine retrieval, citing exact page numbers `(p. N)`.
- **⏹ Generation Interrupt & Abort Control**: Instant client-side `AbortController` stop button and <kbd>Esc</kbd> support to halt long generations immediately without freezing.
- **📖 Side-by-Side Document Reader**: In-browser page viewer with search match highlighting and page-by-page inspection.
- **🏷️ Offline Topic & Keyword Explorer**: TF-IDF statistical keyword scoring with interactive click-to-inquire topic cards.
- **🎨 Responsive Glassmorphism Design**: Royal purple signature aesthetic with Light/Dark mode toggles, ambient glow orbs, and mobile drawer navigation.
- **📥 Exporter Suite**: Export summaries to Markdown (`.md`), Plain Text (`.txt`), Copy to Clipboard, or Print / Save as PDF.
- **🔄 Automatic Rate-Limit Failover**: Auto-switches across free OpenRouter fallback models if rate limits occur.

---

## 🛠️ Architecture

```
PDF Upload ──> PyMuPDF text extraction ──> Page-aware 180-word overlapping chunks
           ──> all-MiniLM-L6-v2 embeddings ──> FAISS IndexFlatIP cosine index
           ──> Grounded Prompting ──> OpenRouter Free LLMs (Nemotron / Gemma / Qwen)
           ──> Interactive Citations (p. N) + Passage Inspector
```

---

## 🚀 Quick Start (Local Run)

### 1. Clone the repository
```bash
git clone https://github.com/yogeshkandoriya/Summary_AI_system.git
cd Summary_AI_system/bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the `bot` folder:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_FALLBACK_MODELS=nvidia/nemotron-3.5-lightning:free,google/gemma-4-31b-it:free,qwen/qwen3.8-27b:free
```
*(Get a free API key from [OpenRouter.ai](https://openrouter.ai))*

### 4. Run the server
```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🌐 Deployment Guide

### Option 1: Render.com (Recommended Free Hosting)
1. Fork or push this repository to GitHub.
2. Go to [Render.com](https://render.com) and click **New + Web Service**.
3. Connect your GitHub repository.
4. Set:
   - **Root Directory**: `bot`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `OPENROUTER_API_KEY`: Your key

### Option 2: Railway.app
1. Create a new project on [Railway.app](https://railway.app) from your GitHub repo.
2. Set Root Directory to `/bot` or deploy with Docker.
3. Add `OPENROUTER_API_KEY` in Railway dashboard variables.

### Option 3: Hugging Face Spaces (Docker / Python)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with SDK: **Docker** or **Gradio/Streamlit**.
2. Push the `bot` directory with the provided Dockerfile / requirements.

### Option 4: VPS (DigitalOcean / AWS EC2 / Ubuntu)
```bash
sudo apt update && sudo apt install python3-pip python3-venv git -y
git clone https://github.com/yogeshkandoriya/Summary_AI_system.git
cd Summary_AI_system/bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

---

## 📁 Repository Structure

```
Summary_AI_system/
├── bot/                       # Complete FastAPI + Vanilla JS Web Application
│   ├── backend.py             # FastAPI REST endpoints & background indexing
│   ├── ai_engine.py           # OpenRouter LLM grounding & auto-fallback engine
│   ├── vector_store.py        # MiniLM chunking & FAISS index
│   ├── pdf_parser.py          # PyMuPDF page text extraction
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment template
│   └── frontend/              # Modern Responsive UI
│       ├── index.html         # Workspace HTML5 markup & modals
│       ├── style.css          # Purple glassmorphic responsive styles
│       └── app.js             # Controller, citation parser & abort handlers
├── SummaryAI/                 # Modular Python library package & Streamlit UI
└── README.md                  # Project documentation & deployment guide
```
