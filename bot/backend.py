"""Summary AI Bot — FastAPI backend.

Endpoints:
  POST /upload            -> upload a PDF, background-index it
  GET  /status/{id}       -> processing status + document stats
  POST /summary           -> AI summary (short / medium / detailed)
  POST /chat              -> grounded Q&A with page citations
  GET  /keywords/{id}     -> offline TF-IDF keywords
  GET  /suggest/{id}      -> AI-suggested questions
  GET  /export/{id}       -> download the last summary as Markdown
  GET  /health            -> config + model info
"""
import os
import re
import threading
import uuid
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_engine import AIEngine
from pdf_parser import extract_text
from vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = FastAPI(title="Summary AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024        # 50 MB
MAX_SUMMARY_CHARS = 120_000                # ~30k tokens of context

sessions: dict[str, dict] = {}
ai_engine = AIEngine()
_llm_lock = threading.Lock()  # free tier is rate-limited; serialise LLM calls


# --------------------------------------------------------------------------- #
# Background processing
# --------------------------------------------------------------------------- #
def process_pdf_background(file_id: str, file_path: str) -> None:
    """Extract text and build the vector index without blocking the API."""
    session = sessions.get(file_id)
    if session is None:
        return
    try:
        session["stage"] = "extracting"
        pages = extract_text(file_path)
        if not pages:
            session.update(status="error", error="No extractable text found in this PDF "
                                                 "(it may be a scanned image — OCR is not enabled).")
            return

        session["stage"] = "indexing"
        session["pages"] = len(pages)
        word_count = sum(len(text.split()) for _, text in pages)
        session["words"] = word_count
        session["pages_data"] = [
            {"page": p, "text": t, "words": len(t.split())}
            for p, t in pages
        ]

        store = VectorStore()
        chunk_count = store.build_index(pages)

        session["text"] = "\n\n".join(f"[Page {p}]\n{t}" for p, t in pages)
        session["vector_store"] = store
        session["chunks"] = chunk_count
        session["stage"] = "done"
        session["status"] = "ready"
    except Exception as exc:  # noqa: BLE001 - report anything to the client
        session.update(status="error", error=str(exc))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_ready_session(file_id: str) -> dict:
    session = sessions.get(file_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Document not found (server may have restarted).")
    if session["status"] == "error":
        raise HTTPException(status_code=500, detail=session.get("error", "Processing failed."))
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail="Document is still processing. Try again shortly.")
    return session


def _keywords_from_text(text: str, top_n: int) -> list[dict]:
    """Offline TF-IDF keywords — no API needed."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=200,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        )
        matrix = vec.fit_transform([text])
        scores = zip(vec.get_feature_names_out(), matrix.toarray()[0])
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [{"term": term, "score": round(float(s), 4)} for term, s in ranked if s > 0]
    except Exception:
        # Deterministic fallback: most common meaningful words
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
        stops = {"the", "and", "for", "with", "that", "this", "from", "are", "was",
                 "were", "have", "has", "not", "but", "all", "can", "will", "one",
                 "two", "into", "such", "their", "than", "which", "when", "what"}
        freq: dict[str, int] = {}
        for w in words:
            if w not in stops:
                freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"term": t, "score": round(c / max(1, len(words)), 4)} for t, c in ranked]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SummaryRequest(BaseModel):
    file_id: str
    length: str = "short"  # short | medium | detailed


class ChatRequest(BaseModel):
    file_id: str
    question: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks,
                      file: UploadFile = File(...)) -> JSONResponse:
    """Upload a PDF and start background indexing."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    file_id = str(uuid.uuid4())
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", f"{file_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(contents)

    sessions[file_id] = {
        "status": "processing",
        "stage": "uploading",
        "filename": file.filename,
        "path": file_path,
        "text": "",
        "last_summary": "",
    }
    background_tasks.add_task(process_pdf_background, file_id, file_path)
    return JSONResponse({"file_id": file_id, "filename": file.filename})


@app.get("/status/{file_id}")
def check_status(file_id: str) -> JSONResponse:
    """Poll processing status; returns stats once ready."""
    session = sessions.get(file_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    payload = {
        "status": session["status"],
        "stage": session.get("stage", ""),
        "filename": session.get("filename", ""),
    }
    if session["status"] == "error":
        payload["error"] = session.get("error", "Unknown error")
    if session["status"] == "ready":
        payload.update({
            "pages": session.get("pages", 0),
            "words": session.get("words", 0),
            "chunks": session.get("chunks", 0),
        })
    return JSONResponse(payload)


@app.post("/summary")
def generate_summary(req: SummaryRequest) -> JSONResponse:
    """Generate (and cache) an AI summary of the document."""
    session = _get_ready_session(req.file_id)
    allowed_lengths = ("short", "medium", "detailed", "findings", "faq")
    length = req.length if req.length in allowed_lengths else "short"

    cache_key = f"summary_{length}"
    if session.get(cache_key):
        return JSONResponse({
            "summary": session[cache_key],
            "model": session.get(f"{cache_key}_model", ""),
            "cached": True,
            "format": length,
        })

    text = session["text"][:MAX_SUMMARY_CHARS]
    try:
        with _llm_lock:
            summary, model = ai_engine.generate_summary(text, length)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the AI service: {exc}")

    session[cache_key] = summary
    session[f"{cache_key}_model"] = model
    session["last_summary"] = summary
    return JSONResponse({"summary": summary, "model": model, "cached": False, "format": length})


@app.post("/chat")
def chat_with_doc(req: ChatRequest) -> JSONResponse:
    """Answer a question strictly from the document, with page citations."""
    session = _get_ready_session(req.file_id)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    store: VectorStore = session["vector_store"]
    retrieved = store.search(question, top_k=4)
    if not retrieved:
        raise HTTPException(status_code=500, detail="The document index is empty.")

    context = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in retrieved
    )
    try:
        with _llm_lock:
            answer, model = ai_engine.answer_question(context, question)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the AI service: {exc}")

    sources = [
        {"page": c["page"], "score": round(c["score"], 3),
         "snippet": c["text"][:280] + ("…" if len(c["text"]) > 280 else "")}
        for c in retrieved
    ]
    return JSONResponse({
        "answer": answer,
        "model": model,
        "sources": sources,
        "chunks_used": len(retrieved),
    })


@app.get("/keywords/{file_id}")
def get_keywords(file_id: str, top: int = 15) -> JSONResponse:
    """Top keywords for the document (offline TF-IDF)."""
    session = _get_ready_session(file_id)
    top = max(5, min(40, top))
    return JSONResponse({"keywords": _keywords_from_text(session["text"], top)})


@app.get("/suggest/{file_id}")
def suggest_questions(file_id: str) -> JSONResponse:
    """AI-suggested questions based on the document's opening content."""
    session = _get_ready_session(file_id)
    sample = session["text"][:6000]
    try:
        with _llm_lock:
            raw, model = ai_engine.suggest_questions(sample)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    
    # Clean thinking tags
    raw_cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    lines = [
        re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip(" -*\"'")
        for line in raw_cleaned.splitlines()
        if line.strip()
    ]
    q_words = ("what", "how", "why", "when", "where", "which", "who", "can", "is", "are", "explain", "does", "do")
    valid_questions = [
        line for line in lines
        if (line.endswith("?") or line.lower().startswith(q_words))
        and not any(ignore in line.lower() for ignore in ("thinking process", "analyze user", "generate 3", "here's", "output format", "comprehension questions"))
    ]
    return JSONResponse({"questions": valid_questions[:4] if valid_questions else lines[:4]})


@app.get("/document/{file_id}")
def get_document_data(file_id: str) -> JSONResponse:
    """Returns structured page-by-page text for reader view."""
    session = _get_ready_session(file_id)
    pages = session.get("pages_data", [])
    if not pages and session.get("text"):
        # Fallback if pages_data was not created yet
        pages = [{"page": 1, "text": session["text"], "words": session.get("words", 0)}]
    return JSONResponse({
        "filename": session.get("filename", ""),
        "total_pages": session.get("pages", len(pages)),
        "total_words": session.get("words", 0),
        "total_chunks": session.get("chunks", 0),
        "pages": pages,
    })


@app.get("/pdf/{file_id}")
def get_pdf_file(file_id: str):
    """Serve the raw PDF file for inline browser viewing."""
    session = _get_ready_session(file_id)
    pdf_path = session.get("path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")
    filename = session.get("filename", "document.pdf")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )


@app.get("/search/{file_id}")
def search_in_document(file_id: str, q: str = "") -> JSONResponse:
    """Semantic chunk search across the document."""
    session = _get_ready_session(file_id)
    q = q.strip()
    if not q:
        return JSONResponse({"results": []})
    store: VectorStore = session.get("vector_store")
    if not store:
        return JSONResponse({"results": []})
    results = store.search(q, top_k=6)
    return JSONResponse({
        "query": q,
        "results": [
            {
                "page": r["page"],
                "score": round(r["score"], 3),
                "text": r["text"],
            }
            for r in results
        ]
    })


@app.post("/cancel/{file_id}")
def cancel_request(file_id: str) -> JSONResponse:
    """Client interrupt acknowledgement."""
    return JSONResponse({"ok": True, "file_id": file_id, "message": "Interrupted."})


@app.get("/export/{file_id}")
def export_summary(file_id: str, format: str = "md") -> FileResponse:
    """Download the most recent summary in desired format (md, txt)."""
    session = _get_ready_session(file_id)
    summary = session.get("last_summary", "")
    if not summary:
        raise HTTPException(status_code=404, detail="Generate a summary first, then export it.")
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", session["filename"].rsplit(".", 1)[0]) or "document"
    
    if format == "txt":
        out_path = os.path.join("uploads", f"{file_id}_summary.txt")
        # Strip markdown syntax for plain text
        clean_text = re.sub(r"[#*_`~]", "", summary)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Summary — {session['filename']}\n{'='*40}\n\n{clean_text}\n")
        return FileResponse(out_path, media_type="text/plain", filename=f"{stem}_summary.txt")
    else:
        out_path = os.path.join("uploads", f"{file_id}_summary.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Summary — {session['filename']}\n\n{summary}\n")
        return FileResponse(out_path, media_type="text/markdown", filename=f"{stem}_summary.md")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "api_key_configured": bool(ai_engine.api_key),
        "model": ai_engine.model_name,
        "fallbacks": ai_engine.fallback_models,
    })


# --------------------------------------------------------------------------- #
# Static frontend (must be mounted last so /api-like routes take precedence)
# --------------------------------------------------------------------------- #
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
