"""Summary AI Bot — FastAPI backend with SQLite Persistence and History Support.

Endpoints:
  POST   /upload               -> upload a PDF, background-index it, persist to SQLite & disk
  GET    /status/{id}          -> processing status + document stats
  POST   /summary              -> AI summary (short / medium / detailed / findings / faq)
  POST   /chat                 -> grounded Q&A with page citations & persistent history
  GET    /keywords/{id}        -> offline TF-IDF keywords
  GET    /suggest/{id}         -> AI-suggested questions
  GET    /document/{id}        -> structured page-by-page text for reader
  GET    /pdf/{id}             -> raw PDF file for inline browser viewing
  GET    /search/{id}          -> semantic in-document chunk search
  POST   /cancel/{id}          -> interrupt client acknowledgement
  GET    /export/{id}          -> download summary (md, txt)
  GET    /health               -> config + model info

History Endpoints:
  GET    /history/documents    -> list all past documents with metadata & stats
  GET    /history/{id}         -> get document data & full chat conversation history
  DELETE /history/{id}         -> delete a document and its chat history
  DELETE /history/chat/{id}    -> clear chat messages for a document
  DELETE /history/clear        -> clear all documents and history
"""
import glob
import json
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
from database import (
    add_chat_message,
    clear_all_documents,
    clear_chat_history,
    delete_document,
    get_chat_history,
    get_document,
    list_documents,
    save_document,
    update_document_summary,
)
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
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

sessions: dict[str, dict] = {}
ai_engine = AIEngine()
_llm_lock = threading.Lock()  # free tier is rate-limited; serialise LLM calls


# --------------------------------------------------------------------------- #
# Background processing & Persistence
# --------------------------------------------------------------------------- #
def process_pdf_background(file_id: str, file_path: str, file_size: int = 0) -> None:
    """Extract text, build vector index, and persist document to disk & SQLite."""
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
        pages_data = [
            {"page": p, "text": t, "words": len(t.split())}
            for p, t in pages
        ]
        session["pages_data"] = pages_data

        store = VectorStore()
        chunk_count = store.build_index(pages)

        # Save VectorStore index & chunks to disk
        prefix = os.path.join(UPLOADS_DIR, file_id)
        store.save_to_disk(prefix)

        # Save raw page text metadata to disk
        doc_json_path = os.path.join(UPLOADS_DIR, f"{file_id}_doc.json")
        with open(doc_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": session.get("filename", "document.pdf"),
                "pages": len(pages),
                "words": word_count,
                "chunks": chunk_count,
                "pages_data": pages_data,
            }, f, ensure_ascii=False)

        # Persist document record into SQLite
        save_document(
            file_id=file_id,
            filename=session.get("filename", "document.pdf"),
            file_size=file_size,
            page_count=len(pages),
            word_count=word_count,
            chunk_count=chunk_count,
        )

        session["text"] = "\n\n".join(f"[Page {p}]\n{t}" for p, t in pages)
        session["vector_store"] = store
        session["chunks"] = chunk_count
        session["stage"] = "done"
        session["status"] = "ready"
    except Exception as exc:  # noqa: BLE001
        session.update(status="error", error=str(exc))


# --------------------------------------------------------------------------- #
# Helpers & Session Restoration
# --------------------------------------------------------------------------- #
def _restore_session_from_disk(file_id: str) -> Optional[dict]:
    """Attempt to restore a document session from disk & SQLite."""
    doc_meta = get_document(file_id)
    if not doc_meta:
        return None

    pdf_path = os.path.join(UPLOADS_DIR, f"{file_id}.pdf")
    doc_json_path = os.path.join(UPLOADS_DIR, f"{file_id}_doc.json")
    prefix = os.path.join(UPLOADS_DIR, file_id)

    if not os.path.exists(pdf_path) or not os.path.exists(doc_json_path):
        return None

    try:
        with open(doc_json_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        store = VectorStore()
        loaded = store.load_from_disk(prefix)
        if not loaded:
            return None

        pages_data = doc_data.get("pages_data", [])
        full_text = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages_data)

        session = {
            "status": "ready",
            "stage": "done",
            "filename": doc_meta.get("filename", doc_data.get("filename", "document.pdf")),
            "path": pdf_path,
            "pages": doc_meta.get("page_count", doc_data.get("pages", 1)),
            "words": doc_meta.get("word_count", doc_data.get("words", 0)),
            "chunks": doc_meta.get("chunk_count", doc_data.get("chunks", 0)),
            "pages_data": pages_data,
            "text": full_text,
            "vector_store": store,
            "summary_short": doc_meta.get("summary_short", ""),
            "summary_medium": doc_meta.get("summary_medium", ""),
            "summary_detailed": doc_meta.get("summary_detailed", ""),
            "summary_findings": doc_meta.get("summary_findings", ""),
            "summary_faq": doc_meta.get("summary_faq", ""),
            "last_summary": doc_meta.get("summary_short", ""),
        }
        sessions[file_id] = session
        return session
    except Exception as exc:
        print(f"[Backend] Failed to restore session {file_id}: {exc}")
        return None


def _get_ready_session(file_id: str) -> dict:
    session = sessions.get(file_id)
    if session is None:
        # Try to restore session from SQLite & Disk
        session = _restore_session_from_disk(file_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if session["status"] == "error":
        raise HTTPException(status_code=500, detail=session.get("error", "Processing failed."))
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail="Document is still processing. Try again shortly.")
    return session


def _keywords_from_text(text: str, top_n: int) -> list[dict]:
    """Offline TF-IDF keywords — no API needed."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=200,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        )
        matrix = vec.fit_transform([text])
        scores = zip(vec.get_feature_names_out(), matrix.toarray()[0])
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [{"term": term, "score": round(float(s), 4)} for term, s in ranked if s > 0]
    except Exception:
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
    length: str = "short"  # short | medium | detailed | findings | faq


class ChatRequest(BaseModel):
    file_id: str
    question: str


# --------------------------------------------------------------------------- #
# Document Processing Endpoints
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
    file_path = os.path.join(UPLOADS_DIR, f"{file_id}.pdf")
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
    background_tasks.add_task(process_pdf_background, file_id, file_path, len(contents))
    return JSONResponse({"file_id": file_id, "filename": file.filename})


@app.get("/status/{file_id}")
def check_status(file_id: str) -> JSONResponse:
    """Poll processing status; returns stats once ready."""
    session = sessions.get(file_id)
    if session is None:
        # Check if already in persistent database
        session = _restore_session_from_disk(file_id)
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
    """Generate (and cache) an AI summary of the document, saving to SQLite."""
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

    # Persist summary in SQLite document record & history message
    update_document_summary(req.file_id, length, summary)
    add_chat_message(
        file_id=req.file_id,
        sender="ai",
        content=summary,
        msg_type="summary",
        model=model,
    )

    return JSONResponse({"summary": summary, "model": model, "cached": False, "format": length})


@app.post("/chat")
def chat_with_doc(req: ChatRequest) -> JSONResponse:
    """Answer a question strictly from the document, saving to persistent chat history."""
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

    # Save to SQLite Chat History
    add_chat_message(
        file_id=req.file_id,
        sender="user",
        content=question,
        msg_type="chat",
    )
    add_chat_message(
        file_id=req.file_id,
        sender="ai",
        content=answer,
        msg_type="chat",
        sources=sources,
        model=model,
    )

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
        pages = [{"page": 1, "text": session["text"], "words": session.get("words", 0)}]
    return JSONResponse({
        "file_id": file_id,
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
        out_path = os.path.join(UPLOADS_DIR, f"{file_id}_summary.txt")
        clean_text = re.sub(r"[#*_`~]", "", summary)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Summary — {session['filename']}\n{'='*40}\n\n{clean_text}\n")
        return FileResponse(out_path, media_type="text/plain", filename=f"{stem}_summary.txt")
    else:
        out_path = os.path.join(UPLOADS_DIR, f"{file_id}_summary.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Summary — {session['filename']}\n\n{summary}\n")
        return FileResponse(out_path, media_type="text/markdown", filename=f"{stem}_summary.md")


# --------------------------------------------------------------------------- #
# History Management Endpoints
# --------------------------------------------------------------------------- #
@app.get("/history/documents")
def get_documents_history() -> JSONResponse:
    """List all previously analyzed documents with metadata and message counts."""
    docs = list_documents()
    return JSONResponse({"documents": docs})


@app.get("/history/{file_id}")
def get_document_history(file_id: str) -> JSONResponse:
    """Get document details along with full chat conversation history."""
    doc = get_document(file_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document history not found.")
    messages = get_chat_history(file_id)
    return JSONResponse({
        "document": doc,
        "messages": messages,
    })


@app.delete("/history/{file_id}")
def delete_document_history(file_id: str) -> JSONResponse:
    """Delete a document from SQLite and remove all associated files from disk."""
    deleted = delete_document(file_id)
    # Remove in-memory session
    sessions.pop(file_id, None)

    # Clean up files on disk
    patterns = [
        os.path.join(UPLOADS_DIR, f"{file_id}.pdf"),
        os.path.join(UPLOADS_DIR, f"{file_id}_doc.json"),
        os.path.join(UPLOADS_DIR, f"{file_id}.index"),
        os.path.join(UPLOADS_DIR, f"{file_id}_chunks.json"),
        os.path.join(UPLOADS_DIR, f"{file_id}_summary.*"),
    ]
    for pattern in patterns:
        for p in glob.glob(pattern):
            try:
                os.remove(p)
            except OSError:
                pass

    return JSONResponse({"ok": True, "file_id": file_id, "deleted": deleted})


@app.delete("/history/chat/{file_id}")
def clear_chat_history_endpoint(file_id: str) -> JSONResponse:
    """Clear chat messages for a specific document."""
    cleared = clear_chat_history(file_id)
    return JSONResponse({"ok": True, "file_id": file_id, "cleared": cleared})


@app.delete("/history/clear")
def clear_all_history_endpoint() -> JSONResponse:
    """Delete all documents and chat records from database and clean disk."""
    clear_all_documents()
    sessions.clear()
    
    # Remove files in uploads
    for p in glob.glob(os.path.join(UPLOADS_DIR, "*")):
        if not p.endswith(".gitkeep"):
            try:
                os.remove(p)
            except OSError:
                pass

    return JSONResponse({"ok": True, "message": "All history and documents cleared."})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "api_key_configured": bool(ai_engine.api_key),
        "model": ai_engine.model_name,
        "fallbacks": ai_engine.fallback_models,
    })


# --------------------------------------------------------------------------- #
# Static frontend (mounted last)
# --------------------------------------------------------------------------- #
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
