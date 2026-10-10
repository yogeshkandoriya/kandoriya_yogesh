"""Summary AI — SQLite Database Module for Document & Chat History Persistence.

Provides persistent storage for:
- Uploaded PDF metadata and generated summaries.
- Complete Q&A chat message history with citations and model metadata.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "history.db")


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite database connection with row factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        
        # Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                page_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                summary_short TEXT DEFAULT '',
                summary_medium TEXT DEFAULT '',
                summary_detailed TEXT DEFAULT '',
                summary_findings TEXT DEFAULT '',
                summary_faq TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Chat History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                msg_type TEXT DEFAULT 'chat',
                content TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                model TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES documents (file_id) ON DELETE CASCADE
            );
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_file_id ON chat_history (file_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_created ON documents (created_at DESC);")
        conn.commit()


# --------------------------------------------------------------------------- #
# Document Operations
# --------------------------------------------------------------------------- #
def save_document(
    file_id: str,
    filename: str,
    file_size: int = 0,
    page_count: int = 0,
    word_count: int = 0,
    chunk_count: int = 0,
    summary_short: str = "",
) -> None:
    """Insert or update a document record."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (
                file_id, filename, file_size, page_count, word_count, chunk_count, summary_short, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_id) DO UPDATE SET
                filename = excluded.filename,
                file_size = excluded.file_size,
                page_count = excluded.page_count,
                word_count = excluded.word_count,
                chunk_count = excluded.chunk_count,
                summary_short = CASE WHEN excluded.summary_short != '' THEN excluded.summary_short ELSE documents.summary_short END,
                updated_at = CURRENT_TIMESTAMP;
        """, (file_id, filename, file_size, page_count, word_count, chunk_count, summary_short))
        conn.commit()


def get_document(file_id: str) -> Optional[dict[str, Any]]:
    """Retrieve document metadata by file_id."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE file_id = ?", (file_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_documents() -> list[dict[str, Any]]:
    """List all documents ordered by most recent, including chat message counts."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                d.file_id,
                d.filename,
                d.file_size,
                d.page_count,
                d.word_count,
                d.chunk_count,
                d.summary_short,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS message_count
            FROM documents d
            LEFT JOIN chat_history c ON d.file_id = c.file_id
            GROUP BY d.file_id
            ORDER BY d.updated_at DESC;
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_document_summary(file_id: str, format_key: str, summary_text: str) -> None:
    """Save generated summary for a given format."""
    allowed_keys = {
        "short": "summary_short",
        "medium": "summary_medium",
        "detailed": "summary_detailed",
        "findings": "summary_findings",
        "faq": "summary_faq",
    }
    col = allowed_keys.get(format_key, "summary_short")
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE documents 
            SET {col} = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE file_id = ?;
        """, (summary_text, file_id))
        conn.commit()


def delete_document(file_id: str) -> bool:
    """Delete a document and cascade delete its chat history."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE file_id = ?", (file_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_documents() -> None:
    """Delete all documents and chat history from database."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history;")
        cursor.execute("DELETE FROM documents;")
        conn.commit()


# --------------------------------------------------------------------------- #
# Chat History Operations
# --------------------------------------------------------------------------- #
def add_chat_message(
    file_id: str,
    sender: str,
    content: str,
    msg_type: str = "chat",
    sources: Optional[list[dict]] = None,
    model: str = "",
) -> int:
    """Record a chat message in history."""
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (file_id, sender, msg_type, content, sources_json, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, (file_id, sender, msg_type, content, sources_json, model))
        # Update parent document's updated_at timestamp
        cursor.execute("UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE file_id = ?", (file_id,))
        conn.commit()
        return cursor.lastrowid


def get_chat_history(file_id: str) -> list[dict[str, Any]]:
    """Retrieve full chat history for a document ordered chronologically."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, file_id, sender, msg_type, content, sources_json, model, created_at
            FROM chat_history
            WHERE file_id = ?
            ORDER BY id ASC;
        """, (file_id,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["sources"] = json.loads(item.get("sources_json") or "[]")
            except Exception:
                item["sources"] = []
            result.append(item)
        return result


def clear_chat_history(file_id: str) -> bool:
    """Clear chat history messages for a specific document."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE file_id = ?", (file_id,))
        conn.commit()
        return cursor.rowcount > 0


# Automatically ensure tables exist on module load
init_db()
