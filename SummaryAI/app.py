"""Summary AI - Streamlit user interface.

This module contains presentation logic only. All document processing lives in
the ``summaryai`` package, so the UI stays small and the core stays testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the src/ layout importable without installing the package.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from summaryai import __version__  # noqa: E402
from summaryai.config import FREE_MODELS, get_settings  # noqa: E402
from summaryai.analysis import find_term_pages  # noqa: E402
from summaryai.export import safe_filename, to_docx, to_markdown, to_pdf, to_plain_text, to_pptx  # noqa: E402
from summaryai.ingest import is_ocr_available  # noqa: E402
from summaryai.llm import OpenRouterEngine  # noqa: E402
from summaryai.pipeline import DocumentAssistant  # noqa: E402

st.set_page_config(
    page_title="Summary AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUMMARY_FORMATS = ["Bullet Points", "Key Findings", "Paragraph", "Structured", "FAQ"]
SUMMARY_LENGTHS = ["Short", "Medium", "Detailed"]

CSS = """
<style>
.app-header { text-align: center; padding: 1rem 0 1.5rem; }
.app-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-subtitle { color: #64748b; font-size: 1.05rem; margin-top: .25rem; }
.doc-card {
    border: 1px solid #e2e8f0; border-left: 4px solid #2563eb;
    border-radius: 8px; padding: .85rem 1rem; margin-bottom: .6rem;
    background: #f8fafc;
}
.doc-card h4 { margin: 0 0 .3rem 0; font-size: 1rem; }
.evid { border-left: 3px solid #cbd5e1; padding: .4rem .8rem; margin: .4rem 0; color: #475569; font-size: .88rem; }
.refusal { background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:.8rem 1rem; color:#991b1b; }
</style>
"""

SLASH_COMMANDS = {
    "/summary": "Summarise the whole document",
    "/keypoints": "List the key findings",
    "/explain <topic>": "Explain a topic from the document",
    "/find <term>": "Locate a term and show its pages",
    "/define <term>": "Define a term using the document",
    "/keywords [n]": "Extract the top n keywords",
    "/faq": "Generate frequently asked questions",
    "/compare <doc>": "Compare with another uploaded document",
    "/help": "Show this list of commands",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_assistant(model: str, api_key: str) -> DocumentAssistant:
    """Build (and cache) the assistant. Cached per model+key combination."""
    return DocumentAssistant(settings=get_settings(model=model, api_key=api_key or None))


def init_state() -> None:
    """Seed session state on first run."""
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("active_doc", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("debug_mode", False)


def _qa_payload(result) -> dict:
    """Convert an AnswerResult into the dict shape the UI renders."""
    return {
        "answer": result.answer,
        "refused": result.refused,
        "reason": result.reason,
        "footer": result.citation_footer() if result.citations else "",
        "evidence": result.evidence,
        "error": result.error,
    }


def run_query(assistant: DocumentAssistant, doc: str, text: str) -> dict:
    """Route a user message to the right operation and normalise the result."""
    text = text.strip()

    if text.startswith("/"):
        parts = text.split(" ", 1)
        cmd, args = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

        if cmd == "/help":
            lines = "\n".join(f"- `{c}` — {d}" for c, d in SLASH_COMMANDS.items())
            return {"answer": f"**Available commands**\n\n{lines}"}

        if cmd in ("/summary", "/keypoints"):
            length = {"short": "Short", "medium": "Medium", "detailed": "Detailed"}.get(
                args.lower(), "Medium"
            )
            fmt = "Key Findings" if cmd == "/keypoints" else "Bullet Points"
            outcome = assistant.summarize(doc, length=length, fmt=fmt)
            if not outcome.ok:
                return {"error": outcome.error}
            return {
                "answer": outcome.text,
                "footer": f"{doc} · {outcome.actual_words} words · {outcome.model}",
                "citations": [],
            }

        if cmd == "/explain":
            if not args:
                return {"error": "Usage: `/explain <topic>`"}
            return _qa_payload(assistant.explain(doc, args))

        if cmd in ("/find", "/define"):
            if not args:
                return {"error": f"Usage: `/{cmd.lstrip('/')} <term>`"}
            hits = find_term_pages(assistant.documents[doc], args, top_n=5)
            if not hits:
                return {"answer": f"The term **{args}** does not appear in this document."}
            body = "\n\n".join(f"**Page {h['page']}**\n\n{h['excerpt']}" for h in hits)
            return {"answer": f"Occurrences of **{args}**:\n\n{body}"}

        if cmd == "/keywords":
            try:
                count = max(1, min(30, int(args))) if args else 12
            except ValueError:
                count = 12
            terms = assistant.keywords(doc, count)
            return {"answer": "**Keywords**\n\n" + "\n".join(f"- {k.term}" for k in terms)}

        if cmd == "/faq":
            from summaryai.generation import build_faq_prompt
            from summaryai.generation.prompts import GROUNDED_SYSTEM

            chunks, _ = assistant.search(doc, "key topics main concepts", top_k=6)
            context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c in chunks)
            response = assistant.engine.generate(
                build_faq_prompt(context, 6), system=GROUNDED_SYSTEM
            )
            return {"answer": response.text} if response.ok else {"error": response.error}

        if cmd == "/compare":
            others = [n for n in assistant.names if n != doc]
            other = args if args in others else (others[0] if others else None)
            if not other:
                return {"error": "Upload a second PDF to compare with."}
            with st.spinner("Comparing documents..."):
                return {"answer": assistant.compare(doc, other)}

        return {"error": f"Unknown command `{cmd}`. Try `/help`."}

    return _qa_payload(assistant.ask(doc, text))


def render_answer(result: dict) -> None:
    """Render one result payload inside a chat bubble."""
    if result.get("error"):
        st.error(result["error"])
        return

    if result.get("refused"):
        st.markdown(
            f'<div class="refusal">{result["answer"]}</div>', unsafe_allow_html=True
        )
        if result.get("reason"):
            st.caption(result["reason"])
    else:
        st.markdown(result["answer"])

    if result.get("footer"):
        st.caption(result["footer"].replace("**", ""))

    if st.session_state.debug_mode and result.get("evidence"):
        with st.expander("Developer: retrieval trace"):
            for chunk in result["evidence"]:
                st.markdown(
                    f'<div class="evid"><b>Page {chunk["page"]}</b> &nbsp; '
                    f'sem={chunk["semantic_score"]} &nbsp; lex={chunk["lexical_score"]} &nbsp; '
                    f'score={chunk["score"]}<br>{chunk["text"][:300]}…</div>',
                    unsafe_allow_html=True,
                )


def render_chat(assistant: DocumentAssistant) -> None:
    """Render conversation history and the input box."""
    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input("Ask a question, or type /help for commands...")
    if not question:
        return

    doc = st.session_state.active_doc
    if not doc:
        st.warning("Upload a PDF first.")
        return

    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            result = run_query(assistant, doc, question)
        render_answer(result)
        st.session_state.last_result = result

    st.session_state.history.append(("assistant", result.get("answer", "")))


def render_exports(title: str, content: str, doc_name: str) -> None:
    """Render the download buttons for a generated result."""
    stem = safe_filename(f"{doc_name}_{title}")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.download_button(
            "⬇️ TXT", to_plain_text(title, content), f"{stem}.txt", "text/plain",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Markdown", to_markdown(title, content), f"{stem}.md", "text/markdown",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "⬇️ Word", to_docx(title, content), f"{stem}.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with c4:
        st.download_button(
            "⬇️ PDF", to_pdf(title, content), f"{stem}.pdf", "application/pdf",
            use_container_width=True,
        )
    st.download_button(
        "⬇️ PowerPoint deck", to_pptx(title, content), f"{stem}.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Entry point: build the sidebar, then render the selected tab."""
    st.markdown(
        '<div class="app-header"><div class="app-title">SUMMARY AI</div>'
        '<div class="app-subtitle">AI-Powered PDF Summarization &amp; Document Assistant</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.sidebar.title("⚙️ Settings")
    model = st.sidebar.selectbox(
        "AI model (free)",
        list(FREE_MODELS),
        index=0,
        format_func=lambda m: FREE_MODELS[m]["label"],
        help="All options are zero-cost OpenRouter models. The app falls back to "
             "another automatically if one is rate-limited.",
    )
    st.sidebar.caption(FREE_MODELS[model]["note"])

    api_key = st.sidebar.text_input(
        "OpenRouter API key", value="", type="password", placeholder="sk-or-v1-...",
        help="Optional if OPENROUTER_API_KEY is set in .env",
    )
    if get_settings(api_key=api_key or None).has_key():
        st.sidebar.success("✅ API key detected")
    else:
        st.sidebar.error("No API key found. Add one to `.env` or paste it above.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Upload")
    uploaded = st.sidebar.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    enable_ocr = st.sidebar.checkbox(
        "Enable OCR for scanned pages", value=is_ocr_available(),
        disabled=not is_ocr_available(),
        help="Requires Tesseract; disabled automatically when unavailable.",
    )
    st.session_state.debug_mode = st.sidebar.checkbox(
        "🛠️ Developer mode", value=st.session_state.debug_mode,
        help="Show retrieval scores and evidence chunks.",
    )

    assistant = get_assistant(model, api_key)

    for upload in uploaded or []:
        if upload.name in assistant.names:
            continue
        with st.spinner(f"Indexing {upload.name}..."):
            try:
                assistant.add_document(
                    upload.getvalue(), upload.name,
                    enable_ocr=enable_ocr, file_size=upload.size,
                )
            except Exception as exc:
                st.sidebar.error(f"{upload.name}: {exc}")

    render_sidebar_library(assistant)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Summary AI v{__version__} · OpenRouter")

    tab_summary, tab_ask, tab_history, tab_about = st.tabs(
        ["📝 Summarizer", "💬 Ask PDF", "🕘 History", "ℹ️ About"]
    )
    with tab_summary:
        render_summarizer(assistant)
    with tab_ask:
        render_ask(assistant)
    with tab_history:
        render_history()
    with tab_about:
        render_about(model, enable_ocr)


def render_sidebar_library(assistant: DocumentAssistant) -> None:
    """Document library: selection, stats, and removal."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Documents")

    if not assistant.names:
        st.session_state.active_doc = None
        return

    current = st.session_state.active_doc
    index = assistant.names.index(current) if current in assistant.names else 0
    st.session_state.active_doc = st.sidebar.selectbox("Active document", assistant.names, index=index)

    stats = assistant.index_stats(st.session_state.active_doc)
    doc = assistant.documents[st.session_state.active_doc]
    s1, s2, s3 = st.columns(3)
    s1.metric("Pages", doc.page_count)
    s2.metric("Words", f"{doc.word_count:,}")
    s3.metric("Chunks", stats.get("total_chunks", 0))

    if st.sidebar.button("🗑️ Remove active document"):
        assistant.remove_document(st.session_state.active_doc)
        st.session_state.active_doc = assistant.names[0] if assistant.names else None
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_summarizer(assistant: DocumentAssistant) -> None:
    """Summarizer tab: options, generation and exports."""
    if not assistant.has_documents():
        st.info("Upload a PDF from the sidebar to get started.")
        return

    st.subheader("📝 Generate a summary")
    c1, c2, c3 = st.columns(3)
    length = c1.selectbox("Length", SUMMARY_LENGTHS, index=1)
    fmt = c2.selectbox("Format", SUMMARY_FORMATS, index=0)
    simple = c3.toggle("Simple language", help="Explain in plain, beginner-friendly words.")

    if st.button("✨ Generate summary", type="primary"):
        with st.spinner("Reading the document and writing the summary..."):
            st.session_state.last_result = summary_payload(assistant, length, fmt, simple)

    result = st.session_state.last_result or {}
    if result.get("answer"):
        st.markdown("---")
        st.markdown(result["answer"])
        st.caption(result.get("footer", ""))
        render_exports("summary", result["answer"], st.session_state.active_doc)
    elif result.get("error"):
        st.error(result["error"])

    if st.session_state.debug_mode:
        with st.expander("🛠️ Document audit"):
            st.json(assistant.index_stats(st.session_state.active_doc))
            st.json({k: v for k, v in assistant.documents[st.session_state.active_doc].meta.items() if v})


def summary_payload(assistant: DocumentAssistant, length: str, fmt: str, simple: bool) -> dict:
    """Run a summarisation and shape it for the UI."""
    outcome = assistant.summarize(
        st.session_state.active_doc, length=length, fmt=fmt, simple_language=simple
    )
    if not outcome.ok:
        return {"answer": "", "footer": "", "error": outcome.error}
    return {
        "answer": outcome.text,
        "footer": (
            f"{outcome.actual_words} words (target {outcome.target_words}) · "
            f"{outcome.model} · {outcome.coverage.get('Verdict', '')}"
        ),
        "error": "",
    }


def render_ask(assistant: DocumentAssistant) -> None:
    """Ask PDF tab: chat interface."""
    st.subheader("💬 Ask your document")
    st.caption("Answers are grounded in the PDF and cite page numbers.")
    st.caption(" · ".join(f"`{c}`" for c in SLASH_COMMANDS))
    if not assistant.has_documents():
        st.info("Upload a PDF from the sidebar first.")
        return
    render_chat(assistant)


def render_history() -> None:
    """History tab: past questions and answers."""
    st.subheader("🕘 Conversation history")
    history = st.session_state.history
    if not history:
        st.info("No questions asked yet.")
        return
    for role, message in history:
        with st.chat_message(role):
            st.markdown(message)
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


def render_about(model: str, enable_ocr: bool) -> None:
    """About tab: architecture and configuration."""
    st.subheader("ℹ️ About Summary AI")
    st.markdown(
        "A document intelligence assistant that answers **only** from the PDF you "
        "upload, and cites the page for every claim."
    )
    st.markdown("### Pipeline")
    st.code(
        "PDF -> PyMuPDF extraction / Tesseract OCR\n"
        "    -> text cleaning (headers, page numbers, de-hyphenation)\n"
        "    -> page-aware chunking\n"
        "    -> MiniLM embeddings -> FAISS (cosine)\n"
        "    -> hybrid retrieval (semantic + lexical)\n"
        "    -> OpenRouter LLM -> grounded answer + citations",
        language="text",
    )
    st.markdown("### Configuration")
    st.json(
        {
            "Model": FREE_MODELS[model]["label"],
            "Context window": f"{FREE_MODELS[model]['context']:,} tokens",
            "OCR available": bool(enable_ocr),
            "Version": __version__,
        }
    )
    st.markdown("### Commands")
    st.markdown("\n".join(f"- `{c}` — {d}" for c, d in SLASH_COMMANDS.items()))


if __name__ == "__main__":
    main()
