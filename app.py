"""
app.py — Streamlit Application
AI Meeting Minutes Summarizer
"""

import os
import sys
import io
import re
import pickle
import tempfile

import numpy as np
import streamlit as st

# ─── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🤖",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "seq2seq_model.h5")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "tokenizers.pkl")


@st.cache_resource(show_spinner="Loading AI model…")
def load_model_cached():
    """Load seq2seq model and tokenizers once, cache across sessions."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None

    from tensorflow.keras.models import load_model
    from src.model      import build_inference_models
    from src.preprocessing import load_tokenizers

    seq2seq = load_model(MODEL_PATH)
    enc_model, dec_model = build_inference_models(seq2seq)
    text_tok, sum_tok    = load_tokenizers(TOKENIZER_PATH)
    return enc_model, dec_model, text_tok, sum_tok


def do_summarize(text: str) -> str:
    """Run inference and return the generated summary string."""
    from src.summarizer import generate_summary

    enc_model, dec_model, text_tok, sum_tok = load_model_cached()

    if enc_model is None:
        # Model not trained yet — return a helpful demo message
        return (
            "⚠️ Model not trained yet. Run `python train.py` first to train the model, "
            "then refresh the app."
        )

    summary = generate_summary(text, enc_model, dec_model, text_tok, sum_tok)
    return summary if summary.strip() else "Unable to generate summary. Try a longer input."


def compute_stats(original: str, summary: str) -> dict:
    orig_words = len(original.split())
    summ_words = len(summary.split())
    ratio      = round((1 - summ_words / orig_words) * 100, 1) if orig_words > 0 else 0
    return {
        "Original Words":    orig_words,
        "Summary Words":     summ_words,
        "Compression Ratio": f"{ratio}%",
    }


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from DOCX bytes using python-docx."""
    try:
        from docx import Document
        doc  = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text.strip()
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def create_pdf_download(summary: str) -> bytes:
    """Create a downloadable PDF from the summary text."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AI Generated Meeting Summary", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_draw_color(100, 100, 100)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        # Multi-line text (handles wrapping)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, summary)
        return bytes(pdf.output())
    except Exception as e:
        return f"[PDF creation error: {e}]".encode()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#4F8BF9;'>🤖 AI Meeting Minutes Summarizer</h1>
    <p style='text-align:center; color:gray;'>Seq2Seq Encoder-Decoder · LSTM · Streamlit</p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

# ── Model status banner ────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.warning(
        "⚠️ **Model not found.** Run `python train.py` to train the model first.\n\n"
        "The app will still work — it will show a demo message instead of a real summary.",
        icon="⚠️",
    )
else:
    st.success("✅ Trained model loaded and ready.", icon="✅")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Enter Meeting Notes
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## 📝 Section 1 · Enter Meeting Notes")

meeting_text = st.text_area(
    label="Paste your meeting notes here:",
    placeholder=(
        "e.g.\n"
        "Yesterday the development team completed the authentication APIs. "
        "The frontend team integrated the login functionality. "
        "Testing will begin next Monday..."
    ),
    height=220,
    key="manual_input",
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Generate Summary Button
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## ▶️ Section 2 · Generate Summary")

generate_clicked = st.button("🚀 Generate Summary", use_container_width=True, type="primary")

# State management
if "summary"      not in st.session_state: st.session_state["summary"]      = ""
if "source_text"  not in st.session_state: st.session_state["source_text"]  = ""

if generate_clicked and meeting_text.strip():
    with st.spinner("Generating summary…"):
        st.session_state["summary"]     = do_summarize(meeting_text)
        st.session_state["source_text"] = meeting_text

elif generate_clicked and not meeting_text.strip():
    st.warning("Please paste some meeting notes in Section 1 first.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI Generated Summary
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Section 3 · AI Generated Summary")

if st.session_state["summary"]:
    st.success(st.session_state["summary"])
else:
    st.info("Your AI-generated summary will appear here after clicking **Generate Summary**.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Statistics
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Section 4 · Statistics")

if st.session_state["summary"] and st.session_state["source_text"]:
    stats = compute_stats(st.session_state["source_text"], st.session_state["summary"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Words",    stats["Original Words"])
    col2.metric("Summary Words",     stats["Summary Words"])
    col3.metric("Compression Ratio", stats["Compression Ratio"])
else:
    st.info("Statistics will appear here once a summary is generated.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# BONUS FEATURES
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎁 Bonus Features")

tab1, tab2, tab3 = st.tabs(["📄 Upload PDF", "📝 Upload DOCX", "⬇️ Download Summary"])

# ── Bonus 1: PDF Upload ───────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload a PDF Meeting Document")
    pdf_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_uploader"
    )
    if pdf_file is not None:
        pdf_text = extract_text_from_pdf(pdf_file.read())
        if pdf_text.startswith("[Error"):
            st.error(pdf_text)
        else:
            st.text_area("Extracted Text (preview):", value=pdf_text[:1000] + "…"
                         if len(pdf_text) > 1000 else pdf_text, height=180, disabled=True)
            if st.button("🚀 Generate Summary from PDF", key="pdf_btn", type="primary"):
                with st.spinner("Summarizing PDF…"):
                    pdf_summary = do_summarize(pdf_text)
                    st.session_state["summary"]     = pdf_summary
                    st.session_state["source_text"] = pdf_text
                st.success(pdf_summary)
                stats = compute_stats(pdf_text, pdf_summary)
                c1, c2, c3 = st.columns(3)
                c1.metric("Original Words",    stats["Original Words"])
                c2.metric("Summary Words",     stats["Summary Words"])
                c3.metric("Compression Ratio", stats["Compression Ratio"])

# ── Bonus 2: DOCX Upload ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### Upload a DOCX Meeting Document")
    docx_file = st.file_uploader(
        "Choose a DOCX file", type=["docx"], key="docx_uploader"
    )
    if docx_file is not None:
        docx_text = extract_text_from_docx(docx_file.read())
        if docx_text.startswith("[Error"):
            st.error(docx_text)
        else:
            st.text_area("Extracted Text (preview):", value=docx_text[:1000] + "…"
                         if len(docx_text) > 1000 else docx_text, height=180, disabled=True)
            if st.button("🚀 Generate Summary from DOCX", key="docx_btn", type="primary"):
                with st.spinner("Summarizing DOCX…"):
                    docx_summary = do_summarize(docx_text)
                    st.session_state["summary"]     = docx_summary
                    st.session_state["source_text"] = docx_text
                st.success(docx_summary)
                stats = compute_stats(docx_text, docx_summary)
                c1, c2, c3 = st.columns(3)
                c1.metric("Original Words",    stats["Original Words"])
                c2.metric("Summary Words",     stats["Summary Words"])
                c3.metric("Compression Ratio", stats["Compression Ratio"])

# ── Bonus 3: Download Summary as PDF ──────────────────────────────────────────
with tab3:
    st.markdown("### Download Your Summary as PDF")
    if st.session_state["summary"]:
        pdf_bytes = create_pdf_download(st.session_state["summary"])
        if isinstance(pdf_bytes, bytes):
            st.download_button(
                label="⬇️ Download Summary PDF",
                data=pdf_bytes,
                file_name="meeting_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.error(pdf_bytes)
    else:
        st.info("Generate a summary first, then download it as PDF here.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px;'>"
    "AI Meeting Minutes Summarizer · Seq2Seq LSTM · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
