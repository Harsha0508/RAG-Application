"""
RAG OCR System — Streamlit Frontend Entry Point.

Provides a chat interface for querying indexed documents,
a file upload widget for ingesting new PDFs, and a sidebar
with system status and configuration.
"""

import streamlit as st
import requests

from components.chat import add_assistant_message, add_user_message, render_chat_history
from components.sidebar import render_sidebar
from components.upload import render_upload_section

# ── Page Configuration ───────────────────────────────────────────────

st.set_page_config(
    page_title="RAG OCR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container layout & scroll space */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 900px;
    }

    /* Fixed bottom chat bar container */
    [data-testid="stBottom"] {
        background-color: transparent;
    }

    /* Chat input styling */
    .stChatInput > div {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Metrics styling */
    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #313244;
    }

    /* Upload section in sidebar */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Divider spacing */
    hr {
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ── Initialize Session State ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# ── Sidebar ──────────────────────────────────────────────────────────

render_sidebar(API_BASE)

# ── Main Content ─────────────────────────────────────────────────────

st.title("🔍 RAG OCR Chat")
st.caption("Ask questions about your uploaded PDF and image documents")

# Fixed-height chat history container with its own scroll bar.
# The input stays pinned at the bottom via Streamlit's chat_input.
chat_container = st.container(height=560, border=False)

with chat_container:
    render_chat_history()

# Chat input placed at root level — anchors fixed at bottom of screen
if prompt := st.chat_input("Ask a question about your documents..."):
    add_user_message(prompt)

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/api/chat",
                        json={
                            "query": prompt,
                            "top_k": st.session_state.get("top_k", 5),
                        },
                        timeout=60,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer generated.")
                        sources = data.get("sources", [])
                        llm_used = data.get("llm_used", False)
                        model = data.get("model")

                        st.markdown(answer)

                        if sources:
                            with st.expander("📄 Sources", expanded=False):
                                for src in sources:
                                    st.caption(f"• **{src['file']}** — Page {src['page']}")

                        if llm_used:
                            st.caption(f"🤖 Answered by: {model or 'Groq LLM'}")
                        else:
                            st.caption("📋 Context-only response (set GROQ_API_KEY for LLM answers)")

                        add_assistant_message(answer, sources, llm_used, model)

                    else:
                        error_msg = f"Server error ({response.status_code}): {response.text}"
                        st.error(error_msg)
                        add_assistant_message(f"❌ {error_msg}")

                except requests.ConnectionError:
                    error_msg = (
                        "❌ Could not connect to the backend. "
                        "Please start the FastAPI server with:\n\n"
                        "```\npython main.py\n```"
                    )
                    st.error(error_msg)
                    add_assistant_message(error_msg)

                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    add_assistant_message(error_msg)

    st.rerun()





