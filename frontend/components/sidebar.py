"""
Sidebar Component — System status, document list, and configuration.
"""

import streamlit as st
import requests

from components.upload import render_upload_section


def render_sidebar(api_base: str):
    """
    Render the sidebar with upload section, system status, document list, and configuration.

    Args:
        api_base: Base URL for the FastAPI backend.
    """
    with st.sidebar:
        st.title("🔍 RAG OCR System")
        st.caption("Retrieval-Augmented Generation with OCR")

        st.divider()

        # Upload section
        render_upload_section(api_base)

        st.divider()

        # System status
        _render_status(api_base)

        st.divider()

        # Indexed documents
        _render_documents(api_base)

        st.divider()

        # Configuration
        _render_config()

        st.divider()

        # Info
        st.caption(
            "💡 **How to use:**\n"
            "1. Upload a PDF or Image document in the sidebar\n"
            "2. Wait for processing to complete\n"
            "3. Ask questions in the chat view\n\n"
            "You can also drop files into the `ingestion/` folder "
            "for automatic processing."
        )


def _render_status(api_base: str):
    """Render system status section."""
    st.subheader("📊 System Status")

    try:
        resp = requests.get(f"{api_base}/api/status", timeout=5)
        if resp.status_code == 200:
            status = resp.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", status.get("total_documents", 0))
            with col2:
                st.metric("Chunks", status.get("total_chunks", 0))

            # LLM status
            if status.get("llm_available"):
                st.success(f"🤖 LLM: {status.get('llm_model', 'Connected')}")
            else:
                st.warning("🤖 LLM: Not configured (context-only mode)")

            # File watcher status
            if status.get("file_watcher_running"):
                st.success("👁️ File watcher: Active")
            else:
                st.warning("👁️ File watcher: Stopped")
        else:
            st.error("Could not fetch status")
    except requests.ConnectionError:
        st.error("⚠️ Backend not connected")
    except Exception as e:
        st.error(f"Status error: {e}")


def _render_documents(api_base: str):
    """Render indexed documents list."""
    st.subheader("📄 Indexed Documents")

    try:
        resp = requests.get(f"{api_base}/api/documents", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            documents = data.get("documents", [])

            if not documents:
                st.info("No documents indexed yet. Upload a PDF or image to get started!")
                return

            for doc in documents:
                name = doc.get("name", "Unknown")
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📎 {name}")
                with col2:
                    if st.button("🗑️", key=f"del_{name}", help=f"Delete {name}"):
                        _delete_document(api_base, name)
                        st.rerun()
    except requests.ConnectionError:
        st.info("Connect to backend to see documents")
    except Exception:
        pass


def _delete_document(api_base: str, doc_name: str):
    """Delete a document from the index."""
    try:
        resp = requests.delete(f"{api_base}/api/documents/{doc_name}", timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                st.success(f"Deleted '{doc_name}' ({result.get('chunks_deleted', 0)} chunks)")
            else:
                st.warning(f"Document '{doc_name}' not found in index")
        else:
            st.error(f"Failed to delete: {resp.text}")
    except Exception as e:
        st.error(f"Delete error: {e}")


def _render_config():
    """Render configuration controls."""
    st.subheader("⚙️ Settings")

    top_k = st.slider(
        "Context chunks (top-k)",
        min_value=1,
        max_value=20,
        value=st.session_state.get("top_k", 5),
        help="Number of document chunks to retrieve for each query.",
        key="top_k_slider",
    )
    st.session_state["top_k"] = top_k
