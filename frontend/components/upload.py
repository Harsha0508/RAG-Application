"""
Upload Component — PDF upload with progress and feedback.
"""

import streamlit as st
import requests


def render_upload_section(api_base: str):
    """
    Render the PDF upload section.

    Args:
        api_base: Base URL for the FastAPI backend (e.g., "http://localhost:8000")
    """
    st.subheader("📤 Upload Documents")

    uploaded_file = st.file_uploader(
        "Drop a document or image here or click to browse",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="Upload a PDF or image file (PNG, JPG, JPEG, WEBP) for ingestion.",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

        if st.button("🚀 Process Document", key="process_btn", type="primary"):
            _process_upload(api_base, uploaded_file)


def _process_upload(api_base: str, uploaded_file):
    """Upload and process the file via the FastAPI backend."""
    with st.spinner(f"Processing **{uploaded_file.name}**... This may take a moment."):
        try:
            content_type = uploaded_file.type or "application/octet-stream"
            response = requests.post(
                f"{api_base}/api/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), content_type)},
                timeout=300,  # 5 minutes for large files
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    st.success(
                        f"✅ **{result['file_name']}** processed successfully!\n\n"
                        f"• Pages: {result['pages_processed']}\n"
                        f"• Chunks: {result['chunks_added']}\n"
                        f"• Time: {result['processing_time_seconds']:.1f}s"
                    )
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ Server error ({response.status_code}): {response.text}")

        except requests.ConnectionError:
            st.error(
                "❌ Could not connect to the backend. "
                "Make sure the FastAPI server is running on the correct port."
            )
        except requests.Timeout:
            st.error("❌ Request timed out. The document may be too large or complex.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
