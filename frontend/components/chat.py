"""
Chat Component — Displays chat messages with user/assistant bubbles.

Shows the chat history, retrieved source citations, and handles
message submission.
"""

import streamlit as st


def render_chat_history():
    """Render all messages in the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            st.markdown(content)

            # Show sources if available
            if role == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("Sources", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• **{src['file']}** — Page {src['page']}")

            # Show model info
            if role == "assistant" and msg.get("llm_used"):
                st.caption(f"Answered by: {msg.get('model', 'Groq LLM')}")
            elif role == "assistant" and msg.get("llm_used") is False:
                st.caption("Context-only response (no LLM configured)")


def add_user_message(content: str):
    """Add a user message to the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": content})


def add_assistant_message(
    content: str,
    sources: list[dict] | None = None,
    llm_used: bool = False,
    model: str | None = None,
):
    """Add an assistant message to the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
        "llm_used": llm_used,
        "model": model,
    })
