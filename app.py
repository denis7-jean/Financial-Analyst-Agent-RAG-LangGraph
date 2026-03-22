# app.py
import os
import re
import uuid
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.graph import app, store

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="Financial Analyst Agent", page_icon="📈")

with st.sidebar:
    st.title("📈 Financial Analyst Agent")
    st.markdown(
        """
        An agentic RAG system for autonomous analysis of SEC 10-K
        filings, powered by LangGraph explicit routing and persistent
        memory.

        **Tools**
        - **search_10k**: Hybrid retrieval (BM25 + ChromaDB + RRF)
          over ingested 10-K chunks with cell-level table extraction.
        - **calculator**: Deterministic arithmetic — no mental math.
        - **yfinance_tool**: Live stock price and market data.
        - **web_search**: Macroeconomic and news search via Tavily.

        **Capabilities**
        - Financial metrics with page-level citations
        - Multi-turn memory across conversation turns
        - Cross-tool queries (10-K + market data + math)
        - Narrative sections: risk factors, MD&A, strategy

        **Currently ingested:** Apple 2024 Form 10-K
        """
    )

    st.markdown("---")
    st.markdown("**👤 User Identity**")
    user_id = st.sidebar.text_input(
        "User ID (for long-term memory)",
        value="guest",
        help="Enter a consistent ID to retain memory across sessions."
    )

    st.markdown("---")
    if st.sidebar.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("**🧠 Long-Term Memory**")
    try:
        memories = store.search(("tickers", user_id))
        if memories:
            st.markdown("Remembered tickers:")
            for m in memories:
                st.markdown(f"- **{m.value['ticker']}** "
                           f"(queried {m.value['count']} time(s))")
        else:
            st.markdown("_No tickers remembered yet._")
    except Exception:
        st.markdown("_Memory unavailable._")

st.title("Financial Analyst Chat")


def safe_markdown(text: str) -> None:
    # Escape $ signs that are not intended as LaTeX delimiters
    escaped = re.sub(r'\$(?!\$)', r'\\$', text)
    st.markdown(escaped)

# -------------------- Session State -------------------- #
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
st.session_state.user_id = user_id

# -------------------- Display Chat History -------------------- #
for msg in st.session_state["messages"]:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        safe_markdown(msg.content)

# -------------------- Handle User Input -------------------- #
user_input = st.chat_input("Ask about Apple's 10-K, or request a calculation...")
if user_input:
    # Append and display user message
    user_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user"):
        st.write(user_input)

    # Prepare inputs for the agent (full history)
    inputs = {"messages": st.session_state["messages"]}

    final_response = None

    # Optional: visualize agent steps
    status_box = st.status("🤖 Agent is thinking...", state="running")

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id,
        }
    }

    for event in app.stream(inputs, stream_mode="values", config=config):
        last_msg = event["messages"][-1]

        # Tool call visualization
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.get("name", "tool")
                status_box.update(label=f"🤖 Using tool: {tool_name}", state="running")
        else:
            # Only treat assistant messages as final response
            if getattr(last_msg, "type", None) == "ai":
                final_response = last_msg.content
                status_box.update(label="✅ Response ready", state="complete")

    # Display final AI response
    if final_response:
        ai_msg = AIMessage(content=final_response)
        st.session_state["messages"].append(ai_msg)
        with st.chat_message("assistant"):
            safe_markdown(final_response)
        st.rerun()
