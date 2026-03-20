# app.py
import os
import uuid
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.graph import app

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="Financial Analyst Agent", page_icon="💰")

with st.sidebar:
    st.title("💰 Financial Analyst Agent")
    st.markdown(
        """
        This agent uses **RAG + Tools** to answer questions about
        the 2024 Apple 10-K filing and can also perform calculations.
        - **search_10k**: Retrieves relevant excerpts from the filing.
        - **calculator**: Executes math expressions.
        """
    )

st.title("Financial Analyst Chat")

# -------------------- Session State -------------------- #
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# -------------------- Display Chat History -------------------- #
for msg in st.session_state["messages"]:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

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

    for event in app.stream(inputs, stream_mode="values", config={"configurable": {"thread_id": st.session_state.thread_id}}):
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
            st.write(final_response)
