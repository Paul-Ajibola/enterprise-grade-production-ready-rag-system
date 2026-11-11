import os
import json
import streamlit as st
import requests
import uuid
import logfire
from dotenv import load_dotenv

# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# Initialize LOGFIRE_STATUS globally so it is guaranteed to exist
LOGFIRE_STATUS = "Connected"

# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token: 
        LOGFIRE_STATUS = "Standby (Missing Token)"
    logfire.configure(token=token)
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"

# ----PAGE CONFIG ------
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# ---AVATARS ----
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# ------ SESSION MANAGEMENT -----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f" New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- SIDEBAR --------
with st.sidebar:
    st.title("Agent OS")
    st.markdown("-----")
    
    # Render appropriate message alert based on connection health
    if "Standby" in LOGFIRE_STATUS:
        st.warning(f"Logfire: {LOGFIRE_STATUS}")
    else:
        st.success(f"Logfire: {LOGFIRE_STATUS}")
        
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("Clear History & Memory", use_container_width=True, type="primary"):
        logfire.warning(f"Memory Wipe Triggered for Session: {st.session_state.session_id}")
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
        try:
            response = requests.delete(
                f"{base_url}/threads/{st.session_state.session_id}", timeout=10
            )
            response.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Could not clear saved conversation memory: {e}")
            st.stop()
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ------ MAIN CHAT --------
st.title("Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation...."):
    with logfire.span("User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            answer_placeholder = st.empty()
            streamed_answer = ""
            with st.status("Agent is thinking ....", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
                        url = f"{base_url}/query/stream"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(url, json=payload, timeout=600, stream=True)
                        response.raise_for_status() 
                        for line in response.iter_lines(decode_unicode=True):
                            if isinstance(line, bytes):
                                line = line.decode("utf-8")
                            if not line or not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            if event.get("type") == "token":
                                streamed_answer += event.get("token", "")
                                answer_placeholder.markdown(streamed_answer + "▌")
                            elif event.get("type") == "final":
                                data = event
                            elif event.get("type") == "error":
                                raise RuntimeError(event.get("error", "Backend stream failed."))

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f" * {step}")

                    status.update(label="Answer Synthesized", state='complete')
                    
                    # ---- SHOW SOURCES (NESTED EXPANDABLES) ----
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("View Retrieved Context (Source)"):
                            for i, source in enumerate(sources):
                                if isinstance(source, dict):
                                    content = source.get("content", "")
                                    source_name = source.get("source", "Unknown source")
                                    score = source.get("score")
                                    preview = content[:100].replace("\n", " ")
                                    title = f"Chunk {i+1} · {source_name}"
                                    if score is not None:
                                        title += f" · score {score:.3f}"
                                    with st.expander(f"{title}: {preview}..."):
                                        st.info(content)
                                else:
                                    preview = str(source)[:100].replace("\n", " ")
                                    with st.expander(f"Chunk {i+1}: {preview}..."):
                                        st.info(str(source))
                
                except Exception as e:
                    logfire.error(f"UI-Backend Connection Failed: {e}")
                    status.update(label="Connection Failed", state="error")
                    st.error(f"Backend Offline or Error: {e}")
                    st.stop()

            full_answer = data.get("answer", "No response.")
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("Chat cycle completed successfully")
