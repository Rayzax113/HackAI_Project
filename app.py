import streamlit as st
import google.generativeai as genai
import json
import pyttsx3
import threading
from gtts import gTTS
import tempfile
import os
import base64
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- Configure Gemini API ---
genai.configure(api_key="AIzaSyCG8eJe42KsXWssIfGyL_Hpx44CW5HVP9A")  # Replace with your actual key
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Load ESG Chunks ---
@st.cache_data
def load_chunks(path="output/text_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [chunk["text"] for chunk in chunks if chunk.get("text")]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ESG Chat Agent", page_icon="🌿", layout="centered")
st.title("🌿 ESG Report Q&A Assistant")
st.caption("Ask questions about LTIMindtree’s 2023-24 sustainability report")

# --- Session State Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ➕ Upload Report")
    st.file_uploader("Attach Annual Report (PDF)", type=["pdf"], label_visibility="collapsed")

    st.markdown("### 🔊 Voice Mode")
    tts_enabled = st.checkbox("Read answers aloud", value=False)

# --- Question Input ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([0.88, 0.12])
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            placeholder="e.g., What are LTIMindtree’s ESG goals for 2024?",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

# --- Check if question is a follow-up ---
def check_if_follow_up(model, previous_q: str, new_q: str) -> bool:
    prompt = f"""
Determine whether the new question is a follow-up to the previous one.

Previous Question: "{previous_q}"
New Question: "{new_q}"

If the new question builds upon, references, or seeks clarification on the previous one, respond with "Yes". Otherwise, respond with "No".
Respond with only "Yes" or "No".
"""
    try:
        response = model.generate_content(prompt)
        return "yes" in response.text.strip().lower()
    except Exception:
        return False

# --- Handle Submission ---
if submitted and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Rewriting query and parsing report..."):
        chunks = load_chunks()

        # Step 1: Rewrite question for better search
        rewrite_prompt = f"""
You are helping an AI system search a sustainability report more effectively.
Rewrite the following user question into a short, focused search query that can be used to retrieve relevant content from an ESG document.

User Question: "{user_input}"

Return ONLY the refined query — no commentary or formatting.
"""
        try:
            rewrite_response = model.generate_content(rewrite_prompt)
            refined_query = rewrite_response.text.strip()
        except Exception:
            refined_query = user_input  # fallback

        # Step 2: Find matching report chunks
        matching_chunks = [
            chunk for chunk in chunks
            if any(word in chunk.lower() for word in refined_query.lower().split())
        ]
        context = "\n\n".join(matching_chunks[:10])

        # Step 3: Check if follow-up
        is_follow_up = False
        qa_cache = st.session_state.qa_cache
        context_history = ""
        if len(qa_cache) > 0:
            prev_question = qa_cache[-1][0]
            is_follow_up = check_if_follow_up(model, prev_question, user_input)
            if is_follow_up:
                context_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_cache[-5:]])

        # Step 4: Build final prompt safely
        history_block = f"Here is the conversation so far:\n{context_history}\n\n" if context_history else ""

        final_prompt = f"""
You are an AI assistant reviewing an annual sustainability report.

{history_block}
Use the provided content below to answer the new user question as accurately and concisely as possible.

--- Report Content ---
{context}

--- User Question ---
{user_input}

Answer based only on the provided report content and relevant prior conversation (if included).
"""
        try:
            response = model.generate_content(final_prompt)
            final_answer = response.text.strip()
        except Exception as e:
            final_answer = f"⚠️ Error generating response: {e}"

        # Save new Q&A to cache (limit to 5)
        st.session_state.qa_cache.append((user_input, final_answer))
        if len(st.session_state.qa_cache) > 5:
            st.session_state.qa_cache.pop(0)

        # Append assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# --- Render Chat History (newest on top) ---
history = st.session_state.chat_history
pairs = []
i = 0
last_answer = None

while i < len(history) - 1:
    if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
        pairs.append((history[i], history[i + 1]))
        last_answer = history[i + 1]["content"]
        i += 2
    else:
        i += 1

for user_msg, bot_msg in reversed(pairs):
    with st.chat_message(user_msg["role"]):
        st.write(user_msg["content"])
    with st.chat_message(bot_msg["role"]):
        st.write(bot_msg["content"])

if tts_enabled and last_answer:
    try:
        # Convert text to speech and save to file
        tts = gTTS(text=last_answer, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_path = fp.name

        # Read audio content and encode as base64
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()

        # Create auto-playing hidden audio HTML
        audio_html = f"""
        <audio autoplay="true" hidden="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """

        # Inject into the app
        st.markdown(audio_html, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"🔈 TTS playback failed: {e}")


