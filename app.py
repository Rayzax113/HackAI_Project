import streamlit as st
import google.generativeai as genai
import json
import threading
from gtts import gTTS
import tempfile
import base64
import os
import speech_recognition as sr
import time

# --- Configure Gemini API ---
genai.configure(api_key="AIzaSyCG8eJe42KsXWssIfGyL_Hpx44CW5HVP9A")
model = genai.GenerativeModel("gemini-1.5-flash")

@st.cache_data
def load_chunks(path="output/text_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [chunk["text"] for chunk in chunks if chunk.get("text")]

st.set_page_config(page_title="Annual Report Chat Agent", page_icon="🌿", layout="centered")
st.title("🌿 LTIMindtree's AI Assistant")
st.caption("Ask questions about LTIMindtree’s 2023-24 sustainability report")

# --- State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = []
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ➕ Upload Report")
    st.file_uploader("Attach Annual Report (PDF)", type=["pdf"], label_visibility="collapsed")
    st.markdown("### 🔊 Voice Mode")
    tts_enabled = st.checkbox("Read answers aloud", value=False)
    st.markdown("### 🎙️ Voice Input")
    voice_input_enabled = st.checkbox("Use microphone to speak question", value=False)

# --- Button to Trigger Recording ---
if st.button("🎙 Speak Now"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("🧠 Transcribing...")
            transcribed = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {transcribed}")
            st.session_state.voice_input = transcribed
            st.rerun()
        except sr.WaitTimeoutError:
            st.warning("⏱️ Timeout: No speech detected.")
        except sr.UnknownValueError:
            st.warning("🤷 Sorry, I couldn't understand what you said.")
        except Exception as e:
            st.warning(f"⚠️ Speech error: {e}")

# --- Question Input ---
transcribed_text = st.session_state.voice_input
user_input = transcribed_text
submitted = bool(user_input)

if not submitted:
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([0.88, 0.12])
        with col1:
            user_input = st.text_input(
                "Ask your question:",
                value="",
                placeholder="e.g., What are LTIMindtree’s ESG goals for 2024?",
                label_visibility="collapsed",
                key="question_input"
            )
        with col2:
            submitted = st.form_submit_button("➤", use_container_width=True)

# --- Follow-up Check ---
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
    st.session_state.voice_input = ""  # reset
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Processing..."):
        chunks = load_chunks()

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
            refined_query = user_input

        matching_chunks = [
            chunk for chunk in chunks
            if any(word in chunk.lower() for word in refined_query.lower().split())
        ]
        context = "\n\n".join(matching_chunks[:10])

        is_follow_up = False
        context_history = ""
        if st.session_state.qa_cache:
            last_q = st.session_state.qa_cache[-1][0]
            if check_if_follow_up(model, last_q, user_input):
                context_history = "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_cache[-5:]]
                )

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

        st.session_state.qa_cache.append((user_input, final_answer))
        if len(st.session_state.qa_cache) > 5:
            st.session_state.qa_cache.pop(0)

        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# --- Chat Display ---
last_answer = None
for i in reversed(range(0, len(st.session_state.chat_history) - 1, 2)):
    user = st.session_state.chat_history[i]
    assistant = st.session_state.chat_history[i + 1]
    with st.chat_message("user"):
        st.write(user["content"])
    with st.chat_message("assistant"):
        st.write(assistant["content"])
        if i == len(st.session_state.chat_history) - 2:
            last_answer = assistant["content"]


# --- TTS Output ---
if tts_enabled and last_answer:
    try:
        tts = gTTS(text=last_answer, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
        with open(fp.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <audio autoplay="true" hidden="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

        if voice_input_enabled:
            time.sleep(1.5)
            st.rerun()
    except Exception as e:
        st.warning(f"🔈 TTS playback failed: {e}")
