import streamlit as st
import google.generativeai as genai
import json

# Configure Gemini API
genai.configure(api_key="AIzaSyCG8eJe42KsXWssIfGyL_Hpx44CW5HVP9A")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load pre-extracted ESG report chunks
@st.cache_data
def load_chunks(path="output/text_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [chunk["text"] for chunk in chunks if chunk.get("text")]

# Page config
st.set_page_config(page_title="ESG Chat Agent", page_icon="🌿", layout="centered")
st.title("🌿 ESG Report Q&A Assistant")
st.caption("Ask questions about LTIMindtree’s 2023-24 sustainability report")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload UI (placeholder for future functionality)
with st.sidebar:
    st.markdown("### ➕ Upload Report")
    st.file_uploader("Attach Annual Report (PDF)", type=["pdf"], label_visibility="collapsed")

# Text input and submit button
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

# Generate response on submission
if submitted and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Rewriting query and parsing report..."):
        chunks = load_chunks()

        # STEP 1: Rewrite question into a focused search query
        rewrite_prompt = f"""
You are helping an AI system search a sustainability report more effectively.
Rewrite the following user question into a short, focused search query that can be used to retrieve relevant content from an ESG document.

User Question: "{user_input}"

Return ONLY the refined query — no commentary or formatting.
"""
        try:
            rewrite_response = model.generate_content(rewrite_prompt)
            refined_query = rewrite_response.text.strip()
        except Exception as e:
            refined_query = user_input  # fallback if rewriting fails

        # STEP 2: Filter chunks using refined query (simple keyword match)
        matching_chunks = [
            chunk for chunk in chunks
            if any(word in chunk.lower() for word in refined_query.lower().split())
        ]
        context = "\n\n".join(matching_chunks[:10])  # limit to top 10

        # STEP 3: Ask Gemini to answer the original question using the matched context
        final_prompt = f"""
You are an AI assistant reviewing an annual sustainability report.
Use the provided content below to answer the user's question as accurately and concisely as possible.

--- Report Content ---
{context}

--- User Question ---
{user_input}

Answer based only on the provided content. If the answer isn't available, say so.
"""

        try:
            response = model.generate_content(final_prompt)
            final_answer = response.text.strip()
        except Exception as e:
            final_answer = f"⚠️ Error generating response: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# Render chat history as reversed Q&A pairs (newest at top)
history = st.session_state.chat_history
pairs = []
i = 0
while i < len(history) - 1:
    if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
        pairs.append((history[i], history[i + 1]))
        i += 2
    else:
        i += 1  # skip unpaired

for user_msg, bot_msg in reversed(pairs):
    with st.chat_message(user_msg["role"]):
        st.write(user_msg["content"])
    with st.chat_message(bot_msg["role"]):
        st.write(bot_msg["content"])
