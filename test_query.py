import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
VECTOR_PATH = "output/vector_store.npy"
TEXT_CHUNK_PATH = "output/text_chunks.json"

# Load overlapped text chunks from original chunks
def chunk_with_overlap(text: str, size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks

def load_text_chunks():
    with open(TEXT_CHUNK_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    all_chunks = []
    for chunk in raw_chunks:
        split = chunk_with_overlap(chunk["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(split)
    return all_chunks

def extract_keywords(query, top_n=5):
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

def rerank_by_keywords(chunks, keywords):
    scored_chunks = []
    for chunk in chunks:
        score = sum(kw.lower() in chunk.lower() for kw in keywords)
        scored_chunks.append((chunk, score))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks]

if __name__ == "__main__":
    # Load vectors and text
    vectors = np.load(VECTOR_PATH)
    texts = load_text_chunks()

    # Input and keyword extraction
    query = input("🔍 Enter your question: ")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([query])[0]
    keywords = extract_keywords(query)
    print(f"🧠 Keywords extracted: {keywords}")

    # Vector search
    scores = cosine_similarity([query_vector], vectors)[0]
    top_k = scores.argsort()[::-1][:25]
    top_chunks = [texts[i] for i in top_k]

    # Rerank
    refined_chunks = rerank_by_keywords(top_chunks, keywords)

    # Display
    for i, chunk in enumerate(refined_chunks[:3]):
        print(f"\n--- MATCH [{i}] ---\n{chunk[:1000]}\n")
