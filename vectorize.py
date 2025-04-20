from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

TEXT_CHUNK_PATH = "output/text_chunks.json"
VECTOR_PATH = "output/vector_store.npy"

# Configurable chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHAIN = 3  # ⬅️ limit merging to 2 chunks max

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

def is_truncated(text):
    # If the last character is not proper punctuation, assume truncation
    return text.strip()[-1:] not in [".", "!", "?", "\"", "”", "’", ")", "]"]

def chain_truncated_chunks(chunks, max_chain=2):
    combined = []
    buffer = ""
    chain_count = 0

    for i, chunk in enumerate(chunks):
        if buffer:
            chunk = buffer + " " + chunk
            chain_count += 1
        else:
            chain_count = 0

        # If still truncated and not over the limit, hold to merge next one
        if is_truncated(chunk) and chain_count < max_chain and i < len(chunks) - 1:
            buffer = chunk
        else:
            combined.append(chunk)
            buffer = ""
            chain_count = 0

    if buffer:
        combined.append(buffer)

    return combined

def vectorize_chunks(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    with open(TEXT_CHUNK_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    # Step 1: Apply overlapping chunking
    texts = []
    for chunk in raw_chunks:
        split_chunks = chunk_with_overlap(chunk['text'], CHUNK_SIZE, CHUNK_OVERLAP)
        texts.extend(split_chunks)

    # Step 2: Clean and merge truncated sequences
    cleaned_texts = chain_truncated_chunks(texts, max_chain=MAX_CHAIN)

    # Step 3: Vectorize
    vectors = model.encode(cleaned_texts, convert_to_numpy=True)

    np.save(VECTOR_PATH, vectors)
    print(f"✅ Vectorized {len(cleaned_texts)} cleaned chunks (with overlap + chaining).")

    return cleaned_texts, vectors

if __name__ == "__main__":
    vectorize_chunks()
