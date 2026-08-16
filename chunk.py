"""
Step 3: Load ticket documents from data/ and split them into chunks.
Chunks are built from whole sentences, so we never cut a word in half.
"""

import os
import re

DATA_DIR = "data"


def load_documents(data_dir=DATA_DIR):
    """Read every .txt file in data_dir. Returns a list of (filename, text)."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))
    return documents


def split_into_sentences(text):
    """Split text into a flat list of sentences, dropping empty ones."""
    # First split on blank lines (paragraph breaks) — this treats headings like
    # "Subject: ..." as their own unit even though they have no trailing
    # punctuation, instead of letting them merge into the next real sentence.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sentences = []
    for paragraph in paragraphs:
        # Within a paragraph, collapse wrapped newlines into single spaces,
        # then split after '.', '!', or '?' followed by whitespace — a simple,
        # readable sentence splitter (not perfect for abbreviations, but fine
        # for our tickets).
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def chunk_text(text, chunk_size=300, overlap_sentences=1):
    """
    Group whole sentences into chunks of roughly chunk_size characters.
    overlap_sentences controls how many trailing sentences repeat at the
    start of the next chunk, so context isn't lost at the seam.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        # If adding this sentence would blow past chunk_size, close the
        # current chunk out (as long as it already has something in it).
        if current and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # start the next chunk with the last `overlap_sentences` sentences
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def load_and_chunk_all(data_dir=DATA_DIR, chunk_size=300, overlap_sentences=1):
    """Load every document and chunk it. Returns a list of dicts with text + source."""
    all_chunks = []
    for filename, text in load_documents(data_dir):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap_sentences=overlap_sentences)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_index": i,
            })
    return all_chunks


if __name__ == "__main__":
    result = load_and_chunk_all()
    print(f"Loaded and chunked {len(result)} chunks total.\n")
    for item in result:
        print(f"--- {item['source']} (chunk {item['chunk_index']}) ---")
        print(item["text"])
        print()
