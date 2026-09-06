"""
Ingestion pipeline: documents -> extracted text -> chunks -> embeddings ->
stored vectors in Qdrant.

This is the consolidated ingestion side of the app (previously split across
loaders.py, chunk.py, embed.py, vectorstore.py, store.py, and ingest.py --
merged into one file since they were always one pipeline, just split into
many small files while we were learning each step individually).

Run this file directly to (re)build the index:
    python ingestion.py
"""

import os
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_DIM,
    EMBEDDING_MODEL_NAME,
    OVERLAP_SENTENCES,
    QDRANT_URL,
)

# ===========================================================================
# 1. Document loading -- one function per file format, dispatched by extension
# ===========================================================================


def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_md(filepath):
    # Markdown is treated as plain text -- headings/lists still read fine
    # through the paragraph/sentence splitter below.
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_docx(filepath):
    import docx  # python-docx

    doc = docx.Document(filepath)
    blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Also pull in table content -- ticket data can live in tables, not just
    # paragraphs. Each row becomes one pipe-joined line.
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    return "\n\n".join(blocks)


def load_pdf(filepath):
    # PyMuPDF (import name "fitz"), not pypdf -- pypdf's plain text extraction
    # doesn't reconstruct reading order for multi-column/list layouts and
    # mangles some Unicode (smart quotes, ligatures came out as "�", hyphenated
    # line-wraps came out as "custom - ers"). PyMuPDF preserves both far more
    # reliably, which matters here: a "Follow these steps:" chunk followed by
    # scrambled/garbled text is a chunk the LLM correctly can't answer from,
    # even though the real content was right there in the source PDF.
    import pymupdf

    pages = []
    with pymupdf.open(filepath) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text() or ""
            if text.strip():
                pages.append((page_number, text))
    return pages
    # Unlike the other loaders (a single string), this returns a list of
    # (page_number, page_text) -- so chunk_text can be run per page below and
    # every chunk knows which PDF page it came from. Scanned/image-only
    # pages yield "" and are dropped here; if every page is empty the file
    # is skipped by the caller with a warning. No OCR support by design.


def load_xlsx(filepath):
    import openpyxl

    wb = openpyxl.load_workbook(filepath, data_only=True)
    lines = []

    for sheet in wb.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue

        # First row is the header -- each other row becomes a "Column: value"
        # sentence-like line, which reads naturally through the sentence
        # splitter (unlike raw pipe-joined cell values).
        header = [str(h).strip() if h is not None else "" for h in rows[0]]
        lines.append(f"Sheet: {sheet.title}")
        for row in rows[1:]:
            pairs = [
                f"{col}: {val}"
                for col, val in zip(header, row)
                if col and val is not None and str(val).strip()
            ]
            if pairs:
                lines.append(". ".join(pairs) + ".")

    return "\n\n".join(lines)


# Dispatch by file extension -- add one line here + one function above to
# support a new format later.
LOADERS = {
    ".txt": load_txt,
    ".md": load_md,
    ".docx": load_docx,
    ".pdf": load_pdf,
    ".xlsx": load_xlsx,
}


def load_document_text(filepath):
    """
    Return a list of (page_number, text) units for one file, or None if
    unreadable. Never raises -- the caller decides how to warn/skip.

    A PDF yields one unit per page (page_number starting at 1), so a later
    chunk can cite exactly which page it came from. Every other format has
    no notion of a "page", so it yields a single unit with page_number=None
    -- those chunks just cite the filename, same as before this existed.
    """
    ext = os.path.splitext(filepath)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        return None
    try:
        result = loader(filepath)
    except Exception as e:
        print(f"WARNING: could not read '{filepath}' ({e}). Skipping.")
        return None

    if ext == ".pdf":
        return result  # already a list of (page_number, text)
    return [(None, result)]


def load_documents(data_dir=DATA_DIR):
    """
    Read every supported document in data_dir (.txt, .md, .docx, .pdf, .xlsx).
    Unsupported or unreadable files are skipped with a warning instead of
    crashing the whole run. Returns a list of (filename, units), where units
    is a list of (page_number, text) -- see load_document_text().
    """
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in LOADERS:
            print(f"WARNING: skipping unsupported file type '{filename}'.")
            continue

        units = load_document_text(filepath)
        if not units or not any(text.strip() for _, text in units):
            print(f"WARNING: no extractable text in '{filename}', skipping.")
            continue

        documents.append((filename, units))
    return documents


# ===========================================================================
# 2. Chunking -- sentence-aware, so a chunk boundary never falls mid-word
# ===========================================================================


def split_into_sentences(text):
    """Split text into a flat list of sentences, dropping empty ones."""
    # First split on blank lines (paragraph breaks) -- this treats headings
    # like "Subject: ..." as their own unit even with no trailing
    # punctuation, instead of merging them into the next real sentence.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sentences = []
    for paragraph in paragraphs:
        # Collapse wrapped newlines into single spaces, then split after
        # '.', '!', or '?' followed by whitespace.
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES):
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
        if current and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def load_and_chunk_all(data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES):
    """
    Load every document and chunk it. Returns a list of dicts with text,
    source, chunk_index, and page (the PDF page number a chunk came from,
    or None for every other file format).

    Chunking runs per page for a PDF -- so a sentence never straddles a
    page boundary and every chunk can be attributed to exactly one page.
    The trade-off: overlap_sentences doesn't carry across a page break
    (there's nothing to overlap with, since the previous page's chunking
    already finished). chunk_index stays a single sequence across the
    whole file, same as before.
    """
    all_chunks = []
    for filename, units in load_documents(data_dir):
        chunk_index = 0
        for page, text in units:
            for chunk in chunk_text(text, chunk_size=chunk_size, overlap_sentences=overlap_sentences):
                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_index": chunk_index,
                    "page": page,
                })
                chunk_index += 1
    return all_chunks


# ===========================================================================
# 3. Embedding -- turn chunk text (or a question) into a vector
# ===========================================================================

MODEL_NAME = EMBEDDING_MODEL_NAME


def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks, model=None):
    """
    Given a list of chunk dicts, add an 'embedding' key to each one. Returns
    the same list, enriched.
    """
    if model is None:
        model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


# ===========================================================================
# 4. Vector store -- thin wrapper around Qdrant
# ===========================================================================


def get_client(url=QDRANT_URL):
    """Connect to a running Qdrant server, or an in-memory instance if url=None."""
    return QdrantClient(url=url) if url else QdrantClient(":memory:")


def build_collection(client, chunks, embeddings, collection_name=COLLECTION_NAME):
    """
    Delete-and-recreate the collection, then store every chunk + its vector.
    Rebuilding from scratch (rather than upserting) avoids two problems:
      1) stale chunks lingering after a source file is renamed/removed;
      2) leftover vectors from a previous embedding model silently mixing
         with new ones (same vector size doesn't mean same vector space).
    """
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "page": chunk.get("page"),
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(client, query_vector, top_k=3, collection_name=COLLECTION_NAME):
    """
    Find the top_k closest chunks to query_vector. Returns a list of dicts:
    text, source, chunk_index, distance (lower = more relevant).
    """
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )

    matches = []
    for point in result.points:
        # Qdrant's COSINE "score" is a similarity (higher = closer). We
        # convert it to a "distance" (lower = closer) so the rest of the
        # app has one consistent convention.
        matches.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "page": point.payload.get("page"),
            "distance": 1 - point.score,
        })
    return matches


def count(client, collection_name=COLLECTION_NAME):
    """How many chunks are currently stored."""
    return client.count(collection_name=collection_name).count


def all_chunks(client, collection_name=COLLECTION_NAME):
    """Return every chunk's payload -- used by the React document browser."""
    records, _ = client.scroll(collection_name=collection_name, limit=1000, with_payload=True, with_vectors=False)
    return [{"id": point.id, **point.payload} for point in records]


# ===========================================================================
# 5. Orchestration -- chunk -> embed -> store, in one call
# ===========================================================================


def build_database():
    """
    Chunk the documents, embed them, and store everything in Qdrant.
    Rebuilds the collection from scratch every run -- see build_collection()'s
    docstring for why.
    """
    chunks = load_and_chunk_all()
    model = get_embedding_model()
    chunks = embed_chunks(chunks, model=model)

    client = get_client()
    embeddings = [c["embedding"].tolist() for c in chunks]
    build_collection(client, chunks, embeddings)

    return client


if __name__ == "__main__":
    client = build_database()
    print(f"Ingestion complete: {count(client)} chunks stored in Qdrant at "
          f"'{QDRANT_URL}' (collection: '{COLLECTION_NAME}').")
