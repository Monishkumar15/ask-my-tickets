"""
Ingestion entry point: rebuild the vector database from everything in data/.

Run this after adding, editing, or removing files in data/, or after
changing EMBEDDING_MODEL_NAME in .env. Run `python app.py` afterward to
ask questions against the freshly-built index.
"""

from store import build_database
from vectorstore import count


def main():
    client = build_database()
    print(f"Ingestion complete: {count(client)} chunks stored.")


if __name__ == "__main__":
    main()
