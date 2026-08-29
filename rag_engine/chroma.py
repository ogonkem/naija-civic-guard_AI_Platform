"""ChromaDB client factory - local persistent dir or a networked server.

Set CHROMA_HOST (+ CHROMA_PORT, default 8000) to talk to a ChromaDB server
container; otherwise a local PersistentClient at CHROMA_DB_PATH is used, which
keeps `runserver` / tests working with no extra services.

Pure module: no Django, so rag_engine/mcp_server.py can import it too.
"""

import os

import chromadb

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "langchain")  # langchain_chroma default


def get_chroma_client():
    host = os.getenv("CHROMA_HOST")
    if host:
        return chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")))
    return chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "chroma_db"))
