"""Standalone MCP server exposing three constitution-retrieval tools.

Run directly:  python -m rag_engine.mcp_server   (stdio transport)

It talks to ChromaDB directly via the `chromadb` client (metadata `.get`,
no embeddings) so the subprocess starts fast - no torch / sentence-transformers.
The LangGraph agent's retrieve and chain nodes call these over an MCP client
session instead of calling retrieval functions in-process.
"""

import json
import os
import re

from mcp.server.fastmcp import FastMCP

from rag_engine.chroma import COLLECTION_NAME, get_chroma_client
from rag_engine.sections import find_section_references

mcp = FastMCP(
    "naija-civic-guard-tools",
    log_level=os.getenv("MCP_LOG_LEVEL", "WARNING"),
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8100")),
)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_chroma_client().get_collection(COLLECTION_NAME)
    return _collection


def _section_chunks(number: int, limit: int = 6) -> list[str]:
    res = _get_collection().get(
        where={"section": f"Section {number}"}, include=["documents"], limit=limit
    )
    seen, out = set(), []
    for d in res.get("documents") or []:
        d = (d or "").strip()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


@mcp.tool()
def lookup_section(number: int) -> str:
    """Fetch a constitutional section's text by its number via a direct
    metadata lookup - bypasses semantic search entirely.

    Returns JSON: {"section", "found", "chunks": [...]}.
    """
    chunks = _section_chunks(number)
    return json.dumps({
        "section": f"Section {number}",
        "found": bool(chunks),
        "chunks": chunks[:4],
    })


@mcp.tool()
def find_related_sections(section_id: str) -> str:
    """Given a section (e.g. "Section 45" or "45"), return the sections its
    own text cross-references (via the shared regex tagging) together with
    their text.

    Returns JSON: {"section", "found", "references": [...],
                   "related": [{"section", "text"}, ...]}.
    """
    m = re.search(r"\d{1,3}", str(section_id))
    if not m:
        return json.dumps({"section": section_id, "found": False,
                           "error": f"no section number in {section_id!r}"})
    num = int(m.group())
    own = _section_chunks(num)
    if not own:
        return json.dumps({"section": f"Section {num}", "found": False,
                           "references": [], "related": []})

    refs = find_section_references("\n\n".join(own), exclude={f"Section {num}"})
    related = []
    for ref in refs[:5]:
        rn = int(re.search(r"\d{1,3}", ref).group())
        rc = _section_chunks(rn, limit=2)
        if rc:
            related.append({"section": ref, "text": rc[0][:900]})
    return json.dumps({
        "section": f"Section {num}",
        "found": True,
        "references": refs,
        "related": related,
    })


@mcp.tool()
def search_precedent(query: str) -> str:
    """Search Nigerian case law for precedent relevant to a query.

    Not yet implemented - case law integration is planned. Returns a clear
    message rather than an error so callers can carry on.
    """
    return json.dumps({
        "implemented": False,
        "query": query,
        "message": "not yet implemented — case law integration planned",
    })


if __name__ == "__main__":
    # stdio for a gateway-spawned subprocess (local dev); streamable-http when
    # the MCP server runs as its own container.
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
