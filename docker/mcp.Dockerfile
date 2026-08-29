# Standalone MCP tool server (Phase 5). Small image: MCP SDK + chromadb client,
# no torch / Django. Serves streamable-http on :8100.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8100

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mcp.txt .
RUN pip install -r requirements-mcp.txt

# Just the modules mcp_server.py needs (rag_engine/__init__.py is empty).
COPY rag_engine/__init__.py rag_engine/sections.py rag_engine/chroma.py rag_engine/mcp_server.py ./rag_engine/

EXPOSE 8100
CMD ["python", "-m", "rag_engine.mcp_server"]
