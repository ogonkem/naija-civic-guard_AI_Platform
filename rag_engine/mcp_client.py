"""Persistent MCP client for the constitution-retrieval tools.

One subprocess + one ClientSession for the whole process lifetime - every
tool call is a JSON-RPC round trip over the already-open stdio pipes, NOT a
new connection. The MCP SDK is async-only, so the session lives on a private
event loop in a daemon thread and sync callers block on it.

`call(name, args)` returns (payload_dict, latency_ms, ok, error) and never
raises - a tool/transport failure comes back as ok=False so the agent can
fall back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "15"))


class McpToolClient:
    def __init__(self, server_args=None, env=None):
        self._params = StdioServerParameters(
            command=sys.executable,
            args=server_args or ["-m", "rag_engine.mcp_server"],
            env={**os.environ, **(env or {})},
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-client-loop", daemon=True
        )
        self._thread.start()
        self._session: ClientSession | None = None
        self._stack = None
        self._ready = threading.Event()
        self._start_err: Exception | None = None
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    async def _connect(self):
        from contextlib import AsyncExitStack
        try:
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(stdio_client(self._params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            logger.info("MCP client connected (%s)", " ".join(self._params.args))
        except Exception as exc:  # noqa: BLE001
            self._start_err = exc
            logger.warning("MCP client failed to start: %s", exc)
        finally:
            self._ready.set()

    def wait_ready(self, timeout: float = 30) -> bool:
        self._ready.wait(timeout)
        return self._session is not None

    def call(self, name: str, arguments: dict):
        """-> (payload: dict|None, latency_ms: float, ok: bool, error: str|None)"""
        t0 = time.perf_counter()
        if not self._ready.is_set():
            self._ready.wait(30)
        if self._session is None:
            return (None, (time.perf_counter() - t0) * 1000.0, False,
                    f"mcp session unavailable: {self._start_err}")
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, arguments), self._loop
            )
            result = fut.result(timeout=_CALL_TIMEOUT)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if getattr(result, "isError", False):
                return (None, latency_ms, False, _text(result) or "tool returned isError")
            payload = _json(_text(result))
            return (payload, latency_ms, True, None)
        except Exception as exc:  # noqa: BLE001
            return (None, (time.perf_counter() - t0) * 1000.0, False,
                    f"{type(exc).__name__}: {exc}")

    # convenience wrappers
    def lookup_section(self, number: int):
        return self.call("lookup_section", {"number": int(number)})

    def find_related_sections(self, section_id):
        return self.call("find_related_sections", {"section_id": str(section_id)})

    def search_precedent(self, query: str):
        return self.call("search_precedent", {"query": query})

    def close(self):
        if self._loop.is_closed():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._aclose(), self._loop)
            fut.result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def _aclose(self):
        if self._stack is not None:
            await self._stack.aclose()


def _text(result) -> str:
    for item in getattr(result, "content", None) or []:
        if getattr(item, "type", None) == "text" or hasattr(item, "text"):
            return item.text
    return ""


def _json(s: str):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return {"raw": s}


# --- process-wide singleton -------------------------------------------------- #
_client: McpToolClient | None = None
_lock = threading.Lock()


def get_mcp_client(env=None) -> McpToolClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = McpToolClient(env=env)
                _client.wait_ready(timeout=30)
    return _client
