#!/usr/bin/env python3
"""Blocking and non-blocking gateways for the vLLM concurrency experiment."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException


UPSTREAM_URL = os.environ.get("VLLM_UPSTREAM_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_S = float(os.environ.get("GATEWAY_REQUEST_TIMEOUT_S", "600"))


def blocking_post(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{UPSTREAM_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=body) from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/blocking")
async def blocking_gateway(payload: dict[str, Any]) -> dict[str, Any]:
    # This intentionally blocks the single Uvicorn event loop. It is the
    # negative control that demonstrates why `async def` alone is insufficient.
    return blocking_post(payload)


@app.post("/threaded")
async def threaded_gateway(payload: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(blocking_post, payload)


@app.post("/async")
async def async_gateway(payload: dict[str, Any]) -> dict[str, Any]:
    response = await app.state.client.post("/v1/chat/completions", json=payload)
    if response.is_error:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
