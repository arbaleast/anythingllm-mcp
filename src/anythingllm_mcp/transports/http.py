#!/usr/bin/env python3
"""
FastAPI HTTP wrapper for anythingllm-mcp tools.

Upstream AnythingLLM REST API uses /api/v1/... and Bearer authentication,
same as anythingllm_mcp.py.
"""
import os
import httpx
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001").rstrip("/")
API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")
HTTP_API_KEY = os.environ.get("HTTP_API_KEY", "")

# ──────────────────────────────────────────────
# AnythingLLM API calls
# ──────────────────────────────────────────────


async def _allm(endpoint: str, method: str = "GET", body: Optional[dict] = None) -> Any:
    """Call AnythingLLM; endpoint is e.g. '/workspaces', '/workspace/{slug}/chat'."""
    if not API_KEY.strip():
        raise RuntimeError("ANYTHINGLLM_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}/api/v1{endpoint}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.request(method, url, headers=headers, json=body)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        return resp.json()
    return resp.text


def _json_response(data: Any) -> dict:
    return {"success": True, "data": data}


def _wrap_error(e: Exception) -> dict:
    return {"success": False, "error": str(e)}


def verify_key(authorization: Optional[str] = Header(None)) -> str:
    if not HTTP_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="HTTP_API_KEY is not set; configure it to use this gateway.",
        )
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token != HTTP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return authorization


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(title="AnythingLLM HTTP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "anythingllm_api_key_configured": bool(API_KEY.strip()),
        "http_gateway_key_configured": bool(HTTP_API_KEY),
        "base_url": API_BASE_URL,
    }


# ──────────────────────────────────────────────
# Workspaces
# ──────────────────────────────────────────────


@app.get("/api/workspaces")
async def list_workspaces(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/workspaces")
        workspaces = result.get("workspaces", [])
        summary = [
            {
                "name": ws.get("name"),
                "slug": ws.get("slug"),
                "chatMode": ws.get("chatMode", "chat"),
                "vectorSearchMode": ws.get("vectorSearchMode"),
                "threads": len(ws.get("threads", [])),
                "createdAt": ws.get("createdAt"),
            }
            for ws in workspaces
        ]
        return _json_response({"total": len(summary), "workspaces": summary})
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/workspaces/{slug}")
async def get_workspace(slug: str, authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm(f"/workspace/{slug}")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.post("/api/workspaces")
async def create_workspace(payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/workspace/new", method="POST", body={"name": payload["name"]})
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.patch("/api/workspaces/{slug}")
async def update_workspace(slug: str, payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        body = {k: v for k, v in payload.items() if v is not None}
        if not body:
            return _wrap_error(ValueError("No updates provided"))
        result = await _allm(f"/workspace/{slug}/update", method="POST", body=body)
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────


@app.post("/api/workspaces/{slug}/chat")
async def chat_with_workspace(slug: str, payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm(
            f"/workspace/{slug}/chat",
            method="POST",
            body={"message": payload["message"], "mode": payload.get("mode", "chat")},
        )
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/workspaces/{slug}/chats")
async def get_chat_history(slug: str, authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm(f"/workspace/{slug}/chats")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


# ──────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────


@app.get("/api/documents")
async def list_documents(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/documents")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/documents/accepted-file-types")
async def get_accepted_file_types(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/document/accepted-file-types")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.post("/api/documents/raw-text")
async def upload_raw_text(payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm(
            "/document/raw-text",
            method="POST",
            body={"textContent": payload["text_content"], "metadata": {"title": payload.get("title", "")}},
        )
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.post("/api/documents/create-folder")
async def create_folder(payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/document/create-folder", method="POST", body={"name": payload["name"]})
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.post("/api/documents/move-files")
async def move_files(payload: dict = Body(...), authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/document/move-files", method="POST", body={"files": payload["files"]})
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


# ──────────────────────────────────────────────
# System
# ──────────────────────────────────────────────


@app.get("/api/system")
async def get_system(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/system")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/system/vector-count")
async def get_vector_count(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/system/vector-count")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/system/export-chats")
async def export_chats(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/system/export-chats")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/embed")
async def get_embed(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/embed")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


@app.get("/api/openai/models")
async def get_models(authorization: str = Header(None)):
    verify_key(authorization)
    try:
        result = await _allm("/openai/models")
        return _json_response(result)
    except Exception as e:
        return _wrap_error(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
