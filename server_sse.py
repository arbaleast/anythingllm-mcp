#!/usr/bin/env python3
"""
MCP SSE Server for AnythingLLM with a simplified OAuth-style token exchange.

Security: authorization auto-approves and does not verify PKCE code_verifier.
Use only on a trusted network; set HTTP_API_KEY to a strong secret for static Bearer access.

Endpoints:
  GET  /                         → redirect to /oauth/authorize
  GET  /.well-known/mcp          → service discovery JSON
  GET  /oauth/authorize           → authorization page (GET with query params)
  POST /oauth/token               → exchange code for token (PKCE)
  GET  /sse                       → SSE stream (must have valid Bearer token)
  POST /messages                 → JSON-RPC requests (must have valid Bearer token)
  POST /upload                    → file upload (multipart, optional auth)
  GET  /health                   → health check (no auth)
"""
import os
import json
import time
import uuid
import asyncio
import hashlib
import base64
from urllib.parse import urlencode
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header, Form, Response, Query, UploadFile, File
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
)
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001").rstrip("/")
API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")
SERVER_API_KEY = os.environ.get("HTTP_API_KEY", "zed-hermes-key")
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8766")

# ──────────────────────────────────────────────
# AnythingLLM API
# ──────────────────────────────────────────────

async def allm_api(path: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.request(method, f"{API_BASE_URL}/api/v1{path}",
                                     headers=headers, json=body)
    if resp.status_code >= 400:
        # Try to parse error; if not JSON, use status text
        try:
            err = resp.json()
            msg = err.get("error", err.get("message", resp.text))
        except Exception:
            msg = resp.text or resp.reason_phrase
        raise httpx.HTTPStatusError(message=msg, request=resp.request, response=resp)
    resp.raise_for_status()
    return resp.json()

# ──────────────────────────────────────────────
# OAuth 2.0 State Management (in-memory for simplicity)
# ──────────────────────────────────────────────

oauth_states: dict = {}    # state → {client_id, redirect_uri, scope, code_verifier, created_at}
auth_codes: dict = {}      # code → {client_id, redirect_uri, access_token, created_at, scopes}
access_tokens: dict = {}   # token → {client_id, created_at, scopes}

AUTH_CODE_EXPIRY = 600      # 10 minutes
TOKEN_EXPIRY = 3600        # 1 hour

def generate_code() -> str:
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")

def generate_token() -> str:
    return base64.urlsafe_b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes).decode().rstrip("=")

# ──────────────────────────────────────────────
# MCP Tool Handlers
# ──────────────────────────────────────────────

TOOL_HANDLERS = {}

def register(name: str, schema: dict, handler):
    TOOL_HANDLERS[name] = {"name": name, "inputSchema": schema, "handler": handler}


# ── Auth ──

async def h_check_auth(**kwargs):
    result = await allm_api("/auth")
    return json.dumps(result)


# ── Workspaces ──

async def h_list_workspaces(**kwargs):
    result = await allm_api("/workspaces")
    workspaces = result.get("workspaces", [])
    summary = [{
        "name": ws.get("name"), "slug": ws.get("slug"),
        "chatMode": ws.get("chatMode", "chat"),
        "vectorSearchMode": ws.get("vectorSearchMode"),
        "threads": len(ws.get("threads", [])),
        "createdAt": ws.get("createdAt"),
    } for ws in workspaces]
    return json.dumps({"total": len(summary), "workspaces": summary})


async def h_get_workspace(slug: str, **kwargs):
    result = await allm_api(f"/workspace/{slug}")
    return json.dumps(result)


async def h_create_workspace(name: str, **kwargs):
    result = await allm_api("/workspace/new", method="POST", body={"name": name})
    return json.dumps(result)


async def h_update_workspace(slug: str, name=None, openAiTemp=None, openAiHistory=None,
                             openAiPrompt=None, similarityThreshold=None, topN=None,
                             chatMode=None, **kwargs):
    body = {k: v for k, v in {
        "name": name, "openAiTemp": openAiTemp, "openAiHistory": openAiHistory,
        "openAiPrompt": openAiPrompt, "similarityThreshold": similarityThreshold,
        "topN": topN, "chatMode": chatMode
    }.items() if v is not None}
    if not body:
        return json.dumps({"success": False, "error": "No updates provided."})
    result = await allm_api(f"/workspace/{slug}/update", method="POST", body=body)
    return json.dumps(result)


async def h_delete_workspace(slug: str, **kwargs):
    try:
        await allm_api(f"/workspace/{slug}", method="DELETE")
        return json.dumps({"success": True, "message": f"Workspace '{slug}' deleted successfully."})
    except httpx.HTTPStatusError as e:
        return json.dumps({"success": False, "error": str(e.message)})


# ── Chat ──

async def h_chat(slug: str, message: str, mode: str = "chat", **kwargs):
    result = await allm_api(f"/workspace/{slug}/chat", method="POST",
                            body={"message": message, "mode": mode})
    return json.dumps(result)


async def h_get_chat_history(slug: str, **kwargs):
    result = await allm_api(f"/workspace/{slug}/chats")
    return json.dumps(result)


# ── Threads ──

async def h_create_thread(slug: str, name: Optional[str] = None, **kwargs):
    body = {}
    if name:
        body["name"] = name
    result = await allm_api(f"/workspace/{slug}/thread/new", method="POST", body=body)
    return json.dumps(result)


async def h_delete_thread(slug: str, thread_slug: str, **kwargs):
    try:
        await allm_api(f"/workspace/{slug}/thread/{thread_slug}", method="DELETE")
        return json.dumps({"success": True, "message": f"Thread '{thread_slug}' deleted."})
    except httpx.HTTPStatusError as e:
        return json.dumps({"success": False, "error": str(e.message)})


async def h_update_thread(slug: str, thread_slug: str, name: str, **kwargs):
    result = await allm_api(f"/workspace/{slug}/thread/{thread_slug}/update",
                            method="POST", body={"name": name})
    return json.dumps(result)


async def h_chat_in_thread(slug: str, thread_slug: str, message: str,
                           mode: str = "chat", **kwargs):
    result = await allm_api(f"/workspace/{slug}/thread/{thread_slug}/chat", method="POST",
                            body={"message": message, "mode": mode})
    return json.dumps(result)


async def h_get_thread_chats(slug: str, thread_slug: str, **kwargs):
    result = await allm_api(f"/workspace/{slug}/thread/{thread_slug}/chats")
    return json.dumps(result)


async def h_get_thread(slug: str, thread_slug: str, **kwargs):
    result = await allm_api(f"/workspace/{slug}/thread/{thread_slug}")
    return json.dumps(result)


# ── Documents ──

async def h_list_documents(**kwargs):
    result = await allm_api("/documents")
    return json.dumps(result)


async def h_get_accepted_file_types(**kwargs):
    result = await allm_api("/document/accepted-file-types")
    return json.dumps(result)


async def h_upload_link(link: str, **kwargs):
    result = await allm_api("/document/upload-link", method="POST",
                            body={"link": link})
    return json.dumps(result)


async def h_upload_raw_text(text_content: str, title: str, **kwargs):
    result = await allm_api(
        "/document/raw-text",
        method="POST",
        body={"textContent": text_content, "metadata": {"title": title}},
    )
    return json.dumps(result)


async def h_upload_file(file_path: str, **kwargs):
    """Upload a file from the local filesystem. file_path must be an absolute path accessible by the server."""
    import pathlib
    path = pathlib.Path(file_path)
    if not path.exists() or not path.is_file():
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f)}
            resp = await client.post(
                f"{API_BASE_URL}/api/v1/document/upload",
                headers=headers, files=files)
    resp.raise_for_status()
    return json.dumps(resp.json())


async def h_upload_file_to_folder(file_path: str, folder_name: str, **kwargs):
    """Upload a file from the local filesystem into a specific folder."""
    import pathlib
    path = pathlib.Path(file_path)
    if not path.exists() or not path.is_file():
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f)}
            resp = await client.post(
                f"{API_BASE_URL}/api/v1/document/upload/{folder_name}",
                headers=headers, files=files)
    resp.raise_for_status()
    return json.dumps(resp.json())


async def h_list_documents_in_folder(folder_name: str, **kwargs):
    result = await allm_api(f"/documents/folder/{folder_name}")
    return json.dumps(result)


async def h_get_document(doc_name: str, **kwargs):
    result = await allm_api(f"/document/{doc_name}")
    return json.dumps(result)


async def h_get_document_metadata_schema(**kwargs):
    result = await allm_api("/document/metadata-schema")
    return json.dumps(result)


async def h_create_folder(name: str, **kwargs):
    result = await allm_api("/document/create-folder", method="POST", body={"name": name})
    return json.dumps(result)


async def h_remove_folder(name: str, **kwargs):
    result = await allm_api("/document/remove-folder", method="DELETE", body={"name": name})
    return json.dumps(result)


async def h_move_files(files: list, **kwargs):
    result = await allm_api("/document/move-files", method="POST", body={"files": files})
    return json.dumps(result)


# ── Embeddings ──

async def h_update_embeddings(slug: str, adds: Optional[list] = None,
                              deletes: Optional[list] = None, **kwargs):
    body = {}
    if adds is not None:
        body["adds"] = adds
    if deletes is not None:
        body["deletes"] = deletes
    if not body:
        return json.dumps({"success": False, "error": "Provide at least 'adds' or 'deletes'."})
    result = await allm_api(f"/workspace/{slug}/update-embeddings", method="POST", body=body)
    return json.dumps(result)


async def h_update_pin(slug: str, doc_path: str, pinned: bool, **kwargs):
    result = await allm_api(f"/workspace/{slug}/update-pin", method="POST",
                            body={"docPath": doc_path, "pinStatus": pinned})
    return json.dumps(result)


# ── Search ──

async def h_search_workspace(slug: str, query: str, top_n: int = 4,
                              score_threshold: Optional[float] = None, **kwargs):
    body = {"query": query, "topN": top_n}
    if score_threshold is not None:
        body["scoreThreshold"] = score_threshold
    result = await allm_api(f"/workspace/{slug}/vector-search", method="POST", body=body)
    return json.dumps(result)


# ── System ──

async def h_get_system_settings(**kwargs):
    result = await allm_api("/system")
    return json.dumps(result)


async def h_get_vector_count(**kwargs):
    result = await allm_api("/system/vector-count")
    return json.dumps(result)


async def h_export_chats(**kwargs):
    result = await allm_api("/system/export-chats")
    return json.dumps(result)


async def h_remove_documents(names: list, **kwargs):
    result = await allm_api("/system/remove-documents", method="DELETE", body={"names": names})
    return json.dumps(result)


# ── Embeds ──

async def h_list_embeds(**kwargs):
    result = await allm_api("/embed")
    return json.dumps(result)


# ── Models ──

async def h_list_models(**kwargs):
    result = await allm_api("/openai/models")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Register all tools
# ──────────────────────────────────────────────

register("anythingllm_check_auth", {"type": "object", "properties": {}}, h_check_auth)

register("anythingllm_list_workspaces",
    {"type": "object", "properties": {}}, h_list_workspaces)

register("anythingllm_get_workspace",
    {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]},
    h_get_workspace)

register("anythingllm_create_workspace",
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    h_create_workspace)

register("anythingllm_update_workspace",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"},
         "name": {"type": "string"},
         "openAiTemp": {"type": "number"},
         "openAiHistory": {"type": "integer"},
         "openAiPrompt": {"type": "string"},
         "similarityThreshold": {"type": "number"},
         "topN": {"type": "integer"},
         "chatMode": {"type": "string"}
     },
     "required": ["slug"]},
    h_update_workspace)

register("anythingllm_delete_workspace",
    {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]},
    h_delete_workspace)

register("anythingllm_chat",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"}, "message": {"type": "string"}, "mode": {"type": "string"}
     },
     "required": ["slug", "message"]},
    h_chat)

register("anythingllm_get_chat_history",
    {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]},
    h_get_chat_history)

register("anythingllm_create_thread",
    {"type": "object",
     "properties": {"slug": {"type": "string"}, "name": {"type": "string"}},
     "required": ["slug"]},
    h_create_thread)

register("anythingllm_delete_thread",
    {"type": "object",
     "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}},
     "required": ["slug", "thread_slug"]},
    h_delete_thread)

register("anythingllm_update_thread",
    {"type": "object",
     "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}, "name": {"type": "string"}},
     "required": ["slug", "thread_slug", "name"]},
    h_update_thread)

register("anythingllm_chat_in_thread",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"}, "thread_slug": {"type": "string"},
         "message": {"type": "string"}, "mode": {"type": "string"}
     },
     "required": ["slug", "thread_slug", "message"]},
    h_chat_in_thread)

register("anythingllm_get_thread_chats",
    {"type": "object",
     "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}},
     "required": ["slug", "thread_slug"]},
    h_get_thread_chats)

register("anythingllm_get_thread",
    {"type": "object",
     "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}},
     "required": ["slug", "thread_slug"]},
    h_get_thread)

register("anythingllm_list_documents",
    {"type": "object", "properties": {}}, h_list_documents)

register("anythingllm_get_accepted_file_types",
    {"type": "object", "properties": {}}, h_get_accepted_file_types)

register("anythingllm_upload_link",
    {"type": "object",
     "properties": {"link": {"type": "string"}}, "required": ["link"]},
    h_upload_link)

register("anythingllm_upload_raw_text",
    {"type": "object",
     "properties": {"text_content": {"type": "string"}, "title": {"type": "string"}},
     "required": ["text_content", "title"]},
    h_upload_raw_text)

register("anythingllm_upload_file",
    {"type": "object",
     "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    h_upload_file)

register("anythingllm_upload_file_to_folder",
    {"type": "object",
     "properties": {"file_path": {"type": "string"}, "folder_name": {"type": "string"}},
     "required": ["file_path", "folder_name"]},
    h_upload_file_to_folder)

register("anythingllm_list_documents_in_folder",
    {"type": "object",
     "properties": {"folder_name": {"type": "string"}}, "required": ["folder_name"]},
    h_list_documents_in_folder)

register("anythingllm_get_document",
    {"type": "object",
     "properties": {"doc_name": {"type": "string"}}, "required": ["doc_name"]},
    h_get_document)

register("anythingllm_get_document_metadata_schema",
    {"type": "object", "properties": {}}, h_get_document_metadata_schema)

register("anythingllm_create_folder",
    {"type": "object",
     "properties": {"name": {"type": "string"}}, "required": ["name"]},
    h_create_folder)

register("anythingllm_remove_folder",
    {"type": "object",
     "properties": {"name": {"type": "string"}}, "required": ["name"]},
    h_remove_folder)

register("anythingllm_move_files",
    {"type": "object",
     "properties": {
         "files": {
             "type": "array",
             "items": {
                 "type": "object",
                 "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                 "required": ["from", "to"]
             }
         }
     },
     "required": ["files"]},
    h_move_files)

register("anythingllm_update_embeddings",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"},
         "adds": {"type": "array", "items": {"type": "string"}},
         "deletes": {"type": "array", "items": {"type": "string"}}
     },
     "required": ["slug"]},
    h_update_embeddings)

register("anythingllm_update_pin",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"}, "doc_path": {"type": "string"}, "pinned": {"type": "boolean"}
     },
     "required": ["slug", "doc_path", "pinned"]},
    h_update_pin)

register("anythingllm_search_workspace",
    {"type": "object",
     "properties": {
         "slug": {"type": "string"}, "query": {"type": "string"},
         "top_n": {"type": "integer"}, "score_threshold": {"type": "number"}
     },
     "required": ["slug", "query"]},
    h_search_workspace)

register("anythingllm_get_system_settings",
    {"type": "object", "properties": {}}, h_get_system_settings)

register("anythingllm_get_vector_count",
    {"type": "object", "properties": {}}, h_get_vector_count)

register("anythingllm_export_chats",
    {"type": "object", "properties": {}}, h_export_chats)

register("anythingllm_remove_documents",
    {"type": "object",
     "properties": {"names": {"type": "array", "items": {"type": "string"}}},
     "required": ["names"]},
    h_remove_documents)

register("anythingllm_list_embeds",
    {"type": "object", "properties": {}}, h_list_embeds)

register("anythingllm_list_models",
    {"type": "object", "properties": {}}, h_list_models)


# ──────────────────────────────────────────────
# MCP Session
# ──────────────────────────────────────────────

class MCPSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def send(self, event: dict):
        if not self._closed:
            await self.queue.put(json.dumps(event))

    async def stream(self):
        yield f"event: endpoint\ndata: /messages?sessionId={self.session_id}\n\n"
        while not self._closed:
            try:
                msg = await asyncio.wait_for(self.queue.get(), timeout=30)
                if msg is None:
                    break
                yield f"event: message\ndata: {msg}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

    def close(self):
        self._closed = True
        self.queue.put_nowait(None)

class SessionManager:
    def __init__(self):
        self.sessions: dict[str, MCPSession] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> MCPSession:
        sess = MCPSession(str(uuid.uuid4()))
        async with self._lock:
            self.sessions[sess.session_id] = sess
        return sess

    async def get(self, sid: str) -> Optional[MCPSession]:
        return self.sessions.get(sid)

    async def remove(self, sid: str):
        sess = self.sessions.pop(sid, None)
        if sess:
            sess.close()

sessions = SessionManager()

# ──────────────────────────────────────────────
# Auth Helper
# ──────────────────────────────────────────────

def require_auth(authorization: Optional[str] = Header(None)) -> str:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    # Accept both OAuth-issued tokens and the static HTTP API key
    if token not in access_tokens and token != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid access token")
    return token

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(title="AnythingLLM MCP/SSE Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── OAuth-compliant 404 handler ───
# VSCode MCP client expects {"error": "..."} on unknown routes, not {"detail": "..."}
@app.exception_handler(HTTPException)
async def oauth_exception_handler(request: Request, exc: HTTPException):
    # Only handle 404 — let other errors through normally
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    raise exc

# Middleware to catch non-matched routes (FastAPI default 404 bypasses exception_handler)
@app.middleware("http")
async def catch_404_middleware(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404 and not hasattr(response, "_is_mcp_stream"):
        # Replace FastAPI's {"detail":"Not Found"} with OAuth-style {"error":"not_found"}
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return response

# ─── Service Discovery ───

@app.get("/")
async def root():
    return RedirectResponse(url="/oauth/authorize")

@app.get("/.well-known/mcp")
async def mcp_discovery():
    """MCP service discovery endpoint."""
    return JSONResponse({
        "mcpServers": {
            "anythingllm": {
                "sse": {
                    "url": f"{SERVER_URL}/sse",
                    "auth": {
                        "type": "oauth",
                        "authorizationUrl": f"{SERVER_URL}/oauth/authorize",
                        "tokenUrl": f"{SERVER_URL}/oauth/token",
                    }
                }
            }
        }
    })

# ─── OAuth 2.0 Endpoints ───

@app.get("/oauth/authorize")
async def oauth_authorize(
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: str = Query(""),
    state: str = Query(""),
    code_challenge: str = Query(""),
    code_challenge_method: str = Query("S256"),
):
    """Authorization page. Auto-confirm for local LAN use."""
    if response_type != "code":
        return JSONResponse({"error": "unsupported_response_type"}, status_code=400)

    state_id = state or generate_code()
    oauth_states[state_id] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_verifier": None,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "created_at": time.time(),
    }

    code = generate_code()
    auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state_id,
        "created_at": time.time(),
    }

    params = {"code": code, "state": state_id}
    return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}", status_code=302)


@app.post("/oauth/token")
async def oauth_token(
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    client_id: str = Form(...),
    redirect_uri: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
):
    """Exchange authorization code for access token (PKCE flow)."""
    if grant_type == "authorization_code":
        if not code or code not in auth_codes:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)

        stored = auth_codes.pop(code)
        if stored["client_id"] != client_id:
            return JSONResponse({"error": "invalid_client"}, status_code=400)

        access_token = generate_token()
        access_tokens[access_token] = {
            "client_id": client_id,
            "created_at": time.time(),
            "scope": stored.get("scope", ""),
        }

        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": TOKEN_EXPIRY,
            "scope": stored.get("scope", ""),
        })

    elif grant_type == "client_credentials":
        access_token = generate_token()
        access_tokens[access_token] = {
            "client_id": client_id,
            "created_at": time.time(),
            "scope": scope or "",
        }
        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": TOKEN_EXPIRY,
            "scope": scope or "",
        })

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


# ─── SSE & Messages ───

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "api_key_configured": bool(API_KEY)})

@app.get("/sse")
async def sse_endpoint(authorization: str = Header(None)):
    """Establish SSE channel. Requires valid Bearer token."""
    require_auth(authorization)
    session = await sessions.create()
    return StreamingResponse(
        session.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/messages")
async def messages_endpoint(
    sessionId: str = Query(...),
    authorization: str = Header(None),
    request: Request = None,
):
    """Handle JSON-RPC requests. Results sent via SSE stream."""
    import sys
    print(f"[MESSAGES] sessionId={sessionId}", flush=True)
    try:
        require_auth(authorization)
        session = await sessions.get(sessionId)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        body = await request.json()
        method = body.get("method", "")
        msg_id = body.get("id")
        print(f"[MESSAGES] method={method} id={msg_id}", flush=True)

        # initialize
        if method == "initialize":
            await session.send(
                {
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "anythingllm-mcp-sse", "version": "1.0.0"}
                    }
                }
            )
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"status": "ok"}})

        # tools/list
        if method == "tools/list":
            await session.send(
                {
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"tools": [{"name": t["name"], "inputSchema": t["inputSchema"]} for t in TOOL_HANDLERS.values()]}
                }
            )
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"status": "ok"}})

        # ping
        if method == "ping":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})

        # tools/call
        if method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            tool = TOOL_HANDLERS.get(tool_name)
            if not tool:
                await session.send(
                    {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}}
                )
                return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"status": "ok"}})

            try:
                result = await tool["handler"](**tool_args)
                try:
                    result_data = json.loads(result)
                    await session.send(
                        {
                            "jsonrpc": "2.0", "id": msg_id,
                            "result": {"content": [{"type": "text", "text": json.dumps(result_data, ensure_ascii=False)}]}
                        }
                    )
                except json.JSONDecodeError:
                    await session.send(
                        {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": result}]}}
                    )
            except Exception as e:
                await session.send(
                    {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(e)}}
                )

            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"status": "ok"}})

        # Unknown method
        return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}})

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stdout)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{sessionId}")
async def delete_session(sessionId: str, authorization: str = Header(None)):
    require_auth(authorization)
    await sessions.remove(sessionId)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
