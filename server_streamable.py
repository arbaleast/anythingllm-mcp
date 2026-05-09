#!/usr/bin/env python3
"""
AnythingLLM MCP Server — Streamable HTTP Transport
Compatible with Cline, Claude Desktop, and other MCP clients supporting StreamableHTTP.
"""
import os
import json
import time
import uuid
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Header, Query, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ─── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001").rstrip("/")
SERVER_API_KEY = os.getenv("HTTP_API_KEY", "zed-hermes-key")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8766")
TOKEN_EXPIRY = int(os.getenv("TOKEN_EXPIRY", "3600"))

# ─── State ───────────────────────────────────────────────────────────────────
access_tokens: dict = {}
auth_codes: dict = {}
oauth_states: dict = {}

def generate_token() -> str:
    return f"tk_{uuid.uuid4().hex}{uuid.uuid4().hex[:24]}"

def generate_code() -> str:
    return f"ac_{uuid.uuid4().hex}"

def verify_key(auth: str) -> bool:
    if auth is None:
        return False
    token = auth.replace("Bearer ", "").strip()
    return token == SERVER_API_KEY or token in access_tokens

def require_auth(authorization: Optional[str] = Header(None)) -> str:
    if not verify_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing authorization")
    token = authorization.replace("Bearer ", "").strip()
    return token

# ─── AnythingLLM API ──────────────────────────────────────────────────────────
async def allm_api(endpoint: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    """Call AnythingLLM API. endpoint is like '/auth', '/workspaces', '/system/settings' etc.
    Will be prepended with /api/v1 to form the full URL."""
    if not API_KEY.strip():
        raise RuntimeError("ANYTHINGLLM_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    url = f"{ANYTHINGLLM_BASE_URL}/api/v1{endpoint}"
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.request(method, url, headers=headers, json=body)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "application/json" in content_type:
            return r.json()
        return {"raw": r.text, "status": r.status_code}

# ─── MCP Tool Handlers ───────────────────────────────────────────────────────
TOOL_HANDLERS = {}

def register(name: str, schema: dict, handler):
    TOOL_HANDLERS[name] = {"name": name, "inputSchema": schema, "handler": handler}

async def h_check_auth(**kwargs):
    return await allm_api("/auth")

async def h_list_workspaces(**kwargs):
    return await allm_api("/workspaces")

async def h_get_workspace(slug: str, **kwargs):
    return await allm_api(f"/workspace/{slug}")

async def h_create_workspace(name: str, **kwargs):
    return await allm_api("/workspace/new", "POST", {"name": name})

async def h_update_workspace(slug: str, name=None, openAiTemp=None, openAiHistory=None, **kwargs):
    payload = {}
    if name is not None:
        payload["name"] = name
    if openAiTemp is not None:
        payload["openAiTemp"] = openAiTemp
    if openAiHistory is not None:
        payload["openAiHistory"] = openAiHistory
    if not payload:
        return {"success": False, "error": "No updates provided."}
    return await allm_api(f"/workspace/{slug}/update", "POST", payload)

async def h_delete_workspace(slug: str, **kwargs):
    return await allm_api(f"/workspace/{slug}", "DELETE")

async def h_chat(slug: str, message: str, mode: str = "chat", **kwargs):
    return await allm_api(f"/workspaces/{slug}/chat", "POST", {"message": message, "mode": mode})

async def h_get_chat_history(slug: str, **kwargs):
    return await allm_api(f"/workspaces/{slug}/chats")

async def h_create_thread(slug: str, name: Optional[str] = None, **kwargs):
    payload = {}
    if name: payload["name"] = name
    return await allm_api(f"/workspace/{slug}/thread/new", "POST", payload)

async def h_delete_thread(slug: str, thread_slug: str, **kwargs):
    return await allm_api(f"/workspace/{slug}/thread/{thread_slug}", "DELETE")

async def h_update_thread(slug: str, thread_slug: str, name: str, **kwargs):
    return await allm_api(f"/workspace/{slug}/thread/{thread_slug}/update", "POST", {"name": name})

async def h_chat_in_thread(slug: str, thread_slug: str, message: str, mode: str = "chat", **kwargs):
    return await allm_api(f"/workspace/{slug}/thread/{thread_slug}/chat", "POST", {"message": message, "mode": mode})

async def h_get_thread_chats(slug: str, thread_slug: str, **kwargs):
    return await allm_api(f"/workspace/{slug}/thread/{thread_slug}/chats")

async def h_get_thread(slug: str, thread_slug: str, **kwargs):
    return await allm_api(f"/workspace/{slug}/thread/{thread_slug}")

async def h_list_documents(**kwargs):
    return await allm_api("/documents")

async def h_get_accepted_file_types(**kwargs):
    return await allm_api("/document/accepted-file-types")

async def h_upload_link(link: str, **kwargs):
    return await allm_api("/document/upload-link", "POST", {"link": link})

async def h_upload_raw_text(text_content: str, title: str, **kwargs):
    return await allm_api(
        "/document/raw-text",
        "POST",
        {"textContent": text_content, "metadata": {"title": title}},
    )

async def h_upload_file(file_path: str, **kwargs):
    """Upload a file from the local filesystem. file_path must be an absolute path accessible by the server."""
    if not API_KEY.strip():
        return {"success": False, "error": "ANYTHINGLLM_API_KEY is not set"}
    import pathlib
    path = pathlib.Path(file_path)
    if not path.exists() or not path.is_file():
        return {"success": False, "error": f"File not found: {file_path}"}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f)}
            resp = await client.post(
                f"{ANYTHINGLLM_BASE_URL}/api/v1/document/upload",
                headers=headers, files=files)
    resp.raise_for_status()
    return resp.json()

async def h_upload_file_to_folder(file_path: str, folder_name: str, **kwargs):
    """Upload a file from the local filesystem into a specific folder."""
    if not API_KEY.strip():
        return {"success": False, "error": "ANYTHINGLLM_API_KEY is not set"}
    import pathlib
    path = pathlib.Path(file_path)
    if not path.exists() or not path.is_file():
        return {"success": False, "error": f"File not found: {file_path}"}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f)}
            resp = await client.post(
                f"{ANYTHINGLLM_BASE_URL}/api/v1/document/upload/{folder_name}",
                headers=headers, files=files)
    resp.raise_for_status()
    return resp.json()

async def h_list_documents_in_folder(folder_name: str, **kwargs):
    return await allm_api(f"/documents/folder/{folder_name}")

async def h_get_document(doc_name: str, **kwargs):
    return await allm_api(f"/document/{doc_name}")

async def h_get_document_metadata_schema(**kwargs):
    return await allm_api("/document/metadata-schema")

async def h_create_folder(name: str, **kwargs):
    return await allm_api("/document/create-folder", "POST", {"name": name})

async def h_remove_folder(name: str, **kwargs):
    return await allm_api("/document/remove-folder", "DELETE", {"name": name})

async def h_move_files(files: list, **kwargs):
    return await allm_api("/document/move-files", "POST", {"files": files})

async def h_update_embeddings(slug: str, adds: Optional[list] = None, deletes: Optional[list] = None, **kwargs):
    payload = {}
    if adds:
        payload["adds"] = adds
    if deletes:
        payload["deletes"] = deletes
    if not payload:
        return {"success": False, "error": "Provide at least 'adds' or 'deletes'."}
    return await allm_api(f"/workspace/{slug}/update-embeddings", "POST", payload)

async def h_update_pin(slug: str, doc_path: str, pinned: bool, **kwargs):
    return await allm_api(f"/workspace/{slug}/update-pin", "POST", {"docPath": doc_path, "pinStatus": pinned})

async def h_search_workspace(slug: str, query: str, top_n: int = 4, score_threshold: Optional[float] = None, **kwargs):
    body = {"query": query, "topN": top_n}
    if score_threshold is not None:
        body["scoreThreshold"] = score_threshold
    return await allm_api(f"/workspace/{slug}/vector-search", "POST", body)

async def h_get_system_settings(**kwargs):
    return await allm_api("/system")

async def h_get_vector_count(**kwargs):
    return await allm_api("/system/vector-count")

async def h_export_chats(**kwargs):
    return await allm_api("/system/export-chats")

async def h_remove_documents(names: list, **kwargs):
    return await allm_api("/system/remove-documents", "DELETE", {"names": names})

async def h_list_embeds(**kwargs):
    return await allm_api("/embed")

async def h_list_models(**kwargs):
    return await allm_api("/openai/models")

# Register all tools
register("anythingllm_check_auth", {"type": "object", "properties": {}}, h_check_auth)
register("anythingllm_list_workspaces", {"type": "object", "properties": {}}, h_list_workspaces)
register("anythingllm_get_workspace", {"type": "object", "properties": {"slug": {"type": "string"}}}, h_get_workspace)
register("anythingllm_create_workspace", {"type": "object", "properties": {"name": {"type": "string"}}}, h_create_workspace)
register("anythingllm_update_workspace", {"type": "object", "properties": {"slug": {"type": "string"}, "name": {"type": "string"}, "openAiTemp": {"type": "number"}, "openAiHistory": {"type": "integer"}}}, h_update_workspace)
register("anythingllm_delete_workspace", {"type": "object", "properties": {"slug": {"type": "string"}}}, h_delete_workspace)
register("anythingllm_chat", {"type": "object", "properties": {"slug": {"type": "string"}, "message": {"type": "string"}, "mode": {"type": "string", "default": "chat"}}}, h_chat)
register("anythingllm_get_chat_history", {"type": "object", "properties": {"slug": {"type": "string"}}}, h_get_chat_history)
register("anythingllm_create_thread", {"type": "object", "properties": {"slug": {"type": "string"}, "name": {"type": "string"}}}, h_create_thread)
register("anythingllm_delete_thread", {"type": "object", "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}}}, h_delete_thread)
register("anythingllm_update_thread", {"type": "object", "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}, "name": {"type": "string"}}}, h_update_thread)
register("anythingllm_chat_in_thread", {"type": "object", "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}, "message": {"type": "string"}}}, h_chat_in_thread)
register("anythingllm_get_thread_chats", {"type": "object", "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}}}, h_get_thread_chats)
register("anythingllm_get_thread", {"type": "object", "properties": {"slug": {"type": "string"}, "thread_slug": {"type": "string"}}}, h_get_thread)
register("anythingllm_list_documents", {"type": "object", "properties": {}}, h_list_documents)
register("anythingllm_get_accepted_file_types", {"type": "object", "properties": {}}, h_get_accepted_file_types)
register("anythingllm_upload_link", {"type": "object", "properties": {"link": {"type": "string"}}}, h_upload_link)
register("anythingllm_upload_raw_text", {"type": "object", "properties": {"text_content": {"type": "string"}, "title": {"type": "string"}}}, h_upload_raw_text)
register("anythingllm_upload_file", {"type": "object", "properties": {"file_path": {"type": "string"}}}, h_upload_file)
register("anythingllm_upload_file_to_folder", {"type": "object", "properties": {"file_path": {"type": "string"}, "folder_name": {"type": "string"}}}, h_upload_file_to_folder)
register("anythingllm_list_documents_in_folder", {"type": "object", "properties": {"folder_name": {"type": "string"}}}, h_list_documents_in_folder)
register("anythingllm_get_document", {"type": "object", "properties": {"doc_name": {"type": "string"}}}, h_get_document)
register("anythingllm_get_document_metadata_schema", {"type": "object", "properties": {}}, h_get_document_metadata_schema)
register("anythingllm_create_folder", {"type": "object", "properties": {"name": {"type": "string"}}}, h_create_folder)
register("anythingllm_remove_folder", {"type": "object", "properties": {"name": {"type": "string"}}}, h_remove_folder)
register(
    "anythingllm_move_files",
    {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                    "required": ["from", "to"],
                },
            }
        },
        "required": ["files"],
    },
    h_move_files,
)
register("anythingllm_update_embeddings", {"type": "object", "properties": {"slug": {"type": "string"}, "adds": {"type": "array", "items": {"type": "string"}}, "deletes": {"type": "array", "items": {"type": "string"}}}}, h_update_embeddings)
register("anythingllm_update_pin", {"type": "object", "properties": {"slug": {"type": "string"}, "doc_path": {"type": "string"}, "pinned": {"type": "boolean"}}}, h_update_pin)
register("anythingllm_search_workspace", {"type": "object", "properties": {"slug": {"type": "string"}, "query": {"type": "string"}, "top_n": {"type": "integer", "default": 4}}}, h_search_workspace)
register("anythingllm_get_system_settings", {"type": "object", "properties": {}}, h_get_system_settings)
register("anythingllm_get_vector_count", {"type": "object", "properties": {}}, h_get_vector_count)
register("anythingllm_export_chats", {"type": "object", "properties": {}}, h_export_chats)
register("anythingllm_remove_documents", {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}}}, h_remove_documents)
register("anythingllm_list_embeds", {"type": "object", "properties": {}}, h_list_embeds)
register("anythingllm_list_models", {"type": "object", "properties": {}}, h_list_models)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="AnythingLLM MCP Server (Streamable HTTP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── OAuth 2.0 Endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root path — return 200 so Cline MCP client doesn't get a 404 on initial discovery."""
    return JSONResponse({"status": "ok", "service": "anythingllm-mcp"})

@app.get("/.well-known/mcp")
async def mcp_discovery():
    """MCP service discovery — tells clients the auth scheme and endpoint URL."""
    return JSONResponse({
        "mcpServers": {
            "anythingllm": {
                "streamableHttp": {
                    "url": f"{SERVER_URL}/mcp",
                    "auth": {
                        "type": "bearer",
                        "authorizationUrl": f"{SERVER_URL}/oauth/authorize",
                        "tokenUrl": f"{SERVER_URL}/oauth/token",
                    }
                }
            }
        }
    })

@app.get("/oauth/authorize")
async def oauth_authorize(
    request: Request,
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: str = Query(""),
    state: str = Query(""),
    code_challenge: str = Query(""),
    code_challenge_method: str = Query("S256"),
):
    if response_type != "code":
        return JSONResponse({"error": "unsupported_response_type"}, status_code=400)

    state_id = state or generate_code()
    oauth_states[state_id] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
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

    from urllib.parse import urlencode
    params = {"code": code, "state": state_id}
    from fastapi.responses import RedirectResponse
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
    if grant_type == "authorization_code":
        if not code or code not in auth_codes:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        stored = auth_codes.pop(code)
        if stored["client_id"] != client_id:
            return JSONResponse({"error": "invalid_client"}, status_code=400)
        access_token = generate_token()
        access_tokens[access_token] = {"client_id": client_id, "created_at": time.time(), "scope": stored.get("scope", "")}
        return JSONResponse({"access_token": access_token, "token_type": "Bearer", "expires_in": TOKEN_EXPIRY})

    elif grant_type == "client_credentials":
        access_token = generate_token()
        access_tokens[access_token] = {"client_id": client_id, "created_at": time.time(), "scope": scope or ""}
        return JSONResponse({"access_token": access_token, "token_type": "Bearer", "expires_in": TOKEN_EXPIRY, "scope": scope or ""})

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

# ─── Streamable HTTP MCP Endpoint ─────────────────────────────────────────────

@app.get("/mcp")
async def mcp_get(request: Request):
    auth = request.headers.get("authorization", "")
    if not verify_key(auth):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse({
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "anythingllm-mcp", "version": "2.0.0"}
        }
    })

@app.post("/mcp")
async def mcp_post(request: Request):
    """Handle JSON-RPC requests via Streamable HTTP."""
    import sys
    try:
        auth = request.headers.get("authorization", "")
        if not verify_key(auth):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}, status_code=400)

        method = body.get("method", "")
        msg_id = body.get("id")
        params = body.get("params", {})

        print(f"[MCP] method={method} id={msg_id}", flush=True)

        # initialize
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "anythingllm-mcp", "version": "2.0.0"},
                    "instructions": "AnythingLLM MCP Server. Use anythingllm_* tools."
                }
            })

        # ping
        if method == "ping":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})

        # tools/list
        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "tools": [
                        {"name": t["name"], "description": f"AnythingLLM: {t['name']}", "inputSchema": t["inputSchema"]}
                        for t in TOOL_HANDLERS.values()
                    ]
                }
            })

        # tools/call
        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            tool = TOOL_HANDLERS.get(tool_name)
            if not tool:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}
                })

            try:
                result = await tool["handler"](**tool_args)
                try:
                    result_data = json.loads(result) if isinstance(result, str) else result
                    return JSONResponse({
                        "jsonrpc": "2.0", "id": msg_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result_data, ensure_ascii=False)}]}
                    })
                except (json.JSONDecodeError, TypeError):
                    return JSONResponse({
                        "jsonrpc": "2.0", "id": msg_id,
                        "result": {"content": [{"type": "text", "text": str(result)}]}
                    })
            except Exception as e:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32603, "message": str(e)}
                })

        # resources/list
        if method == "resources/list":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}})

        # prompts/list
        if method == "prompts/list":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}})

        # Unknown method
        return JSONResponse({
            "jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}, status_code=500)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "transport": "streamable-http", "api_key_configured": bool(API_KEY)})

@app.get("/debug-handler")
async def debug_handler():
    """Optional dev-only endpoint; enable with MCP_DEBUG=1."""
    if os.getenv("MCP_DEBUG", "").strip() not in ("1", "true", "yes"):
        return JSONResponse({"error": "disabled"}, status_code=404)
    h = TOOL_HANDLERS.get("anythingllm_get_system_settings")
    if not h:
        return JSONResponse({"error": "handler not found"})
    try:
        result = await h["handler"]()
        # Check if it's a raw HTML string
        if isinstance(result, dict) and result.get("raw"):
            return JSONResponse({"is_raw_html": True, "preview": result["raw"][:200], "status": result.get("status")})
        return JSONResponse({"result": str(result)[:400]})
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()[:500]})

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
