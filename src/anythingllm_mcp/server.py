#!/usr/bin/env python3
"""
MCP Server for AnythingLLM.

Provides tools for managing workspaces, chatting, handling documents,
and administering system settings via the AnythingLLM REST API.

Architecture:
- Uses models.py for Pydantic data models (Data Layer)
- Uses api/client.py for HTTP communication (Service Layer)
- This module provides MCP tools as thin controllers
"""

import json
import logging
import pathlib
from typing import Optional, Any, cast

from mcp.server.fastmcp import FastMCP

from config import get_config
from models import ChatMode, VectorSearchMode, WorkspaceUpdate
from api.client import (
    AnythingLLMAPIClient,
    get_api_client,
    close_api_client,
    format_api_error,
    ValidationError,
)

# ──────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anythingllm_mcp")


# ──────────────────────────────────────────────
# FastMCP Server Setup
# ──────────────────────────────────────────────

_server_config = get_config()

mcp = FastMCP(
    "anythingllm_mcp",
    instructions=(
        "MCP server for AnythingLLM. Provides workspace management, chat, "
        "document handling, embedding, and system administration tools. "
        "All tools require a running AnythingLLM instance."
    ),
    host=_server_config.server_host,
    port=_server_config.server_port,
)


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────

def _validate_range(name: str, value: float | int, minimum: float, maximum: float) -> None:
    """Validate numeric value is within inclusive range."""
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _json_response(data: Any) -> str:
    """Serialize data to pretty JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _handle_error(e: Exception) -> str:
    """Return a user-friendly error message with logging."""
    return format_api_error(e)


# ──────────────────────────────────────────────
# Tools: Authentication
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_check_auth",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def check_auth() -> str:
    """Verify that the API key is valid and AnythingLLM is reachable."""
    try:
        client = get_api_client()
        result = await client.check_auth()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Workspaces
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_list_workspaces",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def list_workspaces() -> str:
    """List all workspaces in the AnythingLLM instance with their slugs, settings, and thread info."""
    try:
        client = get_api_client()
        workspaces = await client.list_workspaces()
        summary = []
        for ws in workspaces:
            summary.append({
                "name": ws.get("name"),
                "slug": ws.get("slug"),
                "chatMode": ws.get("chatMode", "chat"),
                "vectorSearchMode": ws.get("vectorSearchMode"),
                "threads": len(ws.get("threads", [])),
                "createdAt": ws.get("createdAt"),
            })
        return _json_response({"total": len(summary), "workspaces": summary})
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_workspace",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_workspace(slug: str) -> str:
    """Get detailed information about a specific workspace.

    Args:
        slug: Workspace slug (e.g. 'papers', 'lands')
    """
    try:
        client = get_api_client()
        result = await client.get_workspace(slug)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_create_workspace",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def create_workspace(name: str) -> str:
    """Create a new workspace.

    Args:
        name: Name for the new workspace
    """
    try:
        client = get_api_client()
        result = await client.create_workspace(name=name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_update_workspace",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def update_workspace(
    slug: str,
    name: Optional[str] = None,
    openAiTemp: Optional[float] = None,
    openAiHistory: Optional[int] = None,
    openAiPrompt: Optional[str] = None,
    similarityThreshold: Optional[float] = None,
    topN: Optional[int] = None,
    chatMode: Optional[ChatMode] = None,
) -> str:
    """Update workspace settings (name, temperature, prompt, similarity threshold, etc.).

    Args:
        slug: Workspace slug to update
        name: New workspace name
        openAiTemp: LLM temperature (0.0-1.0)
        openAiHistory: Chat history length (0-100)
        openAiPrompt: System prompt override
        similarityThreshold: Similarity threshold (0.0-1.0)
        topN: Top N results for context (1-20)
        chatMode: Chat mode: 'chat' or 'query'
    """
    try:
        if openAiTemp is not None:
            _validate_range("openAiTemp", openAiTemp, 0.0, 1.0)
        if openAiHistory is not None:
            _validate_range("openAiHistory", openAiHistory, 0, 100)
        if similarityThreshold is not None:
            _validate_range("similarityThreshold", similarityThreshold, 0.0, 1.0)
        if topN is not None:
            _validate_range("topN", topN, 1, 20)

        updates = WorkspaceUpdate(
            name=name,
            openAiTemp=openAiTemp,
            openAiHistory=openAiHistory,
            openAiPrompt=openAiPrompt,
            similarityThreshold=similarityThreshold,
            topN=topN,
            chatMode=chatMode,
        )
        client = get_api_client()
        result = await client.update_workspace(slug, updates)
        return _json_response(result)
    except ValidationError as e:
        return _handle_error(e)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_delete_workspace",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def delete_workspace(slug: str) -> str:
    """Permanently delete a workspace. This action cannot be undone.

    Args:
        slug: Workspace slug to delete
    """
    try:
        client = get_api_client()
        await client.delete_workspace(slug)
        return f"Workspace '{slug}' deleted successfully."
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Chat
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_chat",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def chat_with_workspace(
    slug: str,
    message: str,
    mode: ChatMode = ChatMode.CHAT,
    sessionId: Optional[str] = None,
    reset: bool = False,
) -> str:
    """Send a message to a workspace and get a response.

    Mode 'chat' uses LLM general knowledge w/custom embeddings, uses rolling chat history.
    Mode 'query' will not use LLM unless there are relevant sources from vectorDB & does not recall chat history.
    Mode 'automatic' will use tool-calling if the provider supports native tool calling.

    Args:
        slug: Workspace slug
        message: Message to send
        mode: 'chat' (context + history), 'query' (documents only), or 'automatic' (tool-calling)
        sessionId: Optional session ID to partition chats by external ID
        reset: If true, resets the chat session

    Note: Streaming responses are planned for future API support. Currently returns
    the complete response after generation. For real-time streaming, the AnythingLLM
    API must support Server-Sent Events (SSE) for chat responses.
    """
    try:
        client = get_api_client()
        result = await client.chat(
            workspace_slug=slug,
            message=message,
            mode=mode,
            session_id=sessionId,
            reset=reset,
        )
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_chat_history",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_chat_history(
    slug: str,
    apiSessionId: Optional[str] = None,
    limit: Optional[int] = None,
    orderBy: Optional[str] = None,
) -> str:
    """Get the chat history for a workspace.

    Args:
        slug: Workspace slug
        apiSessionId: Optional API session ID to filter by
        limit: Optional number of chat messages to return (default: 100)
        orderBy: Optional order of chat messages ('asc' or 'desc')
    """
    try:
        client = get_api_client()
        result = await client.get_chat_history(
            workspace_slug=slug,
            session_id=apiSessionId,
            limit=limit,
            order_by=orderBy,
        )
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Workspace Threads
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_create_thread",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def create_thread(slug: str, name: Optional[str] = None) -> str:
    """Create a new chat thread within a workspace.

    Args:
        slug: Workspace slug
        name: Optional thread name
    """
    try:
        client = get_api_client()
        result = await client.create_thread(workspace_slug=slug, name=name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_delete_thread",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def delete_thread(slug: str, thread_slug: str) -> str:
    """Delete a chat thread from a workspace.

    Args:
        slug: Workspace slug
        thread_slug: Thread slug to delete
    """
    try:
        client = get_api_client()
        await client.delete_thread(workspace_slug=slug, thread_slug=thread_slug)
        return f"Thread '{thread_slug}' deleted from workspace '{slug}'."
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_update_thread",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def update_thread(slug: str, thread_slug: str, name: str) -> str:
    """Update the name of an existing chat thread.

    Args:
        slug: Workspace slug.
        thread_slug: Thread slug to update.
        name: New name for the thread.
    """
    try:
        client = get_api_client()
        result = await client.update_thread(workspace_slug=slug, thread_slug=thread_slug, name=name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_chat_in_thread",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def chat_in_thread(
    slug: str,
    thread_slug: str,
    message: str,
    mode: ChatMode = ChatMode.CHAT,
) -> str:
    """Send a message within a specific workspace thread.

    Args:
        slug: Workspace slug
        thread_slug: Thread slug
        message: Message to send
        mode: 'chat' or 'query'
    """
    try:
        client = get_api_client()
        result = await client.chat_in_thread(
            workspace_slug=slug,
            thread_slug=thread_slug,
            message=message,
            mode=mode,
        )
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_thread",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_thread(slug: str, thread_slug: str) -> str:
    """Get a specific thread by slug in a workspace.

    Args:
        slug: Workspace slug
        thread_slug: Thread slug
    """
    try:
        client = get_api_client()
        result = await client.get_thread(workspace_slug=slug, thread_slug=thread_slug)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_thread_chats",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_thread_chats(slug: str, thread_slug: str) -> str:
    """Get chat history for a specific thread in a workspace.

    Args:
        slug: Workspace slug
        thread_slug: Thread slug
    """
    try:
        client = get_api_client()
        result = await client.get_thread_chats(workspace_slug=slug, thread_slug=thread_slug)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Documents
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_list_documents",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def list_documents() -> str:
    """List all uploaded documents across all folders."""
    try:
        client = get_api_client()
        result = await client.list_documents()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_accepted_file_types",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_accepted_file_types() -> str:
    """Get the list of file types that AnythingLLM accepts for upload."""
    try:
        client = get_api_client()
        result = await client.get_accepted_file_types()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_upload_link",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def upload_link(link: str) -> str:
    """Scrape a web page and upload it as a document to AnythingLLM.

    Args:
        link: URL to scrape (must start with http:// or https://)
    """
    try:
        if not link.startswith(("http://", "https://")):
            return "Error: Link must start with http:// or https://"
        client = get_api_client()
        result = await client.upload_link(link)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_upload_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def upload_file(file_path: str) -> str:
    """Upload a file from the local filesystem to AnythingLLM.

    Args:
        file_path: Absolute path to the file on the local filesystem.
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists() or not path.is_file():
            return f"Error: File not found: {file_path}"
        client = get_api_client()
        result = await client.upload_file(file_path)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_upload_raw_text",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def upload_raw_text(text_content: str, title: str) -> str:
    """Upload raw text content as a document to AnythingLLM.

    Args:
        text_content: Raw text content to upload
        title: Document title
    """
    try:
        client = get_api_client()
        result = await client.upload_raw_text(text_content=text_content, title=title)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_upload_file_to_folder",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def upload_file_to_folder(file_path: str, folder_name: str) -> str:
    """Upload a file from the local filesystem into a specific document folder in AnythingLLM.

    Args:
        file_path: Absolute path to the file on the local filesystem.
        folder_name: Target folder name in AnythingLLM (must already exist).
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists() or not path.is_file():
            return f"Error: File not found: {file_path}"
        client = get_api_client()
        result = await client.upload_file(file_path, folder_name=folder_name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_list_documents_in_folder",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def list_documents_in_folder(folder_name: str) -> str:
    """List all documents inside a specific folder.

    Args:
        folder_name: Folder name to list documents from.
    """
    try:
        client = get_api_client()
        result = await client.list_documents_in_folder(folder_name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_document",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_document(doc_name: str) -> str:
    """Get metadata and details for a specific document by its stored name.

    The doc_name is the internal document identifier returned by list_documents
    (e.g. 'custom-documents/myfile.pdf-abc123.json').

    Args:
        doc_name: Document name/path as returned by the documents API.
    """
    try:
        client = get_api_client()
        result = await client.get_document(doc_name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_document_metadata_schema",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_document_metadata_schema() -> str:
    """Get the metadata schema that AnythingLLM uses for documents."""
    try:
        client = get_api_client()
        result = await client.get_document_metadata_schema()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_create_folder",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def create_folder(name: str) -> str:
    """Create a new document folder in AnythingLLM.

    Args:
        name: Name for the new folder.
    """
    try:
        client = get_api_client()
        result = await client.create_folder(name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_remove_folder",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def remove_folder(name: str) -> str:
    """Permanently delete a document folder and all its contents.

    Args:
        name: Folder name to delete.
    """
    try:
        client = get_api_client()
        result = await client.remove_folder(name)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_move_files",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def move_files(files: list[dict[str, str]]) -> str:
    """Move documents between folders.

    Each entry in 'files' must have 'from' and 'to' keys with document paths
    as returned by list_documents (e.g. 'custom-documents/file.pdf-uuid.json').

    Args:
        files: List of move operations, each with 'from' and 'to' path strings.
               Example: [{"from": "custom-documents/a.pdf-uuid.json",
                          "to": "custom-documents/my-folder/a.pdf-uuid.json"}]
    """
    try:
        if not files:
            return "Error: 'files' list cannot be empty."
        for entry in files:
            if "from" not in entry or "to" not in entry:
                return "Error: Each entry in 'files' must have 'from' and 'to' keys."
        client = get_api_client()
        result = await client.move_files(files)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_update_embeddings",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def update_embeddings(
    slug: str,
    adds: Optional[list[str]] = None,
    deletes: Optional[list[str]] = None,
) -> str:
    """Embed or un-embed documents in a workspace (equivalent to "Save and Embed" in the UI).

    This is the final step after uploading documents to AnythingLLM storage.
    Typical workflow:
      1. Upload a file     -> upload_file / upload_file_to_folder / upload_link / upload_raw_text
      2. List documents    -> list_documents / list_documents_in_folder  (get the doc paths)
      3. Embed into workspace -> THIS TOOL with 'adds' containing the doc paths from step 2

    Once embedded, the document's content is vectorized and available for
    RAG queries in the workspace. Removing (via 'deletes') un-embeds the
    document from the workspace but does NOT delete it from storage.

    Args:
        slug: Workspace slug where documents will be embedded.
        adds: Document paths to embed, as returned by list_documents
              (e.g. ['custom-documents/Pine Script/file.pine-abc123.json']).
        deletes: Document paths to un-embed (remove from workspace only,
                 the file remains in AnythingLLM storage).
    """
    try:
        if not adds and not deletes:
            return "Error: Provide at least 'adds' or 'deletes'."
        client = get_api_client()
        result = await client.update_embeddings(workspace_slug=slug, adds=adds, deletes=deletes)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_update_pin",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def update_pin(slug: str, doc_path: str, pinned: bool) -> str:
    """Pin or unpin a document in a workspace.

    Pinned documents are always included in the LLM context for every query,
    regardless of vector similarity results.

    Args:
        slug: Workspace slug.
        doc_path: Document path as returned by list_documents
                  (e.g. 'custom-documents/myfile.pdf-abc123.json').
        pinned: True to pin, False to unpin.
    """
    try:
        client = get_api_client()
        result = await client.update_pin(workspace_slug=slug, doc_path=doc_path, pinned=pinned)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Search
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def search_workspace(
    slug: str,
    query: str,
    top_n: int = 4,
    score_threshold: Optional[float] = None,
) -> str:
    """Search for relevant document chunks within a workspace using vector similarity.

    Args:
        slug: Workspace slug to search in
        query: Search query
        top_n: Number of results (1-20, default 4)
        score_threshold: Similarity score threshold (0-1). Lower = more results.
    """
    try:
        _validate_range("top_n", top_n, 1, 20)
        if score_threshold is not None:
            _validate_range("score_threshold", score_threshold, 0.0, 1.0)
        client = get_api_client()
        result = await client.search_workspace(
            workspace_slug=slug,
            query=query,
            top_n=top_n,
            score_threshold=score_threshold,
        )
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: System
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_get_system_settings",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_system_settings() -> str:
    """Get AnythingLLM system settings (LLM provider, vector DB, embeddings, etc.)."""
    try:
        client = get_api_client()
        result = await client.get_system_settings()
        if isinstance(result, dict):
            settings = result.get("settings", {})
            safe_keys = [
                "LLMProvider", "LLMModel", "VectorDB", "EmbeddingEngine",
                "EmbeddingModelPref", "EmbeddingModelMaxChunkLength",
                "MultiUserMode", "DisableTelemetry", "WhisperProvider",
                "TextToSpeechProvider", "OllamaLLMBasePath", "OllamaLLMModelPref",
            ]
            filtered = {k: settings.get(k) for k in safe_keys if settings.get(k) is not None}
            return _json_response({"settings": filtered})
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_get_vector_count",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def get_vector_count() -> str:
    """Get the total number of vectors stored in the system."""
    try:
        client = get_api_client()
        result = await client.get_vector_count()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_export_chats",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def export_chats() -> str:
    """Export all chat logs from all workspaces."""
    try:
        client = get_api_client()
        result = await client.export_chats()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="anythingllm_remove_documents",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def remove_documents(names: list[str]) -> str:
    """Permanently delete one or more documents from the system.

    This removes the documents from AnythingLLM storage entirely.
    They will also be removed from any workspaces that had them embedded.

    Args:
        names: List of document names/paths as returned by list_documents
               (e.g. ['custom-documents/myfile.pdf-abc123.json']).
    """
    try:
        if not names:
            return "Error: 'names' list cannot be empty."
        client = get_api_client()
        result = await client.remove_documents(names)
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: Embeds (public chat widgets)
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_list_embeds",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def list_embeds() -> str:
    """List all embed configurations (public chat widgets)."""
    try:
        client = get_api_client()
        result = await client.list_embeds()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Tools: OpenAI-compatible endpoints
# ──────────────────────────────────────────────

@mcp.tool(
    name="anythingllm_list_models",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def list_models() -> str:
    """List available models via the OpenAI-compatible endpoint."""
    try:
        client = get_api_client()
        result = await client.list_models()
        return _json_response(result)
    except Exception as e:
        return _handle_error(e)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    """Run MCP server using stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
