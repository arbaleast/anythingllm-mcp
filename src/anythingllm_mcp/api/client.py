#!/usr/bin/env python3
"""
API Service Layer for AnythingLLM MCP Server.

Provides a robust API client with:
- Connection pooling via persistent httpx.AsyncClient
- Automatic retry with exponential backoff for transient errors
- Comprehensive error handling
- Type-safe request/response handling

Architecture:
- This module is the Service layer that handles all HTTP communication
- Uses models.py for request/response validation
- Is used by server.py tools (Controller layer)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any, AsyncIterator

import httpx

from ..config import get_config, Config
from ..models import (
    ChatMode,
    WorkspaceCreate,
    WorkspaceUpdate,
    ChatRequest,
    ThreadCreate,
    ThreadUpdate,
    EmbeddingUpdateRequest,
    SearchRequest,
    LinkUploadRequest,
    RawTextUploadRequest,
    FolderCreateRequest,
    FolderDeleteRequest,
    PinUpdateRequest,
    BatchDeleteRequest,
    FileMoveOperation,
)


logger = logging.getLogger("anythingllm_mcp.api")


# ──────────────────────────────────────────────
# Error Types
# ──────────────────────────────────────────────

class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class AuthenticationError(APIError):
    """Authentication or authorization failed."""
    pass


class NotFoundError(APIError):
    """Resource not found."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""
    pass


class ServerError(APIError):
    """Server-side error (5xx)."""
    pass


class ValidationError(APIError):
    """Request validation failed."""
    pass


# ──────────────────────────────────────────────
# HTTP Status Code Mapping
# ──────────────────────────────────────────────

def _get_error_type(status_code: int) -> type[APIError]:
    """Map HTTP status code to appropriate error type."""
    if status_code == 401 or status_code == 403:
        return AuthenticationError
    elif status_code == 404:
        return NotFoundError
    elif status_code == 422:
        return ValidationError
    elif status_code == 429:
        return RateLimitError
    elif status_code >= 500:
        return ServerError
    return APIError


def _is_retryable_status(status_code: int) -> bool:
    """Check if HTTP status code should trigger a retry."""
    return status_code in (429, 500, 502, 503, 504)


def _get_user_message(status_code: int) -> str:
    """Get user-friendly error message for status code."""
    messages = {
        400: "Invalid request. Check your parameters.",
        401: "Authentication failed. Check your ANYTHINGLLM_API_KEY.",
        403: "Permission denied. Your API key may lack required permissions.",
        404: "Resource not found. Check the slug or ID.",
        409: "Conflict. The resource may already exist.",
        422: "Validation error. Check your request parameters.",
        429: "Rate limit exceeded. Please wait a moment before retrying.",
        500: "Internal server error in AnythingLLM. Check LLM provider connectivity.",
        502: "Bad gateway. AnythingLLM server may be restarting.",
        503: "Service unavailable. AnythingLLM may be overloaded.",
        504: "Gateway timeout. AnythingLLM is not responding.",
    }
    return messages.get(status_code, f"API error (HTTP {status_code}).")


# ──────────────────────────────────────────────
# API Client
# ──────────────────────────────────────────────

class AnythingLLMAPIClient:
    """
    Async API client for AnythingLLM with retry logic and connection pooling.
    
    Features:
    - Persistent connection via shared httpx.AsyncClient
    - Automatic retry with exponential backoff for transient errors
    - Comprehensive error handling with typed exceptions
    - Type-safe methods for all API endpoints
    
    Usage:
        async with AnythingLLMAPIClient() as client:
            workspaces = await client.list_workspaces()
            await client.create_workspace(name="My Workspace")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
    ):
        """
        Initialize API client.
        
        Args:
            config: Configuration object. If None, loads from environment.
            max_attempts: Maximum retry attempts for transient errors.
            base_delay: Base delay for exponential backoff (seconds).
            max_delay: Maximum delay between retries (seconds).
        """
        self._config = config or get_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        
        # Validate config
        if not self._config.is_api_key_configured:
            raise RuntimeError(
                "ANYTHINGLLM_API_KEY is not configured. "
                "Set it in environment or .env file."
            )
    
    @property
    def _api_key(self) -> str:
        """Get API key from config."""
        return self._config.api_key
    
    @property
    def _base_url(self) -> str:
        """Get API base URL."""
        return self._config.api_base_url.rstrip("/")
    
    @property
    def _default_timeout(self) -> float:
        """Get default request timeout."""
        return self._config.default_timeout
    
    @property
    def _upload_timeout(self) -> float:
        """Get upload request timeout."""
        return self._config.upload_timeout
    
    def _get_headers(self) -> dict[str, str]:
        """Get common request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def _get_url(self, endpoint: str) -> str:
        """Build full URL for endpoint."""
        return f"{self._base_url}/api/v1{endpoint}"
    
    # ──────────────────────────────────────────────
    # Connection Management
    # ──────────────────────────────────────────────
    
    async def __aenter__(self) -> "AnythingLLMAPIClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection to the API (initialize client)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._default_timeout),
                headers=self._get_headers(),
            )
            logger.info(f"Connected to AnythingLLM API at {self._base_url}")
    
    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Closed AnythingLLM API connection")
    
    async def _ensure_connected(self) -> httpx.AsyncClient:
        """Ensure client is connected, creating if necessary."""
        if self._client is None:
            await self.connect()
        return self._client
    
    # ──────────────────────────────────────────────
    # Retry Logic
    # ──────────────────────────────────────────────
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = self._base_delay * (2 ** (attempt - 1))
        return min(delay, self._max_delay)
    
    async def _sleep_with_backoff(self, attempt: int) -> None:
        """Sleep for exponential backoff delay."""
        delay = self._calculate_delay(attempt)
        logger.debug(f"Retrying in {delay:.1f}s (attempt {attempt})")
        await asyncio.sleep(delay)
    
    # ──────────────────────────────────────────────
    # Core Request Methods
    # ──────────────────────────────────────────────
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retry_on_status: bool = True,
    ) -> Any:
        """
        Make HTTP request with automatic retry for transient errors.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/workspaces")
            json: JSON request body
            params: Query parameters
            timeout: Request timeout override
            retry_on_status: Whether to retry on transient status codes
        
        Returns:
            Parsed JSON response
            
        Raises:
            AuthenticationError: For 401/403 responses
            NotFoundError: For 404 responses
            RateLimitError: For 429 responses
            ServerError: For 5xx responses
            APIError: For other errors
        """
        client = await self._ensure_connected()
        url = self._get_url(endpoint)
        timeout = timeout or self._default_timeout
        headers = self._get_headers()
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params,
                    timeout=timeout,
                )
                
                # Check for retryable status
                if retry_on_status and _is_retryable_status(response.status_code):
                    if attempt < self._max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{self._max_attempts} received HTTP {response.status_code}. "
                            f"Retrying..."
                        )
                        await self._sleep_with_backoff(attempt)
                        continue
                    # Final attempt failed with retryable status
                    error_type = _get_error_type(response.status_code)
                    details = self._parse_response_body(response)
                    raise error_type(
                        _get_user_message(response.status_code),
                        status_code=response.status_code,
                        details=details,
                    )
                
                # Handle success
                if response.is_success:
                    return self._parse_response_body(response)
                
                # Handle non-retryable error
                error_type = _get_error_type(response.status_code)
                details = self._parse_response_body(response)
                raise error_type(
                    _get_user_message(response.status_code),
                    status_code=response.status_code,
                    details=details,
                )
                
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                last_error = e
                if attempt < self._max_attempts:
                    logger.warning(
                        f"Attempt {attempt}/{self._max_attempts} failed with {type(e).__name__}. "
                        f"Retrying..."
                    )
                    await self._sleep_with_backoff(attempt)
                else:
                    raise APIError(
                        f"Request failed after {self._max_attempts} attempts: {e}",
                        details=str(e),
                    )
        
        # Should not reach here, but handle gracefully
        if last_error:
            raise APIError(f"Request failed: {last_error}")
        raise APIError("Request failed after all retries")
    
    def _parse_response_body(self, response: httpx.Response) -> Any:
        """Parse response body based on content type."""
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text
    
    # ──────────────────────────────────────────────
    # HTTP Method Shortcuts
    # ──────────────────────────────────────────────
    
    async def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send GET request."""
        return await self._request("GET", endpoint, params=params, timeout=timeout)
    
    async def post(
        self,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send POST request."""
        return await self._request("POST", endpoint, json=json, timeout=timeout)
    
    async def put(
        self,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send PUT request."""
        return await self._request("PUT", endpoint, json=json, timeout=timeout)
    
    async def patch(
        self,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send PATCH request."""
        return await self._request("PATCH", endpoint, json=json, timeout=timeout)
    
    async def delete(
        self,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send DELETE request."""
        return await self._request("DELETE", endpoint, json=json, timeout=timeout)
    
    # ──────────────────────────────────────────────
    # Authentication
    # ──────────────────────────────────────────────
    
    async def check_auth(self) -> dict[str, Any]:
        """Verify API authentication."""
        return await self.get("/auth")
    
    # ──────────────────────────────────────────────
    # Workspace Operations
    # ──────────────────────────────────────────────
    
    async def list_workspaces(self) -> list[dict[str, Any]]:
        """List all workspaces."""
        result = await self.get("/workspaces")
        if isinstance(result, dict):
            return result.get("workspaces", [])
        return result
    
    async def get_workspace(self, slug: str) -> dict[str, Any]:
        """Get workspace details by slug."""
        return await self.get(f"/workspace/{slug}")
    
    async def create_workspace(
        self,
        name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create a new workspace."""
        request = WorkspaceCreate(name=name, description=description)
        return await self.post("/workspace/new", json=request.model_dump(exclude_none=True))
    
    async def update_workspace(
        self,
        slug: str,
        updates: WorkspaceUpdate,
    ) -> dict[str, Any]:
        """Update workspace settings."""
        # Convert enums to values
        data = updates.model_dump(exclude_none=True, mode="json")
        return await self.post(f"/workspace/{slug}/update", json=data)
    
    async def delete_workspace(self, slug: str) -> None:
        """Delete a workspace."""
        await self.delete(f"/workspace/{slug}")
    
    # ──────────────────────────────────────────────
    # Chat Operations
    # ──────────────────────────────────────────────
    
    async def chat(
        self,
        workspace_slug: str,
        message: str,
        mode: ChatMode = ChatMode.CHAT,
        session_id: Optional[str] = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Send chat message to workspace."""
        body: dict[str, Any] = {
            "message": message,
            "mode": mode.value,
        }
        if session_id is not None:
            body["sessionId"] = session_id
        if reset:
            body["reset"] = reset
        return await self.post(f"/workspace/{workspace_slug}/chat", json=body)
    
    async def get_chat_history(
        self,
        workspace_slug: str,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get chat history for workspace."""
        params: dict[str, Any] = {}
        if session_id is not None:
            params["apiSessionId"] = session_id
        if limit is not None:
            params["limit"] = limit
        if order_by is not None:
            params["orderBy"] = order_by
        return await self.get(f"/workspace/{workspace_slug}/chats", params=params if params else None)
    
    # ──────────────────────────────────────────────
    # Thread Operations
    # ──────────────────────────────────────────────
    
    async def create_thread(
        self,
        workspace_slug: str,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new thread in workspace."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        return await self.post(f"/workspace/{workspace_slug}/thread/new", json=body if body else None)
    
    async def get_thread(
        self,
        workspace_slug: str,
        thread_slug: str,
    ) -> dict[str, Any]:
        """Get thread details."""
        return await self.get(f"/workspace/{workspace_slug}/thread/{thread_slug}")
    
    async def update_thread(
        self,
        workspace_slug: str,
        thread_slug: str,
        name: str,
    ) -> dict[str, Any]:
        """Update thread name."""
        return await self.post(
            f"/workspace/{workspace_slug}/thread/{thread_slug}/update",
            json={"name": name},
        )
    
    async def delete_thread(
        self,
        workspace_slug: str,
        thread_slug: str,
    ) -> None:
        """Delete a thread."""
        await self.delete(f"/workspace/{workspace_slug}/thread/{thread_slug}")
    
    async def chat_in_thread(
        self,
        workspace_slug: str,
        thread_slug: str,
        message: str,
        mode: ChatMode = ChatMode.CHAT,
    ) -> dict[str, Any]:
        """Send chat message in thread."""
        return await self.post(
            f"/workspace/{workspace_slug}/thread/{thread_slug}/chat",
            json={"message": message, "mode": mode.value},
        )
    
    async def get_thread_chats(
        self,
        workspace_slug: str,
        thread_slug: str,
    ) -> dict[str, Any]:
        """Get chat history for a thread."""
        return await self.get(f"/workspace/{workspace_slug}/thread/{thread_slug}/chats")
    
    # ──────────────────────────────────────────────
    # Document Operations
    # ──────────────────────────────────────────────
    
    async def list_documents(self) -> dict[str, Any]:
        """List all documents."""
        return await self.get("/documents")
    
    async def list_documents_in_folder(self, folder_name: str) -> dict[str, Any]:
        """List documents in a folder."""
        return await self.get(f"/documents/folder/{folder_name}")
    
    async def get_document(self, doc_name: str) -> dict[str, Any]:
        """Get document details."""
        return await self.get(f"/document/{doc_name}")
    
    async def get_document_metadata_schema(self) -> dict[str, Any]:
        """Get document metadata schema."""
        return await self.get("/document/metadata-schema")
    
    async def get_accepted_file_types(self) -> dict[str, Any]:
        """Get accepted file types for upload."""
        return await self.get("/document/accepted-file-types")
    
    async def upload_link(self, link: str) -> dict[str, Any]:
        """Upload content from a URL."""
        request = LinkUploadRequest(link=link)
        return await self.post("/document/upload-link", json=request.model_dump())
    
    async def upload_raw_text(
        self,
        text_content: str,
        title: str,
    ) -> dict[str, Any]:
        """Upload raw text as document."""
        request = RawTextUploadRequest(
            textContent=text_content,
            metadata={"title": title} if title else None,
        )
        return await self.post("/document/raw-text", json=request.model_dump(exclude_none=True))
    
    async def create_folder(self, name: str) -> dict[str, Any]:
        """Create a document folder."""
        request = FolderCreateRequest(name=name)
        return await self.post("/document/create-folder", json=request.model_dump())
    
    async def remove_folder(self, name: str) -> dict[str, Any]:
        """Remove a document folder."""
        request = FolderDeleteRequest(name=name)
        return await self.delete("/document/remove-folder", json=request.model_dump())
    
    async def move_files(self, files: list[dict[str, str]]) -> dict[str, Any]:
        """Move documents between folders."""
        return await self.post("/document/move-files", json={"files": files})
    
    # ──────────────────────────────────────────────
    # File Upload (Multipart)
    # ──────────────────────────────────────────────
    
    async def upload_file(
        self,
        file_path: str,
        folder_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Upload a file to AnythingLLM.
        
        Uses multipart form data instead of JSON.
        """
        import pathlib
        
        path = pathlib.Path(file_path)
        if not path.exists() or not path.is_file():
            raise ValidationError(f"File not found: {file_path}")
        
        client = await self._ensure_connected()
        url = self._get_url(
            f"/document/upload/{folder_name}" if folder_name else "/document/upload"
        )
        headers = self._get_headers()
        headers.pop("Content-Type", None)  # Remove Content-Type for multipart
        
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f)}
            response = await client.post(
                url,
                headers=headers,
                files=files,
                timeout=self._upload_timeout,
            )
            response.raise_for_status()
            return self._parse_response_body(response)
    
    # ──────────────────────────────────────────────
    # Embedding Operations
    # ──────────────────────────────────────────────
    
    async def update_embeddings(
        self,
        workspace_slug: str,
        adds: Optional[list[str]] = None,
        deletes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Update workspace embeddings (add or remove documents)."""
        request = EmbeddingUpdateRequest(adds=adds, deletes=deletes)
        return await self.post(
            f"/workspace/{workspace_slug}/update-embeddings",
            json=request.model_dump(exclude_none=True),
        )
    
    async def update_pin(
        self,
        workspace_slug: str,
        doc_path: str,
        pinned: bool,
    ) -> dict[str, Any]:
        """Update document pin status in workspace."""
        request = PinUpdateRequest(docPath=doc_path, pinStatus=pinned)
        return await self.post(
            f"/workspace/{workspace_slug}/update-pin",
            json=request.model_dump(),
        )
    
    # ──────────────────────────────────────────────
    # Search Operations
    # ──────────────────────────────────────────────
    
    async def search_workspace(
        self,
        workspace_slug: str,
        query: str,
        top_n: int = 4,
        score_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Search workspace using vector similarity."""
        body: dict[str, Any] = {"query": query, "topN": top_n}
        if score_threshold is not None:
            body["scoreThreshold"] = score_threshold
        return await self.post(
            f"/workspace/{workspace_slug}/vector-search",
            json=body,
        )
    
    # ──────────────────────────────────────────────
    # System Operations
    # ──────────────────────────────────────────────
    
    async def get_system_settings(self) -> dict[str, Any]:
        """Get system settings."""
        return await self.get("/system")
    
    async def get_vector_count(self) -> dict[str, Any]:
        """Get total vector count."""
        return await self.get("/system/vector-count")
    
    async def export_chats(self) -> dict[str, Any]:
        """Export all chat logs."""
        return await self.get("/system/export-chats")
    
    async def remove_documents(self, names: list[str]) -> dict[str, Any]:
        """Remove documents from system."""
        request = BatchDeleteRequest(names=names)
        return await self.delete("/system/remove-documents", json=request.model_dump())
    
    # ──────────────────────────────────────────────
    # Embed Operations
    # ──────────────────────────────────────────────
    
    async def list_embeds(self) -> dict[str, Any]:
        """List all embed configurations."""
        return await self.get("/embed")
    
    # ──────────────────────────────────────────────
    # OpenAI-Compatible Endpoints
    # ──────────────────────────────────────────────
    
    async def list_models(self) -> dict[str, Any]:
        """List available models via OpenAI-compatible endpoint."""
        return await self.get("/openai/models")


# ──────────────────────────────────────────────
# Global Client Instance
# ──────────────────────────────────────────────

_client: Optional[AnythingLLMAPIClient] = None


def get_api_client() -> AnythingLLMAPIClient:
    """Get the global API client instance (lazy initialization)."""
    global _client
    if _client is None:
        _client = AnythingLLMAPIClient()
    return _client


async def close_api_client() -> None:
    """Close the global API client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


@asynccontextmanager
async def api_client_context() -> AsyncIterator[AnythingLLMAPIClient]:
    """Context manager for API client with automatic cleanup."""
    client = AnythingLLMAPIClient()
    try:
        await client.connect()
        yield client
    finally:
        await client.close()


# ──────────────────────────────────────────────
# Error Handler Utility
# ──────────────────────────────────────────────

def format_api_error(error: Exception) -> str:
    """
    Format API error into user-friendly message.
    
    Args:
        error: Exception from API calls
        
    Returns:
        Formatted error message for user display
    """
    if isinstance(error, APIError):
        details_str = ""
        if error.details:
            import json
            details_str = f"\nDetails: {json.dumps(error.details, indent=2, default=str) if isinstance(error.details, dict) else error.details}"
        return f"Error: {error.message}{details_str}"
    
    if isinstance(error, httpx.TimeoutException):
        return "Error: Request timed out. AnythingLLM may be busy or unreachable."
    
    if isinstance(error, httpx.ConnectError):
        return f"Error: Cannot connect to AnythingLLM. Is it running?"
    
    if isinstance(error, httpx.RequestError):
        return f"Error: Request failed: {error}"
    
    # Unknown error
    logger.exception(f"Unexpected error: {type(error).__name__}: {error}")
    return f"Error: {type(error).__name__}: {error}"
