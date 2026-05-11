import asyncio
import re

import httpx
import respx
from respx.patterns import M

import anythingllm_mcp as server
from config import get_config, set_config, Config


def test_check_auth_requires_api_key() -> None:
    """Test that check_auth returns error when API key is not set."""
    # Save original config
    original_config = get_config()
    try:
        # Create a new config with empty API key
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="",
        )
        set_config(test_config)
        result = asyncio.run(server.check_auth())
        assert "ANYTHINGLLM_API_KEY is not set" in result
    finally:
        # Restore original config
        set_config(original_config)


def test_check_auth_success_json_response() -> None:
    """Test successful auth check with valid API key."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(re.compile(r"http://localhost:3001/api/v1/auth")).respond(
                status_code=200,
                json={"ok": True},
            )
            result = asyncio.run(server.check_auth())

        assert '"ok": true' in result
    finally:
        set_config(original_config)


def test_search_workspace_rejects_invalid_top_n() -> None:
    """Test that search_workspace rejects invalid top_n values."""
    result = asyncio.run(server.search_workspace("demo", "query", top_n=0))
    assert "top_n must be between 1 and 20" in result


def test_handle_error_timeout_message() -> None:
    """Test timeout error message formatting."""
    message = server._handle_error(httpx.TimeoutException("timeout"))
    assert "Request timed out" in message


# ──────────────────────────────────────────────
# Tests for new tools
# ──────────────────────────────────────────────


def test_create_folder_success() -> None:
    """Test successful folder creation."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.post(re.compile(r"http://localhost:3001/api/v1/document/create-folder")).respond(
                status_code=200,
                json={"success": True, "message": "Folder created"},
            )
            result = asyncio.run(server.create_folder("my-folder"))
        assert '"success": true' in result
    finally:
        set_config(original_config)


def test_remove_folder_success() -> None:
    """Test successful folder removal."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.request("DELETE", re.compile(r"http://localhost:3001/api/v1/document/remove-folder")).respond(
                status_code=200,
                json={"success": True},
            )
            result = asyncio.run(server.remove_folder("my-folder"))
        assert '"success": true' in result
    finally:
        set_config(original_config)


def test_list_documents_in_folder_success() -> None:
    """Test listing documents in a folder."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.get(re.compile(r"http://localhost:3001/api/v1/documents/folder/pine-scripts")).respond(
                status_code=200,
                json={"localFiles": {"items": []}},
            )
            result = asyncio.run(server.list_documents_in_folder("pine-scripts"))
        assert '"localFiles"' in result
    finally:
        set_config(original_config)


def test_move_files_rejects_empty_list() -> None:
    """Test that move_files rejects empty file list."""
    result = asyncio.run(server.move_files([]))
    assert "'files' list cannot be empty" in result


def test_move_files_rejects_missing_keys() -> None:
    """Test that move_files rejects entries missing required keys."""
    result = asyncio.run(server.move_files([{"from": "a"}]))
    assert "'from' and 'to'" in result


def test_update_pin_success() -> None:
    """Test successful pin update."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.post(re.compile(r"http://localhost:3001/api/v1/workspace/demo/update-pin")).respond(
                status_code=200,
                json={"success": True},
            )
            result = asyncio.run(server.update_pin("demo", "custom-documents/test.json", True))
        assert '"success": true' in result
    finally:
        set_config(original_config)


def test_update_thread_success() -> None:
    """Test successful thread update."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.post(
                re.compile(r"http://localhost:3001/api/v1/workspace/demo/thread/t-slug/update")
            ).respond(
                status_code=200,
                json={"thread": {"slug": "t-slug", "name": "new-name"}},
            )
            result = asyncio.run(server.update_thread("demo", "t-slug", "new-name"))
        assert '"new-name"' in result
    finally:
        set_config(original_config)


def test_remove_documents_rejects_empty_list() -> None:
    """Test that remove_documents rejects empty names list."""
    result = asyncio.run(server.remove_documents([]))
    assert "'names' list cannot be empty" in result


def test_remove_documents_success() -> None:
    """Test successful document removal."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.request(
                "DELETE", re.compile(r"http://localhost:3001/api/v1/system/remove-documents")
            ).respond(
                status_code=200,
                json={"success": True},
            )
            result = asyncio.run(server.remove_documents(["custom-documents/test.json"]))
        assert '"success": true' in result
    finally:
        set_config(original_config)


def test_get_document_metadata_schema_success() -> None:
    """Test getting document metadata schema."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.get(re.compile(r"http://localhost:3001/api/v1/document/metadata-schema")).respond(
                status_code=200,
                json={"schema": {"title": "string"}},
            )
            result = asyncio.run(server.get_document_metadata_schema())
        assert '"schema"' in result
    finally:
        set_config(original_config)


def test_get_thread_success() -> None:
    """Test getting a thread."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.get(re.compile(r"http://localhost:3001/api/v1/workspace/demo/thread/t-slug")).respond(
                status_code=200,
                json={"thread": {"slug": "t-slug", "name": "test-thread"}},
            )
            result = asyncio.run(server.get_thread("demo", "t-slug"))
        assert '"t-slug"' in result
    finally:
        set_config(original_config)


def test_chat_with_workspace_accepts_automatic_mode() -> None:
    """Test chat with automatic mode."""
    original_config = get_config()
    try:
        test_config = Config(
            api_base_url="http://localhost:3001",
            api_key="test-token",
        )
        set_config(test_config)
        
        with respx.mock(assert_all_called=False) as mock:
            mock.post(re.compile(r"http://localhost:3001/api/v1/workspace/demo/chat")).respond(
                status_code=200,
                json={"textResponse": "hello"},
            )
            result = asyncio.run(
                server.chat_with_workspace("demo", "hi", mode=server.ChatMode.AUTOMATIC)
            )
        assert '"hello"' in result
    finally:
        set_config(original_config)
