#!/usr/bin/env python3
"""
Data Model Layer for AnythingLLM MCP Server.

Provides Pydantic models for API request/response payloads to ensure
type safety and data validation across the codebase.

Architecture:
- Workspace models: creation, update, and response types
- Chat models: message request/response with context
- Thread models: conversation thread management
- Document models: file upload and embedding
- Settings models: system configuration types
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ChatMode(str, Enum):
    """Chat modes for workspace interactions.
    
    From the official AnythingLLM API:
    - chat: Uses LLM general knowledge w/custom embeddings, uses rolling chat history.
    - query: Will not use LLM unless there are relevant sources from vectorDB & 
             does not recall chat history.
    - automatic: Will use tool-calling if the provider supports native tool calling.
    """
    CHAT = "chat"
    QUERY = "query"
    AUTOMATIC = "automatic"


class VectorSearchMode(str, Enum):
    """Vector search modes supported by AnythingLLM."""
    BM25 = "bm25"
    SIMILARITY = "similarity"
    HYBRID = "hybrid"


class MessageRole(str, Enum):
    """Role of a message sender in chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ──────────────────────────────────────────────
# Base Models
# ──────────────────────────────────────────────

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(populate_by_name=True)
    
    success: bool = True
    message: Optional[str] = None
    error: Optional[str] = None


class PaginationInfo(BaseModel):
    """Pagination metadata for list responses."""
    model_config = ConfigDict(populate_by_name=True)
    
    page: int = 1
    pageSize: int = 50
    total: Optional[int] = None
    totalPages: Optional[int] = None


# ──────────────────────────────────────────────
# Workspace Models
# ──────────────────────────────────────────────

class WorkspaceBase(BaseModel):
    """Base model for workspace data."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., min_length=1, max_length=255, description="Workspace name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")


class WorkspaceCreate(WorkspaceBase):
    """Request model for creating a workspace."""
    
    @field_validator('name')
    @classmethod
    def name_alphanumeric(cls, v: str) -> str:
        """Ensure name is safe for slug generation."""
        if not v.strip():
            raise ValueError("Workspace name cannot be empty or whitespace")
        return v.strip()


class WorkspaceUpdate(BaseModel):
    """Request model for updating a workspace."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    openAiTemp: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM temperature")
    openAiHistory: Optional[int] = Field(None, ge=0, le=100, description="Chat history length")
    openAiPrompt: Optional[str] = Field(None, max_length=5000, description="System prompt override")
    similarityThreshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")
    topN: Optional[int] = Field(None, ge=1, le=20, description="Top N results for context")
    chatMode: Optional[ChatMode] = None
    vectorSearchMode: Optional[VectorSearchMode] = None


class WorkspaceSettings(BaseModel):
    """Workspace-specific settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    openAiTemp: float = 0.7
    openAiHistory: int = 40
    openAiPrompt: Optional[str] = None
    similarityThreshold: float = 0.0
    topN: int = 4
    chatMode: ChatMode = ChatMode.CHAT
    vectorSearchMode: VectorSearchMode = VectorSearchMode.SIMILARITY


class ThreadSummary(BaseModel):
    """Summary of a thread (embedded in workspace response)."""
    model_config = ConfigDict(populate_by_name=True)
    
    slug: str
    name: Optional[str] = None
    updatedAt: Optional[str] = None


class WorkspaceResponse(BaseModel):
    """Response model for workspace data."""
    model_config = ConfigDict(populate_by_name=True)
    
    id: Optional[str] = None
    name: str
    slug: str
    description: Optional[str] = None
    chatMode: ChatMode = ChatMode.CHAT
    vectorSearchMode: VectorSearchMode = VectorSearchMode.SIMILARITY
    openAiTemp: float = 0.7
    openAiHistory: int = 40
    openAiPrompt: Optional[str] = None
    similarityThreshold: float = 0.0
    topN: int = 4
    threads: list[ThreadSummary] = []
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class WorkspaceListResponse(BaseModel):
    """Response model for listing workspaces."""
    model_config = ConfigDict(populate_by_name=True)
    
    workspaces: list[WorkspaceResponse]
    total: int


class WorkspaceSummary(BaseModel):
    """Summary model for workspace listings."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str
    slug: str
    chatMode: ChatMode = ChatMode.CHAT
    vectorSearchMode: VectorSearchMode = VectorSearchMode.SIMILARITY
    threads: int = 0
    createdAt: Optional[str] = None


# ──────────────────────────────────────────────
# Chat Models
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Single chat message in a conversation."""
    model_config = ConfigDict(populate_by_name=True)
    
    role: MessageRole
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for sending a chat message."""
    model_config = ConfigDict(populate_by_name=True)
    
    message: str = Field(..., min_length=1, description="User message")
    mode: ChatMode = ChatMode.CHAT
    sessionId: Optional[str] = Field(None, description="Session ID for partitioning chats")
    reset: bool = Field(False, description="Whether to reset the chat session")


class Citation(BaseModel):
    """Citation from a document in chat response."""
    model_config = ConfigDict(populate_by_name=True)
    
    source: str
    location: Optional[dict[str, Any]] = None
    text: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat message."""
    model_config = ConfigDict(populate_by_name=True)
    
    response: Optional[str] = None
    sources: list[Citation] = []
    error: Optional[str] = None
    # Session info
    sessionId: Optional[str] = None
    threadSlug: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    model_config = ConfigDict(populate_by_name=True)
    
    chats: list[ChatMessage] = []
    page: int = 1
    pageSize: int = 100


# ──────────────────────────────────────────────
# Thread Models
# ──────────────────────────────────────────────

class ThreadCreate(BaseModel):
    """Request model for creating a thread."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = Field(None, max_length=255, description="Thread name")


class ThreadUpdate(BaseModel):
    """Request model for updating a thread."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., min_length=1, max_length=255, description="New thread name")


class ThreadResponse(BaseModel):
    """Response model for thread data."""
    model_config = ConfigDict(populate_by_name=True)
    
    slug: str
    name: Optional[str] = None
    workspaceSlug: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    chats: list[ChatMessage] = []


class ThreadListResponse(BaseModel):
    """Response model for listing threads."""
    model_config = ConfigDict(populate_by_name=True)
    
    threads: list[ThreadResponse]
    total: int


# ──────────────────────────────────────────────
# Document Models
# ──────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    model_config = ConfigDict(populate_by_name=True)
    
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    file_type: Optional[str] = None
    size: Optional[int] = None


class DocumentResponse(BaseModel):
    """Response model for document data."""
    model_config = ConfigDict(populate_by_name=True)
    
    id: Optional[str] = None
    name: str
    location: str  # Path/URL where document is stored
    type: Optional[str] = None
    size: Optional[int] = None
    tokens: Optional[int] = None
    metadata: Optional[DocumentMetadata] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    model_config = ConfigDict(populate_by_name=True)
    
    documents: list[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseResponse):
    """Response model for document upload."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    processingProgress: Optional[float] = None


class LinkUploadRequest(BaseModel):
    """Request model for uploading a link."""
    model_config = ConfigDict(populate_by_name=True)
    
    link: str = Field(..., description="URL to scrape and upload")
    
    @field_validator('link')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure link is a valid HTTP(S) URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Link must start with http:// or https://")
        return v


class RawTextUploadRequest(BaseModel):
    """Request model for uploading raw text."""
    model_config = ConfigDict(populate_by_name=True)
    
    textContent: str = Field(..., min_length=1, description="Raw text content")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")


class FileMoveOperation(BaseModel):
    """Model for a single file move operation."""
    model_config = ConfigDict(populate_by_name=True)
    
    from_path: str = Field(..., alias="from", description="Source document path")
    to_path: str = Field(..., alias="to", description="Destination path")
    
    @field_validator('from_path', 'to_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Ensure paths are not empty."""
        if not v.strip():
            raise ValueError("File paths cannot be empty")
        return v


class BatchMoveRequest(BaseModel):
    """Request model for moving multiple files."""
    model_config = ConfigDict(populate_by_name=True, populate_by_alias=True)
    
    files: list[FileMoveOperation] = Field(..., min_length=1, description="List of move operations")


class FolderCreateRequest(BaseModel):
    """Request model for creating a folder."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., min_length=1, max_length=255, description="Folder name")


class FolderDeleteRequest(BaseModel):
    """Request model for deleting a folder."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., min_length=1, max_length=255, description="Folder name to delete")


class FolderResponse(BaseResponse):
    """Response model for folder operations."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = None


# ──────────────────────────────────────────────
# Embedding Models
# ──────────────────────────────────────────────

class EmbeddingUpdateRequest(BaseModel):
    """Request model for updating workspace embeddings."""
    model_config = ConfigDict(populate_by_name=True)
    
    adds: Optional[list[str]] = Field(None, description="Document paths to embed")
    deletes: Optional[list[str]] = Field(None, description="Document paths to un-embed")


class EmbeddingUpdateResponse(BaseResponse):
    """Response model for embedding update."""
    model_config = ConfigDict(populate_by_name=True)
    
    workspaceSlug: Optional[str] = None
    embedded: int = 0
    removed: int = 0


class PinUpdateRequest(BaseModel):
    """Request model for updating document pin status."""
    model_config = ConfigDict(populate_by_name=True)
    
    docPath: str = Field(..., description="Document path")
    pinStatus: bool = Field(..., description="Whether to pin (true) or unpin (false)")


class PinUpdateResponse(BaseResponse):
    """Response model for pin update."""
    model_config = ConfigDict(populate_by_name=True)
    
    docPath: Optional[str] = None
    pinned: bool = False


# ──────────────────────────────────────────────
# Search Models
# ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request model for workspace search."""
    model_config = ConfigDict(populate_by_name=True)
    
    query: str = Field(..., min_length=1, description="Search query")
    topN: int = Field(4, ge=1, le=20, description="Number of results")
    scoreThreshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResult(BaseModel):
    """Single search result."""
    model_config = ConfigDict(populate_by_name=True)
    
    content: str
    score: float
    source: str
    metadata: Optional[dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response model for search results."""
    model_config = ConfigDict(populate_by_name=True)
    
    results: list[SearchResult] = []
    query: Optional[str] = None
    total: int = 0


# ──────────────────────────────────────────────
# System/Settings Models
# ──────────────────────────────────────────────

class LLMSettings(BaseModel):
    """LLM provider settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    provider: Optional[str] = None
    model: Optional[str] = None
    basePath: Optional[str] = None
    temperature: Optional[float] = None
    maxTokens: Optional[int] = None


class VectorDBSettings(BaseModel):
    """Vector database settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    provider: Optional[str] = None
    name: Optional[str] = None


class EmbeddingSettings(BaseModel):
    """Embedding model settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    provider: Optional[str] = None
    model: Optional[str] = None
    maxChunkLength: Optional[int] = None


class SystemSettings(BaseModel):
    """AnythingLLM system settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    llmProvider: Optional[str] = None
    llmModel: Optional[str] = None
    vectorDB: Optional[str] = None
    embeddingEngine: Optional[str] = None
    embeddingModelPref: Optional[str] = None
    embeddingModelMaxChunkLength: Optional[int] = None
    multiUserMode: bool = False
    disableTelemetry: bool = False
    whisperProvider: Optional[str] = None
    textToSpeechProvider: Optional[str] = None


class SystemSettingsResponse(BaseModel):
    """Response model for system settings."""
    model_config = ConfigDict(populate_by_name=True)
    
    settings: SystemSettings


class VectorCountResponse(BaseModel):
    """Response model for vector count."""
    model_config = ConfigDict(populate_by_name=True)
    
    count: int
    vectorDb: Optional[str] = None


class AcceptFileTypesResponse(BaseModel):
    """Response model for accepted file types."""
    model_config = ConfigDict(populate_by_name=True)
    
    accepted_file_types: list[str] = []
    extensions: list[str] = []


# ──────────────────────────────────────────────
# Embed (Public Widget) Models
# ──────────────────────────────────────────────

class EmbedConfig(BaseModel):
    """Configuration for public chat widget."""
    model_config = ConfigDict(populate_by_name=True)
    
    id: Optional[str] = None
    workspaceSlug: Optional[str] = None
    name: Optional[str] = None
    enabled: bool = True
    embedSettings: Optional[dict[str, Any]] = None


class EmbedListResponse(BaseModel):
    """Response model for listing embeds."""
    model_config = ConfigDict(populate_by_name=True)
    
    embeds: list[EmbedConfig] = []


# ──────────────────────────────────────────────
# OpenAI-Compatible Models
# ──────────────────────────────────────────────

class OpenAIModel(BaseModel):
    """OpenAI-compatible model definition."""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None


class OpenAIModelsResponse(BaseModel):
    """Response model for OpenAI models list."""
    model_config = ConfigDict(populate_by_name=True)
    
    object: str = "list"
    data: list[OpenAIModel] = []


# ──────────────────────────────────────────────
# Auth Models
# ──────────────────────────────────────────────

class AuthResponse(BaseModel):
    """Response model for authentication check."""
    model_config = ConfigDict(populate_by_name=True)
    
    authenticated: bool = False
    user: Optional[dict[str, Any]] = None


# ──────────────────────────────────────────────
# Batch/Delete Operations
# ──────────────────────────────────────────────

class BatchDeleteRequest(BaseModel):
    """Request model for batch delete operations."""
    model_config = ConfigDict(populate_by_name=True)
    
    names: list[str] = Field(..., min_length=1, description="Names/paths to delete")


class BatchDeleteResponse(BaseResponse):
    """Response model for batch delete."""
    model_config = ConfigDict(populate_by_name=True)
    
    deleted: int = 0
    failed: list[str] = []
