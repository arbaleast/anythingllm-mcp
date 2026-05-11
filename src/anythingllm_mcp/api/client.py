#!/usr/bin/env python3
"""
统一 API 客户端模块
封装所有与 AnythingLLM API 的交互
"""
import os
from typing import Any, Optional
import httpx

from config import get_config


class AnythingLLMClient:
    """AnythingLLM API 客户端"""
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """初始化 API 客户端
        
        Args:
            api_base_url: API 基础 URL，默认从配置读取
            api_key: API 密钥，默认从配置读取
            timeout: 请求超时时间，默认从配置读取
        """
        config = get_config()
        self.api_base_url = api_base_url or config.api_base_url
        self.api_key = api_key or config.api_key
        self.timeout = timeout or config.default_timeout
        
        self._client: Optional[httpx.Client] = None
    
    def _get_headers(self) -> dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    @property
    def _http_client(self) -> httpx.Client:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.api_base_url,
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._client
    
    def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> "AnythingLLMClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # ========== 工作区操作 ==========
    
    def list_workspaces(self) -> list[dict[str, Any]]:
        """获取所有工作区"""
        response = self._http_client.get("/api/v1/workspaces")
        response.raise_for_status()
        return response.json()
    
    def get_workspace(self, slug: str) -> dict[str, Any]:
        """获取指定工作区"""
        response = self._http_client.get(f"/api/v1/workspaces/{slug}")
        response.raise_for_status()
        return response.json()
    
    def create_workspace(self, name: str, description: str = "") -> dict[str, Any]:
        """创建工作区"""
        response = self._http_client.post(
            "/api/v1/workspaces",
            json={"name": name, "description": description},
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 文档操作 ==========
    
    def list_documents(self, workspace_slug: str) -> list[dict[str, Any]]:
        """获取工作区中的所有文档"""
        response = self._http_client.get(f"/api/v1/workspaces/{workspace_slug}/documents")
        response.raise_for_status()
        return response.json()
    
    def add_document(
        self,
        workspace_slug: str,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """添加文档到工作区"""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {k: v for k, v in kwargs.items() if v is not None}
            response = self._http_client.post(
                f"/api/v1/workspaces/{workspace_slug}/documents",
                files=files,
                data=data,
            )
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, workspace_slug: str, document_id: str) -> dict[str, Any]:
        """删除文档"""
        response = self._http_client.delete(
            f"/api/v1/workspaces/{workspace_slug}/documents/{document_id}"
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 聊天操作 ==========
    
    def chat(
        self,
        workspace_slug: str,
        message: str,
        mode: str = "chat",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送聊天消息"""
        response = self._http_client.post(
            f"/api/v1/workspaces/{workspace_slug}/chat",
            json={"message": message, "mode": mode, **kwargs},
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 向量数据库操作 ==========
    
    def list_vector_dbs(self) -> list[dict[str, Any]]:
        """获取所有向量数据库"""
        response = self._http_client.get("/api/v1/vector-databases")
        response.raise_for_status()
        return response.json()
    
    def get_vector_db(self, db_id: str) -> dict[str, Any]:
        """获取指定向量数据库"""
        response = self._http_client.get(f"/api/v1/vector-databases/{db_id}")
        response.raise_for_status()
        return response.json()
    
    def create_vector_db(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """创建向量数据库"""
        response = self._http_client.post(
            "/api/v1/vector-databases",
            json={"name": name, **kwargs},
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 集合操作 ==========
    
    def list_collections(self, vector_db_id: Optional[str] = None) -> list[dict[str, Any]]:
        """获取所有集合"""
        params = {"vector-db-id": vector_db_id} if vector_db_id else {}
        response = self._http_client.get("/api/v1/collections", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_collection(self, collection_id: str) -> dict[str, Any]:
        """获取指定集合"""
        response = self._http_client.get(f"/api/v1/collections/{collection_id}")
        response.raise_for_status()
        return response.json()
    
    def create_collection(
        self,
        name: str,
        vector_db_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """创建集合"""
        response = self._http_client.post(
            "/api/v1/collections",
            json={"name": name, "vectorDbId": vector_db_id, **kwargs},
        )
        response.raise_for_status()
        return response.json()
    
    # ========== LLM 提供商操作 ==========
    
    def list_llm_providers(self) -> list[dict[str, Any]]:
        """获取所有 LLM 提供商"""
        response = self._http_client.get("/api/v1/llm-providers")
        response.raise_for_status()
        return response.json()
    
    def get_llm_provider(self, provider_id: str) -> dict[str, Any]:
        """获取指定 LLM 提供商"""
        response = self._http_client.get(f"/api/v1/llm-providers/{provider_id}")
        response.raise_for_status()
        return response.json()
    
    def update_llm_provider(
        self,
        provider_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """更新 LLM 提供商"""
        response = self._http_client.patch(
            f"/api/v1/llm-providers/{provider_id}",
            json=kwargs,
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 嵌入模型操作 ==========
    
    def list_embedding_models(self) -> list[dict[str, Any]]:
        """获取所有嵌入模型"""
        response = self._http_client.get("/api/v1/embedding")
        response.raise_for_status()
        return response.json()
    
    def get_embedding_model(self, model_id: str) -> dict[str, Any]:
        """获取指定嵌入模型"""
        response = self._http_client.get(f"/api/v1/embedding/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def update_embedding_model(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """更新嵌入模型"""
        response = self._http_client.patch(
            f"/api/v1/embedding/{model_id}",
            json=kwargs,
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 转录模型操作 ==========
    
    def list_transcription_models(self) -> list[dict[str, Any]]:
        """获取所有转录模型"""
        response = self._http_client.get("/api/v1/transcription")
        response.raise_for_status()
        return response.json()
    
    # ========== 用户设置操作 ==========
    
    def get_user_settings(self) -> dict[str, Any]:
        """获取用户设置"""
        response = self._http_client.get("/api/v1/user/settings")
        response.raise_for_status()
        return response.json()
    
    def update_user_settings(self, **kwargs: Any) -> dict[str, Any]:
        """更新用户设置"""
        response = self._http_client.post(
            "/api/v1/user/settings",
            json=kwargs,
        )
        response.raise_for_status()
        return response.json()
    
    # ========== 通用请求方法 ==========
    
    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        """发送 GET 请求"""
        return self._http_client.get(path, **kwargs)
    
    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        """发送 POST 请求"""
        return self._http_client.post(path, **kwargs)
    
    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        """发送 PUT 请求"""
        return self._http_client.put(path, **kwargs)
    
    def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        """发送 PATCH 请求"""
        return self._http_client.patch(path, **kwargs)
    
    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        """发送 DELETE 请求"""
        return self._http_client.delete(path, **kwargs)


# 全局客户端实例
_client: Optional[AnythingLLMClient] = None


def get_client() -> AnythingLLMClient:
    """获取全局 API 客户端实例（延迟初始化）"""
    global _client
    if _client is None:
        _client = AnythingLLMClient()
    return _client


def close_client() -> None:
    """关闭全局 API 客户端"""
    global _client
    if _client:
        _client.close()
        _client = None
