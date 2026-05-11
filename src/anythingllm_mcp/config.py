#!/usr/bin/env python3
"""
统一配置管理模块
提供集中化的配置管理，支持环境变量和代码配置
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """AnythingLLM MCP 服务器配置类"""
    
    # AnythingLLM API 配置
    api_base_url: str = "http://localhost:3001"
    api_key: str = ""
    
    # MCP 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 8765
    server_url: str = "http://localhost:8765"
    
    # HTTP API 认证配置
    http_api_key: str = ""
    
    # 超时配置
    default_timeout: float = 180.0
    upload_timeout: float = 300.0
    
    # OAuth 配置
    auth_code_expiry: int = 600  # 10 minutes
    token_expiry: int = 3600     # 1 hour
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置
        
        支持旧的环境变量名以保持向后兼容:
        - ANYTHINGLLM_BASE_URL (旧) -> ANYTHINGLLM_API_BASE_URL (新)
        - SERVER_URL (旧) -> MCP_SERVER_URL (新)
        """
        # API base URL: 优先使用新名称，旧名称作为备选
        api_base = os.getenv("ANYTHINGLLM_API_BASE_URL") or os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
        
        # Server URL: 优先使用新名称，旧名称作为备选
        server_url = os.getenv("MCP_SERVER_URL") or os.getenv("SERVER_URL", "http://localhost:8765")
        
        # Server port: 优先使用新名称，旧名称作为备选
        server_port = int(os.getenv("MCP_SERVER_PORT") or os.getenv("SERVER_PORT", "8765"))
        
        return cls(
            api_base_url=api_base,
            api_key=os.getenv("ANYTHINGLLM_API_KEY", ""),
            server_host=os.getenv("MCP_SERVER_HOST", os.getenv("SERVER_HOST", "0.0.0.0")),
            server_port=server_port,
            server_url=server_url,
            http_api_key=os.getenv("HTTP_API_KEY", ""),
            default_timeout=float(os.getenv("DEFAULT_TIMEOUT", "180.0")),
            upload_timeout=float(os.getenv("UPLOAD_TIMEOUT", "300.0")),
            auth_code_expiry=int(os.getenv("AUTH_CODE_EXPIRY", "600")),
            token_expiry=int(os.getenv("TOKEN_EXPIRY", "3600")),
        )
    
    @property
    def is_api_key_configured(self) -> bool:
        """检查 API 密钥是否已配置"""
        return bool(self.api_key and self.api_key.strip())
    
    @property
    def is_http_api_key_configured(self) -> bool:
        """检查 HTTP API 密钥是否已配置"""
        return bool(self.http_api_key and self.http_api_key.strip())
    
    def validate(self) -> list[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        if not self.api_base_url:
            errors.append("API base URL is required")
        elif not self.api_base_url.startswith(("http://", "https://")):
            errors.append("API base URL must start with http:// or https://")
        
        if self.server_port < 1 or self.server_port > 65535:
            errors.append("Server port must be between 1 and 65535")
        
        if self.default_timeout <= 0:
            errors.append("Default timeout must be positive")
        
        if self.upload_timeout <= 0:
            errors.append("Upload timeout must be positive")
        
        return errors


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例（延迟初始化）"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config() -> Config:
    """重新加载配置"""
    global _config
    _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """设置全局配置实例"""
    global _config
    _config = config
