#!/usr/bin/env python3
"""
工具注册中心模块
提供 MCP 工具的统一注册和管理
"""
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from mcp.types import Tool


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    
    def to_mcp_tool(self) -> Tool:
        """转换为 MCP Tool 对象"""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
        )


class ToolRegistry:
    """MCP 工具注册中心"""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, list[str]] = {}
    
    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
        category: str = "general",
        tags: Optional[list[str]] = None,
    ) -> None:
        """注册工具
        
        Args:
            name: 工具名称
            description: 工具描述
            input_schema: 输入模式
            handler: 处理函数
            category: 工具分类
            tags: 标签列表
        """
        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            category=category,
            tags=tags or [],
        )
        self._tools[name] = tool_def
        
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Optional[Callable[..., Any]]:
        """获取工具处理器"""
        tool = self._tools.get(name)
        return tool.handler if tool else None
    
    def list_tools(self) -> list[Tool]:
        """列出所有工具"""
        return [tool.to_mcp_tool() for tool in self._tools.values()]
    
    def list_by_category(self, category: str) -> list[Tool]:
        """按分类列出工具"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name].to_mcp_tool() for name in tool_names if name in self._tools]
    
    def get_categories(self) -> list[str]:
        """获取所有分类"""
        return list(self._categories.keys())
    
    def get_tools_by_tag(self, tag: str) -> list[Tool]:
        """按标签获取工具"""
        return [
            tool.to_mcp_tool()
            for tool in self._tools.values()
            if tag in tool.tags
        ]
    
    def unregister(self, name: str) -> bool:
        """注销工具"""
        if name not in self._tools:
            return False
        
        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category].remove(name)
        return True
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 全局工具注册中心实例
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """获取全局工具注册中心实例"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    handler: Callable[..., Any],
    category: str = "general",
    tags: Optional[list[str]] = None,
) -> None:
    """便捷函数：在全局注册中心注册工具"""
    get_registry().register(name, description, input_schema, handler, category, tags)


# ========== 工具装饰器 ==========

def mcp_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    input_schema: Optional[dict[str, Any]] = None,
    category: str = "general",
    tags: Optional[list[str]] = None,
):
    """MCP 工具装饰器
    
    Usage:
        @mcp_tool(name="my_tool", description="Does something", category="custom")
        def my_tool_handler(arg1: str, arg2: int) -> dict:
            return {"result": f"{arg1} - {arg2}"}
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        tool_schema = input_schema or {"type": "object", "properties": {}}
        
        # 注册工具
        register_tool(
            name=tool_name,
            description=tool_desc,
            input_schema=tool_schema,
            handler=func,
            category=category,
            tags=tags,
        )
        
        # 返回原函数（带元数据）
        func._mcp_tool = True
        func._tool_name = tool_name
        func._tool_category = category
        return func
    
    return decorator
