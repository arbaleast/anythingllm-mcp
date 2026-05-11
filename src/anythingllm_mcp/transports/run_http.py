#!/usr/bin/env python3
"""HTTP wrapper for anythingllm-mcp (FastMCP SSE mode on 0.0.0.0:8765)."""
import anythingllm_mcp
anythingllm_mcp.mcp.run(transport="sse")
