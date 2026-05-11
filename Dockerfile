# AnythingLLM MCP Server Dockerfile
# Multi-stage build for optimized production image

FROM python:3.10-slim AS builder

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (faster than pip)
RUN uv sync --no-install-project --no-dev --no-editable

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set up environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN groupadd --gid 1000 mcpuser && \
    useradd --uid 1000 --gid mcpuser --shell /bin/bash --create-home mcpuser

# Copy application code
COPY --chown=mcpuser:mcpuser . .

# Switch to non-root user
USER mcpuser

# Default environment variables with placeholders
ENV ANYTHINGLLM_BASE_URL=${ANYTHINGLLM_BASE_URL:-http://localhost:3001}
ENV ANYTHINGLLM_API_KEY=${ANYTHINGLLM_API_KEY:-}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('${ANYTHINGLLM_BASE_URL}/api/v1/health', timeout=5)" || exit 1

# Expose default port (if using HTTP transport)
EXPOSE 8766

# Run the MCP server
ENTRYPOINT ["python", "-m", "anythingllm_mcp"]
CMD ["--help"]
