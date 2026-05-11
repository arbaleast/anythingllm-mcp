# AnythingLLM MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that
lets MCP-compatible clients (VS Code, Claude Desktop, Cursor, and others)
interact with [AnythingLLM](https://anythingllm.com).

## Available Tools

- **Workspace**: List, create, update, delete workspaces
- **Chat**: Send messages, get chat history, manage sessions
- **Threads**: Create, update, delete, chat in threads
- **Documents**: Upload files, links, raw text; manage folders
- **Embeddings**: Add/remove documents to/from workspaces
- **Search**: Vector similarity search within workspaces
- **System**: Get settings, vector counts, export chats
- **Auth**: Verify API key validity

## Requirements

- **Python 3.10+** — `python --version` or [download](https://www.python.org/downloads/)
- **uv** — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **AnythingLLM** running at `http://localhost:3001`
- **AnythingLLM API key** — create in AnythingLLM Settings → API Keys

## Installation

```sh
git clone https://github.com/andreperez/anythingllm-mcp.git
cd anythingllm-mcp
uv sync
```

## Configuration

Set environment variables:

| Variable | Required | Default |
|----------|----------|---------|
| `ANYTHINGLLM_BASE_URL` | No | `http://localhost:3001` |
| `ANYTHINGLLM_API_KEY` | Yes | — |

### Using `.env` file

```sh
cp .env.example .env
# Edit .env with your API key
uv run --env-file .env anythingllm-mcp
```

## Client Setup

### VS Code

Add to your `mcp.json` (Command Palette → **MCP: Open User Configuration**):

```json
{
  "servers": {
    "anythingllm": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/anythingllm-mcp", "anythingllm-mcp"],
      "env": {
        "ANYTHINGLLM_API_KEY": "${input:anythingllm_api_key}",
        "ANYTHINGLLM_BASE_URL": "http://localhost:3001"
      }
    }
  }
}
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "anythingllm": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/anythingllm-mcp", "anythingllm-mcp"],
      "env": {
        "ANYTHINGLLM_API_KEY": "YOUR_API_KEY",
        "ANYTHINGLLM_BASE_URL": "http://localhost:3001"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` or `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "anythingllm": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/anythingllm-mcp", "anythingllm-mcp"],
      "env": {
        "ANYTHINGLLM_API_KEY": "YOUR_API_KEY",
        "ANYTHINGLLM_BASE_URL": "http://localhost:3001"
      }
    }
  }
}
```

## Development

```sh
uv sync --extra dev
uv run pytest
```

## Troubleshooting

### "API key is missing" / 401 Unauthorized
- Verify `ANYTHINGLLM_API_KEY` is set correctly
- Check the key is active in AnythingLLM Settings → API Keys

### "Connection refused"
- Confirm AnythingLLM is running (`http://localhost:3001`)
- Check `ANYTHINGLLM_BASE_URL` is correct
- For Docker: use `http://host.docker.internal:3001` or the host-published port

## Security

- Never commit real API keys
- Use environment variables for secrets
- Use least-privileged API tokens when possible

## License

MIT
