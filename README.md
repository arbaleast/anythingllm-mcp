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

### Option 1: Using uv (Recommended)

```sh
git clone https://github.com/arbaleast/anythingllm-mcp.git
cd anythingllm-mcp
uv sync
```

### Option 2: Using pip

```sh
git clone https://github.com/arbaleast/anythingllm-mcp.git
cd anythingllm-mcp
pip install -r requirements.txt
```

### Option 3: Using Docker

```sh
# Build the image
docker build -t anythingllm-mcp .

# Run with environment variables
docker run -d \
  --name anythingllm-mcp \
  -e ANYTHINGLLM_API_BASE_URL=http://host.docker.internal:3001 \
  -e ANYTHINGLLM_API_KEY=your_api_key_here \
  anythingllm-mcp

# Or use a .env file
docker run -d \
  --name anythingllm-mcp \
  --env-file .env \
  -e ANYTHINGLLM_API_BASE_URL=http://host.docker.internal:3001 \
  anythingllm-mcp
```

> **Note:** When running Docker, use `http://host.docker.internal:3001` (Windows/macOS) 
> or the host machine's IP address instead of `localhost` to access AnythingLLM running on the host.

## Configuration

Set environment variables:

| Variable | Required | Default |
|----------|----------|---------|
| `ANYTHINGLLM_API_BASE_URL` | No | `http://localhost:3001` |
| `ANYTHINGLLM_API_KEY` | Yes | — |

### Using `.env` file

```sh
cp .env.example .env
# Edit .env with your API key
uv run --env-file .env anythingllm-mcp
```

> **Security:** Ensure your `.env` file has restricted permissions (`chmod 600 .env`) and is never committed to version control.

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
        "ANYTHINGLLM_API_BASE_URL": "http://localhost:3001"
      }
    }
  }
}
```

> **Security Note:** Use VS Code's `input` variables or environment variable injection instead of hardcoding API keys. See `examples/vscode_mcp.json` for a complete example.

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
        "ANYTHINGLLM_API_BASE_URL": "http://localhost:3001"
      }
    }
  }
}
```

> **Security Note:** Consider using environment variable files or secure secret storage for API keys.

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
        "ANYTHINGLLM_API_BASE_URL": "http://localhost:3001"
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
- Check `ANYTHINGLLM_API_BASE_URL` is correct
- For Docker: use `http://host.docker.internal:3001` or the host-published port

## Docker Deployment

The MCP server can be deployed as a Docker container for easy setup and isolation.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANYTHINGLLM_API_BASE_URL` | No | `http://localhost:3001` | AnythingLLM server URL |
| `ANYTHINGLLM_API_KEY` | Yes | — | Your AnythingLLM API key |
| `HTTP_API_KEY` | No | — | Gateway auth key (for SSE/streamable transport) |
| `MCP_SERVER_HOST` | No | `0.0.0.0` | MCP server bind address |
| `MCP_SERVER_PORT` | No | `8765` | MCP server port |
| `DEFAULT_TIMEOUT` | No | `180.0` | Request timeout (seconds) |
| `UPLOAD_TIMEOUT` | No | `300.0` | Upload timeout (seconds) |

### Building the Image

```sh
docker build -t anythingllm-mcp .
```

### Running the Container

```sh
# With inline environment variables
docker run -d \
  --name anythingllm-mcp \
  -e ANYTHINGLLM_API_BASE_URL=http://host.docker.internal:3001 \
  -e ANYTHINGLLM_API_KEY=your_api_key_here \
  anythingllm-mcp

# Using .env file
docker run -d \
  --name anythingllm-mcp \
  --env-file .env \
  anythingllm-mcp
```

### Connecting to AnythingLLM

When AnythingLLM runs on your host machine:

- **Windows/macOS:** Use `http://host.docker.internal:3001`
- **Linux:** Use `http://172.17.0.1:3001` or your host's actual IP address

## Security

### Best Practices

1. **Never commit secrets to version control**
   - Add `.env` to your `.gitignore`
   - Never commit real API keys or secrets

2. **Use strong, randomly generated keys**
   ```bash
   # Generate a secure random key
   openssl rand -hex 32
   ```

3. **Restrict file permissions**
   ```bash
   chmod 600 .env
   ```

4. **Rotate keys periodically**
   - Change API keys regularly for production environments
   - Immediately rotate if a key is compromised

5. **Production deployments**
   - Use environment variables instead of `.env` files when possible
   - Use secrets management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Enable HTTPS for all network communication
   - Use least-privileged API tokens

6. **API key security**
   - Never expose API keys in client-side code
   - Use MCP client's secure variable injection (e.g., `${input:...}`)
   - Rotate keys immediately if compromised

7. **HTTP API Gateway** (for SSE/streamable transport)
   - Always set a strong `HTTP_API_KEY`
   - Use HTTPS in production
   - Disable `MCP_DEBUG` in production

## License

MIT
