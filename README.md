# 🟣 Purple Agent — τ²-Bench Enhanced

An enhanced Purple Agent for the [AgentX-AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) competition, optimized for the τ²-Bench track.

## Features

- 🧠 **Advanced Prompting** — Structured reasoning with policy-first approach
- 🔄 **Robust Error Handling** — Retry logic with JSON extraction fallback
- 🔧 **Tool Calling** — Precise tool execution following domain policy
- 🐳 **Containerized** — Docker-ready for AgentBeats platform deployment
- 📊 **Multi-LLM Support** — Switch between OpenAI, Anthropic, Google via litellm

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp sample.env .env
# Edit .env and add your API key

# Run the agent
uv run src/server.py --host 127.0.0.1 --port 9019
```

## Running with Docker

```bash
docker build --platform linux/amd64 -t purple-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=your-key purple-agent
```

## Testing with τ²-Bench locally

See the [agentbeats-tutorial](https://github.com/RDI-Foundation/agentbeats-tutorial) for running assessments locally.

## Project Structure

```
src/
├── server.py      # A2A server + agent card configuration
├── executor.py    # A2A request handling and task lifecycle
└── agent.py       # Core agent logic with enhanced prompting
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `AGENT_LLM` | `openai/gpt-4.1` | LLM model (litellm format) |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GOOGLE_API_KEY` | - | Google API key |
