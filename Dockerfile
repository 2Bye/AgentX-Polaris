FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV UV_HTTP_TIMEOUT=300

RUN adduser agentbeats
USER agentbeats
RUN mkdir -p /home/agentbeats/.cache/uv
WORKDIR /home/agentbeats/agent

COPY pyproject.toml README.md ./

RUN \
    --mount=type=cache,target=/home/agentbeats/.cache/uv,uid=1000 \
    uv sync --no-dev --no-install-project

COPY src src

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
