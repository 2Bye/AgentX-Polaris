#!/bin/bash
# Start the enhanced Purple Agent using our venv
# Usage: ./run_agent.sh [--host HOST] [--port PORT] [args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/src/server.py" "$@"
