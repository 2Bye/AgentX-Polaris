#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/_tutorial"
cp "$DIR/.env" .
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$DIR/src"
export TAU2_DATA_DIR="$PWD/scenarios/tau2/tau2-bench/data"
uv run agentbeats-run scenarios/tau2/test_enhanced.toml --show-logs
