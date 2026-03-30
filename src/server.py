import argparse
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the Enhanced τ²-Bench Purple Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=os.getenv("AGENT_LLM", "openai/gpt-4.1"),
        help="LLM model to use (litellm format: provider/model)",
    )
    args = parser.parse_args()

    os.environ.setdefault("AGENT_LLM", args.agent_llm)

    skill = AgentSkill(
        id="task_fulfillment",
        name="Advanced Task Fulfillment",
        description=(
            "Expert customer service agent for τ²-Bench evaluation. "
            "Uses structured reasoning, careful policy adherence, and "
            "precise tool calling to resolve customer issues."
        ),
        tags=["benchmark", "tau2", "customer-service", "tool-calling"],
        examples=[],
    )

    agent_card = AgentCard(
        name="PurpleAgent-Tau2-Enhanced",
        description=(
            "An enhanced Purple Agent for the AgentBeats τ²-Bench competition. "
            "Features advanced prompting with ReAct reasoning, strict policy "
            "compliance, and reliable multi-turn conversation handling."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"Starting Purple Agent on {args.host}:{args.port} with model {args.agent_llm}")
    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
