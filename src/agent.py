"""Enhanced Purple Agent for τ²-Bench.

Key improvements over the baseline:
1. Structured ReAct-style reasoning before each action
2. Detailed system prompt enforcing policy compliance
3. Retry logic with exponential backoff
4. Strict JSON output validation
5. Conversation-aware context management
"""

import json
import os
import time
import traceback

from dotenv import load_dotenv
import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

# Suppress litellm debug noise
litellm.suppress_debug_info = True

SYSTEM_PROMPT = """\
You are an expert customer service agent operating within a strict policy framework.

## Your Core Responsibilities
1. **Understand the customer's problem** — Ask clarifying questions when needed.
2. **Follow the policy EXACTLY** — Never deviate from the provided policy rules. The policy is your single source of truth.
3. **Use tools precisely** — Call tools with correct arguments. Verify tool results before proceeding.
4. **Resolve the issue completely** — Don't leave tasks half-done.

## Critical Rules
- ALWAYS follow the domain policy provided to you. If the policy says you cannot do something, tell the customer you cannot.
- When the policy requires checking specific conditions before taking action, ALWAYS check those conditions first using the appropriate tools.
- NEVER make up information. If you don't know something, use a tool to look it up.
- NEVER assume the result of a tool call — always wait for the actual result.
- When you need to respond to the user, use the "respond" action.
- When you need to perform an action, use the appropriate tool.
- Only use ONE tool at a time.
- Think step-by-step about what the policy requires before taking any action.

## Response Format
You MUST respond with a valid JSON object containing exactly two fields:
- "name": the tool/function name to call (or "respond" to reply to the user)
- "arguments": the arguments object for that function

Example tool call:
{"name": "get_order_details", "arguments": {"order_id": "12345"}}

Example user response:
{"name": "respond", "arguments": {"content": "I'd be happy to help you with that. Let me look into your account."}}

## Reasoning Process
Before each action, think through:
1. What is the customer asking for?
2. What does the policy say about this?
3. What information do I need? Do I have it or do I need to look it up?
4. What is the correct action to take?

IMPORTANT: Output ONLY the JSON object. No markdown, no code blocks, no extra text.\
"""


class Agent:
    """Enhanced agent with structured reasoning and robust error handling."""

    def __init__(self):
        self.model = os.getenv("AGENT_LLM", os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1"))
        self.messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.turn_count = 0
        self.max_retries = 3

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        self.turn_count += 1

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Processing turn {self.turn_count}..."),
        )

        self.messages.append({"role": "user", "content": input_text})

        assistant_json = await self._get_llm_response(updater)

        self.messages.append({"role": "assistant", "content": json.dumps(assistant_json)})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )

    async def _get_llm_response(self, updater: TaskUpdater) -> dict:
        """Call LLM with retry logic and response validation."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                completion = litellm.completion(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=120,
                )
                assistant_content = completion.choices[0].message.content or "{}"

                # Parse and validate JSON
                parsed = json.loads(assistant_content)

                # Validate required fields
                if "name" not in parsed:
                    # Try to extract from nested structure
                    if "function" in parsed:
                        parsed = {
                            "name": parsed["function"].get("name", "respond"),
                            "arguments": parsed["function"].get("arguments", {}),
                        }
                    else:
                        # Default to respond with whatever content we got
                        parsed = {
                            "name": "respond",
                            "arguments": {"content": str(parsed)},
                        }

                if "arguments" not in parsed:
                    parsed["arguments"] = {}

                # Ensure arguments is a dict
                if isinstance(parsed["arguments"], str):
                    try:
                        parsed["arguments"] = json.loads(parsed["arguments"])
                    except json.JSONDecodeError:
                        if parsed["name"] == "respond":
                            parsed["arguments"] = {"content": parsed["arguments"]}
                        else:
                            parsed["arguments"] = {}

                return parsed

            except json.JSONDecodeError as e:
                last_error = e
                # If JSON parsing fails, try to extract JSON from the response
                if assistant_content:
                    extracted = self._extract_json(assistant_content)
                    if extracted:
                        return extracted

                if attempt < self.max_retries - 1:
                    # Add a correction hint
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Please respond with ONLY a JSON object like: "
                            '{"name": "respond", "arguments": {"content": "your message"}}'
                        ),
                    })
                    time.sleep(0.5 * (attempt + 1))

            except Exception as e:
                last_error = e
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        print(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        traceback.print_exc()
        return {
            "name": "respond",
            "arguments": {
                "content": "I apologize, but I'm experiencing technical difficulties. Could you please repeat your request?"
            },
        }

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract a JSON object from text that might have extra content."""
        # Try to find JSON between braces
        depth = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = -1
        return None
