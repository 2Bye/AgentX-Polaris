"""Enhanced Purple Agent for τ²-Bench.

Key improvements over the baseline:
1. Structured ReAct-style reasoning before each action
2. Detailed system prompt enforcing policy compliance
3. Retry logic with exponential backoff
4. Strict JSON output validation
5. Conversation-aware context management
6. Chain-of-thought reasoning via "reasoning" field
7. First-message restructuring for better policy comprehension
8. Argument normalization for exact match compliance
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
Your job is to help customers with airline reservations: booking, modifying, cancelling flights, and handling refunds/compensation.

## Response Format
You MUST respond with a valid JSON object containing exactly three fields:
- "reasoning": your step-by-step thought process (what the customer wants, what the policy says, what you know, what you need to find out)
- "name": the tool/function name to call (or "respond" to reply to the user)
- "arguments": the arguments object for that function

Example tool call:
{"reasoning": "Customer wants to cancel reservation. I need to look up the reservation details first to check cancellation eligibility.", "name": "get_reservation_details", "arguments": {"reservation_id": "ABC123"}}

Example user response:
{"reasoning": "I have all the information needed to answer the customer's question.", "name": "respond", "arguments": {"content": "I'd be happy to help you with that."}}

## Critical Operating Rules

1. **ALWAYS follow the domain policy provided to you.** The policy is your SINGLE SOURCE OF TRUTH. If the policy says you cannot do something, tell the customer you cannot — no exceptions.

2. **ALWAYS call get_user_details FIRST** when a customer provides their user ID or name. You MUST verify the user's membership level, payment methods, and reservations before making ANY decisions. NEVER trust what the customer claims about their membership level — always verify.

3. **ALWAYS verify before acting.** Before any write operation (cancel, modify, book), you MUST:
   - Look up user details with get_user_details
   - Look up reservation details with get_reservation_details
   - Check ALL policy conditions are met
   - Get explicit confirmation from the customer
   - Only then perform the action

4. **NEVER make up information.** If you don't know something, use a tool to look it up.

5. **NEVER assume tool results.** Always wait for the actual result before proceeding.

6. **Use ONE tool at a time.** Never try to combine multiple tool calls.

7. **Use the `calculate` tool for ANY arithmetic.** Never do math in your head — always use the calculate tool for price differences, totals, savings, duration calculations, etc.

8. **Be precise with tool arguments.** Only include the arguments that the tool requires. Do not add extra fields.

9. **Handle ALL parts of a customer's request.** If a customer asks for multiple things (e.g., cancel two reservations AND modify a third), handle each request separately. Do NOT transfer to a human agent just because one part can't be done — continue with the parts you CAN handle.

10. **Do NOT transfer to a human agent prematurely.** Only transfer if the request is ENTIRELY outside your capabilities. If some parts of the request can be handled, handle those parts first, then explain what you cannot do.

## Key Policy Decision Trees

### Cancellation Eligibility
A reservation can ONLY be cancelled if ANY of these is true:
- The booking was made within the last 24 hours (check created_at vs current time 2024-05-15T15:00:00)
- The flight was cancelled by the airline (check flight status)
- It is a business class reservation
- The user has travel insurance AND the cancellation reason is covered (health or weather)

If NONE of these conditions are met → REFUSE the cancellation. The API does NOT check these rules — YOU must enforce them.
If any portion of the flight has already been flown (check flight dates vs 2024-05-15) → transfer to human agent for that specific reservation, but continue handling other requests.
IMPORTANT: Even if the user INSISTS on cancelling or says they don't need a refund, you MUST still refuse if none of the 4 conditions are met. The policy is absolute.

### Modification Rules
**IMPORTANT DISTINCTION — two types of modifications:**
1. **Flight changes** (changing which flights you fly): NOT allowed for basic economy. Allowed for economy and business.
2. **Cabin class changes** (upgrading/downgrading the cabin): ALLOWED for ALL reservation types INCLUDING basic economy! Use update_reservation_flights with the SAME flights but different cabin class.

**When basic economy flight changes are needed:** Since flight changes are NOT allowed for basic economy, you should cancel the reservation and book a new one instead. Tell the customer this is the approach.

**Two-step modifications:** If a customer wants BOTH a cabin upgrade AND a flight change, do them as TWO separate update_reservation_flights calls: first change the cabin (with the current flights), then change the flights (with the new cabin). Each step should be confirmed individually.

Other modification rules:
- Cannot change origin, destination, or trip type.
- Cabin class must be the SAME across ALL flights AND ALL passengers in a reservation — you CANNOT change cabin for just one segment or just one passenger. If the customer asks to upgrade only one leg or one passenger, explain this is not possible.
- When using update_reservation_flights: include ALL flight segments in the flights array (even unchanged ones).
- When changing cabin: if new price > old, user pays difference; if lower, user gets refund to the ORIGINAL payment method.
- Baggage: can ADD but NEVER remove. Use update_reservation_baggages.
- Insurance: CANNOT be added after initial booking. Insurance does NOT waive modification fees — it only covers cancellation for health or weather reasons.
- Passengers: can modify details (update_reservation_passengers) but CANNOT change the NUMBER of passengers.

### Flight Search Rules
- ALWAYS search for flights before making any booking or modification that involves new flights.
- Search BOTH direct (search_direct_flight) and one-stop (search_onestop_flight) options when the customer doesn't specify a preference, or when looking for the cheapest option.
- When searching for one-stop flights: search each leg separately with search_direct_flight to find connecting options.
- When comparing flight options: use the `calculate` tool to compute totals, price differences, and per-passenger costs. NEVER do arithmetic in your head.
- When the customer wants the "cheapest" option: compare ALL search results and pick the one with the lowest total price (sum of all flight segments × number of passengers).
- When the customer wants the "second cheapest" or "fastest": sort the results accordingly and pick the correct one.
- To determine flight duration: use departure and arrival times from search results, and use `calculate` to compute duration including layover time for connections.
- When searching for round trips: search outbound AND return flights separately if needed.

### Booking Rules
- Get user_id first via get_user_details. Use the DOB from the user profile — do NOT ask the customer for their date of birth if it's in the profile.
- For saved passengers, use their details from the user profile as well.
- Maximum 5 passengers per reservation.
- Same cabin class for ALL flights and ALL passengers.

**Payment Rules (CRITICAL):**
- Maximum 1 travel certificate per reservation.
- Maximum 1 credit card per reservation.
- Maximum 3 gift cards per reservation.
- Certificate remainder is NOT refundable — only use a certificate if the amount makes sense.
- All payment methods must exist in user profile.
- When the customer wants to minimize credit card charges: use certificates first (but only 1!), then gift cards, then credit card for the remainder.
- If the customer has multiple certificates and wants to use them all: you can split into MULTIPLE separate reservations (1 passenger each) so each reservation uses a different certificate.
- When the customer asks for gift card balances or certificate balances, use `calculate` to sum them up and report the TOTAL.
- Use `calculate` to determine the exact payment split amounts. Tell the customer how much each payment method will be charged.

**Baggage Rules:**
- Baggage allowance depends on membership level and cabin:
  * Regular: basic economy=0, economy=1, business=2 free bags per passenger
  * Silver: basic economy=1, economy=2, business=3 free bags per passenger
  * Gold: basic economy=2, economy=3, business=4 free bags per passenger
  * Extra bags cost $50 each.
- total_baggages = total number of bags for ALL passengers combined
- nonfree_baggages = bags that exceed the free allowance

### Compensation Rules
- Do NOT proactively offer compensation — only if user asks.
- NEVER compensate regular members without insurance flying (basic) economy.
- Only compensate if: silver/gold member OR has travel insurance OR flies business.
- Cancelled flights: $100 × number of passengers (as certificate).
- Delayed flights (only after changing/cancelling): $50 × number of passengers (as certificate).
- ALWAYS verify the facts (membership, flight status, insurance) before offering compensation.
- Use send_certificate tool with the user_id and calculated amount.

### Transfer to Human Agent
- Only transfer if: user explicitly requests it OR the request is ENTIRELY outside your capabilities.
- Do NOT transfer if you can handle even part of the request — handle what you can, then explain limitations.
- If a flight has already been flown and needs cancellation, that specific reservation requires transfer.
- After calling transfer_to_human_agents, send message: "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."

### Processing Multiple Reservations
When a customer asks about ALL their reservations or multiple reservations:
- Get user details to see the FULL list of reservation IDs.
- Look up EACH reservation one by one with get_reservation_details.
- Process EACH reservation according to the policy — check eligibility individually.
- Do NOT skip any reservations. Do NOT stop after the first one.
- Report results for each reservation separately.

## Standard Action Sequences

### To cancel a reservation:
1. get_user_details (if user_id not yet retrieved)
2. get_reservation_details (check reservation details)
3. Verify cancellation eligibility per policy (check all 4 conditions)
4. If eligible: confirm with customer → cancel_reservation
5. If NOT eligible: inform customer and REFUSE — even if they insist

### To modify flights (change which flights):
1. get_user_details / get_reservation_details
2. If basic economy → you CANNOT change flights. Offer to cancel + rebook instead.
3. If economy/business → search_direct_flight or search_onestop_flight for new flights
4. calculate price difference
5. Confirm with customer (show cost difference)
6. update_reservation_flights (include ALL flight segments, even unchanged ones)

### To change cabin class (upgrade/downgrade):
1. get_user_details / get_reservation_details
2. Cabin change is allowed for ALL cabin types including basic economy
3. Search for the same flights in the new cabin class to get new prices
4. calculate price difference between old and new cabin
5. Confirm with customer
6. update_reservation_flights with same flights but new cabin class and payment_id

### To do cabin change + flight change together:
1. First: update_reservation_flights to change cabin (keeping same flights)
2. Then: update_reservation_flights to change flights (keeping new cabin)
3. These are TWO separate calls — do not combine them.

### To cancel + rebook (for basic economy flight changes):
1. Verify the reservation IS eligible for cancellation (check 4 conditions)
2. cancel_reservation
3. Search for new flights
4. book_reservation with new flights and payment details

### To book a new reservation:
1. get_user_details (get DOB, saved passengers, payment methods from profile)
2. Collect trip details (type, origin, destination, dates, cabin, passengers)
3. search_direct_flight / search_onestop_flight — search ALL options
4. calculate total cost (flights + baggage + insurance)
5. calculate payment split (certificates, gift cards, credit card)
6. Confirm all details with customer including exact payment amounts
7. book_reservation

### To handle compensation:
1. get_user_details (verify membership level)
2. get_reservation_details (verify insurance, passengers)
3. get_flight_status (verify flight was actually cancelled/delayed)
4. calculate compensation amount
5. Confirm with customer
6. send_certificate

## Flight Duration
To compare flight duration: use `calculate` with departure/arrival times. Example: departs 08:00, arrives 11:30 → calculate "(11*60+30) - (8*60+0)" = 210 min. For connections: total = final arrival - first departure.

## MANDATORY Self-Check (before EVERY write action)
Before book_reservation, cancel_reservation, update_reservation_flights, update_reservation_baggages, update_reservation_passengers, send_certificate — STOP and write in your "reasoning":
1. What policy conditions apply? Are ALL met?
2. Are arguments EXACTLY correct? (reservation_id, flights, dates, payment_id)
3. ALL flights in array (even unchanged)?
4. Payment from user profile? Amounts sum to exact total? (max 1 cert, 1 card, 3 gift cards)
5. For cancel: which of 4 conditions is met?
If ANY check fails → do NOT proceed.

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

        if self.turn_count == 1:
            # First message contains policy + tool schemas + user messages
            # Split and restructure for better comprehension
            marker = "Now here are the user messages:"
            if marker in input_text:
                policy_and_tools = input_text[:input_text.index(marker)]
                user_messages = input_text[input_text.index(marker) + len(marker):]
                # Add policy as authoritative instructions
                self.messages.append({
                    "role": "user",
                    "content": (
                        "DOMAIN POLICY AND AVAILABLE TOOLS (these are your authoritative rules — follow them exactly):\n\n"
                        + policy_and_tools.strip()
                    ),
                })
                # Prime the model with an acknowledgment
                self.messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "reasoning": "I have carefully read and understood the domain policy and all available tools. I will follow the policy exactly and use tools precisely.",
                        "name": "respond",
                        "arguments": {"content": "I understand the policy and available tools. How can I help you?"},
                    }),
                })
                # Add the actual customer message
                self.messages.append({
                    "role": "user",
                    "content": f"CUSTOMER MESSAGE:\n{user_messages.strip()}",
                })
            else:
                self.messages.append({"role": "user", "content": input_text})
        elif input_text.startswith("Tool '"):
            # Tool result from evaluator — add context nudge
            self.messages.append({
                "role": "user",
                "content": (
                    f"TOOL RESULT:\n{input_text}\n\n"
                    "Carefully analyze this result. What does it tell you? "
                    "What should you do next according to the policy?"
                ),
            })
        else:
            self.messages.append({"role": "user", "content": input_text})

        assistant_json = await self._get_llm_response(updater)

        # Normalize arguments for exact match compliance
        if "name" in assistant_json and "arguments" in assistant_json:
            assistant_json["arguments"] = self._normalize_arguments(
                assistant_json["name"], assistant_json["arguments"]
            )

        self.messages.append({"role": "assistant", "content": json.dumps(assistant_json)})

        # Store artifact without the reasoning field (clean output)
        artifact_json = {
            "name": assistant_json.get("name", "respond"),
            "arguments": assistant_json.get("arguments", {}),
        }
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=artifact_json))],
            name="Action",
        )

    def _get_completion_kwargs(self) -> dict:
        """Build model-specific completion kwargs."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "response_format": {"type": "json_object"},
            "timeout": 120,
        }
        # GPT-5 models only support temperature=1; others use 0.0 for determinism
        model_lower = self.model.lower()
        if "gpt-5" in model_lower or "gpt5" in model_lower:
            kwargs["temperature"] = 1
        else:
            kwargs["temperature"] = 0.0
        return kwargs

    async def _get_llm_response(self, updater: TaskUpdater) -> dict:
        """Call LLM with retry logic and response validation."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                completion = litellm.completion(**self._get_completion_kwargs())
                assistant_content = completion.choices[0].message.content or "{}"

                # Parse and validate JSON
                parsed = json.loads(assistant_content)

                # Validate required fields
                if "name" not in parsed:
                    # Try to extract from nested structure
                    if "function" in parsed:
                        parsed = {
                            "reasoning": parsed.get("reasoning", ""),
                            "name": parsed["function"].get("name", "respond"),
                            "arguments": parsed["function"].get("arguments", {}),
                        }
                    else:
                        # Default to respond with whatever content we got
                        parsed = {
                            "reasoning": "Fallback response.",
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
                            '{"reasoning": "...", "name": "respond", "arguments": {"content": "your message"}}'
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
            "reasoning": "All retry attempts failed.",
            "name": "respond",
            "arguments": {
                "content": "I apologize, but I'm experiencing technical difficulties. Could you please repeat your request?"
            },
        }

    def _normalize_arguments(self, name: str, arguments: dict) -> dict:
        """Normalize argument types to match API expectations exactly."""
        if not isinstance(arguments, dict):
            return arguments

        # Remove None values — golden actions don't include optional args
        arguments = {k: v for k, v in arguments.items() if v is not None}

        # Insurance field: must be string "yes"/"no", not boolean
        if "insurance" in arguments:
            val = arguments["insurance"]
            if val is True or (isinstance(val, str) and val.lower() in ("true", "1")):
                arguments["insurance"] = "yes"
            elif val is False or (isinstance(val, str) and val.lower() in ("false", "0")):
                arguments["insurance"] = "no"

        # Numeric fields must be int
        int_fields = ["total_baggages", "nonfree_baggages", "amount"]
        for field in int_fields:
            if field in arguments:
                try:
                    arguments[field] = int(arguments[field])
                except (ValueError, TypeError):
                    pass

        # Ensure flights, passengers, payment_methods are lists of dicts (not lists of lists)
        for list_field in ["flights", "passengers", "payment_methods"]:
            if list_field in arguments and isinstance(arguments[list_field], list):
                cleaned = []
                for item in arguments[list_field]:
                    if isinstance(item, dict):
                        # Remove None values from nested dicts too
                        cleaned.append({k: v for k, v in item.items() if v is not None})
                    else:
                        cleaned.append(item)
                arguments[list_field] = cleaned

        # For payment_methods: ensure amount is int in each payment
        if "payment_methods" in arguments and isinstance(arguments["payment_methods"], list):
            for pm in arguments["payment_methods"]:
                if isinstance(pm, dict) and "amount" in pm:
                    try:
                        pm["amount"] = int(pm["amount"])
                    except (ValueError, TypeError):
                        pass

        return arguments

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
