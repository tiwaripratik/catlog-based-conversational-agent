"""
LLM Integration Module
========================
Wraps the HuggingFace Inference API for meta-llama/Llama-3.1-8B-Instruct.
Handles:
  - System prompt engineering for the SHL assessment recommender agent
  - Structured JSON output parsing (reply + recommendations + end_of_conversation)
  - Timeout handling (must respond within 30 seconds)
  - Error recovery and fallback responses
"""

import json
import os
import re
import time
from typing import Optional
from huggingface_hub import InferenceClient

# ─── Configuration ───────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:novita"
MAX_TOKENS = 1024
TEMPERATURE = 0.3  # Low temperature for consistent, grounded responses
TIMEOUT_SECONDS = 25  # Leave 5s buffer for network + processing (total 30s limit)

# ─── System Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an SHL Assessment Recommender Agent. Your job is to help hiring managers find the right SHL assessments through natural multi-turn conversation.

## YOUR RULES (STRICTLY FOLLOW):

1. **CLARIFY first**: If the user's query is vague (no role, no level, no specific need), ask ONE focused clarifying question. Do NOT recommend on vague queries.

2. **RECOMMEND**: Once you have enough context (role, level, or specific need), recommend 1-10 assessments from the catalog results provided. Every recommendation MUST come from the catalog data — never invent assessments.

3. **REFINE**: If the user changes constraints mid-conversation, UPDATE the existing shortlist. Don't start over.

4. **COMPARE**: If the user asks to compare specific assessments, explain differences using ONLY the catalog data provided.

5. **REFUSE**: If the user asks about salary, legal advice, general HR, or anything unrelated to SHL assessments, politely decline and redirect to assessment selection.

## RESPONSE FORMAT:
You MUST respond with valid JSON in this EXACT format:
```json
{
  "reply": "Your conversational response text here",
  "recommendations": [
    {"name": "Assessment Name", "url": "https://...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## FIELD RULES:
- `reply`: Your natural language response. Be helpful and professional.
- `recommendations`: Empty array `[]` when clarifying or refusing. Array of 1-10 items when recommending.
- Each recommendation MUST have: `name` (exact catalog name), `url` (exact catalog URL), `test_type` (letter codes: A/B/C/D/E/K/P/S)
- `end_of_conversation`: Set `true` ONLY when the user explicitly confirms they're done. Default `false`.

## TEST TYPE CODES:
- A = Ability & Aptitude
- B = Biodata & Situational Judgment
- C = Competencies
- D = Development & 360
- E = Assessment Exercises
- K = Knowledge & Skills
- P = Personality & Behavior
- S = Simulations

## IMPORTANT:
- NEVER hallucinate assessment names or URLs. Only use what's in the catalog data provided.
- Keep responses concise but informative.
- Always respond with valid JSON only. No text before or after the JSON.
"""


def _create_client() -> InferenceClient:
    """Create a HuggingFace InferenceClient."""
    token = HF_TOKEN
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")
    return InferenceClient(api_key=token)


def _extract_json_from_response(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response, handling various formats:
    - Pure JSON
    - JSON wrapped in ```json ... ```
    - JSON with surrounding text
    """
    text = text.strip()

    # Try 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code block
    json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Find JSON object in text (first { to last })
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try 4: Fix common JSON issues (trailing commas, single quotes)
    try:
        # Replace single quotes with double quotes
        fixed = text.replace("'", '"')
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


def _validate_response(parsed: dict) -> dict:
    """
    Validate and normalize the parsed LLM response.
    Ensures it matches the required schema.
    """
    result = {
        "reply": "",
        "recommendations": [],
        "end_of_conversation": False,
    }

    # Extract reply
    result["reply"] = str(parsed.get("reply", parsed.get("response", "")))

    # Extract end_of_conversation
    eoc = parsed.get("end_of_conversation", parsed.get("end", False))
    result["end_of_conversation"] = bool(eoc)

    # Extract and validate recommendations
    recs = parsed.get("recommendations", parsed.get("assessments", []))
    if recs is None:
        recs = []

    validated_recs = []
    for rec in recs:
        if isinstance(rec, dict) and rec.get("name") and rec.get("url"):
            validated_recs.append({
                "name": str(rec["name"]),
                "url": str(rec["url"]),
                "test_type": str(rec.get("test_type", rec.get("test_types", ""))),
            })

    result["recommendations"] = validated_recs[:10]  # Max 10 recommendations

    return result


def _build_fallback_response(error_msg: str = "") -> dict:
    """Build a safe fallback response when LLM fails."""
    return {
        "reply": "I apologize, but I'm having a temporary issue. Could you please rephrase your question about SHL assessments?",
        "recommendations": [],
        "end_of_conversation": False,
    }


def call_llm(
    messages: list[dict],
    catalog_context: str = "",
    max_retries: int = 2,
) -> dict:
    """
    Call the LLM with conversation history and catalog context.
    
    Args:
        messages: Conversation history [{"role": "user/assistant", "content": "..."}]
        catalog_context: Retrieved assessment data to ground the response
        max_retries: Number of retry attempts on failure
    
    Returns:
        dict with keys: reply, recommendations, end_of_conversation
    """
    client = _create_client()

    # Build the full message list
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add catalog context to the latest user message
    if catalog_context and messages:
        # Clone messages to avoid mutating the input
        full_messages.extend(messages[:-1])

        last_msg = messages[-1].copy()
        if last_msg["role"] == "user":
            last_msg["content"] = (
                f"{last_msg['content']}\n\n"
                f"--- CATALOG SEARCH RESULTS (use these for recommendations) ---\n"
                f"{catalog_context}\n"
                f"--- END CATALOG DATA ---\n\n"
                f"Respond with valid JSON only."
            )
        full_messages.append(last_msg)
    else:
        full_messages.extend(messages)
        # Add JSON instruction to last message
        if full_messages and full_messages[-1]["role"] == "user":
            full_messages[-1] = full_messages[-1].copy()
            full_messages[-1]["content"] += "\n\nRespond with valid JSON only."

    # Call the LLM with retries
    for attempt in range(max_retries + 1):
        try:
            start = time.time()

            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=full_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            elapsed = time.time() - start

            if elapsed > TIMEOUT_SECONDS:
                print(f"[LLM] WARNING: Response took {elapsed:.1f}s (limit: {TIMEOUT_SECONDS}s)")

            raw_response = completion.choices[0].message.content
            # print(f"[LLM] Raw response ({elapsed:.1f}s): {raw_response[:200]}...")

            # Parse JSON from response
            parsed = _extract_json_from_response(raw_response)

            if parsed is None:
                print(f"[LLM] Failed to parse JSON (attempt {attempt + 1})")
                if attempt < max_retries:
                    continue
                # Last resort: use the raw text as reply
                return {
                    "reply": raw_response.strip(),
                    "recommendations": [],
                    "end_of_conversation": False,
                }

            # Validate and normalize
            result = _validate_response(parsed)

            if not result["reply"]:
                result["reply"] = "I can help you find the right SHL assessment. What role are you hiring for?"

            return result

        except Exception as e:
            print(f"[LLM] Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            return _build_fallback_response(str(e))

    return _build_fallback_response("Max retries exceeded")


def format_catalog_context(assessments: list) -> str:
    """
    Format retrieved assessments into a text context for the LLM.
    
    Args:
        assessments: List of Assessment objects from the retriever
    
    Returns:
        Formatted string with assessment details
    """
    if not assessments:
        return "No matching assessments found in the catalog."

    lines = []
    for i, a in enumerate(assessments, 1):
        parts = [f"{i}. {a.name}"]
        parts.append(f"   URL: {a.url}")
        parts.append(f"   Test Type: {', '.join(a.test_types)}")

        if a.description:
            # Truncate description to keep context manageable
            desc = a.description[:200] + "..." if len(a.description) > 200 else a.description
            parts.append(f"   Description: {desc}")

        if a.duration:
            parts.append(f"   Duration: {a.duration}")

        if a.job_levels:
            parts.append(f"   Job Levels: {', '.join(a.job_levels)}")

        parts.append(f"   Remote Testing: {'Yes' if a.remote_testing else 'No'}")
        parts.append(f"   Adaptive/IRT: {'Yes' if a.adaptive else 'No'}")

        lines.append("\n".join(parts))

    return "\n\n".join(lines)
