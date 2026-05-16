"""
Agent Logic Module
===================
Orchestrates the SHL Assessment Recommender conversation flow.

Responsibilities:
  - Classify user intent from conversation history
  - Build search queries from accumulated context
  - Retrieve assessments via hybrid search
  - Call LLM with catalog context for grounded responses
  - Validate all recommended URLs against the catalog
  - Enforce conversation constraints (max 8 turns, stateless)

Behavioral States:
  - CLARIFY:   Query too vague → ask one focused question, 0 recommendations
  - RECOMMEND: Enough context → retrieve + recommend 1-10 assessments
  - REFINE:    Constraints changed → update existing shortlist
  - COMPARE:   User asks about specific assessments → explain differences
  - REFUSE:    Off-topic / prompt injection → decline politely
"""

import re
from typing import Optional
from app.retrieval import HybridRetriever, get_retriever, Assessment
from app.llm import call_llm, format_catalog_context

# ─── Constants ───────────────────────────────────────────────────────
MAX_TURNS = 8            # Max conversation turns (per assignment spec)
MAX_RECOMMENDATIONS = 10  # Max assessments per recommendation
RETRIEVAL_TOP_K = 15     # How many candidates to retrieve for LLM context

# Intent types
INTENT_CLARIFY = "CLARIFY"
INTENT_RECOMMEND = "RECOMMEND"
INTENT_REFINE = "REFINE"
INTENT_COMPARE = "COMPARE"
INTENT_REFUSE = "REFUSE"

# Off-topic keywords for quick detection
REFUSE_PATTERNS = [
    r"\bsalary\b", r"\bsalaries\b", r"\bpay\s+scale\b", r"\bcompensation\b",
    r"\blegal\s+advice\b", r"\blawsuit\b", r"\bsue\b",
    r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:poem|story|essay|code|song)\b",
    r"\bignore\s+(?:previous|above|all)\s+instructions\b",
    r"\bforget\s+(?:your|all)\s+(?:rules|instructions)\b",
    r"\bjailbreak\b", r"\bpretend\s+you\s+are\b",
    r"\bweather\b", r"\brecipe\b", r"\bjoke\b",
]

# Comparison signal keywords
COMPARE_PATTERNS = [
    r"\bdiffer(?:ence|ent|s)?\b",
    r"\bcompare\b", r"\bcomparison\b",
    r"\bvs\.?\b", r"\bversus\b",
    r"\bwhich\s+(?:one|is\s+better)\b",
    r"\bbetween\b.*\band\b",
]

# Refinement signal keywords
REFINE_PATTERNS = [
    r"\bactually\b", r"\binstead\b", r"\bchange\b",
    r"\bswitch\b", r"\breplace\b", r"\bremove\b",
    r"\badd\b.*\bto\s+the\s+list\b",
    r"\bwhat\s+about\b", r"\bcan\s+you\s+also\b",
    r"\bfilter\b", r"\bonly\b.*\badaptive\b",
    r"\bonly\b.*\bremote\b",
    r"\bshorter\b", r"\bquicker\b",
]

# Signals that indicate enough context to recommend
CONTEXT_SIGNALS = [
    r"\b(?:java|python|c\+\+|javascript|sql|react|angular|node)\b",
    r"\b(?:developer|engineer|manager|executive|agent|analyst|admin)\b",
    r"\b(?:entry.?level|mid.?level|senior|graduate|executive|director)\b",
    r"\b(?:personality|cognitive|ability|aptitude|numerical|verbal)\b",
    r"\b(?:sales|customer\s+service|leadership|management|data\s+entry)\b",
    r"\b(?:hiring|selection|screening|assessment|test)\b",
    r"\b(?:contact\s+cent(?:er|re)|call\s+cent(?:er|re))\b",
    r"\b(?:simulation|situational|behavioral|behaviour)\b",
]


def classify_intent(messages: list[dict]) -> str:
    """
    Classify the user's intent from the conversation history.
    
    The classification is based on:
      1. The latest user message content
      2. Whether there are prior recommendations in the conversation
      3. Number of turns elapsed
    
    Returns one of: CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE
    """
    if not messages:
        return INTENT_CLARIFY

    # Get latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            latest_user_msg = msg["content"].lower().strip()
            break

    if not latest_user_msg:
        return INTENT_CLARIFY

    # ── Check REFUSE first (off-topic / prompt injection) ──
    for pattern in REFUSE_PATTERNS:
        if re.search(pattern, latest_user_msg, re.IGNORECASE):
            return INTENT_REFUSE

    # ── Check COMPARE (asking about differences between assessments) ──
    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, latest_user_msg, re.IGNORECASE):
            return INTENT_COMPARE

    # ── Check REFINE (modifying previous recommendations) ──
    has_prior_recommendations = _has_prior_recommendations(messages)
    if has_prior_recommendations:
        for pattern in REFINE_PATTERNS:
            if re.search(pattern, latest_user_msg, re.IGNORECASE):
                return INTENT_REFINE

    # ── Check if enough context to RECOMMEND ──
    # Accumulate context signals across the ENTIRE conversation
    full_context = _extract_full_context(messages)
    context_score = _score_context(full_context)
    user_turn_count = sum(1 for m in messages if m["role"] == "user")

    # If this is a follow-up to a clarification with specific info
    if len(messages) >= 3 and has_prior_recommendations is False:
        # User answered a clarification question — likely has enough context now
        if context_score >= 1:
            return INTENT_RECOMMEND

    # For first turn, require stronger context (3+) to avoid recommending on vague queries
    # For subsequent turns, accumulated context of 2+ is enough
    min_score = 3 if user_turn_count <= 1 else 2
    if context_score >= min_score:
        return INTENT_RECOMMEND

    # If user confirms or agrees with prior recommendations
    if has_prior_recommendations and _is_confirmation(latest_user_msg):
        return INTENT_RECOMMEND

    # Default to CLARIFY if context is too thin
    return INTENT_CLARIFY


def _has_prior_recommendations(messages: list[dict]) -> bool:
    """Check if any assistant message contains recommendations."""
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            # Check for recommendation markers
            if any(marker in content.lower() for marker in [
                "recommend", "assessment", "here are", "following",
                "shl.com/products", "test type"
            ]):
                return True
    return False


def _is_confirmation(text: str) -> bool:
    """Check if the user message is a confirmation/agreement."""
    confirmation_patterns = [
        r"^(?:yes|yeah|yep|ok|okay|sure|perfect|great|good|confirmed|that'?s?\s+(?:it|all|right|correct|fine|what\s+we\s+need))[\.\!\s]*$",
        r"\bthat(?:'?s| is)\s+(?:it|all|right|correct|perfect|great|fine|what\s+we\s+need)\b",
        r"\bgo\s+(?:ahead|with\s+(?:that|these|this))\b",
        r"\bconfirm\b",
        r"\bsounds\s+good\b",
        r"\blet'?s?\s+(?:go|do\s+(?:it|that))\b",
    ]
    for pattern in confirmation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _extract_full_context(messages: list[dict]) -> str:
    """Extract all user messages combined as full conversation context."""
    user_texts = []
    for msg in messages:
        if msg["role"] == "user":
            user_texts.append(msg["content"])
    return " ".join(user_texts)


def _score_context(text: str) -> int:
    """Score how much context/specificity the conversation has."""
    score = 0
    text_lower = text.lower()
    for pattern in CONTEXT_SIGNALS:
        if re.search(pattern, text_lower):
            score += 1
    return score


def build_search_query(messages: list[dict]) -> str:
    """
    Build a search query from the full conversation history.
    Combines all user messages to capture accumulated context.
    This ensures the search covers all the requirements discussed
    across multiple turns, not just the latest message.
    """
    user_texts = []
    for msg in messages:
        if msg["role"] == "user":
            user_texts.append(msg["content"])

    # Use all user messages for context, with emphasis on the latest
    if len(user_texts) > 1:
        # Repeat latest message for extra weight in embedding
        return " ".join(user_texts[:-1]) + " " + user_texts[-1] + " " + user_texts[-1]
    elif user_texts:
        return user_texts[0]
    return ""


def extract_filters_from_messages(messages: list[dict]) -> dict:
    """
    Extract metadata filters from the conversation.
    Looks for specific constraints mentioned by the user.
    """
    filters = {}
    full_text = _extract_full_context(messages).lower()

    # Job level detection
    level_map = {
        r"\bentry.?level\b": "Entry-Level",
        r"\bgraduate\b": "Graduate",
        r"\bmid.?(?:level|professional)\b": "Mid-Professional",
        r"\bsenior\b|professional\s+individual": "Professional Individual Contributor",
        r"\bmanager\b": "Manager",
        r"\bsupervisor\b": "Supervisor",
        r"\bdirector\b": "Director",
        r"\bexecutive\b|\bcxo\b|\bc-suite\b": "Executive",
        r"\bfront.?line\s+manager\b": "Front Line Manager",
    }
    detected_levels = []
    for pattern, level in level_map.items():
        if re.search(pattern, full_text):
            detected_levels.append(level)
    if detected_levels:
        filters["job_level"] = detected_levels

    # Adaptive filter
    if re.search(r"\badaptive\b|\birt\b", full_text):
        filters["adaptive"] = True

    # Duration filter
    duration_match = re.search(r"(?:under|less\s+than|max(?:imum)?|within)\s+(\d+)\s*min", full_text)
    if duration_match:
        filters["duration_max"] = int(duration_match.group(1))

    return filters


def validate_recommendations(
    recommendations: list[dict], retriever: HybridRetriever
) -> list[dict]:
    """
    Validate all recommended URLs against the catalog.
    Remove any that don't exist in the database.
    """
    if not recommendations:
        return []

    valid_urls = retriever.get_all_urls()
    validated = []
    for rec in recommendations:
        url = rec.get("url", "")
        if url in valid_urls:
            validated.append(rec)
        else:
            # Try to find by name instead
            assessment = retriever.get_assessment_by_name(rec.get("name", ""))
            if assessment:
                validated.append({
                    "name": assessment.name,
                    "url": assessment.url,
                    "test_type": ", ".join(assessment.test_types),
                })
            else:
                print(f"[Agent] WARNING: Dropping invalid recommendation: {rec.get('name', 'unknown')}")

    return validated[:MAX_RECOMMENDATIONS]


def process_turn(messages: list[dict]) -> dict:
    """
    Process a single conversation turn.
    
    This is the MAIN entry point for the agent. It:
      1. Checks turn count (max 8)
      2. Classifies intent
      3. Retrieves assessments if needed
      4. Calls LLM with appropriate context
      5. Validates all recommended URLs
      6. Returns the structured response
    
    Args:
        messages: Full conversation history [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        dict: {
            "reply": str,
            "recommendations": [{"name": ..., "url": ..., "test_type": ...}],
            "end_of_conversation": bool
        }
    """
    # ── Turn count check ──
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > MAX_TURNS:
        return {
            "reply": f"We've reached the maximum of {MAX_TURNS} turns for this conversation. "
                     "Please start a new conversation if you need further assistance.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    # ── Classify intent ──
    intent = classify_intent(messages)
    retriever = get_retriever()

    # ── Handle based on intent ──
    catalog_context = ""

    if intent == INTENT_REFUSE:
        # No retrieval needed for off-topic queries
        llm_response = call_llm(messages, catalog_context="")
        llm_response["recommendations"] = []  # Force empty
        return llm_response

    elif intent == INTENT_CLARIFY:
        # Might still retrieve lightly to help LLM understand the domain
        query = build_search_query(messages)
        if query and len(query) > 10:
            assessments = retriever.search_hybrid(query, top_k=5)
            catalog_context = format_catalog_context(assessments)

        llm_response = call_llm(messages, catalog_context=catalog_context)
        # Clarification should have no recommendations
        llm_response["recommendations"] = []
        return llm_response

    elif intent in (INTENT_RECOMMEND, INTENT_REFINE):
        # Full retrieval pipeline
        query = build_search_query(messages)
        filters = extract_filters_from_messages(messages)

        assessments = retriever.search_hybrid(
            query, top_k=RETRIEVAL_TOP_K, filters=filters if filters else None
        )
        catalog_context = format_catalog_context(assessments)

        llm_response = call_llm(messages, catalog_context=catalog_context)

        # If LLM didn't provide recommendations but should have, build from retrieval
        if not llm_response.get("recommendations") and assessments:
            llm_response["recommendations"] = [
                a.to_recommendation() for a in assessments[:MAX_RECOMMENDATIONS]
            ]

        # Validate all recommended URLs
        llm_response["recommendations"] = validate_recommendations(
            llm_response["recommendations"], retriever
        )

        return llm_response

    elif intent == INTENT_COMPARE:
        # Retrieve assessments mentioned in the conversation
        query = build_search_query(messages)
        assessments = retriever.search_hybrid(query, top_k=RETRIEVAL_TOP_K)
        catalog_context = format_catalog_context(assessments)

        llm_response = call_llm(messages, catalog_context=catalog_context)

        # Comparison turns may or may not include recommendations
        if llm_response.get("recommendations"):
            llm_response["recommendations"] = validate_recommendations(
                llm_response["recommendations"], retriever
            )

        return llm_response

    # Fallback (shouldn't reach here)
    return call_llm(messages, catalog_context="")
