"""
SHL Assessment Recommender — FastAPI Service
==============================================
Stateless conversational API that recommends SHL assessments.

Endpoints:
    GET  /health  → {"status": "ok"}
    POST /chat    → Stateless conversation handler

Request/Response schemas are FIXED per assignment spec.
Each call must respond within 30 seconds.
"""

import time
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import process_turn

# ─── Pydantic Schemas ────────────────────────────────────────────────

class Message(BaseModel):
    """A single conversation message."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message content text")


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.
    Contains the full conversation history (stateless — no server-side session).
    """
    messages: list[Message] = Field(
        ...,
        description="Ordered conversation history",
        min_length=1,
    )


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from SHL catalog")
    url: str = Field(..., description="Canonical SHL catalog URL")
    test_type: str = Field("", description="Test type code(s): A/B/C/D/E/K/P/S")


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.
    Schema is NON-NEGOTIABLE per assignment spec.
    """
    reply: str = Field(..., description="Agent's conversational response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Assessment recommendations (empty when clarifying/refusing)",
    )
    end_of_conversation: bool = Field(
        False,
        description="True only when user confirms they're done",
    )


# ─── Lifespan (startup/shutdown) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models and connections on startup."""
    print("[Server] Starting up...")
    # Eagerly load the retriever and embedding model
    try:
        from app.retrieval import get_retriever
        retriever = get_retriever()
        # Warm up the model with a dummy query
        _ = retriever.search_hybrid("test", top_k=1)
        print("[Server] Retriever and model loaded successfully")
    except Exception as e:
        print(f"[Server] WARNING: Failed to pre-load retriever: {e}")

    yield

    # Shutdown
    print("[Server] Shutting down...")
    try:
        from app.retrieval import get_retriever
        get_retriever().close()
    except Exception:
        pass


# ─── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="AI-powered conversational agent for recommending SHL assessments from their product catalog.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for evaluation/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint. Returns {"status": "ok"} with HTTP 200."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversation handler.
    
    Accepts the full conversation history and returns the agent's response
    with optional assessment recommendations.
    
    The API is STATELESS — all context comes from the messages array.
    Each call must complete within 30 seconds.
    """
    start_time = time.time()

    # Validate messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Validate roles
    for msg in messages:
        if msg["role"] not in ("user", "assistant"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role '{msg['role']}'. Must be 'user' or 'assistant'.",
            )

    # Last message must be from user
    if messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message must be from the user.",
        )

    try:
        # Process the conversation turn
        result = process_turn(messages)

        elapsed = time.time() - start_time

        # Timeout warning (should never happen, but log it)
        if elapsed > 25:
            print(f"[Server] WARNING: Request took {elapsed:.1f}s (limit: 30s)")

        # Build response
        response = ChatResponse(
            reply=result.get("reply", "I can help you find the right SHL assessment."),
            recommendations=[
                Recommendation(**rec) for rec in result.get("recommendations", [])
            ],
            end_of_conversation=result.get("end_of_conversation", False),
        )

        return response

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Server] ERROR after {elapsed:.1f}s: {traceback.format_exc()}")

        # Return a graceful error response instead of 500
        return ChatResponse(
            reply="I apologize, but I encountered an issue processing your request. "
                  "Could you please try rephrasing your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


# ─── Request timing middleware ───────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header and enforce 30s timeout logging."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response
