# SHL Assessment Recommender — Approach Document

## 1. Architecture Overview

The system is a **stateless conversational agent** built as a FastAPI service with two endpoints:
- `GET /health` — returns `{"status": "ok"}`
- `POST /chat` — accepts conversation history, returns grounded assessment recommendations

**Core Pipeline:** User Message → Intent Classification → Hybrid Retrieval → LLM Grounding → URL Validation → Response

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│  FastAPI     │───▶│ Agent Logic  │───▶│ Hybrid Retriever │───▶│ PostgreSQL  │
│  /chat       │    │ (intent +    │    │ BM25 + Vector    │    │ + pgvector  │
│              │    │  context)    │    │ Score Fusion     │    │ 377 items   │
└─────────────┘    └──────┬───────┘    └──────────────────┘    └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Llama 3.1-8B │
                   │ (HuggingFace │
                   │  Inference)  │
                   └──────────────┘
```

## 2. Design Choices

### Data Source
Used the **official SHL product catalog JSON** (377 Individual Test Solutions) instead of web scraping. This ensures data accuracy and completeness.

### Hybrid Retrieval (BM25 + Vector Cosine)
I chose hybrid search because keyword-only search misses semantic intent, and vector-only search misses exact product names:
- **BM25** via PostgreSQL `tsvector/tsquery` with `ts_rank` — strong for exact keyword matches ("OPQ32r", "Java 8")
- **Vector cosine similarity** via `pgvector` with `all-MiniLM-L6-v2` (384-dim) — strong for semantic queries ("hiring managers for leadership roles")
- **Score fusion**: `final = α × normalize(BM25) + (1-α) × cosine`, where α=0.4

Each assessment's embedding is generated from a **rich text** combining: name + description + test categories + job levels + duration + remote/adaptive flags. This gives the embedding model maximum semantic context.

### Agent State Machine
Intent is classified from conversation history using regex pattern matching (not LLM-based, to save latency):
- **CLARIFY**: Vague queries (context score < 3 on turn 1) → ask one focused question, 0 recommendations
- **RECOMMEND**: Sufficient context → retrieve top-15, pass to LLM for grounded selection
- **REFINE**: Prior recommendations exist + change keywords detected → re-retrieve with updated context
- **COMPARE**: Difference/comparison language detected → explain using catalog data
- **REFUSE**: Off-topic, salary, prompt injection → decline politely, 0 recommendations

### LLM Integration
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API (~3-4s latency)
- **Grounding**: Retrieved assessments are injected as structured context in the user message
- **Output**: Forced JSON format (`{reply, recommendations[], end_of_conversation}`)
- **Validation**: Every recommended URL is checked against the catalog database; hallucinated URLs are dropped

### Prompt Design
The system prompt encodes the 5 behavioral rules, test type codes (A/B/C/D/E/K/P/S), the exact JSON schema, and a strict instruction to never hallucinate assessment names. Catalog results are appended to the latest user message as structured text.

## 3. Evaluation Results

| Metric | Result |
|--------|--------|
| Behavioral Probes | **5/5 passed** (clarify, refuse salary, refuse injection, recommend, 8-turn limit) |
| Schema Compliance | **0 errors** across all 10 traces |
| URL Validation | **0 hallucinated URLs** across all traces |
| Mean Recall@10 | **0.3562** |

Per-trace: C4=0.67, C3=0.50, C9=0.43, C5/C7=0.40, C1/C8=0.33, C2/C10=0.25, C6=0.00.

## 4. What Didn't Work & Improvements

1. **Pure vector search** missed exact product names (e.g., "OPQ32r" → low cosine match). Adding BM25 hybrid fixed this for keyword-heavy queries.
2. **LLM sometimes over-clarifies** instead of recommending when context is sufficient. Tuning the intent classifier's context score threshold (raised to 3 for turn-1) helped balance clarification vs. recommendation.
3. **OPQ32r under-recommended**: The LLM frequently omits OPQ32r even when retrieved, because it focuses on more specific assessments. A post-processing rule to boost general-purpose instruments like OPQ32r could improve recall.
4. **C6 scored 0.00**: The expected shortlist included "Graduate Scenarios" which the retriever correctly surfaces, but the LLM chose different assessments. Better few-shot examples in the prompt could address this.

## 5. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Web Framework | FastAPI | Async, auto-docs, Pydantic validation |
| Database | PostgreSQL 16 + pgvector | Hybrid search in one DB, no external vector store |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Fast, lightweight, good semantic quality |
| LLM | Llama 3.1-8B-Instruct (HF) | Free tier, sufficient quality for structured output |
| Deployment | Render.com | Free tier, supports PostgreSQL + Python |

## 6. AI Tools Used
- **Antigravity (Gemini-based coding assistant)**: Used for code generation, debugging, and iterating on the architecture. All code was reviewed and understood before committing.
