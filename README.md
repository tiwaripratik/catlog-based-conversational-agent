# 🎯 SHL Assessment Recommender Agent

> A conversational AI agent that transforms vague hiring intents into grounded SHL assessment shortlists — through natural multi-turn dialogue using **Hybrid Search (BM25 + Vector Cosine Similarity)**.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Deployed on Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

---

## 📌 Problem Statement

Hiring managers often don't know exactly what assessment they need — they know the *role*, not the *test*. Traditional catalog search requires knowing the right keywords upfront, which makes assessment selection slow and shallow.

This agent bridges that gap by:
- **Clarifying** vague queries before recommending
- **Recommending** 1–10 relevant SHL assessments once enough context is gathered
- **Refining** recommendations mid-conversation when constraints change
- **Comparing** specific assessments using actual catalog data (not model hallucinations)
- **Refusing** off-topic queries (salary, legal, general HR advice, prompt injections)

> **Scope:** Only [Individual Test Solutions](https://www.shl.com/solutions/products/product-catalog/) from the SHL catalog. Pre-packaged Job Solutions are out of scope.

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Client
        U[User / Evaluator]
    end

    subgraph FastAPI Service
        H["GET /health"]
        C["POST /chat"]
    end

    subgraph Agent Core
        CL[Classify Intent]
        Q[Ask Clarifying Question]
        R[Hybrid Retrieval]
        RE[Rank & Recommend]
        RF[Refine Shortlist]
        CMP[Compare Assessments]
        REF[Refuse Off-topic]
    end

    subgraph Hybrid Search Layer
        BM25["BM25 (tsvector/tsquery)"]
        VS["Vector Search (pgvector cosine)"]
        MERGE["Score Fusion: α·BM25 + (1-α)·Cosine"]
    end

    subgraph Data Layer
        PG[(PostgreSQL + pgvector)]
        EMB[Sentence Transformer Embeddings]
    end

    subgraph LLM
        HF[Llama-3.1-8B-Instruct via HuggingFace]
    end

    U -->|Conversation History| C
    U --> H
    C --> CL
    CL -->|Vague| Q
    CL -->|Enough Context| R
    CL -->|Edit Request| RF
    CL -->|Comparison| CMP
    CL -->|Off-topic| REF
    R --> BM25 --> PG
    R --> VS --> EMB --> PG
    BM25 --> MERGE
    VS --> MERGE
    MERGE -->|Top-K Results| RE
    RE --> HF
    RF --> HF
    CMP --> PG
    Q --> HF
    HF -->|Structured Response| C
    C -->|JSON Reply + Recommendations| U
```

---

## 🔍 Hybrid Search: BM25 + Vector Cosine Similarity

Traditional keyword search (BM25) misses semantic meaning. Pure vector search misses exact keyword matches. We combine **both** for the best of both worlds.

### How It Works

| Component | Method | Strength |
|---|---|---|
| **BM25** | PostgreSQL `tsvector` + `ts_rank` | Exact keyword matching — "Java", "OPQ32r", "personality" |
| **Vector Search** | pgvector cosine similarity (384-dim) | Semantic understanding — "team collaboration" → finds "interpersonal skills" |
| **Hybrid Score** | `α × BM25_norm + (1-α) × cosine_score` | Best of both worlds |

### Score Fusion Formula

```
final_score = α × normalize(bm25_score) + (1 - α) × cosine_similarity_score
```

- **α = 0.4** (default): balanced between keyword and semantic matching
- BM25 scores are normalized to [0, 1] range using min-max normalization
- Cosine similarity already in [0, 1] range

### Database Schema (Hybrid Search Ready)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE assessments (
    id               SERIAL PRIMARY KEY,
    name             TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    test_types       TEXT[],
    description      TEXT,
    duration         TEXT,
    job_levels       TEXT[],
    remote_testing   BOOLEAN DEFAULT FALSE,
    adaptive         BOOLEAN DEFAULT FALSE,
    embedding        vector(384),
    search_vector    tsvector  -- For BM25 full-text search
);

-- Vector search index (cosine similarity)
CREATE INDEX ON assessments
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

-- BM25 full-text search index
CREATE INDEX ON assessments
USING GIN (search_vector);
```

---

## ⚙️ Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace | Free tier, strong instruction following, fast enough for 30s timeout |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, 384-dim, runs locally, great for semantic similarity |
| **Vector Store** | PostgreSQL + pgvector (Neon.tech) | Persistent, scalable, SQL filtering, supports both vector + BM25 |
| **BM25 Search** | PostgreSQL `tsvector` + `ts_rank` | Native full-text search, no extra dependencies |
| **API Framework** | FastAPI | Async, auto-generated docs, Pydantic validation, production-ready |
| **Deployment** | Render.com | Free tier, GitHub auto-deploy, supports cold start (2 min allowed) |

---

## 📂 Project Structure

```
shl-recommender/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — /health + /chat endpoints
│   ├── agent.py             # Core agent logic — intent classification, state machine
│   ├── retrieval.py         # Hybrid search: BM25 + pgvector cosine similarity
│   └── llm.py               # HuggingFace Inference API wrapper
├── scripts/
│   ├── scrape_catalog.py    # One-time SHL catalog scraper (Individual Tests only)
│   └── ingest.py            # Embed + tsvector → insert into PostgreSQL
├── data/
│   └── catalog.json         # Scraped SHL product catalog
├── tests/
│   └── test_agent.py        # Replay tests against public conversation traces
├── .env                     # API keys (never committed)
├── .gitignore
├── requirements.txt
├── render.yaml              # Render deployment config
├── TODO.md                  # Task tracker
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone & Install

```bash
git clone https://github.com/your-username/shl-recommender.git
cd shl-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token
DATABASE_URL=postgresql://user:pass@host:5432/shl_recommender
```

| Variable | Source |
|---|---|
| `HF_TOKEN` | [HuggingFace Settings → Tokens](https://huggingface.co/settings/tokens) |
| `DATABASE_URL` | [Neon.tech](https://neon.tech) (production) or local PostgreSQL (dev) |

### 3. Database Setup

Run this SQL once in your PostgreSQL console:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS assessments (
    id               SERIAL PRIMARY KEY,
    name             TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    test_types       TEXT[],
    description      TEXT,
    duration         TEXT,
    job_levels       TEXT[],
    remote_testing   BOOLEAN DEFAULT FALSE,
    adaptive         BOOLEAN DEFAULT FALSE,
    embedding        vector(384),
    search_vector    tsvector
);

-- IVFFlat index for vector cosine similarity search
CREATE INDEX ON assessments
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

-- GIN index for BM25 full-text search
CREATE INDEX ON assessments
USING GIN (search_vector);
```

### 4. Scrape the SHL Catalog

```bash
python scripts/scrape_catalog.py
# Output: data/catalog.json (~384 Individual Test Solutions)
```

> Only **Individual Test Solutions** are scraped. Pre-packaged Job Solutions are excluded per assignment scope.

### 5. Embed & Ingest

```bash
python scripts/ingest.py
# Generates embeddings + tsvector, inserts into PostgreSQL
```

### 6. Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Quick test:

```bash
# Health check
curl http://localhost:8000/health

# Start a conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"}
    ]
  }'
```

---

## 📡 API Specification

> ⚠️ **The API schema is non-negotiable.** Deviating from this schema will break the automated evaluator and result in a zero score.

### `GET /health`

Returns service readiness. Cold-start hosting services get up to **2 minutes** for this endpoint.

```json
{"status": "ok"}
```

### `POST /chat`

**Stateless** — send the full conversation history on every call. The service stores no per-conversation state.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Response field rules:**

| Field | When |
|---|---|
| `recommendations: []` | Agent is clarifying, gathering context, or refusing |
| `recommendations: [1–10 items]` | Agent has committed to a shortlist |
| `end_of_conversation: true` | Agent considers the task complete |
| `end_of_conversation: false` | Conversation is still active |

**Constraints:**
- Max **8 turns** per conversation (user + assistant combined)
- Each API call must respond within **30 seconds**
- All recommended URLs must come from the scraped catalog (no hallucinated links)

---

## 🤖 Agent Behavior

The agent handles five distinct conversational intents:

### 1. Clarify → Ask before acting
When the query is too vague to recommend, the agent asks **one focused clarifying question**.

```
User:  "I need an assessment"
Agent: "Sure! What role are you hiring for, and what level of seniority?"
```

### 2. Recommend → Grounded shortlist
Once enough context is gathered, uses **hybrid search** to return 1–10 assessments with names and catalog URLs.

```
User:  "Senior backend engineer, Python, 6 years experience"
Agent: "Here are 4 assessments that fit..." + [shortlist with URLs]
```

### 3. Refine → Update, don't restart
Mid-conversation constraint changes modify the existing shortlist.

```
User:  "Actually, also add a personality test"
Agent: "Updated your list to include OPQ32r..." + [updated shortlist]
```

### 4. Compare → Catalog-grounded answers
Comparison questions are answered using **catalog data**, not the LLM's prior knowledge.

```
User:  "What's the difference between OPQ and VERIFY?"
Agent: "Based on the catalog data: OPQ measures personality across 32 dimensions, while VERIFY..."
```

### 5. Refuse → Stay in scope
Off-topic, harmful, or prompt-injection attempts are declined politely. The agent redirects to assessment selection.

```
User:  "What salary should I offer?"
Agent: "I can only help with SHL assessment selection. What role are you assessing for?"
```

---

## 📊 Evaluation Criteria

The submission is scored by an **automated replay harness** that simulates realistic users against your `/chat` endpoint.

### Scoring Breakdown

| Component | Type | Description |
|---|---|---|
| **Hard Evals** | Must pass | Schema compliance on every response, catalog-only URLs, 8-turn cap honored |
| **Recall@10** | Metric | Mean Recall@10 across all conversation traces (public + holdout) |
| **Behavior Probes** | Pass/Fail | Binary assertions: refuses off-topic, doesn't recommend on turn 1 for vague queries, honors edits, % hallucinations |

### Recall@K Definition

```
Recall@K = (# relevant assessments in top K) / (Total relevant assessments for the query)
Mean Recall@K = (1/N) × Σ Recall@K_i
```

### Test Type Codes

| Code | Category |
|---|---|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgement |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |

### Public Conversation Traces

10 public conversation traces are provided for development. Each trace contains:
- A **persona** with a fact set
- A **labeled expected shortlist**

```bash
python tests/test_agent.py   # Run public trace tests locally
```

---

## 🌐 Deployment (Render)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set environment variables: `HF_TOKEN`, `DATABASE_URL`
5. **Build command:** `pip install -r requirements.txt`
6. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Your public endpoint: `https://shl-recommender.onrender.com`

> **Note:** Render free tier sleeps after inactivity. The evaluator allows up to 2 minutes for `/health` cold-start wake-up.

---

## 📦 Requirements

```txt
fastapi
uvicorn
psycopg2-binary
sentence-transformers
huggingface-hub
numpy
requests
beautifulsoup4
python-dotenv
pydantic
```

---

## 🔒 Security

- **Never** commit `.env` to Git
- Add `.env` and `data/catalog.json` to `.gitignore`
- Never hardcode API keys in source files
- All recommended URLs are validated against the scraped catalog before returning
- Prompt injection attempts are caught and refused by the agent

---

## 📝 Submission Checklist

- [ ] `/health` endpoint returns `{"status": "ok"}` with HTTP 200
- [ ] `/chat` endpoint returns correct JSON schema on every response
- [ ] All recommended URLs exist in the scraped SHL catalog
- [ ] Hybrid search (BM25 + vector) returns relevant results
- [ ] Agent clarifies vague queries (doesn't recommend on turn 1)
- [ ] Agent refines shortlist on constraint changes
- [ ] Agent compares assessments using catalog data
- [ ] Agent refuses off-topic queries
- [ ] Conversations complete within 8 turns
- [ ] Each API call responds within 30 seconds
- [ ] Public conversation traces pass with good Recall@10
- [ ] Approach document (2 pages max) covering design, retrieval, prompts, evaluation
- [ ] Deployed endpoint URL submitted via the form

---

## 📄 Approach Document

> *Submit a 2-page document covering:*
> - Design choices and architecture rationale
> - **Hybrid retrieval setup**: why BM25 + vector cosine, how scores are fused
> - Prompt design and agent state management
> - Evaluation approach and what didn't work
> - AI tools used (if any) and what they were used for

---

## 👤 Author

Built as a take-home assignment for the **SHL Labs AI Intern** role.
