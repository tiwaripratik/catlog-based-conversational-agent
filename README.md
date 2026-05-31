# catlog-based-conversational-agent

AI-powered conversational agent that helps hiring managers find the right SHL assessments through natural dialogue. Built as a stateless FastAPI service with hybrid retrieval (BM25 + Vector Cosine Similarity).

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/catlog-based-conversational-agent.git
cd catlog-based-conversational-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN and database credentials

# 3. Setup database (PostgreSQL + pgvector)
python scripts/setup_db.py

# 4. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need a Java programming test for mid-level developers."},
    {"role": "assistant", "content": "What level of Java proficiency?"},
    {"role": "user", "content": "Advanced Java, they'll work on microservices."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are the recommended assessments for advanced Java developers...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Architecture

```
User → FastAPI /chat → Intent Classifier → Hybrid Retriever → LLM (Llama 3.1-8B) → Validated Response
                              ↓                    ↓
                        5 States:            PostgreSQL + pgvector
                        CLARIFY              377 SHL assessments
                        RECOMMEND            384-dim embeddings
                        REFINE               BM25 tsvector
                        COMPARE              IVFFlat index
                        REFUSE
```

### Hybrid Retrieval
- **BM25**: PostgreSQL `tsvector/tsquery` with `ts_rank` for keyword matching
- **Vector**: pgvector cosine similarity with `all-MiniLM-L6-v2` embeddings
- **Fusion**: `score = 0.4 × BM25_normalized + 0.6 × cosine_similarity`

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI endpoints (/health, /chat)
│   ├── agent.py         # Intent classification + conversation orchestration
│   ├── retrieval.py     # Hybrid retriever (BM25 + vector + filters)
│   └── llm.py           # LLM wrapper (Llama 3.1-8B via HuggingFace)
├── scripts/
│   ├── setup_db.py      # Database setup + catalog ingestion
│   └── ingest.py        # Embedding generation + PostgreSQL import
├── tests/
│   └── test_agent.py    # Behavioral probes + Recall@10 evaluation
├── data/
│   └── catalog.json     # Official SHL product catalog (377 assessments)
├── APPROACH.md          # Design choices & evaluation (2-page submission doc)
├── requirements.txt
├── Procfile             # Render deployment
├── render.yaml          # Render blueprint
└── build.sh             # Deployment build script
```

## Evaluation Results

| Metric | Result |
|--------|--------|
| Behavioral Probes | 5/5 passed |
| Schema Compliance | 0 errors (10 traces) |
| URL Validation | 0 hallucinated URLs |
| Mean Recall@10 | 0.3562 |

## Test Type Codes

| Code | Category |
|------|----------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: PostgreSQL 16 + pgvector
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **LLM**: Llama 3.1-8B-Instruct (HuggingFace Inference API)
- **Deployment**: Render.com

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to run tests, format code, and open pull requests. Please follow the project's Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

