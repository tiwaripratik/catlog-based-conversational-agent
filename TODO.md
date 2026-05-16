# SHL Assessment Recommender — TODO

> PDF Assignment Requirements Tracker  
> Hybrid Search: BM25 (keyword) + Vector Cosine Similarity (semantic)

---

## Phase 1: Database Setup (PostgreSQL + pgvector) ✅
- [x] Install pgvector extension (already available v0.8.2)
- [x] Create PostgreSQL user `pratik`
- [x] Create database `shl_recommender`
- [x] Enable `vector` extension
- [x] Create `assessments` table with `vector(384)` column
- [x] Add `tsvector` column (`search_vector`) for BM25 full-text search
- [x] Create GIN index on tsvector column (for BM25) → `idx_assessments_search_vector`
- [x] Create IVFFlat index on embedding column (tested, will rebuild after data ingestion)
- [x] Verify hybrid search setup works (BM25 + vector + hybrid score fusion all tested)

---

## Phase 2: SHL Catalog Data ✅
- [x] ~~Write `scripts/scrape_catalog.py`~~ → Used **official SHL catalog JSON** instead
- [x] Downloaded from `https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json`
- [x] **377 Individual Test Solutions** loaded (official data, not scraped)
- [x] Fields: entity_id, name, link, description, duration, job_levels, remote, adaptive, keys
- [x] Test type distribution: K=240, P=67, S=43, A=32, C=19, B=17, D=7, E=2
- [x] All 377 have descriptions, valid URLs, remote=yes
- [x] 37/377 are adaptive
- [x] Saved to `data/catalog.json`
- [x] 10 sample conversations available at `sample_conversations/GenAI_SampleConversations/C1-C10.md`

---

## Phase 3: Embedding & Ingestion ✅
- [x] Write `scripts/ingest.py`
- [x] Load `data/catalog.json` (377 assessments)
- [x] Create rich text per assessment: name + description + keys + job_levels + duration + remote/adaptive
- [x] Generate 384-dim embeddings using `all-MiniLM-L6-v2` (1.2s for 377 items)
- [x] Generate `tsvector` for BM25 from name + description + keys + job_levels
- [x] Batch insert into PostgreSQL (pgvector + tsvector) — 377/377 rows
- [x] Rebuild IVFFlat index after data insertion (lists=19)
- [x] Verify: `SELECT count(*) FROM assessments;` = 377 ✅
- [x] Verify vector search: "Java developer" → Java 8 (New) ✅
- [x] Verify BM25 search: "personality leadership" → OPQ Leadership Report ✅
- [x] Verify hybrid search works end-to-end ✅

---

## Phase 4: Hybrid Retrieval (BM25 + Vector Search) ✅
- [x] Write `app/retrieval.py`
- [x] Implement **BM25 search** using PostgreSQL `ts_rank` + `tsvector/tsquery`
- [x] Implement **Vector search** using pgvector cosine similarity
- [x] Implement **Hybrid scoring**: `final_score = α × bm25_score + (1-α) × cosine_score`
- [x] Tunable weight parameter α (default: 0.4 BM25, 0.6 vector)
- [x] Optional metadata filtering (test_type, job_level, remote_testing, adaptive, duration_max)
- [x] Return top-K assessments with all fields
- [x] Assessment lookup by name and URL
- [x] URL validation against catalog
- [x] `to_recommendation()` format matching API schema
- [x] Test: BM25 "personality questionnaire" → OPQ32r ✅
- [x] Test: Vector "Java developer" → Java 8 (New) ✅
- [x] Test: Hybrid "numerical reasoning for graduates" → Verify - Numerical Ability ✅
- [x] Test: Filter(adaptive=True) → Only adaptive tests returned ✅
- [x] Test: Filter(Executive) → Only executive-level tests returned ✅
- [x] Test: URL validation catches fake URLs ✅

---

## Phase 5: LLM Integration ✅
- [x] Write `app/llm.py`
- [x] HuggingFace Inference API wrapper for `meta-llama/Llama-3.1-8B-Instruct:novita`
- [x] System prompt with agent persona & rules (Clarify/Recommend/Refine/Compare/Refuse)
- [x] JSON output parsing — handles pure JSON, markdown blocks, embedded JSON
- [x] Response validation against required schema (reply, recommendations, end_of_conversation)
- [x] Timeout handling (25s buffer within 30s limit) — actual response: ~3.4s
- [x] Retry logic with error recovery and fallback response
- [x] `format_catalog_context()` — formats retrieved assessments for LLM context
- [x] Test: Clarification on vague query → "What role?" ✅
- [x] Test: Refuse off-topic → "I don't have salary info" ✅
- [x] Test: Multi-turn → Turn 1 clarify, Turn 2 recommend 4 assessments ✅
- [x] Test: JSON parsing edge cases → 3/3 passed ✅

---

## Phase 6: Agent Logic ✅
- [x] Write `app/agent.py`
- [x] Intent classifier (from conversation history):
  - [x] CLARIFY — query too vague, ask one focused question (0 recs)
  - [x] RECOMMEND — enough context, retrieve + return 1-10 assessments
  - [x] REFINE — user changed constraints, update shortlist
  - [x] COMPARE — user asks about specific assessments, use catalog data
  - [x] REFUSE — off-topic / prompt injection / salary / weather
- [x] Context extraction from full conversation history (stateless API)
- [x] Filter extraction (job_level, adaptive, duration_max) from messages
- [x] Search query built from accumulated user messages across all turns
- [x] All recommended URLs validated against catalog (name fallback if URL invalid)
- [x] Max 8 turns per conversation enforced → `end_of_conversation: true`
- [x] Turn 1 conservative — vague queries always clarify first
- [x] Test: Intent classification 8/8 cases correct ✅
- [x] Test: Turn 1 "senior leadership" → CLARIFY, 0 recs ✅
- [x] Test: Turn 2 "CXOs, personality" → RECOMMEND, 5 validated recs ✅
- [x] Test: Off-topic → REFUSE, 0 recs ✅
- [x] Test: Max turns → end_of_conversation=true ✅

---

## Phase 7: FastAPI Service ✅
- [x] Write `app/main.py`
- [x] `GET /health` → `{"status": "ok"}` (HTTP 200) ✅
- [x] `POST /chat` → stateless conversation handler ✅
- [x] Request schema: `{"messages": [{"role": "user/assistant", "content": "..."}]}`
- [x] Response schema: `{"reply": "...", "recommendations": [...], "end_of_conversation": bool}`
- [x] `recommendations` = `[]` when clarifying or refusing ✅
- [x] `recommendations` = 1-10 items when recommending ✅
- [x] Each recommendation has: `name`, `url`, `test_type` ✅
- [x] Input validation: empty messages → 422 ✅
- [x] CORS middleware enabled for evaluation ✅
- [x] Model pre-loaded on startup for fast first response ✅
- [x] Response time: health=0.002s, chat<5s ✅
- [x] Graceful error handling (no 500s) ✅

---

## Phase 8: Config & Environment ✅
- [x] Write `requirements.txt` — 11 deps with minimum version pins
- [x] Write `.env` — HF_TOKEN, DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- [x] Write `.env.example` — safe template with placeholders (committed to repo)
- [x] Write `.gitignore` — .env, __pycache__, venv, *.safetensors, logs
- [x] Write `render.yaml` — Render blueprint: web service + PostgreSQL 16

---

## Phase 9: Testing & Evaluation ✅
- [x] Write `tests/test_agent.py`
- [x] Ground-truth shortlists extracted from all 10 sample conversations
- [x] Behavioral probes: **5/5 passed**
  - [x] Clarify vague queries (0 recs) ✅
  - [x] Refuse off-topic (salary → 0 recs) ✅
  - [x] Refuse prompt injection ✅
  - [x] Recommend on specific query (>0 recs) ✅
  - [x] 8-turn limit (end_of_conversation=true) ✅
- [x] Schema compliance: **0 errors** across all 10 traces ✅
- [x] URL validation: **0 invalid URLs** across all traces ✅
- [x] Per-trace Recall@10: C4=0.67, C3=0.50, C9=0.43, C5/C7=0.40, C1/C8=0.33, C2/C10=0.25, C6=0.00
- [x] **Mean Recall@10: 0.3562**

---

## Phase 10: Deployment ✅ (local ready, needs GitHub push)
- [x] Initialize git repo with clean `.gitignore`
- [x] Committed: 29 files, 15,000+ lines
- [x] SSH key generated for GitHub push
- [x] `build.sh` — Render build script (pip install + DB setup)
- [x] `scripts/setup_db.py` — creates tables + ingests catalog with embeddings
- [x] `Procfile` — `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- [x] `render.yaml` — Render blueprint (web service + PostgreSQL 16)
- [x] Local verification: `/health` = ✅, `/chat` = ✅ (4.1s response)
- [ ] **ACTION NEEDED: Add SSH key to GitHub → Create repo → Push**
- [ ] **ACTION NEEDED: Deploy on Render.com → Set HF_TOKEN env var**
- [ ] Verify deployed `/health` and `/chat` endpoints

---

## Phase 11: Submission Materials ✅
- [x] `APPROACH.md` — 2-page approach document:
  - [x] Architecture overview with pipeline diagram
  - [x] Hybrid retrieval design (BM25 + vector) & rationale
  - [x] Agent state machine (Clarify/Recommend/Refine/Compare/Refuse)
  - [x] Prompt design & LLM grounding strategy
  - [x] Evaluation results: 5/5 probes, 0 schema errors, Mean Recall@10 = 0.3562
  - [x] What didn't work & improvement ideas
  - [x] AI tools disclosure
- [x] `README.md` — production-quality with API examples, architecture, project structure
- [x] All commits pushed to git (3 commits, 30+ files)

---

## Evaluation Criteria (from PDF)

| Component | Type | What's Checked |
|---|---|---|
| **Hard Evals** | Must Pass | Schema compliance, catalog-only URLs, 8-turn cap |
| **Recall@10** | Metric | Mean Recall@10 across public + holdout traces |
| **Behavior Probes** | Pass/Fail | Refuses off-topic, no turn-1 recommend for vague, honors edits, % hallucinations |

### Recall@K Formula
```
Recall@K = (# relevant assessments in top K) / (Total relevant assessments)
Mean Recall@K = (1/N) × Σ Recall@K_i
```

---

## Test Type Codes
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
