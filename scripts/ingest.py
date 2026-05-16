"""
SHL Catalog Ingestion Pipeline
================================
Loads the official SHL product catalog JSON, generates embeddings
using all-MiniLM-L6-v2, creates tsvector for BM25, and inserts
everything into PostgreSQL with pgvector.

Input:  data/catalog.json (official SHL catalog, 377 assessments)
Output: PostgreSQL `assessments` table with embeddings + tsvector

Usage:
    python scripts/ingest.py
"""

import json
import os
import sys
import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────
CATALOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
EMBEDDING_DIM = 384
BATCH_SIZE = 50

# Database connection
DB_CONFIG = {
    "dbname": "shl_recommender",
    "user": "pratik",
    "password": "pratik123",
    "host": "localhost",
    "port": 5432,
}

# Map full key names to single-letter test type codes
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",  # alternate spelling
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",  # alternate spelling
    "Simulations": "S",
}


def load_catalog(filepath):
    """Load the official SHL catalog JSON."""
    print(f"Loading catalog from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    print(f"  Loaded {len(catalog)} assessments")
    
    # Validate
    valid = []
    for item in catalog:
        if not item.get("name") or not item.get("link"):
            print(f"  SKIP: missing name or link: {item}")
            continue
        if item.get("status") != "ok":
            print(f"  SKIP: status={item.get('status')}: {item.get('name')}")
            continue
        valid.append(item)
    
    print(f"  Valid assessments: {len(valid)}")
    return valid


def build_rich_text(assessment):
    """
    Build a rich text representation for embedding generation.
    Combines: name + description + test type categories + job levels + duration
    
    This gives the embedding model maximum semantic context for similarity search.
    """
    parts = []
    
    # Name (most important)
    name = assessment.get("name", "").strip()
    parts.append(name)
    
    # Description (detailed info)
    description = assessment.get("description", "").strip()
    if description:
        parts.append(description)
    
    # Test type categories (e.g., "Knowledge & Skills", "Personality & Behavior")
    keys = assessment.get("keys", [])
    if keys:
        parts.append("Test categories: " + ", ".join(keys))
    
    # Job levels
    job_levels = assessment.get("job_levels", [])
    if job_levels:
        parts.append("Job levels: " + ", ".join(job_levels))
    
    # Duration
    duration = assessment.get("duration", "").strip()
    if duration:
        parts.append(f"Duration: {duration}")
    
    # Remote & Adaptive
    if assessment.get("remote") == "yes":
        parts.append("Supports remote testing.")
    if assessment.get("adaptive") == "yes":
        parts.append("Adaptive/IRT enabled.")
    
    return " | ".join(parts)


def build_bm25_text(assessment):
    """
    Build text for BM25 tsvector generation.
    Uses name + description + keys + job levels for keyword matching.
    """
    parts = []
    
    name = assessment.get("name", "").strip()
    if name:
        parts.append(name)
    
    description = assessment.get("description", "").strip()
    if description:
        parts.append(description)
    
    keys = assessment.get("keys", [])
    if keys:
        parts.append(" ".join(keys))
    
    job_levels = assessment.get("job_levels", [])
    if job_levels:
        parts.append(" ".join(job_levels))
    
    duration = assessment.get("duration", "").strip()
    if duration:
        parts.append(duration)
    
    return " ".join(parts)


def map_keys_to_codes(keys):
    """Map full key names to single-letter test type codes."""
    codes = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code and code not in codes:
            codes.append(code)
    return codes


def generate_embeddings(texts, model):
    """Generate embeddings for a list of texts using sentence-transformers."""
    print(f"  Generating {len(texts)} embeddings with {EMBEDDING_MODEL}...")
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )
    
    elapsed = time.time() - start
    print(f"  Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dtype: {embeddings.dtype}")
    
    return embeddings


def insert_into_db(assessments, embeddings, conn):
    """
    Insert assessments with embeddings and tsvector into PostgreSQL.
    Uses batch insert with execute_values for performance.
    """
    cursor = conn.cursor()
    
    # Clear existing data
    print("  Clearing existing data...")
    cursor.execute("DELETE FROM assessments;")
    cursor.execute("ALTER SEQUENCE assessments_id_seq RESTART WITH 1;")
    
    # Drop IVFFlat index if exists (will rebuild after insert)
    cursor.execute("DROP INDEX IF EXISTS idx_assessments_embedding;")
    
    print(f"  Inserting {len(assessments)} assessments...")
    
    # Prepare batch data
    rows = []
    for i, assessment in enumerate(assessments):
        name = assessment["name"]
        url = assessment["link"]
        
        # Map keys to test type codes
        keys = assessment.get("keys", [])
        test_types = map_keys_to_codes(keys)
        
        description = assessment.get("description", "")
        duration = assessment.get("duration", "")
        job_levels = assessment.get("job_levels", [])
        remote_testing = assessment.get("remote") == "yes"
        adaptive = assessment.get("adaptive") == "yes"
        
        # Convert embedding to PostgreSQL vector format: [0.1, 0.2, ...]
        embedding_list = embeddings[i].tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        # Build BM25 text for tsvector
        bm25_text = build_bm25_text(assessment)
        
        rows.append((
            name,
            url,
            test_types,
            description,
            duration,
            job_levels,
            remote_testing,
            adaptive,
            embedding_str,
            bm25_text,
        ))
    
    # Batch insert using execute_values
    insert_sql = """
        INSERT INTO assessments 
            (name, url, test_types, description, duration, job_levels, 
             remote_testing, adaptive, embedding, search_vector)
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            name = EXCLUDED.name,
            test_types = EXCLUDED.test_types,
            description = EXCLUDED.description,
            duration = EXCLUDED.duration,
            job_levels = EXCLUDED.job_levels,
            remote_testing = EXCLUDED.remote_testing,
            adaptive = EXCLUDED.adaptive,
            embedding = EXCLUDED.embedding,
            search_vector = EXCLUDED.search_vector
    """
    
    # Template with proper casting
    template = """(
        %s, %s, %s, %s, %s, %s, %s, %s, 
        %s::vector,
        to_tsvector('english', %s)
    )"""
    
    execute_values(cursor, insert_sql, rows, template=template, page_size=50)
    conn.commit()
    
    # Verify insertion count
    cursor.execute("SELECT count(*) FROM assessments;")
    count = cursor.fetchone()[0]
    print(f"  Inserted {count} assessments into database")
    
    return count


def rebuild_indexes(conn):
    """Rebuild IVFFlat index after data insertion."""
    cursor = conn.cursor()
    
    # Calculate optimal number of lists for IVFFlat
    cursor.execute("SELECT count(*) FROM assessments;")
    row_count = cursor.fetchone()[0]
    
    # Rule of thumb: lists = sqrt(row_count), minimum 1
    num_lists = max(1, int(row_count ** 0.5))
    print(f"  Building IVFFlat index with lists={num_lists} (for {row_count} rows)...")
    
    start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_assessments_embedding 
        ON assessments USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = {num_lists});
    """)
    conn.commit()
    
    elapsed = time.time() - start
    print(f"  IVFFlat index built in {elapsed:.1f}s")
    
    # Verify all indexes
    cursor.execute("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'assessments';
    """)
    indexes = cursor.fetchall()
    print(f"\n  Active indexes on 'assessments':")
    for name, definition in indexes:
        print(f"    {name}: {definition[:100]}...")


def verify_hybrid_search(conn):
    """Run test queries to verify both BM25 and vector search work."""
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("VERIFICATION: Hybrid Search Tests")
    print("=" * 60)
    
    # Test 1: BM25 search for "Java"
    print("\n--- BM25: 'Java programming' ---")
    cursor.execute("""
        SELECT name, test_types, 
               ts_rank(search_vector, plainto_tsquery('english', 'Java programming')) AS score
        FROM assessments
        WHERE search_vector @@ plainto_tsquery('english', 'Java programming')
        ORDER BY score DESC
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} | types={row[1]} | score={row[2]:.4f}")
    
    # Test 2: BM25 search for "personality"
    print("\n--- BM25: 'personality leadership' ---")
    cursor.execute("""
        SELECT name, test_types,
               ts_rank(search_vector, plainto_tsquery('english', 'personality leadership')) AS score
        FROM assessments
        WHERE search_vector @@ plainto_tsquery('english', 'personality leadership')
        ORDER BY score DESC
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} | types={row[1]} | score={row[2]:.4f}")
    
    # Test 3: Vector search for "hiring a senior Java developer"
    # We need to generate an embedding for the query
    print("\n--- Vector: 'hiring a senior Java developer' ---")
    print("  (requires model to generate query embedding, skipping in ingest)")
    
    # Test 4: Verify data integrity
    print("\n--- Data Integrity ---")
    cursor.execute("SELECT count(*) FROM assessments;")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM assessments WHERE embedding IS NOT NULL;")
    has_embedding = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM assessments WHERE search_vector IS NOT NULL;")
    has_tsvector = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM assessments WHERE description IS NOT NULL AND description != '';")
    has_desc = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT unnest(test_types) AS tt, count(*) 
        FROM assessments 
        GROUP BY tt 
        ORDER BY count(*) DESC;
    """)
    type_dist = cursor.fetchall()
    
    print(f"  Total rows: {total}")
    print(f"  With embeddings: {has_embedding}/{total}")
    print(f"  With tsvector: {has_tsvector}/{total}")
    print(f"  With description: {has_desc}/{total}")
    print(f"\n  Test type distribution:")
    for tt, count in type_dist:
        print(f"    {tt}: {count}")


def main():
    print("=" * 60)
    print("SHL CATALOG INGESTION PIPELINE")
    print(f"Model: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
    print("=" * 60)
    
    # Step 1: Load catalog
    print("\n[STEP 1/5] Loading catalog...")
    catalog = load_catalog(CATALOG_FILE)
    
    if not catalog:
        print("ERROR: No valid assessments found!")
        sys.exit(1)
    
    # Step 2: Build rich text for each assessment
    print("\n[STEP 2/5] Building rich text representations...")
    rich_texts = [build_rich_text(a) for a in catalog]
    
    # Show a few examples
    for i in range(min(3, len(rich_texts))):
        print(f"\n  Example {i+1}: {catalog[i]['name']}")
        print(f"  Rich text: {rich_texts[i][:150]}...")
    
    # Step 3: Generate embeddings
    print(f"\n[STEP 3/5] Loading model and generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = generate_embeddings(rich_texts, model)
    
    assert embeddings.shape == (len(catalog), EMBEDDING_DIM), \
        f"Expected shape ({len(catalog)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    
    # Step 4: Insert into PostgreSQL
    print(f"\n[STEP 4/5] Inserting into PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        count = insert_into_db(catalog, embeddings, conn)
        
        assert count == len(catalog), \
            f"Expected {len(catalog)} rows, got {count}"
        
        # Step 5: Rebuild indexes and verify
        print(f"\n[STEP 5/5] Rebuilding indexes and verifying...")
        rebuild_indexes(conn)
        verify_hybrid_search(conn)
        
        conn.close()
    except psycopg2.Error as e:
        print(f"DATABASE ERROR: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE!")
    print(f"  Assessments: {count}")
    print(f"  Embeddings: {EMBEDDING_DIM}-dim ({EMBEDDING_MODEL})")
    print(f"  BM25: tsvector generated from name + description + keys")
    print(f"  Indexes: IVFFlat (vector) + GIN (tsvector)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
