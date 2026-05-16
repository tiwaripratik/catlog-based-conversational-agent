"""
Database Setup & Ingestion Script for Deployment
==================================================
Run this ONCE after deploying to set up the PostgreSQL database
with pgvector, create tables, and ingest the SHL catalog.

Usage:
    DATABASE_URL=postgresql://... python scripts/setup_db.py
    
    Or with individual vars:
    DB_NAME=... DB_USER=... DB_PASSWORD=... DB_HOST=... python scripts/setup_db.py
"""

import json
import os
import sys
import time
import psycopg2
from psycopg2.extras import execute_values

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CATALOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME", "shl_recommender"),
    "user": os.environ.get("DB_USER", "pratik"),
    "password": os.environ.get("DB_PASSWORD", "pratik123"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
}

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}


def get_connection():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(**DB_CONFIG)


def setup_extensions(conn):
    """Enable pgvector extension."""
    cursor = conn.cursor()
    print("[Setup] Enabling pgvector extension...")
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("[Setup] pgvector enabled ✅")
    except Exception as e:
        conn.rollback()
        print(f"[Setup] WARNING: Could not enable pgvector: {e}")
        print("[Setup] Trying without extension (may already exist)...")


def create_tables(conn):
    """Create the assessments table."""
    cursor = conn.cursor()
    print("[Setup] Creating tables...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id              SERIAL PRIMARY KEY,
            name            TEXT NOT NULL,
            url             TEXT UNIQUE NOT NULL,
            test_types      TEXT[] DEFAULT '{}',
            description     TEXT DEFAULT '',
            duration        TEXT DEFAULT '',
            job_levels      TEXT[] DEFAULT '{}',
            remote_testing  BOOLEAN DEFAULT FALSE,
            adaptive        BOOLEAN DEFAULT FALSE,
            embedding       vector(384),
            search_vector   tsvector
        );
    """)

    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_assessments_search_vector 
        ON assessments USING gin(search_vector);
    """)

    conn.commit()
    print("[Setup] Tables created ✅")


def ingest_catalog(conn):
    """Load catalog and generate embeddings."""
    from sentence_transformers import SentenceTransformer

    print("[Setup] Loading catalog...")
    with open(CATALOG_FILE, "r") as f:
        catalog = json.load(f)

    # Filter valid
    catalog = [a for a in catalog if a.get("name") and a.get("link") and a.get("status") == "ok"]
    print(f"[Setup] {len(catalog)} valid assessments")

    # Check if already ingested
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM assessments;")
    existing = cursor.fetchone()[0]
    if existing >= len(catalog):
        print(f"[Setup] Already ingested ({existing} rows). Skipping.")
        return existing

    # Generate embeddings
    print("[Setup] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    rich_texts = []
    for a in catalog:
        parts = [a.get("name", "")]
        if a.get("description"):
            parts.append(a["description"])
        if a.get("keys"):
            parts.append("Test categories: " + ", ".join(a["keys"]))
        if a.get("job_levels"):
            parts.append("Job levels: " + ", ".join(a["job_levels"]))
        if a.get("duration"):
            parts.append(f"Duration: {a['duration']}")
        if a.get("remote") == "yes":
            parts.append("Supports remote testing.")
        if a.get("adaptive") == "yes":
            parts.append("Adaptive/IRT enabled.")
        rich_texts.append(" | ".join(parts))

    print("[Setup] Generating embeddings...")
    embeddings = model.encode(rich_texts, batch_size=50, show_progress_bar=True, normalize_embeddings=True)

    # Clear and insert
    print("[Setup] Inserting into database...")
    cursor.execute("DELETE FROM assessments;")
    cursor.execute("ALTER SEQUENCE assessments_id_seq RESTART WITH 1;")

    rows = []
    for i, a in enumerate(catalog):
        keys = a.get("keys", [])
        test_types = []
        for k in keys:
            code = KEY_TO_CODE.get(k)
            if code and code not in test_types:
                test_types.append(code)

        bm25_parts = [a.get("name", ""), a.get("description", "")]
        bm25_parts.extend(a.get("keys", []))
        bm25_parts.extend(a.get("job_levels", []))
        bm25_text = " ".join(bm25_parts)

        emb_str = "[" + ",".join(str(float(x)) for x in embeddings[i]) + "]"

        rows.append((
            a["name"], a["link"], test_types, a.get("description", ""),
            a.get("duration", ""), a.get("job_levels", []),
            a.get("remote") == "yes", a.get("adaptive") == "yes",
            emb_str, bm25_text,
        ))

    insert_sql = """
        INSERT INTO assessments 
            (name, url, test_types, description, duration, job_levels,
             remote_testing, adaptive, embedding, search_vector)
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            name = EXCLUDED.name, test_types = EXCLUDED.test_types,
            description = EXCLUDED.description, duration = EXCLUDED.duration,
            job_levels = EXCLUDED.job_levels, remote_testing = EXCLUDED.remote_testing,
            adaptive = EXCLUDED.adaptive, embedding = EXCLUDED.embedding,
            search_vector = EXCLUDED.search_vector
    """
    template = """(
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s::vector, to_tsvector('english', %s)
    )"""

    execute_values(cursor, insert_sql, rows, template=template, page_size=50)
    conn.commit()

    # Build IVFFlat index
    cursor.execute("DROP INDEX IF EXISTS idx_assessments_embedding;")
    num_lists = max(1, int(len(catalog) ** 0.5))
    cursor.execute(f"""
        CREATE INDEX idx_assessments_embedding 
        ON assessments USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = {num_lists});
    """)
    conn.commit()

    cursor.execute("SELECT count(*) FROM assessments;")
    count = cursor.fetchone()[0]
    print(f"[Setup] Ingested {count} assessments ✅")
    return count


def main():
    print("=" * 60)
    print("SHL RECOMMENDER — DATABASE SETUP & INGESTION")
    print("=" * 60)

    conn = get_connection()
    setup_extensions(conn)
    create_tables(conn)
    count = ingest_catalog(conn)
    conn.close()

    print(f"\nSetup complete! {count} assessments in database.")


if __name__ == "__main__":
    main()
