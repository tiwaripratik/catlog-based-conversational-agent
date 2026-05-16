"""
Hybrid Retrieval Module (BM25 + Vector Cosine Similarity)
==========================================================
Implements hybrid search combining:
  - BM25 full-text search (PostgreSQL tsvector/tsquery + ts_rank)
  - Vector cosine similarity search (pgvector)
  - Score fusion: final = α × BM25_normalized + (1-α) × cosine_score

All recommended URLs are validated against the catalog stored in PostgreSQL.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_ALPHA = 0.4  # Weight for BM25 in hybrid score (0.4 BM25, 0.6 vector)
DEFAULT_TOP_K = 10

DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME", "shl_recommender"),
    "user": os.environ.get("DB_USER", "pratik"),
    "password": os.environ.get("DB_PASSWORD", "pratik123"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
}

# Try DATABASE_URL first (for Render deployment)
DATABASE_URL = os.environ.get("DATABASE_URL")


@dataclass
class Assessment:
    """Represents an SHL assessment from the catalog."""
    id: int
    name: str
    url: str
    test_types: list[str] = field(default_factory=list)
    description: str = ""
    duration: str = ""
    job_levels: list[str] = field(default_factory=list)
    remote_testing: bool = False
    adaptive: bool = False
    score: float = 0.0
    bm25_score: float = 0.0
    cosine_score: float = 0.0

    def to_recommendation(self) -> dict:
        """Convert to the API response recommendation format."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": ", ".join(self.test_types) if self.test_types else "",
        }

    def to_detail_dict(self) -> dict:
        """Full detail representation for comparison/context."""
        return {
            "name": self.name,
            "url": self.url,
            "test_types": self.test_types,
            "description": self.description,
            "duration": self.duration,
            "job_levels": self.job_levels,
            "remote_testing": self.remote_testing,
            "adaptive": self.adaptive,
        }


class HybridRetriever:
    """
    Hybrid retrieval engine combining BM25 and vector cosine similarity.
    
    Score fusion formula:
        final_score = α × normalize(bm25_score) + (1 - α) × cosine_similarity
    
    Where:
        - α = 0.4 (default, tunable)
        - BM25 scores normalized to [0, 1] via min-max normalization
        - Cosine similarity already in [0, 1] after normalized embeddings
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        self._model = None
        self._conn = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            print("[Retriever] Loading embedding model...")
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[Retriever] Model loaded: {EMBEDDING_MODEL_NAME}")
        return self._model

    def _get_connection(self):
        """Get a database connection, creating one if needed."""
        if self._conn is None or self._conn.closed:
            if DATABASE_URL:
                self._conn = psycopg2.connect(DATABASE_URL)
            else:
                self._conn = psycopg2.connect(**DB_CONFIG)
        return self._conn

    def _encode_query(self, query: str) -> str:
        """Encode a text query into a pgvector-formatted string."""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return "[" + ",".join(str(float(x)) for x in embedding) + "]"

    def _build_filter_clause(self, filters: Optional[dict]) -> tuple[str, list]:
        """
        Build SQL WHERE clause from filter dictionary.
        
        Supported filters:
            - test_type: str or list[str] — filter by test type code(s)
            - job_level: str or list[str] — filter by job level(s)
            - remote_testing: bool — only remote-capable assessments
            - adaptive: bool — only adaptive/IRT assessments
            - duration_max: int — max duration in minutes
        """
        if not filters:
            return "", []

        clauses = []
        params = []

        # Test type filter
        if "test_type" in filters:
            tt = filters["test_type"]
            if isinstance(tt, str):
                tt = [tt]
            clauses.append("test_types && %s")
            params.append(tt)

        # Job level filter
        if "job_level" in filters:
            jl = filters["job_level"]
            if isinstance(jl, str):
                jl = [jl]
            clauses.append("job_levels && %s")
            params.append(jl)

        # Remote testing filter
        if "remote_testing" in filters:
            clauses.append("remote_testing = %s")
            params.append(filters["remote_testing"])

        # Adaptive filter
        if "adaptive" in filters:
            clauses.append("adaptive = %s")
            params.append(filters["adaptive"])

        # Duration filter (extract number from "30 minutes")
        if "duration_max" in filters:
            clauses.append("""
                duration != '' AND 
                CAST(REGEXP_REPLACE(duration, '[^0-9]', '', 'g') AS INTEGER) <= %s
            """)
            params.append(filters["duration_max"])

        if clauses:
            return "AND " + " AND ".join(clauses), params
        return "", []

    def search_bm25(
        self, query: str, top_k: int = DEFAULT_TOP_K,
        filters: Optional[dict] = None
    ) -> list[Assessment]:
        """
        Pure BM25 full-text search using PostgreSQL tsvector/tsquery.
        Uses ts_rank for scoring.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        filter_clause, filter_params = self._build_filter_clause(filters)

        sql = f"""
            SELECT id, name, url, test_types, description, duration,
                   job_levels, remote_testing, adaptive,
                   ts_rank(search_vector, plainto_tsquery('english', %s)) AS bm25_score
            FROM assessments
            WHERE search_vector @@ plainto_tsquery('english', %s)
            {filter_clause}
            ORDER BY bm25_score DESC
            LIMIT %s
        """

        params = [query, query] + filter_params + [top_k]
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(Assessment(
                id=row["id"],
                name=row["name"],
                url=row["url"],
                test_types=row["test_types"] or [],
                description=row["description"] or "",
                duration=row["duration"] or "",
                job_levels=row["job_levels"] or [],
                remote_testing=row["remote_testing"],
                adaptive=row["adaptive"],
                bm25_score=float(row["bm25_score"]),
            ))

        return results

    def search_vector(
        self, query: str, top_k: int = DEFAULT_TOP_K,
        filters: Optional[dict] = None
    ) -> list[Assessment]:
        """
        Pure vector cosine similarity search using pgvector.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query_embedding = self._encode_query(query)
        filter_clause, filter_params = self._build_filter_clause(filters)

        sql = f"""
            SELECT id, name, url, test_types, description, duration,
                   job_levels, remote_testing, adaptive,
                   1 - (embedding <=> %s::vector) AS cosine_score
            FROM assessments
            {("WHERE TRUE " + filter_clause) if filter_clause else ""}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params = [query_embedding] + filter_params + [query_embedding, top_k]

        # If there are filters, adjust SQL
        if filter_clause:
            sql = f"""
                SELECT id, name, url, test_types, description, duration,
                       job_levels, remote_testing, adaptive,
                       1 - (embedding <=> %s::vector) AS cosine_score
                FROM assessments
                WHERE TRUE {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [query_embedding] + filter_params + [query_embedding, top_k]
        else:
            sql = f"""
                SELECT id, name, url, test_types, description, duration,
                       job_levels, remote_testing, adaptive,
                       1 - (embedding <=> %s::vector) AS cosine_score
                FROM assessments
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [query_embedding, query_embedding, top_k]

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(Assessment(
                id=row["id"],
                name=row["name"],
                url=row["url"],
                test_types=row["test_types"] or [],
                description=row["description"] or "",
                duration=row["duration"] or "",
                job_levels=row["job_levels"] or [],
                remote_testing=row["remote_testing"],
                adaptive=row["adaptive"],
                cosine_score=float(row["cosine_score"]),
            ))

        return results

    def search_hybrid(
        self, query: str, top_k: int = DEFAULT_TOP_K,
        alpha: Optional[float] = None,
        filters: Optional[dict] = None
    ) -> list[Assessment]:
        """
        Hybrid search combining BM25 and vector cosine similarity.
        
        Score fusion:
            final_score = α × normalize(bm25_score) + (1 - α) × cosine_score
        
        Implementation:
            1. Fetch top candidates from BOTH BM25 and vector search (wider net)
            2. Merge results by assessment ID
            3. Normalize BM25 scores to [0, 1]
            4. Compute fused hybrid score
            5. Return top-K by hybrid score
        
        Args:
            query: Natural language search query
            top_k: Number of results to return (max 10 per assignment)
            alpha: BM25 weight (0.0 = pure vector, 1.0 = pure BM25)
            filters: Optional metadata filters
        
        Returns:
            List of Assessment objects sorted by hybrid score
        """
        alpha = alpha if alpha is not None else self.alpha

        # Fetch wider candidate pool from both search methods
        candidate_k = max(top_k * 3, 30)  # Get more candidates for fusion

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query_embedding = self._encode_query(query)
        filter_clause, filter_params = self._build_filter_clause(filters)

        # ── Single SQL query that computes hybrid score ──────────────
        # This approach runs both searches in one query for efficiency
        sql = f"""
            WITH bm25_results AS (
                SELECT id,
                       ts_rank(search_vector, plainto_tsquery('english', %s)) AS bm25_raw
                FROM assessments
                WHERE TRUE {filter_clause}
            ),
            bm25_normalized AS (
                SELECT id,
                    CASE
                        WHEN MAX(bm25_raw) OVER() = 0 THEN 0
                        WHEN MAX(bm25_raw) OVER() = MIN(bm25_raw) OVER() THEN
                            CASE WHEN bm25_raw > 0 THEN 1.0 ELSE 0.0 END
                        ELSE (bm25_raw - MIN(bm25_raw) OVER()) / 
                             NULLIF(MAX(bm25_raw) OVER() - MIN(bm25_raw) OVER(), 0)
                    END AS bm25_score
                FROM bm25_results
            ),
            vector_results AS (
                SELECT id,
                       1 - (embedding <=> %s::vector) AS cosine_score
                FROM assessments
                WHERE TRUE {filter_clause}
            )
            SELECT 
                a.id, a.name, a.url, a.test_types, a.description, 
                a.duration, a.job_levels, a.remote_testing, a.adaptive,
                COALESCE(b.bm25_score, 0) AS bm25_score,
                COALESCE(v.cosine_score, 0) AS cosine_score,
                (%s * COALESCE(b.bm25_score, 0) + %s * COALESCE(v.cosine_score, 0)) AS hybrid_score
            FROM assessments a
            LEFT JOIN bm25_normalized b ON a.id = b.id
            LEFT JOIN vector_results v ON a.id = v.id
            WHERE TRUE {filter_clause}
            ORDER BY hybrid_score DESC
            LIMIT %s
        """

        # Build params: query for BM25, embedding for vector, filter params (3x), alpha weights, top_k
        params = (
            [query] + filter_params +       # BM25 CTE
            [query_embedding] + filter_params +  # Vector CTE
            [alpha, 1.0 - alpha] +           # Score weights
            filter_params +                  # Final WHERE
            [top_k]                          # LIMIT
        )

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(Assessment(
                id=row["id"],
                name=row["name"],
                url=row["url"],
                test_types=row["test_types"] or [],
                description=row["description"] or "",
                duration=row["duration"] or "",
                job_levels=row["job_levels"] or [],
                remote_testing=row["remote_testing"],
                adaptive=row["adaptive"],
                score=float(row["hybrid_score"]),
                bm25_score=float(row["bm25_score"]),
                cosine_score=float(row["cosine_score"]),
            ))

        return results

    def get_assessment_by_url(self, url: str) -> Optional[Assessment]:
        """Look up a single assessment by its catalog URL."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT id, name, url, test_types, description, duration,
                   job_levels, remote_testing, adaptive
            FROM assessments WHERE url = %s
        """, (url,))

        row = cursor.fetchone()
        if not row:
            return None

        return Assessment(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            test_types=row["test_types"] or [],
            description=row["description"] or "",
            duration=row["duration"] or "",
            job_levels=row["job_levels"] or [],
            remote_testing=row["remote_testing"],
            adaptive=row["adaptive"],
        )

    def get_assessment_by_name(self, name: str) -> Optional[Assessment]:
        """Look up a single assessment by exact or partial name match."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Try exact match first
        cursor.execute("""
            SELECT id, name, url, test_types, description, duration,
                   job_levels, remote_testing, adaptive
            FROM assessments WHERE LOWER(name) = LOWER(%s)
        """, (name,))

        row = cursor.fetchone()
        if not row:
            # Try partial match
            cursor.execute("""
                SELECT id, name, url, test_types, description, duration,
                       job_levels, remote_testing, adaptive
                FROM assessments WHERE LOWER(name) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{name}%",))
            row = cursor.fetchone()

        if not row:
            return None

        return Assessment(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            test_types=row["test_types"] or [],
            description=row["description"] or "",
            duration=row["duration"] or "",
            job_levels=row["job_levels"] or [],
            remote_testing=row["remote_testing"],
            adaptive=row["adaptive"],
        )

    def get_all_urls(self) -> set[str]:
        """Get all valid catalog URLs for validation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM assessments")
        return {row[0] for row in cursor.fetchall()}

    def validate_urls(self, urls: list[str]) -> list[str]:
        """Validate that all URLs exist in the catalog. Return invalid ones."""
        valid_urls = self.get_all_urls()
        return [url for url in urls if url not in valid_urls]

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()


# ─── Module-level singleton ─────────────────────────────────────────
_retriever_instance = None


def get_retriever(alpha: float = DEFAULT_ALPHA) -> HybridRetriever:
    """Get or create the singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(alpha=alpha)
    return _retriever_instance
