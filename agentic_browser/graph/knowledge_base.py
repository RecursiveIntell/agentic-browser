"""
Knowledge Base - Unified API for Agent Memory.

Provides:
- Agent-specific Strategy and Apocalypse banks (encrypted)
- Tiered recall: Strategies > Apocalypse > Raw Runs
- Single write path for all knowledge operations
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from .secure_store import get_secure_store, SecureStore


@dataclass
class RecallResult:
    """Result from a tiered recall query."""
    strategies: list[dict]
    apocalypse: list[dict]
    raw_runs: list[dict]
    
    def to_prompt_injection(self) -> str:
        """Format results for LLM prompt injection."""
        sections = []
        
        if self.strategies:
            strat_text = "\n".join([
                f"  • {s['name']} (used {s.get('usage_count', 1)}x): {s['description']}"
                for s in self.strategies
            ])
            sections.append(f"🏆 PROVEN STRATEGIES (PRIORITIZE THESE):\n{strat_text}")
        
        if self.apocalypse:
            apoc_text = "\n".join([
                f"  ⚠️ {a['error_pattern']} (seen {a.get('occurrence_count', 1)}x): {a['description']}"
                for a in self.apocalypse
            ])
            sections.append(f"💀 MISTAKES TO AVOID:\n{apoc_text}")
        
        if self.raw_runs:
            runs_text = "\n".join([
                f"  📜 [{r.get('status', 'unknown')}] {r['goal'][:60]}..."
                for r in self.raw_runs[:3]
            ])
            sections.append(f"💡 RAW INSIGHTS (May contain new ideas):\n{runs_text}")
        
        if sections:
            sections.append("\nWEIGHTING: Trust encrypted banks most, but consider raw insights for innovation.")
            return "\n\n".join(sections)
        return ""


class KnowledgeBase:
    """Unified API for agent memory operations.
    
    Performance optimized with:
    - 500-entry embedding cache (~200MB for 16GB RAM)
    - ThreadPoolExecutor for parallel batch operations
    - tiered_recall_async for parallel knowledge retrieval
    """
    
    VALID_AGENTS = ("planner", "research")
    
    # Class-level caches and thread pool (shared across instances)
    _embedding_cache: dict[str, any] = {}
    _CACHE_MAX_SIZE = 500  # 16GB RAM - ~200MB for embeddings
    _executor: any = None  # ThreadPoolExecutor - lazy init
    
    def __init__(self, secure_store: Optional[SecureStore] = None):
        """Initialize knowledge base.
        
        Args:
            secure_store: SecureStore instance (uses global if not provided)
        """
        self.store = secure_store or get_secure_store()
        self._init_databases()
    
    @classmethod
    def _get_executor(cls):
        """Get or create shared ThreadPoolExecutor."""
        if cls._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            cls._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb_")
        return cls._executor
    
    @classmethod
    def preload_embedding_model(cls) -> bool:
        """Preload embedding model at startup to avoid cold-start delays.
        
        Call this early (e.g., during GUI init) to front-load the 3-5s model load.
        This prevents timeouts in tiered_recall_async() on first invocation.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(cls, '_shared_embedding_model'):
                print("[KNOWLEDGE] Preloading embedding model...")
                # CRITICAL: Use explicit device='cpu' to avoid meta tensor issues
                cls._shared_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("[KNOWLEDGE] Embedding model ready")
            return True
        except ImportError:
            print("[KNOWLEDGE] sentence-transformers not installed, semantic search disabled")
            return False
        except Exception as e:
            error_type = type(e).__name__
            print(f"[KNOWLEDGE] Preload failed ({error_type}): {e}")
            return False
        
    def _init_databases(self) -> None:
        """Initialize all agent-specific databases."""
        for agent in self.VALID_AGENTS:
            self._init_strategy_db(agent)
            self._init_apocalypse_db(agent)
        self._init_runs_db()
    
    def _init_strategy_db(self, agent: str) -> None:
        """Initialize strategy database for an agent."""
        conn = self.store.open_encrypted_db(agent, "strategies")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name_enc BLOB NOT NULL,
                description_enc BLOB NOT NULL,
                embedding BLOB,
                usage_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _init_apocalypse_db(self, agent: str) -> None:
        """Initialize apocalypse database for an agent."""
        conn = self.store.open_encrypted_db(agent, "apocalypse")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS apocalypse (
                id TEXT PRIMARY KEY,
                error_pattern_enc BLOB NOT NULL,
                description_enc BLOB NOT NULL,
                embedding BLOB,
                occurrence_count INTEGER DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _init_runs_db(self) -> None:
        """Initialize the shared runs database."""
        db_path = self.store.get_runs_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                final_answer TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embedding BLOB,
                hmac_signature TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    # ==================== STRATEGY OPERATIONS ====================
    
    def save_strategy(self, agent: str, name: str, description: str, 
                      embedding: Optional[bytes] = None) -> str:
        """Save a strategy to the agent's encrypted bank.
        
        Args:
            agent: 'planner' or 'research'
            name: Short strategy name
            description: Detailed description
            embedding: Optional vector embedding
            
        Returns:
            Strategy ID
        """
        if agent not in self.VALID_AGENTS:
            raise ValueError(f"Invalid agent: {agent}")
            
        now = datetime.utcnow().isoformat()
        
        # Check for existing similar strategy via embedding
        existing_id = self._find_similar_strategy(agent, embedding) if embedding else None
        
        conn = self.store.open_encrypted_db(agent, "strategies")
        
        if existing_id:
            # Update existing
            conn.execute("""
                UPDATE strategies 
                SET usage_count = usage_count + 1, last_used_at = ?
                WHERE id = ?
            """, (now, existing_id))
            strategy_id = existing_id
            print(f"[KNOWLEDGE] Updated existing strategy: {existing_id}")
        else:
            # Insert new
            strategy_id = str(uuid.uuid4())
            name_enc = self.store.encrypt_field(name)
            desc_enc = self.store.encrypt_field(description)
            
            conn.execute("""
                INSERT INTO strategies 
                (id, name_enc, description_enc, embedding, usage_count, created_at, last_used_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (strategy_id, name_enc, desc_enc, embedding, now, now))
            print(f"[KNOWLEDGE] Created new strategy for {agent}: {name}")
        
        conn.commit()
        conn.close()
        return strategy_id
    
    def _find_similar_strategy(self, agent: str, embedding: bytes, 
                                threshold: float = 0.85) -> Optional[str]:
        """Find an existing strategy with similar embedding."""
        import numpy as np
        
        conn = self.store.open_encrypted_db(agent, "strategies")
        rows = conn.execute(
            "SELECT id, embedding FROM strategies WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()
        
        if not rows or not embedding:
            return None
            
        query_emb = np.frombuffer(embedding, dtype=np.float32)
        
        for row in rows:
            if row['embedding']:
                db_emb = np.frombuffer(row['embedding'], dtype=np.float32)
                score = np.dot(query_emb, db_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(db_emb)
                )
                if score > threshold:
                    return row['id']
        return None
    
    def search_strategies(self, agent: str, query: str, limit: int = 3,
                          min_score: float = 0.3) -> list[dict]:
        """Search strategies for an agent using semantic similarity.
        
        Only returns strategies that are RELEVANT to the query (score >= min_score).
        
        Args:
            agent: 'planner' or 'research'
            query: Search query
            limit: Max results
            min_score: Minimum similarity score to include (0.0-1.0)
            
        Returns:
            List of strategy dicts with decrypted fields, sorted by relevance
        """
        conn = self.store.open_encrypted_db(agent, "strategies")
        rows = conn.execute("""
            SELECT id, name_enc, description_enc, embedding, usage_count
            FROM strategies
        """).fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Compute query embedding
        query_embedding = self._compute_query_embedding(query)
        if query_embedding is None:
            # Fallback: return top by usage count if no embedding model
            return self._fallback_strategy_search(rows, limit)
        
        # Score each row by semantic similarity
        import numpy as np
        scored_results = []
        
        for row in rows:
            try:
                name = self.store.decrypt_field(row['name_enc'])
                desc = self.store.decrypt_field(row['description_enc'])
                
                # If entry has embedding, use it; otherwise embed the description
                if row['embedding']:
                    entry_emb = np.frombuffer(row['embedding'], dtype=np.float32)
                else:
                    entry_emb = self._compute_query_embedding(f"{name} {desc}")
                    if entry_emb is None:
                        continue
                
                # Cosine similarity
                score = float(np.dot(query_embedding, entry_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_emb)
                ))
                
                # Only include if above threshold
                if score >= min_score:
                    scored_results.append({
                        'id': row['id'],
                        'name': name,
                        'description': desc,
                        'usage_count': row['usage_count'],
                        'relevance_score': score
                    })
            except Exception as e:
                print(f"[KNOWLEDGE] Strategy scoring error: {e}")
        
        # Sort by relevance, then by usage count as tiebreaker
        scored_results.sort(key=lambda x: (x['relevance_score'], x['usage_count']), reverse=True)
        
        return scored_results[:limit]
    
    def _fallback_strategy_search(self, rows: list, limit: int) -> list[dict]:
        """Fallback: return top strategies by usage count."""
        results = []
        for row in sorted(rows, key=lambda r: r['usage_count'], reverse=True)[:limit]:
            try:
                results.append({
                    'id': row['id'],
                    'name': self.store.decrypt_field(row['name_enc']),
                    'description': self.store.decrypt_field(row['description_enc']),
                    'usage_count': row['usage_count']
                })
            except Exception:
                pass
        return results
    
    def _compute_query_embedding(self, text: str) -> Optional[any]:
        """Compute embedding for a query string with LRU caching.
        
        Uses class-level cache (500 entries) to avoid redundant computations.
        """
        import hashlib
        
        # Create cache key from text hash
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Check cache first (fast path)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use class-level shared model if preloaded, else load instance model
            if hasattr(self.__class__, '_shared_embedding_model'):
                model = self.__class__._shared_embedding_model
            elif not hasattr(self, '_embedding_model'):
                # CRITICAL: Use explicit device='cpu' to avoid PyTorch meta tensor issues
                # The meta tensor error occurs when accelerate library lazy-loads on meta device
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                model = self._embedding_model
            else:
                model = self._embedding_model
            
            embedding = model.encode(text)
            
            # Add to cache with LRU eviction
            if len(self._embedding_cache) >= self._CACHE_MAX_SIZE:
                # Batch eviction: remove oldest 50 entries
                keys_to_remove = list(self._embedding_cache.keys())[:50]
                for k in keys_to_remove:
                    del self._embedding_cache[k]
                print(f"[KNOWLEDGE] Evicted 50 cached embeddings (cache at {self._CACHE_MAX_SIZE})")
            
            self._embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            # More specific error logging for debugging
            error_type = type(e).__name__
            print(f"[KNOWLEDGE] Embedding computation failed ({error_type}): {e}")
            return None
    
    def tiered_recall_async(self, agent: str, query: str) -> RecallResult:
        """Perform tiered recall with parallel queries using ThreadPoolExecutor.
        
        Runs strategy, apocalypse, and raw run searches in parallel.
        Uses single 2s total timeout for ALL queries (not 2s each).
        """
        from concurrent.futures import wait, FIRST_EXCEPTION
        
        executor = self._get_executor()
        
        # Submit all three searches in parallel
        strategy_future = executor.submit(self.search_strategies, agent, query, 3)
        apocalypse_future = executor.submit(self.search_apocalypse, agent, query, 3)
        runs_future = executor.submit(self.search_runs, query, agent, 3)
        
        all_futures = [strategy_future, apocalypse_future, runs_future]
        
        # Wait for ALL with single 2s total timeout (not 2s each!)
        done, not_done = wait(all_futures, timeout=2.0, return_when=FIRST_EXCEPTION)
        
        # Extract results from completed futures
        strategies = []
        apocalypse = []
        raw_runs = []
        
        if strategy_future in done and not strategy_future.exception():
            try:
                strategies = strategy_future.result(timeout=0)
            except Exception:
                pass
        
        if apocalypse_future in done and not apocalypse_future.exception():
            try:
                apocalypse = apocalypse_future.result(timeout=0)
            except Exception:
                pass
        
        if runs_future in done and not runs_future.exception():
            try:
                raw_runs = runs_future.result(timeout=0)
            except Exception:
                pass
        
        if not_done:
            print(f"[KNOWLEDGE] {len(not_done)} queries timed out (2s total)")
        
        return RecallResult(
            strategies=strategies,
            apocalypse=apocalypse,
            raw_runs=raw_runs,
        )
    
    # ==================== APOCALYPSE OPERATIONS ====================
    
    def save_apocalypse(self, agent: str, error_pattern: str, description: str,
                        embedding: Optional[bytes] = None) -> str:
        """Record a failure pattern to the agent's apocalypse bank.
        
        Args:
            agent: 'planner' or 'research'
            error_pattern: Short error pattern (e.g., "Clicked non-existent element")
            description: How to avoid this mistake
            embedding: Optional vector embedding
            
        Returns:
            Apocalypse entry ID
        """
        if agent not in self.VALID_AGENTS:
            raise ValueError(f"Invalid agent: {agent}")
            
        now = datetime.utcnow().isoformat()
        
        # Check for existing similar entry
        existing_id = self._find_similar_apocalypse(agent, embedding) if embedding else None
        
        conn = self.store.open_encrypted_db(agent, "apocalypse")
        
        if existing_id:
            # Update existing
            conn.execute("""
                UPDATE apocalypse 
                SET occurrence_count = occurrence_count + 1, last_seen_at = ?
                WHERE id = ?
            """, (now, existing_id))
            entry_id = existing_id
            print(f"[APOCALYPSE] Updated existing failure pattern: {existing_id}")
        else:
            # Insert new
            entry_id = str(uuid.uuid4())
            pattern_enc = self.store.encrypt_field(error_pattern)
            desc_enc = self.store.encrypt_field(description)
            
            conn.execute("""
                INSERT INTO apocalypse 
                (id, error_pattern_enc, description_enc, embedding, occurrence_count, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (entry_id, pattern_enc, desc_enc, embedding, now, now))
            print(f"[APOCALYPSE] Recorded new failure for {agent}: {error_pattern}")
        
        conn.commit()
        conn.close()
        return entry_id
    
    def _find_similar_apocalypse(self, agent: str, embedding: bytes,
                                  threshold: float = 0.85) -> Optional[str]:
        """Find existing apocalypse entry with similar embedding."""
        import numpy as np
        
        conn = self.store.open_encrypted_db(agent, "apocalypse")
        rows = conn.execute(
            "SELECT id, embedding FROM apocalypse WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()
        
        if not rows or not embedding:
            return None
            
        query_emb = np.frombuffer(embedding, dtype=np.float32)
        
        for row in rows:
            if row['embedding']:
                db_emb = np.frombuffer(row['embedding'], dtype=np.float32)
                score = np.dot(query_emb, db_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(db_emb)
                )
                if score > threshold:
                    return row['id']
        return None
    
    def search_apocalypse(self, agent: str, query: str, limit: int = 3,
                          min_score: float = 0.3) -> list[dict]:
        """Search apocalypse entries for an agent using semantic similarity.
        
        Only returns entries that are RELEVANT to the query (score >= min_score).
        
        Args:
            agent: 'planner' or 'research'
            query: Search query
            limit: Max results
            min_score: Minimum similarity score to include (0.0-1.0)
            
        Returns:
            List of apocalypse dicts with decrypted fields, sorted by relevance
        """
        conn = self.store.open_encrypted_db(agent, "apocalypse")
        rows = conn.execute("""
            SELECT id, error_pattern_enc, description_enc, embedding, occurrence_count
            FROM apocalypse
        """).fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Compute query embedding
        query_embedding = self._compute_query_embedding(query)
        if query_embedding is None:
            # Fallback: return top by occurrence count if no embedding model
            return self._fallback_apocalypse_search(rows, limit)
        
        # Score each row by semantic similarity
        import numpy as np
        scored_results = []
        
        for row in rows:
            try:
                pattern = self.store.decrypt_field(row['error_pattern_enc'])
                desc = self.store.decrypt_field(row['description_enc'])
                
                # If entry has embedding, use it; otherwise embed the description
                if row['embedding']:
                    entry_emb = np.frombuffer(row['embedding'], dtype=np.float32)
                else:
                    entry_emb = self._compute_query_embedding(f"{pattern} {desc}")
                    if entry_emb is None:
                        continue
                
                # Cosine similarity
                score = float(np.dot(query_embedding, entry_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_emb)
                ))
                
                # Only include if above threshold
                if score >= min_score:
                    scored_results.append({
                        'id': row['id'],
                        'error_pattern': pattern,
                        'description': desc,
                        'occurrence_count': row['occurrence_count'],
                        'relevance_score': score
                    })
            except Exception as e:
                print(f"[APOCALYPSE] Entry scoring error: {e}")
        
        # Sort by relevance, then by occurrence count as tiebreaker
        scored_results.sort(key=lambda x: (x['relevance_score'], x['occurrence_count']), reverse=True)
        
        return scored_results[:limit]
    
    def _fallback_apocalypse_search(self, rows: list, limit: int) -> list[dict]:
        """Fallback: return top apocalypse entries by occurrence count."""
        results = []
        for row in sorted(rows, key=lambda r: r['occurrence_count'], reverse=True)[:limit]:
            try:
                results.append({
                    'id': row['id'],
                    'error_pattern': self.store.decrypt_field(row['error_pattern_enc']),
                    'description': self.store.decrypt_field(row['description_enc']),
                    'occurrence_count': row['occurrence_count']
                })
            except Exception:
                pass
        return results
    
    # ==================== RAW RUNS OPERATIONS ====================
    
    def search_runs(self, query: str, agent: Optional[str] = None, 
                    limit: int = 5) -> list[dict]:
        """Search raw run history.
        
        Args:
            query: Search query
            agent: Optional agent filter
            limit: Max results
            
        Returns:
            List of run dicts
        """
        db_path = self.store.get_runs_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        if agent:
            rows = conn.execute("""
                SELECT id, agent, goal, status, final_answer
                FROM runs
                WHERE agent = ? AND status = 'success'
                ORDER BY updated_at DESC
                LIMIT ?
            """, (agent, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, agent, goal, status, final_answer
                FROM runs
                WHERE status = 'success'
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        conn.close()
        return [dict(row) for row in rows]
    
    # ==================== TIERED RECALL ====================
    
    def tiered_recall(self, agent: str, query: str) -> RecallResult:
        """Perform tiered recall for an agent.
        
        Order: Strategies > Apocalypse > Raw Runs
        
        Args:
            agent: 'planner' or 'research'
            query: Search query
            
        Returns:
            RecallResult with all tiers
        """
        strategies = self.search_strategies(agent, query, limit=3)
        apocalypse = self.search_apocalypse(agent, query, limit=3)
        raw_runs = self.search_runs(query, agent=agent, limit=3)
        
        return RecallResult(
            strategies=strategies,
            apocalypse=apocalypse,
            raw_runs=raw_runs
        )


# Singleton instance
_knowledge_base: Optional[KnowledgeBase] = None

def get_knowledge_base() -> KnowledgeBase:
    """Get the global KnowledgeBase instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
