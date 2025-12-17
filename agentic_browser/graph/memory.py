"""
Memory and persistence for LangGraph sessions.

Provides SQLite-based checkpointing for session resume.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from ..config import get_base_dir


class SessionStore:
    """SQLite-based session storage for agent state.
    
    Enables:
    - Session persistence across restarts
    - Resume interrupted tasks
    - Session history for debugging
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize session store.
        
        Args:
            db_path: Path to SQLite database (default: ~/.agentic_browser/sessions.db)
        """
        if db_path is None:
            db_path = get_base_dir() / "sessions.db"
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                final_answer TEXT,
                state_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                agent TEXT NOT NULL,
                action TEXT,
                result TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()
    
    def create_session(self, session_id: str, goal: str, state: dict) -> None:
        """Create a new session.
        
        Args:
            session_id: Unique session identifier
            goal: User's goal
            state: Initial state dict
        """
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions (id, goal, created_at, updated_at, state_json)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, goal, now, now, json.dumps(state)))
        conn.commit()
    
    def update_session(self, session_id: str, state: dict) -> None:
        """Update session state.
        
        Args:
            session_id: Session identifier
            state: Updated state dict
        """
        now = datetime.utcnow().isoformat()
        completed = state.get("task_complete", False)
        final_answer = state.get("final_answer")
        
        conn = self._get_conn()
        conn.execute("""
            UPDATE sessions 
            SET state_json = ?, updated_at = ?, completed = ?, final_answer = ?
            WHERE id = ?
        """, (json.dumps(state), now, completed, final_answer, session_id))
        conn.commit()
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dict or None if not found
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        
        if row:
            return {
                "id": row["id"],
                "goal": row["goal"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "completed": bool(row["completed"]),
                "final_answer": row["final_answer"],
                "state": json.loads(row["state_json"]),
            }
        return None
    
    def add_step(
        self, 
        session_id: str, 
        step_number: int,
        agent: str,
        action: Optional[str] = None,
        result: Optional[str] = None,
    ) -> None:
        """Log a step in the session.
        
        Args:
            session_id: Session identifier
            step_number: Step number
            agent: Agent that executed the step
            action: Action taken
            result: Result or error message
        """
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO session_steps (session_id, step_number, agent, action, result, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, step_number, agent, action, result, now))
        conn.commit()
    
    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Get recent sessions.
        
        Args:
            limit: Maximum sessions to return
            
        Returns:
            List of session dicts (without full state)
        """
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT id, goal, created_at, updated_at, completed, final_answer
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session and its steps.
        
        Args:
            session_id: Session identifier
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM session_steps WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
