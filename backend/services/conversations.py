from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from threading import RLock


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = RLock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
                document_hash TEXT, document_name TEXT, messages TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at)")

    def _connect(self):
        return sqlite3.connect(self.path)

    def upsert(self, conversation_id: str, user_id: str, title: str,
               document_hash: str | None, document_name: str | None,
               messages: list[dict[str, object]]) -> None:
        with self.lock, self._connect() as db:
            db.execute("""INSERT INTO conversations
                (id,user_id,title,document_hash,document_name,messages)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                document_hash=excluded.document_hash, document_name=excluded.document_name,
                messages=excluded.messages, updated_at=CURRENT_TIMESTAMP
                WHERE conversations.user_id=excluded.user_id""",
                (conversation_id, user_id, title, document_hash, document_name,
                 json.dumps(messages, ensure_ascii=False)))

    def list(self, user_id: str) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute("""SELECT id,title,document_hash,document_name,created_at,updated_at
                FROM conversations WHERE user_id=? ORDER BY updated_at DESC""", (user_id,)).fetchall()
        return [{"id": row[0], "title": row[1], "document_hash": row[2],
                 "document_name": row[3], "created_at": row[4], "updated_at": row[5]} for row in rows]

    def get(self, conversation_id: str, user_id: str) -> dict[str, object] | None:
        with self._connect() as db:
            row = db.execute("""SELECT id,title,document_hash,document_name,messages,created_at,updated_at
                FROM conversations WHERE id=? AND user_id=?""", (conversation_id, user_id)).fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "document_hash": row[2], "document_name": row[3],
                "messages": json.loads(row[4]), "created_at": row[5], "updated_at": row[6]}
