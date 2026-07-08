"""SQLite: users, documents, token usage."""
import sqlite3
from contextlib import contextmanager
from app import config


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doc_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunks INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                embed_tokens INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )


def create_user(email: str, pw_hash: str) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO users(email, pw_hash) VALUES (?,?)", (email, pw_hash))
        return cur.lastrowid


def get_user_by_email(email: str):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_user(uid: int):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def add_document(user_id: int, doc_id: str, filename: str, chunks: int):
    with conn() as c:
        c.execute("INSERT INTO documents(user_id, doc_id, filename, chunks) VALUES (?,?,?,?)",
                  (user_id, doc_id, filename, chunks))


def list_documents(user_id: int):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT doc_id, filename, chunks, created_at FROM documents WHERE user_id=? ORDER BY id DESC", (user_id,))]


def log_usage(user_id: int, kind: str, embed_t=0, prompt_t=0, output_t=0):
    with conn() as c:
        c.execute("INSERT INTO usage(user_id, kind, embed_tokens, prompt_tokens, output_tokens) VALUES (?,?,?,?,?)",
                  (user_id, kind, embed_t, prompt_t, output_t))


def usage_stats(user_id: int) -> dict:
    with conn() as c:
        tot = c.execute(
            "SELECT COALESCE(SUM(embed_tokens),0) e, COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(output_tokens),0) o, COUNT(*) n FROM usage WHERE user_id=?",
            (user_id,)).fetchone()
        chats = c.execute("SELECT COUNT(*) n FROM usage WHERE user_id=? AND kind='chat'", (user_id,)).fetchone()["n"]
        docs = c.execute("SELECT COUNT(*) n FROM documents WHERE user_id=?", (user_id,)).fetchone()["n"]
        return {
            "embed_tokens": tot["e"], "prompt_tokens": tot["p"], "output_tokens": tot["o"],
            "total_tokens": tot["e"] + tot["p"] + tot["o"], "chats": chats, "documents": docs,
        }
