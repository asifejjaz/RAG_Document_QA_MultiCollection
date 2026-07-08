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
        # additive migrations (safe to re-run)
        for stmt in (
            "ALTER TABLE users ADD COLUMN signup_ip TEXT",
            "ALTER TABLE users ADD COLUMN provider TEXT DEFAULT 'password'",
            "ALTER TABLE usage ADD COLUMN ip TEXT",
            "ALTER TABLE documents ADD COLUMN folder TEXT DEFAULT 'General'",
            "ALTER TABLE documents ADD COLUMN numeric_score REAL DEFAULT 1.0",
        ):
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass


def create_user(email: str, pw_hash: str, signup_ip: str = "", provider: str = "password") -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO users(email, pw_hash, signup_ip, provider) VALUES (?,?,?,?)",
            (email, pw_hash, signup_ip, provider),
        )
        return cur.lastrowid


def signups_from_ip(ip: str, hours: int = 24) -> int:
    if not ip:
        return 0
    with conn() as c:
        return c.execute(
            f"SELECT COUNT(*) n FROM users WHERE signup_ip=? AND created_at > datetime('now','-{int(hours)} hours')",
            (ip,),
        ).fetchone()["n"]


def tokens_today_user(user_id: int) -> int:
    with conn() as c:
        r = c.execute(
            "SELECT COALESCE(SUM(embed_tokens+prompt_tokens+output_tokens),0) t FROM usage WHERE user_id=? AND date(created_at)=date('now')",
            (user_id,),
        ).fetchone()
        return r["t"]


def tokens_today_ip(ip: str) -> int:
    if not ip:
        return 0
    with conn() as c:
        r = c.execute(
            "SELECT COALESCE(SUM(embed_tokens+prompt_tokens+output_tokens),0) t FROM usage WHERE ip=? AND date(created_at)=date('now')",
            (ip,),
        ).fetchone()
        return r["t"]


def accounts_sharing_ip():
    """Flag report: IPs with >1 account (possible abuse)."""
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT signup_ip, COUNT(*) accounts, GROUP_CONCAT(email) emails FROM users "
            "WHERE signup_ip IS NOT NULL AND signup_ip!='' GROUP BY signup_ip HAVING accounts>1 ORDER BY accounts DESC")]


def get_user_by_email(email: str):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_user(uid: int):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def add_document(user_id: int, doc_id: str, filename: str, chunks: int, folder: str = "General", numeric_score: float = 1.0):
    with conn() as c:
        c.execute("INSERT INTO documents(user_id, doc_id, filename, chunks, folder, numeric_score) VALUES (?,?,?,?,?,?)",
                  (user_id, doc_id, filename, chunks, folder, numeric_score))


def list_documents(user_id: int, folder: str | None = None):
    q = "SELECT doc_id, filename, chunks, folder, numeric_score, created_at FROM documents WHERE user_id=?"
    args: list = [user_id]
    if folder:
        q += " AND folder=?"
        args.append(folder)
    q += " ORDER BY id DESC"
    with conn() as c:
        return [dict(r) for r in c.execute(q, args)]


def list_folders(user_id: int):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT folder, COUNT(*) docs, COALESCE(SUM(chunks),0) chunks FROM documents "
            "WHERE user_id=? GROUP BY folder ORDER BY folder", (user_id,))]


def log_usage(user_id: int, kind: str, embed_t=0, prompt_t=0, output_t=0, ip: str = ""):
    with conn() as c:
        c.execute("INSERT INTO usage(user_id, kind, embed_tokens, prompt_tokens, output_tokens, ip) VALUES (?,?,?,?,?,?)",
                  (user_id, kind, embed_t, prompt_t, output_t, ip))


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
