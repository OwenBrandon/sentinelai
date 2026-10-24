"""
database.py — SQLite persistence layer for SentinelAI
=====================================================
Stores prediction logs and dataset metadata so they survive restarts.
SQLite needs no server and no install (built into Python). The whole
database is a single file: data/sentinel.db

Tables:
  predictions   — every analysis the system has ever made (the log)
  dataset_info  — metadata about the training data / model versions

Used by api/app.py.
"""

import os, sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sentinel.db"
)

@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row      # rows behave like dicts
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    """Create tables if they don't already exist. Safe to call on every startup."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                text         TEXT    NOT NULL,
                label        TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                action       TEXT    NOT NULL,
                risk_level   TEXT    NOT NULL,
                model_used   TEXT    NOT NULL,
                source       TEXT    NOT NULL,
                author       TEXT,
                platform     TEXT,
                timestamp    TEXT    NOT NULL
            )
        """)
        # Migrate older databases that lack author/platform columns
        cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
        if "author" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN author TEXT")
        if "platform" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN platform TEXT")
        con.execute("""
            CREATE TABLE IF NOT EXISTS dataset_info (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                total_rows      INTEGER,
                sympathizer     INTEGER,
                not_sympathizer INTEGER,
                model_name      TEXT,
                trained_at      TEXT
            )
        """)
        # index for fast recent-first queries
        con.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(id DESC)")

def log_prediction(p: dict):
    """Insert one prediction record."""
    with _conn() as con:
        con.execute("""
            INSERT INTO predictions
              (text,label,confidence,action,risk_level,model_used,source,author,platform,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            p["text"], p["label"], p["confidence"], p["action"],
            p["risk_level"], p["model_used"], p["source"],
            p.get("author"), p.get("platform"), p["timestamp"],
        ))

def get_predictions(limit=50, only_action=None):
    """Return most-recent predictions, optionally filtered by action."""
    q = "SELECT * FROM predictions"
    args = []
    if only_action:
        q += " WHERE action = ?"
        args.append(only_action)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as con:
        rows = con.execute(q, args).fetchall()
    return [dict(r) for r in rows]

def get_stats():
    """Aggregate counts straight from the database (survives restarts)."""
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        symp  = con.execute("SELECT COUNT(*) FROM predictions WHERE label='sympathizer'").fetchone()[0]
        nots  = con.execute("SELECT COUNT(*) FROM predictions WHERE label='not_sympathizer'").fetchone()[0]
        flag  = con.execute("SELECT COUNT(*) FROM predictions WHERE action='flag'").fetchone()[0]
        rev   = con.execute("SELECT COUNT(*) FROM predictions WHERE action='review'").fetchone()[0]
        ign   = con.execute("SELECT COUNT(*) FROM predictions WHERE action='ignore'").fetchone()[0]
    return {
        "total_predictions": total,
        "sympathizer_count": symp,
        "not_sympathizer_count": nots,
        "flag_count": flag,
        "review_count": rev,
        "ignore_count": ign,
    }

def set_dataset_info(total, sympathizer, not_sympathizer, model_name):
    """Record dataset/model metadata (called after training, or manually)."""
    with _conn() as con:
        con.execute("""
            INSERT INTO dataset_info
              (total_rows,sympathizer,not_sympathizer,model_name,trained_at)
            VALUES (?,?,?,?,?)
        """, (total, sympathizer, not_sympathizer, model_name,
              datetime.now(timezone.utc).isoformat()))

def get_dataset_info():
    """Return the most recent dataset metadata, or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM dataset_info ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None

def clear_predictions():
    """Wipe the prediction log (for demos / resets)."""
    with _conn() as con:
        con.execute("DELETE FROM predictions")

# Quick self-test when run directly
if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")
    log_prediction({
        "text":"test sentence","label":"not_sympathizer","confidence":88.0,
        "action":"ignore","risk_level":"low","model_used":"afroxlmr",
        "source":"selftest","timestamp":datetime.now(timezone.utc).isoformat()
    })
    print("Inserted 1 test row.")
    print("Stats:", get_stats())
    print("Recent:", get_predictions(limit=1))
