import sqlite3
from flask import current_app, g
from contextlib import closing
from click import command
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def get_db():
    if "db" not in g:
        conn = sqlite3.connect(current_app.config["DATABASE"])
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_schema():
    db = get_db()
    with closing(db.cursor()) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sender TEXT NOT NULL CHECK(sender IN ('user','ai')),
                message_text TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        # seed admin
        row = cur.execute("SELECT id FROM users WHERE username=?", ("admin",)).fetchone()
        if not row:
            cur.execute("INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)",
                        ("admin", "admin123", "Administrator"))
        db.commit()

def init_app_db(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_schema()

@command("init-db")
def init_db_command():
    """flask --app run.py init-db"""
    init_schema()
    print("Initialized the database.")

def create_message(db, session_id, sender, text, timestamp=None, metadata=None):
    ts = timestamp or datetime.now(IST)
    cur = db.execute(
        "INSERT INTO messages (session_id, sender, message_text, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
        (session_id, sender, text, ts.isoformat(), None if metadata is None else __import__("json").dumps(metadata))
    )
    db.commit()
    return {
        "id": cur.lastrowid,
        "session_id": session_id,
        "sender": sender,
        "message_text": text,
        "timestamp": ts.isoformat(),
        "formatted_timestamp": ts.astimezone(IST).strftime("%b %d, %Y %I:%M %p IST"),
        "metadata": metadata,
    }
