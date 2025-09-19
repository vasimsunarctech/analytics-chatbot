from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
    jsonify,
)
from sql_rag import run as sql_rag_run

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database.db"
SECRET_KEY = "replace-with-a-random-secret"

app = Flask(__name__)
app.config.update(SECRET_KEY=SECRET_KEY, DATABASE=str(DATABASE_PATH))


# --- Database helpers -----------------------------------------------------

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        db_path = app.config["DATABASE"]
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exception: Optional[BaseException]) -> None:  # noqa: ARG001
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    with closing(db.cursor()) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sender TEXT NOT NULL CHECK(sender IN ('user', 'ai')),
                message_text TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                    ON DELETE CASCADE
            )
            """
        )

        ensure_full_name_column(db)
        ensure_message_metadata_column(db)

        cur.execute(
            "SELECT id, full_name FROM users WHERE username = ?",
            ("admin",),
        )
        existing_user = cur.fetchone()
        if existing_user is None:
            cur.execute(
                "INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)",
                ("admin", "admin123", "Administrator"),
            )
        elif existing_user["full_name"] is None:
            cur.execute(
                "UPDATE users SET full_name = ? WHERE username = ?",
                ("Administrator", "admin"),
            )

        db.commit()


def ensure_full_name_column(db: sqlite3.Connection) -> None:
    columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
    if "full_name" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN full_name TEXT")


def ensure_message_metadata_column(db: sqlite3.Connection) -> None:
    columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
    if "metadata" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")


# --- LLM helpers -----------------------------------------------------------


def format_timestamp_value(value: Optional[str | datetime]) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            return str(value)

    return dt.strftime("%b %d, %Y %I:%M %p")


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value

def derive_chart_from_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    sample = rows[0]
    string_keys = [
        k
        for k, v in sample.items()
        if isinstance(v, (str, datetime))
    ]
    numeric_keys = [
        k
        for k, v in sample.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    if not string_keys or not numeric_keys:
        return None

    label_key = string_keys[0]
    value_key = numeric_keys[0]

    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        label = row.get(label_key)
        value = row.get(value_key)
        if label is None or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
        labels.append(str(label))

    if not labels or not values:
        return None

    return {
        "type": "bar",
        "label": value_key.replace("_", " ").title(),
        "labels": labels,
        "data": values,
    }

def run_sql_rag(question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        result = sql_rag_run(question)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("SQL RAG execution failed: %s", exc)
        return (
            "I couldn't retrieve the requested information right now. Please try again later.",
            None,
        )

    if result.get("status") != "ok":
        message = result.get("message") or "I couldn't find data for that just now."
        return (message, None)

    answer = result.get("answer") or "I hope that helps."
    rows = result.get("rows") or []

    serializable_rows: List[Dict[str, Any]] = []
    for row in rows:
        serializable_rows.append({k: make_json_serializable(v) for k, v in row.items()})

    metadata: Dict[str, Any] = {}
    if serializable_rows:
        metadata["rows"] = serializable_rows
        metadata["columns"] = list(serializable_rows[0].keys())
        chart = derive_chart_from_rows(rows)
        if chart:
            metadata["chart"] = chart

    return answer, (metadata or None)


# --- Message helpers ------------------------------------------------------


def serialize_message(row: sqlite3.Row) -> dict[str, Any]:
    raw_timestamp = row["timestamp"]
    if isinstance(raw_timestamp, datetime):
        iso_timestamp = raw_timestamp.isoformat()
    else:
        try:
            iso_timestamp = datetime.fromisoformat(str(raw_timestamp)).isoformat()
        except ValueError:
            iso_timestamp = str(raw_timestamp)

    keys = row.keys()
    metadata_raw = row["metadata"] if "metadata" in keys else None
    metadata: Optional[Dict[str, Any]] = None
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = None

    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "sender": row["sender"],
        "message_text": row["message_text"],
        "timestamp": iso_timestamp,
        "formatted_timestamp": format_timestamp_value(raw_timestamp),
        "metadata": metadata,
    }


def create_message(
    db: sqlite3.Connection,
    session_id: int,
    sender: str,
    text: str,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    timestamp = timestamp or datetime.utcnow()
    metadata_json = json.dumps(metadata) if metadata else None
    cur = db.execute(
        "INSERT INTO messages (session_id, sender, message_text, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
        (session_id, sender, text, timestamp, metadata_json),
    )
    db.commit()
    return {
        "id": cur.lastrowid,
        "session_id": session_id,
        "sender": sender,
        "message_text": text,
        "timestamp": timestamp.isoformat(),
        "formatted_timestamp": format_timestamp_value(timestamp),
        "metadata": metadata,
    }


# --- Authentication helpers ----------------------------------------------

def is_authenticated() -> bool:
    return session.get("user_id") is not None


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if not is_authenticated():
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


# --- Routes ---------------------------------------------------------------


@app.before_request
def ensure_db_initialized() -> None:
    if not getattr(app, "_db_initialized", False):
        if not DATABASE_PATH.exists():
            DATABASE_PATH.touch()
        init_db()
        app._db_initialized = True


@app.route("/")
def index():
    if is_authenticated():
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("chat"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT id, username, password, full_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user and user["password"] == password:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]

            new_session_id = create_new_session(db)
            session["current_session_id"] = new_session_id

            flash("Logged in successfully.", "success")
            return redirect(url_for("chat", session_id=new_session_id))

        flash("Invalid credentials. Please try again.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/chat")
@login_required
def chat():
    db = get_db()
    requested_session_id = request.args.get("session_id", type=int)
    current_session_id = requested_session_id or session.get("current_session_id")

    selected_session = None
    if current_session_id:
        selected_session = db.execute(
            "SELECT id, session_name FROM chat_sessions WHERE id = ?",
            (current_session_id,),
        ).fetchone()

    if selected_session is None:
        current_session_id = create_new_session(db)
        selected_session = db.execute(
            "SELECT id, session_name FROM chat_sessions WHERE id = ?",
            (current_session_id,),
        ).fetchone()

    session["current_session_id"] = current_session_id

    messages = db.execute(
        """
        SELECT id, session_id, sender, message_text, timestamp, metadata
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        (current_session_id,),
    ).fetchall()
    messages_data = [serialize_message(row) for row in messages]

    return render_template(
        "chat.html",
        selected_session=selected_session,
        messages=messages,
        messages_data=messages_data,
    )


def create_new_session(db: sqlite3.Connection, session_name: Optional[str] = None) -> int:
    timestamp = datetime.utcnow()
    name = (session_name.strip() if session_name else None) or timestamp.strftime(
        "New Chat %Y-%m-%d %H:%M:%S"
    )
    cur = db.execute(
        "INSERT INTO chat_sessions (session_name, created_at) VALUES (?, ?)",
        (name, timestamp),
    )
    db.commit()
    return cur.lastrowid


@app.route("/sessions/new", methods=["POST"])
@login_required
def new_session():
    db = get_db()
    name = request.form.get("session_name")
    session_id = create_new_session(db, name if name else None)
    session["current_session_id"] = session_id
    flash("New chat session created.", "success")
    return redirect(url_for("chat", session_id=session_id))


@app.route("/messages", methods=["POST"])
@login_required
def send_message():
    db = get_db()
    session_id = request.form.get("session_id", type=int)
    prompt = request.form.get("message", "").strip()

    if not session_id:
        flash("Please select a chat session before sending messages.", "error")
        return redirect(url_for("chat"))

    session_exists = db.execute(
        "SELECT id FROM chat_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if session_exists is None:
        flash("The selected chat session was not found.", "error")
        return redirect(url_for("chat"))

    if not prompt:
        flash("Message cannot be empty.", "error")
        return redirect(url_for("chat", session_id=session_id))

    create_message(db, session_id, "user", prompt)

    ai_reply, metadata = run_sql_rag(prompt)
    create_message(db, session_id, "ai", ai_reply, metadata=metadata)

    session["current_session_id"] = session_id

    return redirect(url_for("chat", session_id=session_id))


@app.route("/api/messages", methods=["POST"])
@login_required
def api_send_message():
    if not request.is_json:
        return jsonify({"status": "error", "error": "Expected JSON payload."}), 400

    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    prompt = (payload.get("message") or "").strip()

    if not session_id:
        return jsonify({"status": "error", "error": "Missing session_id."}), 400

    try:
        session_id = int(session_id)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "Invalid session_id."}), 400

    db = get_db()
    session_exists = db.execute(
        "SELECT id FROM chat_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if session_exists is None:
        return jsonify({"status": "error", "error": "Chat session not found."}), 404

    if not prompt:
        return jsonify({"status": "error", "error": "Message cannot be empty."}), 400

    user_message = create_message(db, session_id, "user", prompt)
    ai_reply, metadata = run_sql_rag(prompt)
    ai_message = create_message(db, session_id, "ai", ai_reply, metadata=metadata)

    session["current_session_id"] = session_id

    return jsonify({"status": "ok", "messages": [user_message, ai_message]})


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db()
    user = db.execute(
        "SELECT id, username, password, full_name FROM users WHERE id = ?",
        (session["user_id"],),
    ).fetchone()

    if user is None:
        abort(404)

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username:
            flash("Username cannot be empty.", "error")
            return redirect(url_for("profile"))

        existing = db.execute(
            "SELECT id FROM users WHERE username = ? AND id != ?",
            (username, user["id"]),
        ).fetchone()
        if existing:
            flash("Username is already taken.", "error")
            return redirect(url_for("profile"))

        if password and password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("profile"))

        new_password = password if password else user["password"]
        db.execute(
            "UPDATE users SET username = ?, password = ?, full_name = ? WHERE id = ?",
            (username, new_password, full_name or None, user["id"]),
        )
        db.commit()

        session["username"] = username
        session["full_name"] = full_name or None

        flash("Profile updated successfully.", "success")
        return redirect(url_for("profile"))

    return render_template(
        "profile.html",
        user={
            "username": user["username"],
            "full_name": user["full_name"] or "",
        },
    )


@app.route("/chat-history")
@login_required
def chat_history():
    db = get_db()
    sessions = db.execute(
        "SELECT id, session_name, created_at FROM chat_sessions ORDER BY created_at DESC",
    ).fetchall()

    return render_template("chat_history.html", sessions=sessions)


@app.context_processor
def inject_user():
    return {
        "logged_in": is_authenticated(),
        "username": session.get("username"),
        "full_name": session.get("full_name"),
    }


@app.template_filter("format_timestamp")
def format_timestamp(value: Optional[str | datetime]) -> str:
    return format_timestamp_value(value)


if __name__ == "__main__":
    app.run(debug=True)
