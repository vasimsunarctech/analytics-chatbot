from __future__ import annotations

import sqlite3
import os
from contextlib import closing
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


print()
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
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database.db"
SECRET_KEY = "replace-with-a-random-secret"

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = os.getenv("CHAT_SYSTEM_PROMPT")


app = Flask(__name__)
app.config.update(SECRET_KEY=SECRET_KEY, DATABASE=str(DATABASE_PATH))

llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OLLAMA_API_BASE)


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
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                    ON DELETE CASCADE
            )
            """
        )

        ensure_full_name_column(db)

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


def get_session_messages(db: sqlite3.Connection, session_id: int) -> List[sqlite3.Row]:
    return db.execute(
        """
        SELECT sender, message_text
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        (session_id,),
    ).fetchall()


def build_conversation_payload(rows: List[sqlite3.Row]) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for row in rows:
        role = "assistant" if row["sender"] == "ai" else "user"
        messages.append({"role": role, "content": row["message_text"]})

    return messages


def call_llm(db: sqlite3.Connection, session_id: int) -> str:
    """Call the Ollama-backed model using the OpenAI SDK."""
    rows = get_session_messages(db, session_id)
    messages = build_conversation_payload(rows)

    if len(messages) == (1 if SYSTEM_PROMPT else 0):
        # Ensure the assistant has at least one user message to respond to.
        return "I need a question to respond to."

    try:
        response = llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=False,
            timeout=120,
        )
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)

        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content)
        else:
            text = content or ""

        text = text.strip()
        if not text:
            raise ValueError("Received empty response from model")
        return text
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("LLM call failed: %s", exc)
        return "I ran into an issue generating a response. Please try again."


# --- Message helpers ------------------------------------------------------


def serialize_message(row: sqlite3.Row) -> dict[str, str | int]:
    raw_timestamp = row["timestamp"]
    if isinstance(raw_timestamp, datetime):
        iso_timestamp = raw_timestamp.isoformat()
    else:
        try:
            iso_timestamp = datetime.fromisoformat(str(raw_timestamp)).isoformat()
        except ValueError:
            iso_timestamp = str(raw_timestamp)

    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "sender": row["sender"],
        "message_text": row["message_text"],
        "timestamp": iso_timestamp,
        "formatted_timestamp": format_timestamp_value(raw_timestamp),
    }


def create_message(
    db: sqlite3.Connection,
    session_id: int,
    sender: str,
    text: str,
    timestamp: Optional[datetime] = None,
) -> dict[str, str | int]:
    timestamp = timestamp or datetime.utcnow()
    cur = db.execute(
        "INSERT INTO messages (session_id, sender, message_text, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, sender, text, timestamp),
    )
    db.commit()
    return {
        "id": cur.lastrowid,
        "session_id": session_id,
        "sender": sender,
        "message_text": text,
        "timestamp": timestamp.isoformat(),
        "formatted_timestamp": format_timestamp_value(timestamp),
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
        SELECT id, session_id, sender, message_text, timestamp
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

    ai_reply = call_llm(db, session_id)
    create_message(db, session_id, "ai", ai_reply)

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
    ai_reply = call_llm(db, session_id)
    ai_message = create_message(db, session_id, "ai", ai_reply)

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
