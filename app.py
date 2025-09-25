from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
# from sql_agent import sql_agent

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
IST = ZoneInfo("Asia/Kolkata")

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

CURRENCY_KEYWORDS = (
    "amount",
    "revenue",
    "fare",
    "cost",
    "tariff",
    "collection",
    "expense",
    "value",
)


SMALL_TALK_TRIGGERS = {
    "greeting": {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"},
    "gratitude": {"thanks", "thank you", "appreciate"},
    "status": {"how are", "how's it going"},
    "time": {"what time", "current time", "time now"},
}

OUT_OF_SCOPE_KEYWORDS = {
    "prime minister",
    "president",
    "news",
    "movie",
    "weather",
    "stock",
    "football",
    "cricket",
    "politics",
    "song",
    "india today",
    "world",
    "celebrity",
}


def classify_prompt(text: str) -> Tuple[str, Optional[str]]:
    value = (text or "").strip().lower()
    if not value:
        return "empty", None

    for kind, triggers in SMALL_TALK_TRIGGERS.items():
        if any(re.search(r"\b" + re.escape(trigger) + r"\b", value) for trigger in triggers):
            return "small_talk", kind

    if any(keyword in value for keyword in OUT_OF_SCOPE_KEYWORDS):
        return "out_of_scope", None

    return "tas_query", None


def build_time_context() -> Dict[str, str]:
    now = datetime.now(IST)
    today = now.date()
    first_of_month = today.replace(day=1)
    last_month_last_day = first_of_month - timedelta(days=1)
    last_month_first_day = last_month_last_day.replace(day=1)
    return {
        "current_date_ist": today.strftime("%d-%b-%Y"),
        "current_time_ist": now.strftime("%I:%M %p"),
        "current_month_start": first_of_month.isoformat(),
        "last_month_start": last_month_first_day.isoformat(),
        "last_month_end": last_month_last_day.isoformat(),
    }


def get_recent_messages_for_context(
    db: sqlite3.Connection,
    session_id: int,
    limit: int = 6,
) -> List[Tuple[str, str]]:
    rows = db.execute(
        """
        SELECT sender, message_text
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return [(row["sender"], row["message_text"]) for row in reversed(rows)]


def respond_small_talk(kind: Optional[str], time_context: Dict[str, str]) -> str:
    if kind == "gratitude":
        return "Happy to help! Let me know what TAS insight you’d like to explore next."
    if kind == "status":
        return "I’m all set and ready to analyse your TAS data. What can I dig into for you?"
    if kind == "time":
        current_time = time_context.get("current_time_ist")
        current_date = time_context.get("current_date_ist")
        return f"In India it’s currently {current_time} on {current_date}. What TAS metric should we look at?"
    # greeting or default
    return "Hello! I’m InteriseIQ. Ask me about traffic, revenue, exemptions, or costs across your projects."


def format_for_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in row.items():
        serialized = make_json_serializable(value)
        if isinstance(value, (int, float)) and any(token in key.lower() for token in CURRENCY_KEYWORDS):
            serialized = f"₹{value:,.2f}"
        formatted[key] = serialized
    return formatted


def format_timestamp_value(value: Optional[str | datetime]) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        dt = value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
    else:
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
            else:
                dt = dt.astimezone(IST)
        except ValueError:
            return str(value)

    return dt.strftime("%b %d, %Y %I:%M %p IST")


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
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
    db = get_db()
    session_id = session.get("current_session_id")
    conversation = get_recent_messages_for_context(db, session_id) if session_id else []
    time_context = build_time_context()

    previous_route: Optional[Dict[str, Any]] = None
    if session_id:
        last_ai = db.execute(
            """
            SELECT metadata
            FROM messages
            WHERE session_id = ? AND sender = 'ai' AND metadata IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if last_ai and last_ai["metadata"]:
            try:
                meta_payload = json.loads(last_ai["metadata"])
            except json.JSONDecodeError:
                meta_payload = None
            if isinstance(meta_payload, dict):
                query_id = meta_payload.get("query_id")
                if query_id:
                    previous_route = {
                        "query_id": query_id,
                        "top": meta_payload.get("limit") or meta_payload.get("top"),
                        "start_datetime": meta_payload.get("resolved_start_datetime"),
                        "end_datetime": meta_payload.get("resolved_end_datetime"),
                    }

    prompt_type, detail = classify_prompt(question)
    if prompt_type == "empty":
        return ("Please enter a question so I can help with TAS analytics.", None)

    try:
        result = sql_rag_run(
            question,
            conversation=conversation,
            time_context=time_context,
            previous_route=previous_route,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("SQL RAG execution failed: %s", exc)
        result = {"status": "error", "message": str(exc)}

    if not isinstance(result, dict):
        return ("I couldn't process that request right now.", None)

    if result.get("mode") == "small_talk":
        return result.get("answer", "Hello!"), None

    if result.get("status") != "ok":
        message = result.get("message") or "I couldn't find data for that just now."
        return (message, None)

    rows = result.get("rows")
    sanitized_rows = result.get("sanitized_rows") if isinstance(result.get("sanitized_rows"), list) else None
    answer = result.get("answer")
    sql_used = result.get("sql")
    notes = result.get("notes")

    display_rows = sanitized_rows if sanitized_rows is not None else rows

    if not display_rows:
        message = result.get("message") or "I couldn't find data for that just now."
        return (message, None)

    if sanitized_rows is not None:
        serializable_rows = sanitized_rows
    else:
        serializable_rows = [format_for_metadata(row) for row in rows or []]

    metadata: Dict[str, Any] = {}
    metadata["query_id"] = result.get("query_id")
    metadata["limit"] = result.get("limit")
    metadata["resolved_start_datetime"] = result.get("resolved_start_datetime")
    metadata["resolved_end_datetime"] = result.get("resolved_end_datetime")
    metadata["previous_start_datetime"] = result.get("previous_start_datetime")
    metadata["previous_end_datetime"] = result.get("previous_end_datetime")
    if serializable_rows:
        metadata["rows"] = serializable_rows
        metadata["columns"] = build_column_definitions(list(serializable_rows[0].keys()))
        chart = derive_chart_from_rows(rows)
        if chart:
            metadata["chart"] = chart
    if sql_used:
        metadata["sql"] = sql_used
    if notes:
        metadata["notes"] = notes

    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None or key in {"rows", "columns", "chart"}
    }

    final_answer = answer or "I hope that helps."
    return final_answer, (metadata or None)


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
    timestamp = timestamp or datetime.now(IST)
    metadata_json = json.dumps(metadata, default=make_json_serializable) if metadata else None
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


def get_current_user(db: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return db.execute(
        "SELECT id, username, password, full_name FROM users WHERE id = ?",
        (session["user_id"],),
    ).fetchone()


def update_profile_record(
    db: sqlite3.Connection,
    user_row: sqlite3.Row,
    full_name: str,
    username: str,
    password: str,
    confirm_password: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not username:
        return False, "Username cannot be empty.", {}

    existing = db.execute(
        "SELECT id FROM users WHERE username = ? AND id != ?",
        (username, user_row["id"]),
    ).fetchone()
    if existing:
        return False, "Username is already taken.", {}

    if password and password != confirm_password:
        return False, "Passwords do not match.", {}

    new_password = password if password else user_row["password"]
    db.execute(
        "UPDATE users SET username = ?, password = ?, full_name = ? WHERE id = ?",
        (username, new_password, full_name or None, user_row["id"]),
    )
    db.commit()

    session["username"] = username
    session["full_name"] = full_name or None

    return True, "Profile updated successfully.", {
        "username": username,
        "full_name": full_name or "",
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

    sessions_rows = db.execute(
        """
        SELECT id, session_name, created_at
        FROM chat_sessions
        ORDER BY created_at DESC
        """
    ).fetchall()

    sessions_payload = []
    for row in sessions_rows:
        created = row["created_at"]
        formatted = format_timestamp_value(created)
        sessions_payload.append(
            {
                "id": row["id"],
                "session_name": row["session_name"],
                "created_at": formatted,
            }
        )

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
        sessions=sessions_payload,
        sessions_payload=sessions_payload,
    )


def create_new_session(db: sqlite3.Connection, session_name: Optional[str] = None) -> int:
    timestamp = datetime.now(IST)
    name = (session_name.strip() if session_name else None) or timestamp.strftime(
        "New Chat %Y-%m-%d %I:%M %p"
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
    user = get_current_user(db)

    if user is None:
        abort(404)

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        ok, message, updated = update_profile_record(
            db,
            user,
            full_name,
            username,
            password,
            confirm_password,
        )

        flash(message, "success" if ok else "error")
        if ok:
            return redirect(url_for("profile"))

        return redirect(url_for("profile"))

    return render_template(
        "profile.html",
        user={
            "username": user["username"],
            "full_name": user["full_name"] or "",
        },
    )


@app.route("/api/profile", methods=["GET", "POST"])
@login_required
def profile_api():
    db = get_db()
    user = get_current_user(db)
    if user is None:
        abort(404)

    if request.method == "GET":
        return jsonify(
            {
                "username": user["username"],
                "full_name": user["full_name"] or "",
            }
        )

    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    ok, message, payload = update_profile_record(
        db,
        user,
        full_name,
        username,
        password,
        confirm_password,
    )

    if not ok:
        return jsonify({"status": "error", "message": message}), 400

    return jsonify({"status": "ok", "message": message, "user": payload})


@app.route("/chat-history")
@login_required
def chat_history():
    db = get_db()
    sessions = db.execute(
        "SELECT id, session_name, created_at FROM chat_sessions ORDER BY created_at DESC",
    ).fetchall()

    formatted_sessions = []
    for row in sessions:
        formatted_sessions.append(
            {
                "id": row["id"],
                "session_name": row["session_name"],
                "created_at": format_timestamp_value(row["created_at"]),
            }
        )

    return render_template("chat_history.html", sessions=formatted_sessions)


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
