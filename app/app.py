from flask import Blueprint, render_template, redirect, url_for, request, session, flash
from .db import get_db, create_message
from .utils import format_timestamp_value
from datetime import datetime
from zoneinfo import ZoneInfo

bp = Blueprint("pages", __name__)
IST = ZoneInfo("Asia/Kolkata")

def is_authenticated():
    return session.get("user_id") is not None

def login_required(view):
    from functools import wraps
    @wraps(view)
    def _wrap(*args, **kwargs):
        if not is_authenticated():
            return redirect(url_for("pages.login"))
        return view(*args, **kwargs)
    return _wrap

@bp.before_app_request
def ensure_session_state():
    pass  # DB init handled in init_app

@bp.route("/")
def index():
    if is_authenticated():
        return redirect(url_for("pages.chat"))
    return redirect(url_for("pages.login"))

@bp.route("/login", methods=["GET","POST"])
def login():
    if is_authenticated():
        return redirect(url_for("pages.chat"))

    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        db = get_db()
        user = db.execute("SELECT id, username, password, full_name FROM users WHERE username=?", (username,)).fetchone()
        if user and user["password"] == password:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            # create first chat session
            ts = datetime.now(IST).strftime("New Chat %Y-%m-%d %I:%M %p")
            cur = db.execute("INSERT INTO chat_sessions (session_name, created_at) VALUES (?, ?)",
                             (ts, datetime.now(IST).isoformat()))
            db.commit()
            session["current_session_id"] = cur.lastrowid
            flash("Logged in successfully.", "success")
            return redirect(url_for("pages.chat", session_id=cur.lastrowid))
        flash("Invalid credentials. Please try again.", "error")
    return render_template("login.html")

@bp.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("pages.login"))

@bp.route("/chat")
@login_required
def chat():
    db = get_db()
    requested_session_id = request.args.get("session_id", type=int)
    current_session_id = requested_session_id or session.get("current_session_id")

    row = None
    if current_session_id:
        row = db.execute("SELECT id, session_name, created_at FROM chat_sessions WHERE id=?",(current_session_id,)).fetchone()

    if row is None:
        ts = datetime.now(IST).strftime("New Chat %Y-%m-%d %I:%M %p")
        cur = db.execute("INSERT INTO chat_sessions (session_name, created_at) VALUES (?, ?)",
                         (ts, datetime.now(IST).isoformat()))
        db.commit()
        current_session_id = cur.lastrowid
        row = db.execute("SELECT id, session_name, created_at FROM chat_sessions WHERE id=?",(current_session_id,)).fetchone()

    session["current_session_id"] = current_session_id

    sessions_rows = db.execute("SELECT id, session_name, created_at FROM chat_sessions ORDER BY created_at DESC").fetchall()
    sessions_payload = [{"id": r["id"], "session_name": r["session_name"], "created_at": format_timestamp_value(r["created_at"])} for r in sessions_rows]

    messages = db.execute("SELECT id, session_id, sender, message_text, timestamp, metadata FROM messages WHERE session_id=? ORDER BY timestamp ASC", (current_session_id,)).fetchall()

    # serialize for Alpine
    import json, datetime as dt
    def _ser(row):
        meta = None
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except:
                meta = None
        ts = str(row["timestamp"])
        try:
            ts_iso = dt.datetime.fromisoformat(ts).isoformat()
        except:
            ts_iso = ts
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "sender": row["sender"],
            "message_text": row["message_text"],
            "timestamp": ts_iso,
            "formatted_timestamp": format_timestamp_value(ts),
            "metadata": meta,
        }

    messages_data = [_ser(r) for r in messages]

    return render_template(
        "chat.html",
        selected_session={"id": row["id"], "session_name": row["session_name"]},
        messages=messages,
        messages_data=messages_data,
        sessions=sessions_payload,
        sessions_payload=sessions_payload,
    )

@bp.route("/sessions/new", methods=["POST"])
@login_required
def new_session():
    db = get_db()
    name = request.form.get("session_name")
    ts = (name or datetime.now(IST).strftime("New Chat %Y-%m-%d %I:%M %p"))
    cur = db.execute("INSERT INTO chat_sessions (session_name, created_at) VALUES (?, ?)",
                     (ts, datetime.now(IST).isoformat()))
    db.commit()
    session["current_session_id"] = cur.lastrowid
    flash("New chat session created.", "success")
    return redirect(url_for("pages.chat", session_id=cur.lastrowid))

@bp.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    db = get_db()
    user = db.execute("SELECT id, username, password, full_name FROM users WHERE id=?", (session["user_id"],)).fetchone()
    if not user:
        return redirect(url_for("pages.logout"))

    if request.method == "POST":
        full_name = request.form.get("full_name","").strip()
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        confirm = request.form.get("confirm_password","")
        if not username:
            flash("Username cannot be empty.", "error")
            return redirect(url_for("pages.profile"))
        other = db.execute("SELECT id FROM users WHERE username=? AND id!=?", (username, user["id"])).fetchone()
        if other:
            flash("Username already taken.", "error")
            return redirect(url_for("pages.profile"))
        if password and password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("pages.profile"))

        new_pass = password if password else user["password"]
        db.execute("UPDATE users SET username=?, password=?, full_name=? WHERE id=?",
                   (username, new_pass, full_name or None, user["id"]))
        db.commit()
        session["username"] = username
        session["full_name"] = full_name or None
        flash("Profile updated successfully.", "success")
        return redirect(url_for("pages.profile"))

    return render_template("profile.html", user={"username": user["username"], "full_name": user["full_name"] or ""})

@bp.route("/chat-history")
@login_required
def chat_history():
    db = get_db()
    rows = db.execute("SELECT id, session_name, created_at FROM chat_sessions ORDER BY created_at DESC").fetchall()
    sessions = [{"id": r["id"], "session_name": r["session_name"], "created_at": format_timestamp_value(r["created_at"])} for r in rows]
    return render_template("chat_history.html", sessions=sessions)
