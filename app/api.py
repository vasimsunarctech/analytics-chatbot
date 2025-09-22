import re
import math
import logging
from flask import Blueprint, request, jsonify, session
from .db import get_db, create_message
from .vanna_service import get_vanna

bp = Blueprint("api", __name__)
logger = logging.getLogger("api")
CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", re.MULTILINE)

def clean_sql(sql_text: str) -> str:
    """
    Normalize LLM output to a single SELECT statement:
    - Remove markdown fences and obvious helper lines
    - Extract the first SELECT ... ; (or end-of-string)
    """
    if not sql_text:
        return ""

    # drop fences
    s = CODE_FENCE_RE.sub("", sql_text).strip()

    # drop “INTERMEDIATE_SQL:” and other helper notes
    cleaned_lines = []
    for line in s.splitlines():
        L = line.strip()
        if not L:
            continue
        # ignore helper/meta lines
        if L.upper().startswith("INTERMEDIATE_SQL:") or L.lower().startswith("intermediate_sql") :
            continue
        if L.lower().startswith("--"):
            continue
        cleaned_lines.append(L)
    s = "\n".join(cleaned_lines)

    # Find first SELECT ... (greedy until semicolon if present)
    m = re.search(r"(?is)\bselect\b.*?(;|$)", s)
    if m:
        return m.group(0).rstrip(";").strip()

    # As a fallback, if there’s any line that begins with SELECT
    for line in s.splitlines():
        if line.lower().lstrip().startswith("select "):
            return line.rstrip(";").strip()

    return s.strip()

def is_write_or_dangerous(sql_lower: str) -> bool:
    BAD = (" update ", " delete ", " insert ", " drop ", " alter ", " truncate ", " merge ", " create ", " replace ")
    # contains a bad keyword as a separate token
    return any(b in f" {sql_lower} " for b in BAD)

def ensure_limit(sql: str, default_limit: int = 1000) -> str:
    # Add LIMIT if it’s a simple SELECT without LIMIT already
    if re.search(r"(?is)\blimit\s+\d+", sql):
        return sql
    # naive protection: append LIMIT if not a CTE or has final ORDER BY
    # (keeps it simple; adjust for your needs)
    return f"{sql}\nLIMIT {default_limit}"

def _is_numeric(vals):
    seen = 0
    for v in vals:
        if v is None:
            continue
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            seen += 1
        else:
            return False
        if seen >= 3:
            return True
    return seen > 0

def _chart_from_df(df):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    if len(cols) < 2:
        return None
    x, y = cols[0], cols[1]
    y_vals = df[y].tolist()
    if _is_numeric(y_vals):
        return {
            "type": "bar",
            "label": y,
            "labels": [str(v) for v in df[x].tolist()],
            "data": [0 if v is None else v for v in y_vals],
        }
    return None

@bp.route("/messages", methods=["POST"])
def api_send_message():
    if not request.is_json:
        return jsonify({"status": "error", "error": "Expected JSON."}), 400

    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    text = (payload.get("message") or "").trim() if hasattr(str, "trim") else (payload.get("message") or "").strip()

    if not isinstance(session_id, int):
        return jsonify({"status": "error", "error": "Invalid session_id."}), 400
    if not text:
        return jsonify({"status": "error", "error": "Message cannot be empty."}), 400

    db = get_db()
    row = db.execute("SELECT id FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return jsonify({"status": "error", "error": "Chat session not found."}), 404

    user_msg = create_message(db, session_id, "user", text)

    vn = get_vanna()

    # 1) generate SQL (raw)
    raw_sql = vn.generate_sql(question=text)

    # 2) sanitize
    sql = clean_sql(raw_sql)
    sql_lower = sql.lower()

    # 3) enforce read-only
    if not sql_lower.startswith("select") or is_write_or_dangerous(sql_lower):
        ai_msg = create_message(db, session_id, "ai",
                                "I generated a non-read-only query. Please try rephrasing your question.",
                                metadata={"raw_sql": raw_sql})
        return jsonify({"status": "ok", "messages": [user_msg, ai_msg]})

    # 4) safety LIMIT
    sql_to_run = ensure_limit(sql, 1000)
    logger.info(f"sql_to_run: {sql_to_run}")
    # 5) run + explain
    df = vn.run_sql(sql_to_run)
    # explanation = vn.ask(question=text)

    cols = list(df.columns) if df is not None else []
    rows = df.to_dict(orient="records") if df is not None else []
    chart = _chart_from_df(df)

    metadata = {"columns": cols, "rows": rows, "chart": chart, "sql": ""}
    ai_msg = create_message(
        db, session_id, "ai",
        "",
        # f"{explanation}\n\n```sql\n{sql_to_run}\n```",
        metadata=metadata
    )

    session["current_session_id"] = session_id
    return jsonify({"status": "ok", "messages": [user_msg, ai_msg]})
