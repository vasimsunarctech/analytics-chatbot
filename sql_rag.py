from __future__ import annotations

import calendar
import json
import os
import re
import time
from datetime import date, datetime, timedelta, time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyodbc
from dotenv import load_dotenv
from openai import OpenAI
from zoneinfo import ZoneInfo

load_dotenv()

# --- Configuration ---------------------------------------------------------

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_ANSWER = os.environ.get(
    "SQL_ANSWER_PROMPT",
    "You are a helpful analyst. Summarise SQL query results for business users concisely, cite totals, highlight trends, and mention limits.",
)
MAX_RESULT_ROWS = int(os.environ.get("SQL_RESULT_LIMIT", "1000"))
TEMPLATE_PATH = Path(os.getenv("SQL_TEMPLATE_PATH", "sql_templates.json"))
SCHEMA_PATH = Path(os.getenv("SQL_SCHEMA_PATH", "schema_daily_transaction_final.json"))
TEMPLATE_CACHE_SECONDS = int(os.environ.get("SQL_TEMPLATE_CACHE_SECONDS", "120"))
FISCAL_YEAR_START_MONTH = int(os.environ.get("FISCAL_YEAR_START_MONTH", "4"))
IST = ZoneInfo("Asia/Kolkata")

CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('DB_USERNAME')};"
    f"PWD={os.getenv('DB_PASSWORD')}"
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OLLAMA_API_BASE)

_template_cache: Dict[str, Any] = {"timestamp": 0.0, "templates": []}
DEFAULT_DATE_TEMPLATE: Dict[str, Any] = {"defaults": {"time_window": "current_fiscal_year"}}


# --- Template helpers ------------------------------------------------------


def load_templates_from_disk() -> List[Dict[str, Any]]:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")
    try:
        payload = json.loads(TEMPLATE_PATH.read_text())
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise ValueError(f"Could not parse template file {TEMPLATE_PATH}: {exc}") from exc
    templates = payload.get("templates", [])
    if not isinstance(templates, list):
        raise ValueError("Template file must contain a top-level 'templates' array.")
    return templates


def get_templates() -> List[Dict[str, Any]]:
    now = time.time()
    cache_ok = now - _template_cache["timestamp"] < TEMPLATE_CACHE_SECONDS
    if cache_ok and _template_cache["templates"]:
        return _template_cache["templates"]
    templates = load_templates_from_disk()
    _template_cache["templates"] = templates
    _template_cache["timestamp"] = now
    return templates


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _token_score(question_tokens: List[str], phrase: str) -> int:
    phrase_tokens = _normalize_text(str(phrase)).split()
    if not phrase_tokens:
        return 0
    if all(token in question_tokens for token in phrase_tokens):
        return len(phrase_tokens)
    return 0


def match_template(question: str, templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized_question = _normalize_text(question)
    question_tokens = normalized_question.split()
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for template in templates:
        phrases = template.get("match_phrases", [])
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            score = _token_score(question_tokens, phrase)
            if score > best_score:
                best = template
                best_score = score
    return best


def extract_limit(question: str, template: Optional[Dict[str, Any]] = None) -> Optional[int]:
    defaults = template.get("defaults", {}) if template else {}
    limits_cfg = template.get("limits", {}).get("limit", {}) if template else {}

    limit_default = defaults.get("limit")

    match = re.search(r"top\s+(\d+)", question, re.IGNORECASE)
    if match:
        limit = int(match.group(1))
    elif limit_default is not None:
        limit = int(limit_default)
    else:
        return None

    if limits_cfg:
        if "min" in limits_cfg:
            limit = max(int(limits_cfg["min"]), limit)
        if "max" in limits_cfg:
            limit = min(int(limits_cfg["max"]), limit)

    return limit


# --- Schema helpers --------------------------------------------------------


def load_schema_data() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    try:
        return json.loads(SCHEMA_PATH.read_text())
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise ValueError(f"Could not parse schema file {SCHEMA_PATH}: {exc}") from exc


def build_schema_context(limit_tables: int = 12, limit_columns: int = 12) -> str:
    data = load_schema_data()
    tables = data.get("tables", [])
    lines = [f"Database: {data.get('database', 'unknown')}"]
    for table in tables[:limit_tables]:
        schema_name = table.get("schema", "dbo")
        table_name = table.get("name", "unknown")
        cols = table.get("columns", [])
        column_parts: List[str] = []
        for column in cols[:limit_columns]:
            column_parts.append(f"{column.get('name')} ({column.get('data_type')})")
        if len(cols) > limit_columns:
            column_parts.append("...")
        column_text = ", ".join(column_parts)
        lines.append(f"- {schema_name}.{table_name} -> {column_text}")
    if len(tables) > limit_tables:
        lines.append("(schema truncated)")
    return "\n".join(lines)


def format_conversation_history(
    conversation: Optional[Sequence[Tuple[str, str]]],
) -> str:
    if not conversation:
        return "(none)"
    lines = []
    for role, message in conversation[-6:]:
        label = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{label}: {message}")
    return "\n".join(lines)


FALLBACK_SQL_SYSTEM_PROMPT = (
    "You are a senior TAS analytics SQL assistant for Microsoft SQL Server.\n"
    "Rules:\n"
    "- Produce exactly one SELECT statement.\n"
    "- Never write INSERT, UPDATE, DELETE, DROP, ALTER, or temporary table statements.\n"
    "- Use table and column names exactly as provided; wrap identifiers containing spaces in square brackets.\n"
    "- Prefer aggregated answers and include ORDER BY when using TOP.\n"
    "- Return your response strictly as JSON with keys: sql, notes.\n"
)


def generate_sql_from_schema(
    question: str,
    start_datetime: str,
    end_datetime: str,
    limit: Optional[int],
    conversation: Optional[Sequence[Tuple[str, str]]],
    *,
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        schema_context = build_schema_context()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Schema load failed: {exc}"}

    conversation_text = format_conversation_history(conversation)
    limit_hint = str(limit) if limit is not None else "none"
    if limit is not None:
        limit_guideline = (
            f"- If the user requested top/bottom results, apply TOP {limit} with an ORDER BY that matches the KPI direction.\n"
        )
    else:
        limit_guideline = (
            "- Only introduce TOP when the user explicitly asks for top/bottom results.\n"
        )

    previous_section = ""
    if previous_sql or previous_error:
        previous_lines = ["Previous attempt:"]
        if previous_sql:
            previous_lines.append(f"SQL: {previous_sql}")
        if previous_error:
            previous_lines.append(f"Error: {previous_error}")
        previous_lines.append("Please correct the SQL accordingly.")
        previous_section = "\n\n" + "\n".join(previous_lines)

    user_content = (
        f"Schema:\n{schema_context}\n\n"
        f"Date window (IST):\n- start_datetime: {start_datetime}\n- end_datetime: {end_datetime}\n\n"
        f"Limit hint: {limit_hint}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        "Guidelines:\n"
        "- Constrain the query between the given datetimes using the appropriate date column(s).\n"
        "  • For dbo.ods_tmsdata_revenue use [date].\n"
        "  • For dbo.daily_transaction_final use transaction_date_time.\n"
        "- Honour explicit SPV/plaza filters from the user.\n"
        f"{limit_guideline}"
        "- Use TRY_CONVERT when aggregating numeric values stored as text.\n"
        "- Return column names that are meaningful for an analyst reading the result.\n\n"
        f"Question:\n{question}"
        f"{previous_section}"
    )

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": FALLBACK_SQL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        payload_text = _strip_json_block(raw)
        payload = json.loads(payload_text)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Could not generate SQL: {exc}"}

    sql = payload.get("sql")
    notes = payload.get("notes", "")

    if not sql or not isinstance(sql, str):
        return {"status": "error", "message": "Model did not return a SQL statement."}

    return {"status": "ok", "sql": sql, "notes": notes}


def fiscal_year_range(value: date, delta: int = 0) -> Tuple[date, date]:
    if value.month < FISCAL_YEAR_START_MONTH:
        base_start_year = value.year - 1
    else:
        base_start_year = value.year
    start_year = base_start_year + delta
    start = date(start_year, FISCAL_YEAR_START_MONTH, 1)
    end_start = date(start_year + 1, FISCAL_YEAR_START_MONTH, 1)
    end = end_start - timedelta(days=1)
    return start, end


def shift_years(value: date, years: int) -> date:
    target_year = value.year + years
    try:
        return value.replace(year=target_year)
    except ValueError:
        last_day = calendar.monthrange(target_year, value.month)[1]
        return value.replace(year=target_year, day=last_day)


def resolve_default_window(keyword: str, today: date) -> Tuple[date, date]:
    key = (keyword or "current_fiscal_year").lower()
    if key == "current_fiscal_year":
        return fiscal_year_range(today)
    if key == "last_12_months":
        end = today
        start = today - timedelta(days=365)
        return start, end
    end = today
    start = today - timedelta(days=365)
    return start, end


def start_of_day(value: date) -> datetime:
    return datetime.combine(value, dt_time(0, 0, 0, tzinfo=IST))


def end_of_day(value: date) -> datetime:
    return datetime.combine(value, dt_time(23, 59, 59, tzinfo=IST))


def format_datetime_ist(value: datetime) -> str:
    return value.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")


def _coerce_datetime(value: Optional[str], fallback: datetime) -> datetime:
    if not value:
        return fallback
    text = value.strip()
    if not text:
        return fallback
    text = text.replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text, fmt)
                if fmt == "%Y-%m-%d":
                    dt = datetime.combine(dt.date(), dt_time(0, 0, 0))
                break
            except ValueError:
                dt = None
        if dt is None:
            return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)
    return dt


def _strip_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.lstrip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rstrip("`")
    return stripped.strip()


def resolve_date_range_with_llm(
    question: str,
    template: Dict[str, Any],
    *,
    current_dt: datetime,
) -> Tuple[datetime, datetime]:
    today = current_dt.date()
    default_window = template.get("defaults", {}).get("time_window")
    default_start_date, default_end_date = resolve_default_window(default_window, today)
    default_start = start_of_day(default_start_date)
    default_end = end_of_day(default_end_date)

    system_prompt = (
        "You convert user analytics questions into explicit date ranges.\n"
        "Always work in Asia/Kolkata timezone.\n"
        f"Current IST datetime: {format_datetime_ist(current_dt)}.\n"
        f"If the user provides no timeframe, use start='{format_datetime_ist(default_start)}' and "
        f"end='{format_datetime_ist(default_end)}'.\n"
        "When the user references ranges like 'yesterday', 'today', 'last week', 'last month', 'last 3 months', "
        "'last year', 'previous year', or specific dates, convert them to exact boundaries.\n"
        "Align day-level requests to 00:00:00 for the start and 23:59:59 for the end.\n"
        "Respond strictly as JSON with keys start_datetime and end_datetime using format 'YYYY-MM-DD HH:MM:SS'."
    )

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        raw = response.choices[0].message.content.strip()
        payload_text = _strip_json_block(raw)
        payload = json.loads(payload_text)
        start_value = payload.get("start_datetime")
        end_value = payload.get("end_datetime")
        start_dt = _coerce_datetime(start_value, default_start)
        end_dt = _coerce_datetime(end_value, default_end)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        return start_dt, end_dt
    except Exception:
        return default_start, default_end


# --- SQL execution helpers -------------------------------------------------


FORBIDDEN_PATTERN = re.compile(
    r"(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|EXEC|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK)",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT statements are permitted.")
    if FORBIDDEN_PATTERN.search(cleaned):
        raise ValueError("Detected forbidden keyword in SQL statement.")
    if cleaned.count(";") > 0:
        raise ValueError("Multiple statements are not allowed.")
    return cleaned


def get_connection() -> pyodbc.Connection:
    return pyodbc.connect(CONNECTION_STRING)


def rows_to_dicts(cursor: pyodbc.Cursor, rows: Iterable[tuple]) -> List[Dict[str, Any]]:
    columns = [col[0] for col in cursor.description]
    results: List[Dict[str, Any]] = []
    for row in rows:
        record: Dict[str, Any] = {}
        for idx, value in enumerate(row):
            if isinstance(value, Decimal):
                coerced: Any = float(value)
            elif isinstance(value, datetime):
                coerced = value.isoformat()
            elif isinstance(value, date):
                coerced = value.isoformat()
            else:
                coerced = value
            record[columns[idx]] = coerced
        results.append(record)
    return results


def execute_query(sql: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        fetched = cursor.fetchmany(MAX_RESULT_ROWS)
        rows = rows_to_dicts(cursor, fetched)
        if cursor.fetchone():
            raise ValueError(
                f"Query returned more than {MAX_RESULT_ROWS} rows. Refine your question or add TOP clause."
            )
        return rows


# --- Answer generation -----------------------------------------------------


def generate_answer(
    question: str,
    sql: str,
    rows: List[Dict[str, Any]],
    time_context: Optional[Dict[str, str]] = None,
    conversation: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    if not rows:
        return (
            "I couldn't find any matching data for that request in the TAS warehouse. "
            "Try refining the project, date range, or metric."
        )

    messages = [
        {"role": "system", "content": SYSTEM_ANSWER},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "sql": sql,
                    "rows": rows,
                    "time_context": time_context,
                    "conversation": conversation,
                }
            ),
        },
    ]

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.4,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# --- Command workflow ------------------------------------------------------


def run(
    question: str,
    *,
    conversation: Optional[Sequence[Tuple[str, str]]] = None,
    time_context: Optional[Dict[str, str]] = None,
    retries: int = 0,
) -> Dict[str, Any]:
    def is_follow_up(text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        if re.search(r"\b(this|that|same|again|them|these|those|it)\b", normalized):
            return True
        follow_phrases = (
            "per week",
            "per month",
            "per day",
            "per quarter",
            "for this",
            "with this",
            "same period",
            "same data",
        )
        return any(phrase in normalized for phrase in follow_phrases)

    def last_user_question(conv: Optional[Sequence[Tuple[str, str]]]) -> Optional[str]:
        if not conv:
            return None
        for role, message in reversed(conv):
            if role == "user" and message and message.strip():
                return message.strip()
        return None

    try:
        templates = get_templates()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "message": f"Template load failed: {exc}",
        }

    previous_question = last_user_question(conversation)
    follow_up = bool(previous_question) and is_follow_up(question)
    normalized_follow_text = (question or "").strip().lower()
    dynamic_time_adjustment = any(
        phrase in normalized_follow_text
        for phrase in (
            "per week",
            "weekly",
            "per month",
            "monthly",
            "per quarter",
            "quarterly",
            "per year",
            "yearly",
        )
    )

    augmented_question = question
    if follow_up and previous_question:
        augmented_question = f"{previous_question}. Follow-up request: {question}"

    if follow_up and dynamic_time_adjustment:
        template = None
    else:
        template = match_template(augmented_question if follow_up else question, templates)
    limit = extract_limit(augmented_question, template)

    current_dt = datetime.now(IST)
    date_template = template if template else DEFAULT_DATE_TEMPLATE
    start_dt, end_dt = resolve_date_range_with_llm(
        augmented_question,
        date_template,
        current_dt=current_dt,
    )

    start_date = start_dt.date()
    end_date = end_dt.date()

    start_datetime = format_datetime_ist(start_dt)
    end_datetime = format_datetime_ist(end_dt)

    previous_start_date = shift_years(start_date, -1)
    previous_end_date = shift_years(end_date, -1)
    previous_start_dt = start_of_day(previous_start_date)
    previous_end_dt = end_of_day(previous_end_date)
    previous_start_datetime = format_datetime_ist(previous_start_dt)
    previous_end_datetime = format_datetime_ist(previous_end_dt)

    fallback_mode = template is None
    fallback_attempts = max(1, retries) if fallback_mode else 0
    attempt = 0
    previous_sql = None
    previous_error = None
    sql_text = ""
    template_id = ""
    notes_hint = ""

    while True:
        if fallback_mode:
            fallback = generate_sql_from_schema(
                augmented_question,
                start_datetime,
                end_datetime,
                limit,
                conversation,
                previous_sql=previous_sql,
                previous_error=previous_error,
            )
            if fallback.get("status") != "ok":
                return fallback
            sql_text = fallback["sql"]
            template_id = "fallback_schema"
            notes_hint = fallback.get("notes", "")
        else:
            if "{limit}" in template.get("sql_template", "") and limit is None:
                return {
                    "status": "error",
                    "message": f"Template '{template.get('id')}' requires a limit value.",
                }
            try:
                format_params = {
                    "limit": limit,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                    "previous_start_date": previous_start_date.isoformat(),
                    "previous_end_date": previous_end_date.isoformat(),
                    "previous_start_datetime": previous_start_datetime,
                    "previous_end_datetime": previous_end_datetime,
                }
                sql_text = template["sql_template"].format(**format_params)
            except KeyError as exc:
                return {
                    "status": "error",
                    "message": f"Template '{template.get('id')}' is missing placeholder {exc}.",
                }
            template_id = template.get("id", "template")
            notes_hint = ""

        try:
            validated_sql = validate_sql(sql_text)
        except Exception as exc:  # noqa: BLE001
            if fallback_mode and attempt < fallback_attempts:
                previous_sql = sql_text
                previous_error = str(exc)
                attempt += 1
                continue
            return {
                "status": "error",
                "message": str(exc),
                "sql": sql_text,
            }

        try:
            rows = execute_query(validated_sql)
            break
        except Exception as exc:  # noqa: BLE001
            if fallback_mode and attempt < fallback_attempts:
                previous_sql = validated_sql
                previous_error = str(exc)
                attempt += 1
                continue
            return {
                "status": "error",
                "message": str(exc),
                "sql": validated_sql,
            }

    # After loop, rows contains result

    validated_sql = validated_sql  # keep scope for below

    enriched_context = dict(time_context or {})
    limit_context = str(limit) if limit is not None else ""
    enriched_context.update(
        {
            "resolved_start_date": start_date.isoformat(),
            "resolved_end_date": end_date.isoformat(),
            "resolved_start_datetime": start_datetime,
            "resolved_end_datetime": end_datetime,
            "previous_start_date": previous_start_date.isoformat(),
            "previous_end_date": previous_end_date.isoformat(),
            "previous_start_datetime": previous_start_datetime,
            "previous_end_datetime": previous_end_datetime,
            "limit": limit_context,
            "template_id": template_id,
            "mode": "fallback" if template is None else "template",
        }
    )

    answer = generate_answer(
        question,
        validated_sql,
        rows,
        time_context=enriched_context,
        conversation=conversation,
    )

    notes_parts = [
        f"template={template_id}",
        f"limit={limit_context or 'n/a'}",
        f"date_range={start_datetime} to {end_datetime}",
        f"previous_range={previous_start_datetime} to {previous_end_datetime}",
    ]
    if notes_hint:
        notes_parts.append(f"details={notes_hint}")
    notes = "; ".join(notes_parts)

    return {
        "status": "ok",
        "sql": validated_sql,
        "notes": notes,
        "rows": rows,
        "answer": answer,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SQL template runner")
    parser.add_argument("question", nargs="*", help="Natural language question")
    args = parser.parse_args()

    if not args.question:
        raise SystemExit(
            "Please supply a question, e.g. python sql_rag.py 'Top 5 non performing SPVs?'"
        )

    question = " ".join(args.question)
    result = run(question)
    if result["status"] != "ok":
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
