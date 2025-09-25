from __future__ import annotations

import calendar
import json
import os
import re
import time
from datetime import date, datetime, timedelta, time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

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

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = Path(os.getenv("SQL_QUERY_LOG_PATH") or (BASE_DIR / "var" / "query_logs.json"))
BETA_MESSAGE = (
    "I'm still in the beta phase, and my knowledge is a bit limited right now. Since my training data keeps updating every day, please try again later — I might have a better answer for you then! 🌱"
)
GREETING_RESPONSE = "Hello! I’m InteriseIQ. Ask me about traffic, revenue, exemptions, or costs across your projects."

DEFAULT_DATE_TEMPLATE: Dict[str, Any] = {"defaults": {"time_window": "current_fiscal_year"}}

ROUTER_SYSTEM_PROMPT = (
    "You are a routing assistant for TAS analytics questions.\n"
    "Select the most relevant query id from the provided catalogue.\n"
    "Return only JSON with keys: status, query_id, top, start_datetime, end_datetime, message.\n"
    "- status must be 'ok', 'greeting', or 'none'.\n"
    "- Use status 'greeting' for salutations or casual chit-chat.\n"
    "- Use status 'none' when no query id fits.\n"
    "- When status is 'ok', include a valid query_id, plus explicit start_datetime and end_datetime in 'YYYY-MM-DD HH:MM:SS' (IST).\n"
    "- top must be an integer when the query requires TOP; otherwise set it to null.\n"
    "- Honour user-specified date hints; otherwise use the supplied default range.\n"
    "- Keep start_datetime earlier or equal to end_datetime.\n"
    "- Reuse previous context when the question references the prior answer.\n"
    "- Do not emit Markdown code fences, explanations, or additional text."
)

SANITIZE_SYSTEM_PROMPT = (
    "You are a meticulous data formatter.\n"
    "Return strictly valid JSON with a single key 'rows'.\n"
    "Rules:\n"
    "- Preserve the original row order and column keys.\n"
    "- Never add new columns or drop existing ones.\n"
    "- For columns listed in currency_columns (only revenue, budget, or amount fields), convert each numeric value to Lakhs (value / 100000) and render it as a string prefixed with the rupee symbol '₹', keeping two decimal places and appending the word 'Lakhs'.\n"
    "  • Example: 123456789 -> '₹1,234.57 Lakhs'.\n"
    "  • Treat null/blank values as '₹0.00 Lakhs'.\n"
    "- For columns listed in percentage_columns (or whose name contains 'pct' or 'percent'), render numeric values as strings with up to two decimal places followed by '%'.\n"
    "- Leave every other field exactly as provided (no unit conversions).\n"
    "- Do not include analysis or commentary—only the JSON object."
)


class RouterDecision(TypedDict, total=False):
    status: str
    query_id: str
    top: Optional[int]
    start_datetime: Optional[str]
    end_datetime: Optional[str]
    message: Optional[str]


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


def _template_requires_limit(template: Dict[str, Any]) -> bool:
    sql_template = template.get("sql_template", "")
    return "{limit}" in sql_template


def _template_default_limit(template: Dict[str, Any]) -> Optional[int]:
    defaults = template.get("defaults", {}) or {}
    value = defaults.get("limit")
    return int(value) if value is not None else None


def _template_limit_bounds(template: Dict[str, Any]) -> Dict[str, Optional[int]]:
    limits_cfg = template.get("limits", {}) or {}
    bounds = limits_cfg.get("limit", {}) or {}
    minimum = bounds.get("min")
    maximum = bounds.get("max")
    return {
        "min": int(minimum) if minimum is not None else None,
        "max": int(maximum) if maximum is not None else None,
    }


def build_router_reference(templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reference: List[Dict[str, Any]] = []
    for template in templates:
        reference.append(
            {
                "id": template.get("id"),
                "summary": template.get("description"),
                "patterns": template.get("match_phrases", []),
                "requires_limit": _template_requires_limit(template),
                "default_limit": _template_default_limit(template),
                "limit_bounds": _template_limit_bounds(template),
            }
        )
    return reference


def _sanitize_top(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):  # Guard against True/False being treated as 1/0
        return None
    if isinstance(value, (int, float)):
        try:
            converted = int(value)
        except (TypeError, ValueError):
            return None
        return converted if converted > 0 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if not cleaned.isdigit():
            try:
                cleaned_int = int(float(cleaned))
            except (TypeError, ValueError):
                return None
            return cleaned_int if cleaned_int > 0 else None
        parsed = int(cleaned)
    return parsed if parsed > 0 else None
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _token_score(question_tokens: List[str], phrase: str) -> int:
    phrase_tokens = _normalize_text(str(phrase)).split()
    if not phrase_tokens:
        return 0
    if all(token in question_tokens for token in phrase_tokens):
        return len(phrase_tokens)
    return 0


def _manual_match_template(question: str, templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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


def _extract_limit_from_question(question: str, template: Optional[Dict[str, Any]]) -> Optional[int]:
    match = re.search(r"top\s+(\d+)", question, re.IGNORECASE)
    if match:
        limit = int(match.group(1))
    else:
        limit = _template_default_limit(template or {}) if template else None

    if template and limit is not None:
        bounds = _template_limit_bounds(template)
        if bounds["min"] is not None:
            limit = max(bounds["min"], limit)
        if bounds["max"] is not None:
            limit = min(bounds["max"], limit)
    return limit


def manual_route_question(
    question: str,
    templates: List[Dict[str, Any]],
    *,
    default_start: datetime,
    default_end: datetime,
    previous_route: Optional[Dict[str, Any]] = None,
) -> Optional[RouterDecision]:
    normalized = _normalize_text(question or "")
    if not normalized:
        return None

    selected_template = _manual_match_template(question, templates)

    follow_up_tokens = ("same", "again", "more", "top", "less", "another")
    if selected_template is None and previous_route:
        if any(token in normalized for token in follow_up_tokens):
            query_id = str(previous_route.get("query_id") or "").strip()
            if not query_id:
                return None
            selected_template = next((tpl for tpl in templates if tpl.get("id") == query_id), None)
            if selected_template is None:
                return None
            start_dt = _coerce_datetime(previous_route.get("start_datetime"), default_start)
            end_dt = _coerce_datetime(previous_route.get("end_datetime"), default_end)
            limit_value = _extract_limit_from_question(question, selected_template)
            if limit_value is None:
                limit_value = _sanitize_top(previous_route.get("top")) or _template_default_limit(selected_template)
        else:
            return None
    else:
        if selected_template is None:
            return None
        start_dt = default_start
        end_dt = default_end
        limit_value = _extract_limit_from_question(question, selected_template)

    requires_limit = _template_requires_limit(selected_template)
    if requires_limit and limit_value is None:
        limit_value = _template_default_limit(selected_template)
    if requires_limit and limit_value is None:
        return None

    bounds = _template_limit_bounds(selected_template)
    if limit_value is not None:
        if bounds["min"] is not None and limit_value < bounds["min"]:
            limit_value = bounds["min"]
        if bounds["max"] is not None and limit_value > bounds["max"]:
            limit_value = bounds["max"]

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    decision: RouterDecision = {
        "status": "ok",
        "query_id": str(selected_template.get("id", "")).strip(),
        "top": limit_value,
        "start_datetime": format_datetime_ist(start_dt),
        "end_datetime": format_datetime_ist(end_dt),
    }

    return decision


def _detect_columns(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    currency_columns: List[str] = []
    percentage_columns: List[str] = []
    if not rows:
        return currency_columns, percentage_columns
    sample_row = rows[0]
    for key in sample_row.keys():
        lowered = key.lower()
        if any(token in lowered for token in ("revenue", "budget", "amount")):
            currency_columns.append(key)
        if "pct" in lowered or "percent" in lowered:
            percentage_columns.append(key)
    return currency_columns, percentage_columns


def _format_rows_locally(
    rows: List[Dict[str, Any]],
    currency_columns: Sequence[str],
    percentage_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    formatted_rows: List[Dict[str, Any]] = []
    for row in rows:
        formatted_row: Dict[str, Any] = {}
        for key, value in row.items():
            if key in currency_columns:
                numeric = _to_float(value)
                if numeric is None:
                    formatted_row[key] = "₹0.00"
                    continue
                lakhs_value = numeric / 100000
                formatted_row[key] = f"₹{lakhs_value:,.2f}"
            elif key in percentage_columns:
                numeric = _to_float(value)
                if numeric is None:
                    formatted_row[key] = "0.00%"
                else:
                    formatted_row[key] = f"{numeric:.2f}%"
            else:
                formatted_row[key] = value
        formatted_rows.append(formatted_row)
    return formatted_rows


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("₹", "").replace(",", "").replace("Lakhs", "").replace("lakhs", "").replace("(", "").replace(")", "")
    text = text.replace("INR", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def sanitize_rows_with_llm(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not rows:
        return []

    currency_columns, percentage_columns = _detect_columns(rows)

    if not currency_columns and not percentage_columns:
        return None

    return _format_rows_locally(rows, currency_columns, percentage_columns)


def route_question_with_llm(
    question: str,
    templates: List[Dict[str, Any]],
    *,
    current_dt: datetime,
    default_start: datetime,
    default_end: datetime,
    previous_question: Optional[str] = None,
    previous_route: Optional[Dict[str, Any]] = None,
) -> RouterDecision:
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return RouterDecision(status="none")

    reference = build_router_reference(templates)
    payload: Dict[str, Any] = {
        "question": cleaned_question,
        "templates": reference,
        "default_start_datetime": format_datetime_ist(default_start),
        "default_end_datetime": format_datetime_ist(default_end),
        "current_datetime_ist": format_datetime_ist(current_dt),
    }

    if previous_question:
        payload["previous_question"] = previous_question
    if previous_route:
        payload["previous_route"] = previous_route

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            temperature=0.0,
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()
        payload_text = _strip_json_block(raw)
        data = json.loads(payload_text)
    except Exception:
        return RouterDecision(status="error", message=BETA_MESSAGE)

    status = str(data.get("status", "")).lower()
    decision: RouterDecision = {"status": status}

    if status == "ok":
        decision["query_id"] = str(data.get("query_id", "")).strip()
        decision["top"] = _sanitize_top(data.get("top"))
        start_dt = data.get("start_datetime")
        end_dt = data.get("end_datetime")
        decision["start_datetime"] = str(start_dt).strip() if start_dt else None
        decision["end_datetime"] = str(end_dt).strip() if end_dt else None
    elif status == "greeting":
        decision["message"] = GREETING_RESPONSE
    elif status == "none":
        decision["message"] = BETA_MESSAGE
    else:
        decision["status"] = "error"
        decision["message"] = BETA_MESSAGE

    return decision


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


def append_query_log(question: str, sql: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_entries: List[Dict[str, Any]] = []
        if LOG_FILE.exists():
            try:
                existing = json.loads(LOG_FILE.read_text())
                if isinstance(existing, list):
                    log_entries = existing
            except json.JSONDecodeError:
                log_entries = []
        log_entries.append({"question": question, "sql": sql})
        LOG_FILE.write_text(json.dumps(log_entries, indent=2))
    except Exception:
        # Logging must never break the primary flow
        return


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
                    "notes": (
                        "All monetary columns are strings already formatted in Indian Rupees (Lakhs). "
                        "Treat the numeric portion as Lakhs; do not scale or convert them to crores, millions, or USD. "
                        "Keep the ₹ symbol and the word 'Lakhs' exactly as provided. "
                        "Percent columns already include '%'; restate percentages with the same sign and precision."
                    ),
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
    retries: int = 0,  # noqa: ARG001 - retained for compatibility
    previous_route: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        templates = get_templates()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "message": f"Template load failed: {exc}",
        }

    previous_question: Optional[str] = None
    if conversation:
        for role, message in reversed(conversation):
            if role == "user" and message and message.strip():
                previous_question = message.strip()
                break

    current_dt = datetime.now(IST)
    default_start_date, default_end_date = resolve_default_window("current_fiscal_year", current_dt.date())
    default_start = start_of_day(default_start_date)
    default_end = end_of_day(default_end_date)

    route_decision = route_question_with_llm(
        question,
        templates,
        current_dt=current_dt,
        default_start=default_start,
        default_end=default_end,
        previous_question=previous_question,
        previous_route=previous_route,
    )

    status = route_decision.get("status")
    if status == "greeting":
        return {
            "status": "ok",
            "mode": "small_talk",
            "answer": GREETING_RESPONSE,
        }
    if status != "ok":
        manual_decision = manual_route_question(
            question,
            templates,
            default_start=default_start,
            default_end=default_end,
            previous_route=previous_route,
        )
        if manual_decision:
            route_decision = manual_decision
            status = route_decision.get("status")
        else:
            return {
                "status": "error",
                "message": BETA_MESSAGE,
            }

    query_id = (route_decision.get("query_id") or "").strip()
    template = next((tpl for tpl in templates if tpl.get("id") == query_id), None)
    if not query_id or template is None:
        return {
            "status": "error",
            "message": BETA_MESSAGE,
        }

    limit_value = route_decision.get("top")
    if limit_value is None:
        limit_value = _template_default_limit(template)

    bounds = _template_limit_bounds(template)
    if limit_value is not None:
        if bounds["min"] is not None and limit_value < bounds["min"]:
            limit_value = bounds["min"]
        if bounds["max"] is not None and limit_value > bounds["max"]:
            limit_value = bounds["max"]

    requires_limit = _template_requires_limit(template)
    if requires_limit and limit_value is None:
        return {
            "status": "error",
            "message": BETA_MESSAGE,
        }

    start_dt = _coerce_datetime(route_decision.get("start_datetime"), default_start)
    end_dt = _coerce_datetime(route_decision.get("end_datetime"), default_end)
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

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

    try:
        format_params = {
            "limit": limit_value,
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

    try:
        validated_sql = validate_sql(sql_text)
    except Exception:  # noqa: BLE001
        return {
            "status": "error",
            "message": BETA_MESSAGE,
        }

    try:
        rows = execute_query(validated_sql)
    except Exception:  # noqa: BLE001
        return {
            "status": "error",
            "message": BETA_MESSAGE,
        }

    append_query_log(question, validated_sql)

    sanitized_rows = sanitize_rows_with_llm(rows)

    enriched_context = dict(time_context or {})
    limit_context = str(limit_value) if limit_value is not None else ""
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
            "template_id": query_id,
        }
    )

    rows_for_answer = sanitized_rows if sanitized_rows is not None else rows

    answer = generate_answer(
        question,
        validated_sql,
        rows_for_answer,
        time_context=enriched_context,
        conversation=conversation,
    )

    notes_parts = [
        f"template={query_id}",
        f"limit={limit_context or 'n/a'}",
        f"date_range={start_datetime} to {end_datetime}",
        f"previous_range={previous_start_datetime} to {previous_end_datetime}",
    ]
    notes = "; ".join(notes_parts)

    return {
        "status": "ok",
        "sql": validated_sql,
        "notes": notes,
        "rows": rows,
        "sanitized_rows": sanitized_rows,
        "answer": answer,
        "query_id": query_id,
        "limit": limit_value,
        "resolved_start_datetime": start_datetime,
        "resolved_end_datetime": end_datetime,
        "previous_start_datetime": previous_start_datetime,
        "previous_end_datetime": previous_end_datetime,
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
