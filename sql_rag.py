from __future__ import annotations

import calendar
import json
import os
import re
import time
from datetime import date, datetime, timedelta
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


def extract_limit(question: str, template: Dict[str, Any]) -> int:
    defaults = template.get("defaults", {})
    limits = template.get("limits", {}).get("limit", {})
    default_limit = int(defaults.get("limit", 5))
    min_limit = int(limits.get("min", 1))
    max_limit = int(limits.get("max", max(default_limit, 10)))

    match = re.search(r"top\s+(\d+)", question, re.IGNORECASE)
    if match:
        limit = int(match.group(1))
    else:
        limit = default_limit

    limit = max(min_limit, min(limit, max_limit))
    return limit


def subtract_months(value: date, months: int) -> date:
    year = value.year
    month = value.month - months
    day = value.day
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    if day > last_day:
        day = last_day
    return date(year, month, day)


def previous_calendar_month(value: date) -> Tuple[date, date]:
    first_this_month = value.replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    start_prev_month = last_prev_month.replace(day=1)
    return start_prev_month, last_prev_month


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


def extract_date_range(
    question: str,
    template: Dict[str, Any],
    today: Optional[date] = None,
) -> Tuple[date, date]:
    today = today or datetime.now(IST).date()
    question_lc = question.lower()

    numeric_month = re.search(r"last\s+(\d+)\s+months?", question_lc)
    if "last month" in question_lc:
        start, end = previous_calendar_month(today)
    elif "last year" in question_lc:
        start, end = fiscal_year_range(today, delta=-1)
    elif numeric_month:
        months = max(1, int(numeric_month.group(1)))
        start = subtract_months(today, months)
        end = today
    else:
        default_window = template.get("defaults", {}).get("time_window")
        start, end = resolve_default_window(default_window, today)

    return start, end


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
                record[columns[idx]] = float(value)
            else:
                record[columns[idx]] = value
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
    try:
        templates = get_templates()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "message": f"Template load failed: {exc}",
        }

    template = match_template(question, templates)
    if template is None:
        return {
            "status": "error",
            "message": "I do not yet have a SQL mapped for that request.",
        }

    limit = extract_limit(question, template)
    start_date, end_date = extract_date_range(question, template)

    start_datetime = f"{start_date.strftime('%Y-%m-%d')} 00:00:00"
    end_datetime = f"{end_date.strftime('%Y-%m-%d')} 23:59:59"

    previous_start_date = shift_years(start_date, -1)
    previous_end_date = shift_years(end_date, -1)
    previous_start_datetime = f"{previous_start_date.strftime('%Y-%m-%d')} 00:00:00"
    previous_end_datetime = f"{previous_end_date.strftime('%Y-%m-%d')} 23:59:59"

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
        sql = template["sql_template"].format(**format_params)
    except KeyError as exc:
        return {
            "status": "error",
            "message": f"Template '{template.get('id')}' is missing placeholder {exc}.",
        }

    try:
        validated_sql = validate_sql(sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "message": str(exc),
            "sql": sql,
        }

    try:
        rows = execute_query(validated_sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "message": str(exc),
            "sql": validated_sql,
        }

    enriched_context = dict(time_context or {})
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
            "limit": str(limit),
            "template_id": template.get("id"),
        }
    )

    answer = generate_answer(
        question,
        validated_sql,
        rows,
        time_context=enriched_context,
        conversation=conversation,
    )

    notes = (
        "template={tid}; limit={limit}; date_range={start_start} to {end_end}; "
        "previous_range={prev_start} to {prev_end}"
    ).format(
        tid=template.get("id"),
        limit=limit,
        start_start=start_datetime,
        end_end=end_datetime,
        prev_start=previous_start_datetime,
        prev_end=previous_end_datetime,
    )

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
