from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
import pyodbc
from openai import OpenAI
load_dotenv()

# --- Configuration ---------------------------------------------------------

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_SQL_GEN = os.environ.get(
    "SQL_SYSTEM_PROMPT",
    """You are an expert SQL assistant for a Microsoft SQL Server data warehouse.\n"
    "Generate only SELECT queries. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, MERGE, or EXEC statements.\n"
    "Prefer using TOP 50 to keep result sets manageable unless the user explicitly requests exact counts.\n"
    "Use the provided schema description to pick correct table and column names.\n"
    "When table or column names contain spaces, wrap them in square brackets exactly as shown (e.g., [Cost of Operation Master]). Never invent underscores or abbreviations.\n"
    "Return your answer strictly as JSON with the shape {\"sql\": \"...\", \"notes\": \"...\"}.\n"
    "If you cannot generate a safe query respond with {\"sql\": null, \"notes\": \"reason\"}.\n""",
)
SYSTEM_ANSWER = os.environ.get(
    "SQL_ANSWER_PROMPT",
    "You are a helpful analyst. Summarise SQL query results for business users concisely, cite totals, highlight trends, and mention limits."
)
SCHEMA_PATH = Path(os.environ.get("SQL_SCHEMA_PATH", "schema.json"))
MAX_RESULT_ROWS = int(os.environ.get("SQL_RESULT_LIMIT", "200"))
SCHEMA_CACHE_SECONDS = int(os.environ.get("SQL_SCHEMA_CACHE_SECONDS", "300"))
SQL_MAX_RETRIES = int(os.environ.get("SQL_RETRY_ATTEMPTS", "1"))

BUSINESS_GUIDELINES = """
Terminology guide:
- "SPV" means project or plaza special purpose vehicle; map to columns like project_name, project_description, Project Name, or Project - Code.
- Actual toll revenue is typically stored in collected_fare_tms (daily_transaction_* tables) or collected_fare_tms (Exemption Data). Expected or planned revenue is expected_amount or fields in Total Revenue Master / Budget tables.
- Traffic counts are in traffic_count columns or *_traffic fields (FTD, MTD, QTD, YTD) in daily_transaction_* tables; PCU stands for passenger car units (pcu_traffic_* columns).
- Budget/plan information lives in Budget_Traffic (Traffic Budgets by SPV, Year, Month) and Budget Head report.
- Exemption analysis uses Exemption Data (wim_amount, refund_amount, final_wim_amt, exemption_category, etc.).
- Cost of operations metrics come from Cost of Operation Master (columns like Column26, Revenue Model, etc.) and Total Revenue Master for planned vs actual comparisons.
- Force Majeure events are recorded in Force Majeure 1 table (affected period, claims, approvals by level).
- Accident trends use tables containing columns like fatal, major, minor if available; otherwise respond that data is unavailable.
- Table and column names may contain spaces. Use them exactly as shown and wrap them in square brackets, e.g. [Cost of Operation Master]. Never invent underscores or abbreviations.

SQL best practices:
- Always return aggregated results with aliases e.g. SUM(collected_fare_tms) AS total_actual_revenue.
- When filtering by date phrases like "today" use CAST(transaction_date_time AS DATE) = CAST(GETDATE() AS DATE).
- For month/year comparisons use DATEFROMPARTS or YEAR(), MONTH() on transaction_date_time.
- When computing variance use formula (actual - planned) / NULLIF(planned,0) * 100.
- Always include ORDER BY clauses when returning "top" results and use TOP N.
- Use only single SELECT statement.
"""

TABLE_HINTS: List[Dict[str, Any]] = [
    {
        "table": "daily_transaction_final",
        "keywords": [
            "revenue",
            "traffic",
            "collection",
            "variance",
            "today",
            "daily",
            "ftd",
            "mtd",
            "qtd",
            "ytd",
            "spv",
            "top",
        ],
        "description": "Granular daily transactions with collected_fare_tms (actual revenue), expected_amount (planned), traffic_count, PCU metrics, project_name, plaza_name.",
    },
    {
        "table": "daily_transaction_mtd_ytd",
        "keywords": ["mtd", "ytd", "traffic"],
        "description": "Aggregated traffic counts (FTD, MTD, QTD, YTD) by project/plaza and date.",
    },
    {
        "table": "Budget_Traffic",
        "keywords": ["plan", "planned", "budget", "traffic", "tariff"],
        "description": "Traffic budgets by SPV (project), year, month, vehicle class.",
    },
    {
        "table": "Total Revenue Master",
        "keywords": ["plan", "planned", "revenue", "tariff", "yoY"],
        "description": "Revenue master data with planned figures and project metadata.",
    },
    {
        "table": "Cost of Operation Master",
        "keywords": ["cost", "opex", "mmr", "maintenance", "saving", "exceeding"],
        "description": "Operational cost data per project/SPV with classification columns.",
    },
    {
        "table": "Force Majeure 1",
        "keywords": ["force majeure", "claim", "days", "approved", "dispute", "settlement"],
        "description": "Force majeure events with status by approval level and affected period.",
    },
    {
        "table": "Exemption Data",
        "keywords": ["exemption", "wim", "refund", "pass", "variance"],
        "description": "Vehicle exemptions and revenue adjustments per transaction with wim_amount, refund_amount, final_wim_amt.",
    },
    {
        "table": "Finance Master",
        "keywords": ["valuation", "network impact", "finance"],
        "description": "Finance master data for each project/SPV including revenue model, cluster, concession details.",
    },
    {
        "table": "Route Operation Master",
        "keywords": ["route", "operation", "network", "impact"],
        "description": "Operational metadata for each project, including lane counts, models, integrators.",
    },
]


CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('DB_USERNAME')};"
    f"PWD={os.getenv('DB_PASSWORD')}"
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OLLAMA_API_BASE)

_schema_cache: Dict[str, Any] = {"timestamp": 0.0, "tables": {}}


@dataclass
class SQLGenerationResult:
    sql: Optional[str]
    notes: str


# --- Schema utilities ------------------------------------------------------


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text())


def _fetch_table_schema_from_db(table_name: str) -> str:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """,
            table_name,
        )
        columns = cursor.fetchall()

        cursor.execute(
            """
            SELECT KU.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KU
              ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
            WHERE TC.TABLE_NAME = ? AND TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """,
            table_name,
        )
        pk_cols = {row.COLUMN_NAME for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT 
                fk_col.name AS FK_column,
                pk_tab.name AS PK_table,
                pk_col.name AS PK_column
            FROM sys.foreign_keys fk
            INNER JOIN sys.foreign_key_columns fkc 
                ON fkc.constraint_object_id = fk.object_id
            INNER JOIN sys.tables fk_tab 
                ON fk_tab.object_id = fkc.parent_object_id
            INNER JOIN sys.columns fk_col 
                ON fkc.parent_object_id = fk_col.object_id 
                AND fkc.parent_column_id = fk_col.column_id
            INNER JOIN sys.tables pk_tab 
                ON pk_tab.object_id = fkc.referenced_object_id
            INNER JOIN sys.columns pk_col 
                ON fkc.referenced_object_id = pk_col.object_id 
                AND fkc.referenced_column_id = pk_col.column_id
            WHERE fk_tab.name = ?
            """,
            table_name,
        )
        foreign_keys = [
            f"{row.FK_column} → {row.PK_table}.{row.PK_column}"
            for row in cursor.fetchall()
        ]

    if not columns:
        return f"No schema details found for table '{table_name}'."

    lines = [f"Table {table_name} columns:"]
    for col in columns:
        default = col.COLUMN_DEFAULT if col.COLUMN_DEFAULT is not None else "None"
        pk_marker = " PK" if col.COLUMN_NAME in pk_cols else ""
        lines.append(
            f"  - {col.COLUMN_NAME} ({col.DATA_TYPE}, nullable={col.IS_NULLABLE}, default={default}){pk_marker}"
        )

    if foreign_keys:
        lines.append("  Foreign keys:")
        for fk in foreign_keys:
            lines.append(f"    * {fk}")

    return "\n".join(lines)


def fetch_table_schema(table_name: str) -> str:
    now = time.time()
    cached = _schema_cache["tables"].get(table_name)
    if cached and now - _schema_cache["timestamp"] < SCHEMA_CACHE_SECONDS:
        return cached

    try:
        schema_text = _fetch_table_schema_from_db(table_name)
    except Exception:
        # Fall back to schema.json if live lookup fails
        schema_data = load_schema()
        tbl = next((t for t in schema_data.get("tables", []) if t.get("name") == table_name), None)
        if not tbl:
            schema_text = f"Schema not available for '{table_name}'."
        else:
            cols = tbl.get("columns", [])
            parts = [f"Table {table_name} columns:"]
            for col in cols:
                parts.append(
                    f"  - {col['name']} ({col['data_type']}, nullable={'YES' if col.get('is_nullable') else 'NO'})"
                )
            schema_text = "\n".join(parts)

    _schema_cache["tables"][table_name] = schema_text
    _schema_cache["timestamp"] = now
    return schema_text


def select_relevant_tables(question: str, max_tables: int = 6) -> List[str]:
    q = question.lower()
    scored: List[tuple[int, str]] = []

    for hint in TABLE_HINTS:
        score = sum(1 for kw in hint["keywords"] if kw in q)
        if score > 0:
            scored.append((score, hint["table"]))

    scored.sort(reverse=True)
    ordered = []
    seen = set()
    for _, table in scored:
        if table not in seen:
            ordered.append(table)
            seen.add(table)
        if len(ordered) >= max_tables:
            break
    return ordered


def build_schema_context(
    question: str,
    time_context: Optional[Dict[str, str]] = None,
) -> str:
    data = {}
    if SCHEMA_PATH.exists():
        try:
            data = load_schema()
        except Exception:
            data = {}

    table_names = select_relevant_tables(question)

    defaults = ["daily_transaction_final", "Budget_Traffic", "Exemption Data"]
    for default in defaults:
        if default not in table_names:
            table_names.append(default)

    context_lines = []
    if data:
        context_lines.append(f"Database: {data.get('database', 'unknown')}")

    for table_name in table_names:
        hint = next((h for h in TABLE_HINTS if h["table"] == table_name), None)
        if hint:
            context_lines.append(f"Table {table_name}: {hint['description']}")
        else:
            context_lines.append(f"Table {table_name}.")

        context_lines.append(fetch_table_schema(table_name))

    context_lines.append("Business notes:\n" + BUSINESS_GUIDELINES.strip())

    if time_context:
        formatted_time = "\n".join(f"  - {k}: {v}" for k, v in time_context.items())
        context_lines.append("Time context:\n" + formatted_time)

    return "\n".join(context_lines)


def load_schema_snippet(limit_tables: int = 15, limit_columns: int = 12) -> str:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    data = json.loads(SCHEMA_PATH.read_text())
    tables = data.get("tables", [])[:limit_tables]

    lines: List[str] = [f"Database: {data.get('database', 'unknown')}"]
    for table in tables:
        cols = table.get("columns", [])[:limit_columns]
        column_parts = [f"{col['name']} ({col['data_type']})" for col in cols]
        col_text = ", ".join(column_parts)
        lines.append(f"- {table['name']} [{table['type']}] -> {col_text}")
        if len(table.get("columns", [])) > limit_columns:
            lines.append("  ...")

    if len(data.get("tables", [])) > limit_tables:
        lines.append("(Schema truncated)")

    return "\n".join(lines)


# --- LLM helpers -----------------------------------------------------------


def _compose_sql_prompt(
    question: str,
    schema_context: str,
    conversation: Optional[Sequence[Tuple[str, str]]],
    time_context: Optional[Dict[str, str]],
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
) -> List[Dict[str, str]]:
    sections = [f"Schema:\n{schema_context}"]

    if time_context:
        sections.append(
            "Time context:" + "\n" + "\n".join(f"{k}: {v}" for k, v in time_context.items())
        )

    if conversation:
        history_lines = []
        for role, text in conversation[-6:]:
            label = "USER" if role == "user" else "ASSISTANT"
            history_lines.append(f"{label}: {text}")
        sections.append("Recent conversation:\n" + "\n".join(history_lines))

    if previous_sql or previous_error:
        sections.append(
            "Previous attempt:\n"
            + (f"SQL: {previous_sql}\n" if previous_sql else "")
            + (f"Error: {previous_error}\n" if previous_error else "")
            + "Please provide a corrected SQL statement."
        )

    sections.append(f"Question: {question}")

    user_message = "\n\n".join(sections)

    return [
        {"role": "system", "content": SYSTEM_SQL_GEN},
        {"role": "user", "content": user_message},
    ]


def generate_sql(
    question: str,
    schema_context: str,
    conversation: Optional[Sequence[Tuple[str, str]]] = None,
    time_context: Optional[Dict[str, str]] = None,
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
) -> SQLGenerationResult:
    messages = _compose_sql_prompt(
        question,
        schema_context,
        conversation,
        time_context,
        previous_sql,
        previous_error,
    )

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.0,
        messages=messages,
    )

    raw = response.choices[0].message.content.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse SQL generator response: {raw}") from exc

    sql = payload.get("sql")
    notes = payload.get("notes", "")
    return SQLGenerationResult(sql=sql, notes=notes)


FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|EXEC|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned.lower().startswith("select"):
        raise ValueError("Only SELECT statements are permitted.")

    if FORBIDDEN_PATTERN.search(cleaned):
        raise ValueError("Detected forbidden keyword in SQL statement.")

    if cleaned.count(";") > 0:
        raise ValueError("Multiple statements are not allowed.")

    return cleaned


# --- Database helpers ------------------------------------------------------


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
        if cursor.fetchone():  # extra rows beyond limit
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


# --- Command-line workflow -------------------------------------------------


def run(
    question: str,
    *,
    conversation: Optional[Sequence[Tuple[str, str]]] = None,
    time_context: Optional[Dict[str, str]] = None,
    retries: int = SQL_MAX_RETRIES,
) -> Dict[str, Any]:
    schema_context = build_schema_context(question, time_context=time_context)
    sql_result = generate_sql(
        question,
        schema_context,
        conversation=conversation,
        time_context=time_context,
    )

    if not sql_result.sql:
        return {
            "status": "error",
            "message": sql_result.notes or "Model could not generate a query.",
        }

    attempt_sql = validate_sql(sql_result.sql)

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            rows = execute_query(attempt_sql)
            answer = generate_answer(
                question,
                attempt_sql,
                rows,
                time_context=time_context,
                conversation=conversation,
            )
            return {
                "status": "ok",
                "sql": attempt_sql,
                "notes": sql_result.notes,
                "rows": rows,
                "answer": answer,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            sql_result = generate_sql(
                question,
                schema_context,
                conversation=conversation,
                time_context=time_context,
                previous_sql=attempt_sql,
                previous_error=str(exc),
            )
            if not sql_result.sql:
                break
            attempt_sql = validate_sql(sql_result.sql)

    message = "I wasn't able to execute that query successfully. Please try rephrasing."
    if last_error:
        message += f" (Error: {last_error})"
    return {
        "status": "error",
        "message": message,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SQL RAG assistant")
    parser.add_argument("question", nargs="*", help="Natural language question")
    args = parser.parse_args()

    if not args.question:
        raise SystemExit("Please supply a question, e.g. python sql_rag.py 'What was today\'s revenue?' ")

    question = " ".join(args.question)
    result = run(question)
    if result["status"] != "ok":
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
