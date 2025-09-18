from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
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
    "Return your answer strictly as JSON with the shape {\"sql\": \"...\", \"notes\": \"...\"}.\n"
    "If you cannot generate a safe query respond with {\"sql\": null, \"notes\": \"reason\"}.\n""",
)
SYSTEM_ANSWER = os.environ.get(
    "SQL_ANSWER_PROMPT",
    "You are a helpful analyst. Summarise SQL query results for business users concisely, cite totals, highlight trends, and mention limits."
)
SCHEMA_PATH = Path(os.environ.get("SQL_SCHEMA_PATH", "schema.json"))
MAX_RESULT_ROWS = int(os.environ.get("SQL_RESULT_LIMIT", "200"))

CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('DB_USERNAME')};"
    f"PWD={os.getenv('DB_PASSWORD')}"
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OLLAMA_API_BASE)


@dataclass
class SQLGenerationResult:
    sql: Optional[str]
    notes: str


# --- Schema utilities ------------------------------------------------------


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


def generate_sql(question: str, schema_context: str) -> SQLGenerationResult:
    messages = [
        {"role": "system", "content": SYSTEM_SQL_GEN},
        {"role": "user", "content": f"Schema:\n{schema_context}\n\nQuestion: {question}"},
    ]

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


def generate_answer(question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_ANSWER},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "sql": sql,
                    "rows": rows,
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


def run(question: str) -> Dict[str, Any]:
    schema_context = load_schema_snippet()
    sql_result = generate_sql(question, schema_context)

    if not sql_result.sql:
        return {
            "status": "error",
            "message": sql_result.notes or "Model could not generate a query.",
        }

    validated_sql = validate_sql(sql_result.sql)
    rows = execute_query(validated_sql)
    answer = generate_answer(question, validated_sql, rows)

    return {
        "status": "ok",
        "sql": validated_sql,
        "notes": sql_result.notes,
        "rows": rows,
        "answer": answer,
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
