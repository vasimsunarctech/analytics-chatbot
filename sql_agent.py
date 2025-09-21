from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyodbc
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

SCHEMA_PATH = Path(os.environ.get("SQL_SCHEMA_PATH", "schema.json"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:3.8b")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

MSSQL_DRIVER = os.environ.get("MSSQL_DRIVER", "{ODBC Driver 17 for SQL Server}")
MSSQL_SERVER = os.environ.get("SQL_SERVER", "localhost")
MSSQL_DATABASE = os.environ.get("SQL_DATABASE", "master")
MSSQL_USERNAME = os.environ.get("DB_USERNAME")
MSSQL_PASSWORD = os.environ.get("DB_PASSWORD")
MSSQL_TRUST = os.environ.get("MSSQL_TRUST", "yes")

CONNECTION_STRING = (
    f"DRIVER={MSSQL_DRIVER};"
    f"SERVER={MSSQL_SERVER};"
    f"DATABASE={MSSQL_DATABASE};"
    + (
        f"UID={MSSQL_USERNAME};PWD={MSSQL_PASSWORD};"
        if MSSQL_USERNAME and MSSQL_PASSWORD
        else "Trusted_Connection=yes;"
    )
)

SCHEMA_CACHE_SECONDS = int(os.environ.get("SQL_AGENT_SCHEMA_CACHE", "300"))

_TABLE_HINTS: List[Dict[str, Any]] = [
    {
        "table": "daily_transaction_final",
        "keywords": [
            "traffic",
            "revenue",
            "collection",
            "journey",
            "lane",
            "pcu",
        ],
    },
    {
        "table": "Exemption Data",
        "keywords": ["exemption", "wim", "refund"],
    },
    {
        "table": "Cost of Operation Master",
        "keywords": ["cost", "opex", "maintenance", "saving"],
    },
    {
        "table": "Total Revenue Master",
        "keywords": ["plan", "planned", "tariff", "revenue"],
    },
    {
        "table": "Budget_Traffic",
        "keywords": ["budget", "traffic", "plan"],
    },
    {
        "table": "Budget Head report",
        "keywords": ["budget", "head", "plan"],
    },
    {
        "table": "daily_transaction_mtd_ytd",
        "keywords": ["mtd", "ytd", "qtd"],
    },
    {
        "table": "Force Majeure 1",
        "keywords": ["force majeure", "claim", "days", "covid"],
    },
]

_schema_cache: Dict[str, Any] = {"timestamp": 0.0, "tables": {}}

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text())


def select_relevant_tables(question: str, max_tables: int = 6) -> List[str]:
    question_lower = question.lower()
    scored: List[Tuple[int, str]] = []
    for hint in _TABLE_HINTS:
        score = sum(1 for kw in hint["keywords"] if kw in question_lower)
        if score > 0:
            scored.append((score, hint["table"]))
    scored.sort(reverse=True)
    ordered: List[str] = []
    seen: set[str] = set()
    for score, table in scored:
        if table not in seen:
            ordered.append(table)
            seen.add(table)
        if len(ordered) >= max_tables:
            break
    return ordered


def _fetch_table_schema_from_db(table_name: str) -> str:
    with pyodbc.connect(CONNECTION_STRING) as conn:
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

        if not columns:
            return f"No schema found for table '{table_name}'."

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

        lines = [f"Table {table_name} columns:"]
        for col in columns:
            default = col.COLUMN_DEFAULT if col.COLUMN_DEFAULT is not None else "None"
            pk_marker = " PK" if col.COLUMN_NAME in pk_cols else ""
            lines.append(
                f"  - {col.COLUMN_NAME} ({col.DATA_TYPE}, nullable={col.IS_NULLABLE}, default={default}){pk_marker}"
            )
        return "\n".join(lines)


def build_schema_context(question: str) -> str:
    now = time.time()
    cache_ok = now - _schema_cache["timestamp"] < SCHEMA_CACHE_SECONDS

    table_names = select_relevant_tables(question)
    if not table_names:
        table_names = ["daily_transaction_final"]

    context_lines: List[str] = []
    for table in table_names:
        if cache_ok and table in _schema_cache["tables"]:
            schema_text = _schema_cache["tables"][table]
        else:
            schema_text = _fetch_table_schema_from_db(table)
            _schema_cache["tables"][table] = schema_text
            _schema_cache["timestamp"] = now
        context_lines.append(schema_text)

    return "\n\n".join(context_lines)


# ---------------------------------------------------------------------------
# SQL generation and execution
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned.lower().startswith("select"):
        raise ValueError("Only SELECT statements are permitted.")
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|EXEC|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK)\b",
        re.IGNORECASE,
    )
    if forbidden.search(cleaned):
        raise ValueError("Detected forbidden keyword in SQL statement.")
    if cleaned.count(";") > 0:
        raise ValueError("Multiple SQL statements are not allowed.")
    return cleaned


def execute_sql(sql: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with pyodbc.connect(CONNECTION_STRING) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data: List[Dict[str, Any]] = []
        for row in rows:
            record = {}
            for idx, value in enumerate(row):
                record[columns[idx]] = value
            data.append(record)
        return data, columns


REASONING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior TAS analytics engineer.\n"
        "Your job: analyse the schema, reason through the question, then produce SQL.\n"
        "Rules:\n"
        "- Only use tables/columns that exist in the schema context.\n"
        "- If a name has spaces, wrap it in square brackets exactly as shown (e.g., [Cost of Operation Master]).\n"
        "- If numeric measures are stored as text (nvarchar), convert them using TRY_CONVERT(decimal(18,2), [column]).\n"
        "- Only aggregate numeric values.\n"
        "- Prefer project_name over project_code when the user mentions names.\n"
        "- If data is unavailable, say so instead of guessing.\n"
        "Respond strictly in JSON with keys: reasoning, sql, notes.\n""",
    ),
    (
        "human",
        """Schema:\n{schema}\n\n"
        "Time context:\n{time_context}\n\n"
        "Recent conversation:\n{conversation}\n\n"
        "Question:\n{question}""",
    ),
])


def format_context_list(items: Optional[Sequence[Tuple[str, str]]]) -> str:
    if not items:
        return "(none)"
    lines = []
    for role, content in items[-6:]:
        label = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


@dataclass
class AgentResult:
    status: str
    message: str
    sql: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None
    columns: Optional[List[str]] = None


def run_sql_agent(
    question: str,
    *,
    conversation: Optional[Sequence[Tuple[str, str]]] = None,
    time_context: Optional[Dict[str, str]] = None,
) -> AgentResult:
    try:
        schema_context = build_schema_context(question)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="error", message=f"Schema lookup failed: {exc}")

    time_context_lines = (
        "\n".join(f"- {k}: {v}" for k, v in (time_context or {}).items())
        if time_context
        else "(not provided)"
    )

    conversation_text = format_context_list(conversation)

    chain = REASONING_PROMPT | llm
    response = chain.invoke(
        {
            "schema": schema_context,
            "time_context": time_context_lines,
            "conversation": conversation_text,
            "question": question,
        }
    )

    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        return AgentResult(status="error", message=f"Could not parse agent response: {exc}")

    sql = payload.get("sql")
    notes = payload.get("notes", "")

    if not sql:
        return AgentResult(status="error", message=notes or "Agent did not produce SQL.")

    try:
        validated_sql = validate_sql(sql)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="error", message=str(exc), sql=sql, notes=notes)

    try:
        rows, columns = execute_sql(validated_sql)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="error", message=str(exc), sql=validated_sql, notes=notes)

    return AgentResult(
        status="ok",
        message="ok",
        sql=validated_sql,
        rows=rows,
        notes=notes,
        columns=columns,
    )
