from langchain_community.utilities import SQLDatabase 
from config import ENV
from typing_extensions import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START , StateGraph , END
import json
import pyodbc
import os
import time
from tabulate import tabulate

# os.environ["OLLAMA_NO_GPU"] = "1"
# ____DATABASE_____

DRIVER = ENV.MSSQL_DRIVER
SERVER = ENV.MSSQL_SERVER
DATABASE = ENV.MSSQL_DATABASE
TRUST = ENV.MSSQL_TRUST


conn_str = (
    f"DRIVER={DRIVER};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection={TRUST};"
)

# Schema cache
_MAX_SCHEMA_AGE = 60  # seconds
_schema_cache = {"schema": {}, "ts": 0}


def get_db():
    """Get a new DB connection."""
    try:
        conn = pyodbc.connect(conn_str)
        print("✅ Connected to SQL Server!")
        return conn
    except Exception as e:
        print("❌ DB Connection Error:", str(e))
        raise

def get_schema(table_names):
    """
    Get schema for one or more tables using pyodbc.
    Returns formatted schema(s) in tabular format.
    """

    if isinstance(table_names, str):
        table_names = [table_names]  # make it iterable

    formatted_schemas = []
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()

    for table_name in table_names:
        print("get_schema for", table_name)

        if (
            table_name in _schema_cache["schema"]
            and (now - _schema_cache["ts"] < _MAX_SCHEMA_AGE)
        ):
            formatted_schemas.append(_schema_cache["schema"][table_name])
            continue

        # Get columns
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

        # Primary keys
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
        pk_cols = [row.COLUMN_NAME for row in cursor.fetchall()]

        # Foreign keys
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
        fk_list = [
            f"{row.FK_column} → {row.PK_table}.{row.PK_column}"
            for row in cursor.fetchall()
        ]

        # Format schema
        column_data = []
        for col in columns:
            column_data.append({
                "Column Name": col.COLUMN_NAME,
                "Data Type": col.DATA_TYPE,
                "Nullable": col.IS_NULLABLE,
                "Default": col.COLUMN_DEFAULT if col.COLUMN_DEFAULT else "None",
                "Primary Key": "✔" if col.COLUMN_NAME in pk_cols else "",
            })

        schema_output = tabulate(column_data, headers="keys", tablefmt="grid")
        fk_output = "\nForeign Keys: " + ", ".join(fk_list) if fk_list else ""

        formatted_schema = f"Schema for '{table_name}' table:\n{schema_output}{fk_output}"

        _schema_cache["schema"][table_name] = formatted_schema
        _schema_cache["ts"] = now

        formatted_schemas.append(formatted_schema)

    cursor.close()
    conn.close()

    # Join multiple schemas together
    return "\n\n".join(formatted_schemas)

# _______SQL_AGENT____
class State(TypedDict):
    user_query: str
    table_name : str
    schema: str
    reasoning: str
    sql: str
    results: str


llm = ChatOllama(model=ENV.OLLAMA_MODEL,base_url=ENV.OLLAMA_URL, temperature=0)

# "mssql+pyodbc://<servername>/<dbname>?driver=ODBC+Driver+18+for+SQL+Server&Trusted_connection=yes"
conn = f"mssql+pyodbc://{ENV.MSSQL_SERVER}/{ENV.MSSQL_DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_connection={ENV.MSSQL_TRUST}"

db = SQLDatabase.from_uri(conn)

# print(get_schema("revenue"))

query_tool = QuerySQLDatabaseTool(db=db)


table_detect_prompt = ChatPromptTemplate.from_messages([
   ("system", 
     "You are a database assistant. "
     "The database has two tables: revenue and sells. And both tables has relationships."
     "Your task: extract ONLY the table name from the user query. "
     "Respond strictly in JSON format like this: {{\"table\": \"<table_name>\"}} OR if you want more than one table Respond like this {{\"table\":\"[<table_names>]\"}}"),
    ("human", "User Query: {user_query}")
])

reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant that explains how to answer a SQL question."
     "When user query contain name of place or name of a person, then always prefer name columns over id columns (because you cant geuss the id which not provided to you)"
    ),
    ("human", "User Query: {user_query}\nSchema:\n{schema}\nExplain step by step reasoning before writing SQL.")
])

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL generator. Always return only SQL code, nothing else."),
    ("human", "User Query: {user_query}\nSchema:\n{schema}\nReasoning:\n{reasoning}\nNow write the SQL query.")
])

def call_table_detect(state: State):
    print("now detecting the table..")
    chain = table_detect_prompt | llm
    resp = chain.invoke({"user_query": state["user_query"]})
    table = json.loads(resp.content)
    table_name = table.get("table")
    return {"table_name": table_name}

def call_schema(state: State):
    print("now getting the schema for ",state["table_name"])
    schema = get_schema(state["table_name"])
    print(schema)
    return {"schema": schema}

def call_reasoning(state: State):
    print("NOW LLM Thinking ...")
    chain = reasoning_prompt | llm
    reasoning = chain.invoke({
        "user_query": state["user_query"],
        "schema": state["schema"]
    })
    print(reasoning)
    return {"reasoning": reasoning.content}

def call_sql(state: State):
    print("Now LLM making SQL based on User query")
    chain = sql_prompt | llm
    sql = chain.invoke({
        "user_query": state["user_query"],
        "schema": state["schema"],
        "reasoning": state["reasoning"]
    })
    print(sql)
    return {"sql": sql.content.strip()}

def call_execute(state: State):
    print("NOw getting the result from database")
    result = query_tool.run(state["sql"])
    print(result)
    return {"results": result}

# ----- Build Graph -----
graph = StateGraph(State)
graph.add_node("detect_table", call_table_detect)
graph.add_node("get_schema", call_schema)
graph.add_node("llm_reason", call_reasoning)
graph.add_node("sql_generate", call_sql)
graph.add_node("sql_execute", call_execute)

graph.add_edge(START, "detect_table")
graph.add_edge("detect_table", "get_schema")
graph.add_edge("get_schema", "llm_reason")
graph.add_edge("llm_reason", "sql_generate")
graph.add_edge("sql_generate", "sql_execute")
graph.add_edge("sql_execute", END)

sql_agent = graph.compile()
