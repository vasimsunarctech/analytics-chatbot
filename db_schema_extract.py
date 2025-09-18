import json
import os
from typing import Any, Dict, List
import pyodbc
from dotenv import load_dotenv

load_dotenv()


def get_connection(connection_string: str) -> pyodbc.Connection:
    return pyodbc.connect(connection_string)


def fetch_tables(cursor: pyodbc.Cursor, include_views: bool = False) -> List[Dict[str, str]]:
    object_types = "('BASE TABLE')"
    if include_views:
        object_types = "('BASE TABLE','VIEW')"

    query = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE IN {object_types}
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    rows = cursor.execute(query).fetchall()
    return [
        {
            "schema": row.TABLE_SCHEMA,
            "name": row.TABLE_NAME,
            "type": row.TABLE_TYPE,
        }
        for row in rows
    ]


def fetch_columns(cursor: pyodbc.Cursor, schema: str, table: str) -> List[Dict[str, Any]]:
    query = """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
    """
    rows = cursor.execute(query, schema, table).fetchall()
    return [
        {
            "name": row.COLUMN_NAME,
            "data_type": row.DATA_TYPE,
            "is_nullable": row.IS_NULLABLE == "YES",
            "max_length": row.CHARACTER_MAXIMUM_LENGTH,
            "numeric_precision": row.NUMERIC_PRECISION,
            "numeric_scale": row.NUMERIC_SCALE,
            "default": row.COLUMN_DEFAULT,
        }
        for row in rows
    ]


def fetch_primary_keys(cursor: pyodbc.Cursor, schema: str, table: str) -> List[str]:
    query = """
        SELECT kc.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kc
            ON tc.CONSTRAINT_NAME = kc.CONSTRAINT_NAME
            AND tc.TABLE_SCHEMA = kc.TABLE_SCHEMA
            AND tc.TABLE_NAME = kc.TABLE_NAME
        WHERE tc.TABLE_SCHEMA = ?
          AND tc.TABLE_NAME = ?
          AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY kc.ORDINAL_POSITION
    """
    rows = cursor.execute(query, schema, table).fetchall()
    return [row.COLUMN_NAME for row in rows]


def fetch_foreign_keys(cursor: pyodbc.Cursor, schema: str, table: str) -> List[Dict[str, Any]]:
    query = """
        SELECT
            fk.CONSTRAINT_NAME,
            cu.COLUMN_NAME,
            pk.TABLE_SCHEMA AS REFERENCED_SCHEMA,
            pk.TABLE_NAME AS REFERENCED_TABLE,
            ku.COLUMN_NAME AS REFERENCED_COLUMN
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS fk
        JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS pk
            ON fk.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE cu
            ON fk.CONSTRAINT_NAME = cu.CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
            ON ku.CONSTRAINT_NAME = fk.UNIQUE_CONSTRAINT_NAME
           AND ku.ORDINAL_POSITION = cu.ORDINAL_POSITION
        WHERE cu.TABLE_SCHEMA = ? AND cu.TABLE_NAME = ?
        ORDER BY fk.CONSTRAINT_NAME, cu.ORDINAL_POSITION
    """
    rows = cursor.execute(query, schema, table).fetchall()
    return [
        {
            "constraint": row.CONSTRAINT_NAME,
            "column": row.COLUMN_NAME,
            "referenced_schema": row.REFERENCED_SCHEMA,
            "referenced_table": row.REFERENCED_TABLE,
            "referenced_column": row.REFERENCED_COLUMN,
        }
        for row in rows
    ]


def extract_schema(connection_string: str, include_views: bool = False) -> Dict[str, Any]:
    with get_connection(connection_string) as connection:
        cursor = connection.cursor()
        tables = fetch_tables(cursor, include_views=include_views)

        results: Dict[str, Any] = {
            "database": connection.getinfo(pyodbc.SQL_DATABASE_NAME),
            "tables": [],
        }

        for table in tables:
            schema = table["schema"]
            name = table["name"]
            results["tables"].append(
                {
                    "schema": schema,
                    "name": name,
                    "type": table["type"],
                    "columns": fetch_columns(cursor, schema, name),
                    "primary_key": fetch_primary_keys(cursor, schema, name),
                    "foreign_keys": fetch_foreign_keys(cursor, schema, name),
                }
            )

    return results


def main() -> None:

    connection_string = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={os.getenv('SQL_SERVER')};'
        f'DATABASE={os.getenv('SQL_DATABASE')};'
        f'UID={os.getenv('DB_USERNAME')};'
        f'PWD={os.getenv('DB_PASSWORD')}'
    )

    if not connection_string:
        raise SystemExit("Set MSSQL_CONNECTION_STRING environment variable.")

    include_views = os.environ.get("INCLUDE_VIEWS", "true").lower() == "true"

    schema_data = extract_schema(connection_string, include_views=include_views)
    print(json.dumps(schema_data, indent=2, default=str))


if __name__ == "__main__":
    main()
