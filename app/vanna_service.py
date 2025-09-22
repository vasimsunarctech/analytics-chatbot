import os
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


_vn = None


def get_vanna():
    global _vn
    if _vn is not None:
        return _vn

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    persist_dir = os.getenv("CHROMA_DIR", "./var/chroma")
    os.makedirs(persist_dir, exist_ok=True)

    _vn = MyVanna(config={"model": model, "persist_directory": persist_dir})

    # DB connect (SQLite default)
    # _vn.connect_to_sqlite("tesla_motors_data.db")
    # For SQL Server:
    print(os.getenv("MSSQL_CONN"))
    _vn.connect_to_mssql(odbc_conn_str=os.getenv("MSSQL_CONN"))

    try:
        # df_ddl = _vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")
        df_information_schema = _vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = _vn.get_training_plan_generic(df_information_schema)
        _vn.train(plan=plan)
        _vn.train(
            question="What are the top 5 performing SPVs.",
            sql="""
                SELECT TOP (5)
                    project,
                    SUM(
                        CAST(
                            (COALESCE(cash,0) 
                            + COALESCE(fastag,0) 
                            + COALESCE(others,0) 
                            + COALESCE(overload,0) 
                            + COALESCE(pass_amount,0)) 
                            - COALESCE(double_penalty,0)
                        AS BIGINT)
                    ) AS revenue,
                    SUM(CAST(budget AS BIGINT)) AS budget,
                    ROUND(
                        (
                            (
                                SUM(
                                    CAST(
                                        (COALESCE(cash,0) 
                                        + COALESCE(fastag,0) 
                                        + COALESCE(others,0) 
                                        + COALESCE(overload,0) 
                                        + COALESCE(pass_amount,0)) 
                                        - COALESCE(double_penalty,0)
                                    AS BIGINT)
                                ) 
                                - SUM(CAST(budget AS BIGINT))
                            ) 
                            / NULLIF(SUM(CAST(budget AS BIGINT)), 0.0)
                        ) * 100.0, 
                        2
                    ) AS performance_pct
                FROM dbo.ods_tmsdata_revenue
                WHERE [date] >= '2025-09-01 00:00:00'
                AND [date] <= '2025-09-01 23:59:59'
                GROUP BY project
                ORDER BY performance_pct DESC;
            """,
        )

        # for ddl in df_ddl["sql"].to_list():
        #     _vn.train(ddl=ddl)
    except Exception:
        pass

    return _vn
