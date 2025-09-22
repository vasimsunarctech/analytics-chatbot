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
    _vn.connect_to_mssql(connection_string=os.getenv("MSSQL_CONN"))

    try:
        # df_ddl = _vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")
        df_information_schema = _vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = _vn.get_training_plan_generic(df_information_schema)
        _vn.train(plan=plan)
        # for ddl in df_ddl["sql"].to_list():
        #     _vn.train(ddl=ddl)
    except Exception:
        pass

    return _vn
