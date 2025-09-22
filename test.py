from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
load_dotenv()

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': os.getenv('OLLAMA_MODEL')})
vn.connect_to_mssql(odbc_conn_str=os.getenv('MSSQL_CONN'))
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

plan = vn.get_training_plan_generic(df_information_schema)
print(plan)
vn.train(plan=plan)

# while True:
#     q = input('Enter your question..  ')
#     vn.ask(question=q)

from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()