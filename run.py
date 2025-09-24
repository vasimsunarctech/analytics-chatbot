from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
load_dotenv()

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': os.getenv('OPENAI_API_KEY'), 'model': os.getenv('OLLAMA_MODEL')})
vn.connect_to_mssql(odbc_conn_str=os.getenv('MSSQL_CONN'))

from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run(port=8000)