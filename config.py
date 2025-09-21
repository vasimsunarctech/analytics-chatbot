import os 
from dotenv import load_dotenv 

load_dotenv()

class ENV():
    MSSQL_DRIVER=os.getenv("MSSQL_DRIVER")
    MSSQL_SERVER=os.getenv("MSSQL_SERVER")
    MSSQL_DATABASE=os.getenv("MSSQL_DATABASE")
    MSSQL_TRUST=os.getenv("MSSQL_TRUST")
    OLLAMA_URL=os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL=os.getenv("OLLAMA_MODEL")
