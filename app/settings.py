import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = Path(os.getenv("APP_DATA_DIR") or ROOT / "data")
DOCUMENTS = Path(os.getenv("DOCS_DIR") or ROOT / "docs")
EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL_READY = bool(LLM_MODEL and LLM_API_KEY)
