import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# GCP / Vertex
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")   # leave blank if local, use gcloud default project
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "denim-finrag-vertex")

# Models & API Keys
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = _env_int("TAVILY_MAX_RESULTS", 5)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "Financial-Analyst-Agent")

os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
if LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

# Local filesystem paths
DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data")
CHROMA_PERSIST_DIR = _env_path(
    "CHROMA_PERSIST_DIR",
    Path(os.getenv("VECTOR_DB_LOCAL_DIR", str(PROJECT_ROOT / "vector_db"))),
)
CHUNKS_JSONL_PATH = _env_path("CHUNKS_JSONL_PATH", CHROMA_PERSIST_DIR / "chunks.jsonl")
PARSED_ELEMENTS_PATH = _env_path(
    "PARSED_ELEMENTS_PATH",
    CHROMA_PERSIST_DIR / "artifacts" / "parsed_elements.jsonl",
)
INGEST_SUMMARY_PATH = _env_path(
    "INGEST_SUMMARY_PATH",
    CHROMA_PERSIST_DIR / "artifacts" / "ingestion_summary.json",
)

# Vector DB cache path (Cloud Run friendly)
VECTOR_DB_PREFIX = os.getenv("VECTOR_DB_PREFIX", "vector_db")
VECTOR_DB_LOCAL_DIR = str(CHROMA_PERSIST_DIR)
UPLOAD_VECTOR_DB_TO_GCS = _env_flag("UPLOAD_VECTOR_DB_TO_GCS", "false")
