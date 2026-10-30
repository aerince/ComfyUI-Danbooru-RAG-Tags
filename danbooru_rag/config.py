"""Paths owned by this node; no dependency on another custom-node package."""
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.local.json"


def _settings():
    if not CONFIG_PATH.is_file():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"RAG configuration must be a JSON object: {CONFIG_PATH}")
    return settings


def resolve_path(value):
    path = Path(os.path.expandvars(str(value).strip().strip('\"\''))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _configured(key, default):
    return str(resolve_path(os.environ.get("AERINCE_RAG_" + key.upper()) or _settings().get(key) or default))


DEFAULT_MODEL = _configured("model_path", "models/chat-model.gguf")
DEFAULT_EMBEDDING = _configured("embedding_model_path", "models/Qwen3-Embedding-0.6B-Q8_0.gguf")
DEFAULT_SERVER = _configured("llama_server_path", shutil.which("llama-server") or "bin/llama-server.exe")
DEFAULT_DATABASE = _configured("database_path", "data/danbooru_tags.xlsx")
INDEX_ROOT = Path(_configured("index_root", "data/rag_index"))


def resolve_database_path(value=""):
    if not value or value == "danbooru_tags.xlsx":
        return DEFAULT_DATABASE
    return str(resolve_path(value))
