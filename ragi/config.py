import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 測試資料(嵌入快取等)只寫在這個目錄
SANDBOX_DIR = ROOT / ".sandbox"
RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT  # 由 CLI 設成 results/<label>/

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("RAGI_CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("RAGI_EMBED_MODEL", "bge-m3")


def label_of(chat_model):
    """結果資料夾名稱,例如 llama3.1:8b → minirag-llama3.1-8b"""
    return "minirag-" + chat_model.replace(":", "-").replace("/", "-")


DEFAULT_RUNS = 3
