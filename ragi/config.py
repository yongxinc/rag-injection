import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 所有測試資料(向量庫、假 Notion 紀錄)只能寫在這個目錄,與真實知識庫完全隔離
SANDBOX_DIR = ROOT / ".sandbox"
RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT  # 由 CLI 依目標版本設成 results/<label>/

# 被測的 MyMind 後端(只讀取其程式碼;其 .env、chroma_db、Google 憑證一律不碰)
MYMIND_BACKEND = Path(os.environ.get("MYMIND_BACKEND", Path.home() / "Documents/mymind/backend"))


def label_of(backend):
    """結果資料夾名稱,例如 ~/Documents/mymind/backend → Documents-mymind"""
    b = Path(backend)
    return f"{b.parent.parent.name}-{b.parent.name}"

DEFAULT_RUNS = 3
