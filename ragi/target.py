"""驅動 MiniRAG。每次嘗試都用新的筆記庫,寫入紀錄只屬於那一次。"""
from dataclasses import dataclass, field

from minirag import CachedEmbedder, Document, MiniRAG, NoteStore, OllamaClient, VectorIndex

from . import config
from .defenses import config_for


@dataclass
class Trace:
    message: str
    setting: str
    mode: str = "ask"           # ask:問答;plan:讀書計畫(有寫入副作用)
    prompt: str = ""            # 實際送出的系統提示 + 使用者訊息
    sources: list = field(default_factory=list)  # 檢索到的檔名
    raw_answer: str = ""
    answer: str = ""
    blocked_by_output_check: str = ""
    writes: list = field(default_factory=list)  # {kind, title, body, executed, pending_token}


class MiniRAGTarget:
    def __init__(self, chat_model=None, embed_model=None):
        self.client = OllamaClient(chat_model or config.CHAT_MODEL, embed_model or config.EMBED_MODEL,
                                   url=config.OLLAMA_URL)
        cache = config.SANDBOX_DIR / f"embeddings-{self.client.embed_model.replace(':', '_')}.json"
        self.embedder = CachedEmbedder(self.client, cache)
        self.index = None

    def load(self, docs):
        """乾淨知識庫:每個情境重建一次索引(嵌入有快取,不會重算)。"""
        self.index = VectorIndex(self.embedder)
        self.index.add([Document(d.file, page, i) for d in docs for i, page in enumerate(d.pages, 1)])

    def ask(self, message, setting, mode="ask"):
        app = MiniRAG(self.client, self.index, config_for(setting), NoteStore())
        r = app.plan(message) if mode == "plan" else app.ask(message)
        return Trace(message=message, setting=setting, mode=mode, prompt=r.system + "\n\n" + r.message,
                     sources=r.sources, raw_answer=r.raw, answer=r.text,
                     blocked_by_output_check=r.blocked, writes=r.writes)
