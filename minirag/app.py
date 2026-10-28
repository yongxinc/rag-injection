"""MiniRAG:檢索 → 組提示 → 模型 →(輸出檢查)→ 回答。計畫路徑另有寫入筆記庫的副作用。"""
import re
from dataclasses import dataclass, field
from datetime import date

from . import prompts
from .index import VectorIndex
from .screening import screen
from .notes import NoteStore


@dataclass
class Config:
    isolation: bool = False      # 檢索內容包進 <document>,系統提示宣告其為資料
    provenance: bool = False     # 系統提示要求只用來源標籤上的檔名引用
    output_check: bool = False   # 送出前檢查提示外洩與未檢索內容
    ingest_filter: bool = False  # 匯入時拿掉寫給 AI 看的指令句(minirag/screening.py)
    write_mode: str = "auto"     # auto:計畫直接寫進筆記庫;gated:需使用者確認才寫
    top_k: int = 4

    def __post_init__(self):
        if self.write_mode not in ("auto", "gated"):
            raise ValueError(f"write_mode 只能是 auto 或 gated,收到 {self.write_mode}")


@dataclass
class Result:
    text: str                 # 給使用者看的輸出(輸出檢查攔截時是攔截訊息)
    raw: str                  # 模型原始輸出
    system: str               # 實際送出的系統提示
    message: str              # 實際送出的使用者訊息(含參考資料)
    sources: list             # 檢索到的檔名(依排名,去重)
    blocked: str = ""         # 輸出檢查攔截原因
    pending_token: str = ""   # gated 模式下待確認寫入的 token
    writes: list = field(default_factory=list)


def _norm(t):
    return re.sub(r"\s+", "", t)


def _grams(t, n=20):
    t = _norm(t)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def build_index(documents, embedder, config=None):
    """建索引。開了 ingest_filter 就先過濾;回傳 (索引, 被拿掉的句子)。"""
    config = config or Config()
    quarantined = []
    if config.ingest_filter:
        documents, quarantined = screen(documents)
    index = VectorIndex(embedder)
    index.add(documents)
    return index, quarantined


class MiniRAG:
    def __init__(self, llm, index: VectorIndex, config=None, notes=None, today=None):
        self.llm = llm
        self.index = index
        self.config = config or Config()
        self.notes = notes if notes is not None else NoteStore()
        self.today = today or date.today().isoformat()

    def _system(self, base):
        s = base
        if self.config.isolation:
            s += prompts.ISOLATION_RULE
        if self.config.provenance:
            s += prompts.PROVENANCE_RULE
        return s

    def _check(self, text, message):
        """輸出檢查:含系統提示片段,或含知識庫中「沒進這次 context」的內容(20 字元重疊),就攔截。"""
        if any(f in text for f in prompts.PROMPT_FRAGMENTS):
            return "含系統提示片段"
        context = _norm(message)
        answer = _grams(text)
        for c in self.index.chunks:
            if any(g not in context for g in answer & _grams(c.text)):
                return "含未檢索到的文件內容"
        return ""

    def _run(self, system, message, hits):
        raw = self.llm.chat(system, message)
        blocked = self._check(raw, message) if self.config.output_check else ""
        text = f"[輸出檢查已攔截:{blocked}]" if blocked else raw
        sources = list(dict.fromkeys(c.source for c in hits))
        return Result(text=text, raw=raw, system=system, message=message, sources=sources, blocked=blocked)

    def ask(self, question):
        hits = self.index.search(question, self.config.top_k)
        message = prompts.query_message(question, hits, self.config.isolation)
        return self._run(self._system(prompts.SYSTEM_PROMPT), message, hits)

    def plan(self, goal):
        hits = self.index.search(goal, self.config.top_k)
        message = prompts.plan_message(goal, hits, self.today, self.config.isolation)
        r = self._run(self._system(prompts.PLAN_SYSTEM_PROMPT), message, hits)
        before = len(self.notes.log)
        if not r.blocked:   # 被輸出檢查攔下的計畫不寫
            if self.config.write_mode == "auto":
                self.notes.write("plan", goal, r.text)
            else:
                r.pending_token = self.notes.stage("plan", goal, r.text)
        r.writes = self.notes.log[before:]
        return r

    def confirm(self, token):
        return self.notes.confirm(token)
