"""驅動真正的 MyMindAgent。

隔離措施:
- CHROMA_DIR 指向 .sandbox/,不碰 MyMind 的 chroma_db
- NOTION_TOKEN / NOTION_DATABASE_ID 清除,且 agent.notion 換成只記錄的假物件,不打任何 Notion API
- agent.calendar 換成固定回傳的假物件,不讀 Google 憑證
- 不載入 MyMind 的 .env
在 LLM 邊界放一個 spy,用來:抓最終提示、套用提示類防禦、套用輸出檢查。
"""
import importlib
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field

from . import config, defenses


@dataclass
class Trace:
    message: str
    defense: str
    prompt: str = ""            # 實際送進模型的查詢/計畫提示(已套用防禦)
    sources: list = field(default_factory=list)  # 提示中 [來源: X] 的檔名
    raw_answer: str = ""
    answer: str = ""
    blocked_by_output_check: str = ""
    notion_calls: list = field(default_factory=list)  # {type, payload, executed, blocked_by}
    intent: str = ""


class FakeNotion:
    enabled = True

    def __init__(self, target):
        self.t = target

    def _write(self, kind, payload):
        entry = {"type": kind, "payload": payload, "executed": True, "blocked_by": None}
        # tool_confirm:有副作用的寫入需使用者確認。測試中模擬使用者只下了一個指令,不認得這次寫入內容 → 拒絕
        if self.t.defense == "tool_confirm" and not self.t.confirm_fn(kind, payload):
            entry["executed"], entry["blocked_by"] = False, "user_confirmation"
        self.t.notion_calls.append(entry)
        return "https://notion.invalid/sandbox" if entry["executed"] else ""

    def write_ingest_summary(self, source, kb, chunks, summary):
        return self._write("ingest", f"{source}\n{summary}")

    def write_study_plan(self, goal, plan, calendar_events=None):
        return self._write("plan", f"{goal}\n{plan}")


class FakeCalendar:
    credentials_exist = False
    authenticated = False
    enabled = False

    def get_events(self, days=21):
        return {"available": False, "events": [], "busy": [], "range_days": days}


class MyMindTarget:
    def __init__(self):
        config.SANDBOX_DIR.mkdir(exist_ok=True)
        self.chroma_root = config.SANDBOX_DIR / "chroma"
        os.environ["CHROMA_DIR"] = str(self.chroma_root)
        os.environ.pop("NOTION_TOKEN", None)
        os.environ.pop("NOTION_DATABASE_ID", None)
        sys.path.insert(0, str(config.MYMIND_BACKEND))
        self.mod = importlib.import_module("agent")
        from langchain_core.runnables import RunnableLambda

        self.defense = "baseline"
        self.confirm_fn = lambda kind, payload: False
        self.notion_calls = []
        self.prompts = []
        self.kb_texts = []
        self.agent = self.mod.MyMindAgent()
        self.agent.notion = FakeNotion(self)
        self.agent.calendar = FakeCalendar()
        self._real_llm = self.agent.llm
        self.agent.llm = RunnableLambda(self._spy)

    # ── 乾淨知識庫 ────────────────────────────────────────────────────────────
    def load(self, docs):
        from langchain_core.documents import Document

        shutil.rmtree(self.chroma_root, ignore_errors=True)
        run_dir = self.chroma_root / uuid.uuid4().hex[:8]
        self.mod.CHROMA_DIR = str(run_dir)
        self.agent._kbs = {}
        self.kb_texts = []
        kb_of = {"論文": "thesis", "會議": "meetings", "學習": "study", "技術": "tech"}
        by_kb = {}
        for d in docs:
            for page in d.pages:
                by_kb.setdefault(kb_of[d.category], []).append(
                    Document(page_content=page, metadata={"source": d.file})
                )
                self.kb_texts.append(page)
        for kb, ds in by_kb.items():
            chunks = self.agent.splitter.split_documents(ds)
            self.agent._get_or_create_kb(kb).add_documents(chunks)

    # ── LLM 邊界 ──────────────────────────────────────────────────────────────
    def _spy(self, prompt_value):
        prompt = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        kind = defenses.kind_of(prompt)
        if kind == "other":
            return self._real_llm.invoke(prompt)
        prompt = defenses.transform_prompt(prompt, kind, self.defense)
        out = self._real_llm.invoke(prompt)
        self.prompts.append((kind, prompt, out))
        if self.defense == "output_check":
            reason = defenses.check_output(out, prompt, self.kb_texts)
            if reason:
                self.blocked = reason
                return f"[輸出檢查已攔截:{reason}]"
        return out

    def ask(self, message, defense):
        import re

        self.defense = defense
        self.notion_calls = []
        self.prompts = []
        self.blocked = ""
        res = self.agent.run(message)
        r = res["result"]
        t = Trace(message=message, defense=defense, intent=res["intent"])
        t.answer = r.get("answer") or r.get("plan") or ""
        if self.prompts:
            _, t.prompt, t.raw_answer = self.prompts[-1]
            t.sources = re.findall(r"\[來源: (.*?)\]", t.prompt) or re.findall(r'<document source="(.*?)">', t.prompt)
        t.blocked_by_output_check = self.blocked
        t.notion_calls = list(self.notion_calls)
        return t
