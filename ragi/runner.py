import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from . import config
from .corpus import NORMAL, PRIVATE, sanitized
from .scenarios import BY_ID, SCENARIOS
from .target import MyMindTarget, Trace


def _anchor(s):
    """注入頁拿掉注入句後的第一行:對照組用它確認同一頁仍有被檢索到。"""
    doc = next(d for d in NORMAL if d.file == s.injected_file)
    return sanitized(doc).pages[1].split("\n")[0]


def retrieval_ok(s, t, control=False):
    keys = ([_anchor(s)] if control else list(s.payload_keys)) + list(s.context_keys)
    return all(k in t.prompt for k in keys)


def knowledge_base(s, control=False):
    """乾淨知識庫 = 正常文件 + 機密備忘 + 該情境的一份注入文件(對照組則是拿掉注入句的同一份)。"""
    docs = [d for d in NORMAL if not d.injected]
    injected = next(d for d in NORMAL if d.file == s.injected_file)
    return docs + [sanitized(injected) if control else injected, PRIVATE]


def _git_state(backend):
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(backend), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {}


ENV_PACKAGES = ["sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "torch",
                "chromadb", "langchain-core", "langchain-community", "langgraph"]


def home_relative(path):
    """家目錄底下的路徑存成 ~/...,結果檔才不會帶出使用者名稱。"""
    p, home = Path(path).expanduser(), Path.home()
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def environment():
    """這次執行實際用的直譯器與關鍵套件版本(嵌入與檢索結果取決於它們)。"""
    def ver(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    try:
        ollama = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        ollama = None
    return {"python": home_relative(sys.executable), "prefix": home_relative(sys.prefix), "python_version": platform.python_version(),
            "packages": {n: ver(n) for n in ENV_PACKAGES}, "ollama": ollama}


def run_defense(target, setting, runs=config.DEFAULT_RUNS, scenario_ids=None, progress=print):
    """setting 是四種防禦之一、baseline,或 control(無注入對照:不加防禦、文件裡沒有注入句)。"""
    control = setting == "control"
    defense = "baseline" if control else setting
    results = []
    for s in SCENARIOS:
        if scenario_ids and s.id not in scenario_ids:
            continue
        target.load(knowledge_base(s, control))
        for i in range(runs):
            t = target.ask(s.question, defense)
            hit = retrieval_ok(s, t, control)
            ok = bool(s.judge(t))
            results.append({
                "scenario": s.id, "category": s.category, "run": i + 1,
                "injected_retrieved": hit, "success": ok,
                "intent": t.intent, "sources": t.sources,
                "blocked_by_output_check": t.blocked_by_output_check,
                "notion_calls": t.notion_calls,
                "plan_contains_payload": s.id == "tool_call" and "toeic-fastpass" in t.answer,
                "answer": t.answer, "raw_answer": t.raw_answer,
                "prompt": t.prompt,
            })
            progress(f"[{setting}] {s.id} #{i + 1}: intent={t.intent} retrieved={hit} success={ok}")
    out = {
        "defense": setting, "target": "MyMind",
        "target_path": home_relative(config.MYMIND_BACKEND), "target_git": _git_state(config.MYMIND_BACKEND),
        "environment": environment(),
        "chat_model": target.mod.LLM_MODEL, "embed_model": target.mod.EMBED_MODEL, "runs": runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{setting}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out, path


def rejudge(path):
    """用目前的判定函式重算已存結果的 success(判定修正後不必重跑模型)。回傳改變的筆數。"""
    data = json.loads(path.read_text())
    changed = 0
    for r in data["results"]:
        t = Trace(message="", defense=data["defense"], answer=r["answer"],
                  sources=r["sources"], notion_calls=r["notion_calls"])
        ok = bool(BY_ID[r["scenario"]].judge(t))
        if ok != r["success"]:
            changed += 1
            r["success"] = ok
    data["rejudged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return changed


def new_target():
    return MyMindTarget()
