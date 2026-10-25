import argparse

from . import config
from .defenses import DEFENSES, SETTINGS
from .report import baseline_check, load, sanity_warnings, table
from .runner import new_target, rejudge, run_defense
from .scenarios import SCENARIOS


def main():
    p = argparse.ArgumentParser(prog="ragi", description="RAG 間接提示注入測試(僅限測試作者自有系統)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="以指定防禦設定執行整組測試")
    r.add_argument("--defense", default="baseline", help=f"{'|'.join(SETTINGS)}|all(四種防禦+基線,不含 control)")
    r.add_argument("--runs", type=int, default=config.DEFAULT_RUNS)
    r.add_argument("--chat-model", default=config.CHAT_MODEL, help="Ollama 對話模型")
    r.add_argument("--scenarios", help="逗號分隔的情境 id,預設全部")
    r.add_argument("--label", help="結果資料夾名稱,預設 minirag-<對話模型>")

    rp = sub.add_parser("report", help="輸出對照表")
    rp.add_argument("--chat-model", default=config.CHAT_MODEL)
    rp.add_argument("--label")

    rj = sub.add_parser("rejudge", help="用目前的判定函式重算已存結果(不重跑模型)")
    rj.add_argument("--chat-model", default=config.CHAT_MODEL)
    rj.add_argument("--label")

    sub.add_parser("list", help="列出情境")
    a = p.parse_args()

    if a.cmd == "list":
        for s in SCENARIOS:
            print(f"{s.id:13} {s.category:8} {s.injected_file:30} 判定: {s.criterion}")
        return
    if a.cmd in ("run", "report", "rejudge"):
        config.RESULTS_DIR = config.RESULTS_ROOT / (a.label or config.label_of(a.chat_model))
    if a.cmd == "rejudge":
        for p in [config.RESULTS_DIR / f"{d}.json" for d in SETTINGS]:
            if p.exists():
                print(f"{p.name}: {rejudge(p)} 筆判定改變")
        a.cmd = "report"
    if a.cmd == "run":
        if a.defense != "all" and a.defense not in SETTINGS:
            raise SystemExit(f"未知設定 {a.defense},可用:{', '.join(SETTINGS)}, all")
        target = new_target(a.chat_model)
        ids = a.scenarios.split(",") if a.scenarios else None
        for d in (DEFENSES if a.defense == "all" else [a.defense]):
            _, path = run_defense(target, d, a.runs, ids)
            print(f"→ {path}")
        a.cmd = "report"
    if a.cmd == "report":
        data = load()
        if not data:
            raise SystemExit("找不到結果,請先執行 run")
        print(table(data))
        print("\n" + baseline_check(data))
        for w in sanity_warnings(data):
            print(w)


main()
