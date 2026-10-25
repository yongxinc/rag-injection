"""實驗設定 → MiniRAG 的 Config。防禦本身實作在 minirag/(prompts.py 與 app.py)。"""
from minirag import Config

DEFENSES = ["baseline", "isolation", "provenance", "output_check", "tool_confirm"]
SETTINGS = DEFENSES + ["control"]

CONFIGS = {
    "baseline": Config(),
    "isolation": Config(isolation=True),
    "provenance": Config(provenance=True),
    "output_check": Config(output_check=True),
    # 寫入前要使用者確認。測試中使用者只要了一份計畫,不會去確認寫入
    "tool_confirm": Config(write_mode="gated"),
    # 無注入對照組:設定同 baseline,差別在知識庫(見 runner.knowledge_base)
    "control": Config(),
}


def config_for(setting):
    return CONFIGS[setting]
