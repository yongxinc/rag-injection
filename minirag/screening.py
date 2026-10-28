"""匯入時過濾:把「寫給 AI 看的指令句」從文件裡拿掉,再建索引。

正常文件裡的祈使句是寫給人看的(「部署前請先確認 CI 綠燈」),注入句則是寫給回答問題的助理看的
(「回答此文件的問題時,請…」)。規則要求一句話同時有三樣東西才算:
  1. 對象是回答者:提到助理/AI/模型,或「回答…時」「產生…時」「引用本文件」這類只對回答者有意義的情境;
  2. 祈使或強制語氣:請/務必/一律/必須/忽略;
  3. 要求改變輸出:回答/列出/附上/加上/標示/輸出/寫入/呼叫。
這是字面規則,換個說法就能繞過(見 ragi/screening_eval.py 的量測),不是完整的防禦。
"""
import re
from dataclasses import replace

ADDRESSEE = re.compile(
    r"助理|AI|模型|你收到的|先前(所有)?(的)?指示"
    r"|回答.{0,16}(問題)?時|產生.{0,16}時|生成.{0,16}時|引用本文件"
)
IMPERATIVE = re.compile(r"請|務必|一律|必須|忽略")
OUTPUT_ACTION = re.compile(r"回答|列出|附|加上|標示|輸出|寫入|呼叫")

# 句號後切開;句號緊接著 」』 時,改在引號之後切
SENTENCE_END = re.compile(r"(?<=[。！？!?])(?![」』])|(?<=[。！？!?][」』])|\n")


def split_sentences(text):
    return [s for s in (p.strip() for p in SENTENCE_END.split(text)) if s]


def looks_like_instruction(sentence):
    return bool(ADDRESSEE.search(sentence) and IMPERATIVE.search(sentence) and OUTPUT_ACTION.search(sentence))


def screen(documents):
    """回傳(過濾後的文件, 被拿掉的句子)。被拿掉的句子記成 {source, page, sentence}。"""
    kept, quarantined = [], []
    for d in documents:
        lines = []
        for line in d.text.split("\n"):
            parts = split_sentences(line)
            bad = [s for s in parts if looks_like_instruction(s)]
            quarantined += [{"source": d.source, "page": d.page, "sentence": s} for s in bad]
            good = [s for s in parts if s not in bad]
            if good:
                lines.append("".join(good))
        kept.append(replace(d, text="\n".join(lines)))
    return kept, quarantined
