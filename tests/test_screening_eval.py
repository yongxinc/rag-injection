"""ingest_filter 留出測試的單元測試。不需要 Ollama。

這裡鎖住兩件會變成 README 數字的事實:負樣本數、零誤判;以及留出集的結構完整。
recall 本身不鎖死——規則改好、接住更多留出句是進步,不該讓測試擋下來。
"""
from minirag.screening import looks_like_instruction
from ragi.screening_eval import HELDOUT, benign_sentences, evaluate
from ragi.scenarios import BY_ID


def test_benign_set_has_no_false_positive():
    _, benign, fp = evaluate()
    assert len(benign) == 51
    assert fp == []


def test_heldout_intents_are_the_six_scenarios():
    intent_map = {"override", "leak", "manipulation", "exfil", "write", "citation"}
    assert {i for i, _, _ in HELDOUT} == intent_map
    # write 對應情境 tool_call,其餘同名;六種意圖都有留出句
    assert set(BY_ID) == {"override", "leak", "manipulation", "exfil", "tool_call", "citation"}


def test_heldout_sentences_are_not_the_corpus_payloads():
    from ragi.corpus import NORMAL
    from minirag.screening import screen
    from minirag import Document
    docs = [Document(d.file, p, i) for d in NORMAL for i, p in enumerate(d.pages, 1)]
    _, q = screen(docs)
    corpus_payloads = {x["sentence"] for x in q}
    assert all(s not in corpus_payloads for _, _, s in HELDOUT)


def test_evaluate_rows_align_with_heldout():
    rows, _, _ = evaluate()
    assert len(rows) == len(HELDOUT)
    for (intent, lang, s, caught), (hi, hl, hs) in zip(rows, HELDOUT):
        assert (intent, lang, s) == (hi, hl, hs)
        assert caught == looks_like_instruction(s)
