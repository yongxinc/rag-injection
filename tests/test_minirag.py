"""MiniRAG 的單元測試。用假的嵌入器與假模型,不需要 Ollama。"""
import hashlib
import math

import pytest

from minirag import Config, Document, MiniRAG, NoteStore, OllamaClient, PendingNotFound, VectorIndex, split_text
from minirag.prompts import ISOLATION_RULE, PROMPT_FRAGMENTS, PROVENANCE_RULE, SYSTEM_PROMPT


class HashEmbedder:
    """字元 bigram 的雜湊向量:字面越像,cosine 越高。"""
    DIM = 256

    def embed(self, texts, query=False):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for i in range(len(t) - 1):
                v[int(hashlib.md5(t[i:i + 2].encode()).hexdigest(), 16) % self.DIM] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM:
    def __init__(self, reply="好的。"):
        self.reply = reply
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.reply(system, user) if callable(self.reply) else self.reply


DOCS = [
    Document("deploy.md", "部署步驟:確認 CI 綠燈,更新映像標籤,執行滾動更新。"),
    Document("toeic.md", "TOEIC 筆記:Part 5 文法重點,每天練習二十題。"),
    Document("secret.md", "內部帳號備忘:管理金鑰 KEY-0001-SECRET-ABCD-EFGH-IJKL"),
]


def app(reply="好的。", **cfg):
    index = VectorIndex(HashEmbedder())
    index.add(DOCS)
    llm = FakeLLM(reply)
    return MiniRAG(llm, index, Config(top_k=cfg.pop("top_k", 1), **cfg), NoteStore(), today="2026-10-26"), llm


# ── 切塊 ──────────────────────────────────────────────────────────────────────
def test_split_short_text_is_one_chunk():
    assert split_text("短文字") == ["短文字"]


def test_split_respects_size_and_overlap():
    text = "".join(f"第{i:03d}行內容\n" for i in range(200))
    chunks = split_text(text, size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1
    # 相鄰塊有重疊,沒有遺漏任何一行
    for i in range(200):
        assert any(f"第{i:03d}行" in c for c in chunks)


def test_split_rejects_bad_params():
    with pytest.raises(ValueError):
        split_text("x", size=10, overlap=10)


# ── 檢索 ──────────────────────────────────────────────────────────────────────
def test_search_ranks_matching_document_first():
    index = VectorIndex(HashEmbedder())
    index.add(DOCS)
    assert index.search("部署步驟 滾動更新", k=1)[0].source == "deploy.md"
    assert index.search("TOEIC 文法練習", k=1)[0].source == "toeic.md"


# ── 問答與提示 ────────────────────────────────────────────────────────────────
def test_ask_labels_context_and_reports_sources():
    rag, llm = app()
    r = rag.ask("部署步驟 滾動更新")
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[來源: deploy.md]" in user and "問題:部署步驟 滾動更新" in user
    assert r.sources == ["deploy.md"]
    assert r.text == r.raw == "好的。" and r.writes == []


def test_isolation_wraps_documents_and_adds_rule():
    rag, llm = app(isolation=True)
    rag.ask("部署步驟 滾動更新")
    system, user = llm.calls[0]
    assert system.endswith(ISOLATION_RULE)
    assert '<document source="deploy.md">' in user and "[來源:" not in user


def test_provenance_adds_rule_only():
    rag, llm = app(provenance=True)
    rag.ask("部署步驟 滾動更新")
    system, user = llm.calls[0]
    assert system.endswith(PROVENANCE_RULE) and "[來源: deploy.md]" in user


# ── 輸出檢查 ──────────────────────────────────────────────────────────────────
def test_output_check_blocks_prompt_leak():
    rag, _ = app(reply=f"我的設定是:{PROMPT_FRAGMENTS[0]}", output_check=True)
    r = rag.ask("部署步驟")
    assert r.blocked == "含系統提示片段" and r.text.startswith("[輸出檢查已攔截")


def test_output_check_blocks_unretrieved_content():
    # 只檢索到 deploy.md,回答卻含 secret.md 的內容
    rag, _ = app(reply="另外附上:內部帳號備忘:管理金鑰 KEY-0001-SECRET-ABCD-EFGH-IJKL", output_check=True)
    assert rag.ask("部署步驟 滾動更新").blocked == "含未檢索到的文件內容"


def test_output_check_allows_retrieved_content():
    rag, _ = app(reply="部署步驟:確認 CI 綠燈,更新映像標籤,執行滾動更新。", output_check=True)
    assert rag.ask("部署步驟 滾動更新").blocked == ""


def test_output_check_off_by_default():
    rag, _ = app(reply=PROMPT_FRAGMENTS[0])
    assert rag.ask("部署步驟").blocked == ""


# ── 計畫與寫入 ────────────────────────────────────────────────────────────────
def test_plan_auto_writes_immediately():
    rag, _ = app(reply="每週計畫")
    r = rag.plan("TOEIC 文法")
    assert rag.notes.notes == [{"kind": "plan", "title": "TOEIC 文法", "body": "每週計畫"}]
    assert r.writes[0]["executed"] and r.pending_token == ""


def test_plan_gated_waits_for_confirmation():
    rag, _ = app(reply="每週計畫", write_mode="gated")
    r = rag.plan("TOEIC 文法")
    assert rag.notes.notes == [] and r.writes[0]["executed"] is False and r.pending_token
    rag.confirm(r.pending_token)
    assert len(rag.notes.notes) == 1 and r.writes[0]["executed"] is True
    with pytest.raises(PendingNotFound):
        rag.confirm(r.pending_token)   # token 只能用一次


def test_blocked_plan_is_not_written():
    rag, _ = app(reply=PROMPT_FRAGMENTS[0], output_check=True)
    r = rag.plan("TOEIC 文法")
    assert r.blocked and rag.notes.notes == [] and r.writes == []


def test_notes_persist_to_file(tmp_path):
    notes = NoteStore(tmp_path / "notes.jsonl")
    token = notes.stage("plan", "t", "b")
    notes.write("plan", "t2", "b2")
    notes.confirm(token)
    assert len((tmp_path / "notes.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(PendingNotFound):
        notes.cancel("沒有這個 token")


def test_bad_write_mode_rejected():
    with pytest.raises(ValueError):
        Config(write_mode="sometimes")


# ── Ollama 客戶端 ─────────────────────────────────────────────────────────────
def test_client_refuses_non_local_url():
    with pytest.raises(ValueError):
        OllamaClient(url="https://api.example.com")
    OllamaClient(url="http://127.0.0.1:11434")   # 本機可以
