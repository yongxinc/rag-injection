"""MiniRAG:本專案自帶的最小 RAG 系統,作為注入測試的被測對象。只依賴標準函式庫與本機 Ollama。"""
from .app import Config, MiniRAG, Result, build_index
from .index import Document, VectorIndex, split_text
from .notes import NoteStore, PendingNotFound
from .ollama import CachedEmbedder, OllamaClient

__all__ = ["Config", "MiniRAG", "Result", "build_index", "Document", "VectorIndex", "split_text",
           "NoteStore", "PendingNotFound", "CachedEmbedder", "OllamaClient"]
