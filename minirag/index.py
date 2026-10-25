"""切塊與記憶體內向量檢索。"""
import math
from dataclasses import dataclass


@dataclass
class Document:
    source: str      # 檔名,回答引用時用這個
    text: str
    page: int = 1


@dataclass
class Chunk:
    source: str
    page: int
    text: str
    vector: list


def split_text(text, size=500, overlap=50):
    """切成至多 size 字元的塊,相鄰塊重疊 overlap 字元;盡量在換行處切。"""
    if size <= overlap:
        raise ValueError("size 必須大於 overlap")
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + overlap + 1, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class VectorIndex:
    def __init__(self, embedder, chunk_size=500, overlap=50):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    def add(self, documents):
        pieces = [(d, t) for d in documents for t in split_text(d.text, self.chunk_size, self.overlap)]
        vectors = self.embedder.embed([t for _, t in pieces])
        self.chunks += [Chunk(d.source, d.page, t, v) for (d, t), v in zip(pieces, vectors)]

    def search(self, query, k=4):
        qv = self.embedder.embed([query], query=True)[0]
        scored = sorted(((cosine(qv, c.vector), c) for c in self.chunks), key=lambda x: -x[0])
        return [c for _, c in scored[:k]]
