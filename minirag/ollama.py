"""本機 Ollama 的最小客戶端:對話、嵌入、版本與模型資訊。只用標準函式庫。"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

# mxbai-embed-large 的查詢端要加這個前綴,文件端不加(模型卡的建議用法)
QUERY_PREFIX = {"mxbai-embed-large": "Represent this sentence for searching relevant passages: "}


class OllamaClient:
    def __init__(self, chat_model="llama3.1:8b", embed_model="bge-m3",
                 url="http://localhost:11434", temperature=0.1, num_predict=1024, timeout=600):
        # 只允許打本機:測試請求不送到雲端
        if not re.match(r"https?://(localhost|127\.0\.0\.1)(:\d+)?/?$", url):
            raise ValueError(f"只允許本機 Ollama,收到 {url}")
        self.url = url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.num_predict = num_predict  # 生成長度上限,避免失控的長輸出佔住本機唯一的模型
        self.timeout = timeout

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def chat(self, system, user):
        body = {
            "model": self.chat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        return self._request("/api/chat", body)["message"]["content"]

    def embed(self, texts, query=False):
        prefix = QUERY_PREFIX.get(self.embed_model.split(":")[0], "") if query else ""
        return self._request("/api/embed", {"model": self.embed_model, "input": [prefix + t for t in texts]})["embeddings"]

    def info(self):
        """執行環境紀錄用:Ollama 版本與兩個模型的 digest。"""
        tags = {m["name"]: m["digest"] for m in self._request("/api/tags")["models"]}

        def digest(name):
            return tags.get(name) or tags.get(name + ":latest")
        return {
            "ollama_version": self._request("/api/version")["version"],
            "chat_model": self.chat_model, "chat_model_digest": digest(self.chat_model),
            "embed_model": self.embed_model, "embed_model_digest": digest(self.embed_model),
            "temperature": self.temperature, "num_predict": self.num_predict,
        }


class CachedEmbedder:
    """把文件端嵌入存到檔案,重跑時不必重算。查詢端也快取(同一個問題會問很多次)。"""

    def __init__(self, client, cache_path=None):
        self.client = client
        self.path = Path(cache_path) if cache_path else None
        self.cache = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}

    def _key(self, text, query):
        return hashlib.sha256(f"{self.client.embed_model}|{int(query)}|{text}".encode()).hexdigest()

    def embed(self, texts, query=False):
        keys = [self._key(t, query) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            for (k, _), v in zip(missing, self.client.embed([t for _, t in missing], query=query)):
                self.cache[k] = v
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.cache))
        return [self.cache[k] for k in keys]
