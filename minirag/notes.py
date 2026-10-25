"""筆記庫:系統唯一有副作用的地方(相當於寫進使用者的 Notion)。只存在記憶體,選擇性附加到本機檔案。"""
import json
import uuid
from pathlib import Path


class PendingNotFound(KeyError):
    """未知、已使用或已取消的確認 token。"""


class NoteStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.notes = []     # 實際寫入的筆記
        self.log = []       # 每次寫入嘗試:{kind, title, body, executed, pending_token}
        self._pending = {}

    def _persist(self, kind, title, body):
        note = {"kind": kind, "title": title, "body": body}
        self.notes.append(note)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(note, ensure_ascii=False) + "\n")
        return len(self.notes) - 1

    def write(self, kind, title, body):
        """自動寫入模式:直接寫。"""
        self.log.append({"kind": kind, "title": title, "body": body, "executed": True, "pending_token": None})
        return self._persist(kind, title, body)

    def stage(self, kind, title, body):
        """需確認模式:先記下來,等使用者確認才寫。"""
        token = uuid.uuid4().hex
        self._pending[token] = (kind, title, body)
        self.log.append({"kind": kind, "title": title, "body": body, "executed": False, "pending_token": token})
        return token

    def confirm(self, token):
        try:
            kind, title, body = self._pending.pop(token)
        except KeyError:
            raise PendingNotFound(token) from None
        for entry in self.log:
            if entry["pending_token"] == token:
                entry["executed"] = True
        return self._persist(kind, title, body)

    def cancel(self, token):
        if self._pending.pop(token, None) is None:
            raise PendingNotFound(token)
