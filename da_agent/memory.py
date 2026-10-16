"""Persistent, curated memory.

What we store (never raw chat, never data):
  definition   - what a column/metric means            key='churn', value='status == cancelled'
  named_filter - reusable pandas .query() expression   key='active_customers', value="status != 'cancelled'"
  preference   - how the user likes answers            key='summary_granularity', value='weekly'
  correction   - a fact the user corrected             key='region_codes', value='EU = Europe, not Eurasia'

Each item is scoped to one dataset name or to 'global'. Writing the same (type, key, scope)
again UPDATES the item and keeps the old value in `history` (conflict handling).
"""
import json
import re
import time
import uuid
from pathlib import Path

from .config import MEMORY_PATH

VALID_TYPES = {"definition", "named_filter", "preference", "correction"}


def norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", str(text).lower()))


class MemoryStore:
    def __init__(self, path=MEMORY_PATH):
        self.path = Path(path)
        self._items: list[dict] = []
        self._mtime: float = -1.0
        self._load()

    @property
    def items(self) -> list[dict]:
        self._load()
        return self._items

    @items.setter
    def items(self, val: list[dict]):
        self._items = val

    # ---- persistence ----
    def _load(self):
        if self.path.exists():
            try:
                mtime = self.path.stat().st_mtime
                if mtime != self._mtime or not self._items:
                    self._items = json.loads(self.path.read_text(encoding="utf-8"))
                    self._mtime = mtime
            except Exception:
                if not self._items:
                    self._items = []
        else:
            self._items = []
            self._mtime = -1.0

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.path.exists():
            try:
                self._mtime = self.path.stat().st_mtime
            except OSError:
                pass

    # ---- writes ----
    def upsert(self, type: str, key: str, value: str, scope: str = "global") -> str:
        self._load()
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
        key, value, scope = norm_key(key), str(value).strip(), (scope or "global")
        if not key or not value:
            raise ValueError("key and value must be non-empty")
        now = time.strftime("%Y-%m-%d %H:%M")
        for it in self._items:
            if (it["type"], it["key"], it["scope"]) == (type, key, scope):
                if it["value"] == value:
                    return "unchanged"
                it.setdefault("history", []).append({"value": it["value"], "replaced": now})
                it["value"], it["updated"] = value, now
                self._save()
                return "updated"
        self._items.append({"id": uuid.uuid4().hex[:6], "type": type, "key": key, "value": value,
                           "scope": scope, "created": now, "updated": now, "history": []})
        self._save()
        return "added"

    def delete(self, ref: str, type: str | None = None) -> int:
        """Delete by id or by key (optionally restricted to a type). Returns #removed."""
        self._load()
        ref_n = norm_key(ref)
        keep = [it for it in self._items
                if not ((it["id"] == ref or it["key"] == ref_n) and (type is None or it["type"] == type))]
        removed = len(self._items) - len(keep)
        if removed:
            self._items = keep
            self._save()
        return removed

    def clear(self):
        self._items = []
        self._save()

    # ---- reads ----
    def eligible(self, dataset: str) -> list[dict]:
        self._load()
        if not dataset:
            return list(self._items)
        return [it for it in self._items if it.get("scope") in ("global", dataset)]

    def named_filters(self, dataset: str) -> dict:
        self._load()
        out = {}
        for it in sorted(self.eligible(dataset), key=lambda i: i.get("scope") != "global"):  # dataset overrides global
            if it.get("type") == "named_filter":
                out[it["key"]] = it["value"]
        return out

    def relevant(self, dataset: str, text: str, limit: int = 15) -> list[dict]:
        self._load()
        q = _tokens(text)
        def score(it):
            return len(q & _tokens(it["key"] + " " + it["value"]))
        pool = self.eligible(dataset)
        return sorted(pool, key=lambda i: (score(i), i["updated"]), reverse=True)[:limit]

    @staticmethod
    def render(items: list[dict]) -> str:
        if not items:
            return "(nothing saved yet)"
        return "\n".join(f"[{i['id']}] {i['type']} '{i['key']}' ({i['scope']}): {i['value']}" for i in items)


MEMORY = MemoryStore()
