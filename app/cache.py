import time


class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store = {}

    def get(self, key):
        now = time.time()
        row = self._store.get(key)
        if not row:
            return None
        if (now - row["ts"]) > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return row["value"]

    def set(self, key, value):
        self._store[key] = {"ts": time.time(), "value": value}

    def clear(self):
        self._store.clear()
