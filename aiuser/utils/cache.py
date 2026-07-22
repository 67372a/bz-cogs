from collections import OrderedDict


class Cache(OrderedDict):
    """A simple bounded LRU cache.

    Behaves like a regular dict (missing keys raise ``KeyError``), with
    least-recently-used eviction once ``limit`` entries are exceeded.
    Built on ``OrderedDict`` so deletions/pops/clears stay consistent —
    unlike the previous implementation which kept a parallel key list that
    could go stale and crash eviction with a ``KeyError``.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("Cache limit must be >= 1")
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        if key in self:
            # Refresh recency without changing size
            super().move_to_end(key)
        elif len(self) >= self.limit:
            # Evict least-recently-used entry
            super().popitem(last=False)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)  # raises KeyError for missing keys
        super().move_to_end(key)
        return value
