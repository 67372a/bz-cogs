import asyncio
import hashlib
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger("red.bz_cogs.aiuser")

# Cache TTL: 30 minutes
IMAGE_CACHE_TTL = 1800  # seconds

# Maximum number of entries before pruning
IMAGE_CACHE_MAX_SIZE = 100


class ImageCache:
    """TTL-based cache for downloaded images, keyed by SHA-256 of the URL.

    Stores raw image bytes (pre-base64) so the same cached entry can be
    reused across any caller (image parsing, reference images, etc.).
    Entries expire after IMAGE_CACHE_TTL seconds (default: 30 minutes).
    """

    def __init__(self):
        self._cache: dict[str, Tuple[bytes, str, float]] = {}  # sha256 -> (data, content_type, cached_at)
        self._lock = asyncio.Lock()

    def _sha256(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _is_expired(self, cached_at: float) -> bool:
        return (time.time() - cached_at) > IMAGE_CACHE_TTL

    def get(self, url: str) -> Optional[bytes]:
        """Return cached image bytes for *url*, or None if not cached/expired."""
        key = self._sha256(url)
        entry = self._cache.get(key)
        if entry is None:
            return None
        data, content_type, cached_at = entry
        if self._is_expired(cached_at):
            del self._cache[key]
            logger.debug(f"[ImageCache] Expired entry for {url}")
            return None
        logger.debug(f"[ImageCache] Hit for {url}")
        return data

    def set(self, url: str, data: bytes, content_type: str) -> None:
        """Store image bytes in the cache."""
        key = self._sha256(url)
        now = time.time()
        self._cache[key] = (data, content_type, now)
        logger.debug(f"[ImageCache] Stored {len(data)} bytes for {url}")
        # Lazy prune if the cache is getting large
        if len(self._cache) > IMAGE_CACHE_MAX_SIZE:
            self._prune()

    def invalidate(self, url: str) -> None:
        """Explicitly remove a cached entry (rarely needed)."""
        key = self._sha256(url)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    def _prune(self) -> None:
        """Remove all expired entries. Called lazily when cache exceeds max size."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if (now - v[2]) > IMAGE_CACHE_TTL]
        for k in expired:
            del self._cache[k]
        logger.debug(
            f"[ImageCache] Pruned {len(expired)} expired entries, "
            f"{len(self._cache)} remaining"
        )


# Module-level singleton — imported and used by all download sites.
image_cache = ImageCache()