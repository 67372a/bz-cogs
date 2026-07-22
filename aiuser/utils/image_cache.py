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


# ── Processed Image Cache ───────────────────────────────────────────────────


class ProcessedImageCache:
    """Double-keyed TTL cache for **processed** (resized + WebP-compressed)
    image bytes, keyed by properties of the **original** (pre-processed) image.

    Two lookup keys are maintained per entry so that the same visual content
    arriving via different file encodings (e.g. the same photo uploaded as
    both PNG and JPEG) is only processed once:

    * **Key 1 — byte hash:**  ``SHA-256(original_bytes)``
    * **Key 2 — pixel hash:** ``SHA-256(canonical_RGBA_pixels(original_bytes))``

    Both keys are *namespaced* with the effective ``max_pixels`` processing
    parameter so that a guild with ``max_image_pixels = 16777216`` does not
    serve a 1-mpx-cached image when ``max_image_pixels`` was later changed
    to ``1048576`` (or vice-versa).

    The internal store is ``byte_key → (processed_bytes, pixel_key, cached_at)``.
    A secondary *alias index* ``pixel_key → byte_key`` lets pixel-hash
    lookups resolve to the primary entry.  On a pixel-hash hit, the entry's
    byte-key alias is backfilled so the same content can later be found via
    byte-hash alone.

    Entries expire after ``PROCESSED_CACHE_TTL`` seconds (default 30 min).
    When the cache exceeds ``PROCESSED_CACHE_MAX_SIZE`` entries, expired
    entries are pruned.
    """

    PROCESSED_CACHE_TTL = 1800     # seconds
    PROCESSED_CACHE_MAX_SIZE = 200  # entries

    def __init__(self):
        # byte_key → (processed_bytes, pixel_key_or_None, cached_at)
        self._primary: dict[str, Tuple[bytes, Optional[str], float]] = {}
        # pixel_key → byte_key  (secondary alias index)
        self._pixel_index: dict[str, str] = {}

    # ── key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _byte_key(original_bytes: bytes, max_pixels: int) -> str:
        """SHA-256 of the original bytes * ‖ * max_pixels (namespaced)."""
        h = hashlib.sha256()
        h.update(original_bytes)
        h.update(f"|mp{max_pixels}".encode("ascii"))
        return h.hexdigest()

    @staticmethod
    def _pixel_key(pixel_hash: str, max_pixels: int) -> str:
        """SHA-256 of the pixel hash * ‖ * max_pixels (namespaced)."""
        h = hashlib.sha256()
        h.update(pixel_hash.encode("ascii"))
        h.update(f"|mp{max_pixels}".encode("ascii"))
        return h.hexdigest()

    def _is_expired(self, cached_at: float) -> bool:
        return (time.time() - cached_at) > self.PROCESSED_CACHE_TTL

    # ── public API ───────────────────────────────────────────────────────

    def get(
        self,
        original_bytes: bytes,
        pixel_hash: Optional[str],
        max_pixels: int,
    ) -> Optional[bytes]:
        """Look up processed bytes by byte hash (fast path) or pixel hash.

        Parameters
        ----------
        original_bytes:
            Raw image bytes (the original, before any processing).
        pixel_hash:
            Pre-computed SHA-256 of the canonical RGBA pixels, or ``None``
            if PIL could not decode the image (in which case only the
            byte-hash key is checked).
        max_pixels:
            The effective ``max_pixels`` used when the image was (or would
            be) processed.  Forms part of both cache keys.

        Returns
        -------
        bytes or None
            The cached processed WebP bytes, or ``None`` on miss / expiry.
        """
        # Fast path: byte-hash lookup
        bkey = self._byte_key(original_bytes, max_pixels)
        entry = self._primary.get(bkey)
        if entry is not None:
            processed, _pk, cached_at = entry
            if self._is_expired(cached_at):
                self._remove_entry(bkey, _pk)
            else:
                logger.debug(f"[ProcessedCache] Byte-hash hit ({bkey[:12]}…)")
                return processed

        # Slow path: pixel-hash lookup
        if pixel_hash is not None:
            pkey = self._pixel_key(pixel_hash, max_pixels)
            alias_byte_key = self._pixel_index.get(pkey)
            if alias_byte_key is not None:
                entry = self._primary.get(alias_byte_key)
                if entry is not None:
                    processed, _pk, cached_at = entry
                    if self._is_expired(cached_at):
                        self._remove_entry(alias_byte_key, _pk)
                    else:
                        # Backfill: register under the current byte-hash too
                        self._primary[bkey] = processed, _pk, cached_at
                        logger.debug(
                            f"[ProcessedCache] Pixel-hash hit → backfilled "
                            f"byte-key ({bkey[:12]}…)"
                        )
                        return processed

        return None

    def set(
        self,
        original_bytes: bytes,
        pixel_hash: Optional[str],
        max_pixels: int,
        processed_bytes: bytes,
    ) -> None:
        """Store processed bytes under both byte-hash and pixel-hash keys."""
        bkey = self._byte_key(original_bytes, max_pixels)
        pkey = self._pixel_key(pixel_hash, max_pixels) if pixel_hash else None

        now = time.time()
        self._primary[bkey] = (processed_bytes, pkey, now)
        if pkey is not None:
            self._pixel_index[pkey] = bkey

        logger.debug(
            f"[ProcessedCache] Stored {len(processed_bytes)} processed bytes "
            f"byte={bkey[:12]}… pixel={pkey[:12] if pkey else 'N/A'}…"
        )

        # Lazy prune
        if len(self._primary) > self.PROCESSED_CACHE_MAX_SIZE:
            self._prune()

    # ── housekeeping ─────────────────────────────────────────────────────

    def _remove_entry(self, byte_key: str, pixel_key: Optional[str]) -> None:
        self._primary.pop(byte_key, None)
        if pixel_key is not None:
            self._pixel_index.pop(pixel_key, None)

    def _prune(self) -> None:
        now = time.time()
        expired_bkeys = [
            k for k, v in self._primary.items()
            if (now - v[2]) > self.PROCESSED_CACHE_TTL
        ]
        for bkey in expired_bkeys:
            entry = self._primary.pop(bkey, None)
            if entry is not None and entry[1] is not None:
                self._pixel_index.pop(entry[1], None)
        logger.debug(
            f"[ProcessedCache] Pruned {len(expired_bkeys)} expired entries, "
            f"{len(self._primary)} remaining"
        )

    def clear(self) -> None:
        self._primary.clear()
        self._pixel_index.clear()


# ── Module-level singletons ──────────────────────────────────────────────────

image_cache = ImageCache()
processed_image_cache = ProcessedImageCache()
