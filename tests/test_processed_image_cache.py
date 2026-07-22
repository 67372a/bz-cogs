"""Tests for the double-keyed ProcessedImageCache and unified image processing pipeline.

Covers:
  - Byte-hash lookup (fast path)
  - Pixel-hash lookup with alias backfill (different encodings of the same image)
  - Cache miss on genuinely different images
  - TTL expiry
  - Max-size pruning
  - Pipeline determinism (same input → same output)
  - WebP data URL round-trip
  - Namespace isolation (different max_pixels → different cache entries)
"""

import io
import time
from unittest.mock import patch

import numpy as np
import cv2
import pytest
from PIL import Image

# Use import_module_directly to bypass aiuser/__init__.py which imports redbot.
# This is necessary because the dedup test's _restore_modules() removes the
# redbot mock from sys.modules, which would break normal import paths.
from tests.mock_importer import import_module_directly

_image_cache_mod = import_module_directly(
    "aiuser.utils.image_cache", "aiuser/utils/image_cache.py"
)
ProcessedImageCache = _image_cache_mod.ProcessedImageCache
ImageCache = _image_cache_mod.ImageCache

_image_processing_mod = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
compute_byte_hash = _image_processing_mod.compute_byte_hash
compute_pixel_hash = _image_processing_mod.compute_pixel_hash
process_image_for_llm = _image_processing_mod.process_image_for_llm
build_webp_data_url = _image_processing_mod.build_webp_data_url


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 64, height: int = 64, color=(255, 0, 0)) -> bytes:
    """Create a solid-color PNG image in memory and return its bytes."""
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 64, color=(255, 0, 0)) -> bytes:
    """Create a JPEG image.  Note: JPEG is lossy, so decoded pixels will NOT
    match the lossless PNG version pixel-for-pixel even for a solid color.
    Use for testing miss/scenario, not for cross-encoding hit tests."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_lossless_webp_bytes(width: int = 64, height: int = 64, color=(255, 0, 0)) -> bytes:
    """Create the *same* visual content as lossless WebP (different encoding).

    Lossless WebP preserves exact pixel values, so the RGBA canonical form
    decoded from this file should be byte-identical to the PNG version.
    This makes it the correct choice for cross-encoding pixel-hash tests.
    """
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="WebP", lossless=True)
    return buf.getvalue()


def _make_gradient_png(width: int = 128, height: int = 128) -> bytes:
    """Create a gradient PNG that is visually distinct from the solid images."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [int(255 * x / width), int(255 * y / height), 128]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: compute_byte_hash / compute_pixel_hash
# ---------------------------------------------------------------------------

class TestHashing:

    def test_byte_hash_deterministic(self):
        data = _make_png_bytes()
        assert compute_byte_hash(data) == compute_byte_hash(data)

    def test_byte_hash_differs_for_different_data(self):
        a = _make_png_bytes(color=(255, 0, 0))
        b = _make_png_bytes(color=(0, 255, 0))
        assert compute_byte_hash(a) != compute_byte_hash(b)

    def test_pixel_hash_same_visual_different_lossless_encoding(self):
        """PNG and lossless WebP of the same solid color have the same pixel hash.

        JPEG is lossy so its decoded pixels differ even for a solid color;
        we only test lossless format variants here.
        """
        png = _make_png_bytes(color=(100, 150, 200))
        webp = _make_lossless_webp_bytes(color=(100, 150, 200))

        ph_png = compute_pixel_hash(png)
        ph_webp = compute_pixel_hash(webp)

        assert ph_png is not None
        assert ph_webp is not None
        assert ph_png == ph_webp

    def test_pixel_hash_differs_for_lossy_encoding(self):
        """JPEG is lossy, so its pixel hash will differ from the lossless PNG
        version of the same solid-color image (JPEG artifacts)."""
        png = _make_png_bytes(color=(100, 150, 200))
        jpeg = _make_jpeg_bytes(color=(100, 150, 200))

        ph_png = compute_pixel_hash(png)
        ph_jpeg = compute_pixel_hash(jpeg)

        # JPEG artifacts mean the decoded pixels are NOT identical
        assert ph_png is not None
        assert ph_jpeg is not None
        assert ph_png != ph_jpeg

    def test_pixel_hash_differs_for_different_visuals(self):
        a = _make_png_bytes(color=(255, 0, 0))
        b = _make_png_bytes(color=(0, 255, 0))
        assert compute_pixel_hash(a) != compute_pixel_hash(b)

    def test_pixel_hash_returns_none_for_garbage(self):
        assert compute_pixel_hash(b"not an image at all") is None


# ---------------------------------------------------------------------------
# Tests: ProcessedImageCache (double-keyed)
# ---------------------------------------------------------------------------

class TestProcessedCache:

    def setup_method(self):
        self.cache = ProcessedImageCache()

    # ── byte-hash hit (fast path) ──────────────────────────────────────

    def test_byte_hash_hit(self):
        original = _make_png_bytes()
        processed = process_image_for_llm(original)
        self.cache.set(original, compute_pixel_hash(original), 16_777_216, processed)

        result = self.cache.get(original, compute_pixel_hash(original), 16_777_216)
        assert result is not None
        assert result == processed

    # ── pixel-hash hit (different encoding, same visual) ──────────────

    def test_pixel_hash_hit_different_lossless_encoding(self):
        png = _make_png_bytes(color=(100, 150, 200))
        webp = _make_lossless_webp_bytes(color=(100, 150, 200))

        # Store under PNG bytes
        processed = process_image_for_llm(png)
        self.cache.set(png, compute_pixel_hash(png), 16_777_216, processed)

        # Lookup with lossless WebP bytes — should hit via pixel hash
        result = self.cache.get(webp, compute_pixel_hash(webp), 16_777_216)
        assert result is not None
        assert result == processed

    def test_pixel_hash_backfills_byte_alias(self):
        """After a pixel-hash hit, a subsequent byte-hash lookup should also work."""
        png = _make_png_bytes(color=(100, 150, 200))
        webp = _make_lossless_webp_bytes(color=(100, 150, 200))

        processed = process_image_for_llm(png)
        self.cache.set(png, compute_pixel_hash(png), 16_777_216, processed)

        # First lookup with lossless WebP → pixel-hash hit + backfill
        result = self.cache.get(webp, compute_pixel_hash(webp), 16_777_216)
        assert result is not None

        # Now the WebP byte-hash key should also be stored
        webp_byte_key = ProcessedImageCache._byte_key(webp, 16_777_216)
        assert webp_byte_key in self.cache._primary

    # ── miss ───────────────────────────────────────────────────────────

    def test_miss_for_different_content(self):
        a = _make_png_bytes(color=(255, 0, 0))
        b = _make_png_bytes(color=(0, 255, 0))

        processed_a = process_image_for_llm(a)
        self.cache.set(a, compute_pixel_hash(a), 16_777_216, processed_a)

        result = self.cache.get(b, compute_pixel_hash(b), 16_777_216)
        assert result is None

    def test_miss_when_no_pixel_hash(self):
        """If pixel_hash is None (PIL couldn't decode), only byte-hash is checked."""
        a = _make_png_bytes()
        processed = process_image_for_llm(a)
        self.cache.set(a, compute_pixel_hash(a), 16_777_216, processed)

        # Different bytes with None pixel_hash → miss
        different = b"completely different data"
        result = self.cache.get(different, None, 16_777_216)
        assert result is None

    # ── namespace isolation (max_pixels) ────────────────────────────────

    def test_different_max_pixels_no_cross_hit(self):
        """Images processed at different max_pixels limits should not cross-match."""
        img = _make_png_bytes()
        processed_16m = process_image_for_llm(img, 16_777_216)
        processed_1m = process_image_for_llm(img, 1_048_576)

        self.cache.set(img, compute_pixel_hash(img), 16_777_216, processed_16m)

        # Lookup with different max_pixels should miss
        result = self.cache.get(img, compute_pixel_hash(img), 1_048_576)
        assert result is None

        # Now store at 1M and verify both are accessible
        self.cache.set(img, compute_pixel_hash(img), 1_048_576, processed_1m)

        assert self.cache.get(img, compute_pixel_hash(img), 16_777_216) == processed_16m
        assert self.cache.get(img, compute_pixel_hash(img), 1_048_576) == processed_1m

    # ── TTL expiry ─────────────────────────────────────────────────────

    def test_ttl_expiry(self):
        img = _make_png_bytes()
        processed = process_image_for_llm(img)
        self.cache.set(img, compute_pixel_hash(img), 16_777_216, processed)

        # Simulate time passing beyond TTL
        with patch("aiuser.utils.image_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 1801  # past TTL
            result = self.cache.get(img, compute_pixel_hash(img), 16_777_216)
        assert result is None

    # ── max-size pruning ───────────────────────────────────────────────

    def test_prune_expired_entries_on_overflow(self):
        """When cache exceeds max size, expired entries are pruned."""
        self.cache.PROCESSED_CACHE_MAX_SIZE = 3  # small for testing

        for i in range(3):
            img = _make_png_bytes(color=(i * 50, 0, 0))
            processed = process_image_for_llm(img)
            self.cache.set(img, compute_pixel_hash(img), 16_777_216, processed)

        assert len(self.cache._primary) == 3

        # Age out the first two entries
        now = time.time()
        keys = list(self.cache._primary.keys())
        self.cache._primary[keys[0]] = (
            self.cache._primary[keys[0]][0],
            self.cache._primary[keys[0]][1],
            now - 2000,  # expired
        )
        self.cache._primary[keys[1]] = (
            self.cache._primary[keys[1]][0],
            self.cache._primary[keys[1]][1],
            now - 2000,  # expired
        )

        # Trigger pruning by adding one more
        extra = _make_png_bytes(color=(200, 200, 200))
        processed_extra = process_image_for_llm(extra)
        self.cache.set(extra, compute_pixel_hash(extra), 16_777_216, processed_extra)

        # The two expired entries should have been pruned
        assert len(self.cache._primary) == 2  # 1 survived + 1 new

    # ── clear ──────────────────────────────────────────────────────────

    def test_clear(self):
        img = _make_png_bytes()
        processed = process_image_for_llm(img)
        self.cache.set(img, compute_pixel_hash(img), 16_777_216, processed)
        assert len(self.cache._primary) == 1

        self.cache.clear()
        assert len(self.cache._primary) == 0
        assert len(self.cache._pixel_index) == 0


# ---------------------------------------------------------------------------
# Tests: process_image_for_llm
# ---------------------------------------------------------------------------

class TestProcessImageForLlm:

    def test_small_image_unchanged_dimensions(self):
        """An image smaller than max_pixels is not scaled."""
        img_bytes = _make_png_bytes(32, 32)
        result = process_image_for_llm(img_bytes, max_pixels=1_000_000)
        assert result is not None

        # Decode the result and check dimensions
        pil_result = Image.open(io.BytesIO(result))
        assert pil_result.size == (32, 32)

    def test_large_image_scaled_down(self):
        """An image larger than max_pixels is scaled down."""
        img_bytes = _make_png_bytes(512, 512)
        # Set max_pixels to 64*64 = 4096
        result = process_image_for_llm(img_bytes, max_pixels=4096)
        assert result is not None

        pil_result = Image.open(io.BytesIO(result))
        w, h = pil_result.size
        assert w * h <= 4096 + 1  # allow rounding

    def test_output_is_webp(self):
        img_bytes = _make_png_bytes()
        result = process_image_for_llm(img_bytes)
        assert result is not None

        pil_result = Image.open(io.BytesIO(result))
        assert pil_result.format == "WEBP"

    def test_returns_none_for_garbage(self):
        result = process_image_for_llm(b"not an image")
        assert result is None

    def test_deterministic_output(self):
        """Same input always produces the same processed output."""
        img_bytes = _make_png_bytes()
        a = process_image_for_llm(img_bytes)
        b = process_image_for_llm(img_bytes)
        assert a == b

    def test_webp_quality_parameter(self):
        """Lower quality should produce smaller output."""
        img_bytes = _make_gradient_png(128, 128)
        high = process_image_for_llm(img_bytes, webp_quality=100)
        low = process_image_for_llm(img_bytes, webp_quality=10)
        assert high is not None and low is not None
        assert len(low) < len(high)


# ---------------------------------------------------------------------------
# Tests: build_webp_data_url
# ---------------------------------------------------------------------------

class TestBuildWebpDataUrl:

    def test_roundtrip(self):
        img_bytes = _make_png_bytes()
        processed = process_image_for_llm(img_bytes)
        data_url = build_webp_data_url(processed)

        assert data_url.startswith("data:image/webp;base64,")

        import base64
        b64_part = data_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == processed


# ---------------------------------------------------------------------------
# Tests: ImageCache (URL-level cache — unchanged but verify it still works)
# ---------------------------------------------------------------------------

class TestImageCache:

    def test_get_set_roundtrip(self):
        cache = ImageCache()
        data = b"fake image bytes"
        cache.set("https://example.com/img.png", data, "image/png")
        assert cache.get("https://example.com/img.png") == data

    def test_miss(self):
        cache = ImageCache()
        assert cache.get("https://example.com/nope.png") is None

    def test_different_urls_different_entries(self):
        cache = ImageCache()
        cache.set("https://example.com/a.png", b"aaa", "image/png")
        cache.set("https://example.com/b.png", b"bbb", "image/png")
        assert cache.get("https://example.com/a.png") == b"aaa"
        assert cache.get("https://example.com/b.png") == b"bbb"


# ---------------------------------------------------------------------------
# Tests: pipeline integration (content parts)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_pipeline_produces_content_part(self):
        """Verify that process_image_for_llm_content returns a valid content part."""
        process_image_for_llm_content = _image_processing_mod.process_image_for_llm_content
        img_bytes = _make_png_bytes()
        part = process_image_for_llm_content(img_bytes)
        assert part is not None
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/webp;base64,")

    def test_pipeline_dedup_via_lossless_cache(self):
        """Two different lossless encodings of the same visual produce the same cached result."""
        cache = ProcessedImageCache()
        max_pixels = 16_777_216

        png = _make_png_bytes(color=(120, 80, 40))
        webp = _make_lossless_webp_bytes(color=(120, 80, 40))

        # Process and cache the PNG
        processed_png = process_image_for_llm(png, max_pixels)
        cache.set(png, compute_pixel_hash(png), max_pixels, processed_png)

        # Process the lossless WebP — same visual should yield the same processed bytes
        processed_webp = process_image_for_llm(webp, max_pixels)
        # Same pixels → same scaling + encoding → identical output
        assert processed_png == processed_webp

        # And the cache hit via pixel hash should also return the same thing
        cached = cache.get(webp, compute_pixel_hash(webp), max_pixels)
        assert cached == processed_png

    def test_pipeline_does_not_dedup_lossy_vs_lossless(self):
        """JPEG (lossy) and PNG (lossless) of the same solid color should NOT
        cross-match because JPEG artifacts cause different pixel data."""
        cache = ProcessedImageCache()
        max_pixels = 16_777_216

        png = _make_png_bytes(color=(120, 80, 40))
        jpeg = _make_jpeg_bytes(color=(120, 80, 40))

        processed_png = process_image_for_llm(png, max_pixels)
        cache.set(png, compute_pixel_hash(png), max_pixels, processed_png)

        # JPEG should NOT hit the cache (lossy pixel differences)
        cached = cache.get(jpeg, compute_pixel_hash(jpeg), max_pixels)
        assert cached is None
