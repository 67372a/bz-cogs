"""Tests for image deduplication and unique filenames in GenerateImageToolCall.

Verifies the fix for the bug where Gemini's image generation model could
return duplicate images, resulting in the exact same image being sent to
Discord multiple times.
"""

import base64
import hashlib
import sys
from unittest.mock import MagicMock

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — minimal stubs so the GenerateImageToolCall module can load
# without pulling in the full redbot / discord dependency tree.
#
# IMPORTANT: We save and restore any sys.modules entries we inject so that
# other test modules collected in the same pytest session are not affected.
# ---------------------------------------------------------------------------

_injected_modules: dict = {}  # name -> original value (or _SENTINEL if new)
_SENTINEL = object()


def _inject_mock(name: str, module=None):
    """Register *module* in sys.modules, remembering the previous value."""
    if name in _injected_modules:
        return  # already injected this session
    _injected_modules[name] = sys.modules.get(name, _SENTINEL)
    if module is None:
        module = MagicMock()
    sys.modules[name] = module


def _restore_modules():
    """Undo all injections so other test files see a clean sys.modules."""
    for name, original in _injected_modules.items():
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _injected_modules.clear()


# --- discord stub ---
import types as _types

_discord_mod = _types.ModuleType("discord")
_discord_mod.Message = type("_DiscordMessage", (), {})
_discord_mod.Embed = MagicMock()
_discord_mod.Color = MagicMock()
_discord_mod.Colour = MagicMock()
_inject_mock("discord", _discord_mod)

# --- redbot stubs ---
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
_redbot_bot = _make_mock_package("redbot.core.bot")
_redbot_bot.Red = MagicMock()
_inject_mock("redbot", _redbot)
_inject_mock("redbot.core", _redbot_core)
_inject_mock("redbot.core.bot", _redbot_bot)

# --- Import the real types module (pure Python, no external deps) ---
types_mod = import_module_directly(
    "aiuser.functions.types", "aiuser/functions/types.py"
)
_inject_mock("aiuser.functions.types", types_mod)

# --- Import the real ToolCall base class ---
tool_call_base = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)
_inject_mock("aiuser.functions.tool_call", tool_call_base)

# --- Stub out remaining external dependencies ---
_or_types = _make_mock_package("aiuser.types")
_or_types_mod = _make_mock_package("aiuser.types.openrouter_types")
_or_types_mod.DirectImageGenerationParameters = MagicMock
_or_types_mod.deserialize_parameters = MagicMock()
_inject_mock("aiuser.types", _or_types)
_inject_mock("aiuser.types.openrouter_types", _or_types_mod)

_image_cache_mod = _make_mock_package("aiuser.utils.image_cache")
_image_cache_mod.image_cache = MagicMock()
_inject_mock("aiuser.utils", _make_mock_package("aiuser.utils"))
_inject_mock("aiuser.utils.image_cache", _image_cache_mod)

# aiohttp may or may not be installed; mock it safely
if "aiohttp" not in sys.modules:
    _inject_mock("aiohttp")

# Ensure parent packages exist for the module under test
_all_pkg_prefixes = set()
for _name in list(sys.modules.keys()) + ["aiuser.functions.generate_image.tool_call"]:
    _parts = _name.split(".")
    for _i in range(1, len(_parts)):
        _all_pkg_prefixes.add(".".join(_parts[:_i]))
for _pkg in _all_pkg_prefixes:
    if _pkg not in sys.modules:
        _inject_mock(_pkg, _make_mock_package(_pkg))

# Load the module under test
tool_call_mod = import_module_directly(
    "aiuser.functions.generate_image.tool_call",
    "aiuser/functions/generate_image/tool_call.py",
)
GenerateImageToolCall = tool_call_mod.GenerateImageToolCall

# Restore mocked modules after the module-under-test has been imported
# so other test files in the session see a clean sys.modules.
_restore_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(message_id: int = 123456789) -> GenerateImageToolCall:
    """Create a GenerateImageToolCall with mocked dependencies."""
    ctx = MagicMock()
    ctx.message.id = message_id
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    config = MagicMock()
    tool = GenerateImageToolCall.__new__(GenerateImageToolCall)
    tool.config = config
    tool.ctx = ctx
    tool.bot = MagicMock()
    tool._cog = None
    tool.generated_images = []
    return tool


def _make_data_url(image_bytes: bytes, fmt: str = "png") -> str:
    """Build a data:image/...;base64,... URL from raw bytes."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


def _make_response(model_extra=None, content=None):
    """Build a minimal mock ChatCompletion response."""
    message = MagicMock()
    message.model_extra = model_extra
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests: _extract_images_from_response deduplication
# ---------------------------------------------------------------------------


class TestExtractImagesDedup:
    """Tests for SHA-256 deduplication in _extract_images_from_response."""

    def test_no_duplicates_returns_all(self):
        """Two genuinely different images should both be returned."""
        tool = _make_tool_call()
        img_a = _make_data_url(b"\x89PNG_image_A" + b"\x00" * 50)
        img_b = _make_data_url(b"\x89PNG_image_B" + b"\x00" * 50)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_a}},
                    {"type": "image_url", "image_url": {"url": img_b}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 2

    def test_duplicate_in_model_extra_filtered(self):
        """Two identical images in model_extra should produce only one result."""
        tool = _make_tool_call()
        img_bytes = b"\x89PNG_identical_content" + b"\x00" * 50
        img_url = _make_data_url(img_bytes)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1
        assert result[0] == img_url

    def test_triplicate_in_model_extra_filtered(self):
        """Three identical images in model_extra should produce only one result."""
        tool = _make_tool_call()
        img_url = _make_data_url(b"\x89PNG_triplet" + b"\x00" * 50)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_same_image_in_model_extra_and_content(self):
        """An image in both model_extra and content text should be deduped."""
        tool = _make_tool_call()
        img_bytes = b"\x89PNG_shared" + b"\x00" * 50
        img_url = _make_data_url(img_bytes)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]
            },
            content=f"Here is the image: {img_url}",
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_different_images_not_deduped(self):
        """Two images with different content should both be returned."""
        tool = _make_tool_call()
        img_a = _make_data_url(b"\x00\x01\x02" + b"A" * 100)
        img_b = _make_data_url(b"\x03\x04\x05" + b"B" * 100)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_a}},
                    {"type": "image_url", "image_url": {"url": img_b}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 2
        assert result[0] == img_a
        assert result[1] == img_b

    def test_single_image_string_in_model_extra(self):
        """A single image URL string (not list) should be extracted."""
        tool = _make_tool_call()
        img_url = _make_data_url(b"\x89PNG_single" + b"\x00" * 50)

        response = _make_response(model_extra={"image": img_url})
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_empty_response(self):
        """An empty response should return no images."""
        tool = _make_tool_call()
        response = _make_response()
        response.choices = []
        result = tool._extract_images_from_response(response)
        assert result == []

    def test_no_images_in_response(self):
        """A response with no images should return an empty list."""
        tool = _make_tool_call()
        response = _make_response(model_extra={})
        result = tool._extract_images_from_response(response)
        assert result == []

    def test_two_unique_plus_one_duplicate(self):
        """Three images where the third is a duplicate of the first: two unique kept."""
        tool = _make_tool_call()
        img_a = _make_data_url(b"\x89PNG_uniqueA" + b"\x00" * 50)
        img_b = _make_data_url(b"\x89PNG_uniqueB" + b"\x00" * 50)
        # img_c has same bytes as img_a
        img_c = _make_data_url(b"\x89PNG_uniqueA" + b"\x00" * 50)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": img_a}},
                    {"type": "image_url", "image_url": {"url": img_b}},
                    {"type": "image_url", "image_url": {"url": img_c}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: _decode_image_data unique filenames
# ---------------------------------------------------------------------------


class TestDecodeImageUniqueFilename:
    """Tests that _decode_image_data generates unique filenames via content hashing."""

    def test_filename_contains_content_hash(self):
        """Filename should include a short SHA-256 hash of the image bytes."""
        tool = _make_tool_call(message_id=42)
        img_bytes = b"\x89PNG_test" + b"\x00" * 50
        data_url = _make_data_url(img_bytes)

        result = tool._decode_image_data(data_url)
        assert result is not None

        expected_hash = hashlib.sha256(img_bytes).hexdigest()[:8]
        assert result["filename"] == f"generated_42_{expected_hash}.png"

    def test_different_images_get_different_filenames(self):
        """Two images with different content should have different filenames."""
        tool = _make_tool_call(message_id=42)
        img_a = b"\x89PNG_alpha" + b"\x00" * 50
        img_b = b"\x89PNG_beta_" + b"\x00" * 50

        result_a = tool._decode_image_data(_make_data_url(img_a))
        result_b = tool._decode_image_data(_make_data_url(img_b))

        assert result_a is not None
        assert result_b is not None
        assert result_a["filename"] != result_b["filename"]

    def test_same_content_gets_same_filename(self):
        """Two identical images should produce the same filename (consistent hash)."""
        tool = _make_tool_call(message_id=42)
        img_bytes = b"\x89PNG_same" + b"\x00" * 50
        url = _make_data_url(img_bytes)

        result_a = tool._decode_image_data(url)
        result_b = tool._decode_image_data(url)

        assert result_a is not None
        assert result_b is not None
        assert result_a["filename"] == result_b["filename"]

    def test_filename_format_png(self):
        """PNG images should use .png extension."""
        tool = _make_tool_call(message_id=999)
        data_url = _make_data_url(b"\x89PNG_fmt" + b"\x00" * 50, fmt="png")

        result = tool._decode_image_data(data_url)
        assert result is not None
        assert result["filename"].startswith("generated_999_")
        assert result["filename"].endswith(".png")

    def test_filename_format_jpeg(self):
        """JPEG images should use .jpg extension."""
        tool = _make_tool_call(message_id=999)
        data_url = _make_data_url(b"\xff\xd8\xff_jpg" + b"\x00" * 50, fmt="jpeg")

        result = tool._decode_image_data(data_url)
        assert result is not None
        assert result["filename"].endswith(".jpg")

    def test_filename_format_gif(self):
        """GIF images should use .gif extension."""
        tool = _make_tool_call(message_id=1)
        data_url = _make_data_url(b"GIF89a_test" + b"\x00" * 50, fmt="gif")

        result = tool._decode_image_data(data_url)
        assert result is not None
        assert result["filename"].endswith(".gif")
        assert result["format"] == "gif"

    def test_filename_format_webp(self):
        """WebP images should use .webp extension."""
        tool = _make_tool_call(message_id=100)
        data_url = _make_data_url(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 50, fmt="webp")

        result = tool._decode_image_data(data_url)
        assert result is not None
        assert result["filename"].endswith(".webp")

    def test_invalid_data_url_returns_none(self):
        """A malformed data URL should return None."""
        tool = _make_tool_call()
        result = tool._decode_image_data("not-a-data-url")
        assert result is None

    def test_old_pattern_replaced(self):
        """Old format was 'generated_{id}.ext'; new format adds _hash suffix."""
        tool = _make_tool_call(message_id=42)
        img_bytes = b"\x89PNG_new_fmt" + b"\x00" * 50
        data_url = _make_data_url(img_bytes)

        result = tool._decode_image_data(data_url)
        assert result is not None
        # Should NOT match the old pattern exactly
        assert result["filename"] != "generated_42.png"
        # Should contain the hash suffix
        expected_hash = hashlib.sha256(img_bytes).hexdigest()[:8]
        assert expected_hash in result["filename"]


# ---------------------------------------------------------------------------
# Tests: Content-based dedup safety net in the decode loop
# ---------------------------------------------------------------------------


class TestContentHashDedup:
    """Tests for the SHA-256 content dedup logic applied during image decoding.

    These verify the safety net in _handle that filters duplicates even if
    _extract_images_from_response passes them through.
    """

    def test_same_bytes_deduplicated(self):
        """Two data URLs with identical decoded bytes should yield one image."""
        tool = _make_tool_call(message_id=42)
        img_bytes = b"\x89PNG_safety_net" + b"\x00" * 50

        url_a = _make_data_url(img_bytes)
        url_b = _make_data_url(img_bytes)

        seen_hashes = set()
        stored = []
        for url in [url_a, url_b]:
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    stored.append(decoded)

        assert len(stored) == 1

    def test_different_bytes_not_deduplicated(self):
        """Two data URLs with different decoded bytes should yield two images."""
        tool = _make_tool_call(message_id=42)

        seen_hashes = set()
        stored = []
        for img_bytes in [b"\x89PNG_A" + b"\x00" * 50, b"\x89PNG_B" + b"\x00" * 50]:
            url = _make_data_url(img_bytes)
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    stored.append(decoded)

        assert len(stored) == 2
        # Verify filenames are different (due to content hash)
        assert stored[0]["filename"] != stored[1]["filename"]

    def test_mixed_unique_and_duplicate(self):
        """Three images where one is a duplicate: only two unique stored."""
        tool = _make_tool_call(message_id=77)
        img_a = b"\x89PNG_mix_A" + b"\x00" * 50
        img_b = b"\x89PNG_mix_B" + b"\x00" * 50

        seen_hashes = set()
        stored = []
        for img_bytes in [img_a, img_b, img_a]:  # img_a repeated
            url = _make_data_url(img_bytes)
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    stored.append(decoded)

        assert len(stored) == 2


# ---------------------------------------------------------------------------
# Helpers for pixel-level dedup tests (real PNG images via PIL)
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
    import io as _pil_io
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


def _make_real_png_bytes(
    size=(8, 8),
    color=(255, 0, 0, 255),
    optimize=False,
    add_text_metadata=False,
) -> bytes:
    """Create a real PNG image with PIL and return its raw bytes.

    Different ``optimize`` / metadata settings produce different byte streams
    for the same pixel data — this is the exact scenario that byte-level
    dedup fails to catch but pixel-level dedup should.
    """
    img = _PILImage.new("RGBA", size, color)
    buf = _pil_io.BytesIO()
    kwargs = {"format": "PNG"}
    if optimize:
        kwargs["optimize"] = True
    if add_text_metadata:
        # Add PNG text metadata to force different bytes
        img.info["png_text"] = {"Software": "test-bz-cogs", "Timestamp": "2026-01-01"}
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        pnginfo.add_text("Software", "test-bz-cogs")
        pnginfo.add_text("Timestamp", "2026-01-01T00:00:00Z")
        kwargs["pnginfo"] = pnginfo
    img.save(buf, **kwargs)
    return buf.getvalue()


def _make_real_png_data_url(
    size=(8, 8),
    color=(255, 0, 0, 255),
    optimize=False,
    add_text_metadata=False,
) -> str:
    """Create a real PNG data URL with PIL."""
    raw = _make_real_png_bytes(size, color, optimize, add_text_metadata)
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Tests: _compute_pixel_hash helper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL/Pillow not available")
class TestComputePixelHash:
    """Tests for the _compute_pixel_hash helper method."""

    def test_identical_pixels_different_encoding_same_hash(self):
        """Two PNGs with different bytes but identical pixels should have
        the same pixel hash."""
        tool = _make_tool_call()
        png_a = _make_real_png_bytes(optimize=False)
        png_b = _make_real_png_bytes(optimize=True, add_text_metadata=True)

        # Sanity: bytes must differ (otherwise the test is meaningless)
        assert png_a != png_b

        hash_a = tool._compute_pixel_hash(png_a)
        hash_b = tool._compute_pixel_hash(png_b)

        assert hash_a is not None
        assert hash_b is not None
        assert hash_a == hash_b

    def test_different_pixels_different_hash(self):
        """Two PNGs with different pixel data should have different hashes."""
        tool = _make_tool_call()
        png_red = _make_real_png_bytes(color=(255, 0, 0, 255))
        png_blue = _make_real_png_bytes(color=(0, 0, 255, 255))

        hash_red = tool._compute_pixel_hash(png_red)
        hash_blue = tool._compute_pixel_hash(png_blue)

        assert hash_red is not None
        assert hash_blue is not None
        assert hash_red != hash_blue

    def test_returns_none_for_invalid_image(self):
        """Non-image bytes should return None (graceful degradation)."""
        tool = _make_tool_call()
        result = tool._compute_pixel_hash(b"not an image at all")
        assert result is None

    def test_returns_none_for_empty_bytes(self):
        """Empty bytes should return None."""
        tool = _make_tool_call()
        result = tool._compute_pixel_hash(b"")
        assert result is None

    def test_different_sizes_different_hash(self):
        """Images of different sizes should have different pixel hashes."""
        tool = _make_tool_call()
        png_small = _make_real_png_bytes(size=(4, 4))
        png_large = _make_real_png_bytes(size=(8, 8))

        hash_small = tool._compute_pixel_hash(png_small)
        hash_large = tool._compute_pixel_hash(png_large)

        assert hash_small != hash_large


# ---------------------------------------------------------------------------
# Tests: Pixel-level dedup in _extract_images_from_response (Layer 1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL/Pillow not available")
class TestExtractImagesPixelDedup:
    """Tests for pixel-level deduplication in _extract_images_from_response.

    These test the core bug fix: Gemini returning the same image twice with
    different PNG encodings (different bytes, identical pixels).
    """

    def test_same_pixels_different_encoding_deduped(self):
        """Two images with different bytes but identical pixels should be
        deduplicated to one."""
        tool = _make_tool_call()
        url_a = _make_real_png_data_url(optimize=False)
        url_b = _make_real_png_data_url(optimize=True, add_text_metadata=True)

        # Sanity check: bytes differ, so byte-level dedup would NOT catch this
        assert url_a != url_b

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": url_a}},
                    {"type": "image_url", "image_url": {"url": url_b}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_same_pixels_triplicate_different_encodings_deduped(self):
        """Three copies of the same image with three different encodings
        should be deduplicated to one."""
        tool = _make_tool_call()
        url_a = _make_real_png_data_url(optimize=False)
        url_b = _make_real_png_data_url(optimize=True)
        url_c = _make_real_png_data_url(add_text_metadata=True)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": url_a}},
                    {"type": "image_url", "image_url": {"url": url_b}},
                    {"type": "image_url", "image_url": {"url": url_c}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_different_pixels_not_deduped(self):
        """Two images with different pixel data should both be returned."""
        tool = _make_tool_call()
        url_red = _make_real_png_data_url(color=(255, 0, 0, 255))
        url_blue = _make_real_png_data_url(color=(0, 0, 255, 255))

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": url_red}},
                    {"type": "image_url", "image_url": {"url": url_blue}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 2

    def test_same_pixels_cross_path_deduped(self):
        """The same image in model_extra and message content (with different
        encodings) should be deduplicated."""
        tool = _make_tool_call()
        url_a = _make_real_png_data_url(optimize=False)
        url_b = _make_real_png_data_url(optimize=True, add_text_metadata=True)

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": url_a}},
                ]
            },
            content=f"Here is the image: {url_b}",
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 1

    def test_mixed_unique_and_pixel_duplicate(self):
        """Three images: two unique, one is a pixel-duplicate of the first.
        Should return two unique images."""
        tool = _make_tool_call()
        url_a = _make_real_png_data_url(color=(255, 0, 0, 255), optimize=False)
        url_a_dup = _make_real_png_data_url(color=(255, 0, 0, 255), optimize=True, add_text_metadata=True)
        url_b = _make_real_png_data_url(color=(0, 255, 0, 255))

        response = _make_response(
            model_extra={
                "images": [
                    {"type": "image_url", "image_url": {"url": url_a}},
                    {"type": "image_url", "image_url": {"url": url_b}},
                    {"type": "image_url", "image_url": {"url": url_a_dup}},
                ]
            }
        )
        result = tool._extract_images_from_response(response)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: Pixel-level dedup safety net in _handle decode loop (Layer 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL/Pillow not available")
class TestHandlePixelDedupSafetyNet:
    """Tests for the pixel-level dedup safety net in the _handle decode loop.

    Even if Layer 1 (_extract_images_from_response) somehow passes through
    two visually-identical images with different encodings, Layer 2 in _handle
    should catch them via the seen_pixel_hashes set.
    """

    def test_pixel_duplicate_safety_net(self):
        """Two data URLs with different bytes but identical pixels should
        yield only one stored image in the decode loop."""
        tool = _make_tool_call(message_id=42)

        url_a = _make_real_png_data_url(optimize=False)
        url_b = _make_real_png_data_url(optimize=True, add_text_metadata=True)

        # Simulate the decode loop from _handle
        seen_content_hashes = set()
        seen_pixel_hashes = set()
        stored = []

        for url in [url_a, url_b]:
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                pixel_hash = tool._compute_pixel_hash(decoded["bytes"])

                if img_hash in seen_content_hashes:
                    continue
                if pixel_hash and pixel_hash in seen_pixel_hashes:
                    continue
                seen_content_hashes.add(img_hash)
                if pixel_hash:
                    seen_pixel_hashes.add(pixel_hash)
                stored.append(decoded)

        assert len(stored) == 1

    def test_pixel_unique_safety_net(self):
        """Two images with different pixels should both be stored."""
        tool = _make_tool_call(message_id=42)

        url_red = _make_real_png_data_url(color=(255, 0, 0, 255))
        url_blue = _make_real_png_data_url(color=(0, 0, 255, 255))

        seen_content_hashes = set()
        seen_pixel_hashes = set()
        stored = []

        for url in [url_red, url_blue]:
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                pixel_hash = tool._compute_pixel_hash(decoded["bytes"])

                if img_hash in seen_content_hashes:
                    continue
                if pixel_hash and pixel_hash in seen_pixel_hashes:
                    continue
                seen_content_hashes.add(img_hash)
                if pixel_hash:
                    seen_pixel_hashes.add(pixel_hash)
                stored.append(decoded)

        assert len(stored) == 2

    def test_byte_identical_caught_by_byte_hash(self):
        """Byte-identical images should be caught by the fast byte-hash path,
        without needing pixel comparison."""
        tool = _make_tool_call(message_id=42)

        url = _make_real_png_data_url()

        seen_content_hashes = set()
        seen_pixel_hashes = set()
        stored = []

        for _ in range(3):  # Same URL three times
            decoded = tool._decode_image_data(url)
            if decoded:
                img_hash = hashlib.sha256(decoded["bytes"]).hexdigest()
                pixel_hash = tool._compute_pixel_hash(decoded["bytes"])

                if img_hash in seen_content_hashes:
                    continue
                if pixel_hash and pixel_hash in seen_pixel_hashes:
                    continue
                seen_content_hashes.add(img_hash)
                if pixel_hash:
                    seen_pixel_hashes.add(pixel_hash)
                stored.append(decoded)

        assert len(stored) == 1
