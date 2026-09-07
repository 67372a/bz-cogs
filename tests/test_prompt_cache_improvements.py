"""Tests for the prompt-cache read-hit improvements.

Covers the six recommendations from the cache-hit analysis:

  Rec 1 — Caption-text cache (CaptionCache) pinning captions per image,
          including negative caching (failure placeholder) for AI Horde,
          and LOCAL-mode caption reuse.
  Rec 2 — History watermark (HistoryWatermark): floor pinning, hysteresis
          token stop, gap-break advance, idle expiry, forget reset.
  Rec 3 — Per-channel rolling cache-hit ratio stats.
  Rec 4 — Unified count_tokens() (str + multimodal parts).
  Rec 5 — PDF annotations injected as a tail user message (never into a
          mid-prefix assistant message).
"""

import asyncio
import logging
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing real modules
# ---------------------------------------------------------------------------

discord_mock = types.ModuleType("discord")
discord_mock.__package__ = "discord"
discord_mock.__path__ = []


class _DiscordMessage:
    """Real class to stand in for discord.Message in isinstance checks."""


discord_mock.Message = _DiscordMessage
discord_mock.MessageType = MagicMock()
discord_mock.MessageType.new_member = 7
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
discord_mock.NotFound = type("NotFound", (Exception,), {})
discord_mock.Forbidden = type("Forbidden", (Exception,), {})
discord_mock.HTTPException = type("HTTPException", (Exception,), {})
# Force-assign (like the other test modules): under pytest the real discord
# package may already be imported, and the isinstance(resolved, discord.Message)
# guards in messages.py must see our lightweight stand-in class.
sys.modules["discord"] = discord_mock
sys.modules.setdefault("discord.ext", types.ModuleType("discord.ext"))
sys.modules.setdefault("discord.ext.commands", MagicMock())
sys.modules.setdefault("discord.app_commands", MagicMock())

_redbot_core = types.ModuleType("redbot.core")
_redbot_core.__package__ = "redbot.core"
_redbot_core.__path__ = []
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
_redbot_core.app_commands = MagicMock()
sys.modules.setdefault("redbot", types.ModuleType("redbot"))
sys.modules.setdefault("redbot.core", _redbot_core)
for _n in ["redbot.core.commands", "redbot.core.bot", "redbot.core.checks",
           "redbot.core.utils", "redbot.core.utils.chat_formatting",
           "redbot.core.utils.views", "redbot.core.utils.menus",
           "redbot.core.data_manager"]:
    sys.modules.setdefault(_n, MagicMock())

sys.modules.setdefault("tiktoken", MagicMock())
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("pytesseract", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
try:
    import PIL  # noqa: F401
except ImportError:
    sys.modules.setdefault("PIL", MagicMock())
    sys.modules.setdefault("PIL.Image", MagicMock())

from tests.mock_importer import _make_mock_package, import_module_directly

_sub_pkgs = [
    "aiuser",
    "aiuser.messages_list",
    "aiuser.messages_list.converter",
    "aiuser.messages_list.converter.embed",
    "aiuser.messages_list.converter.image",
    "aiuser.config",
    "aiuser.types",
    "aiuser.utils",
    "aiuser.functions",
    "aiuser.response",
    "aiuser.response.chat",
    "aiuser.common",
    "aiuser.settings",
]
for _pkg in _sub_pkgs:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

_leaf_mocks = [
    "aiuser.config.defaults",
    "aiuser.config.models",
    "aiuser.messages_list.opt_view",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.converter.embed.formatter",
    "aiuser.messages_list.converter.embed.youtube",
    "aiuser.messages_list.converter.image.AI_horde",
    "aiuser.types.abc",
    "aiuser.utils.utilities",
    "aiuser.common.utilities",
    "aiuser.response.chat.function_call_view",
    "aiuser.functions.tool_call",
    "aiuser.functions.types",
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.attach_files.tool_call",
    "aiuser.functions.mermaid.tool_call",
    "aiuser.functions.openrouter",
]
for _m in _leaf_mocks:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

def _ensure_real_module(qualified_name: str, filepath: str):
    """Load *filepath* as *qualified_name* unless a real module already exists.

    Other test modules in this suite load these same files with their own
    mock bootstraps (and pytest imports every test module at collection
    time).  Re-registering a real module would leave earlier modules bound
    to the old object while ``sys.modules`` points at the new one — breaking
    e.g. ``patch("aiuser.utils.image_cache.time")`` in tests that ran
    against the original binding.  Only replace mock placeholders.
    """
    existing = sys.modules.get(qualified_name)
    if existing is not None and str(getattr(existing, "__file__", "")).endswith(".py"):
        return existing
    return import_module_directly(qualified_name, filepath)


# Real modules (order matters: dependencies first)
_ensure_real_module("aiuser.messages_list.entry", "aiuser/messages_list/entry.py")
_ensure_real_module("aiuser.types.enums", "aiuser/types/enums.py")
_ensure_real_module("aiuser.utils.cache", "aiuser/utils/cache.py")
_ensure_real_module("aiuser.utils.image_cache", "aiuser/utils/image_cache.py")

MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry
Cache = sys.modules["aiuser.utils.cache"].Cache
image_cache_mod = sys.modules["aiuser.utils.image_cache"]
CaptionCache = image_cache_mod.CaptionCache

# caption module (real) — for caption-cache wiring tests
_ensure_real_module("aiuser.config.constants", "aiuser/config/constants.py")
_ensure_real_module("aiuser.utils.image_processing", "aiuser/utils/image_processing.py")
_ensure_real_module("aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py")
caption_mod = _ensure_real_module(
    "aiuser.messages_list.converter.image.caption",
    "aiuser/messages_list/converter/image/caption.py")
# Bind the enum class caption.py itself compares against — other test
# modules may have swapped sys.modules["aiuser.types.enums"] for a mock,
# producing a different enum class than the one caption_mod holds.
ScanImageMode = caption_mod.ScanImageMode

# local module (real) — for LOCAL caption reuse test
local_mod = _ensure_real_module(
    "aiuser.messages_list.converter.image.local",
    "aiuser/messages_list/converter/image/local.py")

# messages module (real) — for watermark + count_tokens tests.
# Must be forced past any MagicMock placeholder left by earlier test modules.
messages_mod = _ensure_real_module(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py")
sys.modules["aiuser.messages_list.messages"] = messages_mod
MessagesList = messages_mod.MessagesList
HistoryWatermark = messages_mod.HistoryWatermark
WATERMARK_IDLE_SECONDS = messages_mod.WATERMARK_IDLE_SECONDS

# llm_pipeline module (real) — for cache-hit stats + annotation tail tests.
# Other bootstraps may leave "aiuser.functions.openrouter" as a bare mock
# package without the names llm_pipeline imports; force a MagicMock so the
# from-imports resolve, then (re)load llm_pipeline with real entry/messages.
_cur_openrouter = sys.modules.get("aiuser.functions.openrouter")
if _cur_openrouter is None or not hasattr(_cur_openrouter, "OpenRouterWebSearch"):
    sys.modules["aiuser.functions.openrouter"] = MagicMock()
llm_pipeline_mod = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py")
sys.modules["aiuser.response.chat.llm_pipeline"] = llm_pipeline_mod
LLMPipeline = llm_pipeline_mod.LLMPipeline
update_cache_hit_stats = llm_pipeline_mod.update_cache_hit_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubEncoding:
    """Deterministic token encoder: one token per whitespace-split word."""

    def encode(self, text, disallowed_special=None):
        return str(text).split()


def _make_cog():
    """Fake cog with the watermark + stats stores as real bounded caches."""
    cog = MagicMock()
    cog.ignore_regex = {}
    cog.override_prompt_start_time = {}
    cog.forget_start_times = Cache(limit=100)
    cog.context_front = Cache(limit=100)
    cog.cache_hit_stats = Cache(limit=100)
    cog.pdf_annotations = Cache(limit=100)
    return cog


def _make_ctx(channel_id=123, guild_id=456):
    ctx = MagicMock()
    ctx.channel.id = channel_id
    ctx.guild.id = guild_id
    ctx.guild.name = "TestGuild"
    ctx.message = MagicMock()
    return ctx


def _make_ml(cog=None, token_limit=1000, stop_tokens=None):
    """MessagesList built via __new__ with fully controlled attributes."""
    ml = MessagesList.__new__(MessagesList)
    cog = cog or _make_cog()
    ml._cog = cog
    ml.bot = cog
    ml.config = MagicMock()
    ml.ctx = _make_ctx()
    ml.guild = ml.ctx.guild
    ml.init_message = MagicMock()
    ml.init_message.id = 999
    ml.init_message.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ml.init_message.reference = None
    ml.messages = []
    ml.messages_ids = set()
    ml.tokens = 0
    ml.token_limit = token_limit
    ml._history_stop_tokens = stop_tokens if stop_tokens is not None else int(token_limit * 0.7)
    ml._encoding = _StubEncoding()
    ml.model = "gpt-4"
    ml.start_time = None
    ml._trigger_msg_start = 1
    ml.prefill = None
    ml.ignore_regex = None
    ml.converter = MagicMock()
    ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "x")])
    ml.check_if_add = AsyncMock(return_value=True)
    return ml


def _fake_msg(msg_id, minutes_ago=0, base=None):
    """Lightweight stand-in for a discord.Message in history walks."""
    base = base or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    msg = SimpleNamespace(
        id=msg_id,
        created_at=base - timedelta(minutes=minutes_ago),
        content="hi",
        author=SimpleNamespace(id=111, bot=False),
        embeds=[],
        attachments=[],
        stickers=[],
        reference=None,
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=456),
    )
    return msg


# ===========================================================================
# Rec 1 — CaptionCache semantics
# ===========================================================================

class TestCaptionCacheSemantics:

    def test_set_get_roundtrip(self):
        cache = CaptionCache()
        cache.set("ph1", 100, "a cat")
        assert cache.get("ph1", 100) == "a cat"

    def test_key_namespacing(self):
        cache = CaptionCache()
        cache.set("ph1", 100, "small")
        cache.set("ph1", 200, "big")
        assert cache.get("ph1", 100) == "small"
        assert cache.get("ph1", 200) == "big"

    def test_missing_key_returns_none(self):
        cache = CaptionCache()
        assert cache.get("nope", 100) is None

    def test_ttl_expiry(self):
        cache = CaptionCache()
        cache.set("ph1", 100, "a cat")
        # Force-expire the entry
        key = cache._key("ph1", 100)
        caption, _ = cache._cache[key]
        cache._cache[key] = (caption, 0)  # epoch → expired
        assert cache.get("ph1", 100) is None

    def test_max_size_prune(self):
        cache = CaptionCache()
        cache.CAPTION_CACHE_MAX_SIZE = 5
        for i in range(5):
            cache.set(f"ph{i}", 100, f"caption {i}")
        # Backdate three entries so they are expired
        now = _time.time()
        for i in range(3):
            key = cache._key(f"ph{i}", 100)
            caption, _ = cache._cache[key]
            cache._cache[key] = (caption, now - cache.CAPTION_CACHE_TTL - 10)
        # Adding more entries crosses the size limit → prune drops expired
        for i in range(10, 13):
            cache.set(f"ph{i}", 100, f"caption {i}")
        assert len(cache._cache) == 5
        assert cache.get("ph0", 100) is None  # expired, pruned
        assert cache.get("ph3", 100) == "caption 3"  # fresh, kept

    def test_empty_caption_not_stored(self):
        cache = CaptionCache()
        cache.set("ph1", 100, "")
        assert cache.get("ph1", 100) is None

    def test_failure_placeholder_constant(self):
        assert CaptionCache.FAILURE_PLACEHOLDER == "[Image: caption unavailable]"


# ===========================================================================
# Rec 1 — AI_HORDE wiring: caption pinned across requests
# ===========================================================================

class TestAIHordeCaptionPinning:

    def _run_process(self, caption_side_effects, pixel_hash="pixelhash"):
        """Run _process_attachment in AI_HORDE mode with a patched pipeline."""
        caption_cache = CaptionCache()
        caption_fn = AsyncMock(side_effect=caption_side_effects)

        message = MagicMock()
        message.id = 42
        message.author.name = "user"
        message.author.display_name = "User"
        attachment = MagicMock()
        attachment.filename = "img.png"

        cog = MagicMock()
        cog.config = MagicMock()

        with patch.object(caption_mod, "_fetch_attachment_bytes", new=AsyncMock(return_value=b"raw")), \
             patch.object(caption_mod, "compute_pixel_hash", return_value=pixel_hash), \
             patch.object(caption_mod, "processed_image_cache") as pic, \
             patch.object(caption_mod, "_decode_webp_to_pil", return_value=MagicMock()), \
             patch.object(caption_mod, "caption_image_ai_horde", new=caption_fn), \
             patch.object(caption_mod, "caption_cache", new=caption_cache):
            pic.get.return_value = b"processed"
            results = [
                asyncio.run(caption_mod._process_attachment(
                    cog, message, attachment, ScanImageMode.AI_HORDE, 1000
                ))
                for _ in caption_side_effects
            ]
        return results, caption_fn, caption_cache

    def test_caption_pinned_across_requests(self):
        """Different Horde captions per call must not change the context text."""
        results, caption_fn, cache = self._run_process(["a cat", "a different dog"])
        assert caption_fn.await_count == 1, "caption API must be called exactly once"
        assert "a cat" in results[0]
        assert results[1] == results[0], "second request must reuse the pinned caption"
        assert cache.get("pixelhash", 1000) == "a cat"

    def test_failure_pinned_as_placeholder(self):
        """A failed caption is pinned as a stable placeholder (negative caching)."""
        results, caption_fn, cache = self._run_process([None, "late success"])
        assert caption_fn.await_count == 1
        assert CaptionCache.FAILURE_PLACEHOLDER in results[0]
        assert results[1] == results[0], "placeholder must remain stable on retry"
        assert cache.get("pixelhash", 1000) == CaptionCache.FAILURE_PLACEHOLDER


# ===========================================================================
# Rec 1 — LOCAL wiring: BLIP caption reused from cache
# ===========================================================================

class TestLocalCaptionReuse:

    def test_cached_caption_skips_model(self):
        local_mod.extract_text = AsyncMock(return_value="")
        caption_image = AsyncMock(return_value="blip caption")
        local_mod.caption_image = caption_image
        cache = CaptionCache()
        cache.set("ph", 100, "cached caption")
        cog = MagicMock()
        msg = MagicMock()
        msg.author.name = "user"
        msg.author.display_name = "User"

        with patch.object(local_mod, "caption_cache", new=cache):
            result = asyncio.run(local_mod.process_image_locally(
                cog, msg, MagicMock(), pixel_hash="ph", max_pixels=100))

        caption_image.assert_not_awaited()
        assert "cached caption" in result

    def test_uncached_caption_computed_and_pinned(self):
        local_mod.extract_text = AsyncMock(return_value="")
        caption_image = AsyncMock(return_value="blip caption")
        local_mod.caption_image = caption_image
        cache = CaptionCache()
        cog = MagicMock()
        msg = MagicMock()
        msg.author.name = "user"
        msg.author.display_name = "User"

        with patch.object(local_mod, "caption_cache", new=cache):
            first = asyncio.run(local_mod.process_image_locally(
                cog, msg, MagicMock(), pixel_hash="ph2", max_pixels=100))
            second = asyncio.run(local_mod.process_image_locally(
                cog, msg, MagicMock(), pixel_hash="ph2", max_pixels=100))

        assert caption_image.await_count == 1, "caption model must run once"
        assert "blip caption" in first
        assert second == first
        assert cache.get("ph2", 100) == "blip caption"

    def test_ocr_result_not_overridden_by_caption(self):
        """When OCR finds >10 words, no caption is used at all."""
        local_mod.extract_text = AsyncMock(return_value=" ".join(["word"] * 12))
        caption_image = AsyncMock(return_value="blip caption")
        local_mod.caption_image = caption_image
        cache = CaptionCache()
        cog = MagicMock()
        msg = MagicMock()
        msg.author.name = "user"
        msg.author.display_name = "User"

        with patch.object(local_mod, "caption_cache", new=cache):
            result = asyncio.run(local_mod.process_image_locally(
                cog, msg, MagicMock(), pixel_hash="ph3", max_pixels=100))

        caption_image.assert_not_awaited()
        assert "Image saying" in result


# ===========================================================================
# Rec 2 — History watermark
# ===========================================================================

class TestHistoryWatermarkResolution:

    def test_returns_none_when_forget_active(self):
        ml = _make_ml()
        ml.start_time = datetime(2024, 1, 1)
        ml._cog.context_front[123] = HistoryWatermark(1, 1.0, _monotonic_now())
        assert ml._get_history_watermark() is None

    def test_returns_none_when_absent(self):
        ml = _make_ml()
        assert ml._get_history_watermark() is None

    def test_returns_none_when_stale(self):
        ml = _make_ml()
        stale = _time.monotonic() - (WATERMARK_IDLE_SECONDS + 60)
        ml._cog.context_front[123] = HistoryWatermark(55, 1000.0, stale)
        assert ml._get_history_watermark() is None

    def test_returns_fresh_watermark(self):
        ml = _make_ml()
        wm = HistoryWatermark(55, 1000.0, _monotonic_now())
        ml._cog.context_front[123] = wm
        assert ml._get_history_watermark() is wm

    def test_non_watermark_object_ignored(self):
        """Guards against mock cogs / foreign cache entries."""
        ml = _make_ml()
        ml._cog.context_front[123] = MagicMock()
        assert ml._get_history_watermark() is None


def _monotonic_now():
    import time
    return time.monotonic()


import time as _time  # noqa: E402  (used above)


class TestWatermarkHistoryWalk:

    def test_floor_pinned_older_messages_excluded(self):
        """With a floor, the walk stops at the floor: nothing older is included."""
        cog = _make_cog()
        wm = HistoryWatermark(100, _ts(50), _monotonic_now())
        cog.context_front[123] = wm

        ml = _make_ml(cog=cog)
        # newest → oldest; 100 is the floor, 99/98 must be excluded
        before = [_fake_msg(i, minutes_ago=60 - i) for i in [104, 103, 102, 101, 100, 99, 98]]

        asyncio.run(ml._process_past_messages(before, [], 3600, wm))

        # add_msg records the source message id in messages_ids
        assert ml.messages_ids == {104, 103, 102, 101, 100}
        # Watermark re-confirmed at the floor
        stored = cog.context_front[123]
        assert stored.message_id == 100
        assert stored.created_at == _ts(50)

    def test_cold_start_stores_oldest_included(self):
        """Without a floor, the watermark becomes the oldest included message."""
        cog = _make_cog()
        ml = _make_ml(cog=cog)
        before = [_fake_msg(i, minutes_ago=60 - i) for i in [104, 103, 102]]

        asyncio.run(ml._process_past_messages(before, [], 3600, None))

        assert ml.messages_ids == {104, 103, 102}
        stored = cog.context_front[123]
        assert stored.message_id == 102

    def test_token_hysteresis_advances_front(self):
        """The walk stops at the hysteresis threshold, leaving headroom."""
        cog = _make_cog()
        ml = _make_ml(cog=cog, token_limit=1000, stop_tokens=700)
        # Each converted entry costs 300 stub tokens (1 token per word)
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", " ".join(["w"] * 300))])

        before = [_fake_msg(i, minutes_ago=60 - i) for i in [5, 4, 3, 2, 1]]

        asyncio.run(ml._process_past_messages(before, [], 3600, None))

        # 3 messages × 300 = 900 tokens; the 4th check (900 > 700) stops the walk
        assert ml.messages_ids == {5, 4, 3}
        assert ml.tokens == 900
        assert ml.tokens <= ml.token_limit
        # Front advanced to the oldest included message
        assert cog.context_front[123].message_id == 3

    def test_gap_break_advances_watermark(self):
        """A conversation time-gap forces a semantic reset of the watermark."""
        cog = _make_cog()
        ml = _make_ml(cog=cog)
        # 5, 4 are contiguous; 3 is 2 hours older than 4 → gap break at 3
        before = [
            _fake_msg(5, minutes_ago=0),
            _fake_msg(4, minutes_ago=1),
            _fake_msg(3, minutes_ago=121),
        ]

        asyncio.run(ml._process_past_messages(before, [], 3600, None))

        assert ml.messages_ids == {5, 4}
        assert cog.context_front[123].message_id == 4

    def test_floor_deleted_falls_back_to_oldest_included(self):
        """If the floor message is not in the fetched window, the watermark
        advances to the oldest included message (no stuck floor)."""
        cog = _make_cog()
        wm = HistoryWatermark(1000, _ts(500), _monotonic_now())
        cog.context_front[123] = wm

        ml = _make_ml(cog=cog)
        before = [_fake_msg(i, minutes_ago=60 - i) for i in [104, 103]]

        asyncio.run(ml._process_past_messages(before, [], 3600, wm))

        assert cog.context_front[123].message_id == 103


def _ts(minutes_ago):
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (base - timedelta(minutes=minutes_ago)).timestamp()


class TestAddBackfillWatermark:

    def test_backfill_pins_anchor_watermark(self):
        cog = _make_cog()
        ml = _make_ml(cog=cog, token_limit=100000)

        anchor = _fake_msg(50, minutes_ago=120)
        ml.init_message = _fake_msg(999, minutes_ago=0)
        ml.init_message.channel = MagicMock()
        # No mid messages
        async def _empty(*args, **kwargs):
            if False:
                yield  # pragma: no cover — empty async generator
        ml.init_message.channel.history = MagicMock(return_value=_empty())
        ml.bot.user.id = 789
        ml.config.guild = MagicMock(return_value=MagicMock(
            messages_backread=AsyncMock(return_value=10),
            messages_backread_seconds=AsyncMock(return_value=3600),
        ))

        asyncio.run(ml.add_backfill_history(anchor))

        assert cog.context_front[123].message_id == 50


# ===========================================================================
# Rec 3 — Cache hit stats
# ===========================================================================

class TestUpdateCacheHitStats:

    def _usage(self, cached=None, prompt=1000, details_cached=None):
        return SimpleNamespace(
            cached_tokens=cached,
            prompt_tokens=prompt,
            prompt_tokens_details=(
                SimpleNamespace(cached_tokens=details_cached)
                if details_cached is not None else None
            ),
        )

    def test_accumulates_direct_cached_tokens(self):
        cog = _make_cog()
        for _ in range(3):
            update_cache_hit_stats(cog, 123, self._usage(cached=400))
        assert cog.cache_hit_stats[123] == [3, 1200, 3000]

    def test_falls_back_to_prompt_tokens_details(self):
        cog = _make_cog()
        update_cache_hit_stats(cog, 123, self._usage(cached=None, details_cached=250))
        assert cog.cache_hit_stats[123] == [1, 250, 1000]

    def test_handles_missing_usage_fields(self):
        cog = _make_cog()
        update_cache_hit_stats(cog, 123, SimpleNamespace(prompt_tokens=500))
        assert cog.cache_hit_stats[123] == [1, 0, 500]

    def test_none_usage_and_none_cog_are_noops(self):
        cog = _make_cog()
        update_cache_hit_stats(cog, 123, None)
        update_cache_hit_stats(None, 123, self._usage(cached=1))
        assert 123 not in cog.cache_hit_stats

    def test_cog_without_stats_store_is_noop(self):
        cog = SimpleNamespace()  # no cache_hit_stats attribute
        update_cache_hit_stats(cog, 123, self._usage(cached=1))  # must not raise

    def test_ratio_logged_every_interval(self, caplog):
        cog = _make_cog()
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            for _ in range(20):
                update_cache_hit_stats(cog, 123, self._usage(cached=400))
        assert cog.cache_hit_stats[123] == [20, 8000, 20000]
        assert any("40.0%" in r.getMessage() for r in caplog.records)


# ===========================================================================
# Rec 4 — Unified count_tokens
# ===========================================================================

class TestCountTokens:

    def test_string_content(self):
        ml = _make_ml()
        assert ml.count_tokens("hello world") == 2

    def test_multimodal_content_parts(self):
        ml = _make_ml()
        with patch.object(messages_mod, "estimate_image_part_tokens", lambda url: 500), \
             patch.object(messages_mod, "estimate_file_part_tokens", lambda fd: 250):
            content = [
                {"type": "text", "text": "hello world"},
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,xxx"}},
                {"type": "file", "file": {"filename": "f.pdf", "file_data": "data:application/pdf;base64,yyy"}},
                "not-a-dict",
            ]
            assert ml.count_tokens(content) == 2 + 500 + 250

    def test_no_hardcoded_756_anywhere(self):
        """The reply-context path must use the unified estimator (no 756)."""
        import inspect
        source = inspect.getsource(MessagesList._insert_reply_context)
        assert "756" not in source
        assert "_add_content_tokens" in source

    def test_reply_context_uses_unified_estimator(self):
        """An image reply parent counts via estimate_image_part_tokens."""
        ml = _make_ml()
        parent = _DiscordMessage()
        parent.id = 555
        parent.author = MagicMock()
        parent.author.id = 1
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml._trigger_msg_start = 1
        ml.messages = [MessageEntry("system", "sys")]
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", [
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,abc"}},
        ])])

        with patch.object(messages_mod, "estimate_image_part_tokens", lambda url: 4321):
            asyncio.run(ml._insert_reply_context())

        assert ml.tokens == 4321


# ===========================================================================
# Rec 5 — PDF annotations injected at the tail
# ===========================================================================

class TestAnnotationTailInjection:

    def test_format_annotations_from_dicts(self):
        annotations = [
            {"url_citation": {"url": "https://example.com/a", "title": "A"}},
            {"url": "https://example.com/b"},
        ]
        text = LLMPipeline._format_annotations_as_text(annotations)
        assert text is not None
        assert "- A (https://example.com/a)" in text
        assert "- https://example.com/b" in text

    def test_format_annotations_from_objects(self):
        ann = SimpleNamespace(
            url_citation=SimpleNamespace(url="https://example.com/x", title="X"))
        text = LLMPipeline._format_annotations_as_text([ann])
        assert "- X (https://example.com/x)" in text

    def test_format_annotations_empty_returns_none(self):
        assert LLMPipeline._format_annotations_as_text([]) is None
        assert LLMPipeline._format_annotations_as_text([{"foo": 1}]) is None

    def _make_pipeline_with_annotations(self, annotations):
        pipeline = LLMPipeline.__new__(LLMPipeline)
        cog = _make_cog()
        cog.pdf_annotations[(123, 555)] = annotations
        pipeline.bot = MagicMock()
        pipeline.bot.get_cog = MagicMock(return_value=cog)
        pipeline.ctx = _make_ctx()
        pipeline.msg_list = SimpleNamespace(
            init_message=SimpleNamespace(
                id=999,
                reference=SimpleNamespace(resolved=SimpleNamespace(id=555)),
            ),
            messages=[MessageEntry("system", "sys")],
        )
        return pipeline

    def test_injection_appends_tail_user_message(self):
        pipeline = self._make_pipeline_with_annotations(
            [{"url_citation": {"url": "https://example.com/a", "title": "A"}}])

        asyncio.run(pipeline._inject_pdf_annotations())

        msgs = pipeline.msg_list.messages
        assert len(msgs) == 2
        assert msgs[-1].role == "user"
        assert "https://example.com/a" in msgs[-1].content

    def test_injection_never_mutates_existing_entries(self):
        """The prefix (system message) must remain untouched."""
        pipeline = self._make_pipeline_with_annotations(
            [{"url": "https://example.com/b"}])
        before = pipeline.msg_list.messages[0]

        asyncio.run(pipeline._inject_pdf_annotations())

        assert pipeline.msg_list.messages[0] is before

    def test_no_annotations_no_injection(self):
        pipeline = self._make_pipeline_with_annotations([])
        asyncio.run(pipeline._inject_pdf_annotations())
        assert len(pipeline.msg_list.messages) == 1

    def test_call_client_stops_injecting_into_assistant_messages(self):
        """call_client must no longer read _cached_pdf_annotations."""
        import inspect
        source = inspect.getsource(LLMPipeline.call_client)
        assert "_cached_pdf_annotations" not in source
        assert "annotations_for_assistant" not in source
