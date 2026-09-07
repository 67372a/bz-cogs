"""Tests for tool-related prompt-cache-miss fixes.

Covers:
  Fix C — Deterministic embed policy: only ``rich`` (author-intended) embeds
          are serialized; Discord link-preview unfurls (``link``/``image``/
          ``video``/...) are excluded so a preview arriving late cannot flip
          a message's serialization and diverge the request prefix.
  Fix A — Message-conversion pinning (``cog.converted_history``): the first
          serialization of a message is reused until the message is edited;
          multimodal (list-content) conversions are never pinned.

Also regression-tests the latent IndexError in the YouTube/no-embed path.
"""

import asyncio
import sys
import types
from datetime import datetime, timezone
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
# Force-assign: under pytest the real discord package may already be imported,
# and isinstance checks in the converter must see our lightweight stand-in.
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

from tests.mock_importer import _make_mock_package, import_module_directly


def _ensure_real_module(qualified_name: str, filepath: str):
    """Load *filepath* as *qualified_name* unless a real module already exists.

    Other test modules register MagicMock placeholders for these names; only
    replace those, never re-register a real module (re-registering would
    leave earlier modules bound to a stale object).
    """
    existing = sys.modules.get(qualified_name)
    if existing is not None and str(getattr(existing, "__file__", "")).endswith(".py"):
        return existing
    return import_module_directly(qualified_name, filepath)


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
    "aiuser.functions.scrape",
    "aiuser.response",
    "aiuser.response.chat",
    "aiuser.common",
]
for _pkg in _sub_pkgs:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

_leaf_mocks = [
    "aiuser.config.defaults",
    "aiuser.config.models",
    "aiuser.messages_list.opt_view",
    "aiuser.messages_list.converter.embed.youtube",
    "aiuser.messages_list.converter.image.AI_horde",
    "aiuser.messages_list.converter.image.local",
    "aiuser.types.abc",
    "aiuser.functions.tool_call",
    "aiuser.functions.scrape.tool_call",
    "aiuser.functions.openrouter",
    "aiuser.functions.openrouter.pdf_parsing",
]
for _m in _leaf_mocks:
    if _m not in sys.modules or not str(
            getattr(sys.modules[_m], "__file__", "")).endswith(".py"):
        sys.modules[_m] = MagicMock()

# Real modules (order matters: dependencies first)
sys.modules["aiuser.config.constants"] = _ensure_real_module(
    "aiuser.config.constants", "aiuser/config/constants.py")
_ensure_real_module("aiuser.messages_list.entry", "aiuser/messages_list/entry.py")
_ensure_real_module("aiuser.types.enums", "aiuser/types/enums.py")
_ensure_real_module("aiuser.utils.cache", "aiuser/utils/cache.py")
_ensure_real_module("aiuser.utils.image_cache", "aiuser/utils/image_cache.py")
_ensure_real_module("aiuser.utils.image_processing", "aiuser/utils/image_processing.py")
helpers_mod = _ensure_real_module(
    "aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py")
utilities_mod = _ensure_real_module("aiuser.utils.utilities", "aiuser/utils/utilities.py")
formatter_mod = _ensure_real_module(
    "aiuser.messages_list.converter.embed.formatter",
    "aiuser/messages_list/converter/embed/formatter.py")
_ensure_real_module(
    "aiuser.messages_list.converter.image.caption",
    "aiuser/messages_list/converter/image/caption.py")
# Force-reload the converter after its real dependencies are in place —
# earlier test modules register it while formatter/utilities are still mocks,
# leaving from-imports bound to MagicMock callables.
sys.modules["aiuser.messages_list.converter.converter"] = import_module_directly(
    "aiuser.messages_list.converter.converter",
    "aiuser/messages_list/converter/converter.py")
converter_mod = sys.modules["aiuser.messages_list.converter.converter"]

MessageConverter = converter_mod.MessageConverter
MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry
Cache = sys.modules["aiuser.utils.cache"].Cache
rich_embeds = helpers_mod.rich_embeds
first_rich_embed = helpers_mod.first_rich_embed
is_embed_valid = utilities_mod.is_embed_valid
format_embed_content = formatter_mod.format_embed_content
format_bot_embed_content = formatter_mod.format_bot_embed_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_ID = 789
USER_ID = 111


def _make_cog():
    cog = MagicMock()
    cog.bot_id = BOT_ID
    cog.cached_messages = Cache(limit=100)
    cog.converted_history = Cache(limit=2000)
    cog.bot = MagicMock()
    cog.bot.user.id = BOT_ID
    cog.bot.get_shared_api_tokens = AsyncMock(return_value={})
    guild_cfg = MagicMock(
        scan_images=AsyncMock(return_value=False),
        openrouter_image_parsing_enabled=AsyncMock(return_value=False),
        openrouter_pdf_parsing_enabled=AsyncMock(return_value=False),
        function_calling_functions=AsyncMock(return_value=[]),
        max_image_pixels=AsyncMock(return_value=None),
    )
    cog.config = MagicMock()
    cog.config.guild = MagicMock(return_value=guild_cfg)
    return cog


def _make_converter(cog=None):
    cog = cog or _make_cog()
    ctx = MagicMock()
    ctx.message = MagicMock()
    ctx.interaction = MagicMock()  # truthy → trigger-only scan paths disabled
    return MessageConverter(cog, ctx), cog


def _make_message(msg_id=1000, content="hello world", author_id=USER_ID,
                  embeds=None, edited_at=None):
    msg = _DiscordMessage()
    msg.id = msg_id
    msg.content = content
    msg.type = MagicMock()
    msg.type.value = 0
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.name = "testuser"
    msg.author.display_name = "Test User"
    msg.author.bot = author_id == BOT_ID
    msg.guild = MagicMock()
    msg.guild.id = 456
    msg.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    msg.edited_at = edited_at
    msg.reference = None
    msg.attachments = []
    msg.embeds = embeds or []
    msg.stickers = []
    msg.channel = SimpleNamespace(id=123)
    msg.mentions = []
    msg.role_mentions = []
    msg.channel_mentions = []
    return msg


def _preview_embed(etype="link", title="Example Page", description="A page"):
    return SimpleNamespace(type=etype, title=title, description=description)


# ===========================================================================
# Fix C — rich-embed filtering
# ===========================================================================

class TestRichEmbedFilter:

    def test_link_previews_excluded(self):
        msg = _make_message(embeds=[_preview_embed("link"), _preview_embed("image")])
        assert rich_embeds(msg) == []

    def test_rich_embeds_kept(self):
        rich = _preview_embed("rich")
        msg = _make_message(embeds=[_preview_embed("link"), rich])
        assert rich_embeds(msg) == [rich]

    def test_is_embed_valid_rejects_preview_only(self):
        msg = _make_message(embeds=[_preview_embed("link")])
        assert is_embed_valid(msg) is False

    def test_is_embed_valid_accepts_rich(self):
        msg = _make_message(embeds=[_preview_embed("rich")])
        assert is_embed_valid(msg) is True

    def test_is_embed_valid_empty(self):
        assert is_embed_valid(_make_message()) is False


class TestConvertUnfurlStability:

    def test_preview_arrival_does_not_change_serialization(self):
        """The classic web-search miss: a URL message converted before and
        after its link preview unfurls must serialize identically."""
        conv, _ = _make_converter()
        msg = _make_message(content="check https://example.com/page")

        before = asyncio.run(conv.convert(msg))

        msg.embeds = [_preview_embed("link")]  # unfurl arrives
        after = asyncio.run(conv.convert(msg))

        assert before is not None and after is not None
        assert [(e.role, e.content) for e in after] == [(e.role, e.content) for e in before]

    def test_rich_embed_serialized(self):
        conv, _ = _make_converter()
        msg = _make_message(
            content="look at this",
            embeds=[_preview_embed("rich", title="Cool Thing", description="details")],
        )
        res = asyncio.run(conv.convert(msg))
        joined = " ".join(str(e.content) for e in res)
        assert "<embed title=\"Cool Thing\">details</embed>" in joined

    def test_bot_response_uses_own_embed_only(self):
        """The bot's own response embed is read via the rich filter; preview
        embeds appended by Discord must not leak into the serialization."""
        conv, _ = _make_converter()
        msg = _make_message(
            msg_id=2000,
            content="",
            author_id=BOT_ID,
            embeds=[
                _preview_embed("rich", title="Bot's Response",
                               description="The answer with citations"),
                _preview_embed("link", title="Late Preview", description="noise"),
            ],
        )
        res = asyncio.run(conv.convert(msg))
        assert res and res[0].role == "assistant"
        assert "The answer with citations" in str(res[0].content)
        assert "Late Preview" not in str(res[0].content)

    def test_youtube_without_api_key_no_indexerror(self):
        """Regression: a YouTube link with no embeds and no API key must fall
        back to the text path instead of raising IndexError on embeds[0]."""
        conv, _ = _make_converter()
        msg = _make_message(content="watch https://www.youtube.com/watch?v=abc123")
        res = asyncio.run(conv.convert(msg))  # must not raise
        assert res is not None
        assert "youtube" in str(res[0].content)

    def test_youtube_serialization_stable_when_preview_arrives(self):
        conv, _ = _make_converter()
        msg = _make_message(content="watch https://www.youtube.com/watch?v=abc123")
        before = asyncio.run(conv.convert(msg))
        msg.embeds = [_preview_embed("video", title="Video", description="d")]
        after = asyncio.run(conv.convert(msg))
        assert [(e.role, e.content) for e in after] == [(e.role, e.content) for e in before]


# ===========================================================================
# Fix A — conversion pinning (converted_history)
# ===========================================================================

class TestConversionPinning:

    def test_second_convert_reuses_pinned_entries(self):
        conv, cog = _make_converter()
        msg = _make_message(content="hello world")

        first = asyncio.run(conv.convert(msg))
        assert (123, 1000) in cog.converted_history

        # Force a different serialization path on re-conversion: if the
        # pinning works, the stored (first) form is returned instead.
        msg.embeds = [_preview_embed("link", title="Sneaky", description="x")]
        second = asyncio.run(conv.convert(msg))

        assert [(e.role, e.content) for e in second] == [(e.role, e.content) for e in first]
        assert "Sneaky" not in str(second[0].content)

    def test_edit_invalidates_pinned_entries(self):
        conv, cog = _make_converter()
        msg = _make_message(content="original")

        first = asyncio.run(conv.convert(msg))
        assert cog.converted_history[(123, 1000)][0] == 0.0

        msg.content = "edited content"
        msg.edited_at = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
        second = asyncio.run(conv.convert(msg))

        assert "edited content" in str(second[0].content)
        assert cog.converted_history[(123, 1000)][0] == msg.edited_at.timestamp()

    def test_multimodal_results_not_pinned(self):
        """PDF/image-URL parts must stay per-request (they are trigger-only
        enrichments and can carry large base64 payloads)."""
        conv, cog = _make_converter()
        conv.init_msg.id = 1000  # PDF parsing only fires for the trigger message
        msg = _make_message(content="read https://example.com/doc.pdf")

        pdf_mod = sys.modules["aiuser.functions.openrouter.pdf_parsing"]
        pdf_mod.OpenRouterPdfParsing = MagicMock()
        pdf_mod.OpenRouterPdfParsing.return_value.process_message_for_pdfs = AsyncMock(
            return_value=(
                [{"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,xxx"}}],
                ["doc.pdf"],
            ))

        with patch.object(conv.config, "guild", new=MagicMock(return_value=MagicMock(
                openrouter_pdf_parsing_enabled=AsyncMock(return_value=True),
                scan_images=AsyncMock(return_value=False),
                openrouter_image_parsing_enabled=AsyncMock(return_value=False),
                function_calling_functions=AsyncMock(return_value=[]),
                max_image_pixels=AsyncMock(return_value=None)))):
            res = asyncio.run(conv.convert(msg))

        assert res is not None
        assert any(isinstance(e.content, list) for e in res)
        assert (123, 1000) not in cog.converted_history

    def test_role_pinned_for_bot_messages(self):
        conv, _ = _make_converter()
        msg = _make_message(msg_id=3000, content="bot speaking", author_id=BOT_ID)

        first = asyncio.run(conv.convert(msg))
        second = asyncio.run(conv.convert(msg))

        assert first[0].role == "assistant"
        assert second[0].role == "assistant"

    def test_missing_cache_attr_is_tolerated(self):
        """A cog without converted_history (older shim/tests) still works."""
        conv, cog = _make_converter()
        del cog.converted_history
        msg = _make_message(content="hello")
        res = asyncio.run(conv.convert(msg))  # must not raise
        assert res is not None

    def test_separate_channels_do_not_collide(self):
        conv, cog = _make_converter()
        msg_a = _make_message(msg_id=1000, content="channel A message")
        msg_a.channel = SimpleNamespace(id=1)
        msg_b = _make_message(msg_id=1000, content="channel B message")
        msg_b.channel = SimpleNamespace(id=2)

        res_a = asyncio.run(conv.convert(msg_a))
        res_b = asyncio.run(conv.convert(msg_b))

        assert "channel A message" in str(res_a[0].content)
        assert "channel B message" in str(res_b[0].content)
