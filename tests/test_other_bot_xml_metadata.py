"""Tests for XML metadata preservation on OTHER bots' messages.

Regression tests for the bug where another bot's message carrying an
embed titled "X's Response" (e.g. a second aiuser instance) was routed
through ``format_bot_embed_content`` — which returns the raw embed
description with no ``<message ...>`` XML metadata — because the routing
check only matched the embed title regex without verifying the author.

The fix routes only THIS bot's own response embeds through
``format_bot_embed_content``; all other authors' embeds go through
``format_embed_content`` which wraps them in the standard XML schema.
"""

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock discord / redbot before importing aiuser modules
# ---------------------------------------------------------------------------
discord_mock = types.ModuleType("discord")
discord_mock.__package__ = "discord"
discord_mock.__path__ = []


class _DiscordMessage:
    pass


discord_mock.Message = _DiscordMessage
discord_mock.MessageType = MagicMock()
discord_mock.MessageType.new_member = 7
discord_mock.Interaction = MagicMock()
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
discord_mock.Forbidden = type("Forbidden", (Exception,), {})
discord_mock.NotFound = type("NotFound", (Exception,), {})
discord_mock.HTTPException = type("HTTPException", (Exception,), {})
discord_mock.TextChannel = MagicMock()
discord_mock.Thread = MagicMock()
discord_mock.Member = MagicMock()
discord_mock.User = MagicMock()
discord_mock.Guild = MagicMock()
discord_mock.Role = MagicMock()
discord_mock.Attachment = MagicMock()
discord_mock.Sticker = MagicMock()
discord_mock.PartialMessage = MagicMock()
discord_mock.MessageReference = MagicMock()
discord_mock.AllowedMentions = MagicMock()
discord_mock.File = MagicMock()
discord_mock.abc = MagicMock()
discord_mock.utils = MagicMock()
discord_mock.errors = MagicMock()
sys.modules["discord"] = discord_mock

sys.modules.setdefault("discord.ext", types.ModuleType("discord.ext"))
sys.modules.setdefault("discord.ext.commands", MagicMock())
sys.modules.setdefault("discord.app_commands", MagicMock())

_redbot_core = types.ModuleType("redbot.core")
_redbot_core.__package__ = "redbot.core"
_redbot_core.__path__ = []
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
sys.modules.setdefault("redbot", types.ModuleType("redbot"))
sys.modules.setdefault("redbot.core", _redbot_core)
for _n in ["redbot.core.commands", "redbot.core.bot", "redbot.core.utils",
           "redbot.core.utils.chat_formatting", "redbot.core.utils.views"]:
    sys.modules.setdefault(_n, MagicMock())

sys.modules.setdefault("tiktoken", MagicMock())

from tests.mock_importer import _make_mock_package, import_module_directly  # noqa: E402

for _pkg in ["aiuser", "aiuser.messages_list", "aiuser.messages_list.converter",
             "aiuser.messages_list.converter.embed", "aiuser.messages_list.converter.image",
             "aiuser.config", "aiuser.types", "aiuser.utils",
             "aiuser.functions", "aiuser.functions.scrape"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

for _m in ["aiuser.messages_list.converter.embed.formatter",
           "aiuser.messages_list.converter.embed.youtube",
           "aiuser.functions.scrape.tool_call",
           "aiuser.utils.utilities",
           "aiuser.types.abc"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Real modules needed for actual XML output
sys.modules["aiuser.config.constants"] = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
sys.modules["aiuser.messages_list.converter.helpers"] = import_module_directly(
    "aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py"
)

# caption.py imports image_cache / AI_horde; provide a stub so converter.py's
# `from ...caption import transcribe_image` works.
if "aiuser.messages_list.converter.image.caption" not in sys.modules:
    _caption_stub = types.ModuleType("aiuser.messages_list.converter.image.caption")
    _caption_stub.transcribe_image = AsyncMock()
    _caption_stub.transcribe_single = AsyncMock()
    sys.modules["aiuser.messages_list.converter.image.caption"] = _caption_stub

converter_mod = import_module_directly(
    "aiuser.messages_list.converter.converter", "aiuser/messages_list/converter/converter.py"
)

# Import the REAL embed formatter under a unique name so the shared MagicMock
# at aiuser.messages_list.converter.embed.formatter (used by other test files)
# is left untouched.
_real_formatter = import_module_directly(
    "aiuser.messages_list.converter.embed.formatter_realtest",
    "aiuser/messages_list/converter/embed/formatter.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_ID = 999
OTHER_BOT_ID = 777


def _make_author(author_id=42, name="testuser", display="Test User", bot=False):
    author = MagicMock()
    author.id = author_id
    author.name = name
    author.display_name = display
    author.bot = bot
    return author


def _make_guild(me_id=BOT_ID):
    guild = MagicMock()
    guild.me.id = me_id
    return guild


def _make_embed(title, description):
    embed = MagicMock()
    embed.title = title
    embed.description = description
    return embed


def _make_message(author=None, content="", embeds=None, reference=None, msg_id=123456789):
    msg = MagicMock()
    msg.id = msg_id
    msg.author = author or _make_author()
    msg.guild = _make_guild()
    msg.channel.id = 555
    msg.content = content
    msg.attachments = []
    msg.stickers = []
    msg.embeds = embeds or []
    msg.reference = reference
    msg.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return msg


def _make_converter():
    cog = MagicMock()
    cog.bot.user.id = BOT_ID
    cog.cached_messages = {}
    cog.bot.get_shared_api_tokens = AsyncMock(return_value={})
    ctx = MagicMock()
    ctx.message = _make_message(msg_id=1)
    ctx.interaction = None
    return converter_mod.MessageConverter(cog, ctx), cog


def _other_bot_msg(embeds=None, content="", msg_id=123456789):
    return _make_message(
        author=_make_author(author_id=OTHER_BOT_ID, name="otherbot",
                            display="Other Bot", bot=True),
        content=content, embeds=embeds, msg_id=msg_id,
    )


def _own_bot_msg(embeds=None, content="", msg_id=123456790):
    return _make_message(
        author=_make_author(author_id=BOT_ID, name="mybot",
                            display="My Bot", bot=True),
        content=content, embeds=embeds, msg_id=msg_id,
    )


# ---------------------------------------------------------------------------
# _is_own_response_embed predicate
# ---------------------------------------------------------------------------

class TestIsOwnResponseEmbed:
    def test_own_bot_response_embed_matches(self):
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed("My Bot's Response", "hello")])
        assert conv._is_own_response_embed(msg) is True

    def test_other_bot_response_embed_rejected(self):
        """The regression: another aiuser bot's response embed must NOT match."""
        conv, _ = _make_converter()
        msg = _other_bot_msg(embeds=[_make_embed("Other Bot's Response", "hello")])
        assert conv._is_own_response_embed(msg) is False

    def test_human_author_response_titled_embed_rejected(self):
        conv, _ = _make_converter()
        msg = _make_message(embeds=[_make_embed("testuser's Response", "hello")])
        assert conv._is_own_response_embed(msg) is False

    def test_own_bot_non_response_title_rejected(self):
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed("Some Random Embed", "hello")])
        assert conv._is_own_response_embed(msg) is False

    def test_no_embeds(self):
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[])
        assert conv._is_own_response_embed(msg) is False

    def test_none_title_does_not_raise(self):
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed(None, "hello")])
        assert conv._is_own_response_embed(msg) is False


# ---------------------------------------------------------------------------
# handle_embed routing (unit — formatters mocked)
# ---------------------------------------------------------------------------

class TestHandleEmbedRouting:
    @pytest.mark.asyncio
    async def test_other_bot_response_embed_uses_standard_formatter(self, monkeypatch):
        conv, _ = _make_converter()
        msg = _other_bot_msg(embeds=[_make_embed("Other Bot's Response", "a generated cat")])

        fake_standard = AsyncMock(return_value="<message ...><embed/></message>")
        fake_bot = AsyncMock(return_value="a generated cat")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_standard)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content", fake_bot)

        res = []
        await conv.handle_embed(msg, res, "user")

        fake_standard.assert_awaited_once()
        fake_bot.assert_not_called()
        assert res[0].content == "<message ...><embed/></message>"

    @pytest.mark.asyncio
    async def test_own_bot_response_embed_uses_bot_formatter(self, monkeypatch):
        """Own response embeds keep the historical raw-text behaviour."""
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed("My Bot's Response", "a generated cat")])

        fake_standard = AsyncMock(return_value="<message ...><embed/></message>")
        fake_bot = AsyncMock(return_value="a generated cat")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_standard)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content", fake_bot)

        res = []
        await conv.handle_embed(msg, res, "assistant")

        fake_bot.assert_awaited_once()
        fake_standard.assert_not_called()
        assert res[0].content == "a generated cat"


# ---------------------------------------------------------------------------
# _format_embed_content routing (attachment + embed path)
# ---------------------------------------------------------------------------

class TestFormatEmbedContentHelperRouting:
    @pytest.mark.asyncio
    async def test_other_bot_response_embed_uses_standard_formatter(self, monkeypatch):
        conv, _ = _make_converter()
        msg = _other_bot_msg(embeds=[_make_embed("Other Bot's Response", "caption")])

        fake_standard = AsyncMock(return_value="<message ...><embed/></message>")
        fake_bot = AsyncMock(return_value="caption")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_standard)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content", fake_bot)

        out = await conv._format_embed_content(msg)

        fake_standard.assert_awaited_once()
        fake_bot.assert_not_called()
        assert out == "<message ...><embed/></message>"

    @pytest.mark.asyncio
    async def test_own_bot_response_embed_uses_bot_formatter(self, monkeypatch):
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed("My Bot's Response", "caption")])

        fake_standard = AsyncMock(return_value="<message ...><embed/></message>")
        fake_bot = AsyncMock(return_value="caption")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_standard)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content", fake_bot)

        out = await conv._format_embed_content(msg)

        fake_bot.assert_awaited_once()
        fake_standard.assert_not_called()
        assert out == "caption"


# ---------------------------------------------------------------------------
# Functional tests with the REAL format_embed_content
# ---------------------------------------------------------------------------

class TestRealFormatterXmlMetadata:
    @pytest.mark.asyncio
    async def test_other_bot_embed_wrapped_in_xml(self, monkeypatch):
        """Real formatter output for another bot's response embed must carry
        the full <message ...> XML metadata header."""
        conv, _ = _make_converter()
        msg = _other_bot_msg(embeds=[_make_embed("Other Bot's Response", "a generated cat")],
                             content="")

        monkeypatch.setattr(converter_mod, "format_embed_content",
                            _real_formatter.format_embed_content)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content",
                            _real_formatter.format_bot_embed_content)

        res = []
        await conv.handle_embed(msg, res, "user")

        content = res[0].content
        assert content.startswith("<message ")
        assert 'id="123456789"' in content
        assert 'author_id="777"' in content
        assert 'username="otherbot"' in content
        assert 'displayname="Other Bot"' in content
        assert '<embed title="Other Bot\'s Response">a generated cat</embed>' in content
        assert content.endswith("</message>")

    @pytest.mark.asyncio
    async def test_convert_end_to_end_other_bot_keeps_metadata(self, monkeypatch):
        """Full convert() pipeline: another bot's response embed enters context
        as a user-role message with XML author metadata."""
        conv, _ = _make_converter()
        msg = _other_bot_msg(embeds=[_make_embed("Other Bot's Response", "a generated cat")],
                             content="")

        monkeypatch.setattr(converter_mod, "format_embed_content",
                            _real_formatter.format_embed_content)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content",
                            _real_formatter.format_bot_embed_content)

        converted = await conv.convert(msg)

        assert converted is not None
        assert converted[0].role == "user"
        assert converted[0].content.startswith("<message ")
        assert 'author_id="777"' in converted[0].content
        assert 'username="otherbot"' in converted[0].content

    @pytest.mark.asyncio
    async def test_own_bot_response_embed_wrapped_in_xml(self, monkeypatch):
        """Own response embeds are now wrapped in XML metadata like all messages."""
        conv, _ = _make_converter()
        msg = _own_bot_msg(embeds=[_make_embed("My Bot's Response", "a generated cat")],
                           content="")

        monkeypatch.setattr(converter_mod, "format_embed_content",
                            _real_formatter.format_embed_content)
        monkeypatch.setattr(converter_mod, "format_bot_embed_content",
                            _real_formatter.format_bot_embed_content)

        converted = await conv.convert(msg)

        assert converted is not None
        assert converted[0].role == "assistant"
        assert converted[0].content.startswith("<message ")
        assert 'author_id="999"' in converted[0].content
        assert 'username="mybot"' in converted[0].content
        assert converted[0].content.endswith("</message>")
        assert "a generated cat" in converted[0].content
