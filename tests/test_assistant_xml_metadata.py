"""Tests for XML metadata applied to assistant (bot's own) messages.

Validates that the bot's own messages in history are now wrapped in the same
<message> XML tags that user messages receive, providing consistent metadata
(message ID, timestamp, author info, reply context) for ALL messages in context.
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

# Use MagicMock for openrouter package so attribute access auto-creates sub-modules
# (avoids polluting sys.modules with a real ModuleType that breaks other test files)
if "aiuser.functions.openrouter" not in sys.modules:
    sys.modules["aiuser.functions.openrouter"] = MagicMock()

for _m in ["aiuser.messages_list.converter.embed.formatter",
           "aiuser.messages_list.converter.embed.youtube",
           "aiuser.functions.openrouter.image_parsing",
           "aiuser.functions.openrouter.pdf_parsing",
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

# caption.py stub
if "aiuser.messages_list.converter.image.caption" not in sys.modules:
    _caption_stub = types.ModuleType("aiuser.messages_list.converter.image.caption")
    _caption_stub.transcribe_image = AsyncMock()
    _caption_stub.transcribe_single = AsyncMock()
    sys.modules["aiuser.messages_list.converter.image.caption"] = _caption_stub

converter_mod = import_module_directly(
    "aiuser.messages_list.converter.converter", "aiuser/messages_list/converter/converter.py"
)

from aiuser.messages_list.converter.helpers import (  # noqa: E402
    format_text_content,
    format_embed_text_content,
    format_generic_image,
    format_generic_document,
    format_sticker_content,
    _get_msg_header,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_ID = 999


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


def _make_message(author=None, content="", attachments=None, reference=None,
                  msg_id=123456789, embeds=None, stickers=None):
    msg = MagicMock()
    msg.id = msg_id
    msg.author = author or _make_author()
    msg.guild = _make_guild()
    msg.channel.id = 555
    msg.content = content
    msg.attachments = attachments or []
    msg.embeds = embeds or []
    msg.stickers = stickers or []
    msg.reference = reference
    msg.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    msg.type = MagicMock()
    msg.type.new_member = 7
    return msg


def _make_attachment(filename="photo.png", content_type="image/png", size=100,
                     title=None, description=None):
    att = MagicMock()
    att.filename = filename
    att.content_type = content_type
    att.size = size
    att.title = title
    att.description = description
    return att


def _make_converter():
    cog = MagicMock()
    cog.bot.user.id = BOT_ID
    cog.cached_messages = {}
    cog.bot.get_shared_api_tokens = AsyncMock(return_value={})
    config = MagicMock()
    config.guild.return_value.max_image_size = AsyncMock(return_value=10_000_000)
    config.guild.return_value.scan_images = AsyncMock(return_value=False)
    config.guild.return_value.openrouter_image_parsing_enabled = AsyncMock(return_value=False)
    cog.config = config
    ctx = MagicMock()
    ctx.message = _make_message(msg_id=1)
    ctx.interaction = None
    return converter_mod.MessageConverter(cog, ctx), cog


def _own_bot_msg(content="hello", msg_id=123456790, **kwargs):
    author = _make_author(author_id=BOT_ID, name="mybot", display="My Bot", bot=True)
    return _make_message(author=author, content=content, msg_id=msg_id, **kwargs)


def _user_msg(content="hello", msg_id=123456791, **kwargs):
    author = _make_author(author_id=42, name="testuser", display="Test User")
    return _make_message(author=author, content=content, msg_id=msg_id, **kwargs)


# ---------------------------------------------------------------------------
# format_text_content: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatTextContentAssistant:
    def test_bot_text_wrapped_in_xml(self):
        """Bot's own text messages should be wrapped in <message> XML tags."""
        msg = _own_bot_msg(content="Here is my response")
        out = format_text_content(msg)
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert 'username="mybot"' in out
        assert 'displayname="My Bot"' in out
        assert "Here is my response" in out
        assert out.endswith("</message>")

    def test_user_text_wrapped_in_xml(self):
        """User messages continue to be wrapped in <message> XML tags."""
        msg = _user_msg(content="Hello bot")
        out = format_text_content(msg)
        assert out.startswith("<message ")
        assert 'author_id="42"' in out
        assert "Hello bot" in out
        assert out.endswith("</message>")

    def test_bot_empty_content_returns_none(self):
        """Bot messages with empty content should return None."""
        msg = _own_bot_msg(content="")
        assert format_text_content(msg) is None

    def test_bot_whitespace_content_returns_none(self):
        """Bot messages with whitespace-only content should return None."""
        msg = _own_bot_msg(content="   ")
        assert format_text_content(msg) is None


# ---------------------------------------------------------------------------
# format_embed_text_content: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatEmbedTextContentAssistant:
    def test_bot_embed_text_wrapped_in_xml(self):
        """Bot's own embed text should be wrapped in <message> XML tags."""
        msg = _own_bot_msg(content="some text with https://example.com link")
        out = format_embed_text_content(msg)
        assert out is not None
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert out.endswith("</message>")


# ---------------------------------------------------------------------------
# format_generic_image: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatGenericImageAssistant:
    def test_bot_image_wrapped_in_xml(self):
        """Bot's own image attachments should be wrapped in <message> XML tags."""
        att = _make_attachment(filename="selfie.png", content_type="image/png")
        msg = _own_bot_msg(attachments=[att])
        out = format_generic_image(msg)
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert '<image filename="selfie.png"/>' in out
        assert out.endswith("</message>")
        assert "Sent" not in out

    def test_user_image_wrapped_in_xml(self):
        """User image attachments should also be wrapped in <message> XML tags."""
        att = _make_attachment(filename="photo.png", content_type="image/png")
        msg = _user_msg(attachments=[att])
        out = format_generic_image(msg)
        assert out.startswith("<message ")
        assert 'author_id="42"' in out
        assert '<image filename="photo.png"/>' in out
        assert out.endswith("</message>")


# ---------------------------------------------------------------------------
# format_generic_document: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatGenericDocumentAssistant:
    def test_bot_document_wrapped_in_xml(self):
        """Bot's own document attachments should be wrapped in <message> XML tags."""
        att = _make_attachment(filename="report.pdf", content_type="application/pdf")
        msg = _own_bot_msg(attachments=[att])
        out = format_generic_document(msg)
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert '<file filename="report.pdf"/>' in out
        assert out.endswith("</message>")
        assert "Sent" not in out


# ---------------------------------------------------------------------------
# format_extra_attachments: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatExtraAttachmentsAssistant:
    def test_bot_extra_attachments_wrapped_in_xml(self):
        """Bot's extra attachments should be wrapped in <message> XML tags."""
        msg = _own_bot_msg(attachments=[
            _make_attachment(filename="a.png", content_type="image/png"),
            _make_attachment(filename="b.zip", content_type="application/zip"),
        ])
        out = converter_mod.format_extra_attachments(msg)
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert '<file filename="b.zip"/>' in out
        assert out.endswith("</message>")
        assert "Sent" not in out


# ---------------------------------------------------------------------------
# format_generic_attachment: assistant messages get XML wrapper
# ---------------------------------------------------------------------------

class TestFormatGenericAttachmentAssistant:
    def test_bot_unsupported_attachment_wrapped_in_xml(self):
        """Bot's unsupported attachments should be wrapped in <message> XML tags."""
        att = _make_attachment(filename="data.bin", content_type="application/octet-stream")
        msg = _own_bot_msg(attachments=[att])
        out = converter_mod.format_generic_attachment(msg)
        assert out.startswith("<message ")
        assert 'author_id="999"' in out
        assert '<file filename="data.bin"/>' in out
        assert out.endswith("</message>")
        assert "Sent" not in out


# ---------------------------------------------------------------------------
# _get_msg_header: works correctly for bot messages
# ---------------------------------------------------------------------------

class TestGetMsgHeaderAssistant:
    def test_bot_header_contains_all_metadata(self):
        """_get_msg_header should produce complete metadata for bot messages."""
        msg = _own_bot_msg(content="test", msg_id=98765)
        header = _get_msg_header(msg)
        assert header.startswith('<message ')
        assert 'id="98765"' in header
        assert 'timestamp="2024-01-01T12:00:00Z"' in header
        assert 'author_id="999"' in header
        assert 'username="mybot"' in header
        assert 'displayname="My Bot"' in header

    def test_bot_header_with_reply_reference(self):
        """_get_msg_header should include reply info for bot messages replying to someone."""
        replied = _DiscordMessage()
        replied.id = 111
        replied.author = _make_author(author_id=42, name="testuser", display="Test User")
        reference = MagicMock()
        reference.resolved = replied
        msg = _own_bot_msg(content="replying", reference=reference)
        header = _get_msg_header(msg)
        assert 'reply_to_id="111"' in header
        assert 'reply_target_id="42"' in header
        assert 'reply_target_username="testuser"' in header


# ---------------------------------------------------------------------------
# End-to-end: convert() produces assistant role with XML metadata
# ---------------------------------------------------------------------------

class TestConvertAssistantMetadata:
    @pytest.mark.asyncio
    async def test_convert_bot_text_message_has_xml_and_assistant_role(self, monkeypatch):
        """A bot text message should produce an assistant-role entry with XML metadata."""
        monkeypatch.setattr(converter_mod, "contains_youtube_link", lambda content: False)
        monkeypatch.setattr(converter_mod, "is_embed_valid", lambda message: False)
        conv, _ = _make_converter()
        msg = _own_bot_msg(content="I am the bot")
        converted = await conv.convert(msg)
        assert converted is not None
        assert len(converted) == 1
        assert converted[0].role == "assistant"
        assert converted[0].content.startswith("<message ")
        assert 'author_id="999"' in converted[0].content
        assert "I am the bot" in converted[0].content
        assert converted[0].content.endswith("</message>")

    @pytest.mark.asyncio
    async def test_convert_user_text_message_has_xml_and_user_role(self, monkeypatch):
        """A user text message should produce a user-role entry with XML metadata."""
        monkeypatch.setattr(converter_mod, "contains_youtube_link", lambda content: False)
        monkeypatch.setattr(converter_mod, "is_embed_valid", lambda message: False)
        conv, _ = _make_converter()
        msg = _user_msg(content="I am the user")
        converted = await conv.convert(msg)
        assert converted is not None
        assert len(converted) == 1
        assert converted[0].role == "user"
        assert converted[0].content.startswith("<message ")
        assert 'author_id="42"' in converted[0].content
        assert "I am the user" in converted[0].content


# ---------------------------------------------------------------------------
# System prompt: documents unified XML format
# ---------------------------------------------------------------------------

class TestSystemPromptUnifiedFormat:
    def test_mentions_all_messages_including_past_responses(self):
        from aiuser.config.constants import XML_SYSTEM_PROMPT_APPENDIX
        assert "all messages" in XML_SYSTEM_PROMPT_APPENDIX.lower()
        assert "your own past responses" in XML_SYSTEM_PROMPT_APPENDIX.lower()

    def test_no_sent_convention(self):
        from aiuser.config.constants import XML_SYSTEM_PROMPT_APPENDIX
        # The old "Sent" convention should no longer be mentioned
        assert 'Messages prefixed with "Sent"' not in XML_SYSTEM_PROMPT_APPENDIX

    def test_instructs_author_id_matching(self):
        from aiuser.config.constants import XML_SYSTEM_PROMPT_APPENDIX
        assert "author_id" in XML_SYSTEM_PROMPT_APPENDIX
