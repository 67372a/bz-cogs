"""Tests for context XML/media defect fixes.

Covers:
- C1: bot response text included when caching generated images for re-ingestion
- C2: attachments with content_type=None don't crash conversion
- C3: filenames with double quotes are escaped in <file/> and <document> tags
- Cov5: non-image attachments in multi-image messages and attachments[1:]
        get placeholder markers instead of being silently dropped
- Cov6: image-URL text handling does not consume messages that have attachments
- Cov8: XML_SYSTEM_PROMPT_APPENDIX documents <document> and the "Sent" convention
"""

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure real PIL (other test files may have mocked it)
# ---------------------------------------------------------------------------
for _m in list(sys.modules):
    if _m == "PIL" or _m.startswith("PIL."):
        del sys.modules[_m]

# ---------------------------------------------------------------------------
# Mock discord / redbot before importing aiuser modules
# ---------------------------------------------------------------------------
if "discord" not in sys.modules:
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
    discord_mock.AllowedMentions = MagicMock()
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
             "aiuser.response", "aiuser.response.chat"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

for _m in ["aiuser.messages_list.converter.embed.formatter",
           "aiuser.messages_list.converter.embed.youtube",
           "aiuser.messages_list.converter.image.AI_horde",
           "aiuser.messages_list.converter.image.local",
           "aiuser.utils.utilities",
           "aiuser.utils.image_cache",
           "aiuser.types.abc",
           "aiuser.messages_list.messages",
           "aiuser.response.chat.llm_pipeline",
           "aiuser.response.chat.function_call_view",
           "aiuser.utils.latex_converter",
           # mocked (not the real module) so this file never pulls in
           # numpy/cv2, which would break other test files that delete and
           # re-import numpy from sys.modules
           "aiuser.utils.image_processing"]:
    if _m not in sys.modules or not hasattr(sys.modules[_m], "__file__"):
        sys.modules[_m] = MagicMock()

sys.modules["aiuser.config.constants"] = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
sys.modules["aiuser.types.enums"] = import_module_directly(
    "aiuser.types.enums", "aiuser/types/enums.py"
)
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
sys.modules["aiuser.messages_list.converter.helpers"] = import_module_directly(
    "aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py"
)

_caption_stub = types.ModuleType("aiuser.messages_list.converter.image.caption")
_caption_stub.transcribe_image = AsyncMock()
_caption_stub.transcribe_single = AsyncMock()
sys.modules["aiuser.messages_list.converter.image.caption"] = _caption_stub

sys.modules["aiuser.messages_list.converter.converter"] = import_module_directly(
    "aiuser.messages_list.converter.converter", "aiuser/messages_list/converter/converter.py"
)

from aiuser.messages_list.converter import converter as converter_mod  # noqa: E402
from aiuser.messages_list.converter import helpers as helpers_mod  # noqa: E402
from aiuser.config import constants as constants_mod  # noqa: E402

response_mod = import_module_directly(
    "aiuser.response.chat.response", "aiuser/response/chat/response.py"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_author(author_id=42, name="testuser", display="Test User"):
    author = MagicMock()
    author.id = author_id
    author.name = name
    author.display_name = display
    return author


def _make_guild(me_id=999):
    guild = MagicMock()
    guild.me.id = me_id
    return guild


def _make_attachment(filename="photo.png", content_type="image/png", size=100,
                     title=None, description=None):
    att = MagicMock()
    att.filename = filename
    att.content_type = content_type
    att.size = size
    att.title = title
    att.description = description
    return att


def _make_message(author=None, content="", attachments=None, reference=None,
                  msg_id=123456789, embeds=None):
    msg = MagicMock()
    msg.id = msg_id
    msg.author = author or _make_author()
    msg.guild = _make_guild()
    msg.channel.id = 555
    msg.content = content
    msg.attachments = attachments or []
    msg.embeds = embeds or []
    msg.reference = reference
    msg.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return msg


def _make_converter(scan_enabled=True):
    cog = MagicMock()
    cog.cached_messages = {}
    config = MagicMock()
    config.guild.return_value.max_image_size = AsyncMock(return_value=10_000_000)
    cog.config = config
    conv = converter_mod.MessageConverter(cog, MagicMock())
    conv._is_scan_enabled = AsyncMock(return_value=scan_enabled)
    return conv


# ---------------------------------------------------------------------------
# C2: None content_type must not crash
# ---------------------------------------------------------------------------

class TestNoneContentType:
    @pytest.mark.asyncio
    async def test_none_content_type_falls_to_generic_file(self):
        conv = _make_converter()
        msg = _make_message(attachments=[
            _make_attachment(filename="mystery.bin", content_type=None)])
        res = []
        await conv.handle_attachment(msg, res, "user")
        assert res, "no entries produced"
        out = res[0].content
        assert '<file filename="mystery.bin"/>' in out

    def test_format_attachment_marker_none_content_type(self):
        att = _make_attachment(filename="x.dat", content_type=None)
        assert converter_mod.format_attachment_marker(att) == '<file filename="x.dat"/>'

    def test_format_attachment_marker_image(self):
        att = _make_attachment(filename="y.png", content_type="image/png")
        assert converter_mod.format_attachment_marker(att) == '<image filename="y.png"/>'


# ---------------------------------------------------------------------------
# C3: quote escaping in document filenames
# ---------------------------------------------------------------------------

class TestFilenameQuoteEscaping:
    def test_generic_document_escapes_quotes(self):
        msg = _make_message(attachments=[
            _make_attachment(filename='my "report".txt', content_type="text/plain")])
        out = helpers_mod.format_generic_document(msg)
        q = "&" + "quot;"
        assert f'filename="my {q}report{q}.txt"' in out

    @pytest.mark.asyncio
    async def test_text_document_escapes_quotes(self):
        from io import BytesIO
        att = _make_attachment(filename='evil "name".txt', content_type="text/plain")

        async def fake_save(buffer):
            buffer.write(b"hello world")

        att.save = fake_save
        msg = _make_message(attachments=[att])
        parts = await helpers_mod.format_text_document(msg)
        text = parts[-1]["text"]
        q = "&" + "quot;"
        assert f'filename="evil {q}name{q}.txt"' in text
        assert 'filename="evil "name".txt"' not in text


# ---------------------------------------------------------------------------
# Cov5: all attachments get markers
# ---------------------------------------------------------------------------

class TestAllAttachmentsMarked:
    def test_format_extra_attachments_none_for_single(self):
        msg = _make_message(attachments=[_make_attachment()])
        assert converter_mod.format_extra_attachments(msg) is None

    def test_format_extra_attachments_user_message(self):
        msg = _make_message(attachments=[
            _make_attachment(filename="a.png", content_type="image/png"),
            _make_attachment(filename="b.pdf", content_type="application/pdf"),
            _make_attachment(filename="c.png", content_type="image/png"),
        ])
        out = converter_mod.format_extra_attachments(msg)
        assert out.startswith("<message ")
        assert out.endswith("</message>")
        assert '<file filename="b.pdf"/>' in out
        assert '<image filename="c.png"/>' in out
        # first attachment not duplicated
        assert '<image filename="a.png"/>' not in out

    def test_format_extra_attachments_bot_message(self):
        msg = _make_message(author=_make_author(author_id=999), attachments=[
            _make_attachment(filename="a.png", content_type="image/png"),
            _make_attachment(filename="b.zip", content_type="application/zip"),
        ])
        out = converter_mod.format_extra_attachments(msg)
        assert out == 'Sent <file filename="b.zip"/>'

    @pytest.mark.asyncio
    async def test_multi_image_branch_marks_non_image_attachments(self, monkeypatch):
        conv = _make_converter(scan_enabled=True)
        img_part = {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}}
        monkeypatch.setattr(converter_mod, "transcribe_image_single",
                            AsyncMock(return_value=[img_part]))

        msg = _make_message(attachments=[
            _make_attachment(filename="one.png", content_type="image/png"),
            _make_attachment(filename="two.png", content_type="image/png"),
            _make_attachment(filename="notes.pdf", content_type="application/pdf"),
        ])
        res = []
        await conv.handle_attachment(msg, res, "user")

        assert len(res) == 1
        texts = [p["text"] for p in res[0].content if p.get("type") == "text"]
        assert any('<file filename="notes.pdf"/>' in t for t in texts), texts

    @pytest.mark.asyncio
    async def test_fallthrough_marks_additional_attachments(self):
        # single non-image attachment + a second one -> second gets a marker entry
        conv = _make_converter(scan_enabled=False)
        att1 = _make_attachment(filename="doc.pdf", content_type="application/pdf", size=100)
        att2 = _make_attachment(filename="extra.png", content_type="image/png")
        msg = _make_message(attachments=[att1, att2])
        res = []
        await conv.handle_attachment(msg, res, "user")

        all_content = " ".join(
            e.content for e in res if isinstance(e.content, str))
        assert '<image filename="extra.png"/>' in all_content


# ---------------------------------------------------------------------------
# Embed text preserved when a message also has attachments
# ---------------------------------------------------------------------------

class TestEmbedPreservedWithAttachments:
    def _make_embed(self, title="Some Bot", description="a generated cat"):
        embed = MagicMock()
        embed.title = title
        embed.description = description
        return embed

    @pytest.mark.asyncio
    async def test_fallthrough_appends_embed_content(self, monkeypatch):
        conv = _make_converter(scan_enabled=False)
        monkeypatch.setattr(converter_mod, "is_embed_valid", lambda m: True)
        fake_format = AsyncMock(return_value="<message ...><embed title=\"Some Bot\">a generated cat</embed></message>")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_format)

        other_bot = _make_author(author_id=555, name="otherbot", display="OtherBot")
        msg = _make_message(
            author=other_bot,
            attachments=[_make_attachment(filename="cat.png", content_type="image/png")],
            embeds=[self._make_embed()],
        )
        res = []
        await conv.handle_attachment(msg, res, "user")

        fake_format.assert_awaited_once()
        all_content = " ".join(e.content for e in res if isinstance(e.content, str))
        assert "a generated cat</embed>" in all_content

    @pytest.mark.asyncio
    async def test_multi_image_branch_appends_embed_part(self, monkeypatch):
        conv = _make_converter(scan_enabled=True)
        monkeypatch.setattr(converter_mod, "is_embed_valid", lambda m: True)
        fake_format = AsyncMock(return_value="<message ...><embed title=\"Some Bot\">a generated cat</embed></message>")
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_format)
        img_part = {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}}
        monkeypatch.setattr(converter_mod, "transcribe_image_single",
                            AsyncMock(return_value=[img_part]))

        msg = _make_message(
            attachments=[
                _make_attachment(filename="one.png", content_type="image/png"),
                _make_attachment(filename="two.png", content_type="image/png"),
            ],
            embeds=[self._make_embed()],
        )
        res = []
        await conv.handle_attachment(msg, res, "user")

        fake_format.assert_awaited_once()
        texts = [p["text"] for p in res[0].content if p.get("type") == "text"]
        assert any("a generated cat</embed>" in t for t in texts), texts

    @pytest.mark.asyncio
    async def test_no_embed_means_no_extra_entry(self, monkeypatch):
        conv = _make_converter(scan_enabled=False)
        monkeypatch.setattr(converter_mod, "is_embed_valid", lambda m: True)
        fake_format = AsyncMock()
        monkeypatch.setattr(converter_mod, "format_embed_content", fake_format)

        msg = _make_message(attachments=[
            _make_attachment(filename="x.zip", content_type="application/zip")])
        res = []
        await conv.handle_attachment(msg, res, "user")

        fake_format.assert_not_called()


# ---------------------------------------------------------------------------
# Cov6: image-URL handling must not drop attachments
# ---------------------------------------------------------------------------

class TestImageUrlsDontDropAttachments:
    @pytest.mark.asyncio
    async def test_convert_with_url_and_attachment_uses_attachment_path(self):
        conv = _make_converter(scan_enabled=False)
        # handle_image_urls would claim to handle it; ensure it's never consulted
        conv.handle_image_urls = AsyncMock(return_value=True)
        msg = _make_message(
            content="look https://example.com/x.png",
            attachments=[_make_attachment(filename="real.png", content_type="image/png")],
        )
        res = await conv.convert(msg)
        conv.handle_image_urls.assert_not_called()
        assert res, "no entries produced"
        all_content = " ".join(
            e.content for e in res if isinstance(e.content, str))
        assert '<image filename="real.png"/>' in all_content


# ---------------------------------------------------------------------------
# Cov8: system prompt appendix documents the full schema
# ---------------------------------------------------------------------------

class TestSystemPromptAppendix:
    def test_documents_document_tag(self):
        assert "<document" in constants_mod.XML_SYSTEM_PROMPT_APPENDIX

    def test_documents_sent_convention(self):
        assert "Sent" in constants_mod.XML_SYSTEM_PROMPT_APPENDIX

    def test_documents_reply_attributes(self):
        assert "reply_to" in constants_mod.XML_SYSTEM_PROMPT_APPENDIX

    def test_placeholder_means_not_provided(self):
        # the model must be told not to hallucinate placeholder media
        assert "NOT provided" in constants_mod.XML_SYSTEM_PROMPT_APPENDIX


# ---------------------------------------------------------------------------
# C1: bot text included when caching generated images
# ---------------------------------------------------------------------------

class TestCacheGeneratedImagesIncludesBotText:
    def _run_cache(self, sent_message, images):
        ctx = MagicMock()
        ctx.channel.id = 555
        cog = MagicMock()
        cog.cached_messages = {}
        response_mod._cache_generated_images_in_response(ctx, cog, sent_message, images)
        return cog.cached_messages

    def test_bot_text_cached_with_image(self):
        sent = MagicMock()
        sent.id = 777
        sent.content = "Here is the image you asked for!"
        sent.embeds = []
        images = [{"bytes": None, "data_url": "data:image/webp;base64,AAAA"}]

        cache = self._run_cache(sent, images)
        parts = cache[777]
        texts = [p["text"] for p in parts if p.get("type") == "text"]
        imgs = [p for p in parts if p.get("type") == "image_url"]
        assert len(imgs) == 1
        assert "Here is the image you asked for!" in texts

    def test_empty_bot_text_not_added(self):
        sent = MagicMock()
        sent.id = 778
        sent.content = ""
        sent.embeds = []
        images = [{"bytes": None, "data_url": "data:image/webp;base64,AAAA"}]

        cache = self._run_cache(sent, images)
        parts = cache[778]
        texts = [p for p in parts if p.get("type") == "text"]
        assert texts == []

    def test_channel_scoped_key_also_set(self):
        sent = MagicMock()
        sent.id = 779
        sent.content = "caption"
        sent.embeds = []
        images = [{"bytes": None, "data_url": "data:image/webp;base64,AAAA"}]

        cache = self._run_cache(sent, images)
        assert "555:779" in cache
        assert 779 in cache
