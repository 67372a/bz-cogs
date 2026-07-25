"""Tests for XML media context improvements.

Covers:
- Unsupported-attachment fallback now uses the XML schema
- Multi-image messages interleave <image filename/> markers with image parts
- Gemini tiling token estimation and file-part token estimation
"""

import base64
import sys
import types
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure real PIL/numpy/cv2 (other test files may have mocked them)
# ---------------------------------------------------------------------------
for _m in list(sys.modules):
    if _m == "PIL" or _m.startswith("PIL.") or _m == "cv2" or _m == "numpy" or _m.startswith("numpy."):
        del sys.modules[_m]

from PIL import Image  # noqa: E402  real PIL

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
# Attributes other test files expect on the shared discord mock
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
             "aiuser.config", "aiuser.types", "aiuser.utils"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

for _m in ["aiuser.messages_list.converter.embed.formatter",
           "aiuser.messages_list.converter.embed.youtube",
           "aiuser.messages_list.converter.image.AI_horde",
           "aiuser.messages_list.converter.image.local",
           "aiuser.utils.utilities",
           "aiuser.utils.image_cache",
           "aiuser.types.abc"]:
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
sys.modules["aiuser.utils.image_processing"] = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
sys.modules["aiuser.messages_list.converter.helpers"] = import_module_directly(
    "aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py"
)
# caption.py imports image_cache / AI_horde (mocked above); provide a stub so
# converter.py's `from ...caption import transcribe_image` works.
_caption_stub = types.ModuleType("aiuser.messages_list.converter.image.caption")
_caption_stub.transcribe_image = AsyncMock()
_caption_stub.transcribe_single = AsyncMock()
sys.modules["aiuser.messages_list.converter.image.caption"] = _caption_stub
sys.modules["aiuser.messages_list.converter.converter"] = import_module_directly(
    "aiuser.messages_list.converter.converter", "aiuser/messages_list/converter/converter.py"
)

from aiuser.messages_list.converter import converter as converter_mod  # noqa: E402
from aiuser.utils import image_processing as ip  # noqa: E402


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


def _make_message(author=None, content="", attachments=None, reference=None, embeds=None):
    msg = MagicMock()
    msg.id = 123456789
    msg.author = author or _make_author()
    msg.guild = _make_guild()
    msg.channel.id = 555
    msg.content = content
    msg.attachments = attachments or []
    msg.embeds = embeds or []
    msg.reference = reference
    msg.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return msg


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestEstimateImageTokens:
    def test_small_image_flat_cost(self):
        assert ip.estimate_image_tokens(384, 384) == 258
        assert ip.estimate_image_tokens(100, 200) == 258

    def test_gemini_example_960x540(self):
        # crop unit = floor(540/1.5) = 360 -> ceil(960/360)=3, ceil(540/360)=2 -> 6 tiles
        assert ip.estimate_image_tokens(960, 540) == 6 * 258

    def test_invalid_dimensions_fallback(self):
        assert ip.estimate_image_tokens(0, 100) == ip.FALLBACK_IMAGE_TOKENS
        assert ip.estimate_image_tokens(-5, -5) == ip.FALLBACK_IMAGE_TOKENS

    def test_large_image_tiling(self):
        # crop unit = floor(4096/1.5) = 2730 -> ceil(4096/2730)=2 -> 2x2 = 4 tiles
        assert ip.estimate_image_tokens(4096, 4096) == 4 * 258


class TestImagePartTokens:
    def _png_data_url(self, w, h):
        buf = BytesIO()
        Image.new("RGB", (w, h), (255, 0, 0)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def test_dimensions_from_data_url(self):
        url = self._png_data_url(960, 540)
        assert ip.image_dimensions_from_data_url(url) == (960, 540)

    def test_estimate_from_data_url(self):
        url = self._png_data_url(960, 540)
        assert ip.estimate_image_part_tokens(url) == 6 * 258

    def test_bad_data_url_fallback(self):
        assert ip.estimate_image_part_tokens("data:image/png;base64,!!!invalid!!!") == ip.FALLBACK_IMAGE_TOKENS
        assert ip.estimate_image_part_tokens("https://example.com/img.png") == ip.FALLBACK_IMAGE_TOKENS


class TestFilePartTokens:
    def test_scales_with_size(self):
        small = "data:application/pdf;base64," + "A" * 1600
        large = "data:application/pdf;base64," + "A" * 16000
        assert ip.estimate_file_part_tokens(large) == 10 * ip.estimate_file_part_tokens(small)

    def test_empty(self):
        assert ip.estimate_file_part_tokens("") == 0

    def test_minimum_one(self):
        assert ip.estimate_file_part_tokens("data:application/pdf;base64,AA") >= 1


# ---------------------------------------------------------------------------
# XML fallback for unsupported attachments
# ---------------------------------------------------------------------------

class TestGenericAttachmentXml:
    def test_user_message_uses_xml_schema(self):
        msg = _make_message(attachments=[_make_attachment(
            filename="archive.zip", content_type="application/zip",
            title='My "Archive"', description="backup <v2>")])
        out = converter_mod.format_generic_attachment(msg)
        assert out.startswith("<message ")
        assert 'id="123456789"' in out
        assert 'username="testuser"' in out
        assert out.endswith("</message>")
        assert "<file filename=\"archive.zip\"" in out
        # attributes escaped
        q = "&" + "quot;"
        lt = "&" + "lt;"
        gt = "&" + "gt;"
        assert f'title="My {q}Archive{q}"' in out
        assert f'description="backup {lt}v2{gt}"' in out
        # old bracket format gone
        assert "MESSAGE_ID=" not in out

    def test_bot_message_sent_prefix(self):
        bot_author = _make_author(author_id=999, name="bot", display="Bot")
        msg = _make_message(author=bot_author,
                            attachments=[_make_attachment(filename="a.bin", content_type="application/octet-stream")])
        out = converter_mod.format_generic_attachment(msg)
        assert out == 'Sent <file filename="a.bin"/>'

    def test_no_title_desc_omitted(self):
        msg = _make_message(attachments=[_make_attachment(filename="x.bin")])
        out = converter_mod.format_generic_attachment(msg)
        assert "title=" not in out
        assert "description=" not in out


# ---------------------------------------------------------------------------
# Multi-image interleaving of filename markers
# ---------------------------------------------------------------------------

class TestMultiImageInterleave:
    @pytest.mark.asyncio
    async def test_markers_interleaved_and_text_deduped(self, monkeypatch):
        cog = MagicMock()
        cog.cached_messages = {}
        config = MagicMock()
        config.guild.return_value.max_image_size = AsyncMock(return_value=10_000_000)
        cog.config = config

        conv = converter_mod.MessageConverter(cog, MagicMock())
        conv._is_scan_enabled = AsyncMock(return_value=True)

        img_part_1 = {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}}
        img_part_2 = {"type": "image_url", "image_url": {"url": "data:image/webp;base64,BBBB"}}
        # transcription returns image + a text header part that must be stripped
        returns = [
            [img_part_1, {"type": "text", "text": "<message ...>dup</message>"}],
            [img_part_2],
        ]
        mock_transcribe = AsyncMock(side_effect=returns)
        monkeypatch.setattr(converter_mod, "transcribe_image_single", mock_transcribe)

        msg = _make_message(attachments=[
            _make_attachment(filename="one.png"),
            _make_attachment(filename="two.png"),
        ])

        res = []
        await conv.handle_attachment(msg, res, "user")

        assert len(res) == 1
        parts = res[0].content
        texts = [p["text"] for p in parts if p.get("type") == "text"]
        images = [p for p in parts if p.get("type") == "image_url"]

        # markers interleaved: text, image, text, image, header
        assert parts[0] == {"type": "text", "text": '<image filename="one.png"/>'}
        assert parts[1] is img_part_1
        assert parts[2] == {"type": "text", "text": '<image filename="two.png"/>'}
        assert parts[3] is img_part_2
        # duplicated transcription text stripped; single trailing header added
        assert "<message ...>dup</message>" not in texts
        assert texts[-1].startswith("<message ") and texts[-1].endswith("</message>")
        assert len(images) == 2

    @pytest.mark.asyncio
    async def test_oversize_image_gets_marker_only(self, monkeypatch):
        cog = MagicMock()
        cog.cached_messages = {}
        config = MagicMock()
        config.guild.return_value.max_image_size = AsyncMock(return_value=10)
        cog.config = config

        conv = converter_mod.MessageConverter(cog, MagicMock())
        conv._is_scan_enabled = AsyncMock(return_value=True)
        mock_transcribe = AsyncMock()
        monkeypatch.setattr(converter_mod, "transcribe_image_single", mock_transcribe)

        msg = _make_message(attachments=[_make_attachment(filename="big.png", size=9999)])
        res = []
        await conv.handle_attachment(msg, res, "user")

        mock_transcribe.assert_not_called()
        parts = res[0].content
        assert parts[0] == {"type": "text", "text": '<image filename="big.png"/>'}
