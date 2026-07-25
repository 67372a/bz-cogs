"""Tests for image attachment metadata inclusion.

Validates the fix for the gap where a user attaches an image without any text
in a reply, and the XML metadata (author, timestamp, reply info) was not being
included in the LLM content parts.

The fix ensures _process_attachment always includes the <message> XML header
when in LLM mode, even when message.content is empty.
"""

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing real modules
# ---------------------------------------------------------------------------

# Create a real discord module with Message as a proper class
discord_mock = types.ModuleType("discord")
discord_mock.__package__ = "discord"
discord_mock.__path__ = []


class _DiscordMessage:
    """Real class to stand in for discord.Message in isinstance checks."""
    pass


discord_mock.Message = _DiscordMessage
discord_mock.MessageType = MagicMock()
discord_mock.MessageType.new_member = 7
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
sys.modules["discord"] = discord_mock
sys.modules.setdefault("discord.ext", types.ModuleType("discord.ext"))
sys.modules.setdefault("discord.ext.commands", MagicMock())
sys.modules.setdefault("discord.app_commands", MagicMock())

# Mock redbot
_redbot_core = types.ModuleType("redbot.core")
_redbot_core.__package__ = "redbot.core"
_redbot_core.__path__ = []
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
_redbot_core.app_commands = MagicMock()
sys.modules.setdefault("redbot", types.ModuleType("redbot"))
sys.modules.setdefault("redbot.core", _redbot_core)
for _n in ["redbot.core.commands", "redbot.core.bot", "redbot.core.utils",
           "redbot.core.utils.chat_formatting", "redbot.core.utils.views"]:
    sys.modules.setdefault(_n, MagicMock())

# Mock other deps
sys.modules.setdefault("tiktoken", MagicMock())
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

# Create mock packages for aiuser sub-packages
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
    "aiuser.functions.openrouter",
]
for _pkg in _sub_pkgs:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Mock leaf modules
_leaf_mocks = [
    "aiuser.config.defaults",
    "aiuser.messages_list.opt_view",
    "aiuser.messages_list.converter.embed.formatter",
    "aiuser.messages_list.converter.embed.youtube",
    "aiuser.messages_list.converter.image.AI_horde",
    "aiuser.messages_list.converter.image.local",
    "aiuser.utils.utilities",
]
for _m in _leaf_mocks:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Load real modules (order matters: dependencies first)
sys.modules["aiuser.config.constants"] = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
sys.modules["aiuser.config.models"] = import_module_directly(
    "aiuser.config.models", "aiuser/config/models.py"
)
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
sys.modules["aiuser.utils.cache"] = import_module_directly(
    "aiuser.utils.cache", "aiuser/utils/cache.py"
)
sys.modules["aiuser.utils.image_cache"] = import_module_directly(
    "aiuser.utils.image_cache", "aiuser/utils/image_cache.py"
)
sys.modules["aiuser.utils.image_processing"] = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
sys.modules["aiuser.types.enums"] = import_module_directly(
    "aiuser.types.enums", "aiuser/types/enums.py"
)
sys.modules["aiuser.types.abc"] = import_module_directly(
    "aiuser.types.abc", "aiuser/types/abc.py"
)

# Load the modules under test
helpers_mod = import_module_directly(
    "aiuser.messages_list.converter.helpers", "aiuser/messages_list/converter/helpers.py"
)
format_text_content = helpers_mod.format_text_content
_get_msg_header = helpers_mod._get_msg_header

caption_mod = import_module_directly(
    "aiuser.messages_list.converter.image.caption", "aiuser/messages_list/converter/image/caption.py"
)
_process_attachment = caption_mod._process_attachment
ScanImageMode = sys.modules["aiuser.types.enums"].ScanImageMode

converter_mod = import_module_directly(
    "aiuser.messages_list.converter.converter", "aiuser/messages_list/converter/converter.py"
)
MessageConverter = converter_mod.MessageConverter
MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(content="", author_id=111, bot_id=789, has_reference=False, is_bot=False):
    """Create a mock discord.Message with realistic attributes."""
    msg = _DiscordMessage()
    msg.id = 12345
    msg.content = content
    msg.type = MagicMock()
    msg.type.value = 0  # default message type

    # Author
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.name = "testuser"
    msg.author.display_name = "Test User"
    msg.author.bot = is_bot

    # Guild
    msg.guild = MagicMock()
    msg.guild.id = 456
    msg.guild.me = MagicMock()
    msg.guild.me.id = bot_id

    # Timestamp
    msg.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Reply reference
    if has_reference:
        resolved = _DiscordMessage()
        resolved.id = 99999
        resolved.author = MagicMock()
        resolved.author.id = 222
        resolved.author.name = "replyuser"
        resolved.author.display_name = "Reply User"
        msg.reference = MagicMock()
        msg.reference.message_id = 99999
        msg.reference.resolved = resolved
    else:
        msg.reference = None

    # Attachments
    msg.attachments = []
    msg.embeds = []
    msg.stickers = []

    # Channel
    msg.channel = MagicMock()
    msg.channel.id = 789

    # Mentions (needed by mention_to_text in format_text_content)
    msg.mentions = []
    msg.role_mentions = []
    msg.channel_mentions = []

    return msg


def _make_attachment(filename="test.png", size=1024, content_type="image/png"):
    """Create a mock discord attachment."""
    att = MagicMock()
    att.filename = filename
    att.size = size
    att.content_type = content_type
    att.title = None
    att.description = None
    att.url = f"https://cdn.discord.com/attachments/123/456/{filename}"
    return att


# ===========================================================================
# TEST: _process_attachment metadata in LLM mode
# ===========================================================================

class TestProcessAttachmentMetadata:
    """Test that _process_attachment always includes message metadata in LLM mode."""

    @pytest.mark.asyncio
    @patch.object(caption_mod, "processed_image_cache")
    @patch.object(caption_mod, "image_cache")
    @patch.object(caption_mod, "process_image_for_llm")
    @patch.object(caption_mod, "build_webp_data_url")
    @patch.object(caption_mod, "compute_pixel_hash")
    async def test_no_text_includes_metadata(
        self, mock_pixel_hash, mock_build_url, mock_process, mock_image_cache, mock_processed_cache
    ):
        """When message has no text, XML metadata header should still be included."""
        # Setup mocks
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_image_cache.get.return_value = raw_bytes
        mock_pixel_hash.return_value = "abc123"
        mock_processed_cache.get.return_value = None
        mock_process.return_value = b"processed_webp_bytes"
        mock_build_url.return_value = "data:image/webp;base64,dGVzdA=="

        cog = MagicMock()
        message = _make_message(content="")  # No text content
        attachment = _make_attachment()

        result = await _process_attachment(
            cog, message, attachment, ScanImageMode.LLM, 16_777_216
        )

        assert result is not None
        assert isinstance(result, list)

        # Should have image_url part
        image_parts = [p for p in result if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"] == "data:image/webp;base64,dGVzdA=="

        # Should have text part with metadata even though there's no user text
        text_parts = [p for p in result if p.get("type") == "text"]
        assert len(text_parts) == 1
        text_content = text_parts[0]["text"]
        assert '<message' in text_content
        assert 'author_id="111"' in text_content
        assert 'username="testuser"' in text_content
        assert 'displayname="Test User"' in text_content
        assert text_content.endswith("</message>")

    @pytest.mark.asyncio
    @patch.object(caption_mod, "processed_image_cache")
    @patch.object(caption_mod, "image_cache")
    @patch.object(caption_mod, "process_image_for_llm")
    @patch.object(caption_mod, "build_webp_data_url")
    @patch.object(caption_mod, "compute_pixel_hash")
    async def test_with_text_includes_metadata_and_text(
        self, mock_pixel_hash, mock_build_url, mock_process, mock_image_cache, mock_processed_cache
    ):
        """When message has text, both the text content and metadata header should be included."""
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_image_cache.get.return_value = raw_bytes
        mock_pixel_hash.return_value = "abc123"
        mock_processed_cache.get.return_value = None
        mock_process.return_value = b"processed_webp_bytes"
        mock_build_url.return_value = "data:image/webp;base64,dGVzdA=="

        cog = MagicMock()
        message = _make_message(content="Look at this image!")
        attachment = _make_attachment()

        result = await _process_attachment(
            cog, message, attachment, ScanImageMode.LLM, 16_777_216
        )

        assert result is not None
        assert isinstance(result, list)

        # Should have image_url part
        image_parts = [p for p in result if p.get("type") == "image_url"]
        assert len(image_parts) == 1

        # Should have text part with full message and metadata
        text_parts = [p for p in result if p.get("type") == "text"]
        assert len(text_parts) == 1
        text_content = text_parts[0]["text"]
        assert "Look at this image!" in text_content
        assert '<message' in text_content
        assert 'author_id="111"' in text_content
        assert text_content.endswith("</message>")

    @pytest.mark.asyncio
    @patch.object(caption_mod, "processed_image_cache")
    @patch.object(caption_mod, "image_cache")
    @patch.object(caption_mod, "process_image_for_llm")
    @patch.object(caption_mod, "build_webp_data_url")
    @patch.object(caption_mod, "compute_pixel_hash")
    async def test_no_text_with_reply_includes_reply_metadata(
        self, mock_pixel_hash, mock_build_url, mock_process, mock_image_cache, mock_processed_cache
    ):
        """When user replies with an image and no text, reply context metadata should be present."""
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_image_cache.get.return_value = raw_bytes
        mock_pixel_hash.return_value = "abc123"
        mock_processed_cache.get.return_value = None
        mock_process.return_value = b"processed_webp_bytes"
        mock_build_url.return_value = "data:image/webp;base64,dGVzdA=="

        cog = MagicMock()
        message = _make_message(content="", has_reference=True)  # Reply with no text
        attachment = _make_attachment()

        result = await _process_attachment(
            cog, message, attachment, ScanImageMode.LLM, 16_777_216
        )

        assert result is not None
        text_parts = [p for p in result if p.get("type") == "text"]
        assert len(text_parts) == 1
        text_content = text_parts[0]["text"]

        # Should include reply metadata
        assert 'reply_to_id="99999"' in text_content
        assert 'reply_target_id="222"' in text_content
        assert 'reply_target_username="replyuser"' in text_content
        assert 'reply_target_displayname="Reply User"' in text_content

    @pytest.mark.asyncio
    @patch.object(caption_mod, "processed_image_cache")
    @patch.object(caption_mod, "image_cache")
    @patch.object(caption_mod, "process_image_for_llm")
    @patch.object(caption_mod, "build_webp_data_url")
    @patch.object(caption_mod, "compute_pixel_hash")
    async def test_no_text_bot_message_gets_provenance(
        self, mock_pixel_hash, mock_build_url, mock_process, mock_image_cache, mock_processed_cache
    ):
        """Bot's own image messages with no text get a "Sent image" provenance part."""
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_image_cache.get.return_value = raw_bytes
        mock_pixel_hash.return_value = "abc123"
        mock_processed_cache.get.return_value = None
        mock_process.return_value = b"processed_webp_bytes"
        mock_build_url.return_value = "data:image/webp;base64,dGVzdA=="

        bot_id = 789
        cog = MagicMock()
        # Message author IS the bot
        message = _make_message(content="", author_id=bot_id, bot_id=bot_id)
        attachment = _make_attachment()

        result = await _process_attachment(
            cog, message, attachment, ScanImageMode.LLM, 16_777_216
        )

        assert result is not None
        image_parts = [p for p in result if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        text_parts = [p for p in result if p.get("type") == "text"]
        assert text_parts == [{"type": "text", "text": "Sent image"}]


# ===========================================================================
# TEST: handle_attachment multi-image path metadata
# ===========================================================================

class TestHandleAttachmentMultiImageMetadata:
    """Test that handle_attachment includes metadata in the multi-image path."""

    def _make_converter(self):
        """Create a minimal MessageConverter for testing."""
        converter = MessageConverter.__new__(MessageConverter)
        converter.cog = MagicMock()
        converter.config = MagicMock()
        converter.bot_id = 789
        converter.message_cache = {}
        converter.ctx = MagicMock()
        converter.ctx.interaction = None
        converter.init_msg = MagicMock()
        converter.init_msg.id = 999
        converter.init_msg.reference = None
        return converter

    @pytest.mark.asyncio
    @patch.object(converter_mod, "transcribe_image_single", new_callable=AsyncMock)
    async def test_multi_image_no_text_includes_metadata(self, mock_transcribe):
        """Multi-image message with no text should include message metadata."""
        converter = self._make_converter()

        # Setup message with two image attachments and no text
        msg = _make_message(content="")
        att1 = _make_attachment("image1.png")
        att2 = _make_attachment("image2.png")
        msg.attachments = [att1, att2]

        # Make this message the trigger so _is_scan_enabled returns True
        converter.init_msg = msg
        converter.config.guild.return_value.scan_images = AsyncMock(return_value=True)
        converter.config.guild.return_value.max_image_size = AsyncMock(return_value=10_000_000)

        # transcribe_image_single returns image_url content parts (simulating LLM mode)
        mock_transcribe.return_value = [
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,abc"}},
        ]

        res = []
        await converter.handle_attachment(msg, res, "user")

        assert len(res) == 1
        entry = res[0]
        assert isinstance(entry, MessageEntry)

        # Should have image_url parts for both images
        image_parts = [p for p in entry.content if isinstance(p, dict) and p.get("type") == "image_url"]
        assert len(image_parts) == 2

        # Should have text part with metadata
        text_parts = [p for p in entry.content if isinstance(p, dict) and p.get("type") == "text"]
        assert len(text_parts) >= 1

        # At least one text part should contain the message metadata
        metadata_found = any(
            '<message' in p.get("text", "") and 'author_id="111"' in p.get("text", "")
            for p in text_parts
        )
        assert metadata_found, "Expected message metadata in text parts for image-only message"

    @pytest.mark.asyncio
    @patch.object(converter_mod, "transcribe_image_single", new_callable=AsyncMock)
    async def test_multi_image_with_text_no_duplicate_metadata_header(self, mock_transcribe):
        """Multi-image message WITH text should still include text content."""
        converter = self._make_converter()

        msg = _make_message(content="Check these out!")
        att1 = _make_attachment("image1.png")
        att2 = _make_attachment("image2.png")
        msg.attachments = [att1, att2]

        converter.init_msg = msg
        converter.config.guild.return_value.scan_images = AsyncMock(return_value=True)
        converter.config.guild.return_value.max_image_size = AsyncMock(return_value=10_000_000)

        mock_transcribe.return_value = [
            {"type": "image_url", "image_url": {"url": "data:image/webp;base64,abc"}},
        ]

        res = []
        await converter.handle_attachment(msg, res, "user")

        assert len(res) == 1
        entry = res[0]

        # Should have text parts containing the user's message
        text_parts = [p for p in entry.content if isinstance(p, dict) and p.get("type") == "text"]
        assert len(text_parts) >= 1
        text_combined = " ".join(p.get("text", "") for p in text_parts)
        assert "Check these out!" in text_combined


# ===========================================================================
# TEST: format_generic_image helper always includes metadata
# ===========================================================================

class TestFormatGenericImage:
    """Verify format_generic_image behavior as baseline reference."""

    def test_format_generic_image_includes_metadata_for_user(self):
        """format_generic_image always wraps with message header for user messages."""
        msg = _make_message(content="")
        att = _make_attachment("photo.jpg")
        att.title = None
        att.description = None
        msg.attachments = [att]

        result = helpers_mod.format_generic_image(msg)

        assert '<message' in result
        assert 'author_id="111"' in result
        assert '<image filename="photo.jpg"/>' in result
        assert result.endswith("</message>")

    def test_format_generic_image_includes_reply_metadata(self):
        """format_generic_image includes reply metadata when message is a reply."""
        msg = _make_message(content="", has_reference=True)
        att = _make_attachment("photo.jpg")
        msg.attachments = [att]

        result = helpers_mod.format_generic_image(msg)

        assert 'reply_to_id="99999"' in result
        assert 'reply_target_id="222"' in result
