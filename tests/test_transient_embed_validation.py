"""Tests for transient embed validation in core/validators.py.

Covers:
- check_message_content rejects messages that ARE transient embeds (from any bot)
- is_reply_to_transient_embed rejects replies to both function-call AND thoughts embeds
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing modules that depend on
# discord / redbot (not installed in the test environment)
# ---------------------------------------------------------------------------

if "discord" not in sys.modules:
    _discord_mod = types.ModuleType("discord")
    _discord_mod.Message = type("_DiscordMessage", (), {})
    _discord_mod.Thread = type("_DiscordThread", (), {})
    _discord_mod.Interaction = type("_DiscordInteraction", (), {})
    _discord_mod.Guild = type("_DiscordGuild", (), {})
    _discord_mod.NotFound = type("NotFound", (Exception,), {})
    _discord_mod.Forbidden = type("Forbidden", (Exception,), {})
    _discord_mod.HTTPException = type("HTTPException", (Exception,), {})
    sys.modules["discord"] = _discord_mod

# Ensure all needed discord attrs exist even if discord was already partially loaded
_discord_attrs = {
    "Message": type("_DiscordMessage", (), {}),
    "Thread": type("_DiscordThread", (), {}),
    "Interaction": type("_DiscordInteraction", (), {}),
    "Guild": type("_DiscordGuild", (), {}),
    "NotFound": type("NotFound", (Exception,), {}),
    "Forbidden": type("Forbidden", (Exception,), {}),
    "HTTPException": type("HTTPException", (Exception,), {}),
}
for _name, _val in _discord_attrs.items():
    if not hasattr(sys.modules["discord"], _name):
        setattr(sys.modules["discord"], _name, _val)

if "redbot.core" not in sys.modules:
    _redbot = _make_mock_package("redbot")
    _redbot_core = _make_mock_package("redbot.core")
    _redbot_core.Config = MagicMock()
    _redbot_core.commands = MagicMock()
    sys.modules["redbot"] = _redbot
    sys.modules["redbot.core"] = _redbot_core

# Parent packages must exist before any direct imports
for _pkg in [
    "aiuser", "aiuser.config", "aiuser.core", "aiuser.types",
    "aiuser.utils", "aiuser.messages_list",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Mock leaf modules that validators.py imports but we don't need real versions of
_leaf_mocks = [
    "aiuser.types.abc",
    "aiuser.core.openai_utils",
    "aiuser.utils.utilities",
]
for _leaf in _leaf_mocks:
    if _leaf not in sys.modules:
        sys.modules[_leaf] = MagicMock()

# config.constants / config.defaults need real values
sys.modules.pop("aiuser.config.constants", None)
sys.modules.pop("aiuser.config.defaults", None)
sys.modules.pop("aiuser.types.enums", None)
sys.modules.pop("aiuser.messages_list.messages", None)
sys.modules.pop("aiuser.core.validators", None)

import_module_directly("aiuser.types.enums", "aiuser/types/enums.py")
import_module_directly("aiuser.config.constants", "aiuser/config/constants.py")
import_module_directly("aiuser.config.defaults", "aiuser/config/defaults.py")

# Create a lightweight mock of aiuser.messages_list.messages with the real
# regex patterns and is_function_call_or_thoughts_embed so validators.py
# gets the actual matching logic without pulling in all of messages.py's
# heavy dependencies (tiktoken, converter, etc.).
import re

_messages_mock = types.ModuleType("aiuser.messages_list.messages")
_messages_mock.FUNCTION_CALL_EMBED_TITLE_REGEX = re.compile(
    r"^.*is making the following function calls\.\.\.$"
)
_messages_mock.THOUGHTS_EMBED_TITLE_REGEX = re.compile(r"^.*'s Thoughts$")


def _is_function_call_or_thoughts_embed(message):
    if not message.embeds:
        return False
    return any(
        embed.title and (
            _messages_mock.THOUGHTS_EMBED_TITLE_REGEX.search(embed.title)
            or _messages_mock.FUNCTION_CALL_EMBED_TITLE_REGEX.search(embed.title)
        )
        for embed in message.embeds
    )


_messages_mock.is_function_call_or_thoughts_embed = _is_function_call_or_thoughts_embed
sys.modules["aiuser.messages_list.messages"] = _messages_mock

validators = import_module_directly("aiuser.core.validators", "aiuser/core/validators.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Obtain the real discord.Message class so isinstance() checks work
_DiscordMessage = sys.modules["discord"].Message


class _FakeMessage(_DiscordMessage):
    """Real subclass of discord.Message so isinstance() checks succeed.

    validators.py uses ``isinstance(replied, discord.Message)`` — MagicMock
    with spec doesn't reliably pass that check, so we use a real subclass
    instead and set attributes directly.
    """
    def __init__(self, content="", embeds=None, reference=None, author=None):
        self.content = content
        self.embeds = embeds or []
        self.reference = reference
        self.author = author or MagicMock()
        self.channel = MagicMock()
        self.channel.fetch_message = AsyncMock()


def _make_mock_message(content="", embeds=None, reference=None, author=None):
    """Create a fake discord.Message that passes isinstance() checks."""
    return _FakeMessage(content=content, embeds=embeds, reference=reference, author=author)


class _FakeEmbed:
    """Lightweight embed stand-in with a real string title for regex matching."""
    def __init__(self, title=None):
        self.title = title


def _make_embed(title=None):
    """Create a fake embed with the given title (real string, not MagicMock)."""
    return _FakeEmbed(title=title)


def _make_mock_ctx(message=None, guild=None, channel=None, interaction=None):
    """Create a mock commands.Context."""
    ctx = MagicMock()
    ctx.message = message or _make_mock_message()
    ctx.guild = guild or MagicMock()
    ctx.channel = channel or MagicMock()
    ctx.interaction = interaction
    ctx.author = ctx.message.author
    return ctx


def _make_mock_cog(**overrides):
    """Create a mock cog with sensible defaults for check_message_content."""
    cog = MagicMock()
    cog.ignore_regex = overrides.get("ignore_regex", {})

    # Build a config mock where all chained calls return AsyncMock so
    # ``await cog.config.guild(ctx.guild).some_setting()`` works.
    guild_config = MagicMock()
    guild_config.messages_min_length = AsyncMock(return_value=1)
    guild_config.ignore_regex = AsyncMock(return_value=None)
    config = MagicMock()
    config.guild = MagicMock(return_value=guild_config)
    cog.config = overrides.get("config", config)
    return cog


# ---------------------------------------------------------------------------
# Tests: check_message_content rejects transient embeds from any bot
# ---------------------------------------------------------------------------

class TestCheckMessageContentRejectsTransientEmbeds:
    """check_message_content should reject messages whose embeds match
    THOUGHTS_EMBED_TITLE_REGEX or FUNCTION_CALL_EMBED_TITLE_REGEX,
    regardless of who the author is."""

    def _run_check(self, message):
        """Run check_message_content with a minimal mock cog."""
        cog = _make_mock_cog()
        ctx = _make_mock_ctx(message=message)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                validators.check_message_content(cog, ctx)
            )
        finally:
            loop.close()

    def test_rejects_function_call_embed_from_other_bot(self):
        msg = _make_mock_message(
            content="",
            embeds=[_make_embed("SomeBot is making the following function calls...")],
        )
        is_valid, reason = self._run_check(msg)
        assert is_valid is False
        assert "transient" in reason.lower() or "function" in reason.lower()

    def test_rejects_thoughts_embed_from_other_bot(self):
        msg = _make_mock_message(
            content="",
            embeds=[_make_embed("SomeBot's Thoughts")],
        )
        is_valid, reason = self._run_check(msg)
        assert is_valid is False
        assert "transient" in reason.lower() or "thoughts" in reason.lower()

    def test_allows_normal_embed(self):
        msg = _make_mock_message(
            content="hello world",
            embeds=[_make_embed("A Normal Title")],
        )
        is_valid, reason = self._run_check(msg)
        # Should pass the transient embed check (may fail other checks, but
        # not because of the embed title)
        assert "transient" not in reason.lower()

    def test_allows_message_with_no_embeds(self):
        msg = _make_mock_message(content="hello world", embeds=[])
        is_valid, reason = self._run_check(msg)
        assert "transient" not in reason.lower()

    def test_rejects_thoughts_embed_with_emoji_prefix(self):
        """The regex is ^.*'s Thoughts$ so emoji-prefixed names should match."""
        msg = _make_mock_message(
            content="",
            embeds=[_make_embed("🤖 BotName's Thoughts")],
        )
        is_valid, reason = self._run_check(msg)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Tests: is_reply_to_transient_embed
# ---------------------------------------------------------------------------

class TestIsReplyToTransientEmbed:
    """is_reply_to_transient_embed should return True when the replied-to
    message has embeds matching either FUNCTION_CALL or THOUGHTS regex."""

    def _run_check(self, message):
        cog = _make_mock_cog()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                validators.is_reply_to_transient_embed(cog, message)
            )
        finally:
            loop.close()

    def test_reply_to_function_call_embed(self):
        replied = _make_mock_message(
            embeds=[_make_embed("Bot is making the following function calls...")]
        )
        ref = MagicMock()
        ref.resolved = replied
        msg = _make_mock_message(reference=ref)
        assert self._run_check(msg) is True

    def test_reply_to_thoughts_embed(self):
        replied = _make_mock_message(
            embeds=[_make_embed("Bot's Thoughts")]
        )
        ref = MagicMock()
        ref.resolved = replied
        msg = _make_mock_message(reference=ref)
        assert self._run_check(msg) is True

    def test_reply_to_normal_embed(self):
        replied = _make_mock_message(
            embeds=[_make_embed("Some Normal Embed")]
        )
        ref = MagicMock()
        ref.resolved = replied
        msg = _make_mock_message(reference=ref)
        assert self._run_check(msg) is False

    def test_reply_to_message_with_no_embeds(self):
        replied = _make_mock_message(embeds=[])
        ref = MagicMock()
        ref.resolved = replied
        msg = _make_mock_message(reference=ref)
        assert self._run_check(msg) is False

    def test_no_reference_returns_false(self):
        msg = _make_mock_message(reference=None)
        assert self._run_check(msg) is False

    def test_unresolved_reference_fetches_message(self):
        """When reference.resolved is None, the function should fetch the message."""
        replied = _make_mock_message(
            embeds=[_make_embed("Bot's Thoughts")]
        )
        ref = MagicMock()
        ref.resolved = None
        ref.message_id = 12345
        msg = _make_mock_message(reference=ref)
        msg.channel = MagicMock()
        msg.channel.fetch_message = AsyncMock(return_value=replied)
        assert self._run_check(msg) is True

    def test_fetch_failure_returns_false(self):
        """When fetching the replied-to message fails, return False."""
        ref = MagicMock()
        ref.resolved = None
        ref.message_id = 12345
        msg = _make_mock_message(reference=ref)
        msg.channel = MagicMock()
        msg.channel.fetch_message = AsyncMock(
            side_effect=sys.modules["discord"].HTTPException(MagicMock(), "fail")
        )
        assert self._run_check(msg) is False

    def test_reply_to_embed_with_multiple_embeds_one_matches(self):
        """If the replied-to message has multiple embeds and one matches, return True."""
        replied = _make_mock_message(
            embeds=[
                _make_embed("Some Normal Embed"),
                _make_embed("Bot's Thoughts"),
            ]
        )
        ref = MagicMock()
        ref.resolved = replied
        msg = _make_mock_message(reference=ref)
        assert self._run_check(msg) is True
