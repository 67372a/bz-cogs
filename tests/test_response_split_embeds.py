"""Tests for the multi-embed file-attachment bug fix in response.py.

Bug: When the LLM response text exceeds Discord's 4096-character embed
description limit AND file attachments (e.g. from ``attach_files``) are
present, ``send_single_combined_message`` built a single embed with the
full text. Discord rejected it (HTTP 400), and the fallback path sent
each file as its own standalone message with no accompanying text.

Fix (refactored): ``send_response`` now accepts an optional ``files``
parameter. When text > 4096 chars, it splits into multiple embeds and
attaches the files to the *first* message only. ``send_single_combined
_message`` delegates to ``send_response`` for the text+files and
text-only cases, eliminating the duplicated pagination logic.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing the real response module
# ---------------------------------------------------------------------------

# Minimal discord module with a real Embed class (so len checks behave)
_discord_mod = types.ModuleType("discord")


class _FakeEmbed:
    """Minimal Embed stand-in that records its description."""

    def __init__(self, **kwargs):
        self.title = kwargs.get("title")
        self.description = kwargs.get("description")
        self.footer = None

    def set_footer(self, text=None, **kw):
        self.footer = text


class _FakeFile:
    def __init__(self, fp=None, filename=None):
        self.fp = fp
        self.filename = filename


class _FakeAllowedMentions:
    def __init__(self, **kw):
        self.kw = kw


class _FakeHTTPException(Exception):
    def __init__(self, status=400, text="Bad Request"):
        self.status = status
        self.response = MagicMock()
        self.response.text = text
        super().__init__(text)


_discord_mod.Embed = _FakeEmbed
_discord_mod.File = _FakeFile
_discord_mod.AllowedMentions = _FakeAllowedMentions
_discord_mod.HTTPException = _FakeHTTPException
_discord_mod.Message = type("_DiscordMessage", (), {})
sys.modules["discord"] = _discord_mod

# Minimal redbot.core module
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
sys.modules["redbot"] = _redbot
sys.modules["redbot.core"] = _redbot_core

# Leaf aiuser modules imported by response.py — MagicMock is sufficient
_leaf_mocks = [
    "aiuser.config.constants",
    "aiuser.messages_list.messages",
    "aiuser.response.chat.llm_pipeline",
    "aiuser.response.chat.function_call_view",
    "aiuser.types.abc",
    "aiuser.utils.utilities",
    "aiuser.utils.latex_converter",
]
for _leaf in _leaf_mocks:
    sys.modules[_leaf] = MagicMock()

# Ensure every parent package exists as a mock package
_all_pkg_prefixes = set()
for _name in list(sys.modules.keys()) + ["aiuser.response.chat.response"]:
    _parts = _name.split(".")
    for _i in range(1, len(_parts)):
        _all_pkg_prefixes.add(".".join(_parts[:_i]))

for _pkg in _all_pkg_prefixes:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Patch utility functions used by response.py to passthrough so we can
# control the cleaned text length precisely.
_utils = sys.modules["aiuser.utils.utilities"]
_utils.to_thread = lambda *a, **k: (lambda f: f)
_utils.collapse_lines = lambda text, replacement=None: text
_utils.escape_unescaped_backticks = lambda text: text
_utils.resolve_emojis_for_discord = AsyncMock(return_value=None)
_utils.get_enabled_tools = MagicMock(return_value=[])

_latex = sys.modules["aiuser.utils.latex_converter"]
_latex.convert_latex_to_plain = lambda text: text

# Load the module under test
response_mod = import_module_directly(
    "aiuser.response.chat.response", "aiuser/response/chat/response.py"
)

send_single_combined_message = response_mod.send_single_combined_message
send_response = response_mod.send_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx():
    """Create a mock commands.Context with an async reply/send."""
    ctx = MagicMock()
    ctx.bot.user.name = "TestBot"
    ctx.message.id = 1000
    ctx.message.reply = AsyncMock()
    ctx.send = AsyncMock()
    ctx.interaction = None
    ctx.message.guild.me.display_name = "TestBot"
    return ctx


def _make_cog():
    cog = MagicMock()
    cog.config = MagicMock()
    return cog


def _make_images(n=1):
    """Build n image/file dicts as produced by attach_files / generate_image."""
    return [
        {"bytes": f"content-{i}".encode(), "filename": f"file{i}.txt"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# send_response tests (refactored to accept optional files)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_response_short_text_no_files():
    """Short text with no files sends a single embed via ctx.send."""
    ctx = _make_ctx()
    text = "Hello world"
    msg = await send_response(ctx, text, can_reply=False, mentionable_users=[])
    ctx.send.assert_called_once()
    send_kwargs = ctx.send.call_args.kwargs
    assert send_kwargs["embed"].description == text
    assert "files" not in send_kwargs or send_kwargs["files"] is None


@pytest.mark.asyncio
async def test_send_response_short_text_with_files():
    """Short text + files sends a single embed with files attached."""
    ctx = _make_ctx()
    text = "Hello world"
    files = [_FakeFile(filename="a.txt")]
    msg = await send_response(ctx, text, can_reply=False, mentionable_users=[], files=files)
    ctx.send.assert_called_once()
    send_kwargs = ctx.send.call_args.kwargs
    assert send_kwargs["files"] is files


@pytest.mark.asyncio
async def test_send_response_long_text_no_files():
    """Long text (>4096) with no files splits into multiple ctx.send calls."""
    ctx = _make_ctx()
    text = "A" * 9000  # 3 chunks
    await send_response(ctx, text, can_reply=False, mentionable_users=[])
    assert ctx.send.call_count == 3
    # No files on any chunk
    for c in ctx.send.call_args_list:
        assert "files" not in c.kwargs or c.kwargs["files"] is None


@pytest.mark.asyncio
async def test_send_response_long_text_with_files_attaches_to_first_only():
    """Long text + files: files attached to FIRST chunk only, not subsequent."""
    ctx = _make_ctx()
    text = "B" * 9000  # 3 chunks
    files = [_FakeFile(filename="a.txt"), _FakeFile(filename="b.txt")]
    await send_response(ctx, text, can_reply=False, mentionable_users=[], files=files)
    assert ctx.send.call_count == 3

    # First chunk carries files
    first_kwargs = ctx.send.call_args_list[0].kwargs
    assert first_kwargs["files"] is files

    # Subsequent chunks do NOT carry files
    for c in ctx.send.call_args_list[1:]:
        assert c.kwargs.get("files") is None


@pytest.mark.asyncio
async def test_send_response_reply_with_files():
    """Short text + can_reply + should_reply True: reply carries files."""
    ctx = _make_ctx()
    text = "Hello"
    files = [_FakeFile(filename="a.txt")]
    # Force should_reply to return True
    response_mod.should_reply = AsyncMock(return_value=True)
    await send_response(ctx, text, can_reply=True, mentionable_users=[], files=files)
    ctx.message.reply.assert_called_once()
    reply_kwargs = ctx.message.reply.call_args.kwargs
    assert reply_kwargs["files"] is files
    assert ctx.send.call_count == 0


# ---------------------------------------------------------------------------
# send_single_combined_message tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_combined_long_text_with_files_attaches_to_first_message(monkeypatch):
    """When text > 4096 chars + files, files attach to the FIRST embed only,
    and subsequent embeds carry no files. Regression for the original bug."""
    ctx = _make_ctx()
    cog = _make_cog()

    long_text = "A" * 5000  # > 4096
    monkeypatch.setattr(
        response_mod, "remove_patterns_from_response", AsyncMock(return_value=long_text)
    )
    monkeypatch.setattr(
        response_mod, "resolve_emojis_for_discord", AsyncMock(return_value=long_text)
    )
    monkeypatch.setattr(response_mod, "_cache_generated_images_in_response", lambda *a, **k: None)
    monkeypatch.setattr(response_mod, "_attach_reasoning_view", AsyncMock())

    images = _make_images(2)

    result = await send_single_combined_message(
        ctx, cog, long_text, images, can_reply=False, recent_authors=[]
    )

    assert result is True

    # send_response splits into 2 chunks via ctx.send; first carries files
    assert ctx.send.call_count == 2
    first_kwargs = ctx.send.call_args_list[0].kwargs
    assert len(first_kwargs.get("files", [])) == 2
    second_kwargs = ctx.send.call_args_list[1].kwargs
    assert second_kwargs.get("files") is None


@pytest.mark.asyncio
async def test_combined_short_text_with_files_uses_single_embed(monkeypatch):
    """When text <= 4096 chars with files, a single embed+files is used."""
    ctx = _make_ctx()
    cog = _make_cog()

    short_text = "Hello world"
    monkeypatch.setattr(
        response_mod, "remove_patterns_from_response", AsyncMock(return_value=short_text)
    )
    monkeypatch.setattr(
        response_mod, "resolve_emojis_for_discord", AsyncMock(return_value=short_text)
    )
    monkeypatch.setattr(response_mod, "_cache_generated_images_in_response", lambda *a, **k: None)
    monkeypatch.setattr(response_mod, "_attach_reasoning_view", AsyncMock())

    images = _make_images(1)

    result = await send_single_combined_message(
        ctx, cog, short_text, images, can_reply=False, recent_authors=[]
    )

    assert result is True
    assert ctx.send.call_count == 1
    send_kwargs = ctx.send.call_args.kwargs
    assert len(send_kwargs.get("files", [])) == 1


@pytest.mark.asyncio
async def test_combined_long_text_with_files_no_standalone_file_messages(monkeypatch):
    """Regression: previously long text + files caused an HTTPException that
    sent files as standalone messages. Verify no per-file standalone replies
    occur now (reply called at most once for the whole response)."""
    ctx = _make_ctx()
    cog = _make_cog()

    long_text = "C" * 6000
    monkeypatch.setattr(
        response_mod, "remove_patterns_from_response", AsyncMock(return_value=long_text)
    )
    monkeypatch.setattr(
        response_mod, "resolve_emojis_for_discord", AsyncMock(return_value=long_text)
    )
    monkeypatch.setattr(response_mod, "_cache_generated_images_in_response", lambda *a, **k: None)
    monkeypatch.setattr(response_mod, "_attach_reasoning_view", AsyncMock())

    images = _make_images(3)

    await send_single_combined_message(
        ctx, cog, long_text, images, can_reply=False, recent_authors=[]
    )

    # The buggy path called ctx.message.reply once PER file in the fallback.
    # The fix delegates to send_response which attaches all files to the
    # first chunk — reply should never be called in this code path.
    assert ctx.message.reply.call_count == 0
    # All 3 files on the first ctx.send
    first_kwargs = ctx.send.call_args_list[0].kwargs
    assert len(first_kwargs.get("files", [])) == 3


@pytest.mark.asyncio
async def test_combined_files_only_no_text(monkeypatch):
    """Files only (no text): sends files as a reply with no embed."""
    ctx = _make_ctx()
    cog = _make_cog()

    monkeypatch.setattr(response_mod, "_cache_generated_images_in_response", lambda *a, **k: None)
    monkeypatch.setattr(response_mod, "_attach_reasoning_view", AsyncMock())

    images = _make_images(2)

    result = await send_single_combined_message(
        ctx, cog, None, images, can_reply=False, recent_authors=[]
    )

    assert result is True
    ctx.message.reply.assert_called_once()
    reply_kwargs = ctx.message.reply.call_args.kwargs
    assert len(reply_kwargs["files"]) == 2
    assert "embed" not in reply_kwargs


@pytest.mark.asyncio
async def test_combined_text_only_no_files(monkeypatch):
    """Text only (no files): delegates to send_response, no files param."""
    ctx = _make_ctx()
    cog = _make_cog()

    short_text = "Just text"
    monkeypatch.setattr(
        response_mod, "remove_patterns_from_response", AsyncMock(return_value=short_text)
    )
    monkeypatch.setattr(
        response_mod, "resolve_emojis_for_discord", AsyncMock(return_value=short_text)
    )
    monkeypatch.setattr(response_mod, "_attach_reasoning_view", AsyncMock())

    result = await send_single_combined_message(
        ctx, cog, short_text, [], can_reply=False, recent_authors=[]
    )

    assert result is True
    ctx.send.assert_called_once()
    send_kwargs = ctx.send.call_args.kwargs
    assert send_kwargs.get("files") is None
