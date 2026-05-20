"""Tests for stable/dynamic variable splitting in aiuser.utils.utilities.

These tests verify that:
- format_stable_variables() only substitutes stable (per-channel) variables
- build_dynamic_context_message() produces a concise dynamic context summary
- Stable variables produce identical output across calls (critical for caching)
- Dynamic variables change across calls (as expected)
- The variable sets cover all variables used by format_variables()
"""

import asyncio
import re
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiuser.utils.utilities import (
    _STABLE_VARIABLES,
    _DYNAMIC_VARIABLES,
    format_stable_variables,
    build_dynamic_context_message,
    format_variables,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mock_ctx(
    botname="TestBot",
    display_name="TestBotDisplay",
    owner_name="OwnerUser",
    author_name="Alice",
    author_display="AliceDisplay",
    author_top_role="Admin",
    author_mention="<@123>",
    guild_name="TestGuild",
    channel_name="general",
    channel_topic="A test topic",
    is_thread=False,
):
    """Build a mock commands.Context with the minimum surface for format_variables."""
    ctx = MagicMock()

    # guild.me
    ctx.message.guild.me.name = botname
    ctx.message.guild.me.display_name = display_name

    # bot
    ctx.bot.user.name = botname
    ctx.bot.user.display_name = display_name
    app_info = MagicMock()
    app_info.owner.name = owner_name
    ctx.bot.application_info = AsyncMock(return_value=app_info)

    # author
    ctx.message.author.name = author_name
    ctx.message.author.display_name = author_display
    ctx.message.author.top_role.name = author_top_role
    ctx.message.author.mention = author_mention

    # guild
    ctx.guild.name = guild_name

    # channel
    if is_thread:
        parent = MagicMock()
        parent.topic = channel_topic
        parent.name = channel_name
        ctx.message.channel = MagicMock()
        ctx.message.channel.parent = parent
        ctx.message.channel.name = channel_name
        # isinstance check for discord.Thread — we patch it
    else:
        ctx.message.channel = MagicMock()
        ctx.message.channel.topic = channel_topic
        ctx.message.channel.name = channel_name

    return ctx


# ---------------------------------------------------------------------------
# Tests: Variable set coverage
# ---------------------------------------------------------------------------

class TestVariableSetCoverage:
    """Ensure _STABLE_VARIABLES + _DYNAMIC_VARIABLES cover all variables
    used by the original format_variables() function."""

    def test_no_overlap(self):
        """Stable and dynamic variable sets must be disjoint."""
        overlap = _STABLE_VARIABLES & _DYNAMIC_VARIABLES
        assert overlap == set(), f"Overlap found: {overlap}"

    def test_all_format_variables_keys_covered(self):
        """The union of stable + dynamic must cover every key that
        format_variables() passes to str.format()."""
        # These are all the keyword arguments passed in format_variables()
        all_keys = {
            'botname', 'botdisplayname', 'botowner',
            'authorname', 'authordisplayname', 'authortoprole', 'authormention',
            'servername', 'serveremojis', 'channelname', 'channeltopic',
            'currentdate', 'currentweekday', 'currenttime', 'randomnumber',
        }
        covered = _STABLE_VARIABLES | _DYNAMIC_VARIABLES
        assert covered == all_keys, f"Missing: {all_keys - covered}, Extra: {covered - all_keys}"


# ---------------------------------------------------------------------------
# Tests: format_stable_variables
# ---------------------------------------------------------------------------

class TestFormatStableVariables:

    @pytest.mark.asyncio
    async def test_stable_substitution(self):
        """Stable variables should be substituted."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "You are {botname} in {servername} #{channelname}."
            result = await format_stable_variables(ctx, text)
            assert result == "You are TestBot in TestGuild #general."

    @pytest.mark.asyncio
    async def test_dynamic_vars_left_as_placeholders(self):
        """Dynamic variables should remain as literal {…} placeholders."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "Time: {currenttime}, User: {authorname}"
            result = await format_stable_variables(ctx, text)
            # Should still contain the literal placeholders
            assert "{currenttime}" in result
            assert "{authorname}" in result

    @pytest.mark.asyncio
    async def test_mixed_stable_and_dynamic(self):
        """A prompt with both stable and dynamic vars should resolve only stable."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "{botname} talking to {authorname} at {currenttime}"
            result = await format_stable_variables(ctx, text)
            assert result.startswith("TestBot talking to ")
            assert "{authorname}" in result
            assert "{currenttime}" in result

    @pytest.mark.asyncio
    async def test_output_stable_across_calls(self):
        """Multiple calls should produce identical output (critical for caching)."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "{botname} is in {servername} #{channelname}"
            result1 = await format_stable_variables(ctx, text)
            result2 = await format_stable_variables(ctx, text)
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_no_variables(self):
        """Text with no variables should pass through unchanged."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "No variables here."
            result = await format_stable_variables(ctx, text)
            assert result == "No variables here."

    @pytest.mark.asyncio
    async def test_dynamic_only_prompt_safely_handled(self):
        """A prompt with ONLY dynamic vars should not crash (safe formatter fallback)."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "{authorname} said {currenttime} {randomnumber}"
            result = await format_stable_variables(ctx, text)
            # The safe formatter should leave dynamic placeholders as-is
            assert "{authorname}" in result
            assert "{currenttime}" in result
            assert "{randomnumber}" in result

    @pytest.mark.asyncio
    async def test_channel_topic_resolved(self):
        """channeltopic is now stable and should be resolved in system prompt."""
        ctx = _make_mock_ctx(channel_topic="My Topic")
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={}):
            text = "Topic: {channeltopic}"
            result = await format_stable_variables(ctx, text)
            assert result == "Topic: My Topic"

    @pytest.mark.asyncio
    async def test_server_emojis_resolved(self):
        """serveremojis is now stable and should be resolved in system prompt."""
        ctx = _make_mock_ctx()
        with patch("aiuser.utils.utilities.get_guild_emoji_map", new_callable=AsyncMock, return_value={"smile": "<:smile:123>", "wave": "<:wave:456>"}):
            text = "Emojis: {serveremojis}"
            result = await format_stable_variables(ctx, text)
            assert ":smile:" in result
            assert ":wave:" in result


# ---------------------------------------------------------------------------
# Tests: build_dynamic_context_message
# ---------------------------------------------------------------------------

class TestBuildDynamicContextMessage:

    @pytest.mark.asyncio
    async def test_produces_concise_summary(self):
        """Should produce a pipe-separated context summary."""
        ctx = _make_mock_ctx()
        text = "Time: {currenttime}, User: {authorname}"
        result = await build_dynamic_context_message(ctx, text)
        assert result is not None
        # Should contain key dynamic values
        assert "author=AliceDisplay (@Alice)" in result
        assert "time=" in result
        assert "day=" in result
        assert "random=" in result
        # Should be pipe-separated format
        assert " | " in result

    @pytest.mark.asyncio
    async def test_no_dynamic_vars_returns_none(self):
        """If the text has no dynamic variable placeholders, return None."""
        ctx = _make_mock_ctx()
        text = "{botname} in {servername}"
        result = await build_dynamic_context_message(ctx, text)
        assert result is None

    @pytest.mark.asyncio
    async def test_includes_author_info(self):
        """Context should include author name and display name."""
        ctx = _make_mock_ctx(author_name="Bob", author_display="Bobby")
        text = "{authorname}"
        result = await build_dynamic_context_message(ctx, text)
        assert "Bobby (@Bob)" in result

    @pytest.mark.asyncio
    async def test_includes_time_info(self):
        """Context should include time, date, and weekday."""
        ctx = _make_mock_ctx()
        text = "{currenttime}"
        result = await build_dynamic_context_message(ctx, text)
        assert "time=" in result
        assert "date=" in result
        assert "day=" in result

    @pytest.mark.asyncio
    async def test_includes_random_number(self):
        """Context should include a random number."""
        ctx = _make_mock_ctx()
        text = "{randomnumber}"
        result = await build_dynamic_context_message(ctx, text)
        assert "random=" in result
        # Extract the number to verify it's a valid integer
        match = re.search(r'random=(\d+)', result)
        assert match is not None
        assert 0 <= int(match.group(1)) <= 100

    @pytest.mark.asyncio
    async def test_output_changes_between_calls(self):
        """Dynamic output should differ between calls (randomnumber changes)."""
        ctx = _make_mock_ctx()
        text = "Random: {randomnumber}"
        results = set()
        for _ in range(20):
            result = await build_dynamic_context_message(ctx, text)
            results.add(result)
        # With 20 calls and random 0-100, we should get multiple distinct values
        assert len(results) > 1, "Expected different random values across calls"

    @pytest.mark.asyncio
    async def test_none_text_returns_context(self):
        """Passing None text should still build context (unconditional mode)."""
        ctx = _make_mock_ctx()
        result = await build_dynamic_context_message(ctx, None)
        assert result is not None
        assert "author=AliceDisplay (@Alice)" in result

    @pytest.mark.asyncio
    async def test_channel_topic_not_in_dynamic_context(self):
        """channeltopic is now stable, so it should NOT appear in dynamic context."""
        ctx = _make_mock_ctx(channel_topic="My Topic")
        result = await build_dynamic_context_message(ctx, "{channeltopic} {currenttime}")
        assert "channel_topic" not in result
        assert "time=" in result  # but time should still be there

    @pytest.mark.asyncio
    async def test_server_emojis_not_in_dynamic_context(self):
        """serveremojis is now stable, so it should NOT appear in dynamic context."""
        ctx = _make_mock_ctx()
        result = await build_dynamic_context_message(ctx, "{serveremojis} {currenttime}")
        assert "server_emojis" not in result
        assert "time=" in result  # but time should still be there


# ---------------------------------------------------------------------------
# Tests: Integration — format_variables still works unchanged
# ---------------------------------------------------------------------------

class TestFormatVariablesUnchanged:
    """Verify the original format_variables() still works correctly."""

    @pytest.mark.asyncio
    async def test_resolves_all_variables(self):
        ctx = _make_mock_ctx()
        text = "{botname} {servername} {authorname} {currenttime}"
        result = await format_variables(ctx, text)
        assert "TestBot" in result
        assert "TestGuild" in result
        assert "Alice" in result
        # currenttime is HH:MM format
        assert re.search(r'\d{2}:\d{2}', result)


# ---------------------------------------------------------------------------
# Tests: Message construction stability (MessagesList._append_dynamic_context)
# ---------------------------------------------------------------------------

class TestAppendDynamicContext:

    def test_dynamic_context_appended_as_user_message(self):
        """Verify _append_dynamic_context creates a user MessageEntry at the end."""
        from aiuser.messages_list.entry import MessageEntry

        # Create a minimal mock MessagesList
        mock_list = MagicMock()
        mock_list._dynamic_context = "time=14:32 | day=Monday | author=AliceDisplay (@Alice)"
        mock_list.messages = [MessageEntry("system", "test"), MessageEntry("user", "hello")]
        mock_list._encoding = MagicMock()
        mock_list._encoding.encode.return_value = list(range(10))  # 10 tokens

        # Import and call the method directly
        from aiuser.messages_list.messages import MessagesList
        MessagesList._append_dynamic_context(mock_list)

        assert len(mock_list.messages) == 3
        last = mock_list.messages[-1]
        assert last.role == "user"
        assert "14:32" in last.content

    def test_no_dynamic_context_is_noop(self):
        """If _dynamic_context is None, nothing should be appended."""
        from aiuser.messages_list.entry import MessageEntry

        mock_list = MagicMock()
        mock_list._dynamic_context = None
        mock_list.messages = [MessageEntry("system", "test")]

        from aiuser.messages_list.messages import MessagesList
        MessagesList._append_dynamic_context(mock_list)

        assert len(mock_list.messages) == 1


# ---------------------------------------------------------------------------
# Tests: Cache window constants and token limit methods
# ---------------------------------------------------------------------------

class TestCacheWindowConstants:
    """Verify cache window constants and token limit methods."""

    def test_cache_window_seconds(self):
        from aiuser.messages_list.messages import CACHE_WINDOW_SECONDS
        assert CACHE_WINDOW_SECONDS == 180  # 3 minutes

    def test_cache_window_only_seconds_constant(self):
        """Only CACHE_WINDOW_SECONDS remains — backread expansion removed."""
        from aiuser.messages_list.messages import CACHE_WINDOW_SECONDS
        assert CACHE_WINDOW_SECONDS == 180  # 3 minutes
        # CACHE_WINDOW_BACKREAD was removed — only token limit is expanded
        import importlib
        mod = importlib.import_module("aiuser.messages_list.messages")
        assert not hasattr(mod, "CACHE_WINDOW_BACKREAD"), "CACHE_WINDOW_BACKREAD should be removed"

    def test_get_token_limit_has_buffer(self):
        """_get_token_limit subtracts a 1000-token buffer."""
        from aiuser.messages_list.messages import MessagesList
        raw = MessagesList._get_token_limit_raw("google/gemini-2.5-flash")
        buffered = MessagesList._get_token_limit("google/gemini-2.5-flash")
        assert buffered == raw - 1000

    def test_get_token_limit_raw_returns_full(self):
        """_get_token_limit_raw returns the full model limit."""
        from aiuser.messages_list.messages import MessagesList
        limit = MessagesList._get_token_limit_raw("google/gemini-2.5-flash")
        assert limit == 1048576  # from MODELS_LIMITS

    def test_get_token_limit_raw_unknown_model(self):
        """_get_token_limit_raw returns default 7000 for unknown models."""
        from aiuser.messages_list.messages import MessagesList
        limit = MessagesList._get_token_limit_raw("unknown-model-xyz")
        assert limit == 7000
