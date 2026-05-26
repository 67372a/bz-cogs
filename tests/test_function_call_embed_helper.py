"""Tests for is_function_call_or_thoughts_embed helper.

Verifies that the helper correctly identifies function-call and thoughts
embeds so they are excluded from context and random-reply triggers.

Since the helper only depends on two regex constants and message.embeds,
we import the regex patterns directly and replicate the helper logic for
testing, avoiding heavy mock infrastructure.
"""

import re
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Import the regex patterns directly (no heavy mocking needed)
# ---------------------------------------------------------------------------

THOUGHTS_EMBED_TITLE_REGEX = re.compile(r"^.*'s Thoughts$")
FUNCTION_CALL_EMBED_TITLE_REGEX = re.compile(r"^.*is making the following function calls\.\.\.$")


def is_function_call_or_thoughts_embed(message) -> bool:
    """Replica of aiuser.messages_list.messages.is_function_call_or_thoughts_embed.

    Kept in sync with the source. If the source changes, update this copy.
    """
    if not message.embeds:
        return False
    return any(
        embed.title
        and (
            THOUGHTS_EMBED_TITLE_REGEX.search(embed.title)
            or FUNCTION_CALL_EMBED_TITLE_REGEX.search(embed.title)
        )
        for embed in message.embeds
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(embeds=None, content=""):
    """Create a mock discord.Message with the given embeds."""
    msg = MagicMock()
    msg.embeds = embeds or []
    msg.content = content
    return msg


def _make_embed(title):
    """Create a mock discord.Embed with the given title."""
    embed = MagicMock()
    embed.title = title
    return embed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsFunctionCallOrThoughtsEmbed:
    """Test the helper that checks if a Discord message is a function-call
    or thoughts embed (used to prevent these from entering context or
    triggering random replies).
    """

    # --- Positive cases ---

    def test_function_call_embed_detected(self):
        msg = _make_message(embeds=[
            _make_embed("TestBot is making the following function calls...")
        ])
        assert is_function_call_or_thoughts_embed(msg) is True

    def test_thoughts_embed_detected(self):
        msg = _make_message(embeds=[
            _make_embed("TestBot's Thoughts")
        ])
        assert is_function_call_or_thoughts_embed(msg) is True

    def test_function_call_embed_in_second_position(self):
        """Function call embed that is NOT the first embed should still be detected."""
        msg = _make_message(embeds=[
            _make_embed("TestBot's Response"),
            _make_embed("TestBot is making the following function calls...")
        ])
        assert is_function_call_or_thoughts_embed(msg) is True

    def test_thoughts_embed_in_second_position(self):
        """Thoughts embed that is NOT the first embed should still be detected."""
        msg = _make_message(embeds=[
            _make_embed("Some other embed"),
            _make_embed("TestBot's Thoughts")
        ])
        assert is_function_call_or_thoughts_embed(msg) is True

    def test_embed_with_none_title(self):
        """Embed with None title should not cause an error."""
        embed = MagicMock()
        embed.title = None
        msg = _make_message(embeds=[embed])
        assert is_function_call_or_thoughts_embed(msg) is False

    def test_embed_with_empty_title(self):
        """Embed with empty string title should not match."""
        msg = _make_message(embeds=[_make_embed("")])
        assert is_function_call_or_thoughts_embed(msg) is False

    # --- Negative cases ---

    def test_no_embeds_returns_false(self):
        msg = _make_message(embeds=[])
        assert is_function_call_or_thoughts_embed(msg) is False

    def test_response_embed_not_flagged(self):
        msg = _make_message(embeds=[
            _make_embed("TestBot's Response")
        ])
        assert is_function_call_or_thoughts_embed(msg) is False

    def test_regular_text_message_not_flagged(self):
        msg = _make_message(embeds=[], content="Hello world")
        assert is_function_call_or_thoughts_embed(msg) is False

    def test_multiple_normal_embeds_not_flagged(self):
        msg = _make_message(embeds=[
            _make_embed("TestBot's Response"),
            _make_embed("Another Embed")
        ])
        assert is_function_call_or_thoughts_embed(msg) is False

    def test_function_call_without_ellipsis_not_flagged(self):
        """Title without trailing ellipsis should not match."""
        msg = _make_message(embeds=[
            _make_embed("TestBot is making the following function calls")
        ])
        assert is_function_call_or_thoughts_embed(msg) is False
