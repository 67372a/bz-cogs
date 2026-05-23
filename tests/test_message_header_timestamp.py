"""Tests for _get_msg_header timestamp formatting.

Verifies that the XML message header:
- Excludes milliseconds/microseconds from the timestamp
- Uses 'Z' shorthand for UTC timezone instead of offset like '+00:00'
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from aiuser.messages_list.converter.helpers import _get_msg_header


def _make_mock_message(
    msg_id=123456,
    author_name="testuser",
    author_display="Test User",
    author_id=789,
    created_at=None,
    reference=None,
    guild_me_id=9999,
):
    """Create a minimal mock Discord Message for _get_msg_header."""
    msg = MagicMock()
    msg.id = msg_id
    msg.author.name = author_name
    msg.author.display_name = author_display
    msg.author.id = author_id
    msg.reference = reference
    msg.guild.me.id = guild_me_id

    if created_at is None:
        # Default: a UTC datetime with microseconds and a non-zero offset
        created_at = datetime(2025, 7, 15, 14, 30, 45, 123456, tzinfo=timezone.utc)
    msg.created_at = created_at

    return msg


class TestGetMsgHeaderTimestamp:
    """Tests for _get_msg_header timestamp formatting."""

    def test_utc_timestamp_no_milliseconds(self):
        """Timestamp should not contain microseconds/milliseconds."""
        dt = datetime(2025, 7, 15, 14, 30, 45, 999999, tzinfo=timezone.utc)
        msg = _make_mock_message(created_at=dt)
        header = _get_msg_header(msg)
        assert "14:30:45Z" in header
        # Ensure no decimal point (which would indicate microseconds)
        assert "." not in header.split('timestamp="')[1].split('"')[0]

    def test_utc_timestamp_uses_z_suffix(self):
        """Timestamp should use 'Z' instead of '+00:00' offset."""
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        msg = _make_mock_message(created_at=dt)
        header = _get_msg_header(msg)
        assert 'timestamp="2025-01-01T00:00:00Z"' in header
        assert "+00:00" not in header

    def test_non_utc_timezone_converted_to_utc(self):
        """A non-UTC timezone should be converted to UTC with Z suffix."""
        # UTC+5:30 → should subtract 5:30 from the time
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2025, 7, 15, 20, 0, 0, 500000, tzinfo=ist)
        msg = _make_mock_message(created_at=dt)
        header = _get_msg_header(msg)
        # 20:00 IST = 14:30 UTC
        assert 'timestamp="2025-07-15T14:30:00Z"' in header

    def test_negative_utc_offset_converted(self):
        """A negative UTC offset (e.g. US Eastern) should convert to UTC."""
        eastern = timezone(timedelta(hours=-4))  # EDT
        dt = datetime(2025, 7, 15, 10, 30, 30, 1234, tzinfo=eastern)
        msg = _make_mock_message(created_at=dt)
        header = _get_msg_header(msg)
        # 10:30 EDT (-4) = 14:30 UTC
        assert 'timestamp="2025-07-15T14:30:30Z"' in header

    def test_zero_microseconds_preserved_as_clean(self):
        """When microseconds are 0, the timestamp should still end with Z."""
        dt = datetime(2025, 12, 31, 23, 59, 59, 0, tzinfo=timezone.utc)
        msg = _make_mock_message(created_at=dt)
        header = _get_msg_header(msg)
        assert 'timestamp="2025-12-31T23:59:59Z"' in header

    def test_header_structure_with_timestamp(self):
        """Full header should have correct structure with the formatted timestamp."""
        dt = datetime(2025, 3, 10, 8, 15, 30, 456789, tzinfo=timezone.utc)
        msg = _make_mock_message(msg_id=42, created_at=dt)
        header = _get_msg_header(msg)
        assert header.startswith('<message id="42"')
        assert 'timestamp="2025-03-10T08:15:30Z"' in header
        assert 'author_id="789"' in header
        assert header.endswith(">")

    def test_header_with_reply_info_simple_ref(self):
        """Reply info with message_id only (no resolved message) should appear in header."""
        dt = datetime(2025, 5, 20, 12, 0, 0, 0, tzinfo=timezone.utc)
        ref = MagicMock()
        # Simulate reference with only message_id set (no resolved message)
        ref.resolved = None
        ref.message_id = 999
        msg = _make_mock_message(created_at=dt, reference=ref)
        header = _get_msg_header(msg)
        assert 'timestamp="2025-05-20T12:00:00Z"' in header
        assert 'reply_to_id="999"' in header
