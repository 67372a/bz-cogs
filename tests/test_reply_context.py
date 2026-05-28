"""Tests for reply context insertion and trigger message position tracking.

Validates the fix for the out-of-order context construction bug where
reply-chained parent messages were inserted at the END of context instead
of their correct chronological position.

The fix:
  1. Removed recursive reply chaining from add_msg()
  2. Added _insert_reply_context() that inserts the trigger's reply parent
     immediately before the trigger message, preserving prefix stability.
  3. Added _trigger_msg_start tracking to know where the trigger sits.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing real modules
# ---------------------------------------------------------------------------

# Create a real discord module with Message as a proper class
# so isinstance(resolved, discord.Message) works in tests
discord_mock = types.ModuleType("discord")
discord_mock.__package__ = "discord"
discord_mock.__path__ = []


class _DiscordMessage:
    """Real class to stand in for discord.Message in isinstance checks."""
    pass


discord_mock.Message = _DiscordMessage
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

# Mock tiktoken
sys.modules.setdefault("tiktoken", MagicMock())

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
]
for _pkg in _sub_pkgs:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Mock modules that messages.py imports but we don't need the real code for
_mocks_needed = [
    "aiuser.config.defaults",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.opt_view",
    "aiuser.types.abc",
    "aiuser.utils.utilities",
]
for _m in _mocks_needed:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Load real modules we DO need
sys.modules["aiuser.config.models"] = import_module_directly(
    "aiuser.config.models", "aiuser/config/models.py"
)
sys.modules["aiuser.config.constants"] = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
sys.modules["aiuser.types.enums"] = import_module_directly(
    "aiuser.types.enums", "aiuser/types/enums.py"
)

MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry

# Load the real messages module
sys.modules["aiuser.messages_list.messages"] = import_module_directly(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py"
)
MessagesList = sys.modules["aiuser.messages_list.messages"].MessagesList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ml(token_limit=100000):
    """Create a MessagesList instance with minimal mocking."""
    ml = MessagesList.__new__(MessagesList)
    ml.ctx = MagicMock()
    ml.ctx.channel.id = 123
    ml.ctx.guild.id = 456
    ml.ctx.guild.name = "TestGuild"
    ml.ctx.me.id = 789
    ml.model = "gpt-4"
    ml.tokens = 0
    ml.token_limit = token_limit
    ml.messages = []
    ml.messages_ids = set()
    ml.prefill = None
    ml.can_reply = True
    ml._dynamic_context = None
    ml._raw_persona = None
    ml._cache_window_active = False
    ml._trigger_msg_start = None
    ml.init_message = MagicMock()
    ml.init_message.id = 999
    ml.init_message.reference = None
    ml.guild = ml.ctx.guild
    ml.bot = MagicMock()
    ml.bot.user.id = 789
    ml.config = MagicMock()
    ml.converter = MagicMock()
    ml.ignore_regex = None
    ml.start_time = None
    ml.backfill_anchors = {}
    _enc = MagicMock()
    _enc.encode = MagicMock(return_value=[1, 2, 3])
    ml._encoding = _enc
    return ml


def _roles(ml):
    """Extract (role, content) tuples from message list."""
    result = []
    for m in ml.messages:
        if isinstance(m.content, str):
            result.append((m.role, m.content))
        elif isinstance(m.content, list):
            texts = [i.get("text", "") for i in m.content if isinstance(i, dict) and i.get("type") == "text"]
            result.append((m.role, " ".join(texts)))
        else:
            result.append((m.role, str(m.content)))
    return result


def _make_discord_msg(msg_id, content="test"):
    """Create a mock that passes isinstance(..., discord.Message)."""
    msg = _DiscordMessage()
    msg.id = msg_id
    msg.content = content
    msg.author = MagicMock()
    msg.author.id = 111
    msg.author.bot = False
    msg.author.bot = False
    msg.guild = MagicMock()
    msg.guild.id = 456
    msg.embeds = []
    msg.stickers = []
    msg.attachments = []
    msg.reference = None
    msg.created_at = MagicMock()
    msg.channel = MagicMock()
    msg.channel.id = 123
    return msg


# ===========================================================================
# TEST: add_msg no longer has recursive reply chaining
# ===========================================================================
class TestAddMsgNoRecursiveReplyChain:

    def test_add_msg_source_has_no_recursive_self_call(self):
        """add_msg must not recursively call itself (old reply chain removed)."""
        import inspect
        source = inspect.getsource(MessagesList.add_msg)
        lines = source.split('\n')
        recursive_calls = [
            line for line in lines
            if 'await self.add_msg(' in line and 'def add_msg' not in line
        ]
        assert len(recursive_calls) == 0, (
            f"add_msg should not recursively call itself. Found: {recursive_calls}"
        )


# ===========================================================================
# TEST: _insert_reply_context behavior
# ===========================================================================
class TestInsertReplyContext:

    @pytest.mark.asyncio
    async def test_inserts_parent_before_trigger(self):
        """Reply parent is inserted at _trigger_msg_start, just before trigger."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "history"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 2

        parent = _make_discord_msg(100, "parent msg")
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent

        entry = MessageEntry("user", "parent msg")
        ml.converter.convert = AsyncMock(return_value=[entry])

        await ml._insert_reply_context()

        r = _roles(ml)
        assert r[0] == ("system", "prompt")
        assert r[1] == ("user", "history")
        assert r[2] == ("user", "parent msg")   # inserted before trigger
        assert r[3] == ("user", "trigger")
        assert ml._trigger_msg_start == 3

    @pytest.mark.asyncio
    async def test_noop_when_no_reference(self):
        """No-op when trigger message has no reply reference."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt"), MessageEntry("user", "trigger")]
        ml._trigger_msg_start = 1
        ml.init_message.reference = None

        await ml._insert_reply_context()

        assert len(ml.messages) == 2

    @pytest.mark.asyncio
    async def test_noop_when_resolved_not_discord_message(self):
        """No-op when resolved reference is not a discord.Message instance."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt"), MessageEntry("user", "trigger")]
        ml._trigger_msg_start = 1

        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = MagicMock()  # NOT a discord.Message

        await ml._insert_reply_context()

        assert len(ml.messages) == 2

    @pytest.mark.asyncio
    async def test_noop_when_parent_already_in_context(self):
        """No-op when the parent message is already in the message list."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "already here"),
            MessageEntry("user", "trigger"),
        ]
        ml.messages_ids.add(100)
        ml._trigger_msg_start = 2

        parent = _make_discord_msg(100)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent

        await ml._insert_reply_context()

        assert len(ml.messages) == 3  # unchanged

    @pytest.mark.asyncio
    async def test_noop_when_trigger_start_is_none(self):
        """No-op when _trigger_msg_start is None (not initialized)."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt")]
        ml._trigger_msg_start = None

        parent = _make_discord_msg(100)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent

        await ml._insert_reply_context()

        assert len(ml.messages) == 1

    @pytest.mark.asyncio
    async def test_noop_when_converter_returns_none(self):
        """No-op when the converter returns None for the parent."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt"), MessageEntry("user", "trigger")]
        ml._trigger_msg_start = 1

        parent = _make_discord_msg(100)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=None)

        await ml._insert_reply_context()

        assert len(ml.messages) == 2

    @pytest.mark.asyncio
    async def test_prefix_not_shifted(self):
        """Messages before _trigger_msg_start are unaffected."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "old history"),
            MessageEntry("user", "recent history"),
            MessageEntry("user", "trigger"),
            MessageEntry("user", "after msg"),
        ]
        ml._trigger_msg_start = 3

        parent = _make_discord_msg(200)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "parent")])

        await ml._insert_reply_context()

        r = _roles(ml)
        # Prefix unchanged
        assert r[0] == ("system", "prompt")
        assert r[1] == ("user", "old history")
        assert r[2] == ("user", "recent history")
        # Parent inserted just before trigger
        assert r[3] == ("user", "parent")
        assert r[4] == ("user", "trigger")
        # After messages still at end
        assert r[5] == ("user", "after msg")

    @pytest.mark.asyncio
    async def test_multi_entry_parent(self):
        """Multiple entries from parent are inserted in order."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 1

        parent = _make_discord_msg(300)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[
            MessageEntry("user", "embed"),
            MessageEntry("user", "text"),
        ])

        await ml._insert_reply_context()

        r = _roles(ml)
        assert r[0] == ("system", "prompt")
        assert r[1] == ("user", "embed")
        assert r[2] == ("user", "text")
        assert r[3] == ("user", "trigger")
        assert ml._trigger_msg_start == 3

    @pytest.mark.asyncio
    async def test_respects_token_limit(self):
        """Insertion stops when token limit is exceeded."""
        ml = _make_ml(token_limit=2)
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 1
        ml.tokens = 3  # exceeds limit (check is `tokens > token_limit`)

        parent = _make_discord_msg(400)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[
            MessageEntry("user", "should not fit"),
        ])

        await ml._insert_reply_context()

        # No entries should have been inserted (tokens already exceed limit)
        assert len(ml.messages) == 2

    @pytest.mark.asyncio
    async def test_parent_id_added_to_messages_ids(self):
        """Parent message ID is tracked in messages_ids."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 1

        parent = _make_discord_msg(500)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "parent")])

        await ml._insert_reply_context()

        assert 500 in ml.messages_ids

    @pytest.mark.asyncio
    async def test_trigger_start_updated_after_insertion(self):
        """_trigger_msg_start is shifted right by number of entries inserted."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 1

        parent = _make_discord_msg(600)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[
            MessageEntry("user", "a"),
            MessageEntry("user", "b"),
            MessageEntry("user", "c"),
        ])

        await ml._insert_reply_context()

        # 3 entries inserted, trigger moved from 1 to 4
        assert ml._trigger_msg_start == 4


# ===========================================================================
# TEST: _trigger_msg_start tracking during history processing
# ===========================================================================
class TestTriggerMsgStartTracking:

    def test_initialized_to_none(self):
        """Default value is None."""
        ml = _make_ml()
        assert ml._trigger_msg_start is None

    @pytest.mark.asyncio
    async def test_add_msg_single_entry_increments_count(self):
        """A single-entry add_msg increments the message count by 1."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt")]
        ml._trigger_msg_start = 1

        msg = _make_discord_msg(10, "hello")
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "hello")])

        ml.bot.allowed_by_whitelist_blacklist = AsyncMock(return_value=True)
        ml.config.optout = AsyncMock(return_value=[])
        ml.config.optin = AsyncMock(return_value=[111])
        ml.config.guild.return_value.optin_by_default = AsyncMock(return_value=True)

        before_count = len(ml.messages)
        await ml.add_msg(msg, index=1)
        entries_added = len(ml.messages) - before_count

        assert entries_added == 1
        if ml._trigger_msg_start is not None:
            ml._trigger_msg_start += entries_added
        assert ml._trigger_msg_start == 2

    @pytest.mark.asyncio
    async def test_multi_entry_add_msg_increments_correctly(self):
        """A multi-entry add_msg increments by the number of entries."""
        ml = _make_ml()
        ml.messages = [MessageEntry("system", "prompt")]
        ml._trigger_msg_start = 1

        msg = _make_discord_msg(20, "multi")
        ml.converter.convert = AsyncMock(return_value=[
            MessageEntry("user", "part1"),
            MessageEntry("user", "part2"),
            MessageEntry("user", "part3"),
        ])

        ml.bot.allowed_by_whitelist_blacklist = AsyncMock(return_value=True)
        ml.config.optout = AsyncMock(return_value=[])
        ml.config.optin = AsyncMock(return_value=[111])
        ml.config.guild.return_value.optin_by_default = AsyncMock(return_value=True)

        before_count = len(ml.messages)
        await ml.add_msg(msg, index=1)
        entries_added = len(ml.messages) - before_count

        assert entries_added == 3
        if ml._trigger_msg_start is not None:
            ml._trigger_msg_start += entries_added
        assert ml._trigger_msg_start == 4


# ===========================================================================
# TEST: Context layout after full construction
# ===========================================================================
class TestContextLayout:

    @pytest.mark.asyncio
    async def test_reply_parent_inserted_before_trigger_not_at_end(self):
        """The reply parent must appear before the trigger, NOT at the end.

        This is the core regression test for the ordering bug.
        Simulates: system -> history -> [parent inserted here] -> trigger -> after_msgs
        """
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "oldest"),
            MessageEntry("user", "middle"),
            MessageEntry("user", "trigger msg"),
            MessageEntry("user", "after1"),
            MessageEntry("user", "after2"),
        ]
        ml._trigger_msg_start = 3  # trigger at index 3

        parent = _make_discord_msg(777, "reply parent")
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent
        ml.converter.convert = AsyncMock(return_value=[
            MessageEntry("user", "reply parent"),
        ])

        await ml._insert_reply_context()

        r = _roles(ml)
        # Parent is NOT at the end — it's right before the trigger
        assert r[3] == ("user", "reply parent")
        assert r[4] == ("user", "trigger msg")
        # After messages still at end
        assert r[-1] == ("user", "after2")
        assert r[-2] == ("user", "after1")
        # Total: 7 messages
        assert len(ml.messages) == 7

    @pytest.mark.asyncio
    async def test_multiple_sequential_insertions(self):
        """Multiple calls to _insert_reply_context each insert at the updated
        trigger position (if init_message reference changes between calls)."""
        ml = _make_ml()
        ml.messages = [
            MessageEntry("system", "prompt"),
            MessageEntry("user", "trigger"),
        ]
        ml._trigger_msg_start = 1

        parent1 = _make_discord_msg(101)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent1
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "parent1")])

        await ml._insert_reply_context()

        r = _roles(ml)
        assert r[1] == ("user", "parent1")
        assert r[2] == ("user", "trigger")
        assert ml._trigger_msg_start == 2

        # Simulate a second call (different parent)
        parent2 = _make_discord_msg(102)
        ml.init_message.reference = MagicMock()
        ml.init_message.reference.resolved = parent2
        ml.converter.convert = AsyncMock(return_value=[MessageEntry("user", "parent2")])

        await ml._insert_reply_context()

        r = _roles(ml)
        assert r[1] == ("user", "parent1")
        assert r[2] == ("user", "parent2")
        assert r[3] == ("user", "trigger")
        assert ml._trigger_msg_start == 3
