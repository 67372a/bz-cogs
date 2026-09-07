"""Tests for the per-channel [p]aiuser forget command.

Validates that:
  1. The forget command stores its cutoff timestamp keyed by CHANNEL ID
     (not guild ID), so forgetting is scoped to a single channel.
  2. MessagesList resolves start_time with per-channel forget taking
     priority over the guild-wide prompt-reset override.
  3. Forgetting in one channel does not affect other channels.
  4. The guild-wide override (used by prompt resets) still works as a
     fallback for channels without a channel-specific forget.
"""

import sys
import types
import inspect
import functools
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing real modules
# ---------------------------------------------------------------------------

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
           "redbot.core.utils.chat_formatting", "redbot.core.utils.views",
           "redbot.core.utils.menus"]:
    sys.modules.setdefault(_n, MagicMock())

# Mock tiktoken
sys.modules.setdefault("tiktoken", MagicMock())

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

_mocks_needed = [
    "aiuser.config.defaults",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.opt_view",
    "aiuser.types.abc",
    "aiuser.utils.utilities",
    "aiuser.utils.image_processing",
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

# Load the real messages module
sys.modules["aiuser.messages_list.messages"] = import_module_directly(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py"
)
MessagesList = sys.modules["aiuser.messages_list.messages"].MessagesList

# ---------------------------------------------------------------------------
# Controlled mock setup for loading aiuser.settings.base (Settings).
#
# The Settings class inherits from 9 settings mixins, each a separate module.
# We replace redbot.core.commands with a passthrough-decorator module so the
# real `forget` command function survives decoration, and mock the mixins as
# plain classes so the class body can be created.
# ---------------------------------------------------------------------------

if "aiuser.settings" not in sys.modules:
    sys.modules["aiuser.settings"] = _make_mock_package("aiuser.settings")

_settings_mixins = [
    ("functions", "FunctionCallingSettings"),
    ("history", "HistorySettings"),
    ("image_request", "ImageRequestSettings"),
    ("image_scan", "ImageScanSettings"),
    ("owner", "OwnerSettings"),
    ("prompt", "PromptSettings"),
    ("random_message", "RandomMessageSettings"),
    ("response", "ResponseSettings"),
    ("triggers", "TriggerSettings"),
]
for _name, _cls in _settings_mixins:
    _mod = types.ModuleType(f"aiuser.settings.{_name}")
    setattr(_mod, _cls, type(_cls, (), {}))
    sys.modules[f"aiuser.settings.{_name}"] = _mod

# Passthrough commands module so decorated functions remain real functions
_orig_cmds_mod = sys.modules.get("redbot.core.commands")
_orig_checks_mod = sys.modules.get("redbot.core.checks")
_orig_checks_attr = getattr(sys.modules.get("redbot.core"), "checks", None)
_cmds = types.ModuleType("redbot.core.commands")


def _passthrough(*args, **kwargs):
    def deco(func):
        return func
    return deco


class _FakeGroup:
    """Stands in for commands.group(): callable proxy with .command()/.group()."""
    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)
    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
    def command(self, *args, **kwargs):
        return _passthrough(*args, **kwargs)
    def group(self, *args, **kwargs):
        return _make_group(*args, **kwargs)


def _make_group(*args, **kwargs):
    return lambda f: _FakeGroup(f)


_cmds.group = _make_group
_cmds.command = _passthrough
_cmds.hybrid_group = _passthrough
_cmds.guild_only = _passthrough
_cmds.bot_has_permissions = _passthrough
_cmds.cooldown = _passthrough
_cmds.check = _passthrough
_cmds.Context = object
_cmds.Cog = object
sys.modules["redbot.core.commands"] = _cmds
sys.modules["redbot.core"].commands = _cmds

# Fake checks module so `from redbot.core import checks` doesn't pull the real one
_checks = types.ModuleType("redbot.core.checks")
for _attr in ["is_owner", "admin", "guildowner", "admin_or_permissions",
              "guildowner_or_permissions", "is_admin", "bot_in_a_guild",
              "permissions", "bot_has_permissions"]:
    setattr(_checks, _attr, _passthrough)
_checks.check = _passthrough
sys.modules["redbot.core.checks"] = _checks
sys.modules["redbot.core"].checks = _checks

# base.py reads Settings.forget's own module-level imports; ensure these exist
sys.modules.setdefault("aiuser.types.types", MagicMock())
sys.modules.setdefault("aiuser.settings.utilities", MagicMock())

# MixinMeta must be a real class so `class Settings(..., MixinMeta)` can inherit
import abc as _abc
_abc_mod = types.ModuleType("aiuser.types.abc")
_abc_mod.MixinMeta = type("MixinMeta", (_abc.ABC,), {})
sys.modules["aiuser.types.abc"] = _abc_mod

# Load the real base settings module (Settings.forget)
_base_mod = import_module_directly("aiuser.settings.base", "aiuser/settings/base.py")
Settings = _base_mod.Settings

# Restore the original redbot modules so this test's fakes don't leak
# into other test modules collected after this one
sys.modules["redbot.core.commands"] = _orig_cmds_mod
sys.modules["redbot.core"].commands = _orig_cmds_mod
if _orig_checks_mod is not None:
    sys.modules["redbot.core.checks"] = _orig_checks_mod
sys.modules["redbot.core"].checks = _orig_checks_attr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCache(dict):
    """Minimal stand-in for aiuser.utils.cache.Cache (dict-compatible)."""
    def __init__(self, limit=500):
        super().__init__()
        self.limit = limit


def _make_cog():
    """Create a minimal fake cog exposing the override/forget dicts."""
    cog = MagicMock()
    cog.ignore_regex = {}
    cog.override_prompt_start_time = {}
    cog.forget_start_times = _FakeCache()
    return cog


def _make_ctx(channel_id, guild_id=456, created_at=None):
    ctx = MagicMock()
    ctx.channel.id = channel_id
    ctx.guild.id = guild_id
    ctx.guild.name = "TestGuild"
    ctx.message = MagicMock()
    ctx.message.created_at = created_at or datetime(2024, 1, 1, 12, 0, 0)
    return ctx


# ===========================================================================
# TEST: MessagesList start_time resolution
# ===========================================================================
class TestStartTimeResolution:

    def test_channel_forget_takes_priority(self):
        """Per-channel forget time wins over a guild-wide override."""
        cog = _make_cog()
        guild_time = datetime(2024, 1, 1, 8, 0, 0)
        channel_time = datetime(2024, 1, 1, 10, 0, 0)
        cog.override_prompt_start_time[456] = guild_time
        cog.forget_start_times[123] = channel_time

        ml = MessagesList(cog, _make_ctx(channel_id=123, guild_id=456))
        assert ml.start_time == channel_time

    def test_fallback_to_guild_override(self):
        """Channels without a forget fall back to the guild-wide override."""
        cog = _make_cog()
        guild_time = datetime(2024, 1, 1, 8, 0, 0)
        cog.override_prompt_start_time[456] = guild_time

        ml = MessagesList(cog, _make_ctx(channel_id=123, guild_id=456))
        assert ml.start_time == guild_time

    def test_no_override_means_no_start_time(self):
        """With no forget and no guild override, start_time is None."""
        cog = _make_cog()
        ml = MessagesList(cog, _make_ctx(channel_id=123, guild_id=456))
        assert ml.start_time is None

    def test_forget_is_per_channel_isolated(self):
        """Forgetting in channel A must not affect channel B in the same guild."""
        cog = _make_cog()
        forget_time = datetime(2024, 1, 1, 10, 0, 0)
        cog.forget_start_times[111] = forget_time

        ml_a = MessagesList(cog, _make_ctx(channel_id=111, guild_id=456))
        ml_b = MessagesList(cog, _make_ctx(channel_id=222, guild_id=456))

        assert ml_a.start_time == forget_time
        assert ml_b.start_time is None


# ===========================================================================
# TEST: forget command writes a channel-keyed timestamp
# ===========================================================================
class TestForgetCommand:

    def test_forget_keys_by_channel_id(self):
        """The forget command must store the cutoff under ctx.channel.id."""
        source = inspect.getsource(Settings.forget)
        assert "self.forget_start_times[ctx.channel.id]" in source
        # Must NOT write the guild-wide override anymore
        assert "self.override_prompt_start_time" not in source

    def test_forget_command_sets_channel_time(self):
        """Calling forget() as a bound method sets the channel-keyed entry."""
        import asyncio

        cog = _make_cog()
        cog.config.guild = MagicMock()
        cog.config.guild.return_value.public_forget = AsyncMock(return_value=False)

        forget_time = datetime(2024, 1, 1, 12, 0, 0)
        ctx = _make_ctx(channel_id=123, guild_id=456, created_at=forget_time)
        ctx.author = MagicMock()
        ctx.channel.permissions_for = MagicMock(return_value=MagicMock(manage_messages=True))
        ctx.react_quietly = AsyncMock()

        asyncio.run(Settings.forget(cog, ctx))

        assert cog.forget_start_times[123] == forget_time
        # Guild-wide override untouched
        assert 456 not in cog.override_prompt_start_time
        ctx.react_quietly.assert_awaited_once_with("✅")

    def test_forget_command_denied_without_permission(self):
        """Non-mods are rejected when public_forget is disabled."""
        import asyncio

        cog = _make_cog()
        cog.config.guild = MagicMock()
        cog.config.guild.return_value.public_forget = AsyncMock(return_value=False)

        ctx = _make_ctx(channel_id=123, guild_id=456)
        ctx.author = MagicMock()
        ctx.channel.permissions_for = MagicMock(return_value=MagicMock(manage_messages=False))
        ctx.react_quietly = AsyncMock()

        asyncio.run(Settings.forget(cog, ctx))

        assert 123 not in cog.forget_start_times
        ctx.react_quietly.assert_awaited_once_with("❌")


# ===========================================================================
# TEST: add_history honors the per-channel start_time cutoff
# ===========================================================================
class TestAddHistoryCutoff:

    def test_start_time_drives_history_after_cutoff(self):
        """add_history passes start_time (minus 1s) as the 'after' bound."""
        cog = _make_cog()
        forget_time = datetime(2024, 1, 1, 10, 0, 0)
        cog.forget_start_times[123] = forget_time

        ml = MessagesList(cog, _make_ctx(channel_id=123, guild_id=456))
        ml.messages = []
        ml.config.guild = AsyncMock(return_value=AsyncMock())
        guild_cfg = AsyncMock()
        guild_cfg.messages_backread = AsyncMock(return_value=10)
        guild_cfg.messages_backread_seconds = AsyncMock(return_value=3600)
        ml.config.guild = MagicMock(return_value=guild_cfg)
        ml.init_message = MagicMock()
        ml.init_message.id = 999

        captured = {}

        async def fake_get_past(limit, start_time):
            captured["limit"] = limit
            captured["start_time"] = start_time
            return [], []

        async def fake_process(before, after, gap, watermark=None):
            pass

        async def fake_unopted(messages):
            return set()

        ml._get_past_messages = fake_get_past
        ml._process_past_messages = fake_process
        ml._get_unopted_users = fake_unopted

        import asyncio
        asyncio.run(ml.add_history())

        assert captured["start_time"] == forget_time - timedelta(seconds=1)

    def test_no_forget_passes_none_start_time(self):
        """Without a forget, add_history receives start_time=None (no cutoff)."""
        cog = _make_cog()
        ml = MessagesList(cog, _make_ctx(channel_id=123, guild_id=456))
        ml.messages = []

        guild_cfg = AsyncMock()
        guild_cfg.messages_backread = AsyncMock(return_value=10)
        guild_cfg.messages_backread_seconds = AsyncMock(return_value=3600)
        ml.config.guild = MagicMock(return_value=guild_cfg)
        ml.init_message = MagicMock()
        ml.init_message.id = 999

        captured = {}

        async def fake_get_past(limit, start_time):
            captured["start_time"] = start_time
            return [], []

        ml._get_past_messages = fake_get_past
        ml._process_past_messages = AsyncMock()
        ml._get_unopted_users = AsyncMock(return_value=set())

        import asyncio
        asyncio.run(ml.add_history())

        assert captured["start_time"] is None
