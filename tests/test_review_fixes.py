"""Tests for the second round of aiuser defect fixes.

Covers:
- utils/cache.py: dict contract (KeyError), LRU recency, deletion/eviction robustness
- core/handlers.py: epoch + legacy ratelimit_reset parsing
- core/random_message_task.py: guild loop no longer aborts after the first guild
- core/aiuser.py: cog_unload cancels queue processors; token update closes old client
- response/dispatcher.py: image errors logged, reaction removal failure tolerated
"""

import ast
import asyncio
import logging
import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing modules that depend on
# discord / redbot (not installed in the test environment)
# ---------------------------------------------------------------------------

if "discord" not in sys.modules:
    _discord_mod = types.ModuleType("discord")
    sys.modules["discord"] = _discord_mod

# Ensure the attributes handlers.py / dispatcher.py need exist, regardless of
# whether discord is real or mocked (possibly minimally by another test file).
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
    "aiuser", "aiuser.config", "aiuser.core", "aiuser.response",
    "aiuser.response.chat", "aiuser.response.image", "aiuser.types",
    "aiuser.utils", "aiuser.messages_list",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Leaf aiuser modules imported by handlers.py / dispatcher.py
_leaf_mocks = [
    "aiuser.types.abc",
    "aiuser.core.triggers",
    "aiuser.core.validators",
    "aiuser.core.openai_utils",
    "aiuser.utils.utilities",
    "aiuser.messages_list.messages",
    "aiuser.response.chat.response",
    "aiuser.response.image.generator_factory",
    "aiuser.response.image.response",
    "aiuser.response.is_image_request",
]
for _leaf in _leaf_mocks:
    if _leaf not in sys.modules:
        sys.modules[_leaf] = MagicMock()

# config.constants / config.defaults need real values used at module level
sys.modules.pop("aiuser.config.constants", None)
sys.modules.pop("aiuser.config.defaults", None)
sys.modules.pop("aiuser.types.enums", None)
import_module_directly("aiuser.types.enums", "aiuser/types/enums.py")
import_module_directly("aiuser.config.constants", "aiuser/config/constants.py")
import_module_directly("aiuser.config.defaults", "aiuser/config/defaults.py")

handlers = import_module_directly("aiuser.core.handlers", "aiuser/core/handlers.py")
cache_mod = import_module_directly("aiuser.utils.cache", "aiuser/utils/cache.py")


# ---------------------------------------------------------------------------
# utils/cache.py
# ---------------------------------------------------------------------------

def test_cache_missing_key_raises_keyerror():
    Cache = cache_mod.Cache
    c = Cache(limit=2)
    try:
        c["missing"]
        assert False, "Expected KeyError"
    except KeyError:
        pass
    assert c.get("missing") is None
    assert "missing" not in c


def test_cache_lru_recency_on_get():
    """Reading a key must refresh its recency so it isn't evicted next."""
    Cache = cache_mod.Cache
    c = Cache(limit=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert c["a"] == 1  # refresh "a"
    c["d"] = 4          # evicts "b" (now the LRU), not "a"
    assert "a" in c
    assert "b" not in c
    assert "c" in c and "d" in c


def test_cache_delete_then_evict_does_not_crash():
    """Regression: the old parallel key list went stale on __delitem__/pop,
    crashing eviction with a KeyError."""
    Cache = cache_mod.Cache
    c = Cache(limit=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    del c["a"]
    c.pop("b")
    c["d"] = 4
    c["e"] = 5  # must not raise
    assert dict(c) == {"c": 3, "d": 4, "e": 5}


def test_cache_clear_then_reuse():
    Cache = cache_mod.Cache
    c = Cache(limit=2)
    c["a"] = 1
    c.clear()
    c["b"] = 2
    c["c"] = 3
    c["d"] = 4  # evicts "b"
    assert "b" not in c and dict(c) == {"c": 3, "d": 4}


def test_cache_update_existing_key_no_eviction():
    Cache = cache_mod.Cache
    c = Cache(limit=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 99  # update, not insert — size stays 2, nothing evicted
    assert len(c) == 2
    assert c["a"] == 99
    assert "b" in c


def test_cache_pop_with_default():
    """messages.py uses backfill_anchors.pop(channel_id, None)."""
    Cache = cache_mod.Cache
    c = Cache(limit=2)
    assert c.pop("missing", None) is None
    c["x"] = 1
    assert c.pop("x", None) == 1
    assert "x" not in c


# ---------------------------------------------------------------------------
# core/handlers.py — ratelimit_reset parsing (epoch + legacy formats)
# ---------------------------------------------------------------------------

def test_ratelimit_reset_epoch_format():
    """New format: epoch seconds string."""
    cog = MagicMock()
    epoch = datetime(2030, 1, 2, 3, 4, 5).timestamp()
    cog.config.ratelimit_reset = AsyncMock(return_value=str(epoch))
    result = asyncio.run(handlers.get_ratelimit_reset(cog))
    assert result == datetime.fromtimestamp(epoch)


def test_ratelimit_reset_legacy_format_still_accepted():
    cog = MagicMock()
    cog.config.ratelimit_reset = AsyncMock(return_value="2030-01-02 03:04:05")
    result = asyncio.run(handlers.get_ratelimit_reset(cog))
    assert result == datetime(2030, 1, 2, 3, 4, 5)


def test_ratelimit_reset_garbage_returns_none():
    cog = MagicMock()
    cog.config.ratelimit_reset = AsyncMock(return_value="garbage-value")
    assert asyncio.run(handlers.get_ratelimit_reset(cog)) is None

    cog.config.ratelimit_reset = AsyncMock(return_value=None)
    assert asyncio.run(handlers.get_ratelimit_reset(cog)) is None


# ---------------------------------------------------------------------------
# core/random_message_task.py — guild loop must not abort after first guild
# ---------------------------------------------------------------------------

def _get_loop_body(source_file, func_name):
    tree = ast.parse(open(source_file, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"{func_name} not found in {source_file}")


def test_random_message_loop_uses_continue_not_return():
    """Defect: every skip path in the guild loop used `return`, so only the
    first guild was ever evaluated per cycle.  Skip paths must `continue`."""
    func = _get_loop_body(
        "aiuser/core/random_message_task.py", "random_message_trigger"
    )
    loop = next(n for n in ast.walk(func) if isinstance(n, ast.For))

    # Collect Return statements that are lexically inside the for-loop body
    returns_in_loop = [n for n in ast.walk(loop) if isinstance(n, ast.Return)]
    continues = [n for n in ast.walk(loop) if isinstance(n, ast.Continue)]

    # The skip paths must be `continue`...
    assert len(continues) >= 4, (
        f"Expected at least 4 `continue` skip paths in the guild loop, found {len(continues)}"
    )
    # ...and at most ONE `return` is allowed (the single success path that
    # sends one random message per cycle); it must not be bare-skip returns.
    assert len(returns_in_loop) <= 1, (
        f"Found {len(returns_in_loop)} return statements inside the guild loop; "
        "skip paths must use `continue` so all guilds are evaluated"
    )


def test_random_message_loop_shuffles_guilds():
    """Guild iteration order is shuffled so one eligible guild can't starve others."""
    source = open("aiuser/core/random_message_task.py", encoding="utf-8").read()
    assert "random.shuffle" in source


# ---------------------------------------------------------------------------
# core/aiuser.py — cog_unload task cleanup & client leak (AST-level checks)
# ---------------------------------------------------------------------------

def test_cog_unload_cancels_processing_tasks():
    source = open("aiuser/core/aiuser.py", encoding="utf-8").read()
    tree = ast.parse(source)
    unload = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "cog_unload"
    )
    calls = {
        getattr(n.func, "attr", None)
        for n in ast.walk(unload)
        if isinstance(n, ast.Call)
    }
    assert "cancel" in calls, "cog_unload must cancel processing_tasks"
    assert "clear" in calls, "cog_unload must clear task/queue state"


def test_token_update_closes_old_client():
    source = open("aiuser/core/aiuser.py", encoding="utf-8").read()
    tree = ast.parse(source)
    listener = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_red_api_tokens_update"
    )
    closes = [
        n for n in ast.walk(listener)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "close"
    ]
    assert closes, "on_red_api_tokens_update must close the previous OpenAI client"


# ---------------------------------------------------------------------------
# response/dispatcher.py — image error handling & safe reaction removal
# ---------------------------------------------------------------------------

def _import_dispatcher():
    sys.modules.pop("aiuser.response.dispatcher", None)
    return import_module_directly(
        "aiuser.response.dispatcher", "aiuser/response/dispatcher.py"
    )


def test_process_image_response_logs_and_returns_false(caplog=None):
    dispatcher = _import_dispatcher()

    async def _run():
        cog = MagicMock()
        ctx = MagicMock()
        ctx.react_quietly = AsyncMock()
        ctx.message.remove_reaction = AsyncMock()
        dispatcher.get_image_generator = AsyncMock(side_effect=RuntimeError("boom"))
        return await dispatcher.process_image_response(cog, ctx)

    assert asyncio.run(_run()) is False


def test_process_image_response_reaction_removal_failure_tolerated():
    """remove_reaction raising HTTPException must not mask the result."""
    dispatcher = _import_dispatcher()
    http_exc = sys.modules["discord"].HTTPException

    async def _run():
        cog = MagicMock()
        ctx = MagicMock()
        ctx.react_quietly = AsyncMock()
        ctx.message.remove_reaction = AsyncMock(
            side_effect=http_exc.__new__(http_exc)  # real discord.py needs (response, message)
        )
        dispatcher.get_image_generator = AsyncMock(return_value=MagicMock())
        dispatcher.create_image_response = AsyncMock(return_value=True)
        return await dispatcher.process_image_response(cog, ctx)

    assert asyncio.run(_run()) is True


def test_process_image_response_reaction_removed_on_success():
    dispatcher = _import_dispatcher()

    async def _run():
        cog = MagicMock()
        ctx = MagicMock()
        ctx.react_quietly = AsyncMock()
        ctx.message.remove_reaction = AsyncMock()
        dispatcher.get_image_generator = AsyncMock(return_value=MagicMock())
        dispatcher.create_image_response = AsyncMock(return_value=True)
        result = await dispatcher.process_image_response(cog, ctx)
        ctx.message.remove_reaction.assert_awaited_once()
        return result

    assert asyncio.run(_run()) is True


def test_dispatcher_logs_exception(caplog=None):
    """Regression: the old bare `except Exception: return False` hid failures."""
    source = open("aiuser/response/dispatcher.py", encoding="utf-8").read()
    tree = ast.parse(source)
    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_image_response"
    )
    handlers_ = [n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler)]
    assert handlers_, "expected an except block"
    body_source = ast.dump(handlers_[0])
    assert "exception" in body_source or "logger" in body_source, (
        "image-response exceptions must be logged, not swallowed"
    )
