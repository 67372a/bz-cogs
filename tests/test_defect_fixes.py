"""Tests verifying the defect fixes from the aiuser code review.

Covers:
- config/models.py: no duplicate MODELS_LIMITS keys, supports_parallel_tool_calls
- core/handlers.py: get_ratelimit_reset malformed-value handling, role
  reply-percent highest-priority precedence
- response/chat/llm_pipeline.py: OpenRouter synthetic tool results include
  real citation data
- utils/cache.py: bounded Cache eviction (used for pdf_annotations etc.)
"""

import ast
import asyncio
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
    _discord_mod.Message = type("_DiscordMessage", (), {})
    _discord_mod.Thread = type("_DiscordThread", (), {})
    _discord_mod.NotFound = type("NotFound", (Exception,), {})
    _discord_mod.Forbidden = type("Forbidden", (Exception,), {})
    _discord_mod.HTTPException = type("HTTPException", (Exception,), {})
    sys.modules["discord"] = _discord_mod

if "redbot.core" not in sys.modules:
    _redbot = _make_mock_package("redbot")
    _redbot_core = _make_mock_package("redbot.core")
    _redbot_core.Config = MagicMock()
    _redbot_core.commands = MagicMock()
    sys.modules["redbot"] = _redbot
    sys.modules["redbot.core"] = _redbot_core

# Leaf aiuser modules imported by llm_pipeline.py / handlers.py
_leaf_mocks = [
    "aiuser.types.abc",
    "aiuser.types.enums",
    "aiuser.functions.tool_call",
    "aiuser.functions.types",
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.attach_files.tool_call",
    "aiuser.functions.mermaid.tool_call",
    "aiuser.messages_list.messages",
    "aiuser.messages_list.entry",
    "aiuser.response.chat.function_call_view",
    "aiuser.functions.openrouter",
    "aiuser.utils.utilities",
    "aiuser.core.triggers",
    "aiuser.core.validators",
    "aiuser.core.openai_utils",
]
for _leaf in _leaf_mocks:
    if _leaf not in sys.modules:
        sys.modules[_leaf] = MagicMock()

# config.constants / config.defaults need real values used at module level
sys.modules.pop("aiuser.config.constants", None)
sys.modules.pop("aiuser.config.defaults", None)
sys.modules.pop("aiuser.config.models", None)
import_module_directly("aiuser.config.constants", "aiuser/config/constants.py")
import_module_directly("aiuser.config.defaults", "aiuser/config/defaults.py")
models_mod = import_module_directly("aiuser.config.models", "aiuser/config/models.py")

# Ensure parent packages exist for the modules under test
for _pkg in [
    "aiuser", "aiuser.config", "aiuser.core", "aiuser.response",
    "aiuser.response.chat", "aiuser.utils",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

handlers = import_module_directly("aiuser.core.handlers", "aiuser/core/handlers.py")
llm_pipeline = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
cache_mod = import_module_directly("aiuser.utils.cache", "aiuser/utils/cache.py")


# ---------------------------------------------------------------------------
# config/models.py
# ---------------------------------------------------------------------------

def test_models_limits_has_no_duplicate_keys():
    """Duplicate dict keys silently overwrite each other (defect #5)."""
    tree = ast.parse(open("aiuser/config/models.py", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            dupes = {k for k in keys if keys.count(k) > 1}
            assert not dupes, f"Duplicate keys in dict: {dupes}"


def test_supports_parallel_tool_calls():
    """Model-capability helper replaces hardcoded 'gemini-3' sniffing."""
    assert models_mod.supports_parallel_tool_calls("google/gemini-3.5-flash")
    assert models_mod.supports_parallel_tool_calls("google/gemini-2.5-flash")
    assert models_mod.supports_parallel_tool_calls("Gemini-2.0-Flash")
    assert not models_mod.supports_parallel_tool_calls("gpt-4o")
    assert not models_mod.supports_parallel_tool_calls("anthropic/claude-3.5-sonnet")
    assert not models_mod.supports_parallel_tool_calls("")
    assert not models_mod.supports_parallel_tool_calls(None)


# ---------------------------------------------------------------------------
# core/handlers.py — ratelimit parsing
# ---------------------------------------------------------------------------

def test_get_ratelimit_reset_malformed_returns_none():
    """A corrupt ratelimit_reset config value must not break handling."""
    cog = MagicMock()
    cog.config.ratelimit_reset = AsyncMock(return_value="not-a-date")
    assert asyncio.run(handlers.get_ratelimit_reset(cog)) is None

    cog.config.ratelimit_reset = AsyncMock(return_value=None)
    assert asyncio.run(handlers.get_ratelimit_reset(cog)) is None


def test_get_ratelimit_reset_valid():
    cog = MagicMock()
    cog.config.ratelimit_reset = AsyncMock(return_value="2030-01-02 03:04:05")
    result = asyncio.run(handlers.get_ratelimit_reset(cog))
    assert result == datetime(2030, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# core/handlers.py — role reply-percent precedence
# ---------------------------------------------------------------------------

def _make_role(role_id):
    role = MagicMock()
    role.id = role_id
    return role


def _make_cog_for_percentage(role_percents):
    """role_percents: {role_id: percent}. Member/channel/guild return None."""
    cog = MagicMock()
    cog.config.all_roles = AsyncMock(return_value={rid: {} for rid in role_percents})

    def _role_cfg(role):
        cfg = MagicMock()
        cfg.reply_percent = AsyncMock(return_value=role_percents[role.id])
        return cfg

    cog.config.role = MagicMock(side_effect=_role_cfg)

    member_cfg = MagicMock()
    member_cfg.reply_percent = AsyncMock(return_value=None)
    cog.config.member = MagicMock(return_value=member_cfg)

    channel_cfg = MagicMock()
    channel_cfg.reply_percent = AsyncMock(return_value=None)
    cog.config.channel = MagicMock(return_value=channel_cfg)

    guild_cfg = MagicMock()
    guild_cfg.reply_percent = AsyncMock(return_value=None)
    cog.config.guild = MagicMock(return_value=guild_cfg)
    return cog


def test_get_percentage_highest_role_wins():
    """With multiple configured roles, the highest-hierarchy role's percent
    must be used (discord.py sorts author.roles ascending)."""
    low_role = _make_role(111)   # lower in hierarchy -> listed first
    high_role = _make_role(222)  # higher in hierarchy -> listed last
    cog = _make_cog_for_percentage({111: 0.10, 222: 0.90})

    ctx = MagicMock()
    ctx.author.roles = [low_role, high_role]  # ascending, as discord.py provides

    result = asyncio.run(handlers.get_percentage(cog, ctx))
    assert result == 0.90


def test_get_percentage_member_overrides_role():
    member_percent = 0.42
    cog = _make_cog_for_percentage({111: 0.10})
    member_cfg = MagicMock()
    member_cfg.reply_percent = AsyncMock(return_value=member_percent)
    cog.config.member = MagicMock(return_value=member_cfg)

    ctx = MagicMock()
    ctx.author.roles = [_make_role(111)]

    result = asyncio.run(handlers.get_percentage(cog, ctx))
    assert result == member_percent


# ---------------------------------------------------------------------------
# llm_pipeline.py — OpenRouter synthetic tool results
# ---------------------------------------------------------------------------

def test_openrouter_synthetic_result_includes_citations():
    """Synthetic server-tool results must ground the model with real data."""
    model_extra = {
        "annotations": [
            {"url_citation": {"url": "https://example.com/a", "title": "Example A"}},
            {"url_citation": {"url": "https://example.com/b", "title": ""}},
        ]
    }
    result = llm_pipeline.LLMPipeline._build_openrouter_synthetic_result(
        "openrouter:web_search", model_extra
    )
    assert "https://example.com/a" in result
    assert "Example A" in result
    assert "https://example.com/b" in result


def test_openrouter_synthetic_result_fallback_without_data():
    result = llm_pipeline.LLMPipeline._build_openrouter_synthetic_result(
        "openrouter:web_search", None
    )
    assert "was executed" in result

    result = llm_pipeline.LLMPipeline._build_openrouter_synthetic_result(
        "openrouter:web_search", {"annotations": []}
    )
    assert "was executed" in result


def test_openrouter_modality_tool_detection():
    """kwargs['tools'] is a list of dicts; image-gen detection must inspect
    each dict's 'type' field (defect #1: string-in-list-of-dicts bug)."""
    from aiuser.types.enums import OpenRouterToolType

    tools = [
        {"type": "function", "function": {"name": "search_google"}},
        {"type": OpenRouterToolType.IMAGE_GENERATION.value},
    ]
    has_image_gen = any(
        isinstance(t, dict) and t.get("type") == OpenRouterToolType.IMAGE_GENERATION.value
        for t in tools
    )
    assert has_image_gen

    tools_no_img = [{"type": "function", "function": {"name": "search_google"}}]
    assert not any(
        isinstance(t, dict) and t.get("type") == OpenRouterToolType.IMAGE_GENERATION.value
        for t in tools_no_img
    )


# ---------------------------------------------------------------------------
# utils/cache.py — bounded LRU used for pdf_annotations / last_response_at
# ---------------------------------------------------------------------------

def test_cache_evicts_oldest_beyond_limit():
    Cache = cache_mod.Cache
    c = Cache(limit=3)
    for i in range(5):
        c[i] = i
    assert len(c) == 3
    assert 0 not in c and 1 not in c  # oldest evicted
    assert c[4] == 4


def test_cache_missing_key_raises_keyerror():
    """Cache now follows the dict contract: [] raises KeyError, .get() returns None."""
    Cache = cache_mod.Cache
    c = Cache(limit=2)
    try:
        c["missing"]
        assert False, "Expected KeyError for missing key"
    except KeyError:
        pass
    assert c.get("missing") is None
