"""Tests for image tool result content format.

Verifies that GenerateImageToolCall and EditImageToolCall return tool results
in the standard OpenAI content-part format:
    {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
rather than the Gemini-native format:
    {"image": "data:image/...;base64,..."}

The Gemini format lacks a ``type`` field, which means:
- Token estimators skip it (undercounting context size).
- Non-Gemini / strict OpenAI-compatible endpoints reject or ignore it.
"""

import base64
import sys
import types as _types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup
# ---------------------------------------------------------------------------

_injected_modules: dict = {}
_SENTINEL = object()


def _inject_mock(name: str, module=None):
    if name in _injected_modules:
        return
    _injected_modules[name] = sys.modules.get(name, _SENTINEL)
    if module is None:
        module = MagicMock()
    sys.modules[name] = module


def _restore_modules():
    for name, original in _injected_modules.items():
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _injected_modules.clear()


# --- discord stub ---
_discord_mod = _types.ModuleType("discord")
_discord_mod.Message = type("_DiscordMessage", (), {})
_discord_mod.Embed = MagicMock()
_discord_mod.Color = MagicMock()
_discord_mod.Colour = MagicMock()
_inject_mock("discord", _discord_mod)

# --- redbot stubs ---
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
_redbot_bot = _make_mock_package("redbot.core.bot")
_redbot_bot.Red = MagicMock()
_inject_mock("redbot", _redbot)
_inject_mock("redbot.core", _redbot_core)
_inject_mock("redbot.core.bot", _redbot_bot)

# --- Real types + ToolCall base ---
types_mod = import_module_directly(
    "aiuser.functions.types", "aiuser/functions/types.py"
)
_inject_mock("aiuser.functions.types", types_mod)

tool_call_base = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)
_inject_mock("aiuser.functions.tool_call", tool_call_base)

# --- openrouter_types stub with controllable deserialize_parameters ---
_or_types = _make_mock_package("aiuser.types")
_or_types_mod = _make_mock_package("aiuser.types.openrouter_types")
_or_types_mod.DirectImageGenerationParameters = MagicMock
_or_types_mod.DirectImageEditParameters = MagicMock

# Will be set per-test
_test_model = "test-model"


def _deserialize_parameters(params_json, kind):
    return SimpleNamespace(
        model=_test_model,
        image_size=None,
        reasoning_effort=None,
        temperature=None,
        top_p=None,
        service_tier=None,
    )


_or_types_mod.deserialize_parameters = _deserialize_parameters
_inject_mock("aiuser.types", _or_types)
_inject_mock("aiuser.types.openrouter_types", _or_types_mod)

# --- image cache stub ---
_image_cache_mod = _make_mock_package("aiuser.utils.image_cache")
_image_cache_mod.image_cache = MagicMock()
_image_cache_mod.image_cache.get = MagicMock(return_value=None)
_image_cache_mod.processed_image_cache = MagicMock()
_image_cache_mod.processed_image_cache.get = MagicMock(return_value=None)
_image_cache_mod.processed_image_cache.set = MagicMock(return_value=None)
_inject_mock("aiuser.utils", _make_mock_package("aiuser.utils"))
_inject_mock("aiuser.utils.image_cache", _image_cache_mod)

# --- Real image_processing ---
_real_image_processing = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
_inject_mock("aiuser.utils.image_processing", _real_image_processing)

# --- Real config constants + utilities ---
_inject_mock("aiuser.config", _make_mock_package("aiuser.config"))
_real_constants = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
_inject_mock("aiuser.config.constants", _real_constants)
_real_utilities = import_module_directly(
    "aiuser.utils.utilities", "aiuser/utils/utilities.py"
)
_inject_mock("aiuser.utils.utilities", _real_utilities)

# aiohttp may or may not be installed
if "aiohttp" not in sys.modules:
    _inject_mock("aiohttp")

# Ensure parent packages exist
_all_pkg_prefixes = set()
for _name in list(sys.modules.keys()) + [
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
]:
    _parts = _name.split(".")
    for _i in range(1, len(_parts)):
        _all_pkg_prefixes.add(".".join(_parts[:_i]))
for _pkg in _all_pkg_prefixes:
    if _pkg not in sys.modules:
        _inject_mock(_pkg, _make_mock_package(_pkg))

# Load modules under test
_gen_mod = import_module_directly(
    "aiuser.functions.generate_image.tool_call",
    "aiuser/functions/generate_image/tool_call.py",
)
GenerateImageToolCall = _gen_mod.GenerateImageToolCall

_edit_mod = import_module_directly(
    "aiuser.functions.edit_image.tool_call",
    "aiuser/functions/edit_image/tool_call.py",
)
EditImageToolCall = _edit_mod.EditImageToolCall

_restore_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_url(image_bytes: bytes, fmt: str = "png") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


# A valid minimal PNG (1x1 white pixel)
_VALID_PNG = (
    b"\x89PNG\r\n\x1a\n"  # signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx"
    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_generate_tool_call() -> GenerateImageToolCall:
    ctx = MagicMock()
    ctx.message.id = 123456789
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    tool = GenerateImageToolCall.__new__(GenerateImageToolCall)
    tool.config = MagicMock()
    tool.ctx = ctx
    tool.bot = MagicMock()
    tool._cog = None
    tool.generated_images = []
    return tool


def _make_edit_tool_call() -> EditImageToolCall:
    ctx = MagicMock()
    ctx.message.id = 123456789
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    tool = EditImageToolCall.__new__(EditImageToolCall)
    tool.config = MagicMock()
    tool.ctx = ctx
    tool.bot = MagicMock()
    tool._cog = None
    tool.generated_images = []
    return tool


def _make_response_with_data_url(data_url: str):
    """Build a mock ChatCompletion response that returns an image via model_extra."""
    message = MagicMock()
    message.model_extra = {
        "images": [
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
    }
    message.content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateImageToolResultFormat:
    """Verify GenerateImageToolCall._handle() returns OpenAI-format content parts."""

    @pytest.mark.asyncio
    async def test_content_parts_use_openai_image_url_format(self):
        """Content parts must be {"type": "image_url", "image_url": {"url": ...}},
        NOT {"image": ...}."""
        global _test_model
        _test_model = "test-model"

        tool = _make_generate_tool_call()
        data_url = _make_data_url(_VALID_PNG)
        response = _make_response_with_data_url(data_url)

        # Mock the config methods that _handle calls
        tool.config.guild.return_value.direct_image_generation_parameters = AsyncMock(
            return_value='{"model": "test-model"}'
        )
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)
        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(
            return_value=None
        )

        # Mock the API client
        mock_cog = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        mock_cog.openai_client = mock_client
        tool._cog = mock_cog

        # Mock _create_image_completion_with_retry to return our response directly
        with patch.object(
            tool, "_create_image_completion_with_retry", new_callable=AsyncMock, return_value=response
        ):
            result = await tool._handle({"prompt": "a test image"})

        # Result must be a list of content parts
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"
        assert len(result) > 0, "Expected at least one content part"

        for part in result:
            assert isinstance(part, dict), f"Expected dict, got {type(part)}"
            # Must have "type" field
            assert "type" in part, f"Content part missing 'type' field: {part}"
            assert part["type"] == "image_url", f"Expected type='image_url', got '{part.get('type')}'"
            # Must have "image_url" field with nested "url"
            assert "image_url" in part, f"Content part missing 'image_url' field: {part}"
            assert isinstance(part["image_url"], dict), f"Expected image_url to be dict: {part}"
            assert "url" in part["image_url"], f"image_url missing 'url' field: {part}"
            assert part["image_url"]["url"].startswith("data:image/"), "URL should be a data URL"

        # Must NOT have the old Gemini format keys at top level
        for part in result:
            assert "image" not in part, (
                f"Content part uses old Gemini 'image' key (should be type+image_url): {part}"
            )

    @pytest.mark.asyncio
    async def test_content_parts_compatible_with_token_estimator(self):
        """The content parts should be compatible with MessagesList._add_content_tokens().
        Specifically, parts with type='image_url' should have their tokens estimated."""
        global _test_model
        _test_model = "test-model"

        tool = _make_generate_tool_call()
        data_url = _make_data_url(_VALID_PNG)
        response = _make_response_with_data_url(data_url)

        tool.config.guild.return_value.direct_image_generation_parameters = AsyncMock(
            return_value='{"model": "test-model"}'
        )
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)
        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(
            return_value=None
        )

        mock_cog = MagicMock()
        mock_cog.openai_client = MagicMock()
        tool._cog = mock_cog

        with patch.object(
            tool, "_create_image_completion_with_retry", new_callable=AsyncMock, return_value=response
        ):
            result = await tool._handle({"prompt": "a test image"})

        # Verify the format matches what _add_content_tokens expects
        for part in result:
            item_type = part.get("type")
            # _add_content_tokens checks for "text", "image_url", or "file"
            assert item_type in ("text", "image_url", "file"), (
                f"Content part type '{item_type}' would be skipped by token estimator. "
                f"Part: {part}"
            )


class TestEditImageToolResultFormat:
    """Verify EditImageToolCall._handle() returns OpenAI-format content parts."""

    @pytest.mark.asyncio
    async def test_content_parts_use_openai_image_url_format(self):
        """Content parts must be {"type": "image_url", "image_url": {"url": ...}},
        NOT {"image": ...}."""
        global _test_model
        _test_model = "test-model"

        tool = _make_edit_tool_call()
        data_url = _make_data_url(_VALID_PNG)
        response = _make_response_with_data_url(data_url)

        # Mock config
        tool.config.guild.return_value.direct_image_edit_parameters = AsyncMock(
            return_value='{"model": "test-model"}'
        )
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)
        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(
            return_value=None
        )

        # Mock the API client
        mock_cog = MagicMock()
        mock_cog.openai_client = MagicMock()
        tool._cog = mock_cog

        # Mock _resolve_single_reference to return a URL and _download_and_encode_images
        with patch.object(
            tool, "_resolve_single_reference", new_callable=AsyncMock, return_value="https://example.com/image.png"
        ), patch.object(
            tool, "_download_and_encode_images", new_callable=AsyncMock,
            return_value=[{"type": "image_url", "image_url": {"url": data_url}}]
        ), patch.object(
            tool, "_create_image_completion_with_retry", new_callable=AsyncMock, return_value=response
        ):
            result = await tool._handle({
                "prompt": "edit this image",
                "image_to_edit": "123456"
            })

        # Result must be a list of content parts
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"
        assert len(result) > 0, "Expected at least one content part"

        for part in result:
            assert isinstance(part, dict), f"Expected dict, got {type(part)}"
            assert "type" in part, f"Content part missing 'type' field: {part}"
            assert part["type"] == "image_url", f"Expected type='image_url', got '{part.get('type')}'"
            assert "image_url" in part, f"Content part missing 'image_url' field: {part}"
            assert isinstance(part["image_url"], dict), f"Expected image_url to be dict: {part}"
            assert "url" in part["image_url"], f"image_url missing 'url' field: {part}"
            assert part["image_url"]["url"].startswith("data:image/"), "URL should be a data URL"

        # Must NOT have the old Gemini format keys at top level
        for part in result:
            assert "image" not in part, (
                f"Content part uses old Gemini 'image' key (should be type+image_url): {part}"
            )

    @pytest.mark.asyncio
    async def test_content_parts_compatible_with_token_estimator(self):
        """The content parts should be compatible with MessagesList._add_content_tokens()."""
        global _test_model
        _test_model = "test-model"

        tool = _make_edit_tool_call()
        data_url = _make_data_url(_VALID_PNG)
        response = _make_response_with_data_url(data_url)

        tool.config.guild.return_value.direct_image_edit_parameters = AsyncMock(
            return_value='{"model": "test-model"}'
        )
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)
        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(
            return_value=None
        )

        mock_cog = MagicMock()
        mock_cog.openai_client = MagicMock()
        tool._cog = mock_cog

        with patch.object(
            tool, "_resolve_single_reference", new_callable=AsyncMock, return_value="https://example.com/image.png"
        ), patch.object(
            tool, "_download_and_encode_images", new_callable=AsyncMock,
            return_value=[{"type": "image_url", "image_url": {"url": data_url}}]
        ), patch.object(
            tool, "_create_image_completion_with_retry", new_callable=AsyncMock, return_value=response
        ):
            result = await tool._handle({
                "prompt": "edit this image",
                "image_to_edit": "123456"
            })

        for part in result:
            item_type = part.get("type")
            assert item_type in ("text", "image_url", "file"), (
                f"Content part type '{item_type}' would be skipped by token estimator. "
                f"Part: {part}"
            )
