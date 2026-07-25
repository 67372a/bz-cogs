"""Tests for Gemini image model final-image-only filtering.

Gemini image models (e.g. google/gemini-3-pro-image,
google/gemini-3.1-flash-image, google/gemini-3.1-flash-lite-image) can
return multiple draft images before the final one.  Both
``GenerateImageToolCall`` and ``EditImageToolCall`` should keep only the
final image for these models, while non-Gemini models keep all images.
"""

import base64
import sys
import types as _types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — mirrors tests/test_generate_image_dedup.py
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
_params_holder = SimpleNamespace(model=None)

_or_types = _make_mock_package("aiuser.types")
_or_types_mod = _make_mock_package("aiuser.types.openrouter_types")
_or_types_mod.DirectImageGenerationParameters = MagicMock
_or_types_mod.DirectImageEditParameters = MagicMock


def _deserialize_parameters(params_json, kind):
    return SimpleNamespace(
        model=_params_holder.model,
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

# --- Real image_processing (compute_pixel_hash must behave realistically) ---
_real_image_processing = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
_inject_mock("aiuser.utils.image_processing", _real_image_processing)

# --- Real constants + utilities (provides is_gemini_image_model) ---
_inject_mock("aiuser.config", _make_mock_package("aiuser.config"))
_real_constants = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
_inject_mock("aiuser.config.constants", _real_constants)
_real_utilities = import_module_directly(
    "aiuser.utils.utilities", "aiuser/utils/utilities.py"
)
_inject_mock("aiuser.utils.utilities", _real_utilities)

is_gemini_image_model = _real_utilities.is_gemini_image_model

# aiohttp may or may not be installed; mock it safely
if "aiohttp" not in sys.modules:
    _inject_mock("aiohttp")

# Ensure parent packages exist for the modules under test
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

# Load the modules under test
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


# Three distinct "images" (raw bytes need not be valid images — the pixel
# hash helper returns None on decode failure and byte-hash dedup suffices).
_DRAFT_1 = _make_data_url(b"\x89PNG_draft_one" + b"\x00" * 50)
_DRAFT_2 = _make_data_url(b"\x89PNG_draft_two" + b"\x01" * 50)
_FINAL = _make_data_url(b"\x89PNG_final_image" + b"\x02" * 50)


def _make_response_with_images(image_urls):
    message = MagicMock()
    message.model_extra = {
        "images": [
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        ]
    }
    message.content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_config():
    """Config mock whose guild(...) accessor returns AsyncMock attributes."""
    guild_accessor = MagicMock()
    guild_accessor.direct_image_generation_parameters = AsyncMock(return_value="{}")
    guild_accessor.direct_image_edit_parameters = AsyncMock(return_value="{}")
    guild_accessor.direct_image_generation_system_prompt = AsyncMock(return_value=None)
    guild_accessor.direct_image_edit_system_prompt = AsyncMock(return_value=None)
    guild_accessor.service_tier = AsyncMock(return_value=None)
    config = MagicMock()
    config.guild = MagicMock(return_value=guild_accessor)
    return config


def _make_ctx():
    ctx = MagicMock()
    ctx.message.id = 123456789
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    return ctx


def _make_bot_with_client(response):
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    cog = MagicMock()
    cog.openai_client = client
    bot = MagicMock()
    bot.get_cog = MagicMock(return_value=cog)
    return bot


def _make_generate_tool(response) -> GenerateImageToolCall:
    tool = GenerateImageToolCall.__new__(GenerateImageToolCall)
    tool.config = _make_config()
    tool.ctx = _make_ctx()
    tool.bot = _make_bot_with_client(response)
    tool._cog = None
    tool.generated_images = []
    return tool


def _make_edit_tool(response) -> EditImageToolCall:
    tool = EditImageToolCall.__new__(EditImageToolCall)
    tool.config = _make_config()
    tool.ctx = _make_ctx()
    tool.bot = _make_bot_with_client(response)
    tool._cog = None
    tool.generated_images = []
    # Bypass Discord resolution / downloads
    tool._resolve_single_reference = AsyncMock(return_value="https://example.com/src.png")
    tool._download_and_encode_images = AsyncMock(
        return_value=[{"type": "image_url", "image_url": {"url": _DRAFT_1}}]
    )
    return tool


# ---------------------------------------------------------------------------
# Unit tests: is_gemini_image_model
# ---------------------------------------------------------------------------


class TestIsGeminiImageModel:
    @pytest.mark.parametrize(
        "model",
        [
            "google/gemini-3-pro-image",
            "google/gemini-3.1-flash-image",
            "google/gemini-3.1-flash-lite-image",
            "google/gemini-2.5-flash-image-preview",
            "google/gemini-3.1-flash-image-preview",
            "google/gemini-9.9-ultra-image",  # hypothetical future model
            "GOOGLE/GEMINI-3-PRO-IMAGE",  # case-insensitive
        ],
    )
    def test_gemini_image_models_match(self, model):
        assert is_gemini_image_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "google/gemini-3.5-flash",
            "google/gemini-2.5-pro",
            "google/gemini-pro-vision",
            "openai/gpt-5-image",
            "openai/dall-e-3",
            "black-forest-labs/flux-1.1-pro",
            "",
            None,
        ],
    )
    def test_non_gemini_image_models_do_not_match(self, model):
        assert is_gemini_image_model(model) is False


# ---------------------------------------------------------------------------
# Functional tests: generate_image keeps only the final Gemini image
# ---------------------------------------------------------------------------


class TestGenerateImageGeminiFiltering:
    @pytest.mark.asyncio
    async def test_gemini_model_keeps_only_final_image(self):
        _params_holder.model = "google/gemini-3.1-flash-image"
        response = _make_response_with_images([_DRAFT_1, _DRAFT_2, _FINAL])
        tool = _make_generate_tool(response)

        result = await tool._handle({"prompt": "a cat"})

        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["data_url"] == _FINAL
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_gemini_pro_image_keeps_only_final_image(self):
        _params_holder.model = "google/gemini-3-pro-image"
        response = _make_response_with_images([_DRAFT_1, _FINAL])
        tool = _make_generate_tool(response)

        await tool._handle({"prompt": "a dog"})

        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["data_url"] == _FINAL

    @pytest.mark.asyncio
    async def test_non_gemini_model_keeps_all_images(self):
        _params_holder.model = "openai/gpt-5-image"
        response = _make_response_with_images([_DRAFT_1, _DRAFT_2, _FINAL])
        tool = _make_generate_tool(response)

        await tool._handle({"prompt": "a cat"})

        assert len(tool.generated_images) == 3

    @pytest.mark.asyncio
    async def test_gemini_model_single_image_unaffected(self):
        _params_holder.model = "google/gemini-3.1-flash-lite-image"
        response = _make_response_with_images([_FINAL])
        tool = _make_generate_tool(response)

        await tool._handle({"prompt": "a bird"})

        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["data_url"] == _FINAL


# ---------------------------------------------------------------------------
# Functional tests: edit_image keeps only the final Gemini image
# ---------------------------------------------------------------------------


class TestEditImageGeminiFiltering:
    @pytest.mark.asyncio
    async def test_gemini_model_keeps_only_final_image(self):
        _params_holder.model = "google/gemini-3.1-flash-image"
        response = _make_response_with_images([_DRAFT_1, _DRAFT_2, _FINAL])
        tool = _make_edit_tool(response)

        result = await tool._handle(
            {"prompt": "make the sky purple", "image_to_edit": "123456789"}
        )

        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["data_url"] == _FINAL
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_non_gemini_model_keeps_all_images(self):
        _params_holder.model = "black-forest-labs/flux-1.1-pro"
        response = _make_response_with_images([_DRAFT_1, _DRAFT_2, _FINAL])
        tool = _make_edit_tool(response)

        await tool._handle(
            {"prompt": "make the sky purple", "image_to_edit": "123456789"}
        )

        assert len(tool.generated_images) == 3
