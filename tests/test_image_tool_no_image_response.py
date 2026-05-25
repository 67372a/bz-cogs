"""Tests for GenerateImageToolCall and EditImageToolCall no-image response handling.

Verifies that when the image generation/editing API returns a response without images:
- If the model provides text content, it is included in the error message along with finish_reason.
- If the model provides no text content, the message indicates no text response along with finish_reason.
- References to 'model not supporting image generation/editing' are NOT present.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Lightweight mock infrastructure (same pattern as test_tool_loop.py)
# ---------------------------------------------------------------------------

def _make_mock_package(name: str) -> ModuleType:
    pkg = ModuleType(name)
    pkg.__path__ = []
    pkg.__package__ = name
    pkg.__file__ = f"<mocked-{name}>"
    return pkg


def _ensure_package(name: str):
    if name not in sys.modules:
        sys.modules[name] = _make_mock_package(name)


# Pre-register packages
for _pkg in [
    "aiuser", "aiuser.types", "aiuser.config", "aiuser.messages_list",
    "aiuser.messages_list.converter", "aiuser.messages_list.converter.image",
    "aiuser.messages_list.converter.embed", "aiuser.functions",
    "aiuser.functions.openrouter", "aiuser.functions.generate_image",
    "aiuser.functions.edit_image", "aiuser.functions.attach_files",
    "aiuser.utils", "aiuser.response",
    "aiuser.response.chat",
    "redbot", "redbot.core", "redbot.core.utils",
    "redbot.core.utils.chat_formatting",
    "discord",
]:
    _ensure_package(_pkg)

# Mock discord
discord_mock = MagicMock()
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
sys.modules["discord"] = discord_mock
sys.modules["discord.ext"] = MagicMock()
sys.modules["discord.ext.commands"] = MagicMock()
sys.modules["discord.app_commands"] = MagicMock()

# Mock redbot
sys.modules["redbot"] = MagicMock()
sys.modules["redbot.core"] = MagicMock()
sys.modules["redbot.core"].Config = MagicMock()
sys.modules["redbot.core"].commands = MagicMock()
sys.modules["redbot.core.utils"] = MagicMock()
sys.modules["redbot.core.utils.chat_formatting"] = MagicMock()

# Mock openai with real exception types
class _RateLimitError(Exception):
    pass

_openai_mock = MagicMock()
_openai_mock.AsyncOpenAI = MagicMock()
_openai_mock.RateLimitError = _RateLimitError
sys.modules["openai"] = _openai_mock

# Mock aiohttp
sys.modules["aiohttp"] = MagicMock()

# Now import the actual modules under test
import importlib.util
import os

def _load_module(qualified_name: str, filepath: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(qualified_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load dependencies first
_load_module("aiuser.types.enums", "aiuser/types/enums.py")
_load_module("aiuser.types.openrouter_types", "aiuser/types/openrouter_types.py")
_load_module("aiuser.functions.types", "aiuser/functions/types.py")
_load_module("aiuser.functions.tool_call", "aiuser/functions/tool_call.py")
_load_module("aiuser.utils.image_cache", "aiuser/utils/image_cache.py")

# Load the modules under test
_generate_mod = _load_module(
    "aiuser.functions.generate_image.tool_call",
    "aiuser/functions/generate_image/tool_call.py",
)
_edit_mod = _load_module(
    "aiuser.functions.edit_image.tool_call",
    "aiuser/functions/edit_image/tool_call.py",
)

GenerateImageToolCall = _generate_mod.GenerateImageToolCall
EditImageToolCall = _edit_mod.EditImageToolCall


# ---------------------------------------------------------------------------
# Helpers to build mock responses
# ---------------------------------------------------------------------------

def _make_chat_response(finish_reason="stop", content=None, model_extra=None):
    """Create a mock ChatCompletion response."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = MagicMock()
    choice.message.content = content
    if model_extra is not None:
        choice.message.model_extra = model_extra
    else:
        choice.message.model_extra = {}

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_no_choices_response():
    """Create a mock ChatCompletion response with no choices."""
    response = MagicMock()
    response.choices = []
    return response


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _make_tool_instance(tool_cls):
    """Create a tool instance with mocked config/ctx/cog."""
    config = MagicMock()
    ctx = MagicMock()
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    ctx.message = MagicMock()
    ctx.message.id = 12345
    ctx.channel = MagicMock()
    ctx.me = MagicMock()
    ctx.me.id = 99999

    tool = tool_cls(config, ctx)
    # Mock the cog to provide an openai_client
    tool._cog = MagicMock()
    tool._cog.openai_client = MagicMock()
    return tool


# ===================================================================
# Tests for GenerateImageToolCall
# ===================================================================

class TestGenerateImageNoImage:
    """Tests for GenerateImageToolCall when no images are in the response."""

    @pytest.mark.asyncio
    async def test_no_image_with_text_and_finish_reason(self):
        """When model returns text but no image, message includes finish_reason and model output."""
        tool = _make_tool_instance(GenerateImageToolCall)

        response = _make_chat_response(
            finish_reason="content_filter",
            content="I cannot generate this image because it violates policy.",
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        # Mock _get_generation_params
        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_generation_params = AsyncMock(return_value=params)

        # Mock config.guild().direct_image_generation_system_prompt()
        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        result = await tool._handle({"prompt": "A cat riding a unicorn"})

        assert isinstance(result, str)
        assert "No image was generated" in result
        assert "finish_reason: content_filter" in result
        assert "I cannot generate this image because it violates policy." in result
        assert "model not support" not in result.lower()

    @pytest.mark.asyncio
    async def test_no_image_without_text(self):
        """When model returns no text and no image, message indicates no text response."""
        tool = _make_tool_instance(GenerateImageToolCall)

        response = _make_chat_response(
            finish_reason="stop",
            content=None,
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_generation_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        result = await tool._handle({"prompt": "A cat riding a unicorn"})

        assert isinstance(result, str)
        assert "No image was generated" in result
        assert "finish_reason: stop" in result
        assert "did not return any text response" in result
        assert "model not support" not in result.lower()

    @pytest.mark.asyncio
    async def test_no_image_with_empty_string_content(self):
        """When model returns empty string content and no image, message indicates no text response."""
        tool = _make_tool_instance(GenerateImageToolCall)

        response = _make_chat_response(
            finish_reason="stop",
            content="",
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_generation_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        result = await tool._handle({"prompt": "A cat riding a unicorn"})

        assert isinstance(result, str)
        assert "No image was generated" in result
        assert "finish_reason: stop" in result
        assert "did not return any text response" in result

    @pytest.mark.asyncio
    async def test_no_image_no_choices(self):
        """When response has no choices at all, finish_reason shows 'unknown'."""
        tool = _make_tool_instance(GenerateImageToolCall)

        response = _make_no_choices_response()
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_generation_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_generation_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        result = await tool._handle({"prompt": "A cat riding a unicorn"})

        assert isinstance(result, str)
        assert "No image was generated" in result
        assert "finish_reason: unknown" in result
        assert "did not return any text response" in result


# ===================================================================
# Tests for EditImageToolCall
# ===================================================================

class TestEditImageNoImage:
    """Tests for EditImageToolCall when no images are in the response."""

    @pytest.mark.asyncio
    async def test_no_image_with_text_and_finish_reason(self):
        """When model returns text but no edited image, message includes finish_reason and model output."""
        tool = _make_tool_instance(EditImageToolCall)

        response = _make_chat_response(
            finish_reason="content_filter",
            content="The requested edit cannot be performed.",
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_edit_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        # Mock the image resolution/download pipeline
        source_url = "https://example.com/image.png"
        tool._resolve_single_reference = AsyncMock(return_value=source_url)
        tool._download_and_encode_images = AsyncMock(
            return_value=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        )

        result = await tool._handle({
            "prompt": "Make the sky blue",
            "image_to_edit": "123456789",
        })

        assert isinstance(result, str)
        assert "No image was produced after editing" in result
        assert "finish_reason: content_filter" in result
        assert "The requested edit cannot be performed." in result
        assert "model not support" not in result.lower()

    @pytest.mark.asyncio
    async def test_no_image_without_text(self):
        """When model returns no text and no edited image, message indicates no text response."""
        tool = _make_tool_instance(EditImageToolCall)

        response = _make_chat_response(
            finish_reason="stop",
            content=None,
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_edit_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        source_url = "https://example.com/image.png"
        tool._resolve_single_reference = AsyncMock(return_value=source_url)
        tool._download_and_encode_images = AsyncMock(
            return_value=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        )

        result = await tool._handle({
            "prompt": "Make the sky blue",
            "image_to_edit": "123456789",
        })

        assert isinstance(result, str)
        assert "No image was produced after editing" in result
        assert "finish_reason: stop" in result
        assert "did not return any text response" in result
        assert "model not support" not in result.lower()

    @pytest.mark.asyncio
    async def test_no_image_with_empty_string_content(self):
        """When model returns empty string content and no edited image, message indicates no text response."""
        tool = _make_tool_instance(EditImageToolCall)

        response = _make_chat_response(
            finish_reason="stop",
            content="",
        )
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_edit_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        source_url = "https://example.com/image.png"
        tool._resolve_single_reference = AsyncMock(return_value=source_url)
        tool._download_and_encode_images = AsyncMock(
            return_value=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        )

        result = await tool._handle({
            "prompt": "Make the sky blue",
            "image_to_edit": "123456789",
        })

        assert isinstance(result, str)
        assert "No image was produced after editing" in result
        assert "finish_reason: stop" in result
        assert "did not return any text response" in result

    @pytest.mark.asyncio
    async def test_no_image_no_choices(self):
        """When response has no choices at all, finish_reason shows 'unknown'."""
        tool = _make_tool_instance(EditImageToolCall)

        response = _make_no_choices_response()
        tool._cog.openai_client.chat.completions.create = AsyncMock(return_value=response)

        params = MagicMock()
        params.model = "test-model"
        params.image_size = "1024x1024"
        params.reasoning_effort = None
        params.temperature = None
        params.top_p = None
        params.service_tier = None
        tool._get_edit_params = AsyncMock(return_value=params)

        tool.config.guild.return_value.direct_image_edit_system_prompt = AsyncMock(return_value=None)
        tool.config.guild.return_value.service_tier = AsyncMock(return_value=None)

        source_url = "https://example.com/image.png"
        tool._resolve_single_reference = AsyncMock(return_value=source_url)
        tool._download_and_encode_images = AsyncMock(
            return_value=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        )

        result = await tool._handle({
            "prompt": "Make the sky blue",
            "image_to_edit": "123456789",
        })

        assert isinstance(result, str)
        assert "No image was produced after editing" in result
        assert "finish_reason: unknown" in result
        assert "did not return any text response" in result
