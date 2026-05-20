"""Tests for multimodal tool result support (Gemini 3.5 Flash alignment).

Verifies that generated images are embedded inside tool result messages
(multimodal function responses) rather than injected as separate assistant
messages — per Gemini 3.5 Flash guidance:

    "include multimodal content inside the function response, not outside it."

NOTE: The ``_collect_images_from_tool`` tests below replicate the same logic
from ``LLMPipeline._collect_images_from_tool`` as a standalone function,
avoiding the need to import the full llm_pipeline module (which has heavy
transitive dependencies: openai, tenacity, httpx, etc.).
"""

import sys
from types import ModuleType
from typing import List, Union
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight mock infrastructure (mirrors mock_importer approach)
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


# Pre-register packages so imports don't trigger real __init__.py
for _pkg in [
    "aiuser", "aiuser.types", "aiuser.config", "aiuser.messages_list",
    "aiuser.messages_list.converter", "aiuser.messages_list.converter.image",
    "aiuser.messages_list.converter.embed", "aiuser.functions",
    "aiuser.functions.openrouter", "aiuser.functions.generate_image",
    "aiuser.functions.edit_image", "aiuser.utils",
    "redbot", "redbot.core", "redbot.core.utils",
    "redbot.core.utils.chat_formatting",
    "discord",
]:
    _ensure_package(_pkg)

# Mock redbot and discord
sys.modules["redbot.core"] = MagicMock()
sys.modules["redbot.core.utils"] = MagicMock()
sys.modules["redbot.core.utils.chat_formatting"] = MagicMock()
sys.modules["discord"] = MagicMock()

# ---------------------------------------------------------------------------
# Stub classes for GenerateImageToolCall / EditImageToolCall.
#
# The real modules (generate_image/tool_call.py, edit_image/tool_call.py)
# have heavy transitive dependencies (openai, aiohttp, discord) that are
# impractical to mock for isolated unit tests.  Instead, we create minimal
# stub classes and register them in sys.modules so that isinstance() checks
# in the replica of _collect_images_from_tool work correctly.
# ---------------------------------------------------------------------------

class _StubGenerateImageToolCall:
    """Stub for aiuser.functions.generate_image.tool_call.GenerateImageToolCall."""
    function_name: str = "generate_image"

class _StubEditImageToolCall:
    """Stub for aiuser.functions.edit_image.tool_call.EditImageToolCall."""
    function_name: str = "edit_image"

_gen_tc_mod = ModuleType("aiuser.functions.generate_image.tool_call")
_gen_tc_mod.GenerateImageToolCall = _StubGenerateImageToolCall
_gen_tc_mod.__package__ = "aiuser.functions.generate_image"
sys.modules["aiuser.functions.generate_image.tool_call"] = _gen_tc_mod

_edit_tc_mod = ModuleType("aiuser.functions.edit_image.tool_call")
_edit_tc_mod.EditImageToolCall = _StubEditImageToolCall
_edit_tc_mod.__package__ = "aiuser.functions.edit_image"
sys.modules["aiuser.functions.edit_image.tool_call"] = _edit_tc_mod

# Re-export for convenience
GenerateImageToolCall = _StubGenerateImageToolCall
EditImageToolCall = _StubEditImageToolCall


# ---------------------------------------------------------------------------
# Standalone replica of LLMPipeline._collect_images_from_tool for testing.
#
# This mirrors the exact logic in aiuser/response/chat/llm_pipeline.py
# without requiring the full module import (which has heavy dependencies).
# ---------------------------------------------------------------------------

def collect_images_from_tool(enabled_tools: list, tool_function_name: str) -> list:
    """Replica of LLMPipeline._collect_images_from_tool for isolated testing."""
    for tool_obj in enabled_tools:
        if tool_obj.function_name == tool_function_name and (
            isinstance(tool_obj, GenerateImageToolCall) or isinstance(tool_obj, EditImageToolCall)
        ):
            return tool_obj.get_generated_images()
    return []

# Load the real modules under test
import importlib.util


def _load_module(qualified_name: str, filepath: str) -> ModuleType:
    parts = qualified_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        _ensure_package(parent)
    spec = importlib.util.spec_from_file_location(qualified_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod


_entry_mod = _load_module("aiuser.messages_list.entry", "aiuser/messages_list/entry.py")
MessageEntry = _entry_mod.MessageEntry

# Mock tiktoken before loading messages.py
_tiktoken_mock = MagicMock()
_encoding_mock = MagicMock()
_encoding_mock.encode = MagicMock(return_value=[1, 2, 3])  # 3 tokens per call
_tiktoken_mock.encoding_for_model = MagicMock(return_value=_encoding_mock)
sys.modules["tiktoken"] = _tiktoken_mock

# Mock aiuser sub-modules that messages.py imports
sys.modules["aiuser.config.defaults"] = MagicMock()
sys.modules["aiuser.config.models"] = MagicMock()
sys.modules["aiuser.messages_list.converter"] = MagicMock()
sys.modules["aiuser.messages_list.converter.converter"] = MagicMock()
sys.modules["aiuser.messages_list.opt_view"] = MagicMock()
sys.modules["aiuser.types.abc"] = MagicMock()
sys.modules["aiuser.types.enums"] = MagicMock()
sys.modules["aiuser.utils.utilities"] = MagicMock()
sys.modules["aiuser.config.constants"] = MagicMock()

_messages_mod = _load_module("aiuser.messages_list.messages", "aiuser/messages_list/messages.py")
MessagesList = _messages_mod.MessagesList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages_list(token_limit: int = 999999) -> MessagesList:
    """Create a MessagesList with mocked dependencies for unit testing."""
    cog = MagicMock()
    cog.bot = MagicMock()
    cog.cached_messages = {}
    cog.override_prompt_start_time = {}
    cog.ignore_regex = {}
    cog.config = MagicMock()
    ctx = MagicMock()
    ctx.guild = MagicMock()
    ctx.message = MagicMock()
    ctx.message.id = 12345
    ctx.message.author = MagicMock()
    ctx.message.author.id = 111
    ctx.message.author.bot = False
    ctx.message.author.roles = []
    ctx.message.channel = MagicMock()
    ctx.message.channel.id = 999
    ctx.bot = MagicMock()
    ctx.me = MagicMock()
    ml = MessagesList(cog, ctx)
    ml.model = "google/gemini-3.5-flash"
    ml.token_limit = token_limit
    ml.tokens = 0
    ml._encoding = _encoding_mock
    return ml


SAMPLE_IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def _make_image_part(url: str = SAMPLE_IMAGE_URL) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _make_text_part(text: str) -> dict:
    return {"type": "text", "text": text}


# ===================================================================
# Tests for MessageEntry
# ===================================================================

class TestMessageEntryMultimodal:
    """MessageEntry should accept both str and list content."""

    def test_string_content(self):
        entry = MessageEntry("tool", "result text", tool_call_id="tc_1", name="my_func")
        assert entry.content == "result text"
        assert entry.role == "tool"
        assert entry.tool_call_id == "tc_1"
        assert entry.name == "my_func"

    def test_list_content(self):
        parts = [_make_image_part(), _make_text_part("ok")]
        entry = MessageEntry("tool", parts, tool_call_id="tc_2", name="gen_img")
        assert entry.content == parts
        assert isinstance(entry.content, list)
        assert len(entry.content) == 2

    def test_frozen(self):
        entry = MessageEntry("tool", "text", tool_call_id="tc_3")
        with pytest.raises(AttributeError):
            entry.content = "changed"


# ===================================================================
# Tests for MessagesList.add_tool_result()
# ===================================================================

class TestAddToolResult:
    """add_tool_result should handle both string and list content."""

    @pytest.mark.asyncio
    async def test_string_content(self):
        ml = _make_messages_list()
        await ml.add_tool_result(
            content="Image generation successful.",
            tool_call_id="tc_1",
            name="generate_image",
            index=1,
        )
        assert len(ml.messages) == 1
        entry = ml.messages[0]
        assert entry.role == "tool"
        assert entry.content == "Image generation successful."
        assert entry.tool_call_id == "tc_1"
        assert entry.name == "generate_image"
        assert ml.tokens > 0

    @pytest.mark.asyncio
    async def test_list_content_with_images_and_text(self):
        ml = _make_messages_list()
        content_parts = [
            _make_image_part("data:image/png;base64,abc123"),
            _make_image_part("data:image/jpeg;base64,def456"),
            _make_text_part("Generated 2 images."),
        ]
        await ml.add_tool_result(
            content=content_parts,
            tool_call_id="tc_2",
            name="generate_image",
            index=1,
        )
        assert len(ml.messages) == 1
        entry = ml.messages[0]
        assert entry.role == "tool"
        assert isinstance(entry.content, list)
        assert len(entry.content) == 3
        assert entry.content[0]["type"] == "image_url"
        assert entry.content[1]["type"] == "image_url"
        assert entry.content[2]["type"] == "text"
        assert entry.content[2]["text"] == "Generated 2 images."

    @pytest.mark.asyncio
    async def test_list_content_token_counting(self):
        ml = _make_messages_list()
        content_parts = [
            _make_image_part("data:image/png;base64,abc"),
            _make_text_part("some text"),
        ]
        initial_tokens = ml.tokens
        await ml.add_tool_result(
            content=content_parts,
            tool_call_id="tc_3",
            name="gen",
            index=1,
        )
        # 1 image = 756 tokens, text = 3 tokens (mock encode returns [1,2,3])
        assert ml.tokens == initial_tokens + 756 + 3

    @pytest.mark.asyncio
    async def test_list_content_multiple_images_token_counting(self):
        ml = _make_messages_list()
        content_parts = [
            _make_image_part("data:image/png;base64,aaa"),
            _make_image_part("data:image/png;base64,bbb"),
            _make_image_part("data:image/png;base64,ccc"),
            _make_text_part("Three images."),
        ]
        initial_tokens = ml.tokens
        await ml.add_tool_result(
            content=content_parts,
            tool_call_id="tc_4",
            name="gen",
            index=1,
        )
        # 3 images = 3 * 756 = 2268 tokens, text = 3 tokens
        assert ml.tokens == initial_tokens + 3 * 756 + 3

    @pytest.mark.asyncio
    async def test_string_content_token_counting(self):
        ml = _make_messages_list()
        initial_tokens = ml.tokens
        await ml.add_tool_result(
            content="Plain text result",
            tool_call_id="tc_5",
            name="func",
            index=1,
        )
        # Only text tokens counted (mock encode returns [1,2,3] = 3 tokens)
        assert ml.tokens == initial_tokens + 3

    @pytest.mark.asyncio
    async def test_skips_when_over_token_limit(self):
        ml = _make_messages_list(token_limit=0)
        ml.tokens = 100  # over limit
        await ml.add_tool_result(
            content="should be skipped",
            tool_call_id="tc_6",
            name="func",
            index=1,
        )
        assert len(ml.messages) == 0


# ===================================================================
# Tests for MessagesList.get_json() with multimodal tool results
# ===================================================================

class TestGetJsonMultimodalToolResult:
    """get_json() should serialize multimodal tool results correctly."""

    @pytest.mark.asyncio
    async def test_string_tool_result_serialization(self):
        ml = _make_messages_list()
        await ml.add_tool_result(
            content="Image generated.",
            tool_call_id="tc_1",
            name="generate_image",
            index=1,
        )
        result = ml.get_json()
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["content"] == "Image generated."
        assert msg["tool_call_id"] == "tc_1"
        assert msg["name"] == "generate_image"

    @pytest.mark.asyncio
    async def test_multimodal_tool_result_serialization(self):
        ml = _make_messages_list()
        content_parts = [
            _make_image_part("data:image/png;base64,abc"),
            _make_text_part("Generated 1 image."),
        ]
        await ml.add_tool_result(
            content=content_parts,
            tool_call_id="tc_2",
            name="generate_image",
            index=1,
        )
        result = ml.get_json()
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][0]["image_url"]["url"] == "data:image/png;base64,abc"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Generated 1 image."
        assert msg["tool_call_id"] == "tc_2"
        assert msg["name"] == "generate_image"

    @pytest.mark.asyncio
    async def test_multimodal_tool_result_in_full_conversation(self):
        """A multimodal tool result should serialize correctly within a
        multi-message conversation (system + user + assistant + tool)."""
        ml = _make_messages_list()
        await ml.add_system("You are a helpful bot.", index=1)
        await ml.add_msg_result_simulation("user", "Generate an image of a cat", index=2)
        await ml.add_assistant(
            content=None,
            tool_calls=[{"id": "tc_2", "type": "function", "function": {"name": "generate_image", "arguments": '{"prompt":"a cat"}'}}],
            index=3,
        )
        content_parts = [
            _make_image_part("data:image/png;base64,catdata"),
            _make_text_part("Image generated successfully."),
        ]
        await ml.add_tool_result(
            content=content_parts,
            tool_call_id="tc_2",
            name="generate_image",
            index=4,
        )
        result = ml.get_json()
        # Verify the tool message is at position 3 (0-indexed)
        tool_msg = result[3]
        assert tool_msg["role"] == "tool"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "image_url"
        assert tool_msg["content"][1]["type"] == "text"
        # Verify no separate assistant message with images exists after tool result
        assert len(result) == 4


# ===================================================================
# Tests for collect_images_from_tool() logic
# (Replicates LLMPipeline._collect_images_from_tool)
# ===================================================================

class TestCollectImagesFromTool:
    """collect_images_from_tool should collect images from image-producing tools."""

    def test_returns_empty_for_unknown_tool(self):
        result = collect_images_from_tool([], "some_tool")
        assert result == []

    def test_returns_empty_for_non_image_tool(self):
        non_image_tool = MagicMock()
        non_image_tool.function_name = "weather_query"
        # Not an instance of GenerateImageToolCall or EditImageToolCall
        result = collect_images_from_tool([non_image_tool], "weather_query")
        assert result == []

    def test_collects_images_from_generate_image_tool(self):
        gen_tool = MagicMock(spec=GenerateImageToolCall)
        gen_tool.function_name = "generate_image"
        gen_tool.get_generated_images = MagicMock(
            return_value=[{"bytes": b"png", "format": "png", "filename": "img.png", "data_url": SAMPLE_IMAGE_URL}]
        )
        result = collect_images_from_tool([gen_tool], "generate_image")
        assert len(result) == 1
        assert result[0]["data_url"] == SAMPLE_IMAGE_URL
        gen_tool.get_generated_images.assert_called_once()

    def test_collects_images_from_edit_image_tool(self):
        edit_tool = MagicMock(spec=EditImageToolCall)
        edit_tool.function_name = "edit_image"
        edit_tool.get_generated_images = MagicMock(
            return_value=[
                {"bytes": b"png1", "format": "png", "filename": "edit1.png", "data_url": "data:image/png;base64,aaa"},
                {"bytes": b"png2", "format": "png", "filename": "edit2.png", "data_url": "data:image/png;base64,bbb"},
            ]
        )
        result = collect_images_from_tool([edit_tool], "edit_image")
        assert len(result) == 2
        assert result[0]["data_url"] == "data:image/png;base64,aaa"
        assert result[1]["data_url"] == "data:image/png;base64,bbb"

    def test_returns_empty_when_tool_has_no_images(self):
        gen_tool = MagicMock(spec=GenerateImageToolCall)
        gen_tool.function_name = "generate_image"
        gen_tool.get_generated_images = MagicMock(return_value=[])
        result = collect_images_from_tool([gen_tool], "generate_image")
        assert result == []

    def test_returns_empty_for_matching_name_but_wrong_type(self):
        """A tool with the right name but wrong class should not match."""
        wrong_type_tool = MagicMock()  # Not a GenerateImageToolCall or EditImageToolCall
        wrong_type_tool.function_name = "generate_image"
        wrong_type_tool.get_generated_images = MagicMock(return_value=[{"data_url": "x"}])
        result = collect_images_from_tool([wrong_type_tool], "generate_image")
        assert result == []

    def test_multiple_tools_only_collects_from_matching(self):
        """Only the image tool's images should be collected, not other tools."""
        gen_tool = MagicMock(spec=GenerateImageToolCall)
        gen_tool.function_name = "generate_image"
        gen_tool.get_generated_images = MagicMock(
            return_value=[{"bytes": b"img", "format": "png", "filename": "out.png", "data_url": SAMPLE_IMAGE_URL}]
        )
        weather_tool = MagicMock()
        weather_tool.function_name = "weather"
        result = collect_images_from_tool([weather_tool, gen_tool], "generate_image")
        assert len(result) == 1
        assert result[0]["data_url"] == SAMPLE_IMAGE_URL


# ===================================================================
# Integration-style: verify images embedded in tool result (not assistant)
# ===================================================================

class TestMultimodalToolResultIntegration:
    """Verify the overall pattern: images go into tool result content,
    not as a separate assistant message afterward."""

    @pytest.mark.asyncio
    async def test_conversation_structure_with_multimodal_tool_result(self):
        """After a generate_image tool call, the conversation should be:
        [system] [user] [assistant w/ tool_calls] [tool w/ images+text]
        NOT:
        [system] [user] [assistant w/ tool_calls] [tool text-only] [assistant images]
        """
        ml = _make_messages_list()

        # System prompt
        await ml.add_system("You are a bot.", index=1)
        # User message
        await ml.add_msg_result_simulation("user", "Draw me a cat", index=2)
        # Assistant with tool_calls
        await ml.add_assistant(
            content=None,
            tool_calls=[{"id": "call_abc", "type": "function", "function": {"name": "generate_image", "arguments": '{"prompt":"cute cat"}'}}],
            index=3,
        )
        # Tool result with embedded images (the new pattern)
        multimodal_content = [
            _make_image_part("data:image/png;base64,catimage"),
            _make_text_part("Image generation successful. Generated 1 image(s)."),
        ]
        await ml.add_tool_result(
            content=multimodal_content,
            tool_call_id="call_abc",
            name="generate_image",
            index=4,
        )

        result = ml.get_json()
        assert len(result) == 4

        # Verify structure
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["tool_calls"][0]["id"] == "call_abc"
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_abc"
        assert result[3]["name"] == "generate_image"

        # The tool result content is multimodal (list)
        assert isinstance(result[3]["content"], list)
        assert result[3]["content"][0]["type"] == "image_url"
        assert result[3]["content"][1]["type"] == "text"
        assert "Image generation successful" in result[3]["content"][1]["text"]

        # There is NO separate assistant message after the tool result
        # containing images. The images are INSIDE the tool result.
        for msg in result:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                # Only the tool_calls assistant should have list content
                # (which would be tool_calls, not image parts)
                pytest.fail(
                    "Found a separate assistant message with list content — "
                    "images should be in the tool result, not a separate assistant message."
                )

    @pytest.mark.asyncio
    async def test_plain_tool_result_still_works(self):
        """Non-image tool results should remain plain strings."""
        ml = _make_messages_list()
        await ml.add_system("You are a bot.", index=1)
        await ml.add_msg_result_simulation("user", "What's the weather?", index=2)
        await ml.add_assistant(
            content=None,
            tool_calls=[{"id": "call_xyz", "type": "function", "function": {"name": "weather", "arguments": '{"city":"London"}'}}],
            index=3,
        )
        await ml.add_tool_result(
            content="The weather in London is 15°C and cloudy.",
            tool_call_id="call_xyz",
            name="weather",
            index=4,
        )
        result = ml.get_json()
        assert result[3]["role"] == "tool"
        assert isinstance(result[3]["content"], str)
        assert "15°C" in result[3]["content"]


# ===================================================================
# Tests for cache_control on system messages
# ===================================================================

class TestCacheControlSystemMessage:
    """get_json() should add cache_control to system messages for prompt caching."""

    @pytest.mark.asyncio
    async def test_system_message_has_cache_control(self):
        ml = _make_messages_list()
        await ml.add_system("You are a helpful assistant.", index=1)
        result = ml.get_json()
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "system"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        part = msg["content"][0]
        assert part["type"] == "text"
        assert part["text"] == "You are a helpful assistant."
        assert part["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_system_message_content_wrapped_in_list(self):
        """System content should be converted from str to list-of-parts."""
        ml = _make_messages_list()
        await ml.add_system("System prompt here.", index=1)
        result = ml.get_json()
        # Original string content is now wrapped in a content-parts array
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["text"] == "System prompt here."

    @pytest.mark.asyncio
    async def test_non_system_messages_unchanged(self):
        """User and assistant messages should NOT get cache_control."""
        ml = _make_messages_list()
        await ml.add_system("System.", index=1)
        await ml.add_msg_result_simulation("user", "Hello", index=2)
        await ml.add_assistant(content="Hi there!", index=3)
        result = ml.get_json()
        # System has cache_control
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # User and assistant do NOT
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], str)
        assert result[2]["role"] == "assistant"
        assert isinstance(result[2]["content"], str)

    @pytest.mark.asyncio
    async def test_only_last_contiguous_system_gets_cache_control(self):
        """Only the last system message in the leading contiguous block gets cache_control."""
        ml = _make_messages_list()
        # Insert in order: [system1, system2] at the start
        await ml.add_system("First system prompt.", index=1)
        await ml.add_system("Second system prompt.", index=1)
        result = ml.get_json()
        # The first system message (index 0) should NOT have cache_control
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        # The second system message (index 1, last contiguous) SHOULD have it
        assert result[1]["role"] == "system"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_later_system_messages_no_cache_control(self):
        """System messages after non-system messages should NOT get cache_control."""
        ml = _make_messages_list()
        await ml.add_system("System prompt.", index=1)
        await ml.add_msg_result_simulation("user", "Hello", index=2)
        await ml.add_system("Late system injection.", index=3)
        result = ml.get_json()
        # First system (index 0) gets cache_control
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Late system (index 2) does NOT
        assert result[2]["role"] == "system"
        assert isinstance(result[2]["content"], str)


# ---------------------------------------------------------------------------
# Helper to add "raw" user/assistant entries directly (bypassing Discord
# message conversion) for test setup.
# ---------------------------------------------------------------------------

# Monkey-patch a convenience method onto MessagesList for test setup only.
async def _add_msg_result_simulation(self, role: str, content: str, index: int = None):
    """Add a simple message entry directly (bypasses Discord conversion)."""
    entry = MessageEntry(role, content)
    self.messages.insert(index or len(self.messages), entry)
    await self._add_tokens(content)

MessagesList.add_msg_result_simulation = _add_msg_result_simulation
