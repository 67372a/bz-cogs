"""Tests for reasoning embed and file fallback changes.

Verifies that:
- _send_content_or_file sends embed for short content, file for long content
- _extract_reasoning_from_details extracts reasoning from structured details
- FunctionCallView.view_reasoning() sends embed for short, file for long reasoning
- FunctionCallView.view_outputs() sends embed for short, file for long outputs
- FunctionCallView.view_inputs() sends embed for short, file for long inputs
- ResponseView.view_reasoning() sends embed for short, file for long reasoning
- The fallback path in send_single_combined_message attaches reasoning view
"""

import asyncio
import io
import sys
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_function_call_embed.py)
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

# Create a proper Embed mock that stores kwargs as attributes
class _FakeEmbed:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def set_footer(self, **kwargs):
        return self
    def set_image(self, **kwargs):
        return self
    def set_thumbnail(self, **kwargs):
        return self
    def set_author(self, **kwargs):
        return self

discord_mock.Embed = _FakeEmbed
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
discord_mock.AllowedMentions = MagicMock()
discord_mock.HTTPException = type("HTTPException", (Exception,), {})


class _FakeFile:
    """Mock for discord.File that captures the bytes and filename."""
    def __init__(self, fp, filename=None):
        self.fp = fp
        self.filename = filename
        # Read the bytes so tests can inspect them, then seek back
        if hasattr(fp, 'read'):
            fp.seek(0)
            self.content = fp.read()
            fp.seek(0)  # Reset so tests can read fp again
        else:
            self.content = fp


discord_mock.File = _FakeFile

# Set up discord.ui mock
_ui_mock = MagicMock()
discord_mock.ui = _ui_mock


class _FakeButton:
    pass


_ui_mock.Button = _FakeButton


class _FakeView:
    def __init__(self, timeout=None):
        pass

    def remove_item(self, item):
        pass


_ui_mock.View = _FakeView

# Make discord.ui.button a pass-through decorator so async methods aren't replaced
_ui_mock.button = lambda *args, **kwargs: (lambda func: func)

discord_mock.ButtonStyle = MagicMock()
discord_mock.ButtonStyle.secondary = MagicMock()

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

# Mock other aiuser dependencies
for _m in [
    "aiuser.config.defaults", "aiuser.config.models", "aiuser.config.constants",
    "aiuser.messages_list.converter.converter", "aiuser.messages_list.opt_view",
    "aiuser.types.abc", "aiuser.types.enums", "aiuser.utils.utilities",
    "aiuser.functions.types", "aiuser.functions.tool_call",
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.attach_files.tool_call",
    "aiuser.functions.scrape.tool_call",
    "aiuser.functions.openrouter",
    "aiuser.functions.openrouter.web_search",
    "aiuser.functions.openrouter.web_fetch",
    "aiuser.functions.openrouter.image_generation",
    "aiuser.functions.openrouter.pdf_parsing",
    "aiuser.functions.openrouter.image_parsing",
    "aiuser.messages_list.messages",
    "aiuser.messages_list.entry",
    "tiktoken",
]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Set up OpenRouter mock classes (needed by llm_pipeline imports)
_or = sys.modules["aiuser.functions.openrouter"]
_or.OpenRouterWebSearch = MagicMock()
_or.OpenRouterWebFetch = MagicMock()
_or.OpenRouterImageGeneration = MagicMock()

# Mock openai with REAL exception types
_openai_mock = MagicMock()
_openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
_openai_mock.APIConnectionError = type("APIConnectionError", (Exception,), {})
_openai_mock.InternalServerError = type("InternalServerError", (Exception,), {})
_openai_mock.APIError = type("APIError", (Exception,), {})
_openai_mock.APIStatusError = type("APIStatusError", (Exception,), {})
_openai_mock.ChatCompletion = MagicMock()
_openai_mock.ChatCompletionMessageToolCall = MagicMock()
sys.modules["openai"] = _openai_mock
sys.modules["openai.types"] = MagicMock()
sys.modules["openai.types.chat"] = MagicMock()

# Mock httpx
_httpx_mock = MagicMock()
_httpx_mock.ReadTimeout = type("ReadTimeout", (Exception,), {})
sys.modules["httpx"] = _httpx_mock

# Mock tenacity
sys.modules["tenacity"] = MagicMock()

# Load real modules
from tests.mock_importer import import_module_directly

_real_fcv = import_module_directly(
    "aiuser.response.chat.function_call_view",
    "aiuser/response/chat/function_call_view.py",
)

# Mock mermaid package (needed by llm_pipeline imports)
if "aiuser.functions.mermaid" not in sys.modules:
    sys.modules["aiuser.functions.mermaid"] = MagicMock()
if "aiuser.functions.mermaid.tool_call" not in sys.modules:
    sys.modules["aiuser.functions.mermaid.tool_call"] = MagicMock()

# Ensure aiuser.utils.utilities mock has attributes needed by llm_pipeline imports
# (may be a plain MagicMock from prior test files that don't set these)
_utils_mock = sys.modules.get("aiuser.utils.utilities")
if _utils_mock is not None and not hasattr(_utils_mock, "get_enabled_tools"):
    _utils_mock.get_enabled_tools = MagicMock()

_real_pipeline = import_module_directly(
    "aiuser.response.chat.llm_pipeline",
    "aiuser/response/chat/llm_pipeline.py",
)

_send_content_or_file = _real_fcv._send_content_or_file
_extract_reasoning_from_details = _real_pipeline._extract_reasoning_from_details
EMBED_DESCRIPTION_MAX_CHARS = _real_fcv.EMBED_DESCRIPTION_MAX_CHARS


# ---------------------------------------------------------------------------
# Tests: _send_content_or_file
# ---------------------------------------------------------------------------

class TestSendContentOrFile:
    """Test that _send_content_or_file sends embed for short content, file for long."""

    def _make_interaction(self):
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        return interaction

    @pytest.mark.asyncio
    async def test_short_content_sends_embed(self):
        interaction = self._make_interaction()
        content = "Short content"
        await _send_content_or_file(
            interaction, "Test Title", content, 0x5865F2, "test.txt"
        )
        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert embed.description == content
        assert embed.title == "Test Title"
        assert call_args.kwargs.get("file") is None
        assert call_args.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_exact_limit_sends_embed(self):
        interaction = self._make_interaction()
        content = "A" * EMBED_DESCRIPTION_MAX_CHARS
        await _send_content_or_file(
            interaction, "Test Title", content, 0x5865F2, "test.txt"
        )
        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert call_args.kwargs.get("file") is None

    @pytest.mark.asyncio
    async def test_long_content_sends_file(self):
        interaction = self._make_interaction()
        content = "A" * (EMBED_DESCRIPTION_MAX_CHARS + 1)
        await _send_content_or_file(
            interaction, "Test Title", content, 0x5865F2, "test.txt"
        )
        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        # Should have an embed with size notice
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert "too large" in embed.description.lower()
        # Character count is comma-formatted (e.g. "4,097")
        expected_count = f"{EMBED_DESCRIPTION_MAX_CHARS + 1:,}"
        assert expected_count in embed.description
        # Should have a file
        file = call_args.kwargs.get("file")
        assert file is not None
        assert file.filename == "test.txt"
        assert isinstance(file.fp, io.BytesIO)
        assert call_args.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_file_contains_full_content(self):
        interaction = self._make_interaction()
        content = "Full content here " * 500  # ~9000 chars
        await _send_content_or_file(
            interaction, "Test Title", content, 0x5865F2, "test.txt"
        )
        call_args = interaction.response.send_message.call_args
        file = call_args.kwargs.get("file")
        file_content = file.fp.read().decode("utf-8")
        assert file_content == content

    @pytest.mark.asyncio
    async def test_empty_content_sends_embed(self):
        interaction = self._make_interaction()
        await _send_content_or_file(
            interaction, "Test Title", "", 0x5865F2, "test.txt"
        )
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert embed.description == ""


# ---------------------------------------------------------------------------
# Tests: _extract_reasoning_from_details
# ---------------------------------------------------------------------------

class TestExtractReasoningFromDetails:
    """Test that _extract_reasoning_from_details extracts reasoning from structured details."""

    def test_dict_with_reasoning_key(self):
        details = [{"reasoning": "Think step by step..."}]
        result = _extract_reasoning_from_details(details)
        assert result == "Think step by step..."

    def test_dict_with_text_key(self):
        details = [{"text": "Internal reasoning here"}]
        result = _extract_reasoning_from_details(details)
        assert result == "Internal reasoning here"

    def test_dict_with_summary_key(self):
        details = [{"summary": "Summary of reasoning"}]
        result = _extract_reasoning_from_details(details)
        assert result == "Summary of reasoning"

    def test_object_with_reasoning_attr(self):
        detail = MagicMock(spec=["reasoning"])
        detail.reasoning = "Object-based reasoning"
        result = _extract_reasoning_from_details([detail])
        assert result == "Object-based reasoning"

    def test_object_with_text_attr(self):
        detail = MagicMock(spec=["text"])
        detail.text = "Object text reasoning"
        result = _extract_reasoning_from_details([detail])
        assert result == "Object text reasoning"

    def test_multiple_parts_concatenated(self):
        details = [
            {"reasoning": "Part one"},
            {"reasoning": "Part two"},
        ]
        result = _extract_reasoning_from_details(details)
        assert "Part one" in result
        assert "Part two" in result
        assert result == "Part one\n\nPart two"

    def test_empty_list(self):
        result = _extract_reasoning_from_details([])
        assert result is None

    def test_none_values_filtered(self):
        details = [{"reasoning": None}, {"text": ""}]
        result = _extract_reasoning_from_details(details)
        assert result is None

    def test_mixed_valid_and_empty(self):
        details = [
            {"reasoning": ""},
            {"reasoning": "Valid reasoning"},
            {"text": ""},
        ]
        result = _extract_reasoning_from_details(details)
        assert result == "Valid reasoning"

    def test_preference_order_reasoning_before_text(self):
        """When both 'reasoning' and 'text' keys exist, 'reasoning' should be preferred."""
        details = [{"reasoning": "From reasoning key", "text": "From text key"}]
        result = _extract_reasoning_from_details(details)
        assert result == "From reasoning key"


# ---------------------------------------------------------------------------
# Tests: FunctionCallView reasoning
# ---------------------------------------------------------------------------

class TestFunctionCallViewReasoning:
    """Test that FunctionCallView.view_reasoning() sends embed for short, file for long."""

    def _make_view_with_reasoning(self, reasoning: str):
        view_cls = _real_fcv.FunctionCallView
        view_cls._data_cache.clear()
        msg_id = 12345
        view_cls._data_cache[msg_id] = {
            "inputs": [],
            "outputs": [],
            "reasoning": reasoning,
        }
        view = view_cls(message_id=msg_id, has_outputs=False, has_reasoning=True)
        return view, msg_id

    def _make_interaction(self):
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()
        return interaction

    @pytest.mark.asyncio
    async def test_short_reasoning_single_embed(self):
        view, msg_id = self._make_view_with_reasoning("Short reasoning text")
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert embed.description == "Short reasoning text"

    @pytest.mark.asyncio
    async def test_long_reasoning_sends_file(self):
        reasoning = "A" * (EMBED_DESCRIPTION_MAX_CHARS + 1000)
        view, msg_id = self._make_view_with_reasoning(reasoning)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        # Should have embed with size notice
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert "too large" in embed.description.lower()
        # Should have a file with full reasoning
        file = call_args.kwargs.get("file")
        assert file is not None
        assert file.filename == "reasoning.txt"
        file_content = file.fp.read().decode("utf-8")
        assert file_content == reasoning

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        reasoning = "X" * 5000
        view, msg_id = self._make_view_with_reasoning(reasoning)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert "(truncated)" not in (embed.description or "")

    @pytest.mark.asyncio
    async def test_empty_reasoning_sends_message(self):
        view, msg_id = self._make_view_with_reasoning("")
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        assert call_args.kwargs.get("embed") is None


# ---------------------------------------------------------------------------
# Tests: FunctionCallView inputs
# ---------------------------------------------------------------------------

class TestFunctionCallViewInputs:
    """Test that FunctionCallView.view_inputs() sends embed for short, file for long."""

    def _make_view_with_inputs(self, inputs: list):
        view_cls = _real_fcv.FunctionCallView
        view_cls._data_cache.clear()
        msg_id = 11111
        view_cls._data_cache[msg_id] = {
            "inputs": inputs,
            "outputs": [],
            "reasoning": "",
        }
        view = view_cls(message_id=msg_id, has_outputs=False, has_reasoning=False)
        return view, msg_id

    def _make_interaction(self):
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        return interaction

    @pytest.mark.asyncio
    async def test_short_inputs_single_embed(self):
        inputs = [
            {"name": "web_search", "args": '{"query": "test"}'},
        ]
        view, msg_id = self._make_view_with_inputs(inputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_inputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert "web_search" in embed.description
        assert call_args.kwargs.get("file") is None

    @pytest.mark.asyncio
    async def test_long_inputs_sends_file(self):
        inputs = [
            {"name": "big_search", "args": '{"query": "' + "A" * 5000 + '"}'},
        ]
        view, msg_id = self._make_view_with_inputs(inputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_inputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert "too large" in embed.description.lower()
        file = call_args.kwargs.get("file")
        assert file is not None
        assert file.filename == "inputs.txt"
        file_content = file.fp.read().decode("utf-8")
        assert "big_search" in file_content

    @pytest.mark.asyncio
    async def test_no_inputs_sends_message(self):
        view, msg_id = self._make_view_with_inputs([])
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_inputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        assert call_args.kwargs.get("embed") is None


# ---------------------------------------------------------------------------
# Tests: FunctionCallView outputs
# ---------------------------------------------------------------------------

class TestFunctionCallViewOutputs:
    """Test that FunctionCallView.view_outputs() sends embed for short, file for long."""

    def _make_view_with_outputs(self, outputs: list):
        view_cls = _real_fcv.FunctionCallView
        view_cls._data_cache.clear()
        msg_id = 54321
        view_cls._data_cache[msg_id] = {
            "inputs": [],
            "outputs": outputs,
            "reasoning": "",
        }
        view = view_cls(message_id=msg_id, has_outputs=True, has_reasoning=False)
        return view, msg_id

    def _make_interaction(self):
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()
        return interaction

    @pytest.mark.asyncio
    async def test_view_outputs_no_outputs_sends_message(self):
        """When no outputs are cached, sends a plain message."""
        view_cls = _real_fcv.FunctionCallView
        view_cls._data_cache.clear()
        msg_id = 99999
        view = view_cls(message_id=msg_id, has_outputs=True, has_reasoning=False)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        assert call_args.kwargs.get("embed") is None

    @pytest.mark.asyncio
    async def test_short_outputs_single_embed(self):
        """Short outputs should fit in a single embed."""
        outputs = [
            {"name": "web_search", "result": "Search result text"},
            {"name": "web_fetch", "result": "Fetched page content"},
        ]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert "web_search" in embed.description
        assert "web_fetch" in embed.description

    @pytest.mark.asyncio
    async def test_long_output_sends_file(self):
        """A very long output should be sent as a file."""
        long_result = "A" * 8000
        outputs = [{"name": "web_fetch", "result": long_result}]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        # Should have embed with size notice
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert "too large" in embed.description.lower()
        # Should have a file with full output
        file = call_args.kwargs.get("file")
        assert file is not None
        assert file.filename == "outputs.txt"
        file_content = file.fp.read().decode("utf-8")
        assert "web_fetch" in file_content
        assert long_result in file_content

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        """Verify no '(truncated)' or '...' truncation markers."""
        large_result = "X" * 5000
        outputs = [
            {"name": "tool_a", "result": large_result},
            {"name": "tool_b", "result": "Y" * 3000},
        ]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        desc = embed.description or ""
        assert "(truncated)" not in desc
        assert not desc.endswith("...")

    @pytest.mark.asyncio
    async def test_individual_result_not_capped_at_500(self):
        """Individual results should not be capped at 500 characters."""
        result_600 = "B" * 600
        outputs = [{"name": "big_tool", "result": result_600}]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        # The full 600 chars should be present (not truncated to 497 + "...")
        assert "B" * 600 in embed.description

    @pytest.mark.asyncio
    async def test_file_is_ephemeral(self):
        """File message should be sent as ephemeral."""
        outputs = [{"name": "tool", "result": "A" * 8000}]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_outputs(interaction, button)

        first_call = interaction.response.send_message.call_args
        assert first_call.kwargs.get("ephemeral") is True


# ---------------------------------------------------------------------------
# Tests: ResponseView reasoning
# ---------------------------------------------------------------------------

class TestResponseViewReasoning:
    """Test that ResponseView.view_reasoning() sends embed for short, file for long."""

    def _make_view_with_steps(self, steps: list):
        view_cls = _real_fcv.ResponseView
        view_cls._reasoning_cache.clear()
        msg_id = 67890
        view_cls._reasoning_cache[msg_id] = steps
        view = view_cls(message_id=msg_id, has_reasoning=True)
        return view, msg_id

    def _make_interaction(self):
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()
        return interaction

    @pytest.mark.asyncio
    async def test_single_short_step(self):
        view, msg_id = self._make_view_with_steps(["Simple reasoning"])
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert "Simple reasoning" in embed.description

    @pytest.mark.asyncio
    async def test_multiple_steps_with_labels(self):
        steps = ["Reasoning for round 1", "Reasoning for round 2", "Final reasoning"]
        view, msg_id = self._make_view_with_steps(steps)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        desc = embed.description or ""
        assert "Round 1" in desc
        assert "Round 2" in desc
        assert "Final Response" in desc
        assert "Reasoning for round 1" in desc
        assert "Reasoning for round 2" in desc
        assert "Final reasoning" in desc

    @pytest.mark.asyncio
    async def test_long_steps_sends_file(self):
        long_step = "A" * 3000
        steps = [long_step, long_step]
        view, msg_id = self._make_view_with_steps(steps)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        assert embed is not None
        assert "too large" in embed.description.lower()
        file = call_args.kwargs.get("file")
        assert file is not None
        assert file.filename == "reasoning.txt"
        file_content = file.fp.read().decode("utf-8")
        # Both long steps should be fully present
        assert long_step in file_content
        assert file_content.count(long_step) == 2

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        steps = ["X" * 5000]
        view, msg_id = self._make_view_with_steps(steps)
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed")
        desc = embed.description or ""
        assert "(truncated)" not in desc

    @pytest.mark.asyncio
    async def test_empty_steps_sends_message(self):
        view, msg_id = self._make_view_with_steps([])
        button = MagicMock()
        interaction = self._make_interaction()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Fallback path attaches reasoning view
# ---------------------------------------------------------------------------

class TestFallbackPathReasoningAttachment:
    """Test that the fallback path in send_single_combined_message attaches reasoning."""

    def test_fallback_attaches_reasoning(self):
        """When combined send fails, fallback should still attach reasoning view."""
        with open("aiuser/response/chat/response.py", "r") as f:
            lines = f.readlines()

        # Find the HTTPException except block with "Fallback" comment
        found_fallback_block = False
        block_lines = []
        for i, line in enumerate(lines):
            if "except discord.HTTPException" in line:
                # Collect lines until we hit the next except or end of try block
                for j in range(i, min(i + 15, len(lines))):
                    block_lines.append(lines[j])
                    if lines[j].strip().startswith("return ") and j > i + 2:
                        found_fallback_block = True
                        break
                if found_fallback_block:
                    break

        fallback_text = "".join(block_lines)

        assert len(fallback_text) > 0, "Could not find the HTTPException fallback block"

        # The fallback should capture the return value and attach reasoning
        assert "sent_msg" in fallback_text, (
            "Fallback path should capture send_response() return value in 'sent_msg'"
        )
        assert "_attach_reasoning_view" in fallback_text, (
            "Fallback path should call _attach_reasoning_view()"
        )
