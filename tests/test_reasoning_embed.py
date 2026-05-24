"""Tests for reasoning embed changes.

Verifies that:
- _split_text_to_embed_chunks correctly splits long text
- _extract_reasoning_from_details extracts reasoning from structured details
- FunctionCallView.view_reasoning() sends multiple embeds for long reasoning (no truncation)
- FunctionCallView.view_outputs() sends multiple embeds for long outputs (no truncation)
- ResponseView.view_reasoning() sends multiple embeds for long reasoning (no truncation)
- The fallback path in send_single_combined_message attaches reasoning view
"""

import asyncio
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

_real_pipeline = import_module_directly(
    "aiuser.response.chat.llm_pipeline",
    "aiuser/response/chat/llm_pipeline.py",
)

_split_text_to_embed_chunks = _real_fcv._split_text_to_embed_chunks
_extract_reasoning_from_details = _real_pipeline._extract_reasoning_from_details


# ---------------------------------------------------------------------------
# Tests: _split_text_to_embed_chunks
# ---------------------------------------------------------------------------

class TestSplitTextToEmbedChunks:
    """Test that _split_text_to_embed_chunks correctly splits text."""

    def test_short_text_single_chunk(self):
        text = "Short reasoning"
        chunks = _split_text_to_embed_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_limit_single_chunk(self):
        text = "A" * 4096
        chunks = _split_text_to_embed_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_over_limit_splits(self):
        text = "A" * 5000
        chunks = _split_text_to_embed_chunks(text)
        assert len(chunks) == 2
        assert len(chunks[0]) <= 4096
        assert len(chunks[1]) <= 4096
        assert "".join(chunks) == text

    def test_splits_on_paragraph_boundary(self):
        paragraph = "A" * 2000
        text = paragraph + "\n\n" + "B" * 3000
        chunks = _split_text_to_embed_chunks(text)
        assert len(chunks) == 2
        assert chunks[0] == paragraph
        assert "B" in chunks[1]

    def test_splits_on_line_break(self):
        line = "A" * 200
        lines_text = "\n".join([line] * 30)  # 30 * 201 = 6030 chars
        chunks = _split_text_to_embed_chunks(lines_text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096
        # All original content should be preserved
        rejoined = "\n".join(chunks)
        # The split may strip leading newlines, but content should be intact
        assert line in chunks[0]
        assert line in chunks[-1]

    def test_very_long_text_many_chunks(self):
        text = "X" * 20000
        chunks = _split_text_to_embed_chunks(text)
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk) <= 4096
        assert "".join(chunks) == text

    def test_empty_text(self):
        chunks = _split_text_to_embed_chunks("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_custom_max_chars(self):
        text = "A" * 200
        chunks = _split_text_to_embed_chunks(text, max_chars=100)
        assert len(chunks) == 2
        assert len(chunks[0]) <= 100
        assert len(chunks[1]) <= 100
        assert "".join(chunks) == text

    def test_no_truncation_markers_in_output(self):
        """Verify the function never adds truncation markers."""
        text = "A" * 10000
        chunks = _split_text_to_embed_chunks(text)
        for chunk in chunks:
            assert "truncated" not in chunk.lower()
            assert "..." not in chunk


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
# Tests: FunctionCallView reasoning (no truncation)
# ---------------------------------------------------------------------------

class TestFunctionCallViewReasoning:
    """Test that FunctionCallView.view_reasoning() sends full reasoning without truncation."""

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

    @pytest.mark.asyncio
    async def test_short_reasoning_single_embed(self):
        view, msg_id = self._make_view_with_reasoning("Short reasoning text")
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert embed.description == "Short reasoning text"

    @pytest.mark.asyncio
    async def test_long_reasoning_multiple_embeds(self):
        reasoning = "A" * 8000
        view, msg_id = self._make_view_with_reasoning(reasoning)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count >= 1

        # Reconstruct full text from all embeds
        all_text = ""
        first_call = interaction.response.send_message.call_args
        first_embed = first_call.kwargs.get("embed") or first_call[0][0]
        all_text += first_embed.description
        for call in interaction.followup.send.call_args_list:
            embed = call.kwargs.get("embed") or call[0][0]
            all_text += embed.description

        # Full reasoning should be present (no truncation)
        assert len(all_text) >= len(reasoning)

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        reasoning = "X" * 5000
        view, msg_id = self._make_view_with_reasoning(reasoning)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        all_embeds = []
        first_call = interaction.response.send_message.call_args
        all_embeds.append(first_call.kwargs.get("embed") or first_call[0][0])
        for call in interaction.followup.send.call_args_list:
            all_embeds.append(call.kwargs.get("embed") or call[0][0])

        for embed in all_embeds:
            assert "(truncated)" not in (embed.description or "")

    @pytest.mark.asyncio
    async def test_empty_reasoning_sends_message(self):
        view, msg_id = self._make_view_with_reasoning("")
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.view_reasoning(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        assert call_args.kwargs.get("embed") is None


# ---------------------------------------------------------------------------
# Tests: FunctionCallView outputs (no truncation)
# ---------------------------------------------------------------------------

class TestFunctionCallViewOutputs:
    """Test that FunctionCallView.view_outputs() sends full outputs without truncation."""

    @pytest.mark.asyncio
    async def test_view_outputs_no_outputs_sends_message(self):
        """When no outputs are cached, sends a plain message."""
        view_cls = _real_fcv.FunctionCallView
        view_cls._data_cache.clear()
        msg_id = 99999
        view = view_cls(message_id=msg_id, has_outputs=True, has_reasoning=False)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        call_args = interaction.response.send_message.call_args
        assert call_args.kwargs.get("embed") is None

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

    @pytest.mark.asyncio
    async def test_short_outputs_single_embed(self):
        """Short outputs should fit in a single embed."""
        outputs = [
            {"name": "web_search", "result": "Search result text"},
            {"name": "web_fetch", "result": "Fetched page content"},
        ]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count == 0
        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        assert "web_search" in embed.description
        assert "web_fetch" in embed.description

    @pytest.mark.asyncio
    async def test_long_output_no_truncation(self):
        """A very long single output should not be truncated — it should be split across embeds."""
        long_result = "A" * 8000
        outputs = [{"name": "web_fetch", "result": long_result}]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_outputs(interaction, button)

        assert interaction.response.send_message.call_count == 1
        assert interaction.followup.send.call_count >= 1

        # Reconstruct full text from all embeds
        all_text = ""
        first_call = interaction.response.send_message.call_args
        first_embed = first_call.kwargs.get("embed") or first_call[0][0]
        all_text += first_embed.description
        for call in interaction.followup.send.call_args_list:
            embed = call.kwargs.get("embed") or call[0][0]
            all_text += embed.description

        # Full output text should be present (no truncation)
        assert len(all_text) >= len(long_result)

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        """Verify no '(truncated)' or '...' truncation markers in any embed."""
        large_result = "X" * 5000
        outputs = [
            {"name": "tool_a", "result": large_result},
            {"name": "tool_b", "result": "Y" * 3000},
        ]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_outputs(interaction, button)

        all_embeds = []
        first_call = interaction.response.send_message.call_args
        all_embeds.append(first_call.kwargs.get("embed") or first_call[0][0])
        for call in interaction.followup.send.call_args_list:
            all_embeds.append(call.kwargs.get("embed") or call[0][0])

        for embed in all_embeds:
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
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_outputs(interaction, button)

        call_args = interaction.response.send_message.call_args
        embed = call_args.kwargs.get("embed") or call_args[0][0]
        # The full 600 chars should be present (not truncated to 497 + "...")
        assert "B" * 600 in embed.description

    @pytest.mark.asyncio
    async def test_followup_chunks_are_ephemeral(self):
        """All followup embeds should be sent as ephemeral."""
        outputs = [{"name": "tool", "result": "A" * 8000}]
        view, msg_id = self._make_view_with_outputs(outputs)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_outputs(interaction, button)

        # First response should be ephemeral
        first_call = interaction.response.send_message.call_args
        assert first_call.kwargs.get("ephemeral") is True

        # All followups should be ephemeral
        for call in interaction.followup.send.call_args_list:
            assert call.kwargs.get("ephemeral") is True


# ---------------------------------------------------------------------------
# Tests: ResponseView reasoning (no truncation)
# ---------------------------------------------------------------------------

class TestResponseViewReasoning:
    """Test that ResponseView.view_reasoning() sends full reasoning without truncation."""

    def _make_view_with_steps(self, steps: list):
        view_cls = _real_fcv.ResponseView
        view_cls._reasoning_cache.clear()
        msg_id = 67890
        view_cls._reasoning_cache[msg_id] = steps
        view = view_cls(message_id=msg_id, has_reasoning=True)
        return view, msg_id

    @pytest.mark.asyncio
    async def test_single_short_step(self):
        view, msg_id = self._make_view_with_steps(["Simple reasoning"])
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

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
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        all_text = ""
        first_call = interaction.response.send_message.call_args
        first_embed = first_call.kwargs.get("embed") or first_call[0][0]
        all_text += first_embed.description or ""
        for call in interaction.followup.send.call_args_list:
            embed = call.kwargs.get("embed") or call[0][0]
            all_text += embed.description or ""

        assert "Round 1" in all_text
        assert "Round 2" in all_text
        assert "Final Response" in all_text
        assert "Reasoning for round 1" in all_text
        assert "Reasoning for round 2" in all_text
        assert "Final reasoning" in all_text

    @pytest.mark.asyncio
    async def test_long_steps_no_truncation(self):
        long_step = "A" * 2000
        steps = [long_step, long_step]
        view, msg_id = self._make_view_with_steps(steps)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        all_text = ""
        first_call = interaction.response.send_message.call_args
        first_embed = first_call.kwargs.get("embed") or first_call[0][0]
        all_text += first_embed.description or ""
        for call in interaction.followup.send.call_args_list:
            embed = call.kwargs.get("embed") or call[0][0]
            all_text += embed.description or ""

        # Both long steps should be fully present
        assert all_text.count(long_step) == 2

    @pytest.mark.asyncio
    async def test_no_truncation_markers(self):
        steps = ["X" * 5000]
        view, msg_id = self._make_view_with_steps(steps)
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.view_reasoning(interaction, button)

        all_embeds = []
        first_call = interaction.response.send_message.call_args
        all_embeds.append(first_call.kwargs.get("embed") or first_call[0][0])
        for call in interaction.followup.send.call_args_list:
            all_embeds.append(call.kwargs.get("embed") or call[0][0])

        for embed in all_embeds:
            desc = embed.description or ""
            assert "(truncated)" not in desc

    @pytest.mark.asyncio
    async def test_empty_steps_sends_message(self):
        view, msg_id = self._make_view_with_steps([])
        button = MagicMock()
        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

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
