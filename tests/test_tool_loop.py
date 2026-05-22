"""Tests for the tool-calling loop pipeline (LLMPipeline.run / _run_loop).

Verifies that:
- The loop respects max_tool_rounds configuration
- No-tools responses return immediately after round 1
- Tool calls are executed and results added to message history
- The stop instruction is injected on the last round and removed after
- The Gemini replacement pattern is detected correctly
- Reasoning is accumulated across rounds
- Forced break occurs when the LLM ignores tool_choice="none"
- The error-handling wrapper in run() catches expected exceptions
"""

import sys
from types import ModuleType
from typing import List, Union
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight mock infrastructure
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
    "aiuser.functions.edit_image", "aiuser.utils", "aiuser.response",
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

# Mock tiktoken
sys.modules["tiktoken"] = MagicMock()

# Mock openai with REAL exception types (must inherit from Exception)
class _RateLimitError(Exception):
    pass

class _APIConnectionError(Exception):
    pass

class _InternalServerError(Exception):
    pass

class _APIError(Exception):
    pass

class _APIStatusError(Exception):
    status_code = 500
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.response = MagicMock()

_openai_mock = MagicMock()
_openai_mock.AsyncOpenAI = MagicMock()
_openai_mock.RateLimitError = _RateLimitError
_openai_mock.APIConnectionError = _APIConnectionError
_openai_mock.InternalServerError = _InternalServerError
_openai_mock.APIError = _APIError
_openai_mock.APIStatusError = _APIStatusError
sys.modules["openai"] = _openai_mock
sys.modules["openai.types"] = MagicMock()
sys.modules["openai.types.chat"] = MagicMock()
sys.modules["openai.types.chat"].ChatCompletion = MagicMock()
sys.modules["openai.types.chat"].ChatCompletionMessageToolCall = MagicMock()

# Mock httpx with a real ReadTimeout exception
class _ReadTimeout(Exception):
    pass

_httpx_mock = MagicMock()
_httpx_mock.ReadTimeout = _ReadTimeout
sys.modules["httpx"] = _httpx_mock

# Mock tenacity
sys.modules["tenacity"] = MagicMock()

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
    "aiuser.response.chat.function_call_view",
]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Set up OpenRouter mock classes
_or = sys.modules["aiuser.functions.openrouter"]
_or.OpenRouterWebSearch = MagicMock()
_or.OpenRouterWebFetch = MagicMock()
_or.OpenRouterImageGeneration = MagicMock()

# Load real modules we're testing
from tests.mock_importer import import_module_directly

sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry

# Load real MessagesList (required by llm_pipeline imports)
sys.modules["aiuser.messages_list.messages"] = import_module_directly(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py"
)

sys.modules["aiuser.response.chat.llm_pipeline"] = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
LLMPipeline = sys.modules["aiuser.response.chat.llm_pipeline"].LLMPipeline
ResponsePart = sys.modules["aiuser.response.chat.llm_pipeline"].ResponsePart
PipelineResult = sys.modules["aiuser.response.chat.llm_pipeline"].PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A real dataclass to use as a mock tool schema (asdict() requires real dataclasses)
from dataclasses import dataclass as _real_dataclass

@_real_dataclass
class _MockToolSchema:
    name: str = "mock_tool"
    description: str = "A mock tool"
    parameters: dict = None
    strict: bool = False


def _make_mock_ctx():
    ctx = MagicMock()
    ctx.channel.id = 123
    ctx.guild.id = 456
    ctx.guild.name = "TestGuild"
    ctx.me.id = 789
    ctx.message = MagicMock()
    ctx.message.id = 999
    ctx.message.author = MagicMock()
    ctx.message.author.id = 111
    ctx.message.guild = ctx.guild
    ctx.react_quietly = AsyncMock()
    return ctx


def _make_tool_call(id: str, name: str, arguments: str = "{}"):
    """Create a mock ChatCompletionMessageToolCall."""
    tc = MagicMock()
    tc.id = id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_pipeline(max_rounds=2, tools_schemas=None, openrouter_tools=None):
    """Create an LLMPipeline instance with mocked internals for testing _run_loop."""
    ctx = _make_mock_ctx()

    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline.ctx = ctx
    pipeline.config = MagicMock()
    pipeline.bot = MagicMock()
    pipeline.msg_list = MagicMock()
    pipeline.msg_list.messages = []
    pipeline.msg_list.model = "test-model"
    pipeline.msg_list.can_reply = True
    pipeline.msg_list.init_message = MagicMock()
    pipeline.msg_list.init_message.reference = None
    pipeline.msg_list.__len__ = MagicMock(return_value=0)
    pipeline.msg_list.get_json = MagicMock(return_value=[])
    pipeline.msg_list.add_assistant = AsyncMock()
    pipeline.msg_list.add_tool_result = AsyncMock()
    pipeline.msg_list.add_user = AsyncMock()
    pipeline.model = "test-model"
    pipeline.can_reply = True
    pipeline.openai_client = MagicMock()
    pipeline.enabled_tools = []
    pipeline.available_tools_schemas = tools_schemas or []
    pipeline.openrouter_tools = openrouter_tools or []
    pipeline.completion = None
    pipeline.reasoning = None
    pipeline.response_parts = []
    pipeline.collected_images = []

    # Mock config to return max_rounds (must be AsyncMock for 'await')
    mock_guild_config = MagicMock()
    mock_guild_config.max_tool_rounds = AsyncMock(return_value=max_rounds)
    pipeline.config.guild = MagicMock(return_value=mock_guild_config)

    # Mock the async setup methods that _run_loop calls
    pipeline.get_custom_parameters = AsyncMock(return_value={})
    pipeline.get_service_tier = AsyncMock(return_value=None)
    pipeline.setup_tools = AsyncMock()
    pipeline._inject_pdf_annotations = AsyncMock()
    pipeline._build_plugins = AsyncMock(return_value=[])

    return pipeline


# ---------------------------------------------------------------------------
# Tests: PipelineResult dataclass
# ---------------------------------------------------------------------------

class TestPipelineResult:
    """Test PipelineResult dataclass defaults and behavior."""

    def test_defaults(self):
        result = PipelineResult()
        assert result.text is None
        assert result.reasoning is None
        assert result.images == []
        assert result.response_parts == []
        assert result.has_tools is False
        assert result.pre_text is None
        assert result.pre_text_complete is False

    def test_with_values(self):
        result = PipelineResult(
            text="hello",
            reasoning="thoughts",
            has_tools=True,
            pre_text="pre",
            pre_text_complete=True,
        )
        assert result.text == "hello"
        assert result.reasoning == "thoughts"
        assert result.has_tools is True
        assert result.pre_text == "pre"
        assert result.pre_text_complete is True


# ---------------------------------------------------------------------------
# Tests: run() error handling wrapper
# ---------------------------------------------------------------------------

class TestRunErrorHandling:
    """Test that run() catches expected exceptions and returns empty PipelineResult."""

    @pytest.mark.asyncio
    async def test_run_returns_empty_result_on_timeout(self):
        pipeline = _make_pipeline()
        pipeline._run_loop = AsyncMock(side_effect=_ReadTimeout("timeout"))
        result = await pipeline.run()
        assert isinstance(result, PipelineResult)
        assert result.text is None
        assert result.has_tools is False

    @pytest.mark.asyncio
    async def test_run_returns_empty_result_on_rate_limit(self):
        pipeline = _make_pipeline()
        pipeline._run_loop = AsyncMock(side_effect=_RateLimitError("rate limited"))
        result = await pipeline.run()
        assert isinstance(result, PipelineResult)
        assert result.text is None

    @pytest.mark.asyncio
    async def test_run_returns_empty_result_on_api_connection_error(self):
        pipeline = _make_pipeline()
        pipeline._run_loop = AsyncMock(side_effect=_APIConnectionError("connection error"))
        result = await pipeline.run()
        assert isinstance(result, PipelineResult)
        assert result.text is None

    @pytest.mark.asyncio
    async def test_run_returns_empty_result_on_unexpected_exception(self):
        pipeline = _make_pipeline()
        pipeline._run_loop = AsyncMock(side_effect=RuntimeError("unexpected"))
        result = await pipeline.run()
        assert isinstance(result, PipelineResult)
        assert result.text is None


# ---------------------------------------------------------------------------
# Tests: _run_loop core logic
# ---------------------------------------------------------------------------

class TestRunLoopNoTools:
    """Test _run_loop when no tool calls are returned by the LLM."""

    @pytest.mark.asyncio
    async def test_single_round_no_tools(self):
        """When LLM returns no tool_calls, loop completes after round 1."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        # Mock call_client: return text, no tools
        pipeline.call_client = AsyncMock(return_value=(
            "Hello world",    # text
            "thinking...",    # reasoning
            [],               # tool_calls
            None,             # reasoning_details
            None,             # model_extra
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert result.has_tools is False
        assert result.text == "Hello world"
        assert result.reasoning == "thinking..."
        assert result.pre_text == "Hello world"
        assert result.pre_text_complete is True
        # call_client should have been called exactly once
        assert pipeline.call_client.call_count == 1

    @pytest.mark.asyncio
    async def test_single_round_empty_response(self):
        """When LLM returns empty text and no tools, result reflects that."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        pipeline.call_client = AsyncMock(return_value=(
            None, None, [], None, None,
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert result.has_tools is False
        assert result.text is None
        assert result.pre_text is None


class TestRunLoopWithTools:
    """Test _run_loop when the LLM returns tool calls."""

    @pytest.mark.asyncio
    async def test_two_rounds_with_tools(self):
        """Round 1: tool calls + execute. Round 2: no tool calls = done."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()
        pipeline._process_openrouter_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "weather", '{"city": "NYC"}')

        # Round 1: text + tool_call, Round 2: text only
        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me check", "thinking...", [tool_call], None, None),
            ("It's sunny in NYC", "more thoughts...", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert result.has_tools is True
        assert result.pre_text == "Let me check"
        assert result.pre_text_complete is True
        assert pipeline.call_client.call_count == 2
        # Tools should have been executed once (round 1)
        pipeline._process_and_add_tool_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_incomplete_pre_text_with_trailing_comma(self):
        """Trailing comma marks pre-text as incomplete."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "weather")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me search for that,", "r", [tool_call], None, None),
            ("The weather in NYC is sunny.", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = LLMPipeline._is_text_incomplete

        result = await pipeline._run_loop()

        assert result.has_tools is True
        assert result.pre_text_complete is False


class TestRunLoopGeminiReplacement:
    """Test Gemini replacement pattern detection in the loop."""

    @pytest.mark.asyncio
    async def test_replacement_pattern_detected(self):
        """When round 2 text starts with round 1 pre-text, it's a replacement."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "search")

        pre = "Here is what I found:"
        # Round 2 replays pre-text
        pipeline.call_client = AsyncMock(side_effect=[
            (pre, "r", [tool_call], None, None),
            (pre + " The answer is 42.", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        # Replacement: final text should be the full round 2 text, not concatenated
        assert result.text == pre + " The answer is 42."

    @pytest.mark.asyncio
    async def test_no_replacement_pattern(self):
        """When round 2 text does NOT start with pre-text, concatenate."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "search")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me look that up.", "r", [tool_call], None, None),
            ("The answer is 42.", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        # No replacement: concatenate
        assert result.text == "Let me look that up.The answer is 42."


class TestRunLoopReasoningAccumulation:
    """Test that reasoning is accumulated across rounds."""

    @pytest.mark.asyncio
    async def test_reasoning_from_single_round(self):
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        pipeline.call_client = AsyncMock(return_value=(
            "text", "single thought", [], None, None,
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()
        assert result.reasoning == "single thought"

    @pytest.mark.asyncio
    async def test_reasoning_accumulated_across_rounds(self):
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "search")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me check", "thought 1", [tool_call], None, None),
            ("The answer is 42", "thought 2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()
        assert result.reasoning == "thought 1\n\nthought 2"

    @pytest.mark.asyncio
    async def test_no_reasoning(self):
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        pipeline.call_client = AsyncMock(return_value=(
            "text", None, [], None, None,
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()
        assert result.reasoning is None


class TestRunLoopStopInstruction:
    """Test that the stop instruction is injected on the last round."""

    @pytest.mark.asyncio
    async def test_stop_instruction_removed_after_loop(self):
        """After the loop completes, the stop instruction should be removed."""
        pipeline = _make_pipeline(max_rounds=2, tools_schemas=[_MockToolSchema()])
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call = _make_tool_call("call_1", "search")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me check", "r", [tool_call], None, None),
            ("The answer is 42.", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert result.has_tools is True
        # After the loop, no stop instruction should remain
        stop_entries = [
            e for e in pipeline.msg_list.messages
            if isinstance(e, MessageEntry) and e.role == "user"
            and "maximum number of tool-calling rounds" in str(e.content)
        ]
        assert len(stop_entries) == 0


class TestRunLoopForcedBreak:
    """Test forced break when LLM ignores tool_choice='none' on the last round."""

    @pytest.mark.asyncio
    async def test_forced_break_on_last_round_with_tool_calls(self):
        """When the LLM returns tool_calls on the last round despite tool_choice='none',
        the loop should execute them and force-break."""
        pipeline = _make_pipeline(max_rounds=2)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tool_call_1 = _make_tool_call("call_1", "search")
        tool_call_2 = _make_tool_call("call_2", "fetch")

        # Both rounds return tool_calls
        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me check", "r1", [tool_call_1], None, None),
            ("Found more to check", "r2", [tool_call_2], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert result.has_tools is True
        # Tools executed for both rounds
        assert pipeline._process_and_add_tool_results.call_count == 2
        # call_client called exactly 2 times (max_rounds)
        assert pipeline.call_client.call_count == 2
        # Final text should be the accumulated pre-text (no final response)
        assert result.text == "Let me checkFound more to check"


class TestRunLoopConfigurableRounds:
    """Test that max_tool_rounds controls the loop iterations."""

    @pytest.mark.asyncio
    async def test_max_rounds_1_no_tools(self):
        """With max_rounds=1 and no tools, single call."""
        pipeline = _make_pipeline(max_rounds=1)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        pipeline.call_client = AsyncMock(return_value=(
            "Hello", None, [], None, None,
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert pipeline.call_client.call_count == 1
        assert result.has_tools is False

    @pytest.mark.asyncio
    async def test_max_rounds_1_with_tools(self):
        """With max_rounds=1 and tools available, the single round uses tool_choice='none'
        and the stop instruction is injected."""
        pipeline = _make_pipeline(max_rounds=1, tools_schemas=[_MockToolSchema()])
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        # LLM still returns tool_calls despite tool_choice="none"
        tool_call = _make_tool_call("call_1", "search")
        pipeline._process_and_add_tool_results = AsyncMock()
        pipeline.call_client = AsyncMock(return_value=(
            "Let me search", "r", [tool_call], None, None,
        ))
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert pipeline.call_client.call_count == 1
        assert result.has_tools is True

    @pytest.mark.asyncio
    async def test_max_rounds_3_all_tool_rounds(self):
        """With max_rounds=3, the loop can run up to 3 rounds."""
        pipeline = _make_pipeline(max_rounds=3)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tc1 = _make_tool_call("call_1", "search")
        tc2 = _make_tool_call("call_2", "fetch")
        tc3 = _make_tool_call("call_3", "analyze")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Searching", "r1", [tc1], None, None),
            ("Fetching more", "r2", [tc2], None, None),
            ("Analyzing results", "r3", [tc3], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert pipeline.call_client.call_count == 3
        assert result.has_tools is True
        # Tools executed 3 times
        assert pipeline._process_and_add_tool_results.call_count == 3

    @pytest.mark.asyncio
    async def test_max_rounds_3_early_exit(self):
        """With max_rounds=3, exits early when LLM returns no tools on round 2."""
        pipeline = _make_pipeline(max_rounds=3)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)
        pipeline._process_and_add_tool_results = AsyncMock()

        tc = _make_tool_call("call_1", "search")

        pipeline.call_client = AsyncMock(side_effect=[
            ("Searching", "r1", [tc], None, None),
            ("Here are the results", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        # Should exit after round 2, not call round 3
        assert pipeline.call_client.call_count == 2
        assert result.has_tools is True
        assert result.text is not None


class TestRunLoopOpenRouterTools:
    """Test handling of OpenRouter server-side tools."""

    @pytest.mark.asyncio
    async def test_openrouter_tool_synthetic_result(self):
        """OpenRouter tools get synthetic results added to message history."""
        pipeline = _make_pipeline(max_rounds=2)
        # Make "web_search" recognized as an OpenRouter tool
        pipeline._is_openrouter_tool_name = lambda name: name == "web_search"
        pipeline._process_and_add_tool_results = AsyncMock()
        pipeline._process_openrouter_tool_results = AsyncMock()

        or_tc = _make_tool_call("call_or", "web_search", '{"query": "test"}')

        pipeline.call_client = AsyncMock(side_effect=[
            ("Let me search", "r", [or_tc], None, None),
            ("Found results", "r2", [], None, None),
        ])
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        # Synthetic tool result should have been added
        pipeline.msg_list.add_tool_result.assert_called()


class TestRunLoopDefaultMaxRounds:
    """Test default max_tool_rounds value."""

    @pytest.mark.asyncio
    async def test_default_max_rounds_when_none(self):
        """When config returns None, defaults to 2."""
        pipeline = _make_pipeline(max_rounds=None)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        tc = _make_tool_call("call_1", "search")
        pipeline._process_and_add_tool_results = AsyncMock()

        call_count = 0
        async def count_calls(kwargs_dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("Searching", "r", [tc], None, None)
            return ("Done", "r2", [], None, None)

        pipeline.call_client = count_calls
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        # Should default to 2 rounds
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_default_max_rounds_when_zero(self):
        """When config returns 0, defaults to 2."""
        pipeline = _make_pipeline(max_rounds=0)
        pipeline._is_openrouter_tool_name = MagicMock(return_value=False)

        tc = _make_tool_call("call_1", "search")
        pipeline._process_and_add_tool_results = AsyncMock()

        call_count = 0
        async def count_calls(kwargs_dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("Searching", "r", [tc], None, None)
            return ("Done", "r2", [], None, None)

        pipeline.call_client = count_calls
        pipeline._is_text_incomplete = MagicMock(return_value=False)

        result = await pipeline._run_loop()

        assert call_count == 2
