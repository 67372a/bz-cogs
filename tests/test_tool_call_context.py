"""Tests for tool call context correctness, function call/response matching,
and web tool function call handling.

Verifies that:
- Tool calls and responses are correctly added to context with matching id and name
  per Gemini 3.5 Flash doc requirements
- MessageEntry.tool_call_id is a string (OpenAI uses string IDs like "call_abc123")
- log_messages() shows tool call → tool result pairing verification
- _verify_tool_call_response_matching correctly detects mismatches
- _process_and_add_tool_results handles exceptions from asyncio.gather
- Web search/fetch/answer tool calls properly structure their function responses
- get_json() correctly serializes tool_calls with id, name, and arguments
"""

import json
import logging
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing real modules
# ---------------------------------------------------------------------------
discord_mock = MagicMock()
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
sys.modules.setdefault("discord", discord_mock)
sys.modules.setdefault("discord.ext", MagicMock())
sys.modules.setdefault("discord.ext.commands", MagicMock())
sys.modules.setdefault("discord.app_commands", MagicMock())

# Ensure aiuser packages exist as mock packages
from tests.mock_importer import _make_mock_package, import_module_directly

_sub_pkgs = [
    "aiuser",
    "aiuser.messages_list",
    "aiuser.response",
    "aiuser.response.chat",
    "aiuser.config",
    "aiuser.types",
    "aiuser.utils",
    "aiuser.functions",
    "aiuser.functions.openrouter",
    "aiuser.functions.generate_image",
    "aiuser.functions.edit_image",
    "aiuser.functions.attach_files",
    "aiuser.functions.web_search",
    "aiuser.functions.web_fetch",
    "aiuser.functions.web_answer",
    "aiuser.messages_list.converter",
    "aiuser.messages_list.converter.embed",
    "aiuser.messages_list.converter.image",
]
for _pkg in _sub_pkgs:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Mock redbot
import types as _types


def _make_module(name, attrs=None):
    mod = _types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    mod.__file__ = f"<mocked-{name}>"
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


_redbot = _make_module("redbot", {"__version__": "3.5.0", "version_info": MagicMock()})
sys.modules.setdefault("redbot", _redbot)
_redbot_core = _make_module("redbot.core", {
    "Config": MagicMock(),
    "commands": MagicMock(),
    "app_commands": MagicMock(),
})
sys.modules.setdefault("redbot.core", _redbot_core)
for _name in ["redbot.core.commands", "redbot.core.bot", "redbot.core.utils",
              "redbot.core.utils.chat_formatting", "redbot.core.utils.views"]:
    sys.modules.setdefault(_name, MagicMock())

# Mock tiktoken
sys.modules.setdefault("tiktoken", MagicMock())

# Mock other aiuser dependencies
_mocks_needed = [
    "aiuser.config.defaults",
    "aiuser.config.models",
    "aiuser.config.constants",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.opt_view",
    "aiuser.types.abc",
    "aiuser.types.enums",
    "aiuser.utils.utilities",
]
for _m in _mocks_needed:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Load real MessageEntry
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry

# Load real MessagesList
sys.modules["aiuser.messages_list.messages"] = import_module_directly(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py"
)
MessagesList = sys.modules["aiuser.messages_list.messages"].MessagesList

# Load real tool_call types FIRST (needed by web tool modules)
sys.modules["aiuser.functions.types"] = import_module_directly(
    "aiuser.functions.types", "aiuser/functions/types.py"
)
sys.modules["aiuser.functions.tool_call"] = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)

# Mock the web tool providers modules before importing tool_call modules
for _wp in [
    "aiuser.functions.web_search.providers",
    "aiuser.functions.web_fetch.providers",
    "aiuser.functions.web_answer.providers",
]:
    if _wp not in sys.modules:
        _wp_mock = MagicMock()
        # get_search_provider / get_fetch_provider / get_answer_provider must return an object
        # with an async method. Mock it to return an AsyncMock.
        _provider_mock = MagicMock()
        _provider_mock.search = AsyncMock(return_value="mock search results")
        _provider_mock.fetch = AsyncMock(return_value="mock fetch results")
        _provider_mock.answer = AsyncMock(return_value="mock answer results")
        _wp_mock.get_search_provider = MagicMock(return_value=_provider_mock)
        _wp_mock.get_fetch_provider = MagicMock(return_value=_provider_mock)
        _wp_mock.get_answer_provider = MagicMock(return_value=_provider_mock)
        sys.modules[_wp] = _wp_mock

# Load real web tool modules
sys.modules["aiuser.functions.web_search.tool_call"] = import_module_directly(
    "aiuser.functions.web_search.tool_call", "aiuser/functions/web_search/tool_call.py"
)
WebSearchToolCall = sys.modules["aiuser.functions.web_search.tool_call"].WebSearchToolCall

sys.modules["aiuser.functions.web_fetch.tool_call"] = import_module_directly(
    "aiuser.functions.web_fetch.tool_call", "aiuser/functions/web_fetch/tool_call.py"
)
WebFetchToolCall = sys.modules["aiuser.functions.web_fetch.tool_call"].WebFetchToolCall

sys.modules["aiuser.functions.web_answer.tool_call"] = import_module_directly(
    "aiuser.functions.web_answer.tool_call", "aiuser/functions/web_answer/tool_call.py"
)
WebAnswerToolCall = sys.modules["aiuser.functions.web_answer.tool_call"].WebAnswerToolCall

# Load real LLMPipeline (need additional mocks for its dependencies)
_pipeline_mocks = [
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.openrouter",
    "aiuser.functions.openrouter.web_search",
    "aiuser.functions.openrouter.web_fetch",
    "aiuser.functions.openrouter.image_generation",
    "aiuser.functions.openrouter.pdf_parsing",
    "aiuser.functions.openrouter.image_parsing",
    "aiuser.response.chat.function_call_view",
    "aiuser.response.chat.llm_pipeline",
]
# Mock tenacity and httpx and openai
sys.modules.setdefault("tenacity", MagicMock())
sys.modules.setdefault("tenacity.retry", MagicMock())
sys.modules.setdefault("tenacity.stop_after_attempt", MagicMock())
sys.modules.setdefault("tenacity.wait_random_exponential", MagicMock())
sys.modules.setdefault("tenacity.retry_if_exception_type", MagicMock())
sys.modules.setdefault("tenacity.retry_if_result", MagicMock())
sys.modules.setdefault("httpx", MagicMock())
_openai_mock = MagicMock()
_openai_mock.AsyncOpenAI = MagicMock()
_openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
_openai_mock.APIConnectionError = type("APIConnectionError", (Exception,), {})
_openai_mock.InternalServerError = type("InternalServerError", (Exception,), {})
_openai_mock.APIError = type("APIError", (Exception,), {})
_openai_mock.APIStatusError = type("APIStatusError", (Exception,), {})
sys.modules.setdefault("openai", _openai_mock)
sys.modules.setdefault("openai.types", MagicMock())
sys.modules.setdefault("openai.types.chat", MagicMock())

for _m in _pipeline_mocks:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# The openrouter package mock needs specific class attributes for llm_pipeline imports
_or_mock = sys.modules["aiuser.functions.openrouter"]
_or_mock.OpenRouterWebSearch = MagicMock()
_or_mock.OpenRouterWebFetch = MagicMock()
_or_mock.OpenRouterImageGeneration = MagicMock()
_or_mock.OpenRouterPdfParsing = MagicMock()
_or_mock.OpenRouterImageParsing = MagicMock()

sys.modules["aiuser.response.chat.llm_pipeline"] = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
LLMPipeline = sys.modules["aiuser.response.chat.llm_pipeline"].LLMPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_ctx(channel_id=123, guild_id=456, bot_user_id=789):
    ctx = MagicMock()
    ctx.channel.id = channel_id
    ctx.guild.id = guild_id
    ctx.guild.name = "TestGuild"
    ctx.me.id = bot_user_id
    ctx.message = MagicMock()
    ctx.message.id = 999
    ctx.message.author = MagicMock()
    ctx.message.author.id = 111
    ctx.message.guild = ctx.guild
    return ctx


def _make_mock_cog():
    cog = MagicMock()
    cog.bot = MagicMock()
    cog.config = MagicMock()
    cog.ignore_regex = {}
    cog.override_prompt_start_time = {}
    return cog


def _make_messages_list(messages=None, prefill=None, model="gpt-4", tokens=100, token_limit=8000):
    ml = MessagesList.__new__(MessagesList)
    ml.ctx = _make_mock_ctx()
    ml.model = model
    ml.tokens = tokens
    ml.token_limit = token_limit
    ml.messages = messages or []
    ml.prefill = prefill
    _encoding_mock = MagicMock()
    _encoding_mock.encode = MagicMock(return_value=[1, 2, 3])
    ml._encoding = _encoding_mock
    return ml


def _make_pipeline():
    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline.ctx = _make_mock_ctx()
    pipeline.msg_list = _make_messages_list()
    return pipeline


def _make_tool_call(id: str, name: str, arguments: str = "{}"):
    """Create a mock tool call object matching OpenAI's ChatCompletionMessageToolCall."""
    tc = MagicMock()
    tc.id = id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Tests for MessageEntry.tool_call_id type
# ---------------------------------------------------------------------------
class TestMessageEntryToolCallIdType:
    """Verify that MessageEntry.tool_call_id accepts string IDs (OpenAI format)."""

    def test_tool_call_id_accepts_string(self):
        entry = MessageEntry("tool", "result", tool_call_id="call_abc123", name="web_search")
        assert entry.tool_call_id == "call_abc123"
        assert isinstance(entry.tool_call_id, str)

    def test_tool_call_id_default_is_none(self):
        entry = MessageEntry("user", "hello")
        assert entry.tool_call_id is None

    def test_tool_call_id_accepts_none_explicitly(self):
        entry = MessageEntry("assistant", "reply", tool_call_id=None)
        assert entry.tool_call_id is None

    def test_tool_call_id_typical_openai_ids(self):
        """Test with typical OpenAI-generated tool call ID formats."""
        for tc_id in ["call_abc123def456", "call_1", "chatcmpl-tool-xyz"]:
            entry = MessageEntry("tool", "result", tool_call_id=tc_id, name="func")
            assert entry.tool_call_id == tc_id

    def test_assistant_entry_with_tool_calls(self):
        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        entry = MessageEntry("assistant", None, tool_calls=[tc])
        assert entry.tool_calls == [tc]
        assert entry.tool_call_id is None

    def test_tool_entry_with_name_and_id(self):
        entry = MessageEntry("tool", "search results", tool_call_id="call_1", name="web_search")
        assert entry.name == "web_search"
        assert entry.tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# Tests for get_json() serialization of tool calls and results
# ---------------------------------------------------------------------------
class TestGetJsonToolCallSerialization:
    """Verify that get_json() correctly serializes tool calls with id/name for Gemini 3.5 Flash."""

    def test_assistant_message_with_tool_calls_has_id(self):
        tc = _make_tool_call("call_abc", "web_search", '{"query": "test"}')
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
        ])
        result = ml.get_json()
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1

    def test_tool_result_has_tool_call_id_and_name(self):
        ml = _make_messages_list(messages=[
            MessageEntry("tool", "search results", tool_call_id="call_abc", name="web_search"),
        ])
        result = ml.get_json()
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc"
        assert msg["name"] == "web_search"
        assert msg["content"] == "search results"

    def test_full_tool_call_and_response_cycle(self):
        """Test a complete tool call → tool result cycle in get_json()."""
        tc1 = _make_tool_call("call_1", "web_search", '{"query": "python"}')
        tc2 = _make_tool_call("call_2", "web_fetch", '{"urls": ["https://example.com"]}')
        ml = _make_messages_list(messages=[
            MessageEntry("system", "You are helpful."),
            MessageEntry("user", "Search for python"),
            MessageEntry("assistant", None, tool_calls=[tc1, tc2]),
            MessageEntry("tool", "Python is a programming language", tool_call_id="call_1", name="web_search"),
            MessageEntry("tool", "Example.com content", tool_call_id="call_2", name="web_fetch"),
        ])
        result = ml.get_json()
        assert len(result) == 5

        # Verify assistant message has both tool calls
        assistant_msg = result[2]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 2

        # Verify tool results have correct ids and names
        tool1_msg = result[3]
        assert tool1_msg["role"] == "tool"
        assert tool1_msg["tool_call_id"] == "call_1"
        assert tool1_msg["name"] == "web_search"

        tool2_msg = result[4]
        assert tool2_msg["role"] == "tool"
        assert tool2_msg["tool_call_id"] == "call_2"
        assert tool2_msg["name"] == "web_fetch"

    def test_multimodal_tool_result_serialization(self):
        """Tool results with multimodal content (image + text) should serialize correctly."""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Generated image"},
        ]
        ml = _make_messages_list(messages=[
            MessageEntry("tool", content, tool_call_id="call_img", name="generate_image"),
        ])
        result = ml.get_json()
        assert result[0]["content"] == content
        assert result[0]["tool_call_id"] == "call_img"
        assert result[0]["name"] == "generate_image"


# ---------------------------------------------------------------------------
# Tests for log_messages() tool call → tool result pairing
# ---------------------------------------------------------------------------
class TestLogMessagesToolPairing:
    """Verify that log_messages() shows correct tool call → tool result pairing."""

    def test_matched_pair_logs_success(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_1", name="web_search"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "MATCH" in caplog.text
        assert "call_1" in caplog.text
        assert "web_search" in caplog.text
        assert "All tool calls have matching tool results" in caplog.text

    def test_mismatched_name_logs_warning(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_1", name="web_fetch"),  # wrong name!
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "MISMATCH" in caplog.text
        assert "NAMES DIFFER" in caplog.text

    def test_missing_tool_result_logs_warning(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
            # No tool result for call_1!
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "MISSING" in caplog.text
        assert "NO matching tool_result" in caplog.text

    def test_orphaned_tool_result_logs_warning(self, caplog):
        tc = _make_tool_call("call_real", "web_search")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results for real", tool_call_id="call_real", name="web_search"),
            MessageEntry("tool", "orphan results", tool_call_id="call_orphan", name="web_fetch"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "ORPHAN" in caplog.text
        assert "NO matching tool_call" in caplog.text

    def test_multiple_tool_calls_all_matched(self, caplog):
        tc1 = _make_tool_call("call_1", "web_search")
        tc2 = _make_tool_call("call_2", "web_fetch")
        tc3 = _make_tool_call("call_3", "web_answer")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc1, tc2, tc3]),
            MessageEntry("tool", "search results", tool_call_id="call_1", name="web_search"),
            MessageEntry("tool", "fetch results", tool_call_id="call_2", name="web_fetch"),
            MessageEntry("tool", "answer results", tool_call_id="call_3", name="web_answer"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "All tool calls have matching tool results" in caplog.text
        assert caplog.text.count("MATCH") == 3

    def test_tool_call_details_include_id_and_name(self, caplog):
        tc = _make_tool_call("call_xyz", "generate_image")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "generate_image(id=call_xyz)" in caplog.text

    def test_no_tool_calls_skips_pairing_verification(self, caplog):
        ml = _make_messages_list(messages=[
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "hi there"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "Tool Call/Result Pairing Verification" not in caplog.text

    def test_tool_result_shows_id_and_name(self, caplog):
        ml = _make_messages_list(messages=[
            MessageEntry("tool", "result text", tool_call_id="call_42", name="web_search"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "tool_call_id=call_42" in caplog.text
        assert "name=web_search" in caplog.text


# ---------------------------------------------------------------------------
# Tests for _verify_tool_call_response_matching
# ---------------------------------------------------------------------------
class TestVerifyToolCallResponseMatching:
    """Test the _verify_tool_call_response_matching method on LLMPipeline."""

    def _setup_pipeline_with_messages(self, messages):
        pipeline = _make_pipeline()
        pipeline.msg_list = _make_messages_list(messages=messages)
        return pipeline

    def test_all_matched_logs_ok(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        pipeline = self._setup_pipeline_with_messages([
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_1", name="web_search"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._verify_tool_call_response_matching()
        assert "Context verification OK" in caplog.text

    def test_name_mismatch_logs_warning(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        pipeline = self._setup_pipeline_with_messages([
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_1", name="web_fetch"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._verify_tool_call_response_matching()
        assert "Context verification FAILED" in caplog.text
        assert "does NOT match" in caplog.text

    def test_missing_response_logs_warning(self, caplog):
        tc = _make_tool_call("call_1", "web_search")
        pipeline = self._setup_pipeline_with_messages([
            MessageEntry("assistant", None, tool_calls=[tc]),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._verify_tool_call_response_matching()
        assert "has NO matching functionResponse" in caplog.text

    def test_orphaned_response_logs_warning(self, caplog):
        tc = _make_tool_call("call_real", "web_search")
        pipeline = self._setup_pipeline_with_messages([
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_real", name="web_search"),
            MessageEntry("tool", "orphan results", tool_call_id="call_orphan", name="web_fetch"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._verify_tool_call_response_matching()
        assert "orphaned functionResponse" in caplog.text

    def test_no_tool_calls_silent(self, caplog):
        pipeline = self._setup_pipeline_with_messages([
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "hi"),
        ])
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._verify_tool_call_response_matching()
        # Should not log anything since there are no tool calls
        assert "verification" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Tests for web_search tool call handling
# ---------------------------------------------------------------------------
class TestWebSearchToolCallHandling:
    """Verify WebSearchToolCall properly structures function responses."""

    def test_schema_has_correct_function_name(self):
        """The schema function name must match the function_name attribute."""
        assert WebSearchToolCall.function_name == "web_search"
        assert WebSearchToolCall.schema.function.name == "web_search"

    def test_schema_has_query_parameter(self):
        params = WebSearchToolCall.schema.function.parameters
        assert "search_query" in params.properties
        assert "search_query" in params.required

    def test_schema_type_is_function(self):
        assert WebSearchToolCall.schema.type == "function"

    @pytest.mark.asyncio
    async def test_handle_empty_query_returns_error(self):
        config = MagicMock()
        ctx = MagicMock()
        tool = WebSearchToolCall(config=config, ctx=ctx)
        result = await tool._handle({"query": ""})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_handle_with_none_backend_returns_error(self):
        """When backend is 'none', should return a NoneProvider error string."""
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.web_search_backend = AsyncMock(return_value="none")
        guild_cfg.web_search_config = AsyncMock(return_value=None)
        config.guild.return_value = guild_cfg

        ctx = MagicMock()
        ctx.guild = MagicMock()

        tool = WebSearchToolCall(config=config, ctx=ctx)
        result = await tool._handle({"query": "test query"})
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests for web_fetch tool call handling
# ---------------------------------------------------------------------------
class TestWebFetchToolCallHandling:
    """Verify WebFetchToolCall properly structures function responses."""

    def test_schema_has_correct_function_name(self):
        assert WebFetchToolCall.function_name == "web_fetch"
        assert WebFetchToolCall.schema.function.name == "web_fetch"

    def test_schema_has_urls_parameter(self):
        params = WebFetchToolCall.schema.function.parameters
        assert "urls" in params.properties
        assert "urls" in params.required

    def test_schema_type_is_function(self):
        assert WebFetchToolCall.schema.type == "function"

    @pytest.mark.asyncio
    async def test_handle_empty_urls_returns_error(self):
        config = MagicMock()
        ctx = MagicMock()
        tool = WebFetchToolCall(config=config, ctx=ctx)
        result = await tool._handle({"urls": []})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_handle_accepts_single_url_string(self):
        """web_fetch should normalize a single string URL to a list."""
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.web_fetch_backend = AsyncMock(return_value="none")
        guild_cfg.web_fetch_config = AsyncMock(return_value=None)
        config.guild.return_value = guild_cfg

        ctx = MagicMock()
        ctx.guild = MagicMock()

        tool = WebFetchToolCall(config=config, ctx=ctx)
        result = await tool._handle({"urls": "https://example.com"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests for web_answer tool call handling
# ---------------------------------------------------------------------------
class TestWebAnswerToolCallHandling:
    """Verify WebAnswerToolCall properly structures function responses."""

    def test_schema_has_correct_function_name(self):
        assert WebAnswerToolCall.function_name == "web_answer"
        assert WebAnswerToolCall.schema.function.name == "web_answer"

    def test_schema_has_question_parameter(self):
        params = WebAnswerToolCall.schema.function.parameters
        assert "question" in params.properties
        assert "question" in params.required

    def test_schema_type_is_function(self):
        assert WebAnswerToolCall.schema.type == "function"

    @pytest.mark.asyncio
    async def test_handle_empty_question_returns_error(self):
        config = MagicMock()
        ctx = MagicMock()
        tool = WebAnswerToolCall(config=config, ctx=ctx)
        result = await tool._handle({"question": ""})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_handle_with_none_backend_returns_error(self):
        """When backend is 'none', should return a NoneProvider error string."""
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.web_answer_backend = AsyncMock(return_value="none")
        guild_cfg.web_answer_config = AsyncMock(return_value=None)
        config.guild.return_value = guild_cfg

        ctx = MagicMock()
        ctx.guild = MagicMock()

        tool = WebAnswerToolCall(config=config, ctx=ctx)
        result = await tool._handle({"question": "What is Python?"})
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests for _process_and_add_tool_results exception handling
# ---------------------------------------------------------------------------
class TestProcessAndAddToolResultsExceptionHandling:
    """Verify _process_and_add_tool_results handles asyncio.gather exceptions."""

    @pytest.mark.asyncio
    async def test_exception_in_tool_does_not_crash(self, caplog):
        """If a tool raises an exception via asyncio.gather, it should be logged
        and not crash the pipeline."""
        pipeline = _make_pipeline()
        pipeline.enabled_tools = []
        pipeline.collected_images = []

        tc = _make_tool_call("call_1", "broken_tool", '{"arg": "val"}')
        pipeline.run_tool = AsyncMock(side_effect=RuntimeError("Tool exploded"))

        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            await pipeline._process_and_add_tool_results([tc])

        assert "unhandled exception" in caplog.text.lower() or "exception" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_successful_tool_result_has_correct_id_and_name(self, caplog):
        """Verify that a successful tool result is added with matching id and name."""
        pipeline = _make_pipeline()
        pipeline.enabled_tools = []
        pipeline.collected_images = []

        tc = _make_tool_call("call_abc", "web_search", '{"query": "test"}')
        pipeline.run_tool = AsyncMock(return_value="search results here")

        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            await pipeline._process_and_add_tool_results([tc])

        messages = pipeline.msg_list.messages
        tool_results = [m for m in messages if m.role == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].tool_call_id == "call_abc"
        assert tool_results[0].name == "web_search"
        assert tool_results[0].content == "search results here"

        assert "Adding tool result to context" in caplog.text
        assert "id=call_abc" in caplog.text
        assert "name=web_search" in caplog.text


# ---------------------------------------------------------------------------
# Tests for add_assistant and add_tool_result with string IDs
# ---------------------------------------------------------------------------
class TestAddAssistantAndToolResult:
    """Verify add_assistant and add_tool_result handle tool calls/responses correctly."""

    @pytest.mark.asyncio
    async def test_add_assistant_stores_tool_calls(self):
        ml = _make_messages_list()
        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        await ml.add_assistant(None, tool_calls=[tc], index=len(ml) + 1)
        assert len(ml.messages) == 1
        assert ml.messages[0].role == "assistant"
        assert ml.messages[0].tool_calls == [tc]

    @pytest.mark.asyncio
    async def test_add_tool_result_stores_id_and_name(self):
        ml = _make_messages_list()
        await ml.add_tool_result(
            content="search results",
            tool_call_id="call_xyz",
            name="web_search",
            index=len(ml) + 1,
        )
        assert len(ml.messages) == 1
        entry = ml.messages[0]
        assert entry.role == "tool"
        assert entry.tool_call_id == "call_xyz"
        assert entry.name == "web_search"
        assert entry.content == "search results"

    @pytest.mark.asyncio
    async def test_add_tool_result_with_multimodal_content(self):
        ml = _make_messages_list()
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Generated image"},
        ]
        await ml.add_tool_result(
            content=content,
            tool_call_id="call_img",
            name="generate_image",
            index=len(ml) + 1,
        )
        entry = ml.messages[0]
        assert entry.content == content
        assert entry.tool_call_id == "call_img"
        assert entry.name == "generate_image"

    @pytest.mark.asyncio
    async def test_full_cycle_assistant_then_tool_results(self):
        """Test the full pattern: assistant with 2 tool_calls, then 2 tool results."""
        ml = _make_messages_list()

        tc1 = _make_tool_call("call_1", "web_search", '{"query": "python"}')
        tc2 = _make_tool_call("call_2", "web_fetch", '{"urls": ["https://example.com"]}')

        await ml.add_assistant(None, tool_calls=[tc1, tc2], index=len(ml) + 1)
        await ml.add_tool_result(
            content="Python is a programming language",
            tool_call_id="call_1",
            name="web_search",
            index=len(ml) + 1,
        )
        await ml.add_tool_result(
            content="Example.com content here",
            tool_call_id="call_2",
            name="web_fetch",
            index=len(ml) + 1,
        )

        assert len(ml.messages) == 3
        assert ml.messages[0].role == "assistant"
        assert len(ml.messages[0].tool_calls) == 2
        assert ml.messages[1].role == "tool"
        assert ml.messages[1].tool_call_id == "call_1"
        assert ml.messages[1].name == "web_search"
        assert ml.messages[2].role == "tool"
        assert ml.messages[2].tool_call_id == "call_2"
        assert ml.messages[2].name == "web_fetch"


# ---------------------------------------------------------------------------
# Tests for Gemini 3.5 Flash function calling requirements
# ---------------------------------------------------------------------------
class TestGeminiFunctionCallingRequirements:
    """Verify compliance with Gemini 3.5 Flash strict response matching."""

    def test_id_and_name_match_in_get_json(self):
        tc = _make_tool_call("call_gemini_1", "web_search", '{"query": "test"}')
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "results", tool_call_id="call_gemini_1", name="web_search"),
        ])
        result = ml.get_json()

        assistant = result[0]
        assert "tool_calls" in assistant

        tool = result[1]
        assert tool["role"] == "tool"
        assert tool["tool_call_id"] == "call_gemini_1"
        assert tool["name"] == "web_search"

    def test_one_function_response_per_function_call(self):
        tc1 = _make_tool_call("call_1", "web_search")
        tc2 = _make_tool_call("call_2", "web_fetch")
        ml = _make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc1, tc2]),
            MessageEntry("tool", "result1", tool_call_id="call_1", name="web_search"),
            MessageEntry("tool", "result2", tool_call_id="call_2", name="web_fetch"),
        ])
        result = ml.get_json()

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 2
        ids = {m["tool_call_id"] for m in tool_results}
        assert ids == {"call_1", "call_2"}

    def test_all_three_web_tools_in_single_cycle(self):
        """Test a complete cycle with web_search, web_fetch, and web_answer."""
        tc1 = _make_tool_call("call_ws", "web_search", '{"query": "test"}')
        tc2 = _make_tool_call("call_wf", "web_fetch", '{"urls": ["https://a.com"]}')
        tc3 = _make_tool_call("call_wa", "web_answer", '{"question": "What?"}')

        ml = _make_messages_list(messages=[
            MessageEntry("system", "You are helpful."),
            MessageEntry("user", "Help me research"),
            MessageEntry("assistant", None, tool_calls=[tc1, tc2, tc3]),
            MessageEntry("tool", "Search results", tool_call_id="call_ws", name="web_search"),
            MessageEntry("tool", "Page content", tool_call_id="call_wf", name="web_fetch"),
            MessageEntry("tool", "The answer is 42", tool_call_id="call_wa", name="web_answer"),
            MessageEntry("assistant", "Based on my research, the answer is 42."),
        ])

        result = ml.get_json()
        assert len(result) == 7

        assistant_msg = result[2]
        assert len(assistant_msg["tool_calls"]) == 3

        for i, (expected_id, expected_name) in enumerate([
            ("call_ws", "web_search"),
            ("call_wf", "web_fetch"),
            ("call_wa", "web_answer"),
        ]):
            tool_msg = result[3 + i]
            assert tool_msg["role"] == "tool"
            assert tool_msg["tool_call_id"] == expected_id
            assert tool_msg["name"] == expected_name


# ---------------------------------------------------------------------------
# Tests for logging in _process_and_add_tool_results
# ---------------------------------------------------------------------------
class TestToolResultLogging:
    """Verify structured logging during tool result processing."""

    @pytest.mark.asyncio
    async def test_logs_tool_call_id_and_name(self, caplog):
        pipeline = _make_pipeline()
        pipeline.enabled_tools = []
        pipeline.collected_images = []
        pipeline.run_tool = AsyncMock(return_value="result text")

        tc = _make_tool_call("call_log_test", "web_search", '{"query": "test"}')

        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            await pipeline._process_and_add_tool_results([tc])

        assert "Processing tool call: id=call_log_test name=web_search" in caplog.text
        assert "Tool result: id=call_log_test name=web_search" in caplog.text
        assert "Adding tool result to context: id=call_log_test name=web_search" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_tool_arguments(self, caplog):
        pipeline = _make_pipeline()
        pipeline.enabled_tools = []
        pipeline.collected_images = []
        pipeline.run_tool = AsyncMock(return_value="result")

        tc = _make_tool_call("call_args", "web_search", '{"query": "hello world"}')

        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            await pipeline._process_and_add_tool_results([tc])

        assert "Executing tool: id=call_args name=web_search" in caplog.text
        assert "hello world" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_multiple_tools(self, caplog):
        pipeline = _make_pipeline()
        pipeline.enabled_tools = []
        pipeline.collected_images = []
        pipeline.run_tool = AsyncMock(return_value="result")

        tc1 = _make_tool_call("call_a", "web_search", '{"query": "a"}')
        tc2 = _make_tool_call("call_b", "web_fetch", '{"urls": ["https://b.com"]}')

        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            await pipeline._process_and_add_tool_results([tc1, tc2])

        assert "Processing 2 local tool call(s)" in caplog.text
        assert "id=call_a name=web_search" in caplog.text
        assert "id=call_b name=web_fetch" in caplog.text
