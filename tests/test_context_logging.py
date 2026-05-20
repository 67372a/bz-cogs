"""Tests for the context/message-list logging in MessagesList and LLMPipeline.

Verifies that:
- MessagesList.log_messages() logs the full context with per-message detail
- MessagesList._summarize_content() correctly truncates/summarizes content
- LLMPipeline._log_submitted_payload() logs the full submitted payload
- LLMPipeline._summarize_content_for_log() correctly truncates/summarizes content
- The logging helper methods handle edge cases (empty content, images, tool calls)
"""

import logging
import sys
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

# Mock other aiuser dependencies that messages.py imports
_mocks_needed = [
    "aiuser.config.defaults",
    "aiuser.config.models",
    "aiuser.config.constants",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.entry",
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

# Load real LLMPipeline (need additional mocks for its dependencies)
_pipeline_mocks = [
    "aiuser.functions.tool_call",
    "aiuser.functions.types",
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.openrouter",
    "aiuser.functions.openrouter.web_search",
    "aiuser.functions.openrouter.web_fetch",
    "aiuser.functions.openrouter.image_generation",
    "aiuser.functions.openrouter.pdf_parsing",
    "aiuser.functions.openrouter.image_parsing",
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

sys.modules["aiuser.response.chat.llm_pipeline"] = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
LLMPipeline = sys.modules["aiuser.response.chat.llm_pipeline"].LLMPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_ctx(channel_id=123, guild_id=456, bot_user_id=789):
    """Create a minimal mock context for MessagesList/LLMPipeline."""
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
    """Create a minimal mock cog for MessagesList."""
    cog = MagicMock()
    cog.bot = MagicMock()
    cog.config = MagicMock()
    cog.ignore_regex = {}
    cog.override_prompt_start_time = {}
    return cog


# ---------------------------------------------------------------------------
# Tests for MessagesList._summarize_content
# ---------------------------------------------------------------------------
class TestSummarizeContent:
    """Tests for MessagesList._summarize_content()."""

    def test_short_string_returned_as_is(self):
        ml = MessagesList.__new__(MessagesList)
        result = ml._summarize_content("hello world", max_text_len=500)
        assert result == "hello world"

    def test_long_string_truncated(self):
        ml = MessagesList.__new__(MessagesList)
        long_text = "a" * 1000
        result = ml._summarize_content(long_text, max_text_len=500)
        assert result.startswith("a" * 500)
        assert "truncated" in result
        assert "1000 chars total" in result

    def test_content_list_with_text_item(self):
        ml = MessagesList.__new__(MessagesList)
        content = [{"type": "text", "text": "hello"}]
        result = ml._summarize_content(content)
        assert "text:" in result
        assert "hello" in result

    def test_content_list_with_long_text_truncated(self):
        ml = MessagesList.__new__(MessagesList)
        long_text = "x" * 1000
        content = [{"type": "text", "text": long_text}]
        result = ml._summarize_content(content, max_text_len=200)
        assert "x" * 200 in result
        assert "1000 chars" in result

    def test_content_list_with_base64_image(self):
        ml = MessagesList.__new__(MessagesList)
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}]
        result = ml._summarize_content(content)
        assert "base64 data" in result
        assert "5" in result  # length is in there

    def test_content_list_with_url_image(self):
        ml = MessagesList.__new__(MessagesList)
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
        result = ml._summarize_content(content)
        assert "https://example.com/image.png" in result

    def test_content_list_mixed(self):
        ml = MessagesList.__new__(MessagesList)
        content = [
            {"type": "text", "text": "look at this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = ml._summarize_content(content)
        assert "look at this:" in result
        assert "https://example.com/img.png" in result

    def test_empty_string(self):
        ml = MessagesList.__new__(MessagesList)
        result = ml._summarize_content("")
        assert result == ""

    def test_non_string_non_list(self):
        ml = MessagesList.__new__(MessagesList)
        result = ml._summarize_content(42)
        assert result == "42"

    def test_empty_list(self):
        ml = MessagesList.__new__(MessagesList)
        result = ml._summarize_content([])
        assert result == "[]"


# ---------------------------------------------------------------------------
# Tests for MessagesList.log_messages
# ---------------------------------------------------------------------------
class TestLogMessages:
    """Tests for MessagesList.log_messages()."""

    def _make_messages_list(self, messages=None, prefill=None, model="gpt-4", tokens=100, token_limit=8000):
        """Create a MessagesList with minimal setup for testing log_messages."""
        ml = MessagesList.__new__(MessagesList)
        ml.ctx = _make_mock_ctx()
        ml.model = model
        ml.tokens = tokens
        ml.token_limit = token_limit
        ml.messages = messages or []
        ml.prefill = prefill
        return ml

    def test_log_messages_with_user_and_system(self, caplog):
        ml = self._make_messages_list(messages=[
            MessageEntry("system", "You are a helpful assistant."),
            MessageEntry("user", "Hello!"),
        ])
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "Message List Start" in caplog.text
        assert "role=system" in caplog.text
        assert "You are a helpful assistant." in caplog.text
        assert "role=user" in caplog.text
        assert "Hello!" in caplog.text
        assert "Message List End" in caplog.text

    def test_log_messages_with_tool_calls(self, caplog):
        tc = MagicMock()
        tc.function.name = "search"
        ml = self._make_messages_list(messages=[
            MessageEntry("assistant", None, tool_calls=[tc]),
        ])
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "tool_calls=" in caplog.text
        assert "search" in caplog.text

    def test_log_messages_with_tool_result(self, caplog):
        ml = self._make_messages_list(messages=[
            MessageEntry("tool", "search results here", tool_call_id="call_123", name="search"),
        ])
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "role=tool" in caplog.text
        assert "tool_call_id=call_123" in caplog.text
        assert "name=search" in caplog.text
        assert "search results here" in caplog.text

    def test_log_messages_with_prefill(self, caplog):
        ml = self._make_messages_list(
            messages=[MessageEntry("system", "sys prompt")],
            prefill="Hello, I am"
        )
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "prefill" in caplog.text
        assert "Hello, I am" in caplog.text

    def test_log_messages_summary_line(self, caplog):
        ml = self._make_messages_list(
            messages=[MessageEntry("user", "hi")],
            model="gpt-4o",
            tokens=500,
            token_limit=8000,
        )
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "model=gpt-4o" in caplog.text
        assert "tokens=500/8000" in caplog.text
        assert "messages=1" in caplog.text

    def test_log_messages_no_prefill(self, caplog):
        ml = self._make_messages_list(messages=[MessageEntry("user", "hi")], prefill=None)
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        # Should NOT log a prefill line
        assert "[prefill]" not in caplog.text

    def test_log_messages_channel_and_guild_ids(self, caplog):
        ml = self._make_messages_list(messages=[])
        ml.ctx = _make_mock_ctx(channel_id=42, guild_id=99)
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "channel=42" in caplog.text
        assert "guild=99" in caplog.text

    def test_log_messages_multimodal_content(self, caplog):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        ml = self._make_messages_list(messages=[
            MessageEntry("user", content),
        ])
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "describe this" in caplog.text
        assert "base64 data" in caplog.text

    def test_log_messages_empty_list(self, caplog):
        ml = self._make_messages_list(messages=[])
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            ml.log_messages()
        assert "Message List Start" in caplog.text
        assert "messages=0" in caplog.text
        assert "Message List End" in caplog.text


# ---------------------------------------------------------------------------
# Tests for LLMPipeline._summarize_content_for_log
# ---------------------------------------------------------------------------
class TestPipelineSummarizeContent:
    """Tests for LLMPipeline._summarize_content_for_log()."""

    def test_short_string(self):
        result = LLMPipeline._summarize_content_for_log("hello", max_text_len=500)
        assert result == "hello"

    def test_long_string_truncated(self):
        result = LLMPipeline._summarize_content_for_log("a" * 1000, max_text_len=500)
        assert result.startswith("a" * 500)
        assert "truncated" in result

    def test_base64_image_in_list(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 5000}}]
        result = LLMPipeline._summarize_content_for_log(content)
        assert "base64 data" in result

    def test_url_image_in_list(self):
        content = [{"type": "image_url", "image_url": {"url": "https://img.example.com/a.png"}}]
        result = LLMPipeline._summarize_content_for_log(content)
        assert "https://img.example.com/a.png" in result

    def test_none_content(self):
        result = LLMPipeline._summarize_content_for_log(None)
        assert result == "None"


# ---------------------------------------------------------------------------
# Tests for LLMPipeline._log_submitted_payload
# ---------------------------------------------------------------------------
class TestLogSubmittedPayload:
    """Tests for LLMPipeline._log_submitted_payload()."""

    def _make_pipeline(self, channel_id=123, guild_id=456):
        pipeline = LLMPipeline.__new__(LLMPipeline)
        pipeline.ctx = _make_mock_ctx(channel_id=channel_id, guild_id=guild_id)
        return pipeline

    def test_logs_model_and_message_count(self, caplog):
        pipeline = self._make_pipeline()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4o", messages, "abc123def456", {})
        assert "Submitted Payload Start" in caplog.text
        assert "model=gpt-4o" in caplog.text
        assert "messages=2" in caplog.text

    def test_logs_kwargs(self, caplog):
        pipeline = self._make_pipeline()
        kwargs = {"temperature": 0.7, "max_tokens": 100}
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", [], "abc123", kwargs)
        assert "temperature" in caplog.text
        assert "0.7" in caplog.text

    def test_sanitizes_extra_body_safety_settings(self, caplog):
        """safetySettings and session_id should be stripped from the logged kwargs."""
        pipeline = self._make_pipeline()
        kwargs = {
            "extra_body": {
                "safetySettings": [{"category": "HARM"}],
                "session_id": "secret123",
                "plugins": [{"id": "test"}],
            }
        }
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", [], "abc", kwargs)
        assert "HARM" not in caplog.text
        assert "secret123" not in caplog.text

    def test_logs_per_message_detail_at_debug(self, caplog):
        pipeline = self._make_pipeline()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", messages, "abc", {})
        assert "role=system" in caplog.text
        assert "role=user" in caplog.text
        assert "role=assistant" in caplog.text

    def test_logs_tool_calls_in_debug(self, caplog):
        pipeline = self._make_pipeline()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
                ],
            }
        ]
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", messages, "abc", {})
        assert "tool_calls=" in caplog.text
        assert "search" in caplog.text

    def test_logs_tool_result_messages(self, caplog):
        pipeline = self._make_pipeline()
        messages = [
            {"role": "tool", "content": "result text", "tool_call_id": "call_1", "name": "search"},
        ]
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", messages, "abc", {})
        assert "role=tool" in caplog.text
        assert "tool_call_id=call_1" in caplog.text
        assert "name=search" in caplog.text

    def test_logs_channel_and_guild(self, caplog):
        pipeline = self._make_pipeline(channel_id=777, guild_id=888)
        with caplog.at_level(logging.INFO, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", [], "abc", {})
        assert "channel=777" in caplog.text
        assert "guild=888" in caplog.text

    def test_logs_multimodal_message_content(self, caplog):
        pipeline = self._make_pipeline()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        with caplog.at_level(logging.DEBUG, logger="red.bz_cogs.aiuser"):
            pipeline._log_submitted_payload("gpt-4", messages, "abc", {})
        assert "describe" in caplog.text
        assert "base64 data" in caplog.text


# ---------------------------------------------------------------------------
# Tests for system prompt index 0 guarantee
# ---------------------------------------------------------------------------
class TestSystemPromptIndex:
    """Verify that the system prompt is always at index 0 in the messages list."""

    def _make_minimal_messages_list(self):
        """Create a MessagesList with minimal setup for direct method testing."""
        ml = MessagesList.__new__(MessagesList)
        ml.messages = []
        ml.messages_ids = set()
        ml.tokens = 0
        ml._encoding = MagicMock()
        ml._encoding.encode = MagicMock(return_value=[1, 2, 3])
        ml.token_limit = 8000
        return ml

    @pytest.mark.asyncio
    async def test_add_system_at_index_zero(self):
        """add_system with no explicit index inserts at index 0."""
        ml = self._make_minimal_messages_list()
        await ml.add_system("You are a helpful assistant.")
        assert len(ml.messages) == 1
        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_add_system_with_explicit_index(self):
        """add_system with an explicit index inserts at that index."""
        ml = self._make_minimal_messages_list()
        await ml.add_system("system prompt")
        await ml.add_system("another system prompt", index=1)
        assert ml.messages[0].content == "system prompt"
        assert ml.messages[1].content == "another system prompt"

    @pytest.mark.asyncio
    async def test_add_assistant_index_zero_preserved(self):
        """add_assistant with index=0 inserts before existing messages."""
        ml = self._make_minimal_messages_list()
        ml.messages.append(MessageEntry("system", "sys"))
        ml.messages.append(MessageEntry("user", "user"))
        await ml.add_assistant("reply", index=0)
        assert ml.messages[0].role == "assistant"
        assert ml.messages[1].role == "system"

    @pytest.mark.asyncio
    async def test_add_tool_result_index_zero_preserved(self):
        """add_tool_result with index=0 inserts before existing messages."""
        ml = self._make_minimal_messages_list()
        ml.messages.append(MessageEntry("system", "sys"))
        ml.messages.append(MessageEntry("user", "user"))
        await ml.add_tool_result("result", tool_call_id="call_1", name="search", index=0)
        assert ml.messages[0].role == "tool"
        assert ml.messages[0].tool_call_id == "call_1"
        assert ml.messages[1].role == "system"

    @pytest.mark.asyncio
    async def test_system_prompt_stays_at_index_zero_after_other_inserts(self):
        """System prompt remains at index 0 after inserting messages at other indices."""
        ml = self._make_minimal_messages_list()
        await ml.add_system("system prompt")
        # Simulate adding history messages at index 1
        await ml._add_tokens("msg1")
        ml.messages.insert(1, MessageEntry("user", "msg1"))
        await ml._add_tokens("msg2")
        ml.messages.insert(2, MessageEntry("user", "msg2"))
        # System prompt should still be at index 0
        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == "system prompt"

    @pytest.mark.asyncio
    async def test_init_message_comes_after_system_prompt(self):
        """Verify init_message is inserted at index=1 (after system prompt)."""
        ml = self._make_minimal_messages_list()
        # First add system prompt at index 0
        await ml.add_system("system prompt")
        # Then add a user message at index=1 (as _init now does)
        await ml._add_tokens("user msg")
        ml.messages.insert(1, MessageEntry("user", "user msg"))
        # Verify order: system at 0, user at 1
        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == "system prompt"
        assert ml.messages[1].role == "user"
        assert ml.messages[1].content == "user msg"

    @pytest.mark.asyncio
    async def test_index_zero_is_distinct_from_none(self):
        """Verify that index=0 and index=None behave the same (both insert at 0),
        but don't silently shift the system prompt."""
        ml = self._make_minimal_messages_list()
        await ml.add_system("sys prompt")
        assert ml.messages[0].role == "system"

        # add_msg with index=1 should not displace system prompt
        ml.messages_ids = set()
        ml2 = self._make_minimal_messages_list()
        ml2.messages = list(ml.messages)  # copy
        ml2.messages_ids = set()
        # Simulate inserting at index=1
        await ml2._add_tokens("user msg")
        ml2.messages.insert(1, MessageEntry("user", "user msg"))
        assert ml2.messages[0].role == "system"
        assert ml2.messages[1].content == "user msg"


# ---------------------------------------------------------------------------
# Tests for create_messages_list calling log_messages
# ---------------------------------------------------------------------------
class TestCreateMessagesListLogging:
    """Verify that create_messages_list calls log_messages on the returned MessagesList."""

    @pytest.mark.asyncio
    async def test_create_messages_list_calls_log_messages(self):
        """create_messages_list should call log_messages() on the built list."""
        create_messages_list = sys.modules["aiuser.messages_list.messages"].create_messages_list

        mock_cog = _make_mock_cog()
        mock_cog.backfill_anchors = {}
        ctx = _make_mock_ctx()

        # Mock the MessagesList._init, add_history, _append_dynamic_context
        with patch.object(MessagesList, "_init", new_callable=AsyncMock) as mock_init, \
             patch.object(MessagesList, "add_history", new_callable=AsyncMock) as mock_history, \
             patch.object(MessagesList, "_append_dynamic_context") as mock_dyn, \
             patch.object(MessagesList, "log_messages") as mock_log:
            result = await create_messages_list(mock_cog, ctx)
            mock_log.assert_called_once()
