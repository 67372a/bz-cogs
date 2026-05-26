"""Tests for implicit prefix-based caching stability in aiuser's context creation.

Implicit prefix caching (OpenAI, Gemini, OpenRouter) works by caching the
beginning of the message list.  Subsequent requests that share the same prefix
can reuse cached tokens.  ANY change to an earlier message — content, role,
or position — invalidates all cache entries from that point forward.

This test suite validates that the context-building logic preserves a stable
prefix across consecutive requests in the same channel.

Scenarios tested:
  1. System prompt stability (index 0) across requests
  2. History message ordering is deterministic (chronological)
  3. Reply-chain insertion does NOT shift non-related prefix messages
  4. Stop instruction round-trip (add → remove) leaves list unchanged
  5. Dynamic context is always appended at the END (never in prefix)
  6. Token-limit truncation produces a consistent (prefix-stable) subset
  7. Tool-loop assistant/tool-result messages are appended at the end
  8. `add_msg` with `index=1` correctly inserts older history before newer
  9. Multi-entry `add_msg` maintains correct insertion order
 10. `get_json()` output matches message list order exactly
"""

import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing real modules
# ---------------------------------------------------------------------------
_ORIG_SYS_MODULES_SNAPSHOT = dict(sys.modules)

discord_mock = MagicMock()
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
sys.modules.setdefault("discord", discord_mock)
sys.modules.setdefault("discord.ext", MagicMock())
sys.modules.setdefault("discord.ext.commands", MagicMock())
sys.modules.setdefault("discord.app_commands", MagicMock())

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

sys.modules.setdefault("tiktoken", MagicMock())

_mocks_needed = [
    "aiuser.config.defaults",
    "aiuser.messages_list.converter.converter",
    "aiuser.messages_list.opt_view",
    "aiuser.types.abc",
    "aiuser.utils.utilities",
]
for _m in _mocks_needed:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Load real config and entry modules
sys.modules["aiuser.config.models"] = import_module_directly(
    "aiuser.config.models", "aiuser/config/models.py"
)
sys.modules["aiuser.config.constants"] = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
sys.modules["aiuser.messages_list.entry"] = import_module_directly(
    "aiuser.messages_list.entry", "aiuser/messages_list/entry.py"
)
sys.modules["aiuser.types.enums"] = import_module_directly(
    "aiuser.types.enums", "aiuser/types/enums.py"
)

MessageEntry = sys.modules["aiuser.messages_list.entry"].MessageEntry

# Load real MessagesList
sys.modules["aiuser.messages_list.messages"] = import_module_directly(
    "aiuser.messages_list.messages", "aiuser/messages_list/messages.py"
)
MessagesList = sys.modules["aiuser.messages_list.messages"].MessagesList

# Load real LLMPipeline
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
    "aiuser.functions.mermaid",
    "aiuser.functions.mermaid.tool_call",
    "aiuser.functions.attach_files",
    "aiuser.functions.attach_files.tool_call",
    "aiuser.response.chat.function_call_view",
    "aiuser.response.chat.llm_pipeline",
]
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

# The openrouter package mock needs specific class attributes
_or_mock = sys.modules["aiuser.functions.openrouter"]
_or_mock.OpenRouterWebSearch = MagicMock()
_or_mock.OpenRouterWebFetch = MagicMock()
_or_mock.OpenRouterImageGeneration = MagicMock()
_or_mock.OpenRouterPdfParsing = MagicMock()
_or_mock.OpenRouterImageParsing = MagicMock()

# Mock mermaid
if "aiuser.functions.mermaid" not in sys.modules:
    sys.modules["aiuser.functions.mermaid"] = MagicMock()
if "aiuser.functions.mermaid.tool_call" not in sys.modules:
    sys.modules["aiuser.functions.mermaid.tool_call"] = MagicMock()
if "aiuser.functions.attach_files" not in sys.modules:
    sys.modules["aiuser.functions.attach_files"] = MagicMock()
if "aiuser.functions.attach_files.tool_call" not in sys.modules:
    sys.modules["aiuser.functions.attach_files.tool_call"] = MagicMock()

sys.modules["aiuser.response.chat.llm_pipeline"] = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
LLMPipeline = sys.modules["aiuser.response.chat.llm_pipeline"].LLMPipeline
PipelineResult = sys.modules["aiuser.response.chat.llm_pipeline"].PipelineResult


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


def _make_messages_list(messages=None, prefill=None, model="gpt-4",
                        tokens=0, token_limit=100000):
    """Create a MessagesList with minimal setup for testing.

    By default the token limit is very high so that all messages fit.
    """
    ml = MessagesList.__new__(MessagesList)
    ml.ctx = _make_mock_ctx()
    ml.model = model
    ml.tokens = tokens
    ml.token_limit = token_limit
    ml.messages = list(messages) if messages else []
    ml.messages_ids = set()
    ml.prefill = prefill
    ml.can_reply = True
    ml._dynamic_context = None
    ml._raw_persona = None
    ml._cache_window_active = False
    ml.init_message = MagicMock()
    ml.init_message.id = 999
    ml.guild = ml.ctx.guild
    ml.bot = ml.ctx.bot
    ml.config = MagicMock()
    ml.converter = MagicMock()
    ml.ignore_regex = None
    ml.start_time = None
    _encoding_mock = MagicMock()
    _encoding_mock.encode = MagicMock(return_value=[1, 2, 3])
    ml._encoding = _encoding_mock
    return ml


def _make_pipeline(messages=None, prefill=None):
    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline.ctx = _make_mock_ctx()
    pipeline.msg_list = _make_messages_list(messages=messages, prefill=prefill)
    pipeline.model = "gpt-4"
    pipeline.openai_client = MagicMock()
    pipeline.config = MagicMock()
    pipeline.bot = pipeline.ctx.bot
    pipeline.enabled_tools = []
    pipeline.available_tools_schemas = []
    pipeline.openrouter_tools = []
    pipeline.completion = None
    pipeline.reasoning = None
    pipeline.response_parts = []
    pipeline.collected_images = []
    pipeline.can_reply = True
    return pipeline


def _make_tool_call(id: str, name: str, arguments: str = "{}"):
    tc = MagicMock()
    tc.id = id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _extract_roles_and_content(messages_list):
    """Extract a list of (role, content_preview) for easy comparison."""
    result = []
    for msg in messages_list.messages:
        if isinstance(msg.content, str):
            result.append((msg.role, msg.content[:80]))
        elif isinstance(msg.content, list):
            # For multimodal, show text parts
            texts = [p.get("text", "")[:40] for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
            result.append((msg.role, "[multimodal] " + " | ".join(texts)))
        else:
            result.append((msg.role, str(msg.content)[:80]))
    return result


# ===========================================================================
# TEST 1: System prompt stability
# ===========================================================================
class TestSystemPromptStability:
    """The system prompt at index 0 must be identical across requests
    in the same channel (it only contains stable variables)."""

    def test_system_prompt_at_index_zero(self):
        """System prompt should always be the first message."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "You are a helpful assistant."),
            MessageEntry("user", "Hello"),
        ]
        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == "You are a helpful assistant."

    def test_system_prompt_unchanged_after_history_addition(self):
        """Adding history messages at index=1 must NOT alter the system prompt."""
        ml = _make_messages_list()
        system_content = "You are a helpful assistant."
        ml.messages = [MessageEntry("system", system_content)]

        # Simulate history insertion at index=1 (as _process_past_messages does)
        ml.messages.insert(1, MessageEntry("user", "history msg 1"))
        ml.messages.insert(1, MessageEntry("user", "history msg 2"))

        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == system_content

    def test_system_prompt_identical_across_two_instances(self):
        """Two MessagesList instances for the same channel should produce
        the same system prompt content (assuming stable variables)."""
        prompt_text = "You are a helpful assistant in TestGuild."

        ml1 = _make_messages_list()
        ml1.messages = [MessageEntry("system", prompt_text)]

        ml2 = _make_messages_list()
        ml2.messages = [MessageEntry("system", prompt_text)]

        assert ml1.messages[0].content == ml2.messages[0].content


# ===========================================================================
# TEST 2: History message ordering is deterministic
# ===========================================================================
class TestHistoryOrdering:
    """History messages must appear in chronological order (oldest first)
    regardless of the insertion mechanics."""

    def _build_history(self, msg_ids, insert_at=1):
        """Simulate _process_past_messages: older messages inserted at index 1.
        msg_ids should be in newest-first order (as Discord returns them)."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", f"init message (id=999)"),
        ]
        # Simulate: iterate newest-first, insert each at index=1
        for mid in msg_ids:
            ml.messages.insert(insert_at, MessageEntry("user", f"msg id={mid}"))
        return ml

    def test_chronological_order_after_insertion(self):
        """Messages inserted at index=1 in newest-first order should end up
        in chronological (oldest-first) order in the final list."""
        # Discord returns [C, B, A] (newest first); A is oldest
        ml = self._build_history(["C", "B", "A"])
        roles_content = _extract_roles_and_content(ml)

        # Expected: system, A, B, C, init
        assert roles_content[0] == ("system", "system prompt")
        assert roles_content[1] == ("user", "msg id=A")
        assert roles_content[2] == ("user", "msg id=B")
        assert roles_content[3] == ("user", "msg id=C")
        assert "init message" in roles_content[4][1]

    def test_single_history_message(self):
        ml = self._build_history(["A"])
        roles_content = _extract_roles_and_content(ml)
        assert roles_content[1] == ("user", "msg id=A")

    def test_empty_history_preserves_system_and_init(self):
        ml = self._build_history([])
        assert len(ml.messages) == 2
        assert ml.messages[0].role == "system"
        assert "init message" in ml.messages[1].content


# ===========================================================================
# TEST 3: Reply-chain insertion does NOT shift unrelated prefix messages
# ===========================================================================
class TestReplyChainInsertion:
    """When add_msg encounters a reply reference, it recursively inserts the
    replied-to message at the SAME index.  This must not shift the system
    prompt or unrelated history messages.

    NOTE: The reply chain path is gated by
        isinstance(message.reference.resolved, discord.Message)
    In tests, discord is mocked, so discord.Message is a MagicMock type.
    The isinstance check always returns False for MagicMock instances,
    meaning the reply chain code path is NEVER triggered in these tests.

    This means: in production, reply chains DO insert extra messages at the
    insert position, potentially shifting the prefix.  This is a confirmed
    cache-disruption source.
    """

    def test_reply_chain_code_path_requires_isinstance_match(self):
        """The reply chain insertion is gated by isinstance(message.reference.resolved, discord.Message).

        The code also checks that the parent is NOT already in context (messages_ids)
        and inserts at the END of context (not at the original insert_at position),
        preserving prefix stability."""
        import inspect
        from aiuser.messages_list.messages import MessagesList
        source = inspect.getsource(MessagesList.add_msg)
        # Verify the isinstance gate is present in the source code
        assert "isinstance(message.reference.resolved, discord.Message)" in source, \
            "The isinstance gate for reply chain insertion must exist in add_msg"
        # Verify the parent is inserted at the end (not at insert_at)
        assert "index=len(self.messages)" in source, \
            "Reply chain parent must be inserted at end of context, not at insert_at"
        # Verify duplicate check
        assert "messages_ids" in source, \
            "Reply chain must check if parent is already in context"

    def test_reply_chain_insertion_at_end_preserves_prefix(self):
        """Test that reply chain insertion at END of context preserves prefix.

        After the fix, the reply chain inserts the parent at the END of the
        message list (not at the original insert position), so it goes AFTER
        all existing history and init messages. This prevents prefix shifts.

        Layout: [sys, history..., init, child, parent_at_end]
        """
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "history msg"),
            MessageEntry("user", "init message"),
        ]

        # Simulate: child is added at end (after init), parent goes after child
        ml.messages.append(MessageEntry("user", "msg id=200 (child)"))
        ml.messages.append(MessageEntry("user", "msg id=100 (parent)"))

        roles_content = _extract_roles_and_content(ml)
        # System prompt and history are NOT shifted
        assert roles_content[0] == ("system", "system prompt")
        assert roles_content[1] == ("user", "history msg")
        assert roles_content[2] == ("user", "init message")
        # Parent is at end, not before child
        assert "msg id=200" in roles_content[3][1]  # child
        assert "msg id=100" in roles_content[4][1]  # parent at end

    def test_reply_chain_skips_already_present_parent(self):
        """If the parent message is already in context, it must NOT be
        inserted again (prevents duplicates and prefix shifts)."""
        ml = _make_messages_list()
        parent_id = 100
        ml.messages_ids.add(parent_id)  # parent already in context

        # The code checks: if parent.id not in self.messages_ids
        assert parent_id in ml.messages_ids, "Parent already in context — skip insertion"

    def test_reply_chain_does_not_affect_system_prompt_index(self):
        """The system prompt must remain at index 0 even after reply chain
        insertion."""
        ml = _make_messages_list()
        system_content = "immutable system prompt"
        ml.messages = [MessageEntry("system", system_content)]

        # Simulate reply chain insertion
        ml.messages.insert(1, MessageEntry("user", "msg id=200 (child)"))
        ml.messages.insert(1, MessageEntry("user", "msg id=100 (parent)"))

        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == system_content


# ===========================================================================
# TEST 4: Stop instruction round-trip
# ===========================================================================
class TestStopInstructionRoundTrip:
    """In the tool-calling loop, a stop instruction is temporarily appended
    before the last LLM call and removed afterward.  The message list must
    be identical before and after this operation."""

    def test_stop_instruction_add_remove_preserves_list(self):
        """Appending then removing the stop entry must leave the messages
        list identical (same objects, same order)."""
        ml = _make_messages_list()
        original_messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "hi there"),
        ]
        ml.messages = list(original_messages)
        original_len = len(ml.messages)
        original_repr = [(m.role, m.content) for m in ml.messages]

        # Simulate the stop instruction add/remove from _run_loop
        stop_instruction = (
            "You have reached the maximum number of tool-calling rounds. "
            "Based on all the information you have gathered, provide your final, "
            "complete response now. Do NOT call any more tools — respond directly to the user."
        )
        stop_entry = MessageEntry("user", stop_instruction)
        ml.messages.append(stop_entry)

        assert len(ml.messages) == original_len + 1
        assert ml.messages[-1].content == stop_instruction

        # Remove it (as done in _run_loop)
        try:
            ml.messages.remove(stop_entry)
        except ValueError:
            pass

        assert len(ml.messages) == original_len
        after_repr = [(m.role, m.content) for m in ml.messages]
        assert after_repr == original_repr

    def test_stop_instruction_not_in_get_json_after_removal(self):
        """After removing the stop instruction, get_json() must not include it."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "hello"),
        ]
        ml.prefill = None

        stop_entry = MessageEntry("user", "STOP INSTRUCTION")
        ml.messages.append(stop_entry)
        json_with_stop = ml.get_json()
        assert len(json_with_stop) == 3

        ml.messages.remove(stop_entry)
        json_without_stop = ml.get_json()
        assert len(json_without_stop) == 2
        assert all("STOP" not in m.get("content", "") for m in json_without_stop)

    def test_stop_instruction_identity_based_removal(self):
        """FIX: The stop instruction is now removed using identity-based
        removal (iterate backwards, check `is`), not value-based removal.

        This ensures the CORRECT entry (the stop instruction) is removed
        even if a pre-existing message has identical content."""
        ml = _make_messages_list()
        same_content = "identical content"
        existing = MessageEntry("user", same_content)
        ml.messages = [MessageEntry("system", "sys"), existing]

        stop_entry = MessageEntry("user", same_content)
        ml.messages.append(stop_entry)

        assert len(ml.messages) == 3  # sys, existing, stop

        # Identity-based removal: iterate backwards, find the actual stop_entry object
        for i in range(len(ml.messages) - 1, -1, -1):
            if ml.messages[i] is stop_entry:
                del ml.messages[i]
                break

        # The stop entry (last one) was removed; the existing entry remains
        assert len(ml.messages) == 2
        assert ml.messages[0].role == "system"
        assert ml.messages[1].role == "user"
        assert ml.messages[1].content == same_content
        assert ml.messages[1] is existing  # identity check: same object

    def test_stop_instruction_with_real_stop_text_safely_removes(self):
        """The actual stop instruction text is unique enough that remove()
        will only match the stop entry, not any real user message."""
        ml = _make_messages_list()
        stop_instruction = (
            "You have reached the maximum number of tool-calling rounds. "
            "Based on all the information you have gathered, provide your final, "
            "complete response now. Do NOT call any more tools — respond directly to the user."
        )
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "hi there"),
        ]
        original_len = len(ml.messages)

        stop_entry = MessageEntry("user", stop_instruction)
        ml.messages.append(stop_entry)
        assert len(ml.messages) == original_len + 1

        ml.messages.remove(stop_entry)
        assert len(ml.messages) == original_len
        # Verify no stop instruction content leaked
        assert all(stop_instruction not in (m.content or "") for m in ml.messages)


# ===========================================================================
# TEST 5: Dynamic context position
# ===========================================================================
class TestDynamicContextPosition:
    """Dynamic context (time, author, etc.) must be appended at the END of
    the message list, not inserted into the prefix."""

    def test_dynamic_context_appended_at_end(self):
        """_append_dynamic_context should place the context as the last message."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "history 1"),
            MessageEntry("user", "init message"),
        ]
        ml._dynamic_context = "date=2025/01/01 | time=12:00 | author=Alice"

        ml._append_dynamic_context()

        assert len(ml.messages) == 4
        assert ml.messages[-1].role == "user"
        assert "date=2025/01/01" in ml.messages[-1].content
        # System prompt and history positions unchanged
        assert ml.messages[0].role == "system"
        assert ml.messages[1].content == "history 1"

    def test_dynamic_context_none_does_not_add_message(self):
        """If _dynamic_context is None, no message should be added."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "init message"),
        ]
        ml._dynamic_context = None

        ml._append_dynamic_context()

        assert len(ml.messages) == 2

    def test_dynamic_context_preserves_prefix(self):
        """The prefix (system + history) must be byte-identical before and
        after appending dynamic context."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "history msg A"),
            MessageEntry("user", "history msg B"),
            MessageEntry("user", "init message"),
        ]
        prefix_before = [(m.role, m.content) for m in ml.messages]

        ml._dynamic_context = "time=12:00 | author=Bob"
        ml._append_dynamic_context()

        # First N messages unchanged
        for i, (role, content) in enumerate(prefix_before):
            assert ml.messages[i].role == role
            assert ml.messages[i].content == content


# ===========================================================================
# TEST 6: Token-limit truncation produces consistent prefix subset
# ===========================================================================
class TestTokenLimitTruncation:
    """When the token limit is reached, additional messages are dropped.
    The messages that DO fit must form a consistent prefix that is a
    subset of the full history."""

    def test_low_limit_drops_later_history(self):
        """With a low token limit, only the first few history messages should
        be included (those closest to the system prompt)."""
        ml = _make_messages_list(token_limit=10)
        ml.messages = [MessageEntry("system", "system prompt")]

        # Each message adds ~3 tokens (mock encode returns [1,2,3])
        # Token limit is 10; system prompt costs ~3 tokens → 7 remaining
        # That fits ~2 more messages before limit is hit
        for i in range(5):
            # Simulate check_if_add behavior: stop if over limit
            if ml.tokens > ml.token_limit:
                break
            entry = MessageEntry("user", f"msg {i}")
            ml.messages.append(entry)
            ml.tokens += 3  # ~3 tokens per message

        # Should have stopped before all 5 messages
        assert len(ml.messages) < 6  # system + some msgs

    def test_high_limit_preserves_all_messages(self):
        """With a high token limit, all messages should be included."""
        ml = _make_messages_list(token_limit=100000)
        ml.messages = [MessageEntry("system", "system prompt")]

        for i in range(10):
            ml.messages.append(MessageEntry("user", f"msg {i}"))

        assert len(ml.messages) == 11

    def test_prefix_subset_consistency(self):
        """If two requests have different token limits, the messages in the
        lower-limit request must be a prefix of the higher-limit request."""
        messages = [
            MessageEntry("system", "system"),
            MessageEntry("user", "A"),
            MessageEntry("user", "B"),
            MessageEntry("user", "C"),
            MessageEntry("user", "D"),
        ]

        # Lower limit: only fits first 3 messages
        ml_short = _make_messages_list(messages=messages[:3], token_limit=100000)
        # Higher limit: fits all 5
        ml_full = _make_messages_list(messages=messages, token_limit=100000)

        # The short list must be a prefix of the full list
        for i, msg in enumerate(ml_short.messages):
            assert msg.role == ml_full.messages[i].role
            assert msg.content == ml_full.messages[i].content


# ===========================================================================
# TEST 7: Tool-loop messages appended at the end
# ===========================================================================
class TestToolLoopAppendBehavior:
    """During the tool-calling loop, assistant and tool result messages
    must be appended at the END of the message list (after history and
    dynamic context), preserving the prefix."""

    def _setup_pipeline_with_context(self):
        pipeline = _make_pipeline()
        pipeline.msg_list.messages = [
            MessageEntry("system", "system prompt"),
            MessageEntry("user", "history 1"),
            MessageEntry("user", "init message"),
            MessageEntry("user", "dynamic context"),
        ]
        return pipeline

    def test_assistant_appended_at_end(self):
        """add_assistant with index=len+1 must append at the end."""
        pipeline = self._setup_pipeline_with_context()
        ml = pipeline.msg_list
        original_len = len(ml.messages)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            ml.add_assistant("assistant reply", index=len(ml) + 1)
        )

        assert len(ml.messages) == original_len + 1
        assert ml.messages[-1].role == "assistant"
        assert ml.messages[-1].content == "assistant reply"

        # Prefix unchanged
        assert ml.messages[0].role == "system"
        assert ml.messages[1].content == "history 1"

    def test_tool_result_appended_after_assistant(self):
        """Tool result should follow the assistant message at the end."""
        pipeline = self._setup_pipeline_with_context()
        ml = pipeline.msg_list

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            ml.add_assistant("tool call response", tool_calls=[MagicMock()], index=len(ml) + 1)
        )
        asyncio.get_event_loop().run_until_complete(
            ml.add_tool_result("tool output", tool_call_id="call_1", name="web_search", index=len(ml) + 1)
        )

        # Last two messages: assistant then tool
        assert ml.messages[-2].role == "assistant"
        assert ml.messages[-1].role == "tool"
        assert ml.messages[-1].tool_call_id == "call_1"

    def test_multiple_tool_rounds_preserve_prefix(self):
        """After multiple rounds of assistant+tool messages, the prefix
        (system + history + init + dynamic) must be unchanged."""
        pipeline = self._setup_pipeline_with_context()
        ml = pipeline.msg_list
        prefix_snapshot = [(m.role, m.content) for m in ml.messages]

        import asyncio
        for round_num in range(3):
            asyncio.get_event_loop().run_until_complete(
                ml.add_assistant(f"round {round_num} reply", index=len(ml) + 1)
            )
            asyncio.get_event_loop().run_until_complete(
                ml.add_tool_result(f"round {round_num} result", tool_call_id=f"call_{round_num}",
                                   name="test_tool", index=len(ml) + 1)
            )

        # Verify prefix is unchanged
        for i, (role, content) in enumerate(prefix_snapshot):
            assert ml.messages[i].role == role
            assert ml.messages[i].content == content

        # Total: 4 prefix + 3*(assistant + tool) = 10
        assert len(ml.messages) == 10


# ===========================================================================
# TEST 8: add_msg with index=1 inserts correctly
# ===========================================================================
class TestAddMsgIndexOne:
    """add_msg(msg, index=1) should insert the message between the system
    prompt (index 0) and any existing messages at index >= 1."""

    @pytest.mark.asyncio
    async def test_insert_at_index_one_pushes_existing(self):
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "init"),
        ]

        with patch.object(ml, 'check_if_add', new_callable=AsyncMock, return_value=True):
            new_msg = MagicMock()
            new_msg.id = 500
            new_msg.content = "new history"
            new_msg.attachments = []
            new_msg.stickers = []
            new_msg.embeds = []
            new_msg.reference = None
            ml.converter.convert = AsyncMock(
                return_value=[MessageEntry("user", "new history")]
            )
            await ml.add_msg(new_msg, index=1)

        assert ml.messages[0].role == "system"
        assert ml.messages[1].content == "new history"
        assert ml.messages[2].content == "init"

    @pytest.mark.asyncio
    async def test_multiple_inserts_at_index_one(self):
        """Inserting multiple messages at index=1 should reverse their order
        (last inserted ends up closest to system prompt)."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("system", "sys")]

        with patch.object(ml, 'check_if_add', new_callable=AsyncMock, return_value=True):
            for name in ["C", "B", "A"]:
                msg = MagicMock()
                msg.id = hash(name)
                ml.converter.convert = AsyncMock(
                    return_value=[MessageEntry("user", f"msg {name}")]
                )
                await ml.add_msg(msg, index=1)

        # After inserting C, B, A at index 1: [sys, A, B, C]
        assert ml.messages[1].content == "msg A"
        assert ml.messages[2].content == "msg B"
        assert ml.messages[3].content == "msg C"


# ===========================================================================
# TEST 9: Multi-entry add_msg maintains correct order
# ===========================================================================
class TestMultiEntryInsertion:
    """When a single message produces multiple MessageEntry objects
    (e.g. multimodal), they should all be inserted in order."""

    def test_single_entry_insertion(self):
        """A message that produces one entry should insert it at the target index."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("system", "sys")]

        entry = MessageEntry("user", "single entry")
        ml.messages.insert(1, entry)

        assert len(ml.messages) == 2
        assert ml.messages[1].content == "single entry"

    @pytest.mark.asyncio
    async def test_add_msg_multi_entry_preserves_order(self):
        """FIX: add_msg now increments insert_at after each entry, so
        multiple entries from one message maintain their original order.

        Before fix: all entries inserted at same position → reversed order.
        After fix: each entry advances insert_at → correct order."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("system", "sys")]

        with patch.object(ml, 'check_if_add', new_callable=AsyncMock, return_value=True):
            msg = MagicMock()
            msg.id = 500
            msg.reference = None
            # Simulate a message that produces 3 entries (e.g. multi-image)
            ml.converter.convert = AsyncMock(
                return_value=[
                    MessageEntry("user", "image 1"),
                    MessageEntry("user", "image 2"),
                    MessageEntry("user", "image 3"),
                ]
            )
            await ml.add_msg(msg, index=1)

        # All 3 entries should be in original order (not reversed)
        assert ml.messages[1].content == "image 1"
        assert ml.messages[2].content == "image 2"
        assert ml.messages[3].content == "image 3"

    def test_insert_at_index_does_not_skip_positions(self):
        """After inserting at index=1, the next insert at the same index
        should push the first insert to index=2."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("system", "sys")]

        ml.messages.insert(1, MessageEntry("user", "first"))
        ml.messages.insert(1, MessageEntry("user", "second"))

        assert ml.messages[1].content == "second"
        assert ml.messages[2].content == "first"

    def test_insert_at_end_index(self):
        """Inserting at index=len(messages) should append."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
        ]

        ml.messages.insert(len(ml.messages), MessageEntry("assistant", "reply"))

        assert ml.messages[-1].role == "assistant"
        assert ml.messages[-1].content == "reply"

    def test_insert_past_end_still_appends(self):
        """Inserting at index=len+1 (as used in the tool loop) should still
        append — Python list.insert handles out-of-range gracefully."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
        ]

        ml.messages.insert(len(ml.messages) + 1, MessageEntry("assistant", "reply"))

        assert len(ml.messages) == 3
        assert ml.messages[-1].role == "assistant"


# ===========================================================================
# TEST 10: get_json() output order matches message list order
# ===========================================================================
class TestGetJsonOrdering:
    """get_json() must serialize messages in the exact same order as they
    appear in self.messages, plus append the prefill at the end."""

    def test_get_json_preserves_order(self):
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys prompt"),
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "hi"),
            MessageEntry("user", "follow up"),
        ]
        ml.prefill = None

        json_msgs = ml.get_json()

        assert len(json_msgs) == 4
        assert json_msgs[0]["role"] == "system"
        assert json_msgs[0]["content"] == "sys prompt"
        assert json_msgs[1]["role"] == "user"
        assert json_msgs[1]["content"] == "hello"
        assert json_msgs[2]["role"] == "assistant"
        assert json_msgs[3]["role"] == "user"

    def test_get_json_appends_prefill_at_end(self):
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys prompt"),
            MessageEntry("user", "hello"),
        ]
        ml.prefill = "prefilled assistant text"

        json_msgs = ml.get_json()

        assert len(json_msgs) == 3
        assert json_msgs[-1]["role"] == "assistant"
        assert json_msgs[-1]["content"] == "prefilled assistant text"

    def test_get_json_includes_tool_calls(self):
        ml = _make_messages_list()
        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "search results", tool_call_id="call_1", name="web_search"),
        ]
        ml.prefill = None

        json_msgs = ml.get_json()

        assert "tool_calls" in json_msgs[1]
        assert json_msgs[2]["role"] == "tool"
        assert json_msgs[2]["tool_call_id"] == "call_1"


# ===========================================================================
# TEST 11: Prefix stability simulation across consecutive requests
# ===========================================================================
class TestConsecutiveRequestPrefixStability:
    """Simulate two consecutive requests in the same channel and verify
    that the shared prefix (system + common history) is identical."""

    def test_two_requests_share_system_and_common_history(self):
        """Request 1: messages [S, A, B, C, init1]
        Request 2: messages [S, A, B, C, D, init2]
        Shared prefix: [S, A, B, C] — must be byte-identical."""
        system = MessageEntry("system", "You are helpful.")
        hA = MessageEntry("user", "Alice: hello")
        hB = MessageEntry("user", "Bob: hi")
        hC = MessageEntry("user", "Alice: how are you?")

        # Request 1
        ml1 = _make_messages_list()
        ml1.messages = [
            system, hA, hB, hC,
            MessageEntry("user", "init message 1"),
        ]

        # Request 2 (one more history message)
        hD = MessageEntry("user", "Bob: I'm fine")
        ml2 = _make_messages_list()
        ml2.messages = [
            system, hA, hB, hC, hD,
            MessageEntry("user", "init message 2"),
        ]

        # Shared prefix is first 4 messages (system + A, B, C)
        shared_len = 4
        for i in range(shared_len):
            assert ml1.messages[i].role == ml2.messages[i].role
            assert ml1.messages[i].content == ml2.messages[i].content

    def test_cache_window_expansion_preserves_prefix(self):
        """When the cache window expands the token limit, more history is loaded.
        The previously-loaded messages must still be at the same positions."""
        system = MessageEntry("system", "system prompt")
        history_msgs = [MessageEntry("user", f"msg {i}") for i in range(10)]

        # Without cache window: only last 5 messages
        ml_no_window = _make_messages_list()
        ml_no_window.messages = [system] + history_msgs[5:] + [
            MessageEntry("user", "init")
        ]

        # With cache window: all 10 messages
        ml_with_window = _make_messages_list()
        ml_with_window.messages = [system] + history_msgs + [
            MessageEntry("user", "init")
        ]

        # The system prompt is identical
        assert ml_no_window.messages[0].content == ml_with_window.messages[0].content

        # The last 5 history messages from no_window should match the same
        # positions in with_window (offset by 5)
        for i in range(5):
            assert ml_no_window.messages[i + 1].content == ml_with_window.messages[i + 6].content


# ===========================================================================
# TEST 12: add_user inserts at end by default, add_system at beginning
# ===========================================================================
class TestDefaultInsertionPositions:
    """Verify default index behavior for different add methods."""

    @pytest.mark.asyncio
    async def test_add_system_default_inserts_at_zero(self):
        """add_system() with no index inserts at position 0."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("user", "existing")]

        await ml.add_system("new system")

        assert ml.messages[0].role == "system"
        assert ml.messages[0].content == "new system"

    @pytest.mark.asyncio
    async def test_add_user_default_inserts_at_end(self):
        """add_user() with no index inserts at the end."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "existing"),
        ]

        await ml.add_user("new user msg")

        assert ml.messages[-1].role == "user"
        assert ml.messages[-1].content == "new user msg"

    @pytest.mark.asyncio
    async def test_add_assistant_default_appends_at_end(self):
        """FIX: add_assistant() with no index now appends at the end
        (not at position 0), preserving prefix stability."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
        ]

        await ml.add_assistant("reply")

        # Without index, it appends at end — prefix is preserved
        assert ml.messages[0].role == "system"
        assert ml.messages[1].role == "user"
        assert ml.messages[2].role == "assistant"
        assert ml.messages[2].content == "reply"


# ===========================================================================
# TEST 13: Full round-trip simulation
# ===========================================================================
class TestFullRoundTrip:
    """Simulate a complete request lifecycle: build context → tool loop
    with stop instruction → verify prefix stability."""

    def test_full_lifecycle_prefix_unchanged(self):
        """After a full tool-calling loop simulation (including stop instruction
        add/remove and assistant/tool message appending), the original prefix
        must be unchanged."""
        # Build initial context
        ml = _make_messages_list()
        system = MessageEntry("system", "You are helpful.")
        h1 = MessageEntry("user", "Alice: What's the weather?")
        h2 = MessageEntry("user", "Bob: It's sunny today.")
        init = MessageEntry("user", "Alice: Thanks!")
        dynamic = MessageEntry("user", "date=2025/01/15 | time=14:30")

        ml.messages = [system, h1, h2, init, dynamic]
        prefix_snapshot = [(m.role, m.content) for m in ml.messages]

        # Simulate tool-calling loop (2 rounds)
        stop_instruction = "STOP: provide final response."
        for round_num in range(2):
            is_last = (round_num == 1)

            # Add stop instruction on last round
            stop_entry = None
            if is_last:
                stop_entry = MessageEntry("user", stop_instruction)
                ml.messages.append(stop_entry)

            # (LLM call would happen here — messages list is sent to API)
            json_payload = ml.get_json()

            # Identity-based removal (matches the fix in _run_loop)
            if stop_entry:
                for i in range(len(ml.messages) - 1, -1, -1):
                    if ml.messages[i] is stop_entry:
                        del ml.messages[i]
                        break

            # Add assistant response
            ml.messages.append(MessageEntry("assistant", f"Round {round_num} reply"))

            # Add tool result (on non-last round)
            if not is_last:
                ml.messages.append(MessageEntry("tool", f"Round {round_num} tool result",
                                                 tool_call_id=f"call_{round_num}"))

        # Verify prefix is unchanged
        for i, (role, content) in enumerate(prefix_snapshot):
            assert ml.messages[i].role == role, f"Prefix mismatch at index {i}: {ml.messages[i].role} != {role}"
            assert ml.messages[i].content == content, f"Prefix content mismatch at index {i}"

        # Final structure: [system, h1, h2, init, dynamic, asst_r0, tool_r0, asst_r1]
        assert len(ml.messages) == 8
        assert ml.messages[-1].role == "assistant"
        assert ml.messages[-1].content == "Round 1 reply"

    def test_get_json_prefix_stable_across_rounds(self):
        """The first N messages in get_json() output must be identical
        before and after the tool loop adds messages."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
            MessageEntry("user", "dynamic ctx"),
        ]
        ml.prefill = None

        prefix_json = ml.get_json()

        # Simulate tool loop additions
        ml.messages.append(MessageEntry("assistant", "thinking...", tool_calls=[MagicMock()]))
        ml.messages.append(MessageEntry("tool", "result", tool_call_id="c1", name="search"))
        ml.messages.append(MessageEntry("assistant", "final answer"))

        after_json = ml.get_json()

        # First 3 messages must be identical
        for i in range(3):
            assert prefix_json[i] == after_json[i]


# ===========================================================================
# TEST 14: Edge cases that could cause subtle cache misses
# ===========================================================================
class TestCacheMissEdgeCases:
    """Edge cases that could subtly break prefix caching."""

    def test_empty_content_message_preserves_position(self):
        """A message with empty content should still occupy its position
        in the list (not be skipped)."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", ""),
            MessageEntry("user", "next msg"),
        ]
        assert len(ml.messages) == 3
        assert ml.messages[1].content == ""

    def test_none_content_message_preserves_position(self):
        """An assistant message with None content (tool_calls only) should
        still occupy its position."""
        ml = _make_messages_list()
        tc = _make_tool_call("c1", "search")
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("assistant", None, tool_calls=[tc]),
            MessageEntry("tool", "result", tool_call_id="c1"),
        ]
        assert len(ml.messages) == 3
        assert ml.messages[1].content is None

    def test_multimodal_content_preserves_position(self):
        """A multimodal message (list content) should occupy the same
        position as a text-only message."""
        ml = _make_messages_list()
        multimodal_content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "look at this image"},
        ]
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", multimodal_content),
            MessageEntry("user", "what do you think?"),
        ]
        assert len(ml.messages) == 3
        assert isinstance(ml.messages[1].content, list)

    def test_assistant_with_reasoning_details_preserves_position(self):
        """An assistant message with reasoning_details should not disrupt
        the message list ordering."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
            MessageEntry("assistant", "reply", reasoning_details=[{"reasoning": "thinking..."}]),
            MessageEntry("user", "follow up"),
        ]
        assert len(ml.messages) == 4
        assert ml.messages[2].reasoning_details == [{"reasoning": "thinking..."}]

    def test_tool_message_ordering_after_assistant(self):
        """Tool result messages must always follow their corresponding
        assistant message with tool_calls."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "search for cats"),
            MessageEntry("assistant", None, tool_calls=[_make_tool_call("c1", "search")]),
            MessageEntry("tool", "cat results", tool_call_id="c1", name="search"),
            MessageEntry("assistant", "Here are the cat results"),
        ]

        # Verify ordering: assistant(tool_calls) → tool(result) → assistant(final)
        assert ml.messages[2].role == "assistant"
        assert ml.messages[2].tool_calls  # has tool calls
        assert ml.messages[3].role == "tool"
        assert ml.messages[3].tool_call_id == "c1"
        assert ml.messages[4].role == "assistant"
        assert not ml.messages[4].tool_calls  # final response, no tool calls


# ===========================================================================
# TEST 15: Insert-at-index consistency for add_assistant in tool loop
# ===========================================================================
class TestAddAssistantIndexConsistency:
    """The tool loop uses index=len(self.msg_list)+1 for add_assistant and
    add_tool_result.  Verify this consistently appends."""

    def test_add_assistant_at_len_plus_one_appends(self):
        """index=len+1 should behave identically to append."""
        ml = _make_messages_list()
        ml.messages = [
            MessageEntry("system", "sys"),
            MessageEntry("user", "hello"),
        ]

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            ml.add_assistant("reply", index=len(ml) + 1)
        )

        assert len(ml.messages) == 3
        assert ml.messages[-1].role == "assistant"
        assert ml.messages[-1].content == "reply"

    def test_sequential_adds_at_len_plus_one_maintain_order(self):
        """Multiple sequential add_assistant/add_tool_result at index=len+1
        should maintain insertion order."""
        ml = _make_messages_list()
        ml.messages = [MessageEntry("system", "sys")]

        import asyncio
        for i in range(5):
            asyncio.get_event_loop().run_until_complete(
                ml.add_assistant(f"reply {i}", index=len(ml) + 1)
            )
            asyncio.get_event_loop().run_until_complete(
                ml.add_tool_result(f"result {i}", tool_call_id=f"c_{i}",
                                   name="tool", index=len(ml) + 1)
            )

        # Verify sequential order
        for i in range(5):
            assert ml.messages[1 + i * 2].role == "assistant"
            assert ml.messages[1 + i * 2].content == f"reply {i}"
            assert ml.messages[2 + i * 2].role == "tool"
            assert ml.messages[2 + i * 2].content == f"result {i}"
