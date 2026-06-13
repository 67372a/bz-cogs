"""Tests for LLMPipeline retry/empty-choices handling.

These tests cover the edge case where an LLM response arrives with no
``choices`` list. Previously the code tried to raise ``openai.APIError`` with
only a message string, but the openai SDK (>=1.0) requires a ``request``
argument, so the raise itself raised ``TypeError`` and masked the real
upstream failure.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing the real llm_pipeline module
# ---------------------------------------------------------------------------

# Minimal discord module
_discord_mod = types.ModuleType("discord")
_discord_mod.Message = type("_DiscordMessage", (), {})
_discord_mod.Embed = MagicMock()
_discord_mod.Color = MagicMock()
_discord_mod.Colour = MagicMock()
sys.modules["discord"] = _discord_mod

# Minimal redbot.core module
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
sys.modules["redbot"] = _redbot
sys.modules["redbot.core"] = _redbot_core

# Leaf aiuser modules imported by llm_pipeline.py.  MagicMock is sufficient
# because the unit tests only exercise _create_completion_with_retry and
# is_response_unsatisfactory.
_leaf_mocks = [
    "aiuser.config.models",
    "aiuser.types.abc",
    "aiuser.types.enums",
    "aiuser.functions.tool_call",
    "aiuser.functions.types",
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
    "aiuser.functions.attach_files.tool_call",
    "aiuser.functions.mermaid.tool_call",
    "aiuser.messages_list.messages",
    "aiuser.messages_list.entry",
    "aiuser.response.chat.function_call_view",
    "aiuser.functions.openrouter",
    "aiuser.utils.utilities",
]
for _leaf in _leaf_mocks:
    sys.modules[_leaf] = MagicMock()

# Ensure every parent package exists as a mock package so from-imports resolve.
_all_pkg_prefixes = set()
for _name in list(sys.modules.keys()) + ["aiuser.response.chat.llm_pipeline"]:
    _parts = _name.split(".")
    for _i in range(1, len(_parts)):
        _all_pkg_prefixes.add(".".join(_parts[:_i]))

for _pkg in _all_pkg_prefixes:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Load the module under test
llm_pipeline = import_module_directly(
    "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
)
LLMPipeline = llm_pipeline.LLMPipeline
is_response_unsatisfactory = llm_pipeline.is_response_unsatisfactory

from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import Function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> LLMPipeline:
    """Create a minimally-initialized LLMPipeline for unit tests."""
    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline.ctx = MagicMock()
    pipeline.config = MagicMock()
    pipeline.model = "gpt-4"
    pipeline.openai_client = MagicMock()
    return pipeline


def _make_completion(content=None, finish_reason="stop", tool_calls=None):
    """Build a ChatCompletion for testing.

    ``content=None`` produces an empty ``choices`` list, which is the edge
    case we are validating.
    """
    if content is None and tool_calls is None:
        choices = []
    else:
        message_kwargs = {"role": "assistant"}
        if content is not None:
            message_kwargs["content"] = content
        if tool_calls is not None:
            message_kwargs["tool_calls"] = tool_calls
        choices = [
            Choice(
                index=0,
                message=ChatCompletionMessage(**message_kwargs),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ]
    return ChatCompletion(
        id="test-id",
        choices=choices,
        created=0,
        model="test-model",
        object="chat.completion",
    )


# ---------------------------------------------------------------------------
# Tests for the tenacity predicate
# ---------------------------------------------------------------------------


class TestIsResponseUnsatisfactory:
    def test_empty_choices_is_unsatisfactory(self):
        response = _make_completion(content=None)
        assert is_response_unsatisfactory(response) is True

    def test_empty_content_no_tools_is_unsatisfactory(self):
        response = _make_completion(content="   ", finish_reason="stop")
        assert is_response_unsatisfactory(response) is True

    def test_valid_text_is_satisfactory(self):
        response = _make_completion(content="Hello!")
        assert is_response_unsatisfactory(response) is False

    def test_empty_content_with_tools_is_satisfactory(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            function=Function(arguments="{}", name="test"),
            type="function",
        )
        response = _make_completion(content="", tool_calls=[tool_call])
        assert is_response_unsatisfactory(response) is False


# ---------------------------------------------------------------------------
# Tests for the retry wrapper
# ---------------------------------------------------------------------------


class TestCreateCompletionWithRetry:
    @pytest.mark.asyncio
    async def test_returns_valid_response(self):
        pipeline = _make_pipeline()
        expected = _make_completion(content="Hello!")
        pipeline.openai_client.chat.completions.create = AsyncMock(
            return_value=expected
        )

        result = await pipeline._create_completion_with_retry(
            model="test-model", messages=[]
        )

        assert result == expected
        assert result.choices[0].message.content == "Hello!"
        pipeline.openai_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_choices_retries_then_raises(self, monkeypatch):
        pipeline = _make_pipeline()
        empty = _make_completion(content=None)
        pipeline.openai_client.chat.completions.create = AsyncMock(
            return_value=empty
        )

        # Speed up the test by disabling waits and limiting retry attempts.
        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            pipeline._create_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            pipeline._create_completion_with_retry.retry,
            "stop",
            stop_after_attempt(2),
        )

        with pytest.raises(Exception):  # tenacity.RetryError after exhaustion
            await pipeline._create_completion_with_retry(
                model="test-model", messages=[]
            )

        # The fix removes the broken ``raise openai.APIError(...)`` path, so
        # the empty response is returned and retried instead of crashing on
        # the first attempt.
        assert pipeline.openai_client.chat.completions.create.await_count == 2
