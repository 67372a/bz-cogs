"""Regression tests: multimodal tool results must always contain a text part.

Bug context
-----------
``GenerateImageToolCall`` / ``EditImageToolCall`` returned tool results
containing ONLY ``image_url`` content parts (no ``text`` part).  When the
follow-up LLM request was sent through OpenRouter to a Gemini model, the
OpenAI ``tool`` message could not be translated into a valid Gemini
``functionResponse`` (which requires a JSON ``response`` payload), so the
tool message was dropped and the request ended with the assistant message
containing the functionCall.  Google rejected it with::

    400: "Requests ending with a model turn are not supported."

Two layers of defense are tested here:

1. Pipeline safety net — ``LLMPipeline._process_and_add_tool_results``
   injects a synthetic text part into any list-type tool result that lacks
   one, and coerces empty lists to a plain string.
2. Tool-level guarantee — the image tools themselves prepend a text part
   (covered in ``tests/test_image_tool_result_format.py``).
"""

import json
import sys
import types
from types import SimpleNamespace
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

# Leaf aiuser modules imported by llm_pipeline.py.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tool_result):
    """Create a minimally-initialized LLMPipeline whose run_tool returns
    ``tool_result`` and whose msg_list records add_tool_result calls."""
    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline.ctx = MagicMock()
    pipeline.config = MagicMock()
    pipeline.model = "google/gemini-3.6-flash"
    pipeline.enabled_tools = []
    pipeline.collected_attachments = []

    pipeline.run_tool = AsyncMock(return_value=tool_result)

    msg_list = MagicMock()
    msg_list.messages = []  # real list so pairing verification early-returns
    msg_list.add_tool_result = AsyncMock()
    # MagicMock supports __len__ (returns 0) — used for the insert index.
    pipeline.msg_list = msg_list

    return pipeline


def _make_tool_call(name="generate_image", tool_call_id="tc-1", arguments=None):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments or {"prompt": "a test image"}),
        ),
    )


_IMAGE_PART = {
    "type": "image_url",
    "image_url": {"url": "data:image/webp;base64,UklGRiQAAABXRUJQVlA4TBgAAAAvAAAAAAfQ//73v/+BiOh/AAA="},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolResultTextPartSafetyNet:
    """``_process_and_add_tool_results`` must guarantee that any multimodal
    tool result added to context contains at least one text part."""

    @pytest.mark.asyncio
    async def test_image_only_result_gets_text_part_injected(self):
        """An image-only content-part list must gain a leading text part so
        the Gemini functionResponse translation stays valid."""
        pipeline = _make_pipeline(tool_result=[dict(_IMAGE_PART)])

        outputs = await pipeline._process_and_add_tool_results([_make_tool_call()])

        pipeline.msg_list.add_tool_result.assert_awaited_once()
        kwargs = pipeline.msg_list.add_tool_result.await_args.kwargs
        content = kwargs["content"]

        assert isinstance(content, list), f"Expected list, got {type(content)}"
        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(text_parts) >= 1, "A text part must be present after the safety net"
        assert text_parts[0].get("text"), "Injected text part must be non-empty"
        # Text part must be first so the functionResponse has a payload
        assert content[0]["type"] == "text"
        # Original image part is preserved
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        # Outputs summary should reflect the injected text
        assert outputs and outputs[0]["result"], "Expected a non-empty output summary"

    @pytest.mark.asyncio
    async def test_result_with_text_part_is_not_modified(self):
        """A well-formed multimodal result (text + image) must pass through
        unchanged — no duplicate text parts injected."""
        original = [
            {"type": "text", "text": "Successfully generated 1 image(s)."},
            dict(_IMAGE_PART),
        ]
        pipeline = _make_pipeline(tool_result=list(original))

        await pipeline._process_and_add_tool_results([_make_tool_call()])

        kwargs = pipeline.msg_list.add_tool_result.await_args.kwargs
        content = kwargs["content"]

        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(text_parts) == 1, f"Expected exactly 1 text part, got {len(text_parts)}"
        assert content[0]["text"] == "Successfully generated 1 image(s)."
        assert len(content) == 2

    @pytest.mark.asyncio
    async def test_empty_list_result_coerced_to_string(self):
        """An empty content list would create an empty tool message that
        providers may drop; it must be coerced to a plain string."""
        pipeline = _make_pipeline(tool_result=[])

        outputs = await pipeline._process_and_add_tool_results([_make_tool_call()])

        kwargs = pipeline.msg_list.add_tool_result.await_args.kwargs
        content = kwargs["content"]

        assert isinstance(content, str), f"Expected str, got {type(content)}"
        assert content.strip(), "Fallback string must be non-empty"
        assert outputs[0]["result"] == content

    @pytest.mark.asyncio
    async def test_string_result_passes_through(self):
        """Plain string results (e.g. error messages) must be unaffected."""
        pipeline = _make_pipeline(tool_result="Error: something failed")

        outputs = await pipeline._process_and_add_tool_results([_make_tool_call()])

        kwargs = pipeline.msg_list.add_tool_result.await_args.kwargs
        assert kwargs["content"] == "Error: something failed"
        assert outputs[0]["result"] == "Error: something failed"

    @pytest.mark.asyncio
    async def test_tool_message_is_serializable_and_ends_context(self):
        """The tool result passed to add_tool_result must be JSON-serializable
        (it is sent verbatim in the API payload) and, as the final message,
        guarantees the request does not end with a model turn."""
        pipeline = _make_pipeline(tool_result=[dict(_IMAGE_PART)])

        await pipeline._process_and_add_tool_results([_make_tool_call()])

        kwargs = pipeline.msg_list.add_tool_result.await_args.kwargs

        # Must be JSON-serializable for the chat.completions payload
        serialized = json.dumps(kwargs["content"])
        assert serialized

        # Simulate the serialized message list the API would receive: the
        # last message is the tool (functionResponse) turn, never a bare
        # assistant/model turn.
        tool_message = {
            "role": "tool",
            "tool_call_id": kwargs["tool_call_id"],
            "name": kwargs["name"],
            "content": kwargs["content"],
        }
        payload = [
            {"role": "user", "content": "gen yourself"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": "{}"},
                    }
                ],
            },
            tool_message,
        ]
        assert payload[-1]["role"] == "tool"
        assert payload[-1]["content"][0]["type"] == "text"
