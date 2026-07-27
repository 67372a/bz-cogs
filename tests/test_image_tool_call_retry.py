"""Tests for ImageToolCall retry functionality.

These tests cover the shared retry logic added to the ImageToolCall base class,
including the is_image_response_unsatisfactory predicate and the
_create_image_completion_with_retry method used by both GenerateImageToolCall
and EditImageToolCall.
"""

import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — must happen before importing the real tool_call module
# ---------------------------------------------------------------------------

# Minimal discord module
_discord_mod = types.ModuleType("discord")
_discord_mod.Message = type("_DiscordMessage", (), {})
_discord_mod.Embed = MagicMock()
_discord_mod.Color = MagicMock()
_discord_mod.Colour = MagicMock()
sys.modules.setdefault("discord", _discord_mod)

# Minimal redbot.core module
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
sys.modules.setdefault("redbot", _redbot)
sys.modules.setdefault("redbot.core", _redbot_core)

# Ensure parent packages exist
for _pkg in [
    "aiuser", "aiuser.functions", "aiuser.types", "aiuser.utils",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)

# Mock leaf modules that tool_call.py and image tool calls import
_leaf_mocks = [
    "aiuser.functions.types",
    "aiuser.types.openrouter_types",
    "aiuser.utils.image_cache",
    "aiuser.utils.image_processing",
    "aiuser.utils.utilities",
]
for _leaf in _leaf_mocks:
    if _leaf not in sys.modules:
        sys.modules[_leaf] = MagicMock()

# Load the module under test
tool_call_mod = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)
ImageToolCall = tool_call_mod.ImageToolCall
ToolCall = tool_call_mod.ToolCall
is_image_response_unsatisfactory = tool_call_mod.is_image_response_unsatisfactory
_response_has_image_data = tool_call_mod._response_has_image_data
_unwrap_retry_error = tool_call_mod._unwrap_retry_error

from tenacity import RetryError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal valid base64-encoded 1x1 PNG for test data URLs
_TEST_IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _make_completion(content=None, finish_reason="stop", with_image=False):
    """Build a ChatCompletion for testing.

    ``content=None`` produces an empty ``choices`` list.
    ``with_image=True`` attaches image data in ``model_extra`` so the response
    is considered satisfactory by ``is_image_response_unsatisfactory``.
    Uses ``Choice.model_construct`` to bypass Pydantic validation so that
    non-standard finish_reason values like ``"error"`` can be tested.
    """
    if content is None:
        choices = []
    else:
        message = ChatCompletionMessage.model_construct(
            role="assistant", content=content, tool_calls=None,
            function_call=None, reasoning=None, audio=None,
        )
        if with_image:
            message.__pydantic_extra__ = {
                "images": [
                    {"type": "image_url", "image_url": {"url": _TEST_IMAGE_DATA_URL}}
                ]
            }
        choices = [
            Choice.model_construct(
                index=0,
                message=message,
                finish_reason=finish_reason,
                logprobs=None,
            )
        ]
    return ChatCompletion.model_construct(
        id="test-id",
        choices=choices,
        created=0,
        model="test-model",
        object="chat.completion",
        usage=None,
    )


def _make_image_tool_call() -> ImageToolCall:
    """Create a minimally-initialized ImageToolCall for unit tests."""
    tool = ImageToolCall.__new__(ImageToolCall)
    tool.config = MagicMock()
    tool.ctx = MagicMock()
    tool.bot = MagicMock()
    return tool


# ---------------------------------------------------------------------------
# Tests for is_image_response_unsatisfactory
# ---------------------------------------------------------------------------


class TestResponseHasImageData:
    """Tests for _response_has_image_data helper."""

    def test_none_choices_returns_false(self):
        response = _make_completion(content=None)
        assert _response_has_image_data(response) is False

    def test_text_only_content_returns_false(self):
        response = _make_completion(content="Here is your image")
        assert _response_has_image_data(response) is False

    def test_model_extra_images_list_returns_true(self):
        response = _make_completion(content="Here you go", with_image=True)
        assert _response_has_image_data(response) is True

    def test_model_extra_images_string_returns_true(self):
        message = ChatCompletionMessage.model_construct(
            role="assistant", content="Here", tool_calls=None,
            function_call=None, reasoning=None, audio=None,
        )
        message.__pydantic_extra__ = {"images": _TEST_IMAGE_DATA_URL}
        choice = Choice.model_construct(
            index=0, message=message, finish_reason="stop", logprobs=None,
        )
        response = ChatCompletion.model_construct(
            id="test-id", choices=[choice], created=0,
            model="test-model", object="chat.completion", usage=None,
        )
        assert _response_has_image_data(response) is True

    def test_content_with_data_url_returns_true(self):
        message = ChatCompletionMessage.model_construct(
            role="assistant",
            content=f"Image: {_TEST_IMAGE_DATA_URL}",
            tool_calls=None, function_call=None, reasoning=None, audio=None,
        )
        choice = Choice.model_construct(
            index=0, message=message, finish_reason="stop", logprobs=None,
        )
        response = ChatCompletion.model_construct(
            id="test-id", choices=[choice], created=0,
            model="test-model", object="chat.completion", usage=None,
        )
        assert _response_has_image_data(response) is True


class TestIsImageResponseUnsatisfactory:
    def test_none_response_is_unsatisfactory(self):
        assert is_image_response_unsatisfactory(None) is True

    def test_empty_choices_is_unsatisfactory(self):
        response = _make_completion(content=None)
        assert is_image_response_unsatisfactory(response) is True

    def test_finish_reason_error_is_unsatisfactory(self):
        response = _make_completion(content="Some content", finish_reason="error")
        assert is_image_response_unsatisfactory(response) is True

    def test_finish_reason_content_filter_is_unsatisfactory(self):
        response = _make_completion(content="Some content", finish_reason="content_filter")
        assert is_image_response_unsatisfactory(response) is True

    def test_text_only_stop_response_is_unsatisfactory(self):
        """A stop response with no image data should trigger retry."""
        response = _make_completion(content="I cannot generate images")
        assert is_image_response_unsatisfactory(response) is True

    def test_stop_response_with_image_data_is_satisfactory(self):
        """A stop response that contains image data should be satisfactory."""
        response = _make_completion(content="Here is your image", with_image=True)
        assert is_image_response_unsatisfactory(response) is False

    def test_error_finish_reason_with_image_still_unsatisfactory(self):
        """Error finish_reason is unsatisfactory even if image data is present."""
        response = _make_completion(content="Partial", finish_reason="error", with_image=True)
        assert is_image_response_unsatisfactory(response) is True

    def test_valid_response_with_length_finish_is_satisfactory(self):
        response = _make_completion(content="Partial response", finish_reason="length")
        assert is_image_response_unsatisfactory(response) is False


# ---------------------------------------------------------------------------
# Tests for _create_image_completion_with_retry
# ---------------------------------------------------------------------------


class TestUnwrapRetryError:
    """Tests for _unwrap_retry_error helper."""

    def test_unwrap_exception_based_retry(self):
        """Should describe the inner exception for exception-based retries."""
        mock_attempt = MagicMock()
        mock_attempt.failed = True
        mock_attempt.exception.return_value = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_retry_error = MagicMock(spec=RetryError)
        mock_retry_error.last_attempt = mock_attempt

        result = _unwrap_retry_error(mock_retry_error)

        assert "RateLimitError" in result
        assert "rate limited" in result

    def test_unwrap_result_based_retry_with_chat_completion(self):
        """Should extract finish_reason and content preview from ChatCompletion."""
        unsatisfactory = _make_completion(
            content="I cannot generate images for you", finish_reason="stop"
        )
        mock_attempt = MagicMock()
        mock_attempt.failed = False
        mock_attempt.result.return_value = unsatisfactory
        mock_retry_error = MagicMock(spec=RetryError)
        mock_retry_error.last_attempt = mock_attempt

        result = _unwrap_retry_error(mock_retry_error)

        assert "finish_reason=stop" in result
        assert "I cannot generate images" in result

    def test_unwrap_result_based_retry_with_none(self):
        """Should handle None result gracefully."""
        mock_attempt = MagicMock()
        mock_attempt.failed = False
        mock_attempt.result.return_value = None
        mock_retry_error = MagicMock(spec=RetryError)
        mock_retry_error.last_attempt = mock_attempt

        result = _unwrap_retry_error(mock_retry_error)

        assert "None" in result

    def test_unwrap_result_based_retry_with_no_choices(self):
        """Should handle result with empty choices."""
        no_choices = _make_completion(content=None)
        mock_attempt = MagicMock()
        mock_attempt.failed = False
        mock_attempt.result.return_value = no_choices
        mock_retry_error = MagicMock(spec=RetryError)
        mock_retry_error.last_attempt = mock_attempt

        result = _unwrap_retry_error(mock_retry_error)

        assert "Unsatisfactory" in result

    def test_unwrap_long_content_is_truncated(self):
        """Should truncate long content previews to 200 chars."""
        long_content = "x" * 500
        unsatisfactory = _make_completion(content=long_content, finish_reason="stop")
        mock_attempt = MagicMock()
        mock_attempt.failed = False
        mock_attempt.result.return_value = unsatisfactory
        mock_retry_error = MagicMock(spec=RetryError)
        mock_retry_error.last_attempt = mock_attempt

        result = _unwrap_retry_error(mock_retry_error)

        assert "..." in result
        # Should not contain the full 500-char string
        assert long_content not in result


class TestCreateImageCompletionWithRetry:
    @pytest.mark.asyncio
    async def test_returns_valid_response(self):
        tool = _make_image_tool_call()
        mock_client = MagicMock()
        expected = _make_completion(content="Image generated successfully", with_image=True)
        mock_client.chat.completions.create = AsyncMock(return_value=expected)

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert result.choices[0].message.content == "Image generated successfully"
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self, monkeypatch):
        """Should retry on openai.RateLimitError and eventually succeed."""
        import openai

        tool = _make_image_tool_call()
        mock_client = MagicMock()

        rate_limit_error = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        expected = _make_completion(content="Success after retry", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[rate_limit_error, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_api_connection_error(self, monkeypatch):
        """Should retry on openai.APIConnectionError and eventually succeed."""
        import openai

        tool = _make_image_tool_call()
        mock_client = MagicMock()

        conn_error = openai.APIConnectionError(request=MagicMock())
        expected = _make_completion(content="Success after retry", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[conn_error, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_empty_choices_response(self, monkeypatch):
        """Should retry when response has no choices."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        empty = _make_completion(content=None)
        expected = _make_completion(content="Image data here", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[empty, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_error_finish_reason(self, monkeypatch):
        """Should retry when finish_reason is 'error'."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        error_response = _make_completion(content="Error output", finish_reason="error")
        expected = _make_completion(content="Image data here", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[error_response, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_content_filter_finish_reason(self, monkeypatch):
        """Should retry when finish_reason is 'content_filter'."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        filter_response = _make_completion(
            content="Filtered", finish_reason="content_filter"
        )
        expected = _make_completion(content="Image data here", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[filter_response, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_text_only_stop_response(self, monkeypatch):
        """Should retry when model returns text-only (no image) with stop reason."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        text_only = _make_completion(content="I cannot generate images", finish_reason="stop")
        expected = _make_completion(content="Here is your image", with_image=True)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[text_only, expected]
        )

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(3),
        )

        result = await tool._create_image_completion_with_retry(
            client=mock_client, model="test-model", messages=[], user="test-user"
        )

        assert result == expected
        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_empty_choices(self, monkeypatch):
        """Should raise RetryError after exhausting all attempts."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        empty = _make_completion(content=None)
        mock_client.chat.completions.create = AsyncMock(return_value=empty)

        from tenacity import stop_after_attempt, wait_none

        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry, "wait", wait_none()
        )
        monkeypatch.setattr(
            tool._create_image_completion_with_retry.retry,
            "stop",
            stop_after_attempt(2),
        )

        with pytest.raises(Exception):  # tenacity.RetryError
            await tool._create_image_completion_with_retry(
                client=mock_client, model="test-model", messages=[], user="test-user"
            )

        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_exception(self):
        """Should raise non-retryable exceptions immediately."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()

        mock_client.chat.completions.create = AsyncMock(
            side_effect=ValueError("non-retryable error")
        )

        with pytest.raises(ValueError, match="non-retryable error"):
            await tool._create_image_completion_with_retry(
                client=mock_client, model="test-model", messages=[], user="test-user"
            )

        assert mock_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_passes_extra_kwargs_to_client(self):
        """Should forward extra_body and other kwargs to the client."""
        tool = _make_image_tool_call()
        mock_client = MagicMock()
        expected = _make_completion(content="Success", with_image=True)
        mock_client.chat.completions.create = AsyncMock(return_value=expected)

        extra_body = {"modalities": ["image", "text"]}
        result = await tool._create_image_completion_with_retry(
            client=mock_client,
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            user="test-user",
            extra_body=extra_body,
            temperature=0.7,
        )

        assert result == expected
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            user="test-user",
            stream=False,
            extra_body=extra_body,
            temperature=0.7,
        )


# ---------------------------------------------------------------------------
# Tests for inheritance
# ---------------------------------------------------------------------------


class TestImageToolCallInheritance:
    def test_image_tool_call_inherits_from_tool_call(self):
        assert issubclass(ImageToolCall, ToolCall)

    def test_generate_image_uses_image_tool_call_parent(self):
        """Verify GenerateImageToolCall uses ImageToolCall as its base.

        Note: When run in the full test suite, other test modules may mock
        ``aiuser.functions.tool_call`` as a MagicMock.  We therefore check
        that the class's source code references ImageToolCall and that an
        instance has the retry method, rather than using ``issubclass``.
        """
        gen_mod = import_module_directly(
            "aiuser.functions.generate_image.tool_call",
            "aiuser/functions/generate_image/tool_call.py",
        )
        GenerateImageToolCall = gen_mod.GenerateImageToolCall
        # Verify it's a class (not a MagicMock from another test's mocking)
        if inspect.isclass(GenerateImageToolCall):
            assert issubclass(GenerateImageToolCall, ImageToolCall)
            assert issubclass(GenerateImageToolCall, ToolCall)
        # Either way, verify the source file has the correct inheritance
        source_file = "aiuser/functions/generate_image/tool_call.py"
        with open(source_file) as f:
            source = f.read()
        assert "class GenerateImageToolCall(ImageToolCall):" in source

    def test_edit_image_uses_image_tool_call_parent(self):
        """Verify EditImageToolCall uses ImageToolCall as its base.

        Same rationale as above — guard against MagicMock interference.
        """
        edit_mod = import_module_directly(
            "aiuser.functions.edit_image.tool_call",
            "aiuser/functions/edit_image/tool_call.py",
        )
        EditImageToolCall = edit_mod.EditImageToolCall
        if inspect.isclass(EditImageToolCall):
            assert issubclass(EditImageToolCall, ImageToolCall)
            assert issubclass(EditImageToolCall, ToolCall)
        source_file = "aiuser/functions/edit_image/tool_call.py"
        with open(source_file) as f:
            source = f.read()
        assert "class EditImageToolCall(ImageToolCall):" in source

    def test_image_tool_call_has_retry_method(self):
        tool = _make_image_tool_call()
        assert hasattr(tool, "_create_image_completion_with_retry")
        assert callable(tool._create_image_completion_with_retry)
