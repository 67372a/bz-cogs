
import logging
from typing import Any, Dict, Optional

import httpx2
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from redbot.core import Config, commands
from redbot.core.bot import Red
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)

from aiuser.functions.types import ToolCallSchema

logger = logging.getLogger("red.bz_cogs.aiuser")


def _unwrap_retry_error(error: RetryError) -> str:
    """Extract a human-readable message from a tenacity RetryError.

    When ``retry_if_result`` causes all retries to be exhausted, tenacity
    raises ``RetryError`` whose ``str()`` is an opaque
    ``RetryError[<Future ...>]`` message.  This helper unwraps it to
    produce a meaningful description.

    Args:
        error: The tenacity RetryError to unwrap.

    Returns:
        A human-readable string describing the last failure.
    """
    last_attempt = error.last_attempt

    if last_attempt.failed:
        # Exception-based retry exhaustion
        inner = last_attempt.exception()
        return f"{type(inner).__name__}: {inner}"

    # Result-based retry exhaustion — the last result was unsatisfactory
    result = last_attempt.result()
    if result is None:
        return "API returned None after all retries"

    # ChatCompletion object — extract useful details
    if hasattr(result, "choices") and result.choices:
        choice = result.choices[0]
        finish_reason = getattr(choice, "finish_reason", "unknown")
        content = getattr(choice.message, "content", None) if hasattr(choice, "message") else None
        preview = (content[:200] + "...") if content and len(content) > 200 else (content or "<no content>")
        return f"Unsatisfactory response after all retries (finish_reason={finish_reason}): {preview}"

    # Fallback
    return f"Unsatisfactory response after all retries: {repr(result)[:300]}"


def _response_has_image_data(response: ChatCompletion) -> bool:
    """Check if a ChatCompletion contains image data.

    Looks for image data in model_extra (the primary location for OpenRouter/Gemini
    image responses) and in the message content text (as a fallback).

    Args:
        response: The ChatCompletion object from the OpenAI API call.

    Returns:
        True if image data is found, False otherwise.
    """
    if not response.choices:
        return False

    message = response.choices[0].message

    # Check model_extra for image entries (primary path)
    model_extra = getattr(message, "model_extra", None)
    if model_extra and isinstance(model_extra, dict):
        raw_images = model_extra.get("images") or model_extra.get("image")
        if raw_images:
            if isinstance(raw_images, list) and len(raw_images) > 0:
                return True
            if isinstance(raw_images, str) and raw_images.startswith("data:"):
                return True

    # Check content for embedded data URLs (fallback)
    content = getattr(message, "content", None)
    if content and isinstance(content, str) and "data:image/" in content:
        return True

    return False


def is_image_response_unsatisfactory(response: ChatCompletion) -> bool:
    """Check if an image generation/edit API response is unsatisfactory and should be retried.

    An unsatisfactory response is one that:
    1. Is None or has no choices.
    2. Has a finish_reason of 'error' or 'content_filter'.
    3. Has a finish_reason of 'stop' but contains no image data (text-only response).

    Args:
        response: The ChatCompletion object from the OpenAI API call.

    Returns:
        True if the response is unsatisfactory, False otherwise.
    """
    if response is None or not response.choices:
        logger.warning("Image retry triggered: Response object is None or has no choices.")
        return True

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    if finish_reason in ("error", "content_filter"):
        logger.warning(
            f"Image retry triggered: Unsatisfactory finish_reason '{finish_reason}'."
        )
        return True

    # A "stop" response with no image data is unsatisfactory for image generation/editing.
    # The model may have responded with text only (e.g. "I cannot generate images") without
    # actually producing image data.
    if finish_reason == "stop" and not _response_has_image_data(response):
        logger.warning(
            "Image retry triggered: finish_reason is 'stop' but no image data found in response."
        )
        return True

    return False


class ToolCall:
    schema: ToolCallSchema = None
    function_name: str = None

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx
        self.bot: Red = ctx.bot

    def run(self, arguments: dict):
        return self._handle(arguments)

    def _handle(arguments: dict):
        raise NotImplementedError


class ImageToolCall(ToolCall):
    """Base class for image generation/editing tool calls with retry logic.

    Provides a shared ``_create_image_completion_with_retry`` method that
    wraps the OpenAI chat completions API with tenacity-based retry logic,
    matching the retry behavior of ``LLMPipeline._create_completion_with_retry``.
    """

    @retry(
        wait=wait_random_exponential(min=1, max=4),
        stop=stop_after_attempt(5),
        retry=(
            retry_if_exception_type((
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                httpx2.ReadTimeout,
                httpx2.ConnectTimeout,
            ))
            | retry_if_result(is_image_response_unsatisfactory)
        ),
    )
    async def _create_image_completion_with_retry(
        self, client: AsyncOpenAI, model: str, messages: list, user: str, stream: bool = False, **kwargs
    ) -> ChatCompletion:
        """Create a chat completion for image generation/edit with retry logic.

        Retries on transient API errors (rate limits, connection issues,
        timeouts, server errors) and unsatisfactory responses (no choices,
        error/content_filter finish reasons).

        Args:
            client: The AsyncOpenAI client instance.
            model: The model to use for the completion.
            messages: The messages to send to the API.
            user: A hashed user identifier for the API call.
            stream: Whether to stream the response (default False).
            **kwargs: Additional keyword arguments passed to the API call.

        Returns:
            The ChatCompletion response.
        """
        try:
            result = await client.chat.completions.create(
                model=model, messages=messages, user=user, stream=stream, **kwargs
            )
            if not result.choices:
                try:
                    raw = result.model_dump_json(exclude_none=True)
                except Exception:
                    raw = repr(result)
                logger.warning(
                    "Image generation response has no choices (edge case); will retry. "
                    "id=%s model=%s object=%s raw=%s",
                    getattr(result, "id", None),
                    getattr(result, "model", None),
                    getattr(result, "object", None),
                    raw,
                )
                return result
            logger.info(
                "Image generation finish reason: %s",
                result.choices[0].finish_reason,
            )
            return result
        except Exception as e:
            logger.warning(f"Image generation API call attempt failed: {type(e).__name__}: {e}")
            raise
