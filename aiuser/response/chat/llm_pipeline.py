import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import copy

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from redbot.core import Config, commands

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type, 
    retry_if_result
)

from aiuser.config.models import (
    UNSUPPORTED_LOGIT_BIAS_MODELS,
    VISION_SUPPORTED_MODELS,
)
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import ToolCallSchema
from aiuser.functions.generate_image.tool_call import GenerateImageToolCall
from aiuser.functions.edit_image.tool_call import EditImageToolCall
from aiuser.messages_list.messages import MessagesList
from aiuser.messages_list.entry import MessageEntry
from aiuser.functions.openrouter import (
    OpenRouterWebSearch,
    OpenRouterWebFetch,
    OpenRouterImageGeneration,
)
from aiuser.types.abc import MixinMeta
from aiuser.types.enums import OpenRouterToolType
from aiuser.utils.utilities import get_enabled_tools


@dataclass
class ResponsePart:
    """A single part of the bot's response to send to Discord.

    A part can be pure text, or text with attached images for combined sending.
    Images are sent as Discord file attachments alongside the text embed.
    """
    type: str  # "text" — text-only; "text_and_images" — text + attached images
    content: str
    images: List[Dict] = field(default_factory=list)
    def __bool__(self):
        return bool(self.content.strip()) if isinstance(self.content, str) else bool(self.content)


@dataclass
class PipelineResult:
    """Result of the tool-calling loop execution.

    Contains everything the caller needs to send the final response to Discord.
    """
    text: Optional[str] = None
    reasoning: Optional[str] = None
    images: List[Dict] = field(default_factory=list)
    response_parts: List["ResponsePart"] = field(default_factory=list)
    has_tools: bool = False
    pre_text: Optional[str] = None
    pre_text_complete: bool = False


# --- Predicate function for tenacity ---
def is_response_unsatisfactory(response: ChatCompletion) -> bool:
    """
    Check if the OpenAI response is unsatisfactory and should be retried.

    An unsatisfactory response is one that:
    1. Is None or has no choices.
    2. Has empty message content AND no tool_calls (the model produced nothing useful).

    Previously this only retried for finish_reason 'error'/'content_filter',
    which allowed empty responses with finish_reason 'stop' to silently pass through,
    causing the bot to not reply.

    Args:
        response: The ChatCompletion object from the OpenAI API call.

    Returns:
        True if the response is unsatisfactory, False otherwise.
    """
    if response is None or not response.choices:
        logging.warning("Retry triggered: Response object is None or has no choices.")
        return True

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content
    tool_calls = choice.message.tool_calls or []

    # Empty content with no tool_calls means the model produced nothing useful.
    # This can happen with Gemini models returning finish_reason 'stop' but empty
    # content, especially after a tool-call cycle. Retry to get a real response.
    if (not content or not content.strip()) and not tool_calls:
        logging.warning(
            f"Retry triggered: Empty content with finish_reason '{finish_reason}' "
            f"and no tool_calls."
        )
        return True

    return False


logger = logging.getLogger("red.bz_cogs.aiuser")


class LLMPipeline:
    def __init__(self, cog: MixinMeta, ctx: commands.Context, messages: MessagesList):
        self.ctx: commands.Context = ctx
        self.config: Config = cog.config
        self.bot = cog.bot
        self.msg_list: MessagesList = messages
        self.model: str = messages.model
        self.can_reply: bool = messages.can_reply
        self.openai_client = cog.openai_client
        self.enabled_tools: List[ToolCall] = []
        self.available_tools_schemas: List[ToolCallSchema] = []
        self.openrouter_tools: List[dict] = []
        self.completion: Optional[str] = None
        self.reasoning: Optional[str] = None
        self.response_parts: List[ResponsePart] = []
        self.collected_images: List[Dict] = []

    # ------------------------------------------------------------------    
    # Text completeness heuristic
    # ------------------------------------------------------------------
    _BRIDGING_WORDS = frozenset({
        'a', 'an', 'the', 'and', 'or', 'but', 'for', 'in', 'on', 'to',
        'of', 'with', 'from', 'if', 'as', 'that', 'this', 'let', 'like',
        'just', 'one', 'here', 'there', 'where', 'it', 'its', 'my', 'your',
        'is', 'are', 'was', 'were', 'be', 'been', 'not', 'no', 'so', 'then',
        'at', 'by', 'up', 'down', 'out', 'over', 'all', 'some', 'what', 'how',
        'which', 'who', 'whom', 'when', 'why', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'these', 'those', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'such', 'than', 'too', 'very', 'can', 'will', 'shall',
        'may', 'must', 'need', 'dare', 'ought', 'used', 'does', 'doing',
        'having', 'being', 'getting', 'making', 'going', 'here', 'there',
        'now', 'then', 'also', 'well', 'even', 'only', 'still', 'always',
        'never', 'often', 'really', 'quite', 'rather', 'maybe', 'perhaps',
        'please', 'yes', 'yeah', 'sure', 'alright', 'thanks',
    })

    @staticmethod
    def _is_text_incomplete(text: str) -> bool:
        """Check if a text fragment looks like it was interrupted mid-sentence.

        Signs of incompleteness:
        1. Ends with a trailing comma, semicolon, colon, dash, ellipsis
        2. Last word is a bridging word (article, conjunction, preposition, etc.)
        3. Trailing whitespace suggesting the model was about to continue
        4. Ends with a single trailing `-` or `--` (dash)

        Returns True if the text appears incomplete, False if it looks like
        a complete thought ready for early output.
        """
        if not text:
            return False
        stripped = text.strip()
        if not stripped:
            return False

        # Check 1: trailing punctuation that signals continuation
        # Colon is excluded because it's used for emoji syntax (e.g. :smile:)
        incomplete_endings = (',', ';', '-', '–')
        if stripped.endswith(incomplete_endings):
            return True

        # Check 2: trailing dash (single or double) — model was mid-word
        if stripped.endswith('--') or stripped.endswith(' -'):
            return True

        # Check 3: ends with a bridging word
        last_word = stripped.split()[-1].strip('"\'').rstrip('.!?')
        if last_word.lower() in LLMPipeline._BRIDGING_WORDS:
            return True

        # Check 4: trailing whitespace (the model had more to say)
        if text.endswith(' ') or text.endswith('\t'):
            return True

        return False

    # ------------------------------------------------------------------    
    # Setup & configuration
    # ------------------------------------------------------------------
    async def get_service_tier(self) -> Optional[str]:
        """Return the configured service_tier for this guild, or None if not set."""
        return await self.config.guild(self.ctx.guild).service_tier()

    async def get_custom_parameters(self) -> Dict[str, Any]:
        custom_parameters = await self.config.guild(self.ctx.guild).parameters()
        kwargs = json.loads(custom_parameters) if custom_parameters else {}

        if "logit_bias" not in kwargs:
            weights = await self.config.guild(self.ctx.guild).weights()
            if weights:
                try:
                    logit_bias_values = json.loads(weights)
                    if logit_bias_values:
                        kwargs["logit_bias"] = logit_bias_values
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse logit_bias weights for guild {self.ctx.guild.id}: {weights}"
                    )

        if kwargs.get("logit_bias") and (
            self.model in VISION_SUPPORTED_MODELS
            or self.model in UNSUPPORTED_LOGIT_BIAS_MODELS
        ):
            logger.warning(f"logit_bias is not supported for model {self.model}, removing...")
            del kwargs["logit_bias"]

        # Apply service_tier from dedicated config (takes precedence over any
        # service_tier value that may have been set in custom_parameters JSON)
        service_tier = await self.get_service_tier()
        if service_tier is not None:
            kwargs["service_tier"] = service_tier
        elif "service_tier" in kwargs:
            # Allow service_tier in custom_parameters JSON as a fallback
            pass

        return kwargs

    async def get_openrouter_tools(self) -> List[dict]:
        openrouter_tools: List[dict] = []
        if await self.config.guild(self.ctx.guild).openrouter_web_search_enabled():
            tool = OpenRouterWebSearch(self.config, self.ctx)
            openrouter_tools.append(await tool.get_tool_dict())
        if await self.config.guild(self.ctx.guild).openrouter_web_fetch_enabled():
            tool = OpenRouterWebFetch(self.config, self.ctx)
            openrouter_tools.append(await tool.get_tool_dict())
        if await self.config.guild(self.ctx.guild).openrouter_image_generation_enabled():
            tool = OpenRouterImageGeneration(self.config, self.ctx)
            openrouter_tools.append(await tool.get_tool_dict())
        return openrouter_tools

    @staticmethod
    def _is_openrouter_tool_name(tool_name: str) -> bool:
        return tool_name in {
            OpenRouterToolType.WEB_SEARCH.value,
            OpenRouterToolType.WEB_FETCH.value,
            OpenRouterToolType.IMAGE_GENERATION.value,
        }

    async def _build_plugins(self) -> List[dict]:
        from aiuser.functions.openrouter.pdf_parsing import OpenRouterPdfParsing
        try:
            pdf_enabled = await self.config.guild(self.ctx.guild).openrouter_pdf_parsing_enabled()
            if not pdf_enabled:
                return [{"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}]
            service = OpenRouterPdfParsing(self.config, self.ctx)
            return await service.build_plugins()
        except Exception:
            logger.warning("Error building PDF plugins, using default engine", exc_info=True)
            return [{"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}]

    async def _inject_pdf_annotations(self):
        cog = self.bot.get_cog("AIUser")
        if not cog or not hasattr(cog, "pdf_annotations"):
            return
        init_message = self.msg_list.init_message
        if not init_message or not init_message.reference:
            return
        try:
            original_msg = init_message.reference.resolved
            if not original_msg and init_message.reference.message_id:
                try:
                    original_msg = await self.ctx.channel.fetch_message(init_message.reference.message_id)
                except Exception:
                    pass
            if not original_msg:
                return
            cache_key = (self.ctx.channel.id, original_msg.id)
            annotations = cog.pdf_annotations.get(cache_key)
            if annotations:
                logger.info(f"Injecting {len(annotations)} cached PDF annotation(s) from message {original_msg.id} into reply chain")
                if self.msg_list.messages:
                    last_msg = self.msg_list.messages[-1]
                    if not hasattr(last_msg, "annotations"):
                        self._cached_pdf_annotations = annotations
        except Exception:
            logger.warning("Error injecting PDF annotations", exc_info=True)

    def _store_pdf_annotations_from_response(self, response):
        cog = self.bot.get_cog("AIUser")
        if not cog or not hasattr(cog, "pdf_annotations"):
            return
        try:
            init_message = self.msg_list.init_message
            if not init_message:
                return
            annotations = None
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "annotations") and message.annotations:
                    annotations = message.annotations
                elif hasattr(message, "model_extra") and message.model_extra:
                    annotations = message.model_extra.get("annotations")
            if annotations:
                cache_key = (self.ctx.channel.id, init_message.id)
                cog.pdf_annotations[cache_key] = annotations
                if init_message.reference:
                    try:
                        original_msg = init_message.reference.resolved
                        if not original_msg and init_message.reference.message_id:
                            try:
                                original_msg = self.ctx.channel.fetch_message(init_message.reference.message_id)
                            except Exception:
                                pass
                        if original_msg:
                            alt_key = (self.ctx.channel.id, original_msg.id)
                            if alt_key not in cog.pdf_annotations:
                                cog.pdf_annotations[alt_key] = annotations
                    except Exception:
                        pass
        except Exception:
            logger.warning("Error storing PDF annotations", exc_info=True)

    async def setup_tools(self):
        if not (await self.config.guild(self.ctx.guild).function_calling()):
            self.enabled_tools = []
            self.available_tools_schemas = []
            self.openrouter_tools = []
            return
        self.enabled_tools = await get_enabled_tools(self.config, self.ctx)
        self.available_tools_schemas = [tool.schema for tool in self.enabled_tools]
        self.openrouter_tools = await self.get_openrouter_tools()
        if self.openrouter_tools:
            logger.info(f"OpenRouter server tools enabled: {[t.get('type', 'unknown') for t in self.openrouter_tools]}")

    # ------------------------------------------------------------------    
    # API call
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize_content_for_log(content, max_text_len: int = 500) -> str:
        """Return a truncated/summarized representation of message content for logging."""
        if isinstance(content, str):
            if len(content) <= max_text_len:
                return content
            return content[:max_text_len] + f"... [truncated, {len(content)} chars total]"
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                item_type = item.get("type", "unknown")
                if item_type == "text":
                    text = item.get("text", "")
                    if len(text) > max_text_len:
                        text = text[:max_text_len] + f"... [{len(text)} chars]"
                    parts.append(f"text: {text!r}")
                elif item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        parts.append(f"image_url: <base64 data, {len(url)} chars>")
                    else:
                        parts.append(f"image_url: {url!r}")
                else:
                    parts.append(f"{item_type}: {str(item)[:200]}")
            return "[" + ", ".join(parts) + "]"
        else:
            return str(content)[:max_text_len]

    def _log_submitted_payload(self, model: str, messages_json: list, user_digest: str, localKwargs: dict):
        """Log the full submitted payload (messages + kwargs) at DEBUG level.

        Logs a summary at INFO level and the complete per-message detail
        at DEBUG level so operators can inspect the exact data sent to the LLM API.
        """
        channel_id = self.ctx.channel.id
        guild_id = self.ctx.guild.id if self.ctx.guild else "DM"

        # Sanitize kwargs for logging: remove large/sensitive extra_body fields
        kwargs_for_log = copy.deepcopy(localKwargs)
        if 'extra_body' in kwargs_for_log and isinstance(kwargs_for_log['extra_body'], dict):
            extra = kwargs_for_log['extra_body']
            # Don't log safetySettings or session_id — they are boilerplate
            extra.pop('safetySettings', None)
            extra.pop('session_id', None)

        # Sanitize messages for logging: truncate large image data URLs
        messages_for_log = []
        for msg in messages_json:
            msg_copy = {}
            for k, v in msg.items():
                if k == "content":
                    msg_copy[k] = self._summarize_content_for_log(v)
                else:
                    msg_copy[k] = v
            messages_for_log.append(msg_copy)

        logger.info(
            "=== Submitted Payload Start === model=%s channel=%s guild=%s user=%s messages=%d kwargs=%s ===",
            model, channel_id, guild_id, user_digest[:12],
            len(messages_json),
            json.dumps(kwargs_for_log, default=str, ensure_ascii=False),
        )

        if logger.isEnabledFor(logging.DEBUG):
            for i, msg in enumerate(messages_for_log):
                extra_parts = []
                if "tool_calls" in msg:
                    tc_names = []
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            tc_names.append(tc.get("function", {}).get("name", "?"))
                        else:
                            tc_names.append(str(tc))
                    extra_parts.append(f"tool_calls={tc_names}")
                if "tool_call_id" in msg:
                    extra_parts.append(f"tool_call_id={msg['tool_call_id']}")
                if "name" in msg:
                    extra_parts.append(f"name={msg['name']}")

                extras = f" ({', '.join(extra_parts)})" if extra_parts else ""
                logger.debug("  [%d] role=%s%s content=%s", i, msg.get("role"), extras, msg.get("content"))

        logger.debug("=== Submitted Payload End === model=%s channel=%s ===", model, channel_id)

    async def call_client(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], List[ChatCompletionMessageToolCall], Optional[List[Dict]], Optional[Dict]]:
        injected_annotations = getattr(self, "_cached_pdf_annotations", None)
        current_messages_json = self.msg_list.get_json(annotations_for_assistant=injected_annotations)
        plugins = await self._build_plugins()

        localKwargs = copy.deepcopy(kwargs)

        if 'extra_body' in localKwargs:
            if isinstance(localKwargs['extra_body'], dict):
                existing = localKwargs['extra_body'].get("plugins", [])
                localKwargs['extra_body']["plugins"] = existing + plugins
            else:
                localKwargs['extra_body'] = {"plugins": plugins}
        else:
            localKwargs['extra_body'] = {"plugins": plugins}

        user = f"{self.ctx.me.id}-{self.ctx.channel.id}"
        m = hashlib.sha256()
        m.update(user.encode('utf-8'))
        user_digest = m.hexdigest()

        localKwargs['extra_body']['safetySettings'] = [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ]
        
        localKwargs['extra_body']['session_id'] = user_digest

        # Also inject service_tier into extra_body for endpoints that expect it there
        if localKwargs.get('service_tier'):
            localKwargs['extra_body']['service_tier'] = localKwargs['service_tier']

        # Log the full submitted payload (context, messages, kwargs)
        self._log_submitted_payload(self.model, current_messages_json, user_digest, localKwargs)

        logger.info(f"Sending request to LLM (model: {self.model}) with {len(current_messages_json)} messages. Kwarg keys: {list(localKwargs.keys())}")

        response: ChatCompletion = await self._create_completion_with_retry(
            model=self.model, messages=current_messages_json, user=user_digest, **localKwargs
        )

        self._store_pdf_annotations_from_response(response)

        if response.usage:
            logger.info(f"LLM usage: P{response.usage.prompt_tokens} C{response.usage.completion_tokens} T{response.usage.total_tokens}.")
            # Log Gemini implicit cache hit metrics (available via OpenRouter)
            try:
                cached = getattr(response.usage, 'cached_tokens', None)
                if cached is None:
                    details = getattr(response.usage, 'prompt_tokens_details', None)
                    if details:
                        cached = getattr(details, 'cached_tokens', None)
                if cached and cached > 0:
                    logger.info(f"Cache hit: {cached}/{response.usage.prompt_tokens} prompt tokens served from cache.")
            except Exception:
                pass  # Non-critical; don't let cache logging break the pipeline

        logger.info(f"Raw LLM response (truncated): id={response.id}, finish_reason={response.choices[0].finish_reason}")

        message = response.choices[0].message
        llm_content = getattr(message, "content", None)
        llm_reasoning = getattr(message, "reasoning", None)
        llm_tool_calls = message.tool_calls or []

        llm_reasoning_details = getattr(message, "reasoning_details", None)
        if llm_reasoning_details is None and getattr(message, "model_extra", None):
            llm_reasoning_details = message.model_extra.get("reasoning_details")

        llm_model_extra = getattr(message, "model_extra", None)
        if llm_model_extra is None and len(response.choices) > 0:
            llm_model_extra = getattr(response.choices[0].message, "model_extra", None)

        return llm_content, llm_reasoning, llm_tool_calls, llm_reasoning_details, llm_model_extra

    @retry(
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))
        or retry_if_result(is_response_unsatisfactory),
    )
    async def _create_completion_with_retry(self, model = None, messages = None, user = None, stream=False, **kwargs) -> ChatCompletion:
        try:
            result = await self.openai_client.chat.completions.create(model=model, messages=messages, user=user, stream=stream, **kwargs)
            if not result.choices:
                raise openai.APIError(f"API returned no choices: {result}")
            native_finish_reason = getattr(result.choices[0], "native_finish_reason", None) or result.choices[0].finish_reason
            logger.info(f"Finish reason: {result.choices[0].finish_reason}. Native finish reason: {native_finish_reason}")
            return result
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise

    # ------------------------------------------------------------------
    # Tool-calling loop
    # ------------------------------------------------------------------
    _STOP_INSTRUCTION = (
        "You have reached the maximum number of tool-calling rounds. "
        "Based on all the information you have gathered, provide your final, "
        "complete response now. Do NOT call any more tools — respond directly to the user."
    )

    async def run(self) -> PipelineResult:
        """Run the tool-calling loop with configurable max rounds.

        On the first round, if no tools are called, returns immediately.
        If tool calls are present, executes tools and loops for additional rounds.
        On the final round, uses ``tool_choice="none"`` and injects a user-role
        stop instruction (to preserve system-prompt caching) to discourage
        further tool calls.

        Returns:
            PipelineResult with text, reasoning, images, has_tools, etc.
        """
        try:
            return await self._run_loop()
        except httpx.ReadTimeout:
            logger.error(f"LLM request to {self.model} timed out.")
            await self.ctx.react_quietly("💤", message="`aiuser` request timed out")
        except openai.RateLimitError:
            logger.warning(f"LLM request to {self.model} was rate-limited.")
            await self.ctx.react_quietly("💤", message="`aiuser` request ratelimited")
        except openai.APIConnectionError as e:
            logger.error(f"LLM API connection error: {e}")
            await self.ctx.react_quietly("⚠️", message="`aiuser` could not connect to LLM API")
        except openai.APIStatusError as e:
            logger.error(f"LLM API error (Status {e.status_code}): {e.response.text if e.response else 'No response body'}")
            await self.ctx.react_quietly("⚠️", message=f"`aiuser` LLM API error (Status {e.status_code})")
        except Exception:
            logger.exception("Unexpected error during LLM processing")
            await self.ctx.react_quietly("⚠️", message="`aiuser` request failed")
        return PipelineResult()

    async def _run_loop(self) -> PipelineResult:
        """Execute the tool-calling loop (core logic, no error handling)."""
        # --- Setup (once) ---
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()
        await self._inject_pdf_annotations()

        self.response_parts = []
        self.collected_images = []

        max_rounds = await self.config.guild(self.ctx.guild).max_tool_rounds()
        if not max_rounds or max_rounds < 1:
            max_rounds = 2

        tools_available = bool(self.available_tools_schemas or self.openrouter_tools)

        logger.info("Tool-calling loop: max_rounds=%d, tools_available=%s", max_rounds, tools_available)

        # --- State tracking across rounds ---
        first_round_text = None
        first_round_reasoning = None
        first_round_model_extra = None
        all_pre_texts: List[str] = []
        accumulated_reasoning: List[str] = []
        has_tools = False
        forced_break = False

        # Updated each iteration
        response_text = None
        reasoning_text = None
        model_extra = None
        last_or_tool_calls: List[ChatCompletionMessageToolCall] = []
        last_or_model_extra = None

        for round_num in range(1, max_rounds + 1):
            is_last = (round_num == max_rounds)

            # --- Build kwargs for this round ---
            kwargs = copy.deepcopy(custom_kwargs)

            if tools_available:
                kwargs["tools"] = [asdict(schema) for schema in self.available_tools_schemas] + self.openrouter_tools
                kwargs["tool_choice"] = "none" if is_last else "auto"
                if "gemini-3" in self.model.lower():
                    kwargs["parallel_tool_calls"] = True

            if 'extra_body' not in kwargs or not isinstance(kwargs.get('extra_body'), dict):
                kwargs['extra_body'] = {}
            kwargs['extra_body']['modalities'] = ['text']
            if 'openrouter:image_generation' in kwargs.get("tools", []):
                kwargs['extra_body']['modalities'].append('image')

            # Inject user-role stop instruction on the last round to discourage
            # further tool calls.  A user role is used instead of system to avoid
            # breaking provider-level system-prompt caching (e.g. Google context cache).
            stop_entry = None
            if is_last and tools_available:
                stop_entry = MessageEntry("user", self._STOP_INSTRUCTION)
                self.msg_list.messages.append(stop_entry)

            # --- Call LLM ---
            response_text, reasoning_text, tool_calls, reasoning_details, model_extra = await self.call_client(kwargs)

            # Remove stop instruction (before adding assistant message)
            if stop_entry:
                try:
                    self.msg_list.messages.remove(stop_entry)
                except ValueError:
                    pass

            # --- Logging ---
            logger.info(
                "Round %d/%d: text_length=%d, tool_calls=%d",
                round_num, max_rounds, len(response_text or ''), len(tool_calls)
            )
            if reasoning_text:
                logger.info(
                    "Round %d reasoning: length=%d, preview=%s",
                    round_num, len(reasoning_text), reasoning_text[:500]
                )
            else:
                logger.info("Round %d: No reasoning output from model", round_num)
            for tc in tool_calls:
                logger.info(
                    "Round %d tool call: id=%s name=%s args=%s",
                    round_num, tc.id, tc.function.name, tc.function.arguments[:200]
                )

            # --- Add assistant response to history ---
            await self.msg_list.add_assistant(
                content=response_text,
                tool_calls=tool_calls,
                reasoning_details=reasoning_details,
                index=len(self.msg_list) + 1
            )

            # --- Track first-round state ---
            if round_num == 1:
                first_round_text = response_text
                first_round_reasoning = reasoning_text
                first_round_model_extra = model_extra

            # Track reasoning across rounds
            if reasoning_text:
                accumulated_reasoning.append(reasoning_text)

            # --- No tool calls — loop complete ---
            if not tool_calls:
                logger.info("Round %d/%d: No tool calls, loop complete", round_num, max_rounds)
                break

            # --- Has tool calls ---
            has_tools = True
            all_pre_texts.append(response_text or "")

            # Split into local vs OpenRouter server tools
            local_tool_calls = [tc for tc in tool_calls if not self._is_openrouter_tool_name(tc.function.name)]
            openrouter_tool_calls = [tc for tc in tool_calls if self._is_openrouter_tool_name(tc.function.name)]
            if openrouter_tool_calls:
                last_or_tool_calls = openrouter_tool_calls
                last_or_model_extra = model_extra

            # Execute local tools in parallel
            if local_tool_calls:
                logger.info("Round %d: Executing %d local tool(s)", round_num, len(local_tool_calls))
                await self._process_and_add_tool_results(local_tool_calls)
                logger.info("Round %d: Local tool execution completed", round_num)
            else:
                logger.info("Round %d: No local tools to execute", round_num)

            # Add synthetic tool results for OpenRouter server-side tools
            for tc in openrouter_tool_calls:
                logger.info("Round %d: Adding OpenRouter synthetic result: id=%s name=%s", round_num, tc.id, tc.function.name)
                await self.msg_list.add_tool_result(
                    name=tc.function.name,
                    tool_call_id=tc.id,
                    content=f"OpenRouter server-side tool '{tc.function.name}' was executed. "
                            f"The results have been incorporated into the conversation history by OpenRouter.",
                    index=len(self.msg_list) + 1
                )

            # Last round with tool_calls — force break
            if is_last:
                logger.warning(
                    "Round %d/%d (last): LLM returned tool_calls despite tool_choice='none'. "
                    "Executing tools then force-breaking.", round_num, max_rounds
                )
                forced_break = True
                break

        # ====================================================================
        # Build final result
        # ====================================================================

        combined_pre_text = "".join(all_pre_texts)

        if not has_tools:
            # --- No tools: simple response ---
            self.completion = response_text
            self.reasoning = reasoning_text
            self.response_parts = [ResponsePart(type="text", content=response_text or "")]

            # Handle transparent OpenRouter image gen (no tool_calls but image gen tool present)
            if self.openrouter_tools and response_text:
                has_image_gen_tool = any(t.get("type") == OpenRouterToolType.IMAGE_GENERATION.value for t in self.openrouter_tools)
                if has_image_gen_tool and model_extra:
                    await OpenRouterImageGeneration.handle_tool_response_content(
                        response_text, self.ctx, model_extra=model_extra,
                    )

            return PipelineResult(
                text=response_text,
                reasoning=reasoning_text,
                images=self.collected_images,
                response_parts=self.response_parts,
                has_tools=False,
                pre_text=response_text,
                pre_text_complete=not self._is_text_incomplete(response_text or ""),
            )

        # --- Has tools: combine results across rounds ---

        if forced_break:
            # All rounds had tool_calls — no final response text
            final_text = combined_pre_text
            final_response_text = None
        else:
            # The last round had no tool_calls — its text is the final response
            final_response_text = response_text
            is_replacement = (
                combined_pre_text
                and final_response_text
                and final_response_text.strip().startswith(combined_pre_text.strip())
            )
            if is_replacement:
                final_text = final_response_text
            else:
                final_text = combined_pre_text + (final_response_text or "")

        # Combine reasoning from all rounds
        final_reasoning = None
        if len(accumulated_reasoning) == 1:
            final_reasoning = accumulated_reasoning[0]
        elif len(accumulated_reasoning) > 1:
            final_reasoning = "\n\n".join(accumulated_reasoning)

        # Process OpenRouter server tool results
        if last_or_tool_calls and (final_response_text or combined_pre_text):
            await self._process_openrouter_tool_results(
                last_or_tool_calls,
                final_response_text or combined_pre_text,
                model_extra=last_or_model_extra or first_round_model_extra,
            )

        # Build response parts
        is_incomplete = self._is_text_incomplete(first_round_text or "")

        if is_incomplete:
            part = ResponsePart(
                type="text_and_images",
                content=final_text,
                images=self.collected_images,
            )
            self.response_parts = [part]
        else:
            part = ResponsePart(
                type="text_and_images",
                content=final_response_text or combined_pre_text,
                images=self.collected_images,
            )
            self.response_parts = [part] if part else []

        logger.info(
            "Loop complete: has_tools=%s, forced_break=%s, final_text_length=%d, images=%d, response_parts=%d",
            has_tools, forced_break, len(final_text or ''), len(self.collected_images), len(self.response_parts)
        )

        self.completion = final_text
        self.reasoning = final_reasoning

        return PipelineResult(
            text=final_text,
            reasoning=final_reasoning,
            images=self.collected_images,
            response_parts=self.response_parts,
            has_tools=True,
            pre_text=first_round_text,
            pre_text_complete=not is_incomplete,
        )

    def _collect_images_from_tool(self, tool_function_name: str) -> List[Dict]:
        """Collect generated/edited images from an image-producing tool.

        Returns the list of image dicts (and clears the tool's internal list),
        or an empty list if the tool is not an image-producing tool.
        """
        for tool_obj in self.enabled_tools:
            if tool_obj.function_name == tool_function_name and (
                isinstance(tool_obj, GenerateImageToolCall) or isinstance(tool_obj, EditImageToolCall)
            ):
                return tool_obj.get_generated_images()
        return []

    async def _process_and_add_tool_results(self, tool_calls: List[ChatCompletionMessageToolCall]):
        async def process_single_tool(tool_call):
            tool_function_name = tool_call.function.name
            tool_call_id = tool_call.id
            logger.info("Processing tool call: id=%s name=%s", tool_call_id, tool_function_name)
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON arguments for tool {tool_function_name}: {e}")
                return tool_call_id, tool_function_name, f"Error: Invalid JSON arguments for '{tool_function_name}'."
            logger.info(
                "Executing tool: id=%s name=%s args=%s",
                tool_call_id, tool_function_name, json.dumps(arguments, default=str)[:500]
            )
            result = await self.run_tool(tool_function_name, arguments)
            logger.info("Tool result: id=%s name=%s result_length=%d", tool_call_id, tool_function_name,
                        len(result) if isinstance(result, list) else len(str(result)))
            return tool_call_id, tool_function_name, result

        logger.info("Processing %d local tool call(s) in parallel", len(tool_calls))
        tasks = [process_single_tool(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            # Handle exceptions from asyncio.gather (return_exceptions=True)
            if isinstance(result, Exception):
                logger.error("Tool execution raised an unhandled exception: %s", result, exc_info=result)
                continue
            tool_call_id, tool_function_name, tool_result_content = result

            if isinstance(tool_result_content, list):
                # Tool already returned multimodal content parts (image_url +
                # text).  Per Gemini 3.5 Flash guidance: "include multimodal
                # content inside the function response, not outside it."
                # The tool has already built the correct structure — use it
                # directly.  Still collect images for Discord sending.
                tool_images = self._collect_images_from_tool(tool_function_name)
                for img in tool_images:
                    self.collected_images.append(img)
                logger.info(
                    "Tool '%s' returned %d multimodal content parts directly",
                    tool_function_name, len(tool_result_content)
                )

            # Add tool result to context — this is the "user" turn with functionResponse
            # Per Gemini 3.5 Flash: id and name MUST match the preceding functionCall
            logger.info(
                "Adding tool result to context: id=%s name=%s content_preview=%s",
                tool_call_id, tool_function_name,
                str(tool_result_content)[:200] if isinstance(tool_result_content, str) else f"[{len(tool_result_content)} parts]"
            )
            await self.msg_list.add_tool_result(
                name=tool_function_name, tool_call_id=tool_call_id, content=tool_result_content, index=len(self.msg_list) + 1
            )

        # Verify all function calls have matching function responses in context
        self._verify_tool_call_response_matching()

    def _verify_tool_call_response_matching(self):
        """Verify that every functionCall has a matching functionResponse in context.

        Per Gemini 3.5 Flash doc:
        - Every FunctionResponse must include the id from the corresponding FunctionCall
        - The name in the response must match the name in the call
        - Return exactly one FunctionResponse for each FunctionCall received

        Logs warnings if any mismatches are found.
        """
        messages = self.msg_list.messages
        # Build maps of tool_calls (from assistant messages) and tool_results (from tool messages)
        tool_calls_map = {}  # id -> name
        tool_results_map = {}  # id -> name

        for entry in messages:
            if entry.role == "assistant" and entry.tool_calls:
                for tc in entry.tool_calls:
                    tc_id = getattr(tc, 'id', None)
                    tc_func = getattr(tc, 'function', None)
                    tc_name = getattr(tc_func, 'name', None) if tc_func else None
                    if tc_id and tc_name:
                        tool_calls_map[tc_id] = tc_name
            elif entry.role == "tool" and entry.tool_call_id:
                tool_results_map[entry.tool_call_id] = entry.name

        if not tool_calls_map:
            return

        # Check 1: Every tool_call has a matching tool_result
        for tc_id, tc_name in tool_calls_map.items():
            if tc_id not in tool_results_map:
                logger.warning(
                    "Context verification FAILED: tool_call id=%s name=%s has NO matching functionResponse",
                    tc_id, tc_name
                )
            elif tool_results_map[tc_id] != tc_name:
                logger.warning(
                    "Context verification FAILED: tool_call id=%s name=%s does NOT match functionResponse name=%s",
                    tc_id, tc_name, tool_results_map[tc_id]
                )
            else:
                logger.info(
                    "Context verification OK: tool_call id=%s name=%s has matching functionResponse",
                    tc_id, tc_name
                )

        # Check 2: No orphaned tool_results without a matching tool_call
        for tr_id, tr_name in tool_results_map.items():
            if tr_id not in tool_calls_map:
                logger.warning(
                    "Context verification WARNING: orphaned functionResponse id=%s name=%s with no matching tool_call",
                    tr_id, tr_name
                )

    async def run_tool(self, tool_name: str, arguments: Dict[str, Any]):
        for tool_obj in self.enabled_tools:
            if tool_obj.function_name == tool_name:
                logger.info(f'Executing tool: "{tool_name}" with args: {arguments}')
                arguments_for_tool = copy.deepcopy(arguments)
                arguments_for_tool["request"] = self
                try:
                    tool_output = await tool_obj.run(arguments_for_tool)
                    if tool_output is None:
                        return f"Tool '{tool_name}' executed successfully with no specific textual result."
                    # Preserve list returns (multimodal content parts) as-is;
                    # only convert non-list results to string.
                    if isinstance(tool_output, list):
                        return tool_output
                    return str(tool_output)
                except Exception as e:
                    logger.exception(f"Error during execution of tool '{tool_name}'")
                    return f"Error: Tool '{tool_name}' encountered an unhandled exception: {str(e)}"
        logger.warning(f'Tool "{tool_name}" not found or not enabled.')
        return f"Error: Tool '{tool_name}' is not available or not recognized."

    async def _process_openrouter_tool_results(self, tool_calls, response_text, model_extra=None):
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name == OpenRouterToolType.IMAGE_GENERATION.value:
                logger.info("Processing image_generation server tool response")
                await OpenRouterImageGeneration.handle_tool_response_content(response_text, self.ctx, model_extra=model_extra)
            elif tool_name == OpenRouterToolType.WEB_SEARCH.value:
                logger.info("Web search server tool was used")
            elif tool_name == OpenRouterToolType.WEB_FETCH.value:
                logger.info("Web fetch server tool was used")

    async def get_response_parts(self) -> List[ResponsePart]:
        return self.response_parts