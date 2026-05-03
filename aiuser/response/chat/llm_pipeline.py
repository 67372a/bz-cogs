import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib

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
from aiuser.messages_list.messages import MessagesList
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


# --- Predicate function for tenacity ---
def is_response_unsatisfactory(response: ChatCompletion) -> bool:
    """
    Check if the OpenAI response is unsatisfactory and should be retried.

    An unsatisfactory response is one that:
    1. Is not None and has choices.
    2. Has a finish_reason of 'stop', 'length', or 'content_filter'.
    3. The message content is None or empty.

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

    retryable_reasons = {'stop', 'error', 'content_filter'}
    
    if finish_reason in retryable_reasons:
        if not content or not content.strip():
            logging.warning(f"Retry triggered: Empty content with finish_reason '{finish_reason}'.")
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
        self._phase1_done = False
        self._phase2_done = False

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
    async def call_client(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], List[ChatCompletionMessageToolCall], Optional[List[Dict]], Optional[Dict]]:
        injected_annotations = getattr(self, "_cached_pdf_annotations", None)
        current_messages_json = self.msg_list.get_json(annotations_for_assistant=injected_annotations)
        plugins = await self._build_plugins()

        if 'extra_body' in kwargs:
            if isinstance(kwargs['extra_body'], dict):
                existing = kwargs['extra_body'].get("plugins", [])
                kwargs['extra_body']["plugins"] = existing + plugins
            else:
                kwargs['extra_body'] = {"plugins": plugins}
        else:
            kwargs['extra_body'] = {"plugins": plugins}

        user = f"{self.ctx.me.id}-{self.ctx.channel.id}"
        m = hashlib.sha256()
        m.update(user.encode('utf-8'))
        user_digest = m.hexdigest()

        kwargs['extra_body'].update({
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ],
            "user": user_digest,
        })
        kwargs['user'] = user_digest

        logger.info(f"Sending request to LLM (model: {self.model}) with {len(current_messages_json)} messages. Kwarg keys: {list(kwargs.keys())}")

        response: ChatCompletion = await self._create_completion_with_retry(
            model=self.model, messages=current_messages_json, **kwargs
        )

        self._store_pdf_annotations_from_response(response)

        if response.usage:
            logger.info(f"LLM usage: P{response.usage.prompt_tokens} C{response.usage.completion_tokens} T{response.usage.total_tokens}.")

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
    async def _create_completion_with_retry(self, **kwargs) -> ChatCompletion:
        try:
            result = await self.openai_client.chat.completions.create(**kwargs)
            native_finish_reason = getattr(result.choices[0], "native_finish_reason", None) or result.choices[0].finish_reason
            logger.info(f"Finish reason: {result.choices[0].finish_reason}. Native finish reason: {native_finish_reason}")
            return result
        except Exception as e:
            logging.error(f"Error calling LLM API: {e}")
            raise

    # ------------------------------------------------------------------    
    # Two-phase execution API
    # ------------------------------------------------------------------
    async def phase1(self) -> Tuple[Optional[str], Optional[str], bool, bool]:
        """Phase 1: first LLM call.

        Returns:
            (pre_text, reasoning, has_tool_calls, is_incomplete)
        """
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()
        await self._inject_pdf_annotations()

        self.response_parts = []
        self.collected_images = []

        kwargs1 = custom_kwargs.copy()

        if self.available_tools_schemas or self.openrouter_tools:
            kwargs1["tools"] = [asdict(schema) for schema in self.available_tools_schemas] + self.openrouter_tools
            kwargs1["tool_choice"] = "auto"
            if "gemini-3" in self.model.lower():
                kwargs1["parallel_tool_calls"] = True

        kwargs1['extra_body']['modalities'] = ['text']
        if 'openrouter:image_generation' in kwargs1.get("tools", []):
            kwargs1['extra_body']['modalities'].append('image')

        response_text, reasoning_text, response_tool_calls, response_reasoning_details, response_model_extra = await self.call_client(kwargs1)

        self._phase1_pre_text = response_text
        self._phase1_reasoning = reasoning_text
        self._phase1_tool_calls = response_tool_calls
        self._phase1_model_extra = response_model_extra

        # Add to message history
        await self.msg_list.add_assistant(
            content=response_text,
            tool_calls=response_tool_calls,
            reasoning_details=response_reasoning_details,
            index=len(self.msg_list) + 1
        )

        has_tools = bool(response_tool_calls)
        is_incomplete = self._is_text_incomplete(response_text or "") if has_tools else False

        self._phase1_done = True
        return response_text, reasoning_text, has_tools, is_incomplete

    async def phase2(self) -> Tuple[Optional[str], Optional[str], List[Dict]]:
        """Phase 2: execute tools + second LLM call.

        Must be called after phase1. Returns:
            (final_text, final_reasoning, images)
            - final_text: the text content to send (may differ from raw LLM output
              when Gemini replacement pattern is detected)
            - final_reasoning: combined reasoning from both calls
            - images: list of image-data dicts generated by tools
        """
        if not self._phase1_done:
            raise RuntimeError("phase2() called before phase1()")

        custom_kwargs = await self.get_custom_parameters()
        response_text = self._phase1_pre_text
        reasoning_text = self._phase1_reasoning
        response_tool_calls = self._phase1_tool_calls
        first_call_model_extra = self._phase1_model_extra

        # Filter tool calls
        local_tool_calls = [tc for tc in response_tool_calls if not self._is_openrouter_tool_name(tc.function.name)]
        openrouter_tool_calls = [tc for tc in response_tool_calls if self._is_openrouter_tool_name(tc.function.name)]

        # Execute local tools
        if local_tool_calls:
            for tc in local_tool_calls:
                if tc.function.name == "generate_image":
                    for tool_obj in self.enabled_tools:
                        if isinstance(tool_obj, GenerateImageToolCall):
                            existing = tool_obj.get_generated_images()
                            if existing:
                                self.collected_images.extend(existing)

            await self._process_and_add_tool_results(local_tool_calls)

            for tc in local_tool_calls:
                for tool_obj in self.enabled_tools:
                    if tool_obj.function_name == tc.function.name and isinstance(tool_obj, GenerateImageToolCall):
                        tool_images = tool_obj.get_generated_images()
                        if tool_images:
                            self.collected_images.extend(tool_images)

        # Phase 2 LLM call
        kwargs2 = custom_kwargs.copy()
        kwargs2["tools"] = [asdict(schema) for schema in self.available_tools_schemas] + self.openrouter_tools
        kwargs2["tool_choice"] = "none"
        kwargs2['extra_body']['modalities'] = ['text']
        if 'openrouter:image_generation' in kwargs2.get("tools", []):
            kwargs2['extra_body']['modalities'].append('image')
        if "gemini-3" in self.model.lower():
            kwargs2["parallel_tool_calls"] = True

        tool_response_text, tool_reasoning_text, _, tool_response_reasoning_details, tool_response_model_extra = await self.call_client(kwargs2)

        await self.msg_list.add_assistant(
            content=tool_response_text,
            reasoning_details=tool_response_reasoning_details,
            index=len(self.msg_list) + 1
        )

        # Process OpenRouter server tool results (handles image sending for OpenRouter path)
        if openrouter_tool_calls and tool_response_text:
            await self._process_openrouter_tool_results(
                openrouter_tool_calls,
                tool_response_text,
                model_extra=tool_response_model_extra or first_call_model_extra,
            )

        # Determine final text output based on Gemini replacement pattern
        is_replacement = False
        if response_text and tool_response_text and tool_response_text.strip().startswith(response_text.strip()):
            is_replacement = True

        if is_replacement:
            final_text = tool_response_text
        else:
            final_text = (response_text or "") + (tool_response_text or "")

        # Combine reasoning
        final_reasoning = reasoning_text
        if reasoning_text and tool_reasoning_text:
            final_reasoning = reasoning_text + "\n\n" + tool_reasoning_text
        elif tool_reasoning_text:
            final_reasoning = tool_reasoning_text

        # Build response parts
        is_incomplete = self._is_text_incomplete(response_text or "")

        if is_incomplete:
            # Pre-text is incomplete — combine final text + images into one message
            part = ResponsePart(
                type="text_and_images",
                content=final_text,
                images=self.collected_images,
            )
            self.response_parts = [part]
        else:
            # Pre-text was already sent (or will be sent by the caller).
            # Send final text + images as a single combined message.
            part = ResponsePart(
                type="text_and_images",
                content=tool_response_text or "",
                images=self.collected_images,
            )
            self.response_parts = [part] if part else []

        self.completion = final_text
        self.reasoning = final_reasoning
        self._phase2_done = True

        # Handle transparent OpenRouter image gen (no tool_calls)
        if not response_tool_calls and self.openrouter_tools:
            has_image_gen_tool = any(t.get("type") == OpenRouterToolType.IMAGE_GENERATION.value for t in self.openrouter_tools)
            if has_image_gen_tool and first_call_model_extra:
                await OpenRouterImageGeneration.handle_tool_response_content(
                    response_text or "", self.ctx, model_extra=first_call_model_extra,
                )

        return final_text, final_reasoning, self.collected_images

    # ------------------------------------------------------------------    
    # Legacy API — kept for backward compatibility
    # ------------------------------------------------------------------
    async def create_completion(self) -> Tuple[Optional[str], Optional[str]]:
        """Legacy: runs both phases and returns the combined text + reasoning."""
        pre_text, reasoning, has_tools, _ = await self.phase1()
        if not has_tools:
            self.completion = pre_text
            self.reasoning = reasoning
            self.response_parts = [ResponsePart(type="text", content=pre_text or "")]
            return self.completion, self.reasoning

        final_text, final_reasoning, images = await self.phase2()
        return self.completion, self.reasoning

    def _collect_images_from_tools(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List[Dict]:
        images = []
        for tool_call in tool_calls:
            if tool_call.function.name == "generate_image":
                for tool_obj in self.enabled_tools:
                    if isinstance(tool_obj, GenerateImageToolCall):
                        existing = tool_obj.get_generated_images()
                        if existing:
                            images.extend(existing)
        return images

    async def _process_and_add_tool_results(self, tool_calls: List[ChatCompletionMessageToolCall]):
        async def process_single_tool(tool_call):
            tool_function_name = tool_call.function.name
            tool_call_id = tool_call.id
            logger.info(f"Processing tool call ID {tool_call_id} for function '{tool_function_name}'.")
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON arguments for tool {tool_function_name}: {e}")
                return tool_call_id, tool_function_name, f"Error: Invalid JSON arguments for '{tool_function_name}'."
            result = await self.run_tool(tool_function_name, arguments)
            return tool_call_id, tool_function_name, result

        tasks = [process_single_tool(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for tool_call_id, tool_function_name, tool_result_content in results:
            await self.msg_list.add_tool_result(
                name=tool_function_name, tool_call_id=tool_call_id, content=tool_result_content, index=len(self.msg_list) + 1
            )

    async def run_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        for tool_obj in self.enabled_tools:
            if tool_obj.function_name == tool_name:
                logger.info(f'Executing tool: "{tool_name}" with args: {arguments}')
                arguments_for_tool = arguments.copy()
                arguments_for_tool["request"] = self
                try:
                    tool_output = await tool_obj.run(arguments_for_tool)
                    if tool_output is None:
                        return f"Tool '{tool_name}' executed successfully with no specific textual result."
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

    async def run(self) -> Tuple[List[ResponsePart] | None, str | None, str | None]:
        try:
            response_text, reasoning_text = await self.create_completion()
            return self.response_parts, response_text, reasoning_text
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
        return [], None, None