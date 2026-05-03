import asyncio
import json
import logging
from dataclasses import asdict
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
from aiuser.messages_list.messages import MessagesList
from aiuser.functions.openrouter import (
    OpenRouterWebSearch,
    OpenRouterWebFetch,
    OpenRouterImageGeneration,
)
from aiuser.types.abc import MixinMeta
from aiuser.types.enums import OpenRouterToolType
from aiuser.utils.utilities import get_enabled_tools

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
    # If there's no response or no choices, it's definitely unsatisfactory.
    if response is None or not response.choices:
        logging.warning("Retry triggered: Response object is None or has no choices.")
        return True

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content

    # These are the reasons we might get an empty but otherwise valid response.
    # We want to retry in these cases if the content is empty.
    retryable_reasons = {'stop', 'error', 'content_filter'}
    
    if finish_reason in retryable_reasons:
        # If content is None or just whitespace, it's unsatisfactory.
        if not content or not content.strip():
            logging.warning(f"Retry triggered: Empty content with finish_reason '{finish_reason}'.")
            return True

    # If the finish_reason is something else (e.g., 'tool_calls') or if content is present,
    # the response is considered satisfactory.
    return False

logger = logging.getLogger("red.bz_cogs.aiuser")

class LLMPipeline:
    def __init__(self, cog: MixinMeta, ctx: commands.Context, messages: MessagesList):
        self.ctx: commands.Context = ctx
        self.config: Config = cog.config
        self.bot = cog.bot
        self.msg_list: MessagesList = messages
        self.model: str = messages.model
        self.can_reply: bool = messages.can_reply # Retained if used elsewhere
        self.openai_client = cog.openai_client
        self.enabled_tools: List[ToolCall] = []
        self.available_tools_schemas: List[ToolCallSchema] = []
        self.openrouter_tools: List[dict] = []
        self.completion: Optional[str] = None # Stores the *final* text completion
        self.reasoning: Optional[str] = None # Stores the *final* text reasoning

    async def get_custom_parameters(self) -> Dict[str, Any]:
        custom_parameters = await self.config.guild(self.ctx.guild).parameters()
        kwargs = json.loads(custom_parameters) if custom_parameters else {}

        if "logit_bias" not in kwargs:
            weights = await self.config.guild(self.ctx.guild).weights()
            if weights: # Only process if weights string is not empty
                try:
                    logit_bias_values = json.loads(weights)
                    if logit_bias_values:  # Ensure the parsed dict is not empty
                        kwargs["logit_bias"] = logit_bias_values
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse logit_bias weights for guild {self.ctx.guild.id}: {weights}"
                    )

        if kwargs.get("logit_bias") and (
            self.model in VISION_SUPPORTED_MODELS
            or self.model in UNSUPPORTED_LOGIT_BIAS_MODELS
        ):
            logger.warning(
                f"logit_bias is not supported for model {self.model}, removing..."
            )
            del kwargs["logit_bias"]

        return kwargs

    async def get_openrouter_tools(self) -> List[dict]:
        """Build list of OpenRouter server tool dicts based on config.

        Checks each of the 3 OpenRouter tool config booleans. If enabled,
        instantiates the corresponding class and calls get_tool_dict().

        Returns:
            List of dicts like [{"type": "openrouter:web_search", "parameters": {...}}]
        """
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
        """Check if a tool name matches an OpenRouter server tool type."""
        return tool_name in {
            OpenRouterToolType.WEB_SEARCH.value,
            OpenRouterToolType.WEB_FETCH.value,
            OpenRouterToolType.IMAGE_GENERATION.value,
        }

    async def _build_plugins(self) -> List[dict]:
        """Build the plugins array with configurable PDF parsing engine.

        Reads the configured engine from guild config and returns
        the file-parser plugin block. Defaults to 'cloudflare-ai'.

        Returns:
            List containing the file-parser plugin dict, or empty list
            if PDF parsing is disabled.
        """
        from aiuser.functions.openrouter.pdf_parsing import OpenRouterPdfParsing
        try:
            pdf_enabled = await self.config.guild(self.ctx.guild).openrouter_pdf_parsing_enabled()
            if not pdf_enabled:
                return [{"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}]

            service = OpenRouterPdfParsing(self.config, self.ctx)
            return await service.build_plugins()
        except Exception:
            logger.warning(f"Error building PDF plugins, using default engine", exc_info=True)
            return [{"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}]

    async def _inject_pdf_annotations(self):
        """Inject cached PDF annotations for reply-chain messages.

        If the current message is a reply to a message that had PDFs processed,
        the annotations from that original message are injected into the
        conversation history to skip re-parsing costs.

        This walks the reply chain back to find the original PDF message.
        """
        cog = self.bot.get_cog("AIUser")
        if not cog or not hasattr(cog, "pdf_annotations"):
            return

        init_message = self.msg_list.init_message
        if not init_message or not init_message.reference:
            return

        try:
            # Try to find the original PDF message in the reply chain
            original_msg = init_message.reference.resolved
            if not original_msg and init_message.reference.message_id:
                try:
                    original_msg = await self.ctx.channel.fetch_message(
                        init_message.reference.message_id
                    )
                except Exception:
                    pass

            if not original_msg:
                return

            cache_key = (self.ctx.channel.id, original_msg.id)
            annotations = cog.pdf_annotations.get(cache_key)

            if annotations:
                logger.info(
                    f"Injecting {len(annotations)} cached PDF annotation(s) from message {original_msg.id} "
                    f"into reply chain for channel {self.ctx.channel.id}"
                )
                # Add a synthetic assistant message with annotations before dispatch
                # This tells OpenRouter the PDFs were already parsed
                if self.msg_list.messages:
                    last_msg = self.msg_list.messages[-1]
                    if not hasattr(last_msg, "annotations"):
                        # We need to modify the messages dict directly for the API call
                        # Store annotations in msg_list for get_json() to use
                        self._cached_pdf_annotations = annotations
        except Exception:
            logger.warning("Error injecting PDF annotations for reply chain", exc_info=True)

    def _store_pdf_annotations_from_response(self, response):
        """Extract and cache file annotations from an API response.

        OpenRouter responses include annotations with parsed file hashes
        and content. Caching these allows reply chains to skip re-parsing.

        Args:
            response: The raw ChatCompletion response object.
        """
        cog = self.bot.get_cog("AIUser")
        if not cog or not hasattr(cog, "pdf_annotations"):
            return

        try:
            init_message = self.msg_list.init_message
            if not init_message:
                return

            # Extract annotations from the response
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
                logger.info(
                    f"Stored {len(annotations)} PDF annotation(s) for message {init_message.id} "
                    f"in channel {self.ctx.channel.id}"
                )

                # Also cache for any bot messages in the reply chain so that
                # the bot's own follow-ups can reference the original PDFs
                if init_message.reference:
                    try:
                        original_msg = init_message.reference.resolved
                        if not original_msg and init_message.reference.message_id:
                            try:
                                original_msg = self.ctx.channel.fetch_message(
                                    init_message.reference.message_id
                                )
                            except Exception:
                                pass
                        if original_msg:
                            cache_key = (self.ctx.channel.id, original_msg.id)
                            if cache_key not in cog.pdf_annotations:
                                cog.pdf_annotations[cache_key] = annotations
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
        self.available_tools_schemas = [
            tool.schema for tool in self.enabled_tools
        ]
        self.openrouter_tools = await self.get_openrouter_tools()
        if self.openrouter_tools:
            logger.info(
                f"OpenRouter server tools enabled for guild {self.ctx.guild.name}: {[t.get('type', 'unknown') for t in self.openrouter_tools]}"
            )

    async def call_client(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], List[ChatCompletionMessageToolCall], Optional[List[Dict]], Optional[Dict]]:
        # Inject cached PDF annotations into the messages JSON if available
        injected_annotations = getattr(self, "_cached_pdf_annotations", None)
        current_messages_json = self.msg_list.get_json(
            annotations_for_assistant=injected_annotations
        )

        # Build plugins with configurable PDF parsing engine
        plugins = await self._build_plugins()

        if 'extra_body' in kwargs:
            if isinstance(kwargs['extra_body'], dict):
                if "plugins" in kwargs['extra_body']:
                    # Merge if there are already plugins (unlikely but safe)
                    existing = kwargs['extra_body'].get("plugins", [])
                    kwargs['extra_body']["plugins"] = existing + plugins
                else:
                    kwargs['extra_body']["plugins"] = plugins
            else:
                kwargs['extra_body'] = {"plugins": plugins}
        else:
            kwargs['extra_body'] = {"plugins": plugins}

        user = f"{self.ctx.me.id}-{self.ctx.channel.id}"

        m = hashlib.sha256()
        m.update(user.encode('utf-8'))

        user_digest = m.hexdigest()


        kwargs['extra_body'].update(
            {"safetySettings": 
             [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ]})

        kwargs['extra_body'].update({"user": user_digest})

        kwargs['user'] = user_digest

        logger.info(
            f"Sending request to LLM (model: {self.model}) with {len(current_messages_json)} messages. Kwarg keys: {list(kwargs.keys())}"
        )

        response: ChatCompletion = await self._create_completion_with_retry(
                model=self.model, 
                messages=current_messages_json, 
                **kwargs
        )

        # Store PDF annotations from response for reply-chain caching
        self._store_pdf_annotations_from_response(response)

        if response.usage:
            logger.info(
                f"LLM usage: P{response.usage.prompt_tokens} C{response.usage.completion_tokens} T{response.usage.total_tokens}."
            )
            logger.info(
                f"Raw LLM usage: {response.usage}."
            )

        logger.info(
            f"Raw LLM response: {response}."
        )

        message = response.choices[0].message
        llm_content = getattr(message, "content", None)  # This can be None
        llm_reasoning = getattr(message, "reasoning", None) # This can be None
        llm_tool_calls = message.tool_calls or []

        llm_reasoning_details = getattr(message, "reasoning_details", None)
        if llm_reasoning_details is None and getattr(message, "model_extra", None):
             llm_reasoning_details = message.model_extra.get("reasoning_details")

        # Capture model_extra for use downstream (e.g., image generation metadata from OpenRouter)
        llm_model_extra = getattr(message, "model_extra", None)
        if llm_model_extra is None and len(response.choices) > 0:
            choice = response.choices[0]
            llm_model_extra = getattr(choice.message, "model_extra", None)

        # ---- DIAGNOSTIC: Log model_extra explicitly ----
        if llm_model_extra is not None:
            logger.info(f"[LLMPipeline] model_extra keys: {list(llm_model_extra.keys())}")
            for me_key, me_value in llm_model_extra.items():
                me_preview = str(me_value)[:500]
                logger.info(f"[LLMPipeline] model_extra['{me_key}'] = {me_preview}")
        else:
            logger.info(f"[LLMPipeline] model_extra is None for model {self.model}")
        # ---- End diagnostic ----

        return llm_content, llm_reasoning, llm_tool_calls, llm_reasoning_details, llm_model_extra

    @retry(
        wait=wait_random_exponential(min=1, max=5), # Wait 1-5 seconds between retries
        stop=stop_after_attempt(4), # Stop after 4 attempts
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError
        )) or retry_if_result(is_response_unsatisfactory),
    )
    async def _create_completion_with_retry(self, **kwargs) -> ChatCompletion:
        """
        A wrapper around the OpenAI API call decorated for retries.
        It retries on both exceptions and unsatisfactory content.
        """
        try:
            result = await self.openai_client.chat.completions.create(**kwargs)

            native_finish_reason = getattr(result.choices[0], "native_finish_reason", None) or result.choices[0].finish_reason
            logger.info(f"Finish reason: {result.choices[0].finish_reason}. Native finish reason: {native_finish_reason}")
            return result
        except Exception as e:
            logging.error(f"Error occured while calling LLM API: {e}")
            raise  

    async def create_completion(self) -> Tuple[Optional[str], Optional[str]]:
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()
        await self._inject_pdf_annotations()

        current_llm_text_response: Optional[str] = None
        current_llm_text_reasoning: Optional[str] = None

        kwargs1 = custom_kwargs.copy()

        if self.available_tools_schemas or self.openrouter_tools:
            kwargs1["tools"] = [
                asdict(schema) for schema in self.available_tools_schemas
            ] + self.openrouter_tools
            kwargs1["tool_choice"] = "auto"

            if "gemini-3" in self.model.lower():
                kwargs1["parallel_tool_calls"] = True

        kwargs1['extra_body']['modalities'] = ['text']
        if 'openrouter:image_generation' in kwargs1["tools"]:
            kwargs1['extra_body']['modalities'].append('image')

        response_text, reasoning_text, response_tool_calls, response_reasoning_details, response_model_extra = await self.call_client(
             kwargs1
        )

        current_llm_text_response = response_text
        current_llm_text_reasoning = reasoning_text

        await self.msg_list.add_assistant(
            content=response_text, 
            tool_calls=response_tool_calls, 
            reasoning_details=response_reasoning_details,
            index=len(self.msg_list) + 1
        )

        # Track model_extra from the first call for image generation
        first_call_model_extra = response_model_extra

        if response_tool_calls:
            # Filter out OpenRouter server tool calls - they're handled server-side
            local_tool_calls = [
                tc for tc in response_tool_calls
                if not self._is_openrouter_tool_name(tc.function.name)
            ]
            openrouter_tool_calls = [
                tc for tc in response_tool_calls
                if self._is_openrouter_tool_name(tc.function.name)
            ]

            if openrouter_tool_calls:
                logger.info(
                    f"OpenRouter server tool calls detected: {[tc.function.name for tc in openrouter_tool_calls]}"
                )

            if local_tool_calls:
                await self._process_and_add_tool_results(local_tool_calls)

            kwargs2 = custom_kwargs.copy()

            # Prevent further function calls
            kwargs2["tools"] = [
                asdict(schema) for schema in self.available_tools_schemas
            ] + self.openrouter_tools
            kwargs2["tool_choice"] = "none"

            kwargs2['extra_body']['modalities'] = ['text']
            if 'openrouter:image_generation' in kwargs1["tools"]:
                kwargs2['extra_body']['modalities'].append('image')

            if "gemini-3" in self.model.lower():
                kwargs2["parallel_tool_calls"] = True

            tool_response_text, tool_reasoning_text, _, tool_response_reasoning_details, tool_response_model_extra = await self.call_client(
                kwargs2
            )
 
            # 1. Add ONLY the new chunk to the history (to maintain valid chat structure)
            await self.msg_list.add_assistant(
                content=tool_response_text, 
                reasoning_details=tool_response_reasoning_details,
                index=len(self.msg_list) + 1
            )

            # Process OpenRouter server tool response artifacts (e.g., image URLs)
            # Pass model_extra from both calls so image generation can inspect metadata
            if openrouter_tool_calls and tool_response_text:
                await self._process_openrouter_tool_results(
                    openrouter_tool_calls,
                    tool_response_text,
                    model_extra=tool_response_model_extra or first_call_model_extra,
                )

            # Combine the previous text with the new text for the final Discord message
            if current_llm_text_response and tool_response_text:
                # Fix for models (like Gemini 3.0) that repeat the pre-tool text in the post-tool response
                if tool_response_text.strip().startswith(current_llm_text_response.strip()):
                    current_llm_text_response = tool_response_text
                else:
                    current_llm_text_response += tool_response_text 
            elif tool_response_text:
                current_llm_text_response = tool_response_text
                
            # 3. Combine reasoning if applicable
            if current_llm_text_reasoning and tool_reasoning_text:
                current_llm_text_reasoning += "\n\n" + tool_reasoning_text
            elif tool_reasoning_text:
                current_llm_text_reasoning = tool_reasoning_text

        # ---- TRANSPARENT PATH: OpenRouter may handle image_generation server-side
        # in a single API call without exposing tool_calls to the client. In this case
        # model_extra will contain the image result data. Check for it even when no
        # tool_calls were returned. ----
        else:
            # No tool_calls returned — but if image gen was enabled, check model_extra
            if self.openrouter_tools:
                has_image_gen_tool = any(
                    t.get("type") == OpenRouterToolType.IMAGE_GENERATION.value
                    for t in self.openrouter_tools
                )
                if has_image_gen_tool and first_call_model_extra:
                    logger.info(
                        "[LLMPipeline] No tool_calls returned, but image_generation was "
                        "enabled and model_extra is present. Checking for transparent "
                        "image generation results."
                    )
                    await OpenRouterImageGeneration.handle_tool_response_content(
                        response_text or "",
                        self.ctx,
                        model_extra=first_call_model_extra,
                    )

        self.reasoning = current_llm_text_reasoning
        self.completion = current_llm_text_response

        if self.completion:
            log_preview = f'{self.completion[:200]}{"..." if len(self.completion) > 200 else ""}'
            logger.info(
                f'Final LLM response for guild {self.ctx.guild.name} (model {self.model}): "{log_preview}"'
            )
        else:
            logger.info(
                f"Final LLM response for guild {self.ctx.guild.name} (model {self.model}) is empty/None."
            )

        return self.completion, self.reasoning

    async def _process_and_add_tool_results(
        self, tool_calls: List[ChatCompletionMessageToolCall]
    ):
        async def process_single_tool(tool_call):
            tool_function_name = tool_call.function.name
            tool_call_id = tool_call.id
            logger.info(
                f"Processing tool call ID {tool_call_id} for function '{tool_function_name}'."
            )
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON arguments for tool {tool_function_name} (ID: {tool_call_id}): {tool_call.function.arguments}. Error: {e}"
                )
                return tool_call_id, tool_function_name, f"Error: Invalid JSON arguments provided for tool '{tool_function_name}'."

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
                logger.info(
                    f'Executing tool: "{tool_name}" in guild {self.ctx.guild.name} with args: {arguments}'
                )
                arguments_for_tool = arguments.copy()
                arguments_for_tool["request"] = self # Pass context if tool needs it

                try:
                    tool_output = await tool_obj.run(
                        arguments_for_tool
                    )
                    if tool_output is None:
                        logger.warning(
                            f"Tool '{tool_name}' executed but returned None. Interpreting as success with no textual output."
                        )
                        return f"Tool '{tool_name}' executed successfully with no specific textual result."
                    return str(tool_output)
                except Exception as e:
                    logger.exception(
                        f"Error during execution of tool '{tool_name}' in guild {self.ctx.guild.name}"
                    )
                    return f"Error: Tool '{tool_name}' encountered an unhandled exception: {str(e)}"

        logger.warning(
            f'Tool "{tool_name}" not found or not enabled in guild {self.ctx.guild.name}.'
        )
        return f"Error: Tool '{tool_name}' is not available or not recognized."

    async def _process_openrouter_tool_results(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        response_text: str,
        model_extra: Optional[Dict] = None,
    ):
        """Process OpenRouter server tool results from the response.

        OpenRouter server tools are handled entirely server-side. The results
        are returned to the model automatically. This method processes any
        response artifacts that need client-side handling (e.g., sending
        generated image URLs to Discord).

        Args:
            tool_calls: The server tool calls from the LLM response.
            response_text: The model's text response after tool execution.
            model_extra: Optional dict of extra metadata from the LLM response
                (may contain image generation data from OpenRouter).
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            if tool_name == OpenRouterToolType.IMAGE_GENERATION.value:
                logger.info(
                    f"Processing image_generation server tool response for guild {self.ctx.guild.name}"
                )
                await OpenRouterImageGeneration.handle_tool_response_content(
                    response_text, self.ctx, model_extra=model_extra
                )
            elif tool_name == OpenRouterToolType.WEB_SEARCH.value:
                logger.info(
                    f"Web search server tool was used by the model in guild {self.ctx.guild.name}"
                )
            elif tool_name == OpenRouterToolType.WEB_FETCH.value:
                logger.info(
                    f"Web fetch server tool was used by the model in guild {self.ctx.guild.name}"
                )

    async def run(self) -> Tuple[str|None, str|None]:
        try:
            return await self.create_completion()
        except httpx.ReadTimeout:
            logger.error(f"LLM request to {self.model} timed out for guild {self.ctx.guild.name}.")
            await self.ctx.react_quietly("💤", message="`aiuser` request timed out")
        except openai.RateLimitError:
            logger.warning(f"LLM request to {self.model} was rate-limited for guild {self.ctx.guild.name}.")
            await self.ctx.react_quietly("💤", message="`aiuser` request ratelimited")
        except openai.APIConnectionError as e:
            logger.error(f"LLM API connection error for {self.model} in guild {self.ctx.guild.name}: {e}")
            await self.ctx.react_quietly("⚠️", message="`aiuser` could not connect to LLM API")
        except openai.APIStatusError as e: # Catches 4xx and 5xx errors from OpenAI
            logger.error(f"LLM API error for {self.model} in guild {self.ctx.guild.name} (Status {e.status_code}): {e.response.text if e.response else 'No response body'}")
            await self.ctx.react_quietly("⚠️", message=f"`aiuser` LLM API error (Status {e.status_code})")
        except Exception:
            logger.exception(f"An unexpected error occurred during LLM processing for model {self.model} in guild {self.ctx.guild.name}")
            await self.ctx.react_quietly("⚠️", message="`aiuser` request failed due to an unexpected error")
        return None, None