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
from aiuser.types.abc import MixinMeta
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
    retryable_reasons = {'stop', 'length', 'content_filter'}
    
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

    async def setup_tools(self):
        if not (await self.config.guild(self.ctx.guild).function_calling()):
            self.enabled_tools = []
            self.available_tools_schemas = []
            return
        self.enabled_tools = await get_enabled_tools(self.config, self.ctx)
        self.available_tools_schemas = [
            tool.schema for tool in self.enabled_tools
        ]

    async def call_client(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], List[ChatCompletionMessageToolCall]]:
        current_messages_json = self.msg_list.get_json()

        plugins = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "native"
                }
            }
        ]

        if 'extra_body' in kwargs:
            kwargs['extra_body'].update({"plugins": plugins})
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

        return llm_content, llm_reasoning, llm_tool_calls

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

    async def create_completion(self) -> Optional[str]:
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()

        current_llm_text_response: Optional[str] = None
        current_llm_text_reasoning: Optional[str] = None

        kwargs1 = custom_kwargs.copy()

        if self.available_tools_schemas:
            kwargs1["tools"] = [
                asdict(schema) for schema in self.available_tools_schemas
            ]
            kwargs1["tool_choice"] = "auto"
            

        response_text, reasoning_text, response_tool_calls = await self.call_client(
            kwargs1
        )

        current_llm_text_response = response_text
        current_llm_text_reasoning = reasoning_text

        await self.msg_list.add_assistant(
            content=current_llm_text_response, tool_calls=response_tool_calls, index=len(self.msg_list) + 1
        )

        if response_tool_calls:
            await self._process_and_add_tool_results(response_tool_calls)

            kwargs2 = custom_kwargs.copy()

            kwargs2["tools"] = [
                asdict(schema) for schema in self.available_tools_schemas
            ]
            kwargs2["tool_choice"] = "none"

            response_text, reasoning_text, _ = await self.call_client(
                kwargs2
            )

            current_llm_text_response = response_text
            current_llm_text_reasoning = reasoning_text

            await self.msg_list.add_assistant(
                content=current_llm_text_response, index=len(self.msg_list) + 1
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
        for tool_call in tool_calls:
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
                tool_result_content = f"Error: Invalid JSON arguments provided for tool '{tool_function_name}'."
                await self.msg_list.add_tool_result(
                    name=tool_function_name, tool_call_id=tool_call_id, content=tool_result_content, index=len(self.msg_list) + 1
                )
                continue

            tool_result_content = await self.run_tool(tool_function_name, arguments)
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