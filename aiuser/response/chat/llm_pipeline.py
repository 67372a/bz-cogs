# llm_pipeline.py

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from aiuser.config.models import (
    UNSUPPORTED_LOGIT_BIAS_MODELS,
    VISION_SUPPORTED_MODELS,
)
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import ToolCallSchema
from aiuser.messages_list.messages import MessagesList
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import get_enabled_tools
from redbot.core import Config, commands


logger = logging.getLogger("red.bz_cogs.aiuser")

MAX_TOOL_ITERATIONS = 3
MAX_CONCURRENT_TOOL_CALLS = 5


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
        self.completion: Optional[str] = None
        self.reasoning: Optional[str] = None

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
            {"id": "file-parser", "pdf": {"engine": "native"}}
        ]

        if 'extra_body' in kwargs:
            kwargs['extra_body'].update({"plugins": plugins})
        else:
            kwargs['extra_body'] = {"plugins": plugins}

        user = f"{self.ctx.me.id}-{self.ctx.channel.id}"
        m = hashlib.sha256()
        m.update(user.encode('utf-8'))
        user_digest = m.hexdigest()
        kwargs['extra_body'].update({"user": user_digest})
        kwargs['user'] = user_digest

        logger.info(
            f"Sending request to LLM (model: {self.model}) with {len(current_messages_json)} messages. Kwarg keys: {list(kwargs.keys())}"
        )

        response: ChatCompletion = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=current_messages_json,
            **kwargs
        )

        if response.usage:
            logger.info(
                f"LLM usage: P{response.usage.prompt_tokens} C{response.usage.completion_tokens} T{response.usage.total_tokens}. Finish: {response.choices[0].finish_reason}"
            )
        else:
            logger.info(f"LLM response: {response}")

        message = response.choices[0].message
        llm_content = getattr(message, "content", None)
        llm_reasoning = getattr(message, "reasoning", None)
        llm_tool_calls = message.tool_calls or []

        return llm_content, llm_reasoning, llm_tool_calls

    async def create_completion(self) -> Tuple[Optional[str], Optional[str]]:
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()

        # This will hold the last known text from the LLM in case of a timeout
        last_known_response: Optional[str] = None
        last_known_reasoning: Optional[str] = None

        for i in range(MAX_TOOL_ITERATIONS):
            iteration_kwargs = custom_kwargs.copy()
            if self.available_tools_schemas:
                iteration_kwargs["tools"] = [
                    asdict(schema) for schema in self.available_tools_schemas
                ]
                if i == MAX_TOOL_ITERATIONS - 1:
                    iteration_kwargs["tool_choice"] = "none"
                    logger.info("Max tool iterations reached. Forcing text response from LLM.")
                else:
                    iteration_kwargs["tool_choice"] = "auto"
            else:
                iteration_kwargs.pop("tools", None)
                iteration_kwargs.pop("tool_choice", None)

            response_text, reasoning_text, response_tool_calls = await self.call_client(
                iteration_kwargs
            )

            # --- THE CORRECT FIX ---
            # Unconditionally add the assistant's turn to the message history.
            # This is the most crucial change. The message list now accurately reflects the entire conversation.
            await self.msg_list.add_assistant(
                content=response_text, tool_calls=response_tool_calls
            )

            # Store the latest response text in case we time out.
            last_known_response = response_text
            last_known_reasoning = reasoning_text

            if not response_tool_calls:
                # The LLM has provided a final answer. We're done.
                logger.info(f"LLM returned final text response in iteration {i + 1}.")
                self.completion = response_text
                self.reasoning = reasoning_text
                break
            else:
                # The LLM wants to call tools. Process them and continue the loop.
                logger.info(
                    f"LLM returned {len(response_tool_calls)} tool call(s) in iteration {i + 1}."
                )
                await self._process_tool_calls_parallel(response_tool_calls)
        else:
            # This 'else' block belongs to the 'for' loop. It executes if the loop finishes without a 'break'.
            logger.warning(
                f"Reached max tool iterations ({MAX_TOOL_ITERATIONS}). "
                f"Returning the last known text content from the final iteration."
            )
            self.completion = last_known_response
            self.reasoning = last_known_reasoning

        if self.completion:
            log_preview = f'{self.completion[:250]}{"..." if len(self.completion) > 250 else ""}'
            logger.info(
                f'Final LLM response for guild {self.ctx.guild.name} (model {self.model}): "{log_preview}"'
            )
        else:
            logger.info(
                f"Final LLM response for guild {self.ctx.guild.name} (model {self.model}) is empty/None."
            )
            
        return self.completion, self.reasoning

    async def _process_tool_calls_parallel(
        self, tool_calls: List[ChatCompletionMessageToolCall]
    ):
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOL_CALLS)

        async def run_tool_with_semaphore(tool_call: ChatCompletionMessageToolCall) -> Tuple[str, str]:
            async with semaphore:
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
                    result_content = f"Error: Invalid JSON arguments provided for tool '{tool_function_name}'."
                    return tool_call_id, result_content

                result_content = await self.run_tool(tool_function_name, arguments)
                return tool_call_id, result_content

        tasks = [run_tool_with_semaphore(tc) for tc in tool_calls]
        tool_results = await asyncio.gather(*tasks)

        for tool_call_id, result_content in tool_results:
            await self.msg_list.add_tool_result(
                tool_call_id=tool_call_id, content=result_content
            )

    async def run_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        for tool_obj in self.enabled_tools:
            if tool_obj.function_name == tool_name:
                logger.info(
                    f'Executing tool: "{tool_name}" in guild {self.ctx.guild.name} with args: {arguments}'
                )
                # FIX: Restore the mechanism to pass context to the tool functions
                arguments_for_tool = arguments.copy()
                arguments_for_tool["request"] = self

                try:
                    # FIX: Pass the correct dictionary with the added context
                    tool_output = await tool_obj.run(
                        arguments_for_tool, self.available_tools_schemas
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

    async def run(self) -> Tuple[Optional[str], Optional[str]]:
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
        except openai.APIStatusError as e:
            logger.error(f"LLM API error for {self.model} in guild {self.ctx.guild.name} (Status {e.status_code}): {e.response.text if e.response else 'No response body'}")
            await self.ctx.react_quietly("⚠️", message=f"`aiuser` LLM API error (Status {e.status_code})")
        except Exception:
            logger.exception(f"An unexpected error occurred during LLM processing for model {self.model} in guild {self.ctx.guild.name}")
            await self.ctx.react_quietly("⚠️", message="`aiuser` request failed due to an unexpected error")
        return None, None