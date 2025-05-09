import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.completion import Completion
from redbot.core import Config, commands

from aiuser.config.models import (
    UNSUPPORTED_LOGIT_BIAS_MODELS,
    VISION_SUPPORTED_MODELS,
)
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import ToolCallSchema
from aiuser.messages_list.messages import MessagesList
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import get_enabled_tools

logger = logging.getLogger("red.bz_cogs.aiuser")

MAX_TOOL_CALL_ITERATIONS = 5  # Maximum iterations for tool call sequences


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
    ) -> Tuple[Optional[str], List[ChatCompletionMessageToolCall]]:
        current_messages_json = self.msg_list.get_json()

        logger.info(
            f"Sending request to LLM (model: {self.model}) with {len(current_messages_json)} messages. Kwarg keys: {list(kwargs.keys())}"
        )


        plugins = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "native"
                }
            }
        ]

        response: ChatCompletion = (
            await self.openai_client.chat.completions.create(
                model=self.model, 
                messages=current_messages_json, 
                extra_body={"plugins": plugins},
                **kwargs
            )
        )

        if response.usage:
            logger.info(
                f"LLM usage: P{response.usage.prompt_tokens} C{response.usage.completion_tokens} T{response.usage.total_tokens}. Finish: {response.choices[0].finish_reason}"
            )
        else:
            logger.info(f"LLM Finish reason: {response.choices[0].finish_reason}")

        message = response.choices[0].message
        llm_content = message.content  # This can be None
        llm_tool_calls = message.tool_calls or []

        return llm_content, llm_tool_calls

    async def create_completion(self) -> Optional[str]:
        custom_kwargs = await self.get_custom_parameters()
        await self.setup_tools()

        current_llm_text_response: Optional[str] = None

        for i in range(MAX_TOOL_CALL_ITERATIONS):
            iteration_kwargs = custom_kwargs.copy()
            if self.available_tools_schemas:
                iteration_kwargs["tools"] = [
                    asdict(schema) for schema in self.available_tools_schemas
                ]
                # "tool_choice": "auto" is typically default when tools are provided
            else:
                iteration_kwargs.pop("tools", None)
                iteration_kwargs.pop("tool_choice", None)

            response_text, response_tool_calls = await self.call_client(
                iteration_kwargs
            )
            current_llm_text_response = response_text

            if response_tool_calls:
                logger.info(
                    f"LLM returned {len(response_tool_calls)} tool call(s) in iteration {i + 1}."
                )
                # Add assistant's response (text + tool call requests) to history
                # Assuming MessagesList methods append if index is None
                await self.msg_list.add_assistant(
                    content=current_llm_text_response, tool_calls=response_tool_calls
                )

                await self._process_and_add_tool_results(response_tool_calls)
                # Continue loop to get LLM response based on tool results
            else:
                logger.info(f"LLM returned final response in iteration {i + 1}.")
                break  # No tool calls, this is the final response
        else:
            logger.warning(
                f"Reached max tool iterations ({MAX_TOOL_CALL_ITERATIONS}) for guild {self.ctx.guild.name}. "
                f"Returning the last text content received from LLM, if any."
            )

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
        return self.completion

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
                    tool_call_id=tool_call_id, content=tool_result_content
                )
                continue

            tool_result_content = await self.run_tool(tool_function_name, arguments)
            await self.msg_list.add_tool_result(
                tool_call_id=tool_call_id, content=tool_result_content
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

    async def run(self) -> Optional[str]:
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
        return None
