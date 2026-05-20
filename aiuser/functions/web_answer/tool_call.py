"""WebAnswerToolCall — generic web answer function for LLM tool calling.

Only the question is passed by the LLM. The backend (exa) and all
configuration are set by the bot owner via [p]aiuser functions web_answer_* commands.
"""

import json
import logging

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.functions.web_answer.providers import get_answer_provider

logger = logging.getLogger("red.bz_cogs.aiuser")


class WebAnswerToolCall(ToolCall):
    schema = ToolCallSchema(function=Function(
        name="web_answer",
        description="Gets a direct, cited answer to a question using web search. Returns an answer with citations from source pages.",
        parameters=Parameters(
            properties={
                "question": {
                    "type": "string",
                    "description": "The question to answer",
                },
            },
            required=["question"],
        )))
    function_name = "web_answer"

    async def _handle(self, arguments):
        question = arguments.get("question", "")
        if not question:
            return "Error: No question provided."

        backend = await self.config.guild(self.ctx.guild).web_answer_backend()
        if not backend:
            backend = "none"

        config_raw = await self.config.guild(self.ctx.guild).web_answer_config()
        config = json.loads(config_raw) if config_raw else {}

        logger.info("web_answer: backend=%s question=%s", backend, question[:100])
        provider = get_answer_provider(backend)
        return await provider.answer(question, self.bot, self.ctx, config)
