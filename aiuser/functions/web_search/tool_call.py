"""WebSearchToolCall — generic web search function for LLM tool calling.

Only the query is passed by the LLM. The backend (exa/serper) and all
configuration are set by the bot owner via [p]aiuser functions web_search_* commands.
"""

import json
import logging

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.functions.web_search.providers import get_search_provider

logger = logging.getLogger("red.bz_cogs.aiuser")


class WebSearchToolCall(ToolCall):
    schema = ToolCallSchema(function=Function(
        name="web_search",
        description="Searches the web for current, factual, or unknown information. Use this to look up anything you don't already know.",
        parameters=Parameters(
            properties={
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            required=["query"],
        )))
    function_name = "web_search"

    async def _handle(self, arguments):
        query = arguments.get("query", "")
        if not query:
            return "Error: No search query provided."

        backend = await self.config.guild(self.ctx.guild).web_search_backend()
        if not backend:
            backend = "none"

        config_raw = await self.config.guild(self.ctx.guild).web_search_config()
        config = json.loads(config_raw) if config_raw else {}

        logger.info("web_search: backend=%s query=%s", backend, query[:100])
        provider = get_search_provider(backend)
        return await provider.search(query, self.bot, self.ctx, config)
