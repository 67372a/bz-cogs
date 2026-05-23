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
        description="Search the web for any topic and get clean, ready-to-use content. Best for: " \
        "Finding current information, news, facts, people, companies, or answering questions about " \
        "any topic. Returns: Clean text content from top search results. Search Query tips: should describe the " \
        "ideal page, not keywords. For example, \"blog post comparing React and Vue performance.\" " \
        "Use category:people / category:company to search through Linkedin profiles / companies " \
        "respectively. If highlights are insufficient, follow up with " \
        "web_fetch_exa on the best URLs.",
        parameters=Parameters(
            properties={
                "search_query": {
                    "type": "string",
                    "description": "Natural language search query. Must be a semantically rich "
                    "description of the ideal page, not just keywords. Optionally include "
                    "category:<type> (company, people) to focus results — e.g. 'category:people "
                    "John Doe software engineer'.",
                },
                "guiding_query": {
                    "type": "string",
                    "description": "Optional natural-language description of what to have highlights focus on.",
                },
            },
            required=["search_query"],
        )))
    function_name = "web_search"

    async def _handle(self, arguments):
        search_query = arguments.get("search_query", "")
        if not search_query:
            return "Error: No search_query provided."
        
        guiding_query = arguments.get("guiding_query", None)

        backend = await self.config.guild(self.ctx.guild).web_search_backend()
        if not backend:
            backend = "none"

        config_raw = await self.config.guild(self.ctx.guild).web_search_config()
        config = json.loads(config_raw) if config_raw else {}

        logger.info("web_search: backend=%s search_query=%s guiding_query=%s", backend, search_query, guiding_query)
        provider = get_search_provider(backend)
        return await provider.search(search_query, guiding_query, self.bot, self.ctx, config)
