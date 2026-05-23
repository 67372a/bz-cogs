"""WebFetchToolCall — generic web fetch function for LLM tool calling.

Only URLs are passed by the LLM. The backend (exa/scrape) and all
configuration are set by the bot owner via [p]aiuser functions web_fetch_* commands.
"""

import json
import logging

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.functions.web_fetch.providers import get_fetch_provider

logger = logging.getLogger("red.bz_cogs.aiuser")


class WebFetchToolCall(ToolCall):
    schema = ToolCallSchema(function=Function(
        name="web_fetch",
        description="Read highlights of a webpage's content as clean markdown. Use after web_search " \
        "when search highlights are insufficient OR to read any URL specific URLs provided. Best for: " \
        "Extracting more highlights from known URLs. Batch multiple URLs in one call. " \
        "Returns: Clean text highlights and metadata from the page(s).",
        parameters=Parameters(
            properties={
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to read. Batch multiple URLs in one call.",
                },
                "guiding_query": {
                    "type": "string",
                    "description": "Optional query to guide highlight retrieval, only use if looking for specific information.",
                },
            },
            required=["urls"],
        )))
    function_name = "web_fetch"

    async def _handle(self, arguments):
        urls = arguments.get("urls", [])
        if not urls:
            return "Error: No URLs provided to fetch."
        
        guiding_query = arguments.get("guiding_query", None)

        # Normalize: accept single string as well
        if isinstance(urls, str):
            urls = [urls]

        backend = await self.config.guild(self.ctx.guild).web_fetch_backend()
        if not backend:
            backend = "none"

        config_raw = await self.config.guild(self.ctx.guild).web_fetch_config()
        config = json.loads(config_raw) if config_raw else {}

        logger.info("web_fetch: backend=%s urls=%s guiding_query=%s", backend, urls, guiding_query)
        provider = get_fetch_provider(backend)
        return await provider.fetch(urls, self.bot, self.ctx, config)
