

# DEPRECATED: This file is kept for backward compatibility.
# The open_url function has been superseded by the generic web_fetch
# ToolCall in aiuser/functions/web_fetch/tool_call.py.
# Existing configs using "open_url" will be auto-migrated on cog load.
# New setups should use: [p]aiuser functions web_fetch


import logging

from aiuser.functions.scrape.scrape import scrape_page
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema

logger = logging.getLogger("red.bz_cogs.aiuser")


class ScrapeToolCall(ToolCall):
    schema = ToolCallSchema(function=Function(
        name="open_url",
        description="Opens a URL or link and returns the content of it, does not support non-text content types",
        parameters=Parameters(
            properties={
                    "url": {
                        "type": "string",
                        "description": "The URL or link to open",
                    }
            },
            required=["url"]
        )))
    function_name = schema.function.name

    async def _handle(self, arguments):
        logger.info(f'Attempting scrape of {arguments["url"]} in {self.ctx.guild}')
        try:
            return await scrape_page(arguments["url"])
        except Exception as exc:
            logger.exception(f"Failed to scrape {arguments['url']}")
            return "An error occured while attempting to access the given URL."
