

# DEPRECATED: This file is kept for backward compatibility.
# The search_google function has been superseded by the generic web_search
# ToolCall in aiuser/functions/web_search/tool_call.py.
# Existing configs using "search_google" will be auto-migrated on cog load.
# New setups should use: [p]aiuser functions web_search


from aiuser.functions.search.query import search_google
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import (Function, Parameters,
                                                  ToolCallSchema)


class SearchToolCall(ToolCall):
    schema = ToolCallSchema(function=Function(
        name="search_google",
        description="Searches Google using the query for any unknown information or most current infomation",
        parameters=Parameters(
            properties={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
            },
            required=["query"]
        )))
    function_name = schema.function.name

    async def _handle(self, arguments):
        return await search_google(arguments["query"], (await self.bot.get_shared_api_tokens("serper")).get("api_key"), self.ctx)
