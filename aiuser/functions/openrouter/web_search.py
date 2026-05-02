from redbot.core import Config, commands

from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    WebSearchParameters,
    build_openrouter_tool_dict,
    deserialize_parameters,
)


class OpenRouterWebSearch:
    """Represents the openrouter:web_search server tool.

    This is NOT a ToolCall subclass. It is a server-side tool handled entirely
    by OpenRouter. It only provides the raw dict to include in the OpenAI tools array.
    """
    tool_type = OpenRouterToolType.WEB_SEARCH

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx

    async def get_tool_dict(self) -> dict:
        """Build the raw tool dict for the OpenAI tools array.

        Reads config for enabled status and custom parameters, deserializes
        parameters, and returns a dict like:
            {"type": "openrouter:web_search", "parameters": {...}}
        """
        params_json = await self.config.guild(self.ctx.guild).openrouter_web_search_parameters()
        params: WebSearchParameters = deserialize_parameters(params_json, self.tool_type)
        return build_openrouter_tool_dict(
            self.tool_type,
            {
                "engine": params.engine,
                "max_results": params.max_results,
                "max_total_results": params.max_total_results,
                "search_context_size": params.search_context_size,
                "user_location": params.user_location,
                "allowed_domains": params.allowed_domains,
                "excluded_domains": params.excluded_domains,
            }
        )