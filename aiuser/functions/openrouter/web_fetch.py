from redbot.core import Config, commands

from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    WebFetchParameters,
    build_openrouter_tool_dict,
    deserialize_parameters,
)


class OpenRouterWebFetch:
    """Represents the openrouter:web_fetch server tool.

    This is NOT a ToolCall subclass. It is a server-side tool handled entirely
    by OpenRouter. It only provides the raw dict to include in the OpenAI tools array.
    """
    tool_type = OpenRouterToolType.WEB_FETCH

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx

    async def get_tool_dict(self) -> dict:
        """Build the raw tool dict for the OpenAI tools array.

        Reads config for enabled status and custom parameters, deserializes
        parameters, and returns a dict like:
            {"type": "openrouter:web_fetch", "parameters": {...}}
        """
        params_json = await self.config.guild(self.ctx.guild).openrouter_web_fetch_parameters()
        params: WebFetchParameters = deserialize_parameters(params_json, self.tool_type)
        return build_openrouter_tool_dict(
            self.tool_type,
            {
                "engine": params.engine,
                "max_uses": params.max_uses,
                "max_content_tokens": params.max_content_tokens,
                "allowed_domains": params.allowed_domains,
                "blocked_domains": params.blocked_domains,
            }
        )