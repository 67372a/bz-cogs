import logging

import discord
from redbot.core import Config, commands

from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    ImageGenerationParameters,
    build_openrouter_tool_dict,
    deserialize_parameters,
)

logger = logging.getLogger("red.bz_cogs.aiuser")


class OpenRouterImageGeneration:
    """Represents the openrouter:image_generation server tool.

    This is NOT a ToolCall subclass. It is a server-side tool handled entirely
    by OpenRouter. It only provides the raw dict to include in the OpenAI tools array.

    When the model calls this tool, OpenRouter executes image generation server-side
    and returns the result. The response content includes an imageUrl that should
    be extracted and sent to Discord as an embed.
    """
    tool_type = OpenRouterToolType.IMAGE_GENERATION

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx

    async def get_tool_dict(self) -> dict:
        """Build the raw tool dict for the OpenAI tools array.

        Reads config for enabled status and custom parameters, deserializes
        parameters, and returns a dict like:
            {"type": "openrouter:image_generation", "parameters": {...}}
        """
        params_json = await self.config.guild(self.ctx.guild).openrouter_image_generation_parameters()
        params: ImageGenerationParameters = deserialize_parameters(params_json, self.tool_type)
        return build_openrouter_tool_dict(
            self.tool_type,
            {
                "model": params.model,
                "quality": params.quality,
                "size": params.size,
                "aspect_ratio": params.aspect_ratio,
                "background": params.background,
                "output_format": params.output_format,
                "output_compression": params.output_compression,
                "moderation": params.moderation,
            }
        )

    @staticmethod
    async def handle_tool_response_content(content: str, ctx: commands.Context):
        """Parse the model's response for image URLs and send them as Discord embeds.

        When OpenRouter processes an image_generation tool call, it returns the
        result to the model. The model then writes about the image in its text
        response, which may contain the image URL. This method extracts any
        image URLs from the response and sends them as Discord embeds.

        Args:
            content: The text response from the model after tool execution.
            ctx: The command context for sending Discord messages.
        """
        import re
        # Match image URLs in the response content
        # OpenRouter returns {"status": "ok", "imageUrl": "https://..."}
        # The model may include this URL in its text response
        url_pattern = r'https?://[^\s<>"]+\.(?:png|jpg|jpeg|gif|webp)[^\s<>"]*'
        urls = re.findall(url_pattern, content)

        for url in urls:
            try:
                embed = discord.Embed(color=await ctx.embed_color())
                embed.set_image(url=url)
                await ctx.send(embed=embed)
                logger.info(f"Sent OpenRouter generated image to {ctx.channel.name}: {url}")
            except Exception as e:
                logger.error(f"Failed to send image embed for URL {url}: {e}")