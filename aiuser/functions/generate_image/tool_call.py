import asyncio
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
import discord
from openai import AsyncOpenAI
from redbot.core import Config, commands
from redbot.core.bot import Red

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.types.openrouter_types import (
    DirectImageGenerationParameters,
    deserialize_parameters,
)

logger = logging.getLogger("red.bz_cogs.aiuser")


class GenerateImageToolCall(ToolCall):
    """Local ToolCall that generates images via direct OpenRouter API call.

    Unlike `OpenRouterImageGeneration` (a server-side tool handled entirely
    by OpenRouter), this class makes the API call directly from the bot using
    the existing AsyncOpenAI client, with preconfigured generation parameters
    (model, image_size) while allowing the LLM to pass prompt and aspect_ratio.

    The model and image_size are configured server-side via settings commands.
    The prompt and aspect_ratio are provided by the LLM in the function call.
    """

    schema = ToolCallSchema(
        function=Function(
            name="generate_image",
            description=(
                "Generates an AI image using a configured image generation model. "
                "Use this when the user asks you to create, draw, or generate any kind of picture, "
                "illustration, artwork, or visual representation."
            ),
            parameters=Parameters(
                properties={
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed description of the image to generate. "
                            "Include subject, style, colors, mood, and any other visual details."
                        ),
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": [
                            "1:1", "2:3", "3:2", "3:4", "4:3",
                            "4:5", "5:4", "9:16", "16:9", "21:9",
                        ],
                        "description": (
                            "The desired aspect ratio for the generated image. "
                            "Use '16:9' for widescreen, '1:1' for square, '9:16' for portrait mobile, etc."
                        ),
                    },
                },
                required=["prompt"],
            ),
        )
    )
    function_name = schema.function.name

    def __init__(self, config: Config, ctx: commands.Context):
        super().__init__(config, ctx)
        self._cog = None

    @property
    def cog(self):
        """Lazy-load the AIUser cog to access the OpenAI client."""
        if self._cog is None:
            self._cog = self.bot.get_cog("AIUser")
        return self._cog

    async def _get_generation_params(self) -> DirectImageGenerationParameters:
        """Read the preconfigured generation parameters from guild config."""
        params_json = await self.config.guild(
            self.ctx.guild
        ).direct_image_generation_parameters()
        return deserialize_parameters(params_json, "direct_image_generation")

    async def _handle(self, arguments: Dict[str, Any]) -> str:
        """Execute the image generation via direct OpenRouter API call.

        Reads preconfigured model + image_size from guild config, uses the
        LLM-provided prompt and aspect_ratio, calls the OpenRouter API, and
        sends the resulting image to Discord.

        Args:
            arguments: Dict with 'prompt' (required) and 'aspect_ratio' (optional).

        Returns:
            A confirmation string for the LLM describing what was generated.
        """
        prompt = arguments.get("prompt", "")
        aspect_ratio = arguments.get("aspect_ratio")

        if not prompt or not prompt.strip():
            return "Error: No prompt provided for image generation."

        # Read preconfigured parameters
        params = await self._get_generation_params()
        model = params.model
        image_size = params.image_size

        if not model:
            logger.error(
                f"Image generation failed for guild {self.ctx.guild.name}: "
                "no model configured. Use `[p]aiuser functions generate_image_config` to set one."
            )
            return (
                "Error: Image generation model is not configured. "
                "A server administrator needs to set a model using the bot's configuration commands."
            )

        # Get the OpenAI client from the AIUser cog
        client: Optional[AsyncOpenAI] = None
        if self.cog and hasattr(self.cog, "openai_client"):
            client = self.cog.openai_client

        if not client:
            logger.error(
                f"Image generation failed for guild {self.ctx.guild.name}: OpenAI client not available."
            )
            return "Error: OpenAI client is not available. The bot may not be properly configured."

        # Build the request payload
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        extra_body: Dict[str, Any] = {
            "modalities": ["image", "text"],
        }

        image_config: Dict[str, str] = {}
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_size:
            image_config["image_size"] = image_size
        if image_config:
            extra_body["image_config"] = image_config

        logger.info(
            f"[DirectImageGen] Generating image for guild {self.ctx.guild.name}: "
            f"model={model}, prompt={prompt[:100]}{'...' if len(prompt) > 100 else ''}, "
            f"aspect_ratio={aspect_ratio}, image_size={image_size}"
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                stream=False,
            )
        except Exception as e:
            logger.error(
                f"[DirectImageGen] API call failed for guild {self.ctx.guild.name}: {e}"
            )
            return f"Error: Image generation API call failed: {str(e)}"

        # Extract images from the response
        images = self._extract_images_from_response(response)

        if not images:
            logger.warning(
                f"[DirectImageGen] No images found in response for guild {self.ctx.guild.name}"
            )
            # Fallback: check if there's text content describing the issue
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                return (
                    f"The image generation model returned: {content[:500]}"
                )
            return "Error: No image was generated. The model may not support image generation or returned an empty response."

        # Send each image to Discord
        sent_count = 0
        for image_data_url in images:
            try:
                await self._send_image_to_discord(image_data_url, prompt)
                sent_count += 1
            except Exception as e:
                logger.error(
                    f"[DirectImageGen] Failed to send image to Discord: {e}"
                )

        if sent_count == 0:
            return (
                "Error: Image was generated but could not be sent to the Discord channel."
            )

        logger.info(
            f"[DirectImageGen] Successfully generated and sent {sent_count} image(s) "
            f"for guild {self.ctx.guild.name}"
        )

        return (
            f"Successfully generated {sent_count} image(s) with the prompt: {prompt[:200]}"
            f"{'...' if len(prompt) > 200 else ''}. "
            "The image(s) have been sent to the Discord channel."
        )

    def _extract_images_from_response(self, response) -> List[str]:
        """Extract base64 image data URLs from an OpenRouter ChatCompletion response.

        OpenRouter returns images in the `model_extra` dict under the key "images".
        Each image entry has the format:
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        Args:
            response: The ChatCompletion response object.

        Returns:
            List of base64 data URL strings (e.g., "data:image/png;base64,...").
        """
        images: List[str] = []

        if not response.choices:
            return images

        message = response.choices[0].message

        # Primary extraction: model_extra.images
        model_extra = getattr(message, "model_extra", None)
        if model_extra and isinstance(model_extra, dict):
            raw_images = model_extra.get("images") or model_extra.get("image")
            if raw_images and isinstance(raw_images, list):
                for img_entry in raw_images:
                    if isinstance(img_entry, dict):
                        # Format: {"type": "image_url", "image_url": {"url": "data:..."}}
                        image_url_dict = img_entry.get("image_url") or img_entry.get(
                            "imageUrl"
                        )
                        if isinstance(image_url_dict, dict):
                            url = image_url_dict.get("url")
                            if url and isinstance(url, str) and url.startswith("data:"):
                                images.append(url)
            elif raw_images and isinstance(raw_images, str):
                # Sometimes it might be a single image URL string
                if raw_images.startswith("data:"):
                    images.append(raw_images)

        # Secondary extraction: the message content may contain image URLs for some providers
        content = getattr(message, "content", None)
        if content and isinstance(content, str):
            # Look for data URLs in the content text
            data_url_pattern = re.compile(
                r"(data:image/[a-zA-Z]+;base64,[^\s\"'\]\)]+)"
            )
            for match in data_url_pattern.finditer(content):
                url = match.group(1)
                if url not in images:
                    images.append(url)

        return images

    async def _send_image_to_discord(
        self, image_data_url: str, prompt: str
    ) -> None:
        """Decode a base64 image data URL and send it as a Discord file attachment.

        Args:
            image_data_url: The base64 data URL of the image.
            prompt: The original prompt (used for logging).

        Raises:
            Exception: If sending to Discord fails.
        """
        # Parse the data URL
        # Format: data:image/{format};base64,{encoded_data}
        header_match = re.match(
            r"data:image/(?P<fmt>[a-zA-Z]+);base64,(?P<data>.+)",
            image_data_url,
        )
        if not header_match:
            logger.warning(
                "[DirectImageGen] Could not parse data URL header, trying to send as-is"
            )
            # Fallback: send as embed
            embed = discord.Embed(color=await self.ctx.embed_color())
            embed.set_image(url=image_data_url)
            await self.ctx.send(embed=embed)
            return

        fmt = header_match.group("fmt").lower()
        encoded_data = header_match.group("data")

        import base64

        try:
            image_bytes = base64.b64decode(encoded_data)
        except Exception as e:
            logger.error(f"[DirectImageGen] Failed to decode base64 image data: {e}")
            # Fallback: send as embed
            embed = discord.Embed(color=await self.ctx.embed_color())
            embed.set_image(url=image_data_url)
            await self.ctx.send(embed=embed)
            return

        # Map format to file extension
        ext_map = {
            "png": "png",
            "jpeg": "jpg",
            "jpg": "jpg",
            "gif": "gif",
            "webp": "webp",
            "bmp": "bmp",
            "avif": "avif",
        }
        ext = ext_map.get(fmt, "png")
        filename = f"generated_{self.ctx.message.id}.{ext}"

        file = discord.File(fp=io.BytesIO(image_bytes), filename=filename)
        await self.ctx.send(file=file)

        logger.info(
            f"[DirectImageGen] Sent generated image to #{self.ctx.channel.name} "
            f"as {filename} ({len(image_bytes)} bytes)"
        )