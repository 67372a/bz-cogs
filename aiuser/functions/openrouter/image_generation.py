import asyncio
import io
import logging
import re
from typing import Dict, List, Optional

import aiohttp
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
    async def _extract_urls_from_text(text: str) -> List[str]:
        """Extract image URLs from text using multiple regex strategies.

        Strategy 1: Match URLs with known image extensions (.png, .jpg, etc.)
        Strategy 2: Match generic HTTP/HTTPS URLs that might point to CDN images
                    (no extension required, common with OpenRouter CDN URLs)

        Args:
            text: The text to scan for URLs.

        Returns:
            Deduplicated list of discovered image URLs.
        """
        urls: List[str] = []
        seen: set = set()

        # Strategy 1: URLs with image extensions using the existing image_parsing utility
        extended_image_pattern = re.compile(
            r"(https?://[^\s<>\"'\]\)]+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|svg|avif))"
            r"(\?[^\s<>\"'\]\)]*)?(?=[\s<>\"'\]\)]|$)",
            re.IGNORECASE,
        )
        for match in extended_image_pattern.finditer(text):
            full_url = match.group(1)
            if match.group(2):
                full_url += match.group(2)
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)
                logger.debug(f"Extracted image URL (strategy 1): {full_url}")

        # Strategy 2: Generic HTTP URLs that might be CDN image URLs
        # This catches URLs without file extensions (common with OpenRouter CDN)
        generic_url_pattern = re.compile(
            r"(https?://[^\s<>\"'\]\)]+/(?:cdn|storage|assets|media|images|img|generated|output)"
            r"/[^\s<>\"'\]\)]+)",
            re.IGNORECASE,
        )
        for match in generic_url_pattern.finditer(text):
            full_url = match.group(1)
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)
                logger.debug(f"Extracted potential CDN image URL (strategy 2): {full_url}")

        return urls

    @staticmethod
    async def _download_image(url: str) -> Optional[bytes]:
        """Download image data from a URL, using a global TTL cache.

        Images are cached by SHA-256 of the URL for up to 30 minutes.
        Uses aiohttp with a 30-second timeout and 10MB size limit.

        Args:
            url: The URL to download from.

        Returns:
            Raw image bytes, or None if download failed.
        """
        from aiuser.utils.image_cache import image_cache

        # Check cache first
        cached = image_cache.get(url)
        if cached is not None:
            logger.info(f"[ImageGen] Cache hit for image {url}")
            return cached

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "bz-cogs/aiuser"}) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Image download failed from {url}: HTTP {response.status}"
                        )
                        return None

                    # Check content type
                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"URL {url} has non-image Content-Type: {content_type}"
                        )

                    data = await response.read()
                    max_size = 10 * 1024 * 1024  # 10MB
                    if len(data) > max_size:
                        logger.warning(
                            f"Image from {url} exceeds max size ({len(data)} > {max_size} bytes)"
                        )
                        return None

                    logger.info(
                        f"Downloaded image from {url}: {len(data)} bytes, Content-Type: {content_type}"
                    )

                    # Store in cache
                    image_cache.set(url, data, content_type)
                    return data

        except aiohttp.ClientError as e:
            logger.warning(f"Network error downloading image from {url}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading image from {url}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error downloading image from {url}: {e}")
            return None

    @staticmethod
    async def _send_image_as_attachment(ctx: commands.Context, image_data: bytes, url: str):
        """Send image data as a Discord file attachment.

        Args:
            ctx: The command context for sending.
            image_data: Raw bytes of the image.
            url: The original source URL (used for logging only).
        """
        try:
            # Determine filename extension from the URL or default to png
            ext_match = re.search(r'\.(png|jpg|jpeg|gif|webp|bmp|avif)(?:\?|#|$)', url, re.IGNORECASE)
            ext = ext_match.group(1).lower() if ext_match else "png"
            if ext in ("jpg", "jpeg"):
                ext = "jpg"

            filename = f"generated_{ctx.message.id}.{ext}"
            file = discord.File(fp=io.BytesIO(image_data), filename=filename)
            await ctx.send(file=file)
            logger.info(
                f"Sent OpenRouter generated image to #{ctx.channel.name} "
                f"(from {url}) as attachment: {filename} ({len(image_data)} bytes)"
            )
        except discord.HTTPException as e:
            logger.error(f"Failed to send image attachment to Discord: {e}")
            # Fall back to embed if attachment fails
            try:
                embed = discord.Embed(color=await ctx.embed_color())
                embed.set_image(url=url)
                await ctx.send(
                    embed=embed,
                    content="⚠️ Image too large for attachment, sent as embed link instead."
                )
                logger.info(f"Sent image as embed fallback: {url}")
            except Exception as e2:
                logger.error(f"Failed to send image embed fallback: {e2}")

    @staticmethod
    async def handle_tool_response_content(
        content: str,
        ctx: commands.Context,
        model_extra: Optional[Dict] = None,
    ):
        """Parse the model's response for image data and send to Discord.

        Strategy (in order):
        1. Check model_extra metadata for image URL/data from OpenRouter
        2. Scan the response text for image URLs
        3. Download the image and send as Discord file attachment
        4. Fall back to sending as embed if download fails

        When OpenRouter processes an image_generation tool call, it returns the
        result to the model. The model then writes about the image in its text
        response. The image URL may appear in model_extra metadata or in the
        visible text response.

        Args:
            content: The text response from the model after tool execution.
            ctx: The command context for sending Discord messages.
            model_extra: Optional dict of extra metadata from the LLM response
                (may contain structured image generation data from OpenRouter).
        """
        # ---- DIAGNOSTIC LOGGING ----
        logger.info(
            f"[ImageGen] Processing image_generation response for guild {ctx.guild.name}"
        )
        if content:
            logger.info(
                f"[ImageGen] Response text preview: {content[:500]}"
                f"{'...' if len(content) > 500 else ''}"
            )
        else:
            logger.warning("[ImageGen] Response text is None or empty")

        if model_extra:
            logger.info(f"[ImageGen] model_extra keys: {list(model_extra.keys())}")
            for key, value in model_extra.items():
                value_preview = str(value)[:300]
                logger.info(f"[ImageGen] model_extra['{key}'] = {value_preview}")
        else:
            logger.info("[ImageGen] model_extra is None - no metadata from OpenRouter")

        # ---- CHECK model_extra FOR STRUCTURED IMAGE DATA ----
        image_urls_found: List[str] = []

        if model_extra:
            # Common keys OpenRouter might use for image results
            extra_url = None
            if isinstance(model_extra, dict):
                extra_url = model_extra.get("imageUrl") or model_extra.get("image_url") or model_extra.get("url")
                if extra_url and isinstance(extra_url, str):
                    image_urls_found.append(extra_url)
                    logger.info(f"[ImageGen] Found image URL in model_extra: {extra_url}")

                # Check for nested structures
                if not extra_url:
                    for key in ("data", "result", "response", "generated"):
                        nested = model_extra.get(key)
                        if isinstance(nested, dict):
                            nested_url = nested.get("imageUrl") or nested.get("image_url") or nested.get("url")
                            if nested_url and isinstance(nested_url, str):
                                image_urls_found.append(nested_url)
                                logger.info(
                                    f"[ImageGen] Found image URL in model_extra.{key}: {nested_url}"
                                )

        # ---- SCAN RESPONSE TEXT FOR IMAGE URLS ----
        text_urls = await OpenRouterImageGeneration._extract_urls_from_text(content or "")
        if text_urls:
            logger.info(f"[ImageGen] Found {len(text_urls)} image URL(s) in response text: {text_urls}")
            for url in text_urls:
                if url not in image_urls_found:
                    image_urls_found.append(url)

        if not image_urls_found:
            logger.warning(
                "[ImageGen] No image URLs found in model_extra or response text. "
                "The model may have described the image without including a URL, "
                "or OpenRouter may not have returned the image URL."
            )
            return

        logger.info(f"[ImageGen] Total unique image URLs to process: {len(image_urls_found)}")

        # ---- DOWNLOAD AND SEND EACH IMAGE ----
        for url in image_urls_found:
            logger.info(f"[ImageGen] Attempting to download image from: {url}")

            image_data = await OpenRouterImageGeneration._download_image(url)
            if image_data:
                logger.info(
                    f"[ImageGen] Successfully downloaded image ({len(image_data)} bytes)"
                )
                await OpenRouterImageGeneration._send_image_as_attachment(ctx, image_data, url)
            else:
                logger.warning(f"[ImageGen] Download failed for {url}, falling back to embed")
                # Fall back to embed
                try:
                    embed = discord.Embed(color=await ctx.embed_color())
                    embed.set_image(url=url)
                    await ctx.send(embed=embed)
                    logger.info(f"[ImageGen] Sent image embed fallback for {url}")
                except Exception as e:
                    logger.error(f"[ImageGen] Failed to send embed fallback for {url}: {e}")
