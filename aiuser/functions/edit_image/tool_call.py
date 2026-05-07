import asyncio
import base64
import hashlib
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
    DirectImageEditParameters,
    deserialize_parameters,
)

logger = logging.getLogger("red.bz_cogs.aiuser")

# Maximum number of reference images to extract from Discord messages
MAX_REFERENCE_IMAGES = 14

# Discord message link pattern
# Format: https://discord.com/channels/{guild_id}/{channel_id}/{message_id}
DISCORD_MESSAGE_LINK_PATTERN = re.compile(
    r"https?://(?:www\.)?discord(?:app)?\.com/channels/"
    r"(?P<guild>\d+)/(?P<channel>\d+)/(?P<message>\d+)"
)

IMAGE_EXTENSIONS = frozenset({
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "avif",
})

# Maximum image file size for reference image downloads (10 MB)
MAX_IMAGE_DOWNLOAD_SIZE = 10 * 1024 * 1024


class EditImageToolCall(ToolCall):
    """Local ToolCall that edits existing images via direct OpenRouter API call.

    Similar to GenerateImageToolCall but takes a dedicated `image_to_edit` field
    — a single Discord message ID or link pointing to the image that should be
    modified. The prompt is framed as edit instructions, encouraging the LLM to
    describe *changes to make* rather than generating a new image from scratch.

    The model and image_size are configured server-side via settings commands.
    The prompt, image_to_edit, and optionally aspect_ratio are provided by the
    LLM in the function call.

    Edited image data is stored in `self.generated_images` and is retrieved
    by the pipeline to be sent to Discord alongside the response text.
    """

    schema = ToolCallSchema(
        function=Function(
            name="edit_image",
            description=(
                "Edits an existing image by modifying it based on natural language instructions. "
                "Use this when the user asks you to modify, alter, adjust, change, or transform "
                "something in an image — NOT to create a brand new image from scratch. "
                "For entirely new images, use 'generate_image' instead. "
                "Describe what to change about the image, not what the entire image should look like."
            ),
            parameters=Parameters(
                properties={
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed description of the specific edits to apply to the image. "
                            "Describe what to change or modify — for example "
                            "'make the sky more colorful', 'add a dog in the background', "
                            "'change the lighting to sunset', 'remove the text overlay'. "
                            "Do NOT describe the entire image from scratch."
                        ),
                    },
                    "image_to_edit": {
                        "type": "string",
                        "description": (
                            "A Discord message ID or message link "
                            "(e.g. https://discord.com/channels/guild_id/channel_id/message_id) "
                            "containing the image to edit. This is the primary image that will be modified. "
                            "Provide the message ID or link of the message that contains the image to edit."
                        ),
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": [
                            "1:1", "2:3", "3:2", "3:4", "4:3",
                            "4:5", "5:4", "9:16", "16:9", "21:9",
                            "1:4", "4:1", "1:8", "8:1"
                        ],
                        "description": (
                            "The desired aspect ratio for the edited image. "
                            "Use '16:9' for widescreen, '1:1' for square, '9:16' for portrait mobile, etc. "
                            "If not specified, the original image aspect ratio will be preserved."
                        ),
                    },
                    "reference_messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Discord message IDs or message links "
                            "(e.g. https://discord.com/channels/guild_id/channel_id/message_id) "
                            "containing additional images to use as style or context references for the edit. "
                            "Up to 13 images total will be extracted from the referenced messages. "
                            "(The primary image_to_edit counts toward the 14 image limit.)"
                        ),
                    },
                },
                required=["prompt", "image_to_edit"],
            ),
        )
    )
    function_name = schema.function.name

    def __init__(self, config: Config, ctx: commands.Context):
        super().__init__(config, ctx)
        self._cog = None
        self.generated_images: List[Dict] = []

    @property
    def cog(self):
        """Lazy-load the AIUser cog to access the OpenAI client."""
        if self._cog is None:
            self._cog = self.bot.get_cog("AIUser")
        return self._cog

    def get_generated_images(self) -> List[Dict]:
        """Return stored image data and clear the internal list.

        Each entry has:
            - bytes: Raw decoded image bytes.
            - format: Image format string (e.g., "png", "jpeg").
            - filename: Suggested filename for Discord attachment.
            - data_url: The original base64 data URL (for LLM context injection).
        """
        images = self.generated_images[:]
        self.generated_images.clear()
        return images

    async def _get_edit_params(self) -> DirectImageEditParameters:
        """Read the preconfigured edit parameters from guild config."""
        params_json = await self.config.guild(
            self.ctx.guild
        ).direct_image_edit_parameters()
        return deserialize_parameters(params_json, "direct_image_edit")

    async def _resolve_single_reference(
        self, ref: str
    ) -> Optional[str]:
        """Resolve a single Discord message ID/link to an image URL.

        Args:
            ref: A message ID or full Discord message link.

        Returns:
            The direct image URL, or None if it could not be resolved.
        """
        ref = ref.strip()
        if not ref:
            return None

        channel = None
        message_id = None

        link_match = DISCORD_MESSAGE_LINK_PATTERN.search(ref)
        if link_match:
            guild_id = int(link_match.group("guild"))
            channel_id = int(link_match.group("channel"))
            message_id = int(link_match.group("message"))
            guild = self.bot.get_guild(guild_id)
            if guild:
                channel = guild.get_channel(channel_id)
            if channel is None:
                channel = self.bot.get_channel(channel_id)
        else:
            try:
                message_id = int(ref)
                channel = self.ctx.channel
            except (ValueError, TypeError):
                logger.warning(
                    f"[EditImage] Invalid reference message format: {ref}"
                )
                return None

        if channel is None:
            logger.warning(
                f"[EditImage] Could not resolve channel for reference: {ref}"
            )
            return None

        try:
            message = await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
            logger.warning(
                f"[EditImage] Failed to fetch message {message_id}: {e}"
            )
            return None

        if not message:
            return None

        # Extract the first image URL from attachments
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                return attachment.url

        # Fallback to embed images
        for embed in message.embeds:
            if embed.image and embed.image.url:
                return embed.image.url
            if embed.thumbnail and embed.thumbnail.url:
                return embed.thumbnail.url

        logger.warning(
            f"[EditImage] No images found in message {message_id}"
        )
        return None

    async def _resolve_reference_images(
        self, reference_messages: List[str]
    ) -> List[str]:
        """Resolve Discord message IDs/links to a list of image URLs.

        Each entry can be:
        - A raw numeric Discord message ID (searched in the current channel)
        - A full Discord message link (https://discord.com/channels/...)

        Extracts image URLs from message attachments and embed images,
        up to MAX_REFERENCE_IMAGES total.

        Args:
            reference_messages: List of message IDs or message links.

        Returns:
            List of direct image URLs (up to 14).
        """
        image_urls: List[str] = []
        seen: set = set()

        for ref in reference_messages:
            if len(image_urls) >= MAX_REFERENCE_IMAGES:
                break

            ref = ref.strip()
            if not ref:
                continue

            # Determine channel and message ID
            channel = None
            message_id = None

            link_match = DISCORD_MESSAGE_LINK_PATTERN.search(ref)
            if link_match:
                # Full Discord message link
                guild_id = int(link_match.group("guild"))
                channel_id = int(link_match.group("channel"))
                message_id = int(link_match.group("message"))
                guild = self.bot.get_guild(guild_id)
                if guild:
                    channel = guild.get_channel(channel_id)
                if channel is None:
                    channel = self.bot.get_channel(channel_id)
            else:
                # Try as a raw message ID in the current channel
                try:
                    message_id = int(ref)
                    channel = self.ctx.channel
                except (ValueError, TypeError):
                    logger.warning(
                        f"[EditImage] Invalid reference message format: {ref}"
                    )
                    continue

            if channel is None:
                logger.warning(
                    f"[EditImage] Could not resolve channel for reference: {ref}"
                )
                continue

            # Fetch the message
            try:
                message = await channel.fetch_message(message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
                logger.warning(
                    f"[EditImage] Failed to fetch message {message_id}: {e}"
                )
                continue

            if not message:
                continue

            # Extract image URLs from attachments
            for attachment in message.attachments:
                if len(image_urls) >= MAX_REFERENCE_IMAGES:
                    break
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    url = attachment.url
                    if url not in seen:
                        seen.add(url)
                        image_urls.append(url)
                        logger.debug(
                            f"[EditImage] Found reference image from attachment: {url}"
                        )

            if len(image_urls) >= MAX_REFERENCE_IMAGES:
                break

            # Extract image URLs from embeds
            for embed in message.embeds:
                if len(image_urls) >= MAX_REFERENCE_IMAGES:
                    break
                # Embed image
                if embed.image and embed.image.url:
                    url = embed.image.url
                    if url not in seen:
                        seen.add(url)
                        image_urls.append(url)
                        logger.debug(
                            f"[EditImage] Found reference image from embed.image: {url}"
                        )
                # Embed thumbnail
                if embed.thumbnail and embed.thumbnail.url:
                    url = embed.thumbnail.url
                    if url not in seen:
                        seen.add(url)
                        image_urls.append(url)
                        logger.debug(
                            f"[EditImage] Found reference image from embed.thumbnail: {url}"
                        )

        logger.info(
            f"[EditImage] Resolved {len(image_urls)} reference image(s) "
            f"from {len(reference_messages)} message reference(s)"
        )
        return image_urls

    async def _download_and_encode_images(self, image_urls: List[str]) -> List[dict]:
        """Download reference images and encode them as base64 data URLs.

        Args:
            image_urls: List of image URLs to download.

        Returns:
            List of image_url content part dicts for the multimodal messages array.
        """
        content_parts: List[dict] = []

        for url in image_urls:
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        url, headers={"User-Agent": "bz-cogs/aiuser"}
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"[EditImage] Failed to download reference image "
                                f"from {url}: HTTP {response.status}"
                            )
                            continue

                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            logger.warning(
                                f"[EditImage] URL {url} is not an image "
                                f"(Content-Type: {content_type})"
                            )
                            continue

                        data = await response.read()
                        if len(data) > MAX_IMAGE_DOWNLOAD_SIZE:
                            logger.warning(
                                f"[EditImage] Reference image from {url} exceeds "
                                f"max size ({len(data)} > {MAX_IMAGE_DOWNLOAD_SIZE} bytes)"
                            )
                            continue

                        encoded = base64.b64encode(data).decode("utf-8")
                        data_url = f"data:{content_type};base64,{encoded}"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        })
                        logger.info(
                            f"[EditImage] Downloaded and encoded reference image "
                            f"from {url}: {len(data)} bytes"
                        )
            except aiohttp.ClientError as e:
                logger.warning(
                    f"[EditImage] Network error downloading reference image "
                    f"from {url}: {e}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[EditImage] Timeout downloading reference image from {url}"
                )
            except Exception as e:
                logger.warning(
                    f"[EditImage] Unexpected error downloading reference image "
                    f"from {url}: {e}"
                )

        return content_parts

    async def _handle(self, arguments: Dict[str, Any]) -> str:
        """Execute the image edit via direct OpenRouter API call.

        Reads preconfigured model + image_size from guild config, uses the
        LLM-provided prompt, image_to_edit, and aspect_ratio, resolves the
        target image and optional reference images from Discord messages,
        calls the OpenRouter API, and stores the resulting image data for
        later sending to Discord.

        Args:
            arguments: Dict with 'prompt' (required), 'image_to_edit' (required),
                       'aspect_ratio' (optional), and 'reference_messages' (optional).

        Returns:
            A confirmation string for the LLM describing what was edited.
        """
        prompt = arguments.get("prompt", "")
        image_to_edit = arguments.get("image_to_edit", "")
        aspect_ratio = arguments.get("aspect_ratio")
        reference_messages = arguments.get("reference_messages", [])

        if not prompt or not prompt.strip():
            return "Error: No prompt provided for image edit."

        if not image_to_edit or not image_to_edit.strip():
            return "Error: No image_to_edit provided. Provide a Discord message ID or link containing the image to edit."

        # Read preconfigured parameters
        params = await self._get_edit_params()
        model = params.model
        image_size = params.image_size

        if not model:
            logger.error(
                f"Image edit failed for guild {self.ctx.guild.name}: "
                "no model configured. Use `[p]aiuser functions edit_image_config` to set one."
            )
            return (
                "Error: Image edit model is not configured. "
                "A server administrator needs to set a model using the bot's configuration commands."
            )

        # Get the OpenAI client from the AIUser cog
        client: Optional[AsyncOpenAI] = None
        if self.cog and hasattr(self.cog, "openai_client"):
            client = self.cog.openai_client

        if not client:
            logger.error(
                f"Image edit failed for guild {self.ctx.guild.name}: OpenAI client not available."
            )
            return "Error: OpenAI client is not available. The bot may not be properly configured."

        # ---- Resolve the primary image to edit ----
        source_image_url = await self._resolve_single_reference(image_to_edit)
        if not source_image_url:
            return (
                "Error: Could not find an image in the referenced message. "
                "Make sure the message ID or link contains an image attachment."
            )

        # ---- Resolve and download the source image ----
        source_content_parts = await self._download_and_encode_images([source_image_url])
        if not source_content_parts:
            return (
                "Error: Could not download the image to edit. "
                "The image may be too large or unavailable."
            )

        # ---- Resolve and download additional reference images ----
        reference_content_parts: List[dict] = []

        if reference_messages and isinstance(reference_messages, list):
            reference_image_urls = await self._resolve_reference_images(reference_messages)
            if reference_image_urls:
                reference_content_parts = await self._download_and_encode_images(reference_image_urls)
                logger.info(
                    f"[EditImage] Using {len(reference_content_parts)} additional reference image(s) "
                    f"for edit context"
                )

        # ---- Build the request payload ----
        # Multimodal content: edit prompt + source image + optional reference images
        content: Any = [
            {"type": "text", "text": prompt},
        ]
        content.extend(source_content_parts)
        if reference_content_parts:
            content.extend(reference_content_parts)

        messages = [
            {
                "role": "user",
                "content": content,
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

        # Compute user identifier (same logic as LLMPipeline.call_client)
        user = f"{self.ctx.me.id}-{self.ctx.channel.id}"
        m = hashlib.sha256()
        m.update(user.encode('utf-8'))
        user_digest = m.hexdigest()

        # Inject safety settings and user identifier (same as LLMPipeline.call_client)
        extra_body.update({
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ],
            "user": user_digest,
        })

        logger.info(
            f"[EditImage] Editing image for guild {self.ctx.guild.name}: "
            f"model={model}, prompt={prompt[:1000]}{'...' if len(prompt) > 1000 else ''}, "
            f"aspect_ratio={aspect_ratio}, image_size={image_size}, "
            f"source_image={source_image_url[:80]}..., "
            f"reference_images={len(reference_content_parts)}"
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                tools=[{"google_search": {}}],
                user=user_digest,
                stream=False,
            )
        except Exception as e:
            logger.error(
                f"[EditImage] API call failed for guild {self.ctx.guild.name}: {e}"
            )
            return f"Error: Image edit API call failed: {str(e)}"

        # Extract images from the response
        images = self._extract_images_from_response(response)

        if not images:
            logger.warning(
                f"[EditImage] No images found in response for guild {self.ctx.guild.name}"
            )
            # Fallback: check if there's text content describing the issue
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                return (
                    f"The image edit model returned: {content[:500]}"
                )
            return "Error: No image was generated after editing. The model may not support image editing or returned an empty response."

        # Decode and store images for later sending by the response layer
        for image_data_url in images:
            try:
                decoded = self._decode_image_data(image_data_url)
                if decoded:
                    self.generated_images.append(decoded)
            except Exception as e:
                logger.error(
                    f"[EditImage] Failed to process image data: {e}"
                )

        if not self.generated_images:
            return (
                "Error: Image was edited but could not be processed."
            )

        logger.info(
            f"[EditImage] Successfully edited image, generated {len(self.generated_images)} result(s) "
            f"for guild {self.ctx.guild.name}"
        )

        model_info = model.split("/")[-1] if "/" in model else model
        ratio_info = aspect_ratio or "original"
        ref_info = f" with {len(reference_content_parts)} reference image(s)" if reference_content_parts else ""
        return (
            f"Image edit successful. Generated {len(self.generated_images)} edited version(s) "
            f"using model '{model_info}' at aspect ratio '{ratio_info}'{ref_info}. "
            f"Edit prompt used: \"{prompt[:200]}{'...' if len(prompt) > 200 else ''}\". "
            f"The edited image(s) have been sent to the Discord channel."
        )

    def _decode_image_data(self, image_data_url: str) -> Optional[Dict]:
        """Decode a base64 image data URL into image bytes and metadata.

        Args:
            image_data_url: The base64 data URL of the image.
                           Format: data:image/{format};base64,{encoded_data}

        Returns:
            Dict with 'bytes', 'format', 'filename', and 'data_url', or None if parsing fails.
        """

        # Format: data:image/{format};base64,{encoded_data}
        header_match = re.match(
            r"data:image/(?P<fmt>[a-zA-Z]+);base64,(?P<data>.+)",
            image_data_url,
        )
        if not header_match:
            logger.warning(
                "[EditImage] Could not parse data URL header"
            )
            return None

        fmt = header_match.group("fmt").lower()
        encoded_data = header_match.group("data")

        try:
            image_bytes = base64.b64decode(encoded_data)
        except Exception as e:
            logger.error(f"[EditImage] Failed to decode base64 image data: {e}")
            return None

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
        filename = f"edited_{self.ctx.message.id}.{ext}"

        return {
            "bytes": image_bytes,
            "format": fmt,
            "filename": filename,
            "data_url": image_data_url,
        }

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