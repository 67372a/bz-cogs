import logging
import re
from datetime import timezone

from discord import Message
from redbot.core import commands

from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import contains_youtube_link, is_embed_valid
from aiuser.messages_list.converter.embed.formatter import format_embed_content, format_bot_embed_content
from aiuser.messages_list.converter.helpers import (format_embed_text_content,
                                                    format_generic_image,
                                                    format_sticker_content,
                                                    format_text_content,
                                                    format_generic_document,
                                                    format_binary_document,
                                                    format_text_document)
from aiuser.messages_list.converter.image.caption import transcribe_image
from aiuser.messages_list.entry import MessageEntry

logger = logging.getLogger("red.bz_cogs.aiuser")

SUPPORTED_BINARY_DOCUMENT_CONTENT_TYPES = ["application/pdf"]

SUPPORTED_TEXT_DOCUMENT_CONTENT_TYPES = ["text/", "application/xml", "application/yaml", "application/json", "application/xhtml"]

SUPPORTED_VIDEO_CONTENT_TYPES = ["video/x-flv","video/quicktime","video/mpeg","video/mpegs","video/mpg","video/mp4","video/webm","video/wmv","video/gpp"]

SUPPORTED_AUDIO_CONTENT_TYPES = ["audio/x-aac","audio/flac","audio/mp3","audio/m4a","audio/mpeg","audio/mpga","audio/mp4","audio/opus","audio/pcm","audio/wav","audio/webm"]
RESPONSE_EMBED_TITLE_REGEX = re.compile(r'^.*\'s Response$')

class MessageConverter():
    def __init__(self, cog: MixinMeta, ctx: commands.Context):
        self.cog = cog
        self.config = cog.config
        self.bot_id = cog.bot.user.id
        self.init_msg = ctx.message
        self.message_cache = cog.cached_messages
        self.ctx = ctx

    async def convert(self, message: Message):
        """Converts a Discord message to ChatML format message(s)"""
        res = []
        role = "user" if message.author.id != self.bot_id else "assistant"

        # Check cache first - for messages that were already processed (reply chains)
        # Try channel-scoped key first, then fall back to plain message id
        cache_key = f"{message.channel.id}:{message.id}"
        if cache_key in self.message_cache:
            await self.add_entry(self.message_cache[cache_key], res, role)
            return res
        if message.id in self.message_cache:
            await self.add_entry(self.message_cache[message.id], res, role)
            return res

        # Check for image URLs in message text (before attachment handling)
        if message.content:
            handled_by_urls = await self.handle_image_urls(message, res, role)
            if handled_by_urls:
                return res

        # Check for PDF URLs in message text (before attachment handling)
        if not message.attachments and message.content:
            await self.handle_pdf_urls(message, res, role)
            if res:  # handle_pdf_urls fully handled the message
                return res

        if message.attachments:
            await self.handle_attachment(message, res, role)
        elif message.stickers:
            content = await format_sticker_content(message)
            await self.add_entry(content, res, role)
        elif (len(message.embeds) > 0 and is_embed_valid(message)):
            await self.handle_embed(message, res, role)
        elif contains_youtube_link(message.content):
            await self.handle_embed(message, res, role)
        else:
            content = format_text_content(message)
            await self.add_entry(content, res, role)

        return res or None

    async def _is_scan_enabled(self, message: Message) -> bool:
        """Check if scanning is enabled and the message is eligible for scanning.
        
        Returns True if the message should be scanned (trigger or replied-to), 
        scan_images is enabled, and it's not an interaction.
        """
        return (
            (self.init_msg.id == message.id or
             (self.init_msg.reference and self.init_msg.reference.message_id == message.id))
            and not self.ctx.interaction
            and await self.config.guild(message.guild).scan_images()
        )

    async def handle_image_urls(self, message: Message, res, role) -> bool:
        """Detect image URLs in message text, download and encode them.

        Returns True if image URLs were found and handled (meaning the message
        content was consumed and no further processing is needed), False otherwise.

        This is called before handle_pdf_urls so image URLs take priority.
        When URLs are found, the message content is treated as consumed.
        """
        if not await self._is_scan_enabled(message):
            return False

        from aiuser.functions.openrouter.image_parsing import OpenRouterImageParsing

        try:
            url_enabled = await self.config.guild(message.guild).openrouter_image_parsing_enabled()
            if not url_enabled:
                return False
        except Exception:
            return False

        max_size = await self.config.guild(message.guild).openrouter_image_parsing_max_size()
        content_parts, sources = await OpenRouterImageParsing.process_message_for_images(
            message.content, max_size
        )

        if not content_parts:
            return False

        logger.info(
            f"Found {len(content_parts)} image URL(s) in message {message.id}: {', '.join(sources)}"
        )

        # Add text content after images if there is meaningful text
        text_content = format_text_content(message)
        if text_content:
            content_parts.append({"type": "text", "text": text_content})

        res.append(MessageEntry(role, content_parts))
        return True

    # scans only if the msg is the trigger, or if the msg was replied to by the trigger
    async def handle_attachment(self, message: Message, res, role):
        # Check cache first for any existing cached content for this message
        cache_key = f"{message.channel.id}:{message.id}"
        if cache_key in self.message_cache:
            await self.add_entry(self.message_cache[cache_key], res, role)
            return
        if message.id in self.message_cache:
            await self.add_entry(self.message_cache[message.id], res, role)
            return

        # Process all image attachments first
        image_attachments = [
            att for att in message.attachments
            if att.content_type and att.content_type.startswith('image/')
        ]

        if image_attachments and await self._is_scan_enabled(message):
            # Process multiple images - build a list of content parts
            all_content_parts = []
            filenames = []
            for attachment in image_attachments:
                if attachment.size > await self.config.guild(message.guild).max_image_size():
                    logger.info(f"Image {attachment.filename} exceeds max size, skipping transcription")
                    all_content_parts.append(format_generic_image_single(attachment))
                    filenames.append(attachment.filename)
                    continue

                logger.info(
                    f"Supported image. type=[{attachment.content_type}] filename=[{attachment.filename}]"
                )
                # Create a temporary message with just this attachment for transcription
                content = await transcribe_image_single(self.cog, message, attachment)
                if content:
                    if isinstance(content, list):
                        all_content_parts.extend(content)
                    else:
                        all_content_parts.append({"type": "text", "text": content})
                else:
                    all_content_parts.append(
                        {"type": "text", "text": format_generic_image_single(attachment)}
                    )
                filenames.append(attachment.filename)

            if all_content_parts:
                # Add the message text content
                if message.content and message.content.strip():
                    all_content_parts.append({"type": "text", "text": format_text_content(message)})

                entry = MessageEntry(role, all_content_parts)
                res.append(entry)

                # Cache the result
                self.message_cache[cache_key] = all_content_parts
                return

        # Fall through to single-attachment logic for non-image attachments
        if message.attachments[0].content_type.startswith('image/'):
            if await self._is_scan_enabled(message) and (
                message.attachments[0].size <= await self.config.guild(message.guild).max_image_size()
            ):
                content = await transcribe_image(self.cog, message) or format_generic_image(message)
                await self.add_entry(content, res, role)
                if isinstance(content, list):
                    return
            else:
                content = format_generic_image(message)
                await self.add_entry(content, res, role)

        elif any(message.attachments[0].content_type.startswith(content_type) for content_type in SUPPORTED_TEXT_DOCUMENT_CONTENT_TYPES):
            if (((self.init_msg.id == message.id) or (self.init_msg.reference and self.init_msg.reference.message_id == message.id))
                and not self.ctx.interaction and await self.config.guild(message.guild).scan_images() and
                (message.attachments[0].size <= await self.config.guild(message.guild).max_image_size())):
                logger.info(f"Supported text document. type=[{message.attachments[0].content_type}] filename=[{message.attachments[0].filename}]")
                content = await format_text_document(message) or format_generic_document(message)

                await self.add_entry(content, res, role)
                if isinstance(content, list):
                    return
            else:
                content = format_generic_document(message)
                await self.add_entry(content, res, role)

        elif any(message.attachments[0].content_type.startswith(content_type) for content_type in SUPPORTED_BINARY_DOCUMENT_CONTENT_TYPES):
            if (((self.init_msg.id == message.id) or (self.init_msg.reference and self.init_msg.reference.message_id == message.id))
                and not self.ctx.interaction and await self.config.guild(message.guild).scan_images() and
                (message.attachments[0].size <= await self.config.guild(message.guild).max_image_size())):
                logger.info(f"Supported binary document. type=[{message.attachments[0].content_type}] filename=[{message.attachments[0].filename}]")
                content = await format_binary_document(message) or format_generic_document(message)

                await self.add_entry(content, res, role)
                if isinstance(content, list):
                    return
            else:
                content = format_generic_document(message)
                await self.add_entry(content, res, role)
        elif message.id in self.message_cache:
            await self.add_entry(self.message_cache[message.id], res, role)
        else:
            logger.info(f"Unsupported attachment content Type. type={message.attachments[0].content_type} filename={message.attachments[0].filename}")

            title = f', title "{message.attachments[0].title}"' if message.attachments[0].title else ""
            description = f', description "{message.attachments[0].description}"' if message.attachments[0].description else ""
            timestamp = message.created_at.astimezone(timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')
            content = f'[MESSAGE_ID={message.id} TIMESTAMP={timestamp} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.display_name}"] sent an attachment with filename "{message.attachments[0].filename}"{title}{description}'
            await self.add_entry(content, res, role)

        content = format_text_content(message)
        await self.add_entry(content, res, role)

    async def handle_embed(self, message: Message, res, role):
        if self.bot_id and RESPONSE_EMBED_TITLE_REGEX.search(message.embeds[0].title):
            content = await format_bot_embed_content(self.cog, message)
        else:
            content = await format_embed_content(self.cog, message)

        if not content:
            content = format_text_content(message)
            await self.add_entry(content, res, role)
        else:
            await self.add_entry(content, res, role)
            content = format_embed_text_content(message)
            await self.add_entry(content, res, role)

    async def handle_pdf_urls(self, message: Message, res, role):
        """Detect PDF URLs in message text, download and encode them as file content parts.

        This is a pre-processing step that happens before the main convert() logic.
        When PDF URLs are found and openrouter_pdf_parsing is enabled, the PDFs are
        downloaded, base64-encoded, and added as file content parts to the messages array.
        """
        from aiuser.functions.openrouter.pdf_parsing import OpenRouterPdfParsing

        try:
            pdf_enabled = await self.config.guild(message.guild).openrouter_pdf_parsing_enabled()
            if not pdf_enabled:
                return
        except Exception:
            return

        service = OpenRouterPdfParsing(self.config, self.ctx)

        # Only process the triggering message and messages it replies to
        is_trigger = (self.init_msg.id == message.id) or (
            self.init_msg.reference and self.init_msg.reference.message_id == message.id
        )
        if not is_trigger:
            return

        file_contents, filenames = await service.process_message_for_pdfs(message.content)

        if not file_contents:
            return

        logger.info(
            f"Found {len(file_contents)} PDF(s) in message {message.id}: {', '.join(filenames)}"
        )

        # Add file content parts as the first entry
        content_parts = file_contents.copy()

        # Add text content after files if there is meaningful text
        text_content = format_text_content(message)
        if text_content:
            content_parts.append({"type": "text", "text": text_content})

        res.append(MessageEntry(role, content_parts))

        # Mark that we've already added the text content so it's not duplicated
        self._handled_pdf_text = True

    async def add_entry(self, content, res, role):
        if not content:
            return
        res.append(MessageEntry(role, content))


async def transcribe_image_single(cog: MixinMeta, message: Message, attachment):
    """Transcribe a single attachment (used for multiple image support).

    This mirrors the logic in transcribe_image but operates on a specific
    attachment rather than message.attachments[0].
    """
    import base64
    from io import BytesIO
    import cv2
    import numpy as np
    from PIL import Image

    from aiuser.types.enums import ScanImageMode
    from aiuser.messages_list.converter.image.AI_horde import process_image_ai_horde

    config = cog.config
    mode = ScanImageMode(await config.guild(message.guild).scan_images_mode())

    buffer = BytesIO()
    await attachment.save(buffer)

    file_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_image is None:
        logger.error(f"Failed to decode image from attachment {attachment.filename} in message {message.id}")
        return None

    max_pixels = await config.guild(message.guild).max_image_pixels()

    if max_pixels is not None:
        maxsize = max_pixels
    elif mode == ScanImageMode.LLM:
        maxsize = 16777216  # 4096 * 4096
    else:
        maxsize = 1048576  # 1024 * 1024

    from aiuser.messages_list.converter.image.caption import scale_image
    scaled_cv_image = scale_image(cv_image, maxsize)

    if mode == ScanImageMode.AI_HORDE:
        image = Image.fromarray(cv2.cvtColor(scaled_cv_image, cv2.COLOR_BGR2RGB))
        return await process_image_ai_horde(cog, message, image)
    elif mode == ScanImageMode.LOCAL:
        try:
            from aiuser.messages_list.converter.image.local import process_image_locally
            image = Image.fromarray(cv2.cvtColor(scaled_cv_image, cv2.COLOR_BGR2RGB))
            return await process_image_locally(cog, message, image)
        except ImportError:
            logger.exception("Local image scanning dependencies not installed")
            return None
    elif mode == ScanImageMode.LLM:
        content = []
        webp_quality = 90
        params = [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
        success, encoded_image = cv2.imencode('.webp', scaled_cv_image, params)

        if not success:
            logger.error(f"Failed to encode image {attachment.filename} to WebP format")
            return None

        base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        content.append(
            {"type": "image_url", "image_url": {
             "url": f"data:image/webp;base64,{base64_image}"}
             })
        return content
    else:
        return None


def format_generic_image_single(attachment) -> str:
    """Format a generic image description for a single attachment (used for multiple image support)."""
    from xml.sax.saxutils import escape
    title = f' title="{escape(attachment.title)}"' if attachment.title else ""
    desc = f' description="{escape(attachment.description)}"' if attachment.description else ""
    filename = escape(attachment.filename)
    return f'<image filename="{filename}"{title}{desc}/>'