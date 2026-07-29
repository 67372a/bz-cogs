import logging
import re

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
                                                    format_text_document,
                                                    _get_msg_header)
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

        # Check for image URLs in message text (before attachment handling).
        # Skipped when the message also has attachments so they aren't dropped.
        if message.content and not message.attachments:
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
        max_pixels = await self.config.guild(message.guild).max_image_pixels()
        if max_pixels is None:
            max_pixels = 16_777_216  # 4096*4096 default for LLM mode
        content_parts, sources = await OpenRouterImageParsing.process_message_for_images(
            message.content, max_size, max_pixels
        )

        if not content_parts:
            return False

        logger.info(
            f"Found {len(content_parts)} image URL(s) in message {message.id}: {', '.join(sources)}"
        )

        # Add text content after images if there is meaningful text,
        # otherwise emit the XML header so author/reply metadata is kept.
        text_content = format_text_content(message)
        if text_content:
            content_parts.append({"type": "text", "text": text_content})
        elif message.author.id != message.guild.me.id:
            content_parts.append({"type": "text", "text": f'{_get_msg_header(message)}</message>'})

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
            for attachment in image_attachments:
                # Marker so the model can associate each image with its filename
                marker = {"type": "text", "text": format_generic_image_single(attachment)}

                if attachment.size > await self.config.guild(message.guild).max_image_size():
                    logger.info(f"Image {attachment.filename} exceeds max size, skipping transcription")
                    all_content_parts.append(marker)
                    continue

                logger.info(
                    f"Supported image. type=[{attachment.content_type}] filename=[{attachment.filename}]"
                )
                content = await transcribe_image_single(self.cog, message, attachment)
                all_content_parts.append(marker)
                if content:
                    if isinstance(content, list):
                        # Strip any embedded text parts from the transcription;
                        # the message-level text/header is appended once below.
                        all_content_parts.extend(
                            part for part in content
                            if not (isinstance(part, dict) and part.get("type") == "text")
                        )
                    else:
                        all_content_parts.append({"type": "text", "text": content})

            # Emit markers for non-image attachments so they aren't invisible
            for att in message.attachments:
                if not (att.content_type and att.content_type.startswith('image/')):
                    all_content_parts.append({"type": "text", "text": format_attachment_marker(att)})

            if all_content_parts:
                # Add the message text content or metadata header once, at the end
                if message.content and message.content.strip():
                    all_content_parts.append({"type": "text", "text": format_text_content(message)})
                elif message.author.id != message.guild.me.id:
                    all_content_parts.append({"type": "text", "text": f'{_get_msg_header(message)}</message>'})

                # Preserve any embed text (e.g. another bot's image caption)
                embed_content = await self._format_embed_content(message)
                if embed_content:
                    all_content_parts.append({"type": "text", "text": embed_content})

                entry = MessageEntry(role, all_content_parts)
                res.append(entry)

                # Cache the result
                self.message_cache[cache_key] = all_content_parts
                return

        # Fall through to single-attachment logic for non-image attachments.
        # content_type can be None for some attachments; treat as empty string.
        first_content_type = message.attachments[0].content_type or ""
        if first_content_type.startswith('image/'):
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

        elif any(first_content_type.startswith(content_type) for content_type in SUPPORTED_TEXT_DOCUMENT_CONTENT_TYPES):
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

        elif any(first_content_type.startswith(content_type) for content_type in SUPPORTED_BINARY_DOCUMENT_CONTENT_TYPES):
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
            content = format_generic_attachment(message)
            await self.add_entry(content, res, role)

        # Emit markers for any additional attachments beyond the first so
        # they are not silently dropped from context.
        extra = format_extra_attachments(message)
        if extra:
            await self.add_entry(extra, res, role)

        # Preserve any embed text — convert() prioritizes attachments over
        # embeds, so a message with both (e.g. another bot's image response
        # carrying its caption in an embed) would otherwise lose the embed.
        embed_content = await self._format_embed_content(message)
        await self.add_entry(embed_content, res, role)

        content = format_text_content(message)
        await self.add_entry(content, res, role)

    async def _format_embed_content(self, message: Message):
        """Return formatted embed content for a message, or None.

        Used by handle_attachment() so messages with both attachments and
        embeds don't silently lose the embed text.
        """
        if not message.embeds or not is_embed_valid(message):
            return None
        if self._is_own_response_embed(message):
            return await format_bot_embed_content(self.cog, message)
        return await format_embed_content(self.cog, message)

    def _is_own_response_embed(self, message: Message) -> bool:
        """True only for THIS bot's own "X's Response" embeds.

        The title regex alone would also match other bots' response embeds
        (e.g. another aiuser instance), which must keep their XML metadata.
        """
        if not message.embeds:
            return False
        return (
            message.author.id == self.bot_id
            and bool(RESPONSE_EMBED_TITLE_REGEX.search(message.embeds[0].title or ""))
        )

    async def handle_embed(self, message: Message, res, role):
        if self._is_own_response_embed(message):
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

        # Add text content after files if there is meaningful text,
        # otherwise emit the XML header so author/reply metadata is kept.
        text_content = format_text_content(message)
        if text_content:
            content_parts.append({"type": "text", "text": text_content})
        elif message.author.id != message.guild.me.id:
            content_parts.append({"type": "text", "text": f'{_get_msg_header(message)}</message>'})

        res.append(MessageEntry(role, content_parts))

    async def add_entry(self, content, res, role):
        if not content:
            return
        res.append(MessageEntry(role, content))


async def transcribe_image_single(cog: MixinMeta, message: Message, attachment):
    """Transcribe a single attachment (used for multiple image support).

    Uses the unified image processing pipeline with double-keyed caching
    (shared with ``transcribe_image`` in ``caption.py``).
    """
    from aiuser.messages_list.converter.image.caption import transcribe_single
    return await transcribe_single(cog, message, attachment)


_QUOTE_ESCAPES = {'"': "&" + "quot;"}


def format_attachment_marker(attachment) -> str:
    """Return the XML placeholder tag for a single attachment (image or file)."""
    from xml.sax.saxutils import escape
    title = f' title="{escape(attachment.title, _QUOTE_ESCAPES)}"' if attachment.title else ""
    desc = f' description="{escape(attachment.description, _QUOTE_ESCAPES)}"' if attachment.description else ""
    filename = escape(attachment.filename or "", _QUOTE_ESCAPES)
    tag = "image" if (attachment.content_type or "").startswith("image/") else "file"
    return f'<{tag} filename="{filename}"{title}{desc}/>'


def format_extra_attachments(message: Message):
    """Format attachments beyond the first as placeholder tags in the standard schema.

    Returns None when the message has one or zero attachments.
    """
    if len(message.attachments) <= 1:
        return None
    markers = " ".join(format_attachment_marker(a) for a in message.attachments[1:])
    if message.author.id == message.guild.me.id:
        return f"Sent {markers}"
    return f'{_get_msg_header(message)}{markers}</message>'


def format_generic_image_single(attachment) -> str:
    """Format a generic image description for a single attachment (used for multiple image support)."""
    from xml.sax.saxutils import escape
    title = f' title="{escape(attachment.title, _QUOTE_ESCAPES)}"' if attachment.title else ""
    desc = f' description="{escape(attachment.description, _QUOTE_ESCAPES)}"' if attachment.description else ""
    filename = escape(attachment.filename, _QUOTE_ESCAPES)
    return f'<image filename="{filename}"{title}{desc}/>'


def format_generic_attachment(message: Message) -> str:
    """Format an unsupported attachment using the standard XML schema."""
    from xml.sax.saxutils import escape
    attachment = message.attachments[0]
    title = f' title="{escape(attachment.title, _QUOTE_ESCAPES)}"' if attachment.title else ""
    desc = f' description="{escape(attachment.description, _QUOTE_ESCAPES)}"' if attachment.description else ""
    filename = escape(attachment.filename, _QUOTE_ESCAPES)
    xml_content = f'<file filename="{filename}"{title}{desc}/>'

    if message.author.id == message.guild.me.id:
        return f"Sent {xml_content}"

    return f'{_get_msg_header(message)}{xml_content}</message>'