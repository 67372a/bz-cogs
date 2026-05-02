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
    
    # scans only if the msg is the trigger, or if the msg was replied to by the trigger
    async def handle_attachment(self, message: Message, res, role):
        if message.attachments[0].content_type.startswith('image/'):
            logger.info(f"Supported image. type=[{message.attachments[0].content_type}] filename=[{message.attachments[0].filename}]")
            if (((self.init_msg.id == message.id) or (self.init_msg.reference and self.init_msg.reference.message_id == message.id)) 
                and not self.ctx.interaction and await self.config.guild(message.guild).scan_images() and 
                (message.attachments[0].size <= await self.config.guild(message.guild).max_image_size())):
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
            content = f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.display_name}"] sent an attachment with filename "{message.attachments[0].filename}"{title}{description}'
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
