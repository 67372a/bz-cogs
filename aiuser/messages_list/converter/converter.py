import logging

from discord import Message
from redbot.core import commands

from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import contains_youtube_link, is_embed_valid
from aiuser.messages_list.converter.embed.formatter import format_embed_content
from aiuser.messages_list.converter.helpers import (format_embed_text_content,
                                                    format_generic_image,
                                                    format_sticker_content,
                                                    format_text_content,
                                                    format_generic_document,
                                                    format_supported_document)
from aiuser.messages_list.converter.image.caption import transcribe_image
from aiuser.messages_list.entry import MessageEntry

logger = logging.getLogger("red.bz_cogs.aiuser")

SUPPORTED_DOCUMENT_CONTENT_TYPES = ["application/pdf","text/plain"]


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
        if message.attachments:
            await self.handle_attachment(message, res, role)
        elif message.stickers:
            content = await format_sticker_content(message)
            await self.add_entry(content, res, role)
        elif (len(message.embeds) > 0 and is_embed_valid(message)) or contains_youtube_link(message.content):
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

        elif any(message.attachments[0].content_type.startswith(content_type) for content_type in SUPPORTED_DOCUMENT_CONTENT_TYPES):
            if (((self.init_msg.id == message.id) or (self.init_msg.reference and self.init_msg.reference.message_id == message.id)) 
                and not self.ctx.interaction and await self.config.guild(message.guild).scan_images() and 
                (message.attachments[0].size <= await self.config.guild(message.guild).max_image_size())):
                logger.info(f"Supported document. type=[{message.attachments[0].content_type}] filename=[{message.attachments[0].filename}]")
                content = await format_supported_document(message) or format_generic_document(message)
                
                logger.info(f"Supported document content '{content}'.")
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
            content = f'User "{message.author.display_name}" sent: [Attachment: "{message.attachments[0].filename}"]'
            await self.add_entry(content, res, role)

        content = format_text_content(message)
        await self.add_entry(content, res, role)

    async def handle_embed(self, message: Message, res, role):
        content = await format_embed_content(self.cog, message)
        if not content:
            content = format_text_content(message)
            await self.add_entry(content, res, role)
        else:
            await self.add_entry(content, res, role)
            content = format_embed_text_content(message)
            await self.add_entry(content, res, role)

    async def add_entry(self, content, res, role):
        if not content:
            return
        res.append(MessageEntry(role, content))
