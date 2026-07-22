# response/response_handler.py
import logging

import discord
from redbot.core import commands

from aiuser.messages_list.messages import create_messages_list
from aiuser.response.chat.response import create_chat_response
from aiuser.response.image.generator_factory import get_image_generator
from aiuser.response.image.response import create_image_response
from aiuser.response.is_image_request import is_image_request
from aiuser.types.abc import MixinMeta

logger = logging.getLogger("red.bz_cogs.aiuser")


async def dispatch_response(cog: MixinMeta, ctx: commands.Context, messages_list=None):
    """ Decide which response to send based on the context """
    async with ctx.message.channel.typing():
        if (not messages_list and not ctx.interaction) and await is_image_request(cog, ctx.message):
            if await process_image_response(cog, ctx):
                return

        messages_list = messages_list or await create_messages_list(cog, ctx)
        return await create_chat_response(cog, ctx, messages_list)


async def process_image_response(cog: MixinMeta, ctx: commands.Context) -> bool:
    """Process and send an image response"""
    await ctx.react_quietly("🧐")

    try:
        generator = await get_image_generator(ctx, cog.config)
        success = await create_image_response(cog, ctx, generator)
        return success
    except Exception:
        logger.exception("Error generating image response")
        return False
    finally:
        # The message may have been deleted or perms changed while the image
        # was being generated; failing to remove the reaction must not mask
        # the actual result.
        try:
            await ctx.message.remove_reaction("🧐", ctx.me)
        except discord.HTTPException:
            logger.debug("Could not remove 🧐 reaction", exc_info=True)