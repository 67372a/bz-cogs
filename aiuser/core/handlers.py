# handlers.py

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional

import discord
from redbot.core import commands

from aiuser.config.constants import URL_PATTERN
from aiuser.config.defaults import DEFAULT_REPLY_PERCENT
from aiuser.core.triggers import check_triggers
from aiuser.core.validators import is_valid_message
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import is_embed_valid

logger = logging.getLogger("red.bz_cogs.aiuser")


async def handle_slash_command(cog: MixinMeta, inter: discord.Interaction, text: str):
    """Handle /chat slash command interactions"""
    await inter.response.defer()

    ctx = await commands.Context.from_interaction(inter)
    ctx.message.content = text

    if not (await is_valid_message(cog, ctx)):
        return await ctx.send(
            "You're not allowed to use this command here.", ephemeral=True
        )
    elif await get_percentage(cog, ctx) == 1.0:
        pass
    elif not (await cog.config.guild(ctx.guild).reply_to_mentions_replies()):
        return await ctx.send("This command is not enabled.", ephemeral=True)

    rate_limit_reset = await get_ratelimit_reset(cog)
    if rate_limit_reset and rate_limit_reset > datetime.now():
        return await ctx.send(
            "The command is currently being ratelimited!", ephemeral=True
        )

    try:
        await cog.queue_response(ctx)
    except Exception:
        await ctx.send(":warning: Error in generating response!", ephemeral=True)


async def handle_message(cog: MixinMeta, message: discord.Message):
    """Handle regular message events"""
    ctx: commands.Context = await cog.bot.get_context(message)

    if not (await is_valid_message(cog, ctx)):
        return

    is_triggered = await check_triggers(cog, ctx, message)

    # Note: messages arriving while a response is being processed are NOT
    # dropped — they are queued via queue_response and handled in order.
    # Dropping them would lose conversation context and contradicts the
    # per-channel queue design.

    if is_triggered:
        pass
    elif random.random() > await get_percentage(cog, ctx):
        return

    rate_limit_reset = await get_ratelimit_reset(cog)
    if rate_limit_reset and rate_limit_reset > datetime.now():
        logger.debug(
            f"Want to respond but ratelimited until {rate_limit_reset.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if (
            is_triggered
            or await get_percentage(cog, ctx) == 1.0
        ):
            await ctx.react_quietly("💤", message="`aiuser` is ratedlimited")
        return

    if URL_PATTERN.search(ctx.message.content):
        ctx = await wait_for_embed(ctx)

    await cog.queue_response(ctx)


async def get_ratelimit_reset(cog: MixinMeta) -> Optional[datetime]:
    """Parse the configured ratelimit reset timestamp.

    Accepts the current epoch-seconds format (timezone-independent) as well
    as the legacy ``%Y-%m-%d %H:%M:%S`` local-time string for backward
    compatibility.  Returns None if the value is missing or malformed
    instead of raising, so a corrupt config value can never break message
    handling.
    """
    raw = await cog.config.ratelimit_reset()
    if raw is None:
        return None
    try:
        # Current format: epoch seconds
        return datetime.fromtimestamp(float(raw))
    except (ValueError, TypeError, OSError, OverflowError):
        pass
    try:
        # Legacy format: naive local-time string
        return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning(f"Ignoring malformed ratelimit_reset value: {raw!r}")
        return None


async def get_percentage(cog: MixinMeta, ctx: commands.Context) -> float:
    """Get reply percentage based on member/role/channel/guild settings"""
    role_percent = None
    author = ctx.author

    configured_roles = await cog.config.all_roles()
    # author.roles is sorted ascending by hierarchy; iterate highest-first
    # so the highest-priority configured role wins.
    for role in reversed(author.roles):
        if role.id in configured_roles:
            role_percent = await cog.config.role(role).reply_percent()
            break

    percentage = await cog.config.member(author).reply_percent()
    if percentage is None:
        percentage = role_percent
    if percentage is None:
        percentage = await cog.config.channel(ctx.channel).reply_percent()
    if percentage is None:
        percentage = await cog.config.guild(ctx.guild).reply_percent()
    if percentage is None:
        percentage = DEFAULT_REPLY_PERCENT
    return percentage


async def wait_for_embed(ctx: commands.Context) -> commands.Context:
    """Wait for a possible embed to be populated on the triggering message.

    Re-fetches the message (Discord populates embeds asynchronously after
    send).  Gives up gracefully if the message was deleted or embeds never
    arrive, instead of raising.
    """
    start_time = time.monotonic()
    while not is_embed_valid(ctx.message):
        if time.monotonic() - start_time >= 3:
            break
        await asyncio.sleep(1)
        try:
            ctx.message = await ctx.channel.fetch_message(ctx.message.id)
        except discord.NotFound:
            logger.debug("Message deleted while waiting for embed; aborting wait")
            break
        except discord.HTTPException:
            logger.debug("Failed to re-fetch message while waiting for embed", exc_info=True)
            break
    return ctx
