import asyncio
import io
import logging
import re
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import discord
from discord import AllowedMentions, Embed
from redbot.core import Config, commands

from aiuser.config.constants import REGEX_RUN_TIMEOUT
from aiuser.messages_list.messages import MessagesList
from aiuser.response.chat.llm_pipeline import LLMPipeline, ResponsePart
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import to_thread, resolve_emojis_for_discord, escape_unescaped_backticks, collapse_lines

logger = logging.getLogger("red.bz_cogs.aiuser")

# Use to_thread to compile & apply a regex pattern
@to_thread(timeout=REGEX_RUN_TIMEOUT)
def compile_and_apply(pattern_str: str, text: str) -> str:
    pattern = re.compile(pattern_str)
    return pattern.sub('', text).strip(' \n')

async def remove_patterns_from_response(ctx: commands.Context, config: Config, response: str, recent_authors) -> str:
    # Get patterns from config and replace "{botname}".
    patterns = await config.guild(ctx.guild).removelist_regexes()
    botname = ctx.message.guild.me.name or ctx.bot.user.name
    botdisplayname = ctx.message.guild.me.display_name or ctx.bot.user.display_name

    bot_expanded_patterns = []
    for pattern in patterns:
        bot_expanded_patterns.append(pattern.replace(r'{botname}', botname).replace(r'{botdisplayname}', botdisplayname))

    final_expanded_patterns = []
    for pattern in bot_expanded_patterns:
        if '{authorname}' in pattern or '{authordisplayname}' in pattern:
            for author in recent_authors:
                final_expanded_patterns.append(pattern.replace(r'{authorname}', author.name).replace(r'{authordisplayname}', author.display_name))
        else:
            final_expanded_patterns.append(pattern)

    # Apply each pattern sequentially.
    cleaned = response.strip(' \n')
    for pattern in final_expanded_patterns:
        try:
            cleaned = await compile_and_apply(pattern, cleaned)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout applying regex pattern: {pattern}")
        except Exception:
            logger.warning(f"Error applying regex pattern: {pattern}", exc_info=True)
    return cleaned

async def should_reply(ctx: commands.Context) -> bool:
    if ctx.interaction:
        return False

    try:
        await ctx.fetch_message(ctx.message.id)
    except Exception:
        return False

    if (datetime.now(timezone.utc) - ctx.message.created_at).total_seconds() > 8 or random.random() < 0.25:
        return True

    async for last_msg in ctx.message.channel.history(limit=1):
        if last_msg.author == ctx.message.guild.me:
            return True
    return False

async def send_response(ctx: commands.Context, response: str, can_reply: bool, mentionable_users) -> bool:
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])

    if len(response) > 4096:
        total_embed_count = math.ceil(len(response) / 4096)

        for i in range(0, len(response), 4096):
            embed = Embed(title=f"{ctx.bot.user.name}'s Response", description = response[i:i + 4096])
            embed.set_footer(text=f"{int((i + 4096) / 4096)} of {total_embed_count}")
            await ctx.send(embed=embed, allowed_mentions=allowed)
    elif can_reply and await should_reply(ctx):
        await ctx.message.reply(embed=Embed(title=f"{ctx.bot.user.name}'s Response", description = response), mention_author=False, allowed_mentions=allowed)
    elif ctx.interaction:
        await ctx.interaction.followup.send(embed=Embed(title=f"{ctx.bot.user.name}'s Response", description = response), allowed_mentions=allowed)
    else:
        await ctx.send(embed=Embed(title=f"{ctx.bot.user.name}'s Response", description = response), allowed_mentions=allowed)
    return True

async def send_reasoning(ctx: commands.Context, reasoning: str, can_reply: bool, mentionable_users) -> bool:
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])

    if len(reasoning) > 4092:
        total_embed_count = math.ceil(len(reasoning) / 4092)

        for i in range(0, len(reasoning), 4092):
            embed = Embed(title=f"{ctx.bot.user.name}'s Thoughts", description = f"||{reasoning[i:i + 4092]}||")
            embed.set_footer(text=f"{int((i + 4092) / 4092)} of {total_embed_count}")
            await ctx.send(embed=embed, allowed_mentions=allowed)
    elif can_reply and await should_reply(ctx):
        await ctx.message.reply(embed=Embed(title=f"{ctx.bot.user.name}'s Thoughts", description = f"||{reasoning}||"), mention_author=False, allowed_mentions=allowed)
    elif ctx.interaction:
        await ctx.interaction.followup.send(embed=Embed(title=f"{ctx.bot.user.name}'s Thoughts", description = f"||{reasoning}||"), allowed_mentions=allowed)
    else:
        await ctx.send(embed=Embed(title=f"{ctx.bot.user.name}'s Thoughts", description = f"||{reasoning}||"), allowed_mentions=allowed)
    return True

async def create_chat_response(cog: MixinMeta, ctx: commands.Context, messages_list: MessagesList) -> bool:
    pipeline = LLMPipeline(cog, ctx, messages=messages_list)
    response_parts, response, reasoning = await pipeline.run()
    if not response and not response_parts:
        return False

    recent_authors = [
        msg.author async for msg in ctx.channel.history(limit=20)
        if msg.author != ctx.guild.me
    ]
    
    cleaned_reasoning = None
    if reasoning:
        try:
            cleaned_reasoning = await remove_patterns_from_response(ctx, cog.config, reasoning, recent_authors)
        except Exception:
            cleaned_reasoning = None

    if cleaned_reasoning:
        # Collapse multiple newlines and blank lines for more compact embed
        cleaned_reasoning = collapse_lines(cleaned_reasoning, replacement=r'\n')
        cleaned_reasoning = escape_unescaped_backticks(cleaned_reasoning)
        cleaned_reasoning = await resolve_emojis_for_discord(ctx, cleaned_reasoning)
        await send_reasoning(ctx, cleaned_reasoning, messages_list.can_reply, recent_authors)

    # Send response parts sequentially, all as replies to the original message
    if response_parts:
        return await send_response_parts(ctx, cog, response_parts, messages_list.can_reply, recent_authors)

    # Fallback: if no parts, send the combined response as before
    try:
        cleaned_response = await remove_patterns_from_response(ctx, cog.config, response, recent_authors)
    except Exception:
        cleaned_response = None

    if not cleaned_response:
        return False

    cleaned_response = collapse_lines(cleaned_response, replacement=r'\n\n')
    cleaned_response = escape_unescaped_backticks(cleaned_response)
    cleaned_response = await resolve_emojis_for_discord(ctx, cleaned_response)
    return await send_response(ctx, cleaned_response, messages_list.can_reply, recent_authors)


async def send_response_parts(
    ctx: commands.Context,
    cog: MixinMeta,
    parts: List[ResponsePart],
    can_reply: bool,
    recent_authors
) -> bool:
    """Send response parts sequentially, all as replies to the original triggering message.

    Parts are sent in order:
    1. Text parts are sent as separate embed replies.
    2. An image part (containing all images) is sent as a single message with file attachments.

    Each part is sent as a reply to the original triggering message (ctx.message).

    Args:
        ctx: The command context.
        cog: The AIUser cog instance.
        parts: Ordered list of ResponsePart objects.
        can_reply: Whether the bot can reply to messages.
        recent_authors: List of recent message authors for regex filtering.

    Returns:
        True if any part was sent successfully, False otherwise.
    """
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])

    sent_any = False

    for part in parts:
        if part.type == "text":
            text = str(part.content)
            if not text or not text.strip():
                continue

            # Clean up the text
            try:
                cleaned_text = await remove_patterns_from_response(ctx, cog.config, text, recent_authors)
            except Exception:
                cleaned_text = None

            if not cleaned_text:
                continue

            cleaned_text = collapse_lines(cleaned_text, replacement=r'\n\n')
            cleaned_text = escape_unescaped_backticks(cleaned_text)
            cleaned_text = await resolve_emojis_for_discord(ctx, cleaned_text)

            if len(cleaned_text) > 4096:
                total_embed_count = math.ceil(len(cleaned_text) / 4096)
                for i in range(0, len(cleaned_text), 4096):
                    embed = Embed(title=f"{ctx.bot.user.name}'s Response", description=cleaned_text[i:i + 4096])
                    embed.set_footer(text=f"{int((i + 4096) / 4096)} of {total_embed_count}")
                    await ctx.message.reply(embed=embed, mention_author=False, allowed_mentions=allowed)
            else:
                await ctx.message.reply(
                    embed=Embed(title=f"{ctx.bot.user.name}'s Response", description=cleaned_text),
                    mention_author=False,
                    allowed_mentions=allowed
                )
            sent_any = True

        elif part.type == "images":
            image_data_list = part.content
            if not image_data_list or not isinstance(image_data_list, list) or len(image_data_list) == 0:
                continue

            # Build Discord file attachments for all images
            files = []
            for img_data in image_data_list:
                if not isinstance(img_data, dict):
                    continue
                img_bytes = img_data.get("bytes")
                filename = img_data.get("filename", "generated_image.png")
                if img_bytes:
                    files.append(discord.File(fp=io.BytesIO(img_bytes), filename=filename))

            if not files:
                continue

            # Send all images as a single message in reply to the original message
            try:
                await ctx.message.reply(files=files, mention_author=False, allowed_mentions=allowed)
                sent_any = True
                logger.info(
                    f"[ResponseParts] Sent {len(files)} generated image(s) as reply "
                    f"to message {ctx.message.id} in #{ctx.channel.name}"
                )
            except discord.HTTPException as e:
                logger.error(f"[ResponseParts] Failed to send images as reply: {e}")
                # Fallback: send as separate messages
                for f in files:
                    try:
                        await ctx.message.reply(file=f, mention_author=False, allowed_mentions=allowed)
                    except Exception as e2:
                        logger.error(f"[ResponseParts] Failed to send image {f.filename}: {e2}")

    return sent_any
