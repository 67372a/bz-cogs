import asyncio
import io
import logging
import math
import random
import re
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
    """Create and send a chat response using the two-phase pipeline.

    Phase 1: first LLM call.
    - If no tool calls: send text as a single message, done.
    - If tool calls and pre-text is complete: send pre-text immediately (early output),
      then proceed to phase 2.
    - If tool calls and pre-text incomplete: wait for phase 2 to send combined message.

    Phase 2: execute tools + second LLM call.
    - Sends final text + images as a single combined Discord message
      (embed with file attachments).
    """
    pipeline = LLMPipeline(cog, ctx, messages=messages_list)

    # ---- Phase 1: First LLM call ----
    pre_text, reasoning, has_tools, is_incomplete = await pipeline.phase1()

    if not pre_text and not has_tools:
        return False

    recent_authors = [
        msg.author async for msg in ctx.channel.history(limit=20)
        if msg.author != ctx.guild.me
    ]

    # Send reasoning if present (always before any response text)
    if reasoning:
        try:
            cleaned_reasoning = await remove_patterns_from_response(ctx, cog.config, reasoning, recent_authors)
        except Exception:
            cleaned_reasoning = None
        if cleaned_reasoning:
            cleaned_reasoning = collapse_lines(cleaned_reasoning, replacement=r'\n')
            cleaned_reasoning = escape_unescaped_backticks(cleaned_reasoning)
            cleaned_reasoning = await resolve_emojis_for_discord(ctx, cleaned_reasoning)
            await send_reasoning(ctx, cleaned_reasoning, messages_list.can_reply, recent_authors)

    if not has_tools:
        # No tool calls — single response, send it and done
        if pre_text:
            try:
                cleaned_pre = await remove_patterns_from_response(ctx, cog.config, pre_text, recent_authors)
            except Exception:
                cleaned_pre = None
            if cleaned_pre:
                cleaned_pre = collapse_lines(cleaned_pre, replacement=r'\n\n')
                cleaned_pre = escape_unescaped_backticks(cleaned_pre)
                cleaned_pre = await resolve_emojis_for_discord(ctx, cleaned_pre)
                return await send_response(ctx, cleaned_pre, messages_list.can_reply, recent_authors)
        return True

    # ---- Has tool calls ----
    # Early output: if pre-text doesn't look incomplete, send it immediately
    sent_early = False
    if pre_text and not is_incomplete:
        try:
            cleaned_pre = await remove_patterns_from_response(ctx, cog.config, pre_text, recent_authors)
        except Exception:
            cleaned_pre = None
        if cleaned_pre:
            cleaned_pre = collapse_lines(cleaned_pre, replacement=r'\n\n')
            cleaned_pre = escape_unescaped_backticks(cleaned_pre)
            cleaned_pre = await resolve_emojis_for_discord(ctx, cleaned_pre)
            await send_response(ctx, cleaned_pre, messages_list.can_reply, recent_authors)
            sent_early = True

    # ---- Phase 2: tools + second LLM call ----
    final_text, final_reasoning, images = await pipeline.phase2()

    # If there's a combined final text or images, send them together
    text_to_send = final_text
    if pipeline.response_parts and pipeline.response_parts[0].type == "text_and_images":
        text_to_send = pipeline.response_parts[0].content
        if not images and pipeline.response_parts[0].images:
            images = pipeline.response_parts[0].images

    if not text_to_send and not images:
        return sent_early or False

    await send_single_combined_message(ctx, cog, text_to_send, images, messages_list.can_reply, recent_authors)
    return True


async def send_single_combined_message(
    ctx: commands.Context,
    cog: MixinMeta,
    text: Optional[str],
    images: List[Dict],
    can_reply: bool,
    recent_authors,
) -> bool:
    """Send a single Discord reply with an embed + optional image file attachments.

    If text is empty but images exist, sends only the files.
    If images are empty but text exists, sends only the embed.
    If both, sends embed + files in a single message.

    The message is always a reply to the original triggering message (ctx.message).
    """
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])

    # Build Discord file attachments
    files = []
    for img_data in images:
        if not isinstance(img_data, dict):
            continue
        img_bytes = img_data.get("bytes")
        filename = img_data.get("filename", "generated_image.png")
        if img_bytes:
            files.append(discord.File(fp=io.BytesIO(img_bytes), filename=filename))

    # Clean text
    cleaned_text = None
    if text:
        try:
            cleaned_text = await remove_patterns_from_response(ctx, cog.config, text, recent_authors)
        except Exception:
            cleaned_text = None
        if cleaned_text:
            cleaned_text = collapse_lines(cleaned_text, replacement=r'\n\n')
            cleaned_text = escape_unescaped_backticks(cleaned_text)
            cleaned_text = await resolve_emojis_for_discord(ctx, cleaned_text)

    # Handle the 3 cases
    try:
        if cleaned_text and files:
            # Combined: embed + files in one reply
            embed = Embed(title=f"{ctx.bot.user.name}'s Response", description=cleaned_text)
            sent_message = await ctx.message.reply(
                embed=embed,
                files=files,
                mention_author=False,
                allowed_mentions=allowed,
            )
            logger.info(
                f"[Combined] Sent text embed + {len(files)} image(s) in one reply "
                f"to message {ctx.message.id}"
            )
            # Cache image data URLs for future context turn re-ingestion
            _cache_generated_images_in_response(ctx, cog, sent_message, images)
            return True

        elif cleaned_text:
            # Text only
            return await send_response(ctx, cleaned_text, can_reply, recent_authors)

        elif files:
            # Images only
            sent_message = await ctx.message.reply(files=files, mention_author=False, allowed_mentions=allowed)
            logger.info(f"[Combined] Sent {len(files)} image(s) as reply to message {ctx.message.id}")
            _cache_generated_images_in_response(ctx, cog, sent_message, images)
            return True

    except discord.HTTPException as e:
        logger.error(f"[Combined] Failed to send combined reply: {e}")
        # Fallback: send text and images separately
        if cleaned_text:
            await send_response(ctx, cleaned_text, can_reply, recent_authors)
        for f in files:
            try:
                await ctx.message.reply(file=f, mention_author=False, allowed_mentions=allowed)
            except Exception as e2:
                logger.error(f"[Combined] Failed to send image {f.filename}: {e2}")
        return bool(cleaned_text or files)

    except Exception as e:
        logger.error(f"[Combined] Unexpected error sending reply: {e}")
        return False

    return False


def _cache_generated_images_in_response(
    ctx: commands.Context,
    cog: MixinMeta,
    sent_message: discord.Message,
    images: List[Dict],
):
    """Cache generated image data URLs into the message cache so they're
    available when the bot's response message is re-ingested on future turns.

    Stores image_url content parts under the sent message's channel-scoped key.
    """
    if not images or not sent_message:
        return

    # Build image_url content parts from the generated image data
    content_parts: List[dict] = []
    client_text = getattr(sent_message.embeds[0], "description", "") if sent_message.embeds else ""

    for img_data in images:
        data_url = img_data.get("data_url")
        if data_url:
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    if not content_parts:
        return

    # Add the text content after the images (matching how user images are formatted)
    if client_text:
        content_parts.append({"type": "text", "text": client_text})

    # Cache under the sent message ID for re-ingestion on future turns
    cache_key = f"{ctx.channel.id}:{sent_message.id}"
    cog.cached_messages[cache_key] = content_parts
    cog.cached_messages[sent_message.id] = content_parts

    logger.info(
        f"[Cache] Cached {len(content_parts)} content part(s) for future re-ingestion "
        f"of message {sent_message.id} (channel {ctx.channel.id})"
    )
