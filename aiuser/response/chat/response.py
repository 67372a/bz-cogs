import asyncio
import logging
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

from aiuser.config.constants import REGEX_RUN_TIMEOUT, XML_RESERVED_TAG_PATTERN
from aiuser.messages_list.messages import MessagesList
from aiuser.response.chat.llm_pipeline import LLMPipeline, ResponsePart, PipelineResult
from aiuser.response.chat.function_call_view import ResponseView
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import to_thread, resolve_emojis_for_discord, escape_unescaped_backticks, collapse_lines
from aiuser.utils.latex_converter import convert_latex_to_plain

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

async def _clean_response_text(ctx: commands.Context, cog: MixinMeta, text: str, recent_authors) -> Optional[str]:
    """Run the full text-cleaning pipeline on a response string.

    Applies (in order): regex removal patterns, line collapse, backtick
    escaping, LaTeX-to-plain conversion, and Discord emoji resolution.

    Returns the cleaned text, or None if cleaning produced nothing usable.
    """
    # Hardcoded, non-configurable strip of reserved Semantic XML metadata tags.
    # Prevents leaking the internal context schema to Discord even if a guild
    # empties its removelist. Only the tags are removed; inner text is kept.
    text = XML_RESERVED_TAG_PATTERN.sub('', text)
    try:
        cleaned = await remove_patterns_from_response(ctx, cog.config, text, recent_authors)
    except Exception:
        return None
    if not cleaned:
        return None
    cleaned = collapse_lines(cleaned, replacement=r'\n\n')
    cleaned = escape_unescaped_backticks(cleaned)
    cleaned = convert_latex_to_plain(cleaned)
    cleaned = await resolve_emojis_for_discord(ctx, cleaned)
    return cleaned


async def send_response(
    ctx: commands.Context,
    response: str,
    can_reply: bool,
    mentionable_users,
    files: Optional[List[discord.File]] = None,
) -> Optional[discord.Message]:
    """Send the response embed to Discord, optionally with file attachments.

    When ``files`` are provided, they are attached to the **first** message
    sent (the one carrying the first embed chunk). This keeps file
    attachments alongside the response text even when the text must be
    split across multiple embeds (Discord caps embed descriptions at 4096
    chars).

    Returns the sent message (or None) so callers can attach views.
    """
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])
    sent_message = None

    if len(response) > 4096:
        total_embed_count = math.ceil(len(response) / 4096)

        for i in range(0, len(response), 4096):
            embed = Embed(title=f"{ctx.bot.user.name}'s Response", description=response[i:i + 4096])
            embed.set_footer(text=f"{int((i + 4096) / 4096)} of {total_embed_count}")
            # Attach files only to the first chunk so they accompany the
            # response rather than being sent as standalone messages.
            chunk_files = files if i == 0 else None
            sent_message = await ctx.send(embed=embed, files=chunk_files, allowed_mentions=allowed)
    elif can_reply and await should_reply(ctx):
        sent_message = await ctx.message.reply(
            embed=Embed(title=f"{ctx.bot.user.name}'s Response", description=response),
            files=files,
            mention_author=False,
            allowed_mentions=allowed,
        )
    elif ctx.interaction:
        sent_message = await ctx.interaction.followup.send(
            embed=Embed(title=f"{ctx.bot.user.name}'s Response", description=response),
            files=files,
            allowed_mentions=allowed,
        )
    else:
        sent_message = await ctx.send(
            embed=Embed(title=f"{ctx.bot.user.name}'s Response", description=response),
            files=files,
            allowed_mentions=allowed,
        )
    return sent_message


async def _attach_reasoning_view(message: Optional[discord.Message], reasoning_steps: Optional[List[str]]):
    """Attach a ResponseView with reasoning to a sent message, if reasoning is available."""
    if not message or not reasoning_steps:
        return
    try:
        ResponseView.store_reasoning_steps(message.id, reasoning_steps)
        view = ResponseView(message_id=message.id, has_reasoning=True)
        await message.edit(view=view)
    except Exception:
        logger.warning("Failed to attach reasoning view to response message", exc_info=True)

async def create_chat_response(cog: MixinMeta, ctx: commands.Context, messages_list: MessagesList) -> bool:
    """Create and send a chat response using the tool-calling loop pipeline.

    The pipeline runs a configurable number of LLM rounds, executing tools
    between rounds. If no tools are called, the response is sent immediately.
    If tools are called and first-round text is complete, it is sent early
    while the loop continues executing tools.
    """
    pipeline = LLMPipeline(cog, ctx, messages=messages_list)

    # ---- Run the tool-calling loop ----
    result = await pipeline.run()

    if not result.text and not result.has_tools:
        return False

    recent_authors = [
        msg.author async for msg in ctx.channel.history(limit=20)
        if msg.author != ctx.guild.me
    ]

    if not result.has_tools:
        # No tool calls — single response, send it and done
        if result.text:
            cleaned_text = await _clean_response_text(ctx, cog, result.text, recent_authors)
            if cleaned_text:
                sent_msg = await send_response(ctx, cleaned_text, messages_list.can_reply, recent_authors)
                await _attach_reasoning_view(sent_msg, result.reasoning_steps)
                return True
        return True

    # ---- Has tool calls ----
    # Early output: if pre-text doesn't look incomplete, send it immediately
    sent_early = False
    if result.pre_text and result.pre_text_complete:
        cleaned_pre = await _clean_response_text(ctx, cog, result.pre_text, recent_authors)
        if cleaned_pre:
            sent_msg = await send_response(ctx, cleaned_pre, messages_list.can_reply, recent_authors)
            await _attach_reasoning_view(sent_msg, result.reasoning_steps)
            sent_early = True

    # ---- Final output: text + images ----
    final_text = result.text
    images = result.images

    logger.info(
        "Pipeline run complete: final_text_length=%d, images=%d, response_parts=%d, sent_early=%s",
        len(final_text or ''), len(images), len(result.response_parts), sent_early
    )

    # If there's a combined final text or images, send them together
    text_to_send = final_text
    if result.response_parts and result.response_parts[0].type == "text_and_images":
        text_to_send = result.response_parts[0].content
        if not images and result.response_parts[0].images:
            images = result.response_parts[0].images

    logger.info(
        "Sending combined message: text_length=%d, images=%d, will_send=%s",
        len(text_to_send or ''), len(images), bool(text_to_send or images)
    )

    if not text_to_send and not images:
        logger.warning("No text or images to send after pipeline run — bot will not reply!")
        return sent_early or False

    await send_single_combined_message(ctx, cog, text_to_send, images, messages_list.can_reply, recent_authors, reasoning_steps=result.reasoning_steps)
    return True


async def send_single_combined_message(
    ctx: commands.Context,
    cog: MixinMeta,
    text: Optional[str],
    images: List[Dict],
    can_reply: bool,
    recent_authors,
    reasoning_steps: Optional[List[str]] = None,
) -> bool:
    """Send a Discord reply with an embed + optional file attachments.

    Handles three cases:
      * text + files  -> embed(s) + files in one reply (files on first chunk)
      * text only     -> embed(s) only
      * files only    -> files only (no embed)

    Long text (>4096 chars) is automatically split across multiple embeds by
    ``send_response``; files are attached to the first message so they
    accompany the response rather than being sent as standalone messages.

    The message is always a reply to the original triggering message.
    """
    allowed = AllowedMentions(everyone=False, roles=False, users=[ctx.message.author])

    # Build Discord file attachments from image/file dicts
    files = []
    for img_data in images:
        if not isinstance(img_data, dict):
            continue
        img_bytes = img_data.get("bytes")
        filename = img_data.get("filename", "generated_image.png")
        if img_bytes:
            files.append(discord.File(fp=io.BytesIO(img_bytes), filename=filename))

    cleaned_text = await _clean_response_text(ctx, cog, text, recent_authors) if text else None

    logger.info(
        "[Combined] Branch decision: cleaned_text_len=%d, files=%d, "
        "text_exceeds_4096=%s",
        len(cleaned_text or ''), len(files), (len(cleaned_text or '') > 4096)
    )

    try:
        if cleaned_text:
            # Text (with or without files): send_response handles pagination
            # and attaches files to the first message.
            sent_msg = await send_response(
                ctx, cleaned_text, can_reply, recent_authors,
                files=files or None,
            )
            logger.info(
                "[Combined] Sent text embed%s in reply to message %s",
                f" + {len(files)} file(s)" if files else "",
                ctx.message.id,
            )
            if files:
                _cache_generated_images_in_response(ctx, cog, sent_msg, images)
            await _attach_reasoning_view(sent_msg, reasoning_steps)
            return True

        elif files:
            # Files only, no text
            sent_message = await ctx.message.reply(
                files=files, mention_author=False, allowed_mentions=allowed
            )
            logger.info(
                "[Combined] Sent %d file(s) as reply to message %s",
                len(files), ctx.message.id,
            )
            _cache_generated_images_in_response(ctx, cog, sent_message, images)
            await _attach_reasoning_view(sent_message, reasoning_steps)
            return True

    except discord.HTTPException as e:
        logger.error(
            "[Combined] Failed to send combined reply: %s (status=%s, text_len=%d, files=%d)",
            e, getattr(e, 'status', None), len(cleaned_text or ''), len(files)
        )
        # Fallback: send text and files separately
        sent_msg = None
        if cleaned_text:
            sent_msg = await send_response(ctx, cleaned_text, can_reply, recent_authors)
        for f in files:
            try:
                await ctx.message.reply(file=f, mention_author=False, allowed_mentions=allowed)
            except Exception as e2:
                logger.error(f"[Combined] Failed to send file {f.filename}: {e2}")
        await _attach_reasoning_view(sent_msg, reasoning_steps)
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
    """Cache processed image data URLs into the message cache so they're
    available when the bot's response message is re-ingested on future turns.

    Uses the unified image processing pipeline so re-ingested images match
    the same resize/compress treatment as all other LLM-bound images.
    """
    from aiuser.utils.image_cache import processed_image_cache
    from aiuser.utils.image_processing import (
        compute_pixel_hash,
        process_image_for_llm,
        build_webp_data_url,
    )

    if not images or not sent_message:
        return

    # Build image_url content parts from the generated image data
    content_parts: List[dict] = []
    client_text = getattr(sent_message.embeds[0], "description", "") if sent_message.embeds else ""

    max_pixels = 16_777_216  # 4096×4096 default for LLM context

    for img_data in images:
        raw_bytes = img_data.get("bytes")
        if raw_bytes:
            pixel_hash = compute_pixel_hash(raw_bytes)
            cached_processed = processed_image_cache.get(raw_bytes, pixel_hash, max_pixels)
            if cached_processed is None:
                cached_processed = process_image_for_llm(raw_bytes, max_pixels)
                if cached_processed is not None:
                    processed_image_cache.set(raw_bytes, pixel_hash, max_pixels, cached_processed)

            if cached_processed is not None:
                data_url = build_webp_data_url(cached_processed)
            else:
                # Fallback to the original data_url if processing failed
                data_url = img_data.get("data_url")

            if data_url:
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            # Non-bytes entries (e.g. attach_files) — use data_url directly
            data_url = img_data.get("data_url")
            if data_url:
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    if not content_parts:
        return

    # Add the bot's actual response text so it isn't lost when this message
    # is re-ingested from the cache on future turns (the cache short-circuits
    # conversion, so anything not stored here never re-enters context).
    bot_text = getattr(sent_message, "content", "") or ""
    if bot_text.strip():
        content_parts.append({"type": "text", "text": bot_text})

    # Add the embed description text after the images (matching how user
    # images are formatted)
    if client_text:
        content_parts.append({"type": "text", "text": client_text})

    # Cache under the sent message ID for re-ingestion on future turns
    cache_key = f"{ctx.channel.id}:{sent_message.id}"
    cog.cached_messages[cache_key] = content_parts
    cog.cached_messages[sent_message.id] = content_parts

    logger.info(
        f"[Cache] Cached {len(content_parts)} processed content part(s) for future re-ingestion "
        f"of message {sent_message.id} (channel {ctx.channel.id})"
    )
