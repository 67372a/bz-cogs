import asyncio
import logging
import random
import re
import math
from datetime import datetime, timezone

from discord import AllowedMentions, Embed
from redbot.core import Config, commands

from aiuser.config.constants import REGEX_RUN_TIMEOUT
from aiuser.messages_list.messages import MessagesList
from aiuser.response.chat.llm_pipeline import LLMPipeline
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
    response, reasoning = await pipeline.run()
    if not response:
        return False
    
    recent_authors = [
        msg.author async for msg in ctx.channel.history(limit=20)
        if msg.author != ctx.guild.me
    ]
    
    cleaned_reasoning = None
    if reasoning:
        res = await asyncio.gather(
            remove_patterns_from_response(ctx, cog.config, response, recent_authors),
            remove_patterns_from_response(ctx, cog.config, reasoning, recent_authors),
            return_exceptions=True
        )
        cleaned_response = res[0] if not isinstance(res[0], Exception) else None
        cleaned_reasoning = res[1] if not isinstance(res[1], Exception) else None
    else:
        try:
            cleaned_response = await remove_patterns_from_response(ctx, cog.config, response, recent_authors)
        except Exception:
            cleaned_response = None

    if not cleaned_response:
        return False
    
    if cleaned_reasoning:
        # Collapse multiple newlines and blank lines for more compact embed
        cleaned_reasoning = collapse_lines(cleaned_reasoning, replacement=r'\n')
        cleaned_reasoning = escape_unescaped_backticks(cleaned_reasoning)
        cleaned_reasoning = await resolve_emojis_for_discord(ctx, cleaned_reasoning)
        await send_reasoning(ctx, cleaned_reasoning, messages_list.can_reply, recent_authors)

    # Collapse multiple newlines and blank lines for more compact embed
    cleaned_response = collapse_lines(cleaned_response, replacement=r'\n\n')
    cleaned_response = escape_unescaped_backticks(cleaned_response)
    cleaned_response = await resolve_emojis_for_discord(ctx, cleaned_response)
    return await send_response(ctx, cleaned_response, messages_list.can_reply, recent_authors)
