import asyncio
import logging
import random
import re
import math
from datetime import datetime, timezone

from discord import AllowedMentions, Embed, Emoji
from redbot.core import Config, commands

from aiuser.config.constants import REGEX_RUN_TIMEOUT
from aiuser.messages_list.messages import MessagesList
from aiuser.response.chat.llm_pipeline import LLMPipeline
from aiuser.types.abc import MixinMeta
from aiuser.utils.utilities import to_thread

logger = logging.getLogger("red.bz_cogs.aiuser")

MULTI_NEWLINE_PATTERN = re.compile(r'(?:([ \t]|<br>|\\n)*(?:\r\n|\r|\n)){2,}')

EMOJI_PATTERN = re.compile(r"(?:<a?)?:([a-zA-Z0-9_]+?):(?:\d*>)?")

# Use to_thread to compile & apply a regex pattern
@to_thread(timeout=REGEX_RUN_TIMEOUT)
def compile_and_apply(pattern_str: str, text: str) -> str:
    pattern = re.compile(pattern_str)
    return pattern.sub('', text).strip(' \n')

import re

def collapse_lines(text, replacement: str = r'\n'):
  """
  Collapses sequences of two or more blank lines (or just newlines)
  into a single standard newline.
  Handles LF, CRLF, and CR line endings.
  A "blank line" here means zero or more spaces/tabs followed by a newline sequence.
  """
  # Pattern explanation:
  # (?:               # Start of a non-capturing group for a "blank line unit"
  #   [ \t]*          # Match zero or more spaces or tabs (horizontal whitespace)
  #   (?:\r\n|\r|\n)  # Match any kind of newline: CRLF (Windows), CR (old Mac), or LF (Unix/modern Mac)
  #                    # The order \r\n before \r is important to match CRLF correctly.
  # )                  # End of the non-capturing group for a "blank line unit"
  # {2,}               # Match 2 or more occurrences of the preceding "blank line unit"
  # r'(?:[ \t]*(?:\r\n|\r|\n)){2,}'

  # Replace the matched sequence of multiple blank lines with a single standard newline '\n'
  return re.sub(MULTI_NEWLINE_PATTERN, replacement, text)

async def remove_patterns_from_response(ctx: commands.Context, config: Config, response: str) -> str:
    # Get patterns from config and replace "{botname}".
    patterns = await config.guild(ctx.guild).removelist_regexes()
    botname = ctx.message.guild.me.name or ctx.bot.user.name
    botdisplayname = ctx.message.guild.me.display_name or ctx.bot.user.display_name

    for pattern in patterns:
        pattern.replace(r'{botname}', botname)
        pattern.replace(r'{botdisplayname}', botdisplayname)

    # Expand patterns that have "{authorname}" based on recent authors.
    authors = {
        msg.author async for msg in ctx.channel.history(limit=20)
        if msg.author != ctx.guild.me
    }
    expanded_patterns = []
    for pattern in patterns:
        if '{authorname}' in pattern or '{authordisplayname}' in pattern:
            for author in authors:
                expanded_patterns.append(pattern.replace(r'{authorname}', author.name).replace(r'{authordisplayname}', author.display_name))
        else:
            expanded_patterns.append(pattern)

    # Apply each pattern sequentially.
    cleaned = response.strip(' \n')
    for pattern in expanded_patterns:
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

async def send_response(ctx: commands.Context, response: str, can_reply: bool) -> bool:
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

async def send_reasoning(ctx: commands.Context, reasoning: str, can_reply: bool) -> bool:
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
    
    cleaned_reasoning = None
    if reasoning:
        cleaned_response, cleaned_reasoning = await asyncio.gather(remove_patterns_from_response(ctx, cog.config, response),
                                         remove_patterns_from_response(ctx, cog.config, reasoning),
                                         return_exceptions=True)
    else:
        cleaned_response = await remove_patterns_from_response(ctx, cog.config, response)

    if not cleaned_response:
        return False
    
    if cleaned_reasoning:
        # Collapse multiple newlines and blank lines for more compact embed
        cleaned_reasoning = collapse_lines(cleaned_reasoning, replacement=r'\n')
        cleaned_reasoning = cleaned_reasoning.replace(r'`',r'\`')
        cleaned_reasoning = resolve_emojis_for_discord(ctx, cleaned_reasoning)
        await send_reasoning(ctx, cleaned_reasoning, messages_list.can_reply)

    # Collapse multiple newlines and blank lines for more compact embed
    cleaned_response = collapse_lines(cleaned_response, replacement=r'\n\n')
    cleaned_response = cleaned_response.replace(r'`',r'\`')
    cleaned_response = resolve_emojis_for_discord(ctx, cleaned_response)
    return await send_response(ctx, cleaned_response, messages_list.can_reply)

# --- Emoji Resolver Function (Refined) ---
def resolve_emojis_for_discord(ctx: commands.Context, text_content: str) -> str:
    """
    Resolves :emoji_name: shortcodes in a string to their full Discord format,
    leaving already correctly formatted <:...:id> emojis untouched.

    Args:
        text_content: The string potentially containing emoji shortcodes.
        available_emojis: A list of discord.Emoji objects (e.g., from bot.emojis or guild.emojis).

    Returns:
        The string with emoji shortcodes replaced.
    """

    available_emojis = ctx.message.guild.emojis

    if not available_emojis:
        return text_content

    emoji_map = {emoji.name.lower(): str(emoji) for emoji in available_emojis}

    def replacer(match):
        emoji_name = match.group(1)
        # Only replace if it's a known custom emoji shortcode
        return emoji_map.get(emoji_name, match.group(0)) # Return original if not in map

    resolved_text = re.sub(EMOJI_PATTERN, replacer, text_content)
    return resolved_text