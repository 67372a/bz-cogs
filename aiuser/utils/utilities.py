import asyncio
import functools
import importlib
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine
from cachetools import TTLCache
import re

import discord
from discord import Message
from openai import AsyncOpenAI
from redbot.core import Config, commands

from aiuser.config.constants import OPENROUTER_URL, YOUTUBE_URL_PATTERN
from aiuser.functions.tool_call import ToolCall

logger = logging.getLogger("red.bz_cogs.aiuser")

EMOJI_PATTERN = re.compile(r"(?:<a?)?:([A-Za-z0-9_]{2,32}):(?:[0-9]+>)?")

BACKTICK_PATTERN = re.compile(r'(\\*)`')

MULTI_NEWLINE_PATTERN = re.compile(r'(?:([ \t]|<br>|\\n)*(?:\r\n|\r|\n)){2,}')

# Cache for guild emojis with a ten minute TTL
EMOJI_CACHE_TTL = 600
# Cache structure: {guild_id: {emoji_name_lowercase: str(emoji_object)}}
emoji_cache = TTLCache(maxsize=100, ttl=EMOJI_CACHE_TTL)

def to_thread(timeout=300):
    def decorator(func: Callable) -> Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            func_call = functools.partial(func, *args, **kwargs)
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func_call), timeout
            )
            return result

        return wrapper

    return decorator

async def get_guild_emoji_map(ctx: commands.Context) -> dict[str, str]:
    """
    Retrieves a mapping of lowercase emoji names to their string representation for a guild.
    Results are cached to avoid repeated API calls and reliably fetched to ensure completeness.
    """
    guild_id = ctx.guild.id
    if guild_id in emoji_cache:
        return emoji_cache[guild_id]

    logger.info(f"Emoji cache miss. Fetching emojis for guild {guild_id}")
    try:
        # Reliably fetch all emojis from the guild to ensure the cache is complete
        guild_emojis = await ctx.guild.fetch_emojis()
        sorted_emojis = sorted(guild_emojis, key=lambda e: e.name.lower())
        emoji_map = {emoji.name.lower(): str(emoji) for emoji in sorted_emojis}
        emoji_cache[guild_id] = emoji_map
        return emoji_map
    except Exception:
        logger.warning(f"Failed to fetch emojis for guild {guild_id}. Using potentially incomplete data from guild.emojis.", exc_info=True)
        # Fallback to the potentially incomplete internal cache if fetch fails
        return {emoji.name.lower(): str(emoji) for emoji in ctx.guild.emojis}

async def format_variables(ctx: commands.Context, text: str):
    """
    Insert supported variables into string if they are present
    """
    botname = ctx.message.guild.me.name or ctx.bot.user.name
    botdisplayname = ctx.message.guild.me.display_name or ctx.bot.user.display_name
    app_info = await ctx.bot.application_info()
    botowner = app_info.owner.name
    authorname = ctx.message.author.name
    authordisplayname = ctx.message.author.display_name
    authortoprole = ctx.message.author.top_role.name
    authormention = ctx.message.author.mention

    servername = ctx.guild.name
    channelname = ctx.message.channel.name
    currentdate = datetime.today().strftime("%Y/%m/%d")
    currentweekday = datetime.today().strftime("%A")
    currenttime = datetime.today().strftime("%H:%M")

    randomnumber = random.randint(0, 100)

    if isinstance(ctx.message.channel, discord.Thread):
        channeltopic = ctx.message.channel.parent.topic
    else:
        channeltopic = ctx.message.channel.topic

    serveremojis = [f":{e}:" for e in await get_guild_emoji_map(ctx)]
    serveremojis = ' '.join(serveremojis)

    try:
        res = text.format(
            botname=botname,
            botdisplayname=botdisplayname,
            botowner=botowner,
            authorname=authorname,
            authordisplayname=authordisplayname,
            authortoprole=authortoprole,
            authormention=authormention,
            servername=servername,
            serveremojis=serveremojis,
            channelname=channelname,
            channeltopic=channeltopic,
            currentdate=currentdate,
            currentweekday=currentweekday,
            currenttime=currenttime,
            randomnumber=randomnumber,
        )
        return res
    except KeyError:
        logger.exception("Invalid key in message", exc_info=True)
        return text


# Variables that are stable per-channel and rarely change.
# These are safe to include in the system prompt without breaking
# Gemini implicit caching or OpenRouter sticky routing.
_STABLE_VARIABLES = frozenset({
    'botname', 'botdisplayname', 'botowner',
    'servername', 'channelname',
    'serveremojis', 'channeltopic',
})

# Variables that change per-request (time, user, randomness).
# These must NOT be in the system prompt — they are appended as
# a trailing user message to preserve the stable prefix.
_DYNAMIC_VARIABLES = frozenset({
    'currenttime', 'currentdate', 'currentweekday',
    'randomnumber',
    'authorname', 'authordisplayname', 'authortoprole', 'authormention',
})


async def format_stable_variables(ctx: commands.Context, text: str) -> str:
    """Substitute only stable (per-channel) variables into *text*.

    This is used for the system prompt so that its content is identical
    across requests in the same channel, enabling Gemini implicit caching
    and OpenRouter provider sticky routing.

    Dynamic variables like ``{currenttime}``, ``{authorname}``, etc. are
    left as literal ``{…}`` placeholders — callers should pass the result
    through ``format_variables()`` or ``build_dynamic_context_message()``
    if those placeholders still need to be resolved for a different part
    of the prompt.
    """
    botname = ctx.message.guild.me.name or ctx.bot.user.name
    botdisplayname = ctx.message.guild.me.display_name or ctx.bot.user.display_name
    app_info = await ctx.bot.application_info()
    botowner = app_info.owner.name
    servername = ctx.guild.name
    channelname = ctx.message.channel.name

    if isinstance(ctx.message.channel, discord.Thread):
        channeltopic = ctx.message.channel.parent.topic or ""
    else:
        channeltopic = ctx.message.channel.topic or ""

    try:
        serveremojis = ' '.join(f":{e}:" for e in await get_guild_emoji_map(ctx))
    except Exception:
        serveremojis = ""

    stable_kwargs = dict(
        botname=botname,
        botdisplayname=botdisplayname,
        botowner=botowner,
        servername=servername,
        channelname=channelname,
        serveremojis=serveremojis,
        channeltopic=channeltopic,
    )

    try:
        return text.format(**stable_kwargs)
    except KeyError:
        # If the text contains dynamic placeholders that we intentionally
        # skip here, Python's str.format() will raise KeyError.  Use a
        # safe formatter that leaves unknown placeholders intact.
        import string
        class _SafeFormatter(string.Formatter):
            def get_field(self, field_name, *args, **kwargs):
                try:
                    return super().get_field(field_name, *args, **kwargs)
                except (KeyError, AttributeError):
                    return '{' + field_name + '}', field_name
            def format_field(self, value, format_spec):
                try:
                    return super().format_field(value, format_spec)
                except (ValueError, KeyError):
                    return '{' + str(value) + '}'
        return _SafeFormatter().format(text, **stable_kwargs)


async def build_dynamic_context_message(ctx: commands.Context, text: str = None) -> str | None:
    """Build a concise context string with all dynamic (per-request) values.

    Returns a short, machine-readable summary of dynamic variables that
    should be appended as a **user** message at the end of the message
    list — per OpenRouter guidance, dynamic content must be in a later
    user message rather than the system prompt.

    If *text* is provided, only builds context if the text references
    at least one dynamic variable.  If *text* is ``None``, always builds
    context (used when the caller wants an unconditional context message).

    Returns ``None`` if no dynamic context is needed.
    """
    # If a template is given and it doesn't use any dynamic variables,
    # there is no need for a trailing context message.
    if text is not None:
        has_dynamic = any('{' + var + '}' in text for var in _DYNAMIC_VARIABLES)
        if not has_dynamic:
            return None

    authorname = ctx.message.author.name
    authordisplayname = ctx.message.author.display_name
    authortoprole = ctx.message.author.top_role.name
    authormention = ctx.message.author.mention
    currentdate = datetime.today().strftime("%Y/%m/%d")
    currentweekday = datetime.today().strftime("%A")
    currenttime = datetime.today().strftime("%H:%M")
    randomnumber = random.randint(0, 100)

    parts = [
        f"date={currentdate}",
        f"day={currentweekday}",
        f"time={currenttime}",
        f"author={authordisplayname} (@{authorname})",
        f"author_top_role={authortoprole}",
        f"author_mention={authormention}",
        f"random={randomnumber}",
    ]

    return " | ".join(parts)


def is_embed_valid(message: Message):
    if (
        (len(message.embeds) == 0)
        or (not message.embeds[0].title)
        or (not message.embeds[0].description)
    ):
        return False
    return True


def contains_youtube_link(content):
    match = YOUTUBE_URL_PATTERN.search(content)
    return bool(match)


def is_using_openai_endpoint(client: AsyncOpenAI):
    return str(client.base_url).startswith("https://api.openai.com/")


def is_using_openrouter_endpoint(client: AsyncOpenAI):
    return str(client.base_url).startswith(OPENROUTER_URL)


async def get_enabled_tools(config: Config, ctx: commands.Context) -> list:
    functions_dir = Path(__file__).parent.parent / 'functions'

    for item in functions_dir.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            try:
                importlib.import_module(f'aiuser.functions.{item.name}.tool_call')
            except ImportError:
                continue

    enabled_tools = await config.guild(ctx.guild).function_calling_functions()
    tool_classes = {cls.function_name: cls for cls in ToolCall.__subclasses__()}

    return [tool_classes[name](config=config, ctx=ctx)
            for name in enabled_tools
            if name in tool_classes]

async def resolve_emojis_for_discord(ctx: commands.Context, text_content: str) -> str:
    """
    Resolves :emoji_name: shortcodes in a string to their full Discord format using a cache.

    Args:
        ctx: The command context, used to identify the guild.
        text_content: The string potentially containing emoji shortcodes.

    Returns:
        The string with emoji shortcodes replaced.
    """
    emoji_map = await get_guild_emoji_map(ctx)

    if not emoji_map:
        return text_content

    def replacer(match):
        # Match is case-insensitive by lowercasing, aligning with the keys in our map
        emoji_name = match.group(1)
        if emoji_name:
            emoji_name = emoji_name.lower()

        # Return the full emoji string if found, otherwise return the original text (e.g., :thinking:)
        return emoji_map.get(emoji_name, match.group(0))

    return EMOJI_PATTERN.sub(replacer, text_content)

def escape_unescaped_backticks(text: str) -> str:
    """
    Escapes backticks in a string, but only if they are not already escaped.

    This handles cases with multiple preceding backslashes correctly.
    - `backtick` -> `\`backtick`
    - `\`backtick` -> `\`backtick` (no change)
    - `\\`backtick` -> `\\\`backtick` (the backtick was not escaped)
    - `\\\`backtick` -> `\\\`backtick` (no change)
    """
    # This function is called for every match of the regex.
    def replacer(match):
        # The first group captures all the backslashes before the backtick.
        backslashes = match.group(1)
        
        # If the number of backslashes is even, the backtick is not escaped.
        # So, we add an escape slash.
        if len(backslashes) % 2 == 0:
            return backslashes + r'\`'
        # If the number is odd, the backtick is already escaped.
        # So, we return the original match.
        else:
            return match.group(0)

    # The regex finds any number of backslashes (group 1) followed by a backtick.
    # The 'r' prefix is important for raw strings.
    return BACKTICK_PATTERN.sub(replacer, text)

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