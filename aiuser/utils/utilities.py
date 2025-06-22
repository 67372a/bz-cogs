import asyncio
import functools
import importlib
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine
from cachetools import TTLCache

import discord
from discord import Message
from openai import AsyncOpenAI
from redbot.core import Config, commands

from aiuser.config.constants import OPENROUTER_URL, YOUTUBE_URL_PATTERN
from aiuser.functions.tool_call import ToolCall

logger = logging.getLogger("red.bz_cogs.aiuser")

# Cache for guild emojis with a ten minute TTL
EMOJI_CACHE_TTL = 600
# Cache structure: {guild_id: {emoji_name_lowercase: str(emoji_object)}}
emoji_cache = TTLCache(maxsize=100, ttl=EMOJI_CACHE_TTL)

def to_thread(timeout=300):
    def decorator(func: Callable) -> Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func, *args, **kwargs), timeout
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
