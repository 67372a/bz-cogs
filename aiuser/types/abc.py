import re
from abc import ABC
from datetime import datetime
from asyncio import Queue, Task

from openai import AsyncOpenAI
from redbot.core import Config, commands
from redbot.core.bot import Red

from aiuser.messages_list.entry import MessageEntry
from aiuser.utils.cache import Cache


# for other settings to use
@commands.group(aliases=["ai_user"])
@commands.guild_only()
async def aiuser(self, _):
    """Utilize OpenAI to reply to messages and images in approved channels"""
    pass


class CompositeMetaClass(type(commands.Cog), type(ABC)):
    pass


class MixinMeta(ABC):
    def __init__(self, *args):
        self.bot: Red
        self.config: Config
        self.cached_options: dict
        self.override_prompt_start_time: dict[int, datetime]
        self.forget_start_times: dict[int, datetime]
        self.cached_messages: Cache[int, MessageEntry]
        self.ignore_regex: dict[int, re.Pattern]
        self.channels_whitelist: dict[int, list[int]]
        self.openai_client: AsyncOpenAI
        self.optindefault: dict[int, bool]
        self.generating_channels: set[int]
        self.message_queues: dict[int, Queue]
        self.processing_tasks: dict[int, Task]
        self.backfill_anchors: dict[int, discord.Message]
        # History watermark per channel (HistoryWatermark) — string annotation
        # avoids a circular import with messages_list.messages
        self.context_front: Cache[int, "HistoryWatermark"]
        # Rolling prompt-cache hit stats per channel: [count, cached_sum, prompt_sum]
        self.cache_hit_stats: Cache[int, list]
        # Message-conversion pinning: (channel_id, message_id) ->
        # (edit_token, tuple[MessageEntry, ...])
        self.converted_history: Cache[tuple, tuple]
