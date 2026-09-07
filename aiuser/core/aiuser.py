import logging
import asyncio
import re
from datetime import datetime

import discord
from openai import AsyncOpenAI
from redbot.core import Config, app_commands, commands
from redbot.core.bot import Red

from aiuser.config.defaults import (
    DEFAULT_CHANNEL,
    DEFAULT_GLOBAL,
    DEFAULT_GUILD,
    DEFAULT_MEMBER,
    DEFAULT_ROLE,
)
from aiuser.core.handlers import handle_message, handle_slash_command
from aiuser.core.random_message_task import RandomMessageTask
from aiuser.dashboard.base import DashboardIntegration
from aiuser.messages_list.entry import MessageEntry
from aiuser.messages_list.messages import HistoryWatermark
from aiuser.settings.base import Settings
from aiuser.response.dispatcher import dispatch_response
from aiuser.types.abc import CompositeMetaClass
from aiuser.utils.cache import Cache

from .openai_utils import setup_openai_client

logger = logging.getLogger("red.bz_cogs.aiuser")
logging.getLogger("httpcore").setLevel(logging.WARNING)


class AIUser(
    DashboardIntegration,
    Settings,
    RandomMessageTask,
    commands.Cog,
    metaclass=CompositeMetaClass,
):
    """
        Human-like Discord interactions powered by OpenAI (or compatible endpoints) for messages (and images).
    """

    def __init__(self, bot):
        super().__init__()
        self.bot: Red = bot
        self.config = Config.get_conf(self, identifier=754070)
        self.openai_client: AsyncOpenAI = None
        # cached options
        self.optindefault: dict[int, bool] = {}
        self.channels_whitelist: dict[int, list[int]] = {}
        self.ignore_regex: dict[int, re.Pattern] = {}
        self.override_prompt_start_time: dict[int, datetime] = {}
        # Per-channel forget timestamps (channel_id -> datetime), set by [p]aiuser forget
        self.forget_start_times: Cache[int, datetime] = Cache(limit=500)
        self.cached_messages: Cache[int, MessageEntry] = Cache(limit=100)
        self.message_queues: dict[int, asyncio.Queue] = {}
        self.processing_tasks: dict[int, asyncio.Task] = {}
        # Bounded LRU caches to prevent unbounded memory growth over long uptimes
        # History watermark per channel (see MessagesList.add_history): pins the
        # oldest message included in context so the front of the LLM request
        # prefix stays stable across requests, preserving prompt-cache hits.
        self.context_front: Cache[int, HistoryWatermark] = Cache(limit=2000)
        # Per-channel rolling prompt-cache hit statistics:
        # channel_id -> [request_count, cached_tokens_sum, prompt_tokens_sum]
        self.cache_hit_stats: Cache[int, list] = Cache(limit=1000)
        # Message-conversion pinning (see MessageConverter.convert): the first
        # serialization of a message is reused until it is edited, so history
        # re-serializations (embed unfurls, reply-reference cache state) cannot
        # diverge the LLM request prefix.  Only stores string-content entries;
        # (channel_id, message_id) -> (edit_token, tuple[MessageEntry, ...])
        self.converted_history: Cache[tuple, tuple] = Cache(limit=2000)
        # PDF annotation cache: (channel_id, originating_user_message_id) -> list[dict]
        self.pdf_annotations: Cache[tuple[int, int], list[dict]] = Cache(limit=200)
        # Backfill anchors: channel_id -> anchor message for the next trigger
        self.backfill_anchors: Cache[int, discord.Message] = Cache(limit=500)

        self.config.register_member(**DEFAULT_MEMBER)
        self.config.register_role(**DEFAULT_ROLE)
        self.config.register_channel(**DEFAULT_CHANNEL)
        self.config.register_guild(**DEFAULT_GUILD)
        self.config.register_global(**DEFAULT_GLOBAL)

    async def cog_load(self):
        self.openai_client = await setup_openai_client(self.bot, self.config)

        all_config = await self.config.all_guilds()

        for guild_id, config in all_config.items():
            self.optindefault[guild_id] = config["optin_by_default"]
            self.channels_whitelist[guild_id] = config["channels_whitelist"]
            pattern = config["ignore_regex"]

            self.ignore_regex[guild_id] = re.compile(pattern) if pattern else None

        if logger.isEnabledFor(logging.DEBUG):
            # for development
            test_guild = 744802856074346556
            self.override_prompt_start_time[test_guild] = datetime.now()

        # Migrate legacy function names to new generic web_* system
        from aiuser.settings.functions import FunctionCallingSettings
        await FunctionCallingSettings._migrate_legacy_function_names(self)

        self.random_message_trigger.start()

    async def cog_unload(self):
        if self.openai_client:
            await self.openai_client.close()
        self.random_message_trigger.cancel()
        # Cancel any in-flight per-channel queue processors so they don't
        # keep running (and calling dispatch_response with a closed client)
        # after the cog has been unloaded.
        for task in self.processing_tasks.values():
            task.cancel()
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        self.processing_tasks.clear()
        self.message_queues.clear()

    async def red_delete_data_for_user(self, *, requester, user_id: int):
        for guild in self.bot.guilds:
            member = guild.get_member(user_id)
            if member:
                await self.config.member(member).clear()
            else:
                # The user may have left the guild; clear any stored member
                # data by ID so ex-member data does not persist.
                await self.config.member_from_ids(guild.id, user_id).clear()

        # Remove the user from the global opt-in / opt-out lists
        async with self.config.optin() as optin:
            if user_id in optin:
                optin.remove(user_id)
        async with self.config.optout() as optout:
            if user_id in optout:
                optout.remove(user_id)

        # TODO: remove user messages from cache instead of clearing the whole cache
        self.cached_messages = Cache(limit=100)
        self.converted_history = Cache(limit=2000)

    @commands.Cog.listener()
    async def on_red_api_tokens_update(self, service_name, _):
        if service_name in ["openai", "openrouter"]:
            old_client = self.openai_client
            self.openai_client = await setup_openai_client(self.bot, self.config)
            # Close the previous client so its httpx connection pool isn't leaked
            if old_client and old_client is not self.openai_client:
                try:
                    await old_client.close()
                except Exception:
                    logger.debug("Failed to close previous OpenAI client", exc_info=True)

    @app_commands.command(name="chat")
    @app_commands.describe(text="The prompt you want to send to the AI.")
    @app_commands.checks.cooldown(1, 30)
    @app_commands.checks.cooldown(1, 5, key=None)
    async def slash_command(
        self,
        inter: discord.Interaction,
        *,
        text: app_commands.Range[str, 1, 2000],
    ):
        """Talk directly to this bot's AI. Ask it anything you want!"""
        await handle_slash_command(self, inter, text)

    @commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message):
        await handle_message(self, message)

    async def queue_response(self, ctx: commands.Context, messages_list=None):
        """
        Adds a response request to the channel's queue.
        If no processor is running for this channel, start one.
        """
        channel_id = ctx.channel.id
        
        if channel_id not in self.message_queues:
            self.message_queues[channel_id] = asyncio.Queue()
            
        await self.message_queues[channel_id].put((ctx, messages_list))

        if channel_id not in self.processing_tasks or self.processing_tasks[channel_id].done():
            self.processing_tasks[channel_id] = asyncio.create_task(self.process_queue(channel_id))

    async def process_queue(self, channel_id: int):
        """
        Sequentially processes requests in the channel's queue.
        """
        queue = self.message_queues[channel_id]
        try:
            while True:
                try:
                    ctx, messages_list = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await dispatch_response(self, ctx, messages_list)
                except Exception:
                    logger.exception(f"Error processing queue for channel {channel_id}")
                # Small buffer between messages to ensure order and prevent rate-limit bursts
                await asyncio.sleep(1)
        finally:
            # Atomically (no awaits between the check and the cleanup) decide
            # whether to shut down or keep processing.  This closes the race
            # where an item enqueued after the loop exited but before the task
            # was removed would otherwise be stranded with no processor.
            self.processing_tasks.pop(channel_id, None)
            if not queue.empty():
                self.processing_tasks[channel_id] = asyncio.create_task(self.process_queue(channel_id))
            else:
                self.message_queues.pop(channel_id, None)