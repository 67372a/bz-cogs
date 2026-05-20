import json
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional, Union
import re
import time as _time

import discord
import tiktoken
from discord import Message
from redbot.core import commands

from aiuser.config.defaults import DEFAULT_PROMPT
from aiuser.config.models import MODELS_LIMITS
from aiuser.messages_list.converter.converter import MessageConverter
from aiuser.messages_list.entry import MessageEntry
from aiuser.messages_list.opt_view import OptView
from aiuser.types.abc import MixinMeta
from aiuser.types.enums import ScanImageMode
from aiuser.utils.utilities import format_variables, format_stable_variables, build_dynamic_context_message
from aiuser.config.constants import XML_SYSTEM_PROMPT_APPENDIX, OPENROUTER_CITATION_INSTRUCTIONS

logger = logging.getLogger("red.bz_cogs.aiuser")

OPTIN_EMBED_TITLE = ":information_source: AI User Opt-In / Opt-Out"

# When a channel was last processed within this many seconds, the
# token limit is expanded to the model's full context size so that
# already-cached messages are not prematurely dropped.
CACHE_WINDOW_SECONDS = 180  # 3 minutes
THOUGHTS_EMBED_TITLE_REGEX = re.compile(r'^.*\'s Thoughts$')


async def create_messages_list(
    cog: MixinMeta, ctx: commands.Context, prompt: str = None, history: bool = True
):
    """to manage messages in ChatML format"""
    thread = MessagesList(cog, ctx)

    # Check for a backfill anchor for this channel
    backfill_anchor = cog.backfill_anchors.pop(ctx.channel.id, None)
    if backfill_anchor:
        await thread._init(prompt=prompt, skip_init_msg=True)
        if history:
            await thread.add_backfill_history(backfill_anchor)
        thread._append_dynamic_context()
        return thread

    await thread._init(prompt=prompt)
    if history:
        await thread.add_history()
    thread._append_dynamic_context()
    return thread


class MessagesList:
    def __init__(
        self,
        cog: MixinMeta,
        ctx: commands.Context,
    ):
        self._cog = cog
        self.bot = cog.bot
        self.config = cog.config
        self.ctx = ctx
        self.converter = MessageConverter(cog, ctx)
        self.init_message = ctx.message
        self.guild = ctx.guild
        self.ignore_regex = cog.ignore_regex.get(self.guild.id, None)
        self.start_time = cog.override_prompt_start_time.get(
            self.guild.id)
        self.messages: List[MessageEntry] = []
        self.messages_ids = set()
        self.tokens = 0
        self.model = None
        self.can_reply = True
        self.prefill: Optional[str] = None
        self._dynamic_context: Optional[str] = None
        self._raw_persona: Optional[str] = None
        self._cache_window_active: bool = False

    def __len__(self):
        return len(self.messages)

    def __repr__(self) -> str:
        return json.dumps(self.get_json(), indent=4)

    async def _init(self, prompt=None, skip_init_msg=False):
        self.model = await self.config.guild(self.guild).model()
        self.token_limit = await self.config.guild(self.guild).custom_model_tokens_limit() or self._get_token_limit(self.model)
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Check if this channel was recently processed — if so, expand
        # limits to load more history for a longer shared cache prefix.
        channel_id = self.ctx.channel.id
        last_at = self._cog.last_response_at.get(channel_id)
        if last_at is not None and (_time.monotonic() - last_at) < CACHE_WINDOW_SECONDS:
            self._cache_window_active = True
            # Expand token limit to the model's full context (no buffer)
            model_limit = self._get_token_limit_raw(self.model)
            if model_limit > self.token_limit:
                logger.info(
                    f"Cache window active for channel {channel_id} — "
                    f"expanding token limit {self.token_limit} → {model_limit}"
                )
                self.token_limit = model_limit

        if not prompt and not skip_init_msg:
            await self.add_msg(self.init_message)

        # 1. Get the user-defined persona (e.g., "You are a helpful assistant...")
        raw_persona = prompt or await self._pick_prompt()
        self._raw_persona = raw_persona

        # 2. Format only stable variables for the system prompt.
        #    Dynamic variables (time, author, random) are deferred to a
        #    trailing user message to preserve Gemini implicit caching and
        #    OpenRouter provider sticky routing.
        formatted_persona = await format_stable_variables(self.ctx, raw_persona)

        # 3. Combine Persona + XML Protocol
        # We add a double newline to separate the personality from the technical instructions
        final_system_prompt = f"{formatted_persona}\n\n{XML_SYSTEM_PROMPT_APPENDIX}"

        # 3b. Append OpenRouter citation instructions if web search/fetch tools are enabled
        search_enabled = await self.config.guild(self.guild).openrouter_web_search_enabled()
        fetch_enabled = await self.config.guild(self.guild).openrouter_web_fetch_enabled()
        if search_enabled or fetch_enabled:
            final_system_prompt += f"\n\n{OPENROUTER_CITATION_INSTRUCTIONS}"

        self.prefill = await self._pick_prefill()

        # Account for prefill tokens without adding to history
        if self.prefill:
            await self._add_tokens(self.prefill)

        # 4. Add the combined system prompt
        await self.add_system(final_system_prompt)

        # 5. Build dynamic context (time, author, etc.) for later insertion
        #    as a trailing user message after history is loaded.
        self._dynamic_context = await build_dynamic_context_message(
            self.ctx, raw_persona
        )

        if await self._check_if_inital_img():
            self.model = await self.config.guild(self.guild).scan_images_model()

    def _append_dynamic_context(self):
        """Append dynamic per-request context as a trailing user message.

        This is called after history loading to place time-varying and
        user-varying variables (currenttime, authorname, etc.) at the end
        of the message list rather than in the system prompt.  This
        preserves Gemini implicit caching and OpenRouter provider sticky
        routing by keeping the system prompt immutable.

        The context is added as a ``user`` role message, per OpenRouter
        guidance: "move dynamic content into a later user message."
        """
        if not self._dynamic_context:
            return
        entry = MessageEntry("user", self._dynamic_context)
        self.messages.append(entry)
        # Count tokens for the dynamic context
        if isinstance(self._dynamic_context, str):
            self.tokens += len(self._encoding.encode(self._dynamic_context, disallowed_special=()))

    async def _check_if_inital_img(self) -> bool:
        if (
            self.ctx.interaction
            or not await self.config.guild(self.guild).scan_images()
            or await self.config.guild(self.guild).scan_images_mode() != ScanImageMode.LLM.value
        ):
            return False
        if self.init_message.attachments and self.init_message.attachments[0].content_type.startswith('image/'):
            return True
        elif self.init_message.reference:
            ref = self.init_message.reference
            replied = ref.cached_message or await self.bot.get_channel(ref.channel_id).fetch_message(ref.message_id)
            return replied.attachments and replied.attachments[0].content_type.startswith('image/')
        else:
            return False

    async def _pick_prompt(self):
        author = self.init_message.author
        role_prompt = None

        # Roles are ordered from lowest to highest, check highest first
        for role in reversed(author.roles):
            if role.id in (await self.config.all_roles()):
                prompt = await self.config.role(role).custom_text_prompt()
                if prompt:
                    role_prompt = prompt
                    break

        return (await self.config.member(self.init_message.author).custom_text_prompt()
                or role_prompt
                or await self.config.channel(self.init_message.channel).custom_text_prompt()
                or await self.config.guild(self.guild).custom_text_prompt()
                or await self.config.custom_text_prompt()
                or DEFAULT_PROMPT)

    async def _pick_prefill(self) -> Optional[str]:
        """ Pick the most specific prefill, member -> role -> channel -> guild """
        author = self.init_message.author

        prefill = await self.config.member(author).prefill_prompt()
        if prefill:
            return prefill

        # Roles are ordered from lowest to highest, check highest first
        for role in reversed(author.roles):
            role_prefill = await self.config.role(role).prefill_prompt()
            if role_prefill:
                return role_prefill

        channel_prefill = await self.config.channel(self.init_message.channel).prefill_prompt()
        if channel_prefill:
            return channel_prefill

        guild_prefill = await self.config.guild(self.guild).prefill_prompt()
        if guild_prefill:
            return guild_prefill

        return None

    async def check_if_add(self, message: Message, force: bool = False):
        if self.tokens > self.token_limit:
            return False

        if message.id in self.messages_ids and not force:
            logger.debug(
                f"Skipping duplicate message in {message.guild.name} when creating context"
            )
            return False

        if message.embeds and any(embed.title and THOUGHTS_EMBED_TITLE_REGEX.search(embed.title) for embed in message.embeds):
            return False

        if self.ignore_regex and self.ignore_regex.search(message.content):
            return False
        if not await self.bot.allowed_by_whitelist_blacklist(message.author):
            return False
        if message.author.id in await self.config.optout():
            return False
        if (
            (not message.author.id == self.bot.user.id)
            and not message.author.bot
            and message.author.id not in await self.config.optin()
            and not await self.config.guild(self.guild).optin_by_default()
        ):
            return False

        return True

    async def add_msg(self, message: Message, index: int = None, force: bool = False):
        if not await self.check_if_add(message, force):
            return

        converted = await self.converter.convert(message)

        if not converted:
            return

        for entry in converted:
            if self.tokens > self.token_limit:
                return

            self.messages.insert(index or 0, entry)
            self.messages_ids.add(message.id)

            if isinstance(entry.content, list):
                for item in entry.content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        await self._add_tokens(item.get("text"))
                    elif item.get("type") == "image_url":
                        self.tokens += 756  # TODO: calculate actual image token cost
            else:
                await self._add_tokens(entry.content)

        # TODO: proper reply chaining
        if message.reference and isinstance(message.reference.resolved, discord.Message) and message.author.id != self.bot.user.id:
            await self.add_msg(message.reference.resolved, index=0)

    async def add_system(self, content: str, index: int = None):
        if self.tokens > self.token_limit:
            return
        entry = MessageEntry("system", content)
        self.messages.insert(index or 0, entry)
        await self._add_tokens(content)

    async def add_assistant(self, content: str, index: int = None, tool_calls: list = [], reasoning_details: list = None):
        if self.tokens > self.token_limit:
            return
        entry = MessageEntry("assistant", content, tool_calls=tool_calls, reasoning_details=reasoning_details)
        self.messages.insert(index or 0, entry)
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    await self._add_tokens(item.get("text"))
                elif item.get("type") == "image_url":
                    self.tokens += 756  # matches estimate in add_msg()
        else:
            await self._add_tokens(content)

    async def add_tool_result(self, content: Union[str, list], tool_call_id: int, name: str = None, index: int = None):
        """Add a tool result message. Content may be a plain string or a list
        of content parts (text + image_url) for multimodal function responses."""
        if self.tokens > self.token_limit:
            return
        entry = MessageEntry("tool", content, tool_call_id=tool_call_id, name=name)
        self.messages.insert(index or 0, entry)
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    await self._add_tokens(item.get("text"))
                elif item.get("type") == "image_url":
                    self.tokens += 756  # matches estimate in add_assistant()
        else:
            await self._add_tokens(content)

    async def add_history(self):
        limit = await self.config.guild(self.guild).messages_backread()
        max_seconds_gap = await self.config.guild(self.guild).messages_backread_seconds()

        start_time: datetime = (
            self.start_time - timedelta(seconds=1) if self.start_time else None
        )

        before_msgs, after_msgs = await self._get_past_messages(limit, start_time)
        
        all_past_messages = list(reversed(after_msgs)) + [self.init_message] + before_msgs
        users = await self._get_unopted_users(all_past_messages[:10])

        await self._process_past_messages(before_msgs, after_msgs, max_seconds_gap)

        if users and not await self.config.guild(self.guild).optin_disable_embed():
            if (random.random() <= 0.33) or (len(users) > 3):
                await self._send_optin_embed(users)

    async def add_backfill_history(self, anchor: discord.Message):
        """Load messages from the anchor message up to the trigger message,
        respecting backread count, time gap, and token limits.

        Context order:
            [system prompt]
            [anchor message]
            [messages between anchor and trigger, chronological]
            [trigger message]
        """
        limit = await self.config.guild(self.guild).messages_backread()
        max_seconds_gap = await self.config.guild(self.guild).messages_backread_seconds()

        # Add the anchor message at index 1 (right after system prompt)
        await self.add_msg(anchor, index=1)
        if self.tokens > self.token_limit:
            logger.debug(f"{self.tokens} tokens used - token limit hit after anchor for backfill {anchor.id}")
            return

        # Fetch messages between anchor (exclusive) and trigger (exclusive), oldest first
        mid_msgs = [
            message
            async for message in self.init_message.channel.history(
                limit=limit,
                after=anchor,
                before=self.init_message,
                oldest_first=True,
            )
        ]

        # Process mid-messages in chronological order, validating time gap and limits
        last_msg = anchor
        for msg in mid_msgs:
            if not await self._is_valid_time_gap(last_msg, msg, max_seconds_gap):
                logger.debug(f"Time gap exceeded between {last_msg.id} and {msg.id}, stopping backfill")
                break
            if self.tokens > self.token_limit:
                logger.debug(f"{self.tokens} tokens used - nearing limit, stopping backfill for message {self.init_message.id}")
                break
            if (msg.author.id == self.bot.user.id) and (msg.embeds and msg.embeds[0].title == OPTIN_EMBED_TITLE):
                continue
            if msg.embeds and msg.embeds[0].title and THOUGHTS_EMBED_TITLE_REGEX.search(msg.embeds[0].title):
                continue
            # Insert each message at the end (after anchor, before trigger)
            await self.add_msg(msg, index=len(self.messages))
            last_msg = msg

        # Add the trigger message at the end
        await self.add_msg(self.init_message, index=len(self.messages))

        # Ensure the conversation history ends with a user message
        if self.messages and self.messages[-1].role == "assistant":
            self.messages.append(MessageEntry("user", "System Note: Please continue or respond to the latest context."))

        logger.info(f"Backfill complete: {len(self.messages)} messages ({self.tokens} tokens) from anchor {anchor.id} to trigger {self.init_message.id}")

    async def _get_past_messages(self, limit, start_time):
        before_msgs = [
            message
            async for message in self.init_message.channel.history(
                limit=limit,
                before=self.init_message,
                after=start_time,
                oldest_first=False,
            )
        ]
        after_msgs = [
            message
            async for message in self.init_message.channel.history(
                limit=limit,
                after=self.init_message,
                oldest_first=True,
            )
        ]
        return before_msgs, after_msgs

    async def _get_unopted_users(self, messages):
        users = set()

        if await self.config.guild(self.guild).optin_by_default():
            return users

        for message in messages:
            if (
                (message.author.id != self.bot.user.id)
                and (not message.author.bot)
                and (message.author.id not in await self.config.optin())
                and (message.author.id not in await self.config.optout())
            ):
                users.add(message.author)

        return users

    async def _process_past_messages(self, before_msgs, after_msgs, max_seconds_gap):
        last_msg = self.init_message
        for msg in before_msgs:
            if not await self._is_valid_time_gap(last_msg, msg, max_seconds_gap):
                break
            if self.tokens > self.token_limit:
                logger.debug(f"{self.tokens} tokens used - nearing limit, stopping context creation for message {self.init_message.id}")
                break
            if (msg.author.id == self.bot.user.id) and (msg.embeds and msg.embeds[0].title == OPTIN_EMBED_TITLE):
                continue
            # Ignore reasoning
            if msg.embeds and msg.embeds[0].title and THOUGHTS_EMBED_TITLE_REGEX.search(msg.embeds[0].title):
                continue
            await self.add_msg(msg, index=1)
            last_msg = msg

        last_msg = self.init_message
        for msg in after_msgs:
            if not await self._is_valid_time_gap(last_msg, msg, max_seconds_gap):
                break
            if self.tokens > self.token_limit:
                logger.debug(f"{self.tokens} tokens used - nearing limit, stopping context creation for message {self.init_message.id}")
                break
            if (msg.author.id == self.bot.user.id) and (msg.embeds and msg.embeds[0].title == OPTIN_EMBED_TITLE):
                continue
            # Ignore reasoning
            if msg.embeds and msg.embeds[0].title and THOUGHTS_EMBED_TITLE_REGEX.search(msg.embeds[0].title):
                continue
            await self.add_msg(msg, index=len(self.messages))
            last_msg = msg

        # Ensure the conversation history ends with a user message to prevent API strict-sequence errors (e.g. UNEXPECTED_TOOL_CALL)
        if self.messages and self.messages[-1].role == "assistant":
            self.messages.append(MessageEntry("user", "System Note: Please continue or respond to the latest context."))

    async def _send_optin_embed(self, users):
        users = ", ".join([user.mention for user in users])
        embed = discord.Embed(
            title=OPTIN_EMBED_TITLE,
            color=await self.bot.get_embed_color(self.init_message),
        )
        view = OptView(self.config)
        embed.description = f"{users}\nPlease choose whether to allow a subset of your Discord messages from any server with the bot, to be sent to OpenAI or an external party.\nThis will allow the bot to reply to your messages or use your messages.\nThis message will disappear if all current chatters have made a choice."
        await self.init_message.channel.send(embed=embed, view=view)

    def _last_system_prefix_index(self) -> int:
        """Return the index of the last contiguous system message at the
        start of the message list, or -1 if there are none."""
        last = -1
        for i, msg in enumerate(self.messages):
            if msg.role == "system":
                last = i
            else:
                break
        return last

    def get_json(self, annotations_for_assistant: list = None):
        # Determine which system message (if any) should carry the
        # cache_control breakpoint — only the last contiguous system
        # message at the start of the conversation.
        cache_idx = self._last_system_prefix_index()

        messages_as_dict = []
        for i, message in enumerate(self.messages):
            msg_dict = {
                "role": message.role,
                "content": message.content,
            }

            # Add cache_control to the stable system instruction for prompt
            # caching (Anthropic, Alibaba, etc.).  Only the last system
            # message in the leading contiguous block gets the breakpoint —
            # this is the largest cacheable prefix and avoids wasting one
            # of the provider's limited breakpoint slots on smaller
            # fragments.  Providers that don't recognise the field (OpenAI,
            # Gemini with implicit caching) simply ignore it.
            if i == cache_idx and isinstance(message.content, str):
                msg_dict["content"] = [
                    {
                        "type": "text",
                        "text": message.content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            if hasattr(message, 'name') and message.name:
                msg_dict["name"] = message.name

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc
                    for tc in message.tool_calls
                ]

            if hasattr(message, 'tool_call_id') and message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
             
            if hasattr(message, 'reasoning_details') and message.reasoning_details:
                msg_dict["reasoning_details"] = message.reasoning_details

            # Inject PDF annotations into the first assistant message after a PDF user message
            if annotations_for_assistant and message.role == "assistant" and not message.tool_calls:
                msg_dict["annotations"] = annotations_for_assistant
                # Only inject once
                annotations_for_assistant = None

            messages_as_dict.append(msg_dict)

        if self.prefill:
            prefill_message = {
                "role": "assistant",
                "content": self.prefill
            }
            messages_as_dict.append(prefill_message)

        return messages_as_dict

    async def _add_tokens(self, content):
        if not self._encoding:
            await self._initialize_encoding()
        content = str(content)
        tokens = self._encoding.encode(content, disallowed_special=())
        self.tokens += len(tokens)

    @staticmethod
    def _get_token_limit(model) -> int:
        limit = 7000

        model = model.split("/")[-1].split(":")[0]
        if model in MODELS_LIMITS:
            limit = MODELS_LIMITS.get(model, limit) - 1000 # 1000 token buffer

        return limit

    @staticmethod
    def _get_token_limit_raw(model) -> int:
        """Return the model's full token limit without any buffer.

        Used during the cache window to maximize the prefix length
        for Gemini implicit caching.
        """
        limit = 7000
        model = model.split("/")[-1].split(":")[0]
        if model in MODELS_LIMITS:
            limit = MODELS_LIMITS.get(model, limit)
        return limit

    @staticmethod
    async def _is_valid_time_gap(message: discord.Message, next_message: discord.Message, max_seconds_gap: int) -> bool:
        seconds_diff = abs(message.created_at - next_message.created_at).total_seconds()
        if seconds_diff > max_seconds_gap:
            return False
        return True