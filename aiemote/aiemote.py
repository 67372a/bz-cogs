import asyncio
import json
import logging
import random
import re
from typing import Optional

import discord
import tiktoken
from emoji import EMOJI_DATA # type: ignore
from openai import AsyncOpenAI
from redbot.core import Config, checks, commands
from redbot.core.bot import Red
from redbot.core.utils.menus import start_adding_reactions
from redbot.core.utils.predicates import ReactionPredicate
from redbot.core.utils.views import SimpleMenu

logger = logging.getLogger("red.bz_cogs.aiemote")

DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class AIEmote(commands.Cog):
    """ Human-like Discord reacts to messages powered by LLMs. """

    MATCH_DISCORD_EMOJI_REGEX = r"<a?:[A-Za-z0-9]+:[0-9]+>"

    def __init__(self, bot):
        super().__init__()
        self.bot: Red = bot
        self.config = Config.get_conf(self, identifier=75406969)
        self.encoding = None
        self.aclient: Optional[AsyncOpenAI] = None

        self.llm_provider: str = "openai"
        self.llm_model: str = DEFAULT_LLM_MODEL
        self.openrouter_api_key: Optional[str] = None
        self.openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
        self.openrouter_referer: Optional[str] = None
        self.openrouter_title: Optional[str] = None

        default_global = {
            "percent": 50,
            "global_emojis": [
                {"description": "A happy face", "emoji": "😀"},
                {"description": "A sad face", "emoji": "😢"},
            ],
            "extra_instruction": "",
            "optin": [],
            "optout": [],
            "llm_provider": "openai",
            "llm_model": DEFAULT_LLM_MODEL,
            "openrouter_api_key": None,
            "openrouter_base_url": DEFAULT_OPENROUTER_BASE_URL,
            "openrouter_referer": None,
            "openrouter_title": None,
        }

        default_guild = {
            "server_emojis": [],
            "whitelist": [],
            "optin_by_default": False
        }

        self.config.register_guild(**default_guild)
        self.config.register_global(**default_global)

    async def cog_load(self):
        self.whitelist = {}
        all_guild_config = await self.config.all_guilds()
        for guild_id, config_data in all_guild_config.items():
            self.whitelist[guild_id] = config_data["whitelist"]

        global_config = await self.config.all()
        self.percent = global_config["percent"]
        self.optin_users = global_config["optin"]
        self.optout_users = global_config["optout"]
        self.llm_provider = global_config.get("llm_provider", "openai")
        self.llm_model = global_config.get("llm_model", DEFAULT_LLM_MODEL)
        self.openrouter_api_key = global_config.get("openrouter_api_key")
        self.openrouter_base_url = global_config.get("openrouter_base_url", DEFAULT_OPENROUTER_BASE_URL)
        self.openrouter_referer = global_config.get("openrouter_referer")
        self.openrouter_title = global_config.get("openrouter_title")

        asyncio.create_task(self.initialize_llm_client()) # Called without ctx

    @commands.Cog.listener()
    async def on_red_api_tokens_update(self, service_name, api_tokens):
        if service_name in ["openai", "openrouter"]:
            logger.info(f"API token updated for {service_name}, re-initializing LLM client.")
            asyncio.create_task(self.initialize_llm_client()) # Called without ctx

    async def initialize_llm_client(self, ctx: Optional[commands.Context] = None):
        self.aclient = None
        api_key = None
        base_url = None
        default_headers = None
        provider_name_for_msg = "" # For user-friendly messages

        if self.llm_provider == "openai":
            provider_name_for_msg = "OpenAI"
            api_key = (await self.bot.get_shared_api_tokens("openai")).get("api_key")
            if not api_key:
                # Construct clean_prefix part carefully
                prefix_str = ""
                if ctx and hasattr(ctx, 'clean_prefix') and ctx.clean_prefix:
                    prefix_str = ctx.clean_prefix
                elif hasattr(self.bot, ' όταν') and self.bot. όταν: # fallback to a common prefix if possible
                    prefix_str = self.bot. όταν[0] if isinstance(self.bot. όταν, list) else self.bot. όταν

                msg = f"{provider_name_for_msg} API key not set for `aiemote`. Please set it with `{prefix_str}set api openai api_key,API_KEY`"
                if ctx: # Only send message if ctx is provided (user-initiated command)
                    await ctx.send(msg)
                    logger.warning(msg) # Log as warning if user command failed due to this
                else: # Log as error if system-initiated and failing
                    logger.error(msg)
                return
        elif self.llm_provider == "openrouter":
            provider_name_for_msg = "OpenRouter"
            api_key = self.openrouter_api_key
            if not api_key:
                api_key = (await self.bot.get_shared_api_tokens("openrouter")).get("api_key")
            if not api_key:
                api_key = (await self.bot.get_shared_api_tokens("openai")).get("api_key")

            if not api_key:
                msg = f"{provider_name_for_msg} API key not set. Configure via owner commands, or shared tokens for 'openrouter' or 'openai'."
                if ctx:
                    await ctx.send(msg)
                    logger.warning(msg)
                else:
                    logger.error(msg)
                return

            base_url = self.openrouter_base_url or DEFAULT_OPENROUTER_BASE_URL
            default_headers = {}
            if self.openrouter_referer: default_headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_title: default_headers["X-Title"] = self.openrouter_title
            if not default_headers: default_headers = None
        else:
            msg = f"Invalid LLM provider: {self.llm_provider}. Choose 'openai' or 'openrouter'."
            if ctx:
                await ctx.send(msg)
                logger.warning(msg)
            else:
                logger.error(msg)
            return

        try:
            self.aclient = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
                timeout=20.0,
            )
            logger.info(f"LLM client initialized for provider: {self.llm_provider}, model: {self.llm_model}")
        except Exception as e:
            self.aclient = None # Ensure client is None on failure
            logger.exception(f"Failed to initialize AsyncOpenAI client for {self.llm_provider}:", exc_info=e)
            if ctx: await ctx.send(f"Error initializing LLM client for {self.llm_provider}: {e}")
            return # Do not proceed to tiktoken if client init failed

        try:
            self.encoding = tiktoken.encoding_for_model(self.llm_model)
        except Exception:
            logger.warning(f"Could not get tiktoken encoding for model {self.llm_model}. Falling back to cl100k_base.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message):
        if not self.aclient:
            await self.initialize_llm_client() # No ctx, will only log if key missing
            if not self.aclient:
                # initialize_llm_client would have logged the error.
                # No need to log again here, just means cog is not functional for reacting.
                return

        ctx: commands.Context = await self.bot.get_context(message)
        if not (await self.is_valid_to_react(ctx)):
            return
        if (self.percent < random.randint(0, 99)):
            return
        emoji = await self.pick_emoji(message)
        if emoji:
            try:
                await message.add_reaction(emoji)
            except discord.HTTPException as e:
                logger.warning(f"Failed to add reaction {emoji} in {message.guild.name if message.guild else 'DM'}: {e}")

    async def pick_emoji(self, message: discord.Message):
        if not self.aclient: # Should be caught by on_message_without_command, but as a safeguard
            logger.error("LLM client not available in pick_emoji. This should not happen if checks above are working.")
            return None

        options_str = "\n"
        emojis = (await self.config.guild(message.guild).server_emojis() if message.guild else []) or []
        emojis += (await self.config.global_emojis()) or []

        if not emojis:
            logger.warning(f"Skipping react! No valid emojis to use in {message.guild.name if message.guild else 'DM'}")
            return None

        for index, value in enumerate(emojis):
            options_str += f"{index}. {value['description']}\n"

        supports_json_mode = "gemini" in self.llm_model.lower()
        extra_instruction = await self.config.extra_instruction()
        guild_name_log = message.guild.name if message.guild else 'DM'

        request_kwargs = {
            "model": self.llm_model,
            "messages": [],
        }

        if supports_json_mode:
            system_prompt = (
                f"You are in a chat room. You will pick an emoji for the following message. "
                f"{extra_instruction} Here are your options: {options_str}"
                f"Your answer *MUST* be a valid JSON object with a single key 'emoji_index' "
                f"and an integer value corresponding to the chosen option, between 0 and {len(emojis)-1}. "
                f"For example: {{\"emoji_index\": 0}}"
            )
            request_kwargs["max_tokens"] = 15
            request_kwargs["response_format"] = {"type": "json_object"}
            # logger.debug(f"Using JSON mode for model {self.llm_model} in {guild_name_log}.") # Can be noisy
        else:
            system_prompt = (
                f"You are in a chat room. You will pick an emoji for the following message. "
                f"{extra_instruction} Here are your options: {options_str}"
                f"Your answer *MUST* be only an integer corresponding to the option, "
                f"between 0 and {len(emojis)-1}."
            )
            request_kwargs["max_tokens"] = 5
            # logger.debug(f"Using plain/regex mode for model {self.llm_model} in {guild_name_log}.") # Can be noisy

        content = f"{message.author.display_name} : {self.stringify_any_mentions(message)}"
        request_kwargs["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        try:
            response = await self.aclient.chat.completions.create(**request_kwargs)
        except Exception:
            logger.exception(f"Skipping react in {guild_name_log}! Failed to get response from LLM ({self.llm_model})")
            return None

        response_content = response.choices[0].message.content
        if not response_content:
            logger.warning(
                f"Skipping react in {guild_name_log}! LLM ({self.llm_model}) returned empty content."
            )
            return None
        response_content = response_content.strip()

        chosen_index: Optional[int] = None

        if supports_json_mode:
            try:
                data = json.loads(response_content)
                if isinstance(data, dict) and "emoji_index" in data:
                    if isinstance(data["emoji_index"], int):
                        chosen_index = data["emoji_index"]
                    else:
                        logger.warning(
                            f"Skipping react in {guild_name_log}! 'emoji_index' in JSON response from model {self.llm_model} was not an integer: '{response_content}'."
                        )
                else:
                    logger.warning(
                        f"Skipping react in {guild_name_log}! JSON response from model {self.llm_model} did not contain 'emoji_index' key or was not a dict: '{response_content}'."
                    )
            except json.JSONDecodeError:
                logger.warning(
                    f"Skipping react in {guild_name_log}! Failed to parse JSON response from model {self.llm_model} (JSON mode was active): '{response_content}'. Falling back to regex.")
                match = re.search(r'\d+', response_content)
                if match:
                    try:
                        chosen_index = int(match.group(0))
                    except ValueError:
                        logger.warning(f"Skipping react in {guild_name_log}! Could not parse int from model {self.llm_model} (JSON mode fallback regex) response: '{response_content}'")
                else:
                    logger.warning(f"Skipping react in {guild_name_log}! No parsable number in response from model {self.llm_model} (JSON mode fallback regex): '{response_content}'")

        if not supports_json_mode or (chosen_index is None and supports_json_mode):
            if not supports_json_mode:
                 logger.debug(f"Attempting regex parsing for model {self.llm_model} in {guild_name_log}.")
            match = re.search(r'\d+', response_content)
            if match:
                try:
                    potential_index = int(match.group(0))
                    if chosen_index is None:
                        chosen_index = potential_index
                except ValueError:
                    if not supports_json_mode:
                        logger.warning(
                            f"Skipping react in {guild_name_log}! Could not parse int from model {self.llm_model} (regex mode) response: '{response_content}'")
            else:
                if not supports_json_mode:
                    logger.warning(
                        f"Skipping react in {guild_name_log}! No parsable number in response from model {self.llm_model} (regex mode): '{response_content}'.")

        if chosen_index is not None:
            if 0 <= chosen_index < len(emojis):
                try:
                    partial_emoji = discord.PartialEmoji.from_str(emojis[chosen_index]["emoji"])
                    return partial_emoji
                except Exception as e:
                    logger.error(f"Failed to create PartialEmoji from '{emojis[chosen_index]['emoji']}': {e}")
                    return None
            else:
                logger.warning(
                    f"Skipping react in {guild_name_log}! Index {chosen_index} out of range (0-{len(emojis)-1}) for model {self.llm_model}. Response: '{response_content}'")
                return None
        else:
            return None

    async def is_valid_to_react(self, ctx: commands.Context):
        if ctx.guild is None or ctx.author.bot:
            return False

        if not self.aclient:
            await self.initialize_llm_client() # No ctx, will only log if key missing
            if not self.aclient:
                # initialize_llm_client would have logged the error.
                return False

        whitelist = self.whitelist.get(ctx.guild.id, [])
        if await self.bot.cog_disabled_in_guild(self, ctx.guild):
            return False

        if not isinstance(ctx.channel, (discord.TextChannel, discord.VoiceChannel, discord.StageChannel, discord.ForumChannel)):
             return False

        if isinstance(ctx.channel, discord.Thread):
            if ctx.channel.parent_id not in whitelist:
                return False
        elif ctx.channel.id not in whitelist:
            return False

        if await self.bot.ignored_channel_or_guild(ctx):
            return False
        if not await self.bot.allowed_by_whitelist_blacklist(ctx.author):
            return False

        if ctx.author.id in self.optout_users:
            return False
        if (ctx.author.id not in self.optin_users) and (not (await self.config.guild(ctx.guild).optin_by_default())):
            return False

        if not ctx.message.content or (ctx.message.attachments and len(ctx.message.attachments) > 0):
            return False

        if len(ctx.message.content) > 1500 or len(ctx.message.content) < 10:
            # logger.debug(f"Skipping message in {ctx.guild.name} with length {len(ctx.message.content)}") # Can be noisy
            return False

        return True

    def stringify_any_mentions(self, message: discord.Message) -> str:
        content = message.content
        mentions = message.mentions + message.role_mentions + message.channel_mentions

        if not mentions:
            return content

        for mentioned_item in mentions:
            if isinstance(mentioned_item, (discord.TextChannel, discord.VoiceChannel, discord.CategoryChannel, discord.StageChannel, discord.ForumChannel)):
                content = content.replace(mentioned_item.mention, f'#{mentioned_item.name}')
            elif isinstance(mentioned_item, discord.Role):
                content = content.replace(mentioned_item.mention, f'@{mentioned_item.name}')
            elif isinstance(mentioned_item, (discord.User, discord.Member)):
                content = content.replace(mentioned_item.mention, f'@{mentioned_item.display_name}')
        return content

    @commands.group(name="aiemote", alias=["ai_emote"])
    @checks.admin_or_permissions(manage_guild=True)
    async def aiemote(self, _):
        """ Totally not glorified sentiment analysis™

            Picks a reaction for a message using a configurable LLM.
            Default is gpt-4o-mini. Supports OpenAI and OpenRouter.
            Uses JSON mode for 'gemini' models if detected.

            To get started, please add a channel to the whitelist with:
            `[p]aiemote allow <#channel>`
        """
        pass

    # ... (rest of the commands remain the same, they correctly pass `ctx` to initialize_llm_client if needed)
    @aiemote.command(name="whitelist")
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_list(self, ctx: commands.Context):
        """ List all channels in the whitelist """
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        whitelist_ids = self.whitelist.get(ctx.guild.id, [])
        if not whitelist_ids:
            return await ctx.send("No channels in whitelist for this server.")
        
        channels_mentions = []
        for channel_id in whitelist_ids:
            channel = ctx.guild.get_channel(channel_id)
            if channel:
                channels_mentions.append(channel.mention)
            else:
                channels_mentions.append(f"`Unknown Channel (ID: {channel_id})`")

        embed = discord.Embed(title=f"Whitelist for {ctx.guild.name}", color=await ctx.embed_color())
        embed.add_field(name="Channels", value="\n".join(channels_mentions) if channels_mentions else "None")
        await ctx.send(embed=embed)

    @aiemote.command(name="allow", aliases=["add"])
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_add(self, ctx: commands.Context, channel: discord.TextChannel): # type: ignore
        """ Add a channel to the whitelist

            *Arguments*
            - `<channel>` The mention of channel
        """
        if not ctx.guild: return
        whitelist = self.whitelist.get(ctx.guild.id, [])
        if channel.id in whitelist:
            return await ctx.send("Channel already in whitelist")
        whitelist.append(channel.id)
        self.whitelist[ctx.guild.id] = whitelist
        await self.config.guild(ctx.guild).whitelist.set(whitelist)
        return await ctx.tick()

    @aiemote.command(name="remove", aliases=["rm"])
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_remove(self, ctx: commands.Context, channel: discord.TextChannel): # type: ignore
        """ Remove a channel from the whitelist

            *Arguments*
            - `<channel>` The mention of channel
        """
        if not ctx.guild: return
        whitelist = self.whitelist.get(ctx.guild.id, [])
        if channel.id not in whitelist:
            return await ctx.send("Channel not in whitelist")
        whitelist.remove(channel.id)
        self.whitelist[ctx.guild.id] = whitelist
        await self.config.guild(ctx.guild).whitelist.set(whitelist)
        return await ctx.tick()

    @aiemote.command(name="optinbydefault", alias=["optindefault"])
    @checks.admin_or_permissions(manage_guild=True)
    async def optin_by_default(self, ctx: commands.Context):
        """ Toggles whether users are opted in by default in this server

            This command is disabled for servers with more than 150 members by default.
        """
        if not ctx.guild: return
        if len(ctx.guild.members) > 150 and not await self.config.guild(ctx.guild).optin_by_default():
            return await ctx.send("You cannot enable this setting for servers with more than 150 members for privacy reasons. If you understand the implications, this can be changed in the bot's configuration files.")
        value = not await self.config.guild(ctx.guild).optin_by_default()
        await self.config.guild(ctx.guild).optin_by_default.set(value)
        embed = discord.Embed(
            title=f"Users are now opted {'in' if value else 'out'} by default in this server.",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @aiemote.command(name="optin")
    async def optin_user(self, ctx: commands.Context):
        """ Opt in of sending your message to the LLM (bot-wide)

            This will allow the bot to react to your messages.
        """
        if ctx.author.id in self.optin_users:
             if ctx.author.id not in self.optout_users:
                return await ctx.send("You are already opted in bot-wide.")

        if ctx.author.id not in self.optin_users:
            self.optin_users.append(ctx.author.id)
            await self.config.optin.set(self.optin_users)

        if ctx.author.id in self.optout_users:
            self.optout_users.remove(ctx.author.id)
            await self.config.optout.set(self.optout_users)

        await ctx.send("You are now opted in bot-wide. The bot may react to your messages in configured channels.")

    @aiemote.command(name="optout")
    async def optout_user(self, ctx: commands.Context):
        """ Opt out of sending your message to the LLM (bot-wide)

            The bot will no longer react to your messages.
        """
        if ctx.author.id in self.optout_users:
            return await ctx.send("You are already opted out bot-wide.")

        if ctx.author.id not in self.optout_users:
            self.optout_users.append(ctx.author.id)
            await self.config.optout.set(self.optout_users)

        if ctx.author.id in self.optin_users:
            self.optin_users.remove(ctx.author.id)
            await self.config.optin.set(self.optin_users)

        await ctx.send("You are now opted out bot-wide. The bot will no longer react to your messages.")

    @commands.group(name="aiemoteowner", alias=["aiemoteadmin"])
    @checks.is_owner()
    async def aiemote_owner(self, _):
        """ Owner only commands for aiemote """
        pass

    @aiemote_owner.command(name="instruction", aliases=["extra_instruction", "extra"])
    async def set_extra_instruction(self, ctx: commands.Context, *, instruction: Optional[str]):
        """ Add additional (prompting) instruction for the language model.

            Use without arguments to clear.
        """
        await self.config.extra_instruction.set(instruction or "")
        current_instruction = instruction or "None"
        await ctx.send(f"Extra instruction set to: `{current_instruction}`")

    async def check_valid_emoji(self, ctx: commands.Context, emoji_str: str):
        if emoji_str in EMOJI_DATA:
            return True
        match = re.fullmatch(self.MATCH_DISCORD_EMOJI_REGEX, emoji_str)
        if not match:
            await ctx.send(f"'{emoji_str}' is not a valid standard emoji or custom Discord emoji format.")
            return False
        try:
            partial_emoji = discord.PartialEmoji.from_str(emoji_str)
            if partial_emoji.id:
                found_emoji = discord.utils.get(self.bot.emojis, id=partial_emoji.id)
                if not found_emoji:
                    await ctx.send(f"I cannot use the custom emoji {emoji_str}. It might be from a server I'm not in, or I don't have permissions.")
                    return False
        except ValueError:
            await ctx.send(f"Could not parse '{emoji_str}' as an emoji.")
            return False
        return True


    async def _add_emoji_to_list(self, ctx: commands.Context, emoji_list: list, emoji_str: str, description: str):
        if any(item["emoji"] == emoji_str for item in emoji_list):
            await ctx.send(f"Emoji {emoji_str} already in list.")
            return False
        emoji_list.append({"description": description, "emoji": emoji_str})
        return True

    async def _remove_emoji_from_list(self, ctx: commands.Context, emoji_list: list, emoji_str: str):
        index_to_remove = -1
        for i, item in enumerate(emoji_list):
            if item["emoji"] == emoji_str:
                index_to_remove = i
                break
        if index_to_remove == -1:
            await ctx.send(f"Emoji {emoji_str} not found in list.")
            return False
        del emoji_list[index_to_remove]
        return True

    @aiemote_owner.command(name="addglobal")
    async def add_global_emoji(self, ctx: commands.Context, emoji: str, *, description: str):
        """ Add an emoji to the global list. """
        if not await self.check_valid_emoji(ctx, emoji):
            return
        emojis = await self.config.global_emojis()
        if await self._add_emoji_to_list(ctx, emojis, emoji, description):
            await self.config.global_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="rmglobal", aliases=["removeglobal"])
    async def remove_global_emoji(self, ctx: commands.Context, emoji: str):
        """ Remove an emoji from the global list. """
        emojis = await self.config.global_emojis()
        if await self._remove_emoji_from_list(ctx, emojis, emoji):
            await self.config.global_emojis.set(emojis)
            await ctx.tick()

    async def create_emoji_embed_pages(self, ctx: commands.Context, title: str, emojis: list):
        embeds = []
        chunk_size = 8

        if not emojis:
            embed = discord.Embed(title=title, description="None", color=await ctx.embed_color())
            embeds.append(embed)
            return embeds

        for i in range(0, len(emojis), chunk_size):
            embed = discord.Embed(title=title, color=await ctx.embed_color())
            chunk = emojis[i: i + chunk_size]
            for item in chunk:
                embed.add_field(name=item["emoji"], value=item["description"][:1020], inline=False)
            embeds.append(embed)

        if len(embeds) > 1:
            for i, page_embed in enumerate(embeds):
                page_embed.set_footer(text=f"Page {i+1} of {len(embeds)}")
        return embeds

    @aiemote_owner.command(name="config", aliases=["settings", "listconf"])
    async def list_all_config(self, ctx: commands.Context):
        """ List all current settings and emoji lists. """
        settings_embed = discord.Embed(title="AIEmote Main Settings", color=await ctx.embed_color())
        settings_embed.add_field(name="Percent Chance", value=f"{self.percent}%", inline=False)
        settings_embed.add_field(name="Additional Instruction", value=f"`{await self.config.extra_instruction() or 'None'}`", inline=False)
        settings_embed.add_field(name="LLM Provider", value=f"`{self.llm_provider}`", inline=True)
        settings_embed.add_field(name="LLM Model", value=f"`{self.llm_model}`", inline=True)
        if self.llm_provider == "openrouter":
            settings_embed.add_field(name="OpenRouter URL", value=f"`{self.openrouter_base_url}`", inline=False)
            key_status = "Set (hidden)" if self.openrouter_api_key or (await self.bot.get_shared_api_tokens("openrouter")).get("api_key") or (await self.bot.get_shared_api_tokens("openai")).get("api_key") else "Not Set"
            settings_embed.add_field(name="OpenRouter Key", value=key_status, inline=True)
            settings_embed.add_field(name="OpenRouter Referer", value=f"`{self.openrouter_referer or 'Not Set'}`", inline=True)
            settings_embed.add_field(name="OpenRouter Title", value=f"`{self.openrouter_title or 'Not Set'}`", inline=True)

        await ctx.send(embed=settings_embed)

        global_emojis_list = await self.config.global_emojis()
        global_embed_pages = await self.create_emoji_embed_pages(ctx, "Global Emojis", global_emojis_list)
        if global_embed_pages:
            await SimpleMenu(global_embed_pages).start(ctx)

        if ctx.guild:
            server_emojis_list = await self.config.guild(ctx.guild).server_emojis()
            server_embed_pages = await self.create_emoji_embed_pages(ctx, f"Server Emojis for {ctx.guild.name}", server_emojis_list)
            if server_embed_pages:
                await SimpleMenu(server_embed_pages).start(ctx)
        else:
            await ctx.send("Server-specific emoji list can only be shown in a server context.")


    @aiemote_owner.command(name="reset")
    async def reset_all_settings(self, ctx: commands.Context):
        """ Reset *all* global and guild settings to default. """
        embed = discord.Embed(
            title="⚠️ Are you sure? ⚠️",
            description="This will reset ALL global settings (including LLM config, API keys stored in this cog, emojis, etc.) and ALL per-server settings (whitelists, server emojis, opt-in defaults) to their original defaults. This action is irreversible.",
            color=discord.Color.red())
        confirm_msg = await ctx.send(embed=embed)
        start_adding_reactions(confirm_msg, ReactionPredicate.YES_OR_NO_EMOJIS)
        try:
            pred = ReactionPredicate.yes_or_no(confirm_msg, ctx.author)
            await ctx.bot.wait_for("reaction_add", timeout=30.0, check=pred)
        except asyncio.TimeoutError:
            return await confirm_msg.edit(content="Reset cancelled (timed out).", embed=None)

        if pred.result is True:
            await self.config.clear_all_guilds()
            await self.config.clear_all_globals()
            await self.cog_load() # Reloads defaults from config definition
            await self.initialize_llm_client(ctx) # Attempt re-init with (now default) settings, passing ctx for feedback if still issue
            await confirm_msg.edit(content="All AIEmote settings have been reset to default.", embed=None)
        else:
            await confirm_msg.edit(content="Reset cancelled.", embed=None)

    @aiemote_owner.command(name="addserver")
    async def add_server_emoji(self, ctx: commands.Context, emoji: str, *, description: str):
        """ Add an emoji to the current server's list. """
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        if not await self.check_valid_emoji(ctx, emoji):
            return
        emojis = await self.config.guild(ctx.guild).server_emojis()
        if await self._add_emoji_to_list(ctx, emojis, emoji, description):
            await self.config.guild(ctx.guild).server_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="rmserver", aliases=["removeserver"])
    async def remove_server_emoji(self, ctx: commands.Context, emoji: str):
        """ Remove an emoji from the current server's list. """
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        emojis = await self.config.guild(ctx.guild).server_emojis()
        if await self._remove_emoji_from_list(ctx, emojis, emoji):
            await self.config.guild(ctx.guild).server_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="percent")
    async def set_percent(self, ctx: commands.Context, percent_chance: int):
        """ Set the chance (0-100) the bot will react. """
        if not (0 <= percent_chance <= 100):
            return await ctx.send("Percent chance must be between 0 and 100.")
        self.percent = percent_chance
        await self.config.percent.set(percent_chance)
        await ctx.send(f"Reaction chance set to {percent_chance}%.")

    @aiemote_owner.command(name="setprovider")
    async def set_llm_provider(self, ctx: commands.Context, provider: str):
        """Set the LLM provider. Options: 'openai', 'openrouter'."""
        provider = provider.lower()
        if provider not in ["openai", "openrouter"]:
            return await ctx.send("Invalid provider. Use 'openai' or 'openrouter'.")
        self.llm_provider = provider
        await self.config.llm_provider.set(provider)
        await self.initialize_llm_client(ctx) # Pass ctx for feedback on this direct command
        await ctx.send(f"LLM provider set to `{provider}`. Client re-initialized.")

    @aiemote_owner.command(name="setmodel")
    async def set_llm_model(self, ctx: commands.Context, *, model_name: str):
        """Set the LLM model name (e.g., 'gpt-4o-mini', 'google/gemini-pro')."""
        self.llm_model = model_name
        await self.config.llm_model.set(model_name)
        await self.initialize_llm_client(ctx) # Pass ctx
        await ctx.send(f"LLM model set to `{model_name}`. Client re-initialized, TikToken encoding updated.")

    @aiemote_owner.command(name="setopenrouterkey")
    async def set_openrouter_api_key(self, ctx: commands.Context, api_key: Optional[str] = None):
        """Set the dedicated OpenRouter API key for this cog.
           Clears the dedicated key if no argument is provided.
           Will fallback to shared 'openrouter' or 'openai' keys if not set.
        """
        self.openrouter_api_key = api_key
        await self.config.openrouter_api_key.set(api_key)
        await self.initialize_llm_client(ctx) # Pass ctx
        if api_key:
            await ctx.send("OpenRouter API key set for AIEmote.")
        else:
            await ctx.send("Dedicated OpenRouter API key for AIEmote cleared. Will use fallbacks if configured.")

    @aiemote_owner.command(name="setopenrouterurl")
    async def set_openrouter_base_url(self, ctx: commands.Context, url: Optional[str] = None):
        """Set the OpenRouter base URL.
           Clears to default if no argument. Default: https://openrouter.ai/api/v1
        """
        final_url = url or DEFAULT_OPENROUTER_BASE_URL
        self.openrouter_base_url = final_url
        await self.config.openrouter_base_url.set(final_url)
        await self.initialize_llm_client(ctx) # Pass ctx
        await ctx.send(f"OpenRouter base URL set to `{final_url}`.")

    @aiemote_owner.command(name="setopenrouterheaders")
    async def set_openrouter_headers(self, ctx: commands.Context, referer: Optional[str] = None, title: Optional[str] = None):
        """Set custom HTTP-Referer and X-Title headers for OpenRouter.
           Provide "clear" for a header to remove it. No args clears both.
        """
        if referer and referer.lower() == "clear": referer = None
        if title and title.lower() == "clear": title = None

        self.openrouter_referer = referer
        self.openrouter_title = title
        await self.config.openrouter_referer.set(referer)
        await self.config.openrouter_title.set(title)
        await self.initialize_llm_client(ctx) # Pass ctx
        await ctx.send(f"OpenRouter headers updated: Referer=`{referer or 'Not Set'}`, Title=`{title or 'Not Set'}`.")