import asyncio
import json
import logging
import random
import re
from typing import Optional
# Add these imports at the top of aiemote.py
from discord.ui import Modal, TextInput, View, Button, Select # Make sure Select is imported
from discord import ButtonStyle, Interaction # Interaction for type hinting

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

EMOJIS_PER_PAGE = 5 # Keep it small for clarity in the select menu
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-preview"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class AIEmote(commands.Cog):
    """ Human-like Discord reacts to messages powered by LLMs. """

    MATCH_DISCORD_EMOJI_REGEX = r"<a?:[A-Za-z0-9]+:[0-9]+>"

    def __init__(self, bot):
        super().__init__()
        self.bot: Red = bot
        self.config = Config.get_conf(self, identifier=75406969)
        self.aclient: Optional[AsyncOpenAI] = None

        self.llm_provider: str = "openrouter"
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
            "llm_provider": "openrouter",
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
            logger.debug(f"Using JSON mode for model {self.llm_model} in {guild_name_log}.")
        else:
            system_prompt = (
                f"You are in a chat room. You will pick an emoji for the following message. "
                f"{extra_instruction} Here are your options: {options_str}"
                f"Your answer *MUST* be only an integer corresponding to the option, "
                f"between 0 and {len(emojis)-1}."
            )
            request_kwargs["max_tokens"] = 5
            logger.debug(f"Using plain/regex mode for model {self.llm_model} in {guild_name_log}.")

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
            logger.debug(f"Skipping message in {ctx.guild.name} with length {len(ctx.message.content)}")
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

    def _is_discord_emoji(self, emoji_str: str) -> bool:
        return bool(re.fullmatch(self.MATCH_DISCORD_EMOJI_REGEX, emoji_str))

    # Modified create_emoji_embed_pages for the view
    def create_emoji_embed_pages_for_view(self, title: str, emojis: list, chunk_size: int = EMOJIS_PER_PAGE):
        embeds = []
        if not emojis:
            embed = discord.Embed(title=title, description="No emojis configured yet.", color=discord.Color.blue())
            embeds.append(embed)
            return embeds

        for i in range(0, len(emojis), chunk_size):
            embed = discord.Embed(title=title, color=discord.Color.blue()) # Consistent color
            chunk = emojis[i:i + chunk_size]
            for item_idx, item in enumerate(chunk):
                display_emoji = item["emoji"]
                # For custom emojis, discord.utils.get may be needed if we want to display it as emoji in field name
                # but PartialEmoji used in SelectOption handles this. String is fine for field name.
                embed.add_field(name=f"{display_emoji}", value=item["description"][:1020], inline=False)
            
            if len(emojis) > chunk_size : # Only add footer if there are multiple pages potentially
                 embed.set_footer(text=f"Page {(i // chunk_size) + 1} of {-(-len(emojis) // chunk_size)}") # Ceiling division for total pages
            embeds.append(embed)
        return embeds

    # New helper for view to validate emoji using interaction for ephemeral feedback
    async def check_valid_emoji_interaction(self, interaction: Interaction, emoji_str: str) -> bool:
        if emoji_str in EMOJI_DATA:
            return True
        match = re.fullmatch(self.MATCH_DISCORD_EMOJI_REGEX, emoji_str)
        if not match:
            await interaction.response.send_message(f"'{emoji_str}' is not a valid standard emoji or custom Discord emoji format.",ephemeral=True)
            return False
        try:
            partial_emoji = discord.PartialEmoji.from_str(emoji_str)
            if partial_emoji.id:
                found_emoji = discord.utils.get(self.bot.emojis, id=partial_emoji.id)
                if not found_emoji:
                    await interaction.response.send_message(f"I cannot use the custom emoji {emoji_str}. It might be from a server I'm not in, or I don't have permissions.",ephemeral=True)
                    return False
        except ValueError:
            await interaction.response.send_message(f"Could not parse '{emoji_str}' as an emoji.",ephemeral=True)
            return False
        return True

    # Helpers to get/save emoji lists (slightly refactored for clarity)
    async def _get_emoji_list(self, config_group, emoji_list_attr_name: str) -> list:
        if emoji_list_attr_name == "global_emojis":
            return await self.config.global_emojis()
        elif emoji_list_attr_name == "server_emojis" and config_group is not self.config : # Make sure config_group is a guild config
            return await config_group.server_emojis()
        return []

    async def _save_emoji_list(self, config_group, emoji_list_attr_name: str, new_list: list):
        if emoji_list_attr_name == "global_emojis":
            await self.config.global_emojis.set(new_list)
        elif emoji_list_attr_name == "server_emojis" and config_group is not self.config:
            await config_group.server_emojis.set(new_list)

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

    @aiemote_owner.group(name="manageemojis", invoke_without_command=True)
    async def manageemojis_owner(self, ctx: commands.Context):
        """Manage global and server emojis interactively."""
        await ctx.send_help()

    @manageemojis_owner.command(name="global")
    @checks.is_owner() # Ensure owner check
    async def manage_global_emojis(self, ctx: commands.Context):
        """Manage global emojis interactively."""
        view = ManageEmojisView(self, ctx, self.config, "global_emojis")
        await view.start()

    @aiemote.group(name="manageemojis", invoke_without_command=True) # New group under aiemote for guild admins
    @checks.admin_or_permissions(manage_guild=True)
    async def manageemojis_guild(self, ctx: commands.Context):
        """Manage server-specific emojis interactively."""
        await ctx.send_help()

    @manageemojis_guild.command(name="server")
    @checks.admin_or_permissions(manage_guild=True)
    async def manage_server_emojis(self, ctx: commands.Context):
        """Manage this server's specific emojis interactively."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        guild_config = self.config.guild(ctx.guild)
        view = ManageEmojisView(self, ctx, guild_config, "server_emojis")
        await view.start()

    # --- Deprecate old commands ---
    @aiemote_owner.command(name="addglobal", hidden=True)
    async def deprecated_add_global_emoji(self, ctx: commands.Context, emoji: str, *, description: str):
        """(Deprecated) Use `[p]aiemoteowner manageemojis global` instead."""
        await ctx.send(f"This command is deprecated. Please use `{ctx.prefix}aiemoteowner manageemojis global`.")

    @aiemote_owner.command(name="rmglobal", aliases=["removeglobal"], hidden=True)
    async def deprecated_remove_global_emoji(self, ctx: commands.Context, emoji: str):
        """(Deprecated) Use `[p]aiemoteowner manageemojis global` instead."""
        await ctx.send(f"This command is deprecated. Please use `{ctx.prefix}aiemoteowner manageemojis global`.")

    @aiemote_owner.command(name="addserver", hidden=True) # Was owner only, could be admin. New one is admin.
    async def deprecated_add_server_emoji(self, ctx: commands.Context, emoji: str, *, description: str):
        """(Deprecated) Use `[p]aiemote manageemojis server` instead."""
        await ctx.send(f"This command is deprecated. Please use `{ctx.prefix}aiemote manageemojis server`.")

    @aiemote_owner.command(name="rmserver", aliases=["removeserver"], hidden=True)
    async def deprecated_remove_server_emoji(self, ctx: commands.Context, emoji: str):
        """(Deprecated) Use `[p]aiemote manageemojis server` instead."""
        await ctx.send(f"This command is deprecated. Please use `{ctx.prefix}aiemote manageemojis server`.")

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

class AddEmojiModal(Modal, title="Add New Emoji"):
    emoji_str_input = TextInput( # Renamed to avoid conflict with a potential variable
        label="Emoji",
        placeholder="e.g., 👍 or <:custom_emoji:123456789012345678>",
        style=discord.TextStyle.short,
        required=True,
        max_length=100 # Emojis aren't that long
    )
    description_input = TextInput( # Renamed
        label="Description for LLM",
        placeholder="A brief description of when this emoji is appropriate",
        style=discord.TextStyle.long,
        required=True,
        max_length=200,
    )

    def __init__(self, manager_view: View): # manager_view is the instance of ManageEmojisView
        super().__init__(timeout=300) # 5 minutes timeout
        self.manager_view = manager_view

    async def on_submit(self, interaction: Interaction):
        # Defer here as validation and config update might take time
        # await interaction.response.defer(ephemeral=True) # defer before processing
        # The actual processing will send its own ephemeral message
        await self.manager_view.process_add_emoji(
            interaction,
            self.emoji_str_input.value,
            self.description_input.value
        )

    async def on_error(self, interaction: Interaction, error: Exception):
        logger.exception("Error in AddEmojiModal:", exc_info=error)
        await interaction.response.send_message(
            "An unexpected error occurred. Please try again later.", ephemeral=True
        )

class ConfirmRemoveButton(Button):
    def __init__(self, manager_view: View, emoji_to_remove: dict):
        super().__init__(label=f"Yes, remove {emoji_to_remove['emoji']}", style=ButtonStyle.danger, custom_id=f"confirm_remove_{emoji_to_remove['emoji']}")
        self.manager_view = manager_view
        self.emoji_to_remove = emoji_to_remove

    async def callback(self, interaction: Interaction):
        await self.manager_view.process_remove_emoji_confirmed(interaction, self.emoji_to_remove)

class CancelRemoveButton(Button):
    def __init__(self, manager_view: View):
        super().__init__(label="Cancel", style=ButtonStyle.secondary, custom_id="cancel_remove_action")
        self.manager_view = manager_view

    async def callback(self, interaction: Interaction):
        # Simply re-render the view without the confirmation buttons for removal
        self.view.remove_item(self.manager_view.confirm_remove_button_instance)
        self.view.remove_item(self.manager_view.cancel_remove_button_instance)
        self.manager_view.confirm_remove_button_instance = None
        self.manager_view.cancel_remove_button_instance = None
        await self.manager_view.update_message(interaction, "Removal cancelled.")

EMOJIS_PER_PAGE = 5 # Keep it small for clarity in the select menu

class ManageEmojisView(View):
    def __init__(self, cog_instance: AIEmote, ctx: commands.Context, config_group, emoji_list_attr_name: str):
        super().__init__(timeout=300) # View times out after 5 minutes of inactivity
        self.cog = cog_instance
        self.ctx = ctx
        self.author = ctx.author
        self.config_group = config_group # This is either self.config (global) or guild_config
        self.emoji_list_attr_name = emoji_list_attr_name # "global_emojis" or "server_emojis"
        
        self.emojis: list = []
        self.embed_pages: list[discord.Embed] = []
        self.current_page_index: int = 0
        self.message: Optional[discord.Message] = None

        self.remove_select_menu: Optional[Select] = None
        self.confirm_remove_button_instance: Optional[ConfirmRemoveButton] = None
        self.cancel_remove_button_instance: Optional[CancelRemoveButton] = None


    async def interaction_check(self, interaction: Interaction) -> bool:
        if interaction.user.id == self.author.id:
            return True
        await interaction.response.send_message("You are not allowed to interact with this menu.", ephemeral=True)
        return False

    async def on_timeout(self):
        if self.message:
            try:
                await self.message.edit(content="This emoji management menu has timed out.", view=None, embed=None)
            except discord.NotFound:
                pass # Message might have been deleted
        self.stop()

    async def start(self):
        await self._load_emojis()
        self._build_embed_pages()
        self._add_initial_buttons()
        embed_to_show = self.embed_pages[0] if self.embed_pages else self._get_empty_embed()
        self.message = await self.ctx.send(embed=embed_to_show, view=self)

    async def _load_emojis(self):
        self.emojis = await self.cog._get_emoji_list(self.config_group, self.emoji_list_attr_name)

    def _get_empty_embed(self) -> discord.Embed:
        title = "Global Emojis Management" if self.emoji_list_attr_name == "global_emojis" else "Server Emojis Management"
        return discord.Embed(title=title, description="No emojis configured yet.", color=discord.Color.blue())

    def _build_embed_pages(self):
        title = "Global Emojis Management" if self.emoji_list_attr_name == "global_emojis" else f"Server Emojis for {self.ctx.guild.name}"
        # Use the cog's existing method, but pass a dummy context if it expects one for color
        # Or adapt create_emoji_embed_pages to not strictly need ctx for color
        self.embed_pages = self.cog.create_emoji_embed_pages_for_view(title, self.emojis, EMOJIS_PER_PAGE) # New helper needed
        if not self.embed_pages: # Ensure there's at least one page (empty state)
            self.embed_pages.append(self._get_empty_embed())
        self.current_page_index = min(self.current_page_index, len(self.embed_pages) -1) # Adjust if list shrinks

    def _add_initial_buttons(self):
        self.clear_items() # Clear any existing buttons before re-adding

        # Pagination buttons
        prev_button = Button(label="⬅️ Prev", style=ButtonStyle.secondary, custom_id="prev_page_emojis", disabled=self.current_page_index == 0)
        prev_button.callback = self.go_to_prev_page
        self.add_item(prev_button)

        next_button = Button(label="Next ➡️", style=ButtonStyle.secondary, custom_id="next_page_emojis", disabled=self.current_page_index >= len(self.embed_pages) - 1)
        next_button.callback = self.go_to_next_page
        self.add_item(next_button)
        
        # Action buttons
        add_button = Button(label="✨ Add Emoji", style=ButtonStyle.success, custom_id="add_new_emoji")
        add_button.callback = self.show_add_emoji_modal
        self.add_item(add_button)

        remove_button = Button(label="➖ Show Remove Options", style=ButtonStyle.danger, custom_id="toggle_remove_emoji_select", disabled=not self.emojis)
        remove_button.callback = self.toggle_remove_select
        self.add_item(remove_button)

        close_button = Button(label="❌ Close", style=ButtonStyle.grey, custom_id="close_emoji_menu")
        close_button.callback = self.close_menu
        self.add_item(close_button)

        # If a select menu or confirm/cancel buttons were active, re-add them
        if self.remove_select_menu:
            self.add_item(self.remove_select_menu)
        if self.confirm_remove_button_instance and self.cancel_remove_button_instance:
            self.add_item(self.confirm_remove_button_instance)
            self.add_item(self.cancel_remove_button_instance)


    async def update_message(self, interaction: Optional[Interaction] = None, ephemeral_feedback: Optional[str] = None):
        if not self.message: return

        self._build_embed_pages() # Rebuild embeds in case list changed
        self._add_initial_buttons() # Rebuild buttons to update their states (e.g., prev/next, remove select)

        current_embed = self.embed_pages[self.current_page_index] if self.embed_pages else self._get_empty_embed()
        
        if interaction: # If called from an interaction, use its response
            if interaction.response.is_done():
                 await interaction.followup.edit_message(self.message.id, embed=current_embed, view=self)
            else:
                await interaction.response.edit_message(embed=current_embed, view=self)
            if ephemeral_feedback:
                await interaction.followup.send(ephemeral_feedback, ephemeral=True) # Send feedback after edit
        else: # If called without interaction (e.g. initial start), edit self.message
            await self.message.edit(embed=current_embed, view=self)


    async def go_to_prev_page(self, interaction: Interaction):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            # If remove select was active, remove it as page changed
            if self.remove_select_menu:
                self.remove_item(self.remove_select_menu)
                self.remove_select_menu = None
            await self.update_message(interaction)

    async def go_to_next_page(self, interaction: Interaction):
        if self.current_page_index < len(self.embed_pages) - 1:
            self.current_page_index += 1
            if self.remove_select_menu:
                self.remove_item(self.remove_select_menu)
                self.remove_select_menu = None
            await self.update_message(interaction)

    async def show_add_emoji_modal(self, interaction: Interaction):
        modal = AddEmojiModal(manager_view=self)
        await interaction.response.send_modal(modal)

    async def process_add_emoji(self, interaction: Interaction, emoji_str: str, description: str):
        # await interaction.response.defer(ephemeral=True, thinking=True) # Defer before heavy lifting
        
        is_valid = await self.cog.check_valid_emoji_interaction(interaction, emoji_str) # New helper needed
        if not is_valid:
            # check_valid_emoji_interaction should send ephemeral message on failure
            return

        if any(item["emoji"] == emoji_str for item in self.emojis):
            await interaction.response.send_message(f"Emoji {emoji_str} is already in the list.", ephemeral=True)
            return

        self.emojis.append({"description": description, "emoji": emoji_str})
        await self.cog._save_emoji_list(self.config_group, self.emoji_list_attr_name, self.emojis)
        
        feedback = f"Emoji {emoji_str} added successfully!"
        # We need to edit the main message and then send ephemeral feedback
        # The update_message will handle the edit. If called with interaction, it handles response.
        # However, this current interaction is from a MODAL, so its response is for the modal itself.
        # We need to ensure the modal interaction is acked, then edit the original view message.
        await interaction.response.send_message(feedback, ephemeral=True) # Ack the modal interaction
        await self.update_message() # Update the view message (no interaction needed here as modal already responded)


    async def toggle_remove_select(self, interaction: Interaction):
        # If confirm/cancel buttons are present from a previous select, remove them
        if self.confirm_remove_button_instance:
            self.remove_item(self.confirm_remove_button_instance)
            self.confirm_remove_button_instance = None
        if self.cancel_remove_button_instance:
            self.remove_item(self.cancel_remove_button_instance)
            self.cancel_remove_button_instance = None

        if self.remove_select_menu: # If select is already there, remove it (toggle off)
            self.remove_item(self.remove_select_menu)
            self.remove_select_menu = None
            await self.update_message(interaction, "Emoji removal options hidden.")
        else: # Add the select menu
            current_page_emojis = self._get_emojis_on_current_page()
            if not current_page_emojis:
                await interaction.response.send_message("No emojis on this page to remove.", ephemeral=True)
                return

            options = [
                discord.SelectOption(label=f"{item['emoji']} - {item['description'][:50]}", value=item["emoji"], emoji=discord.PartialEmoji.from_str(item["emoji"]) if self.cog._is_discord_emoji(item["emoji"]) else None)
                for item in current_page_emojis
            ]
            self.remove_select_menu = Select(
                placeholder="Choose emoji(s) to remove...", # Multi-select could be an option but complicates things. Let's do single for now.
                options=options,
                custom_id="select_emoji_to_remove",
                # min_values=1, max_values=1 # For single select
            )
            self.remove_select_menu.callback = self.process_remove_selection
            # self.add_item(self.remove_select_menu) # This will be handled by update_message via _add_initial_buttons
            await self.update_message(interaction, "Select an emoji to remove.")

    def _get_emojis_on_current_page(self) -> list:
        start_index = self.current_page_index * EMOJIS_PER_PAGE
        end_index = start_index + EMOJIS_PER_PAGE
        return self.emojis[start_index:end_index]

    async def process_remove_selection(self, interaction: Interaction):
        if not self.remove_select_menu or not interaction.data or not interaction.data.get("values"):
            await interaction.response.send_message("No emoji selected or selection error.",ephemeral=True)
            return
        
        selected_emoji_str = interaction.data["values"][0]
        emoji_to_remove = next((e for e in self.emojis if e["emoji"] == selected_emoji_str), None)

        if not emoji_to_remove:
            await interaction.response.send_message(f"Could not find emoji {selected_emoji_str} to remove. It might have been removed already.",ephemeral=True)
            return

        # Remove the select menu itself
        if self.remove_select_menu:
            self.remove_item(self.remove_select_menu)
            self.remove_select_menu = None
        
        # Add confirmation buttons
        self.confirm_remove_button_instance = ConfirmRemoveButton(self, emoji_to_remove)
        self.cancel_remove_button_instance = CancelRemoveButton(self)
        # these will be added by update_message if _add_initial_buttons is smart
        # self.add_item(self.confirm_remove_button_instance)
        # self.add_item(self.cancel_remove_button_instance)
        
        await self.update_message(interaction, f"Are you sure you want to remove {emoji_to_remove['emoji']}?")


    async def process_remove_emoji_confirmed(self, interaction: Interaction, emoji_dict_to_remove: dict):
        emoji_str_to_remove = emoji_dict_to_remove["emoji"]
        
        original_length = len(self.emojis)
        self.emojis = [e for e in self.emojis if e["emoji"] != emoji_str_to_remove]

        if len(self.emojis) < original_length:
            await self.cog._save_emoji_list(self.config_group, self.emoji_list_attr_name, self.emojis)
            feedback = f"Emoji {emoji_str_to_remove} removed successfully!"
             # Ensure current page index is valid if we removed the last item on a page that no longer exists
            if self.current_page_index * EMOJIS_PER_PAGE >= len(self.emojis) and self.current_page_index > 0:
                self.current_page_index -=1

        else:
            feedback = f"Could not find emoji {emoji_str_to_remove} to remove (it may have already been removed by another action)."

        # Clean up confirmation buttons
        if self.confirm_remove_button_instance:
            self.remove_item(self.confirm_remove_button_instance)
            self.confirm_remove_button_instance = None
        if self.cancel_remove_button_instance:
            self.remove_item(self.cancel_remove_button_instance)
            self.cancel_remove_button_instance = None

        await self.update_message(interaction, feedback)


    async def close_menu(self, interaction: Interaction):
        await interaction.response.defer() # Ack the interaction
        if self.message:
            await self.message.edit(content="Emoji management menu closed.", view=None, embed=None)
        self.stop()