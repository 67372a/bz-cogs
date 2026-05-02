
import json
import logging

import discord
from redbot.core import checks, commands

from aiuser.types.abc import MixinMeta, aiuser
from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    WebSearchParameters,
    WebFetchParameters,
    ImageGenerationParameters,
    serialize_parameters,
    deserialize_parameters,
)

logger = logging.getLogger("red.bz_cogs.aiuser")


class FunctionCallingSettings(MixinMeta):
    @aiuser.group()
    @checks.is_owner()
    async def functions(self, _):
        """ Settings to manage function calling

            (All subcommands are per server)
        """
        pass

    @functions.command(name="toggle")
    async def toggle_function_calling(self, ctx: commands.Context):
        """Toggle functions calling

        Requires a model that is whitelisted or supported for function calling
        If enabled, the LLM will call functions to generate responses when needed
        This will generate additional API calls and token usage!

        """

        current_value = not await self.config.guild(ctx.guild).function_calling()
        await self.config.guild(ctx.guild).function_calling.set(current_value)

        embed = discord.Embed(
            title="Functions Calling now set to:",
            description=f"{current_value}",
            color=await ctx.embed_color(),
        )
        if current_value:
            embed.set_footer(text="⚠️ Ensure selected model supports function calling!")
        await ctx.send(embed=embed)

    @functions.command(name="location")
    async def set_location(self, ctx: commands.Context, latitude: float, longitude: float):
        """ Set the location where the bot will canonically be in

            Used for some functions.

            **Arguments**
            - `latitude` decimal latitude
            - `longitude` decimal longitude
        """
        await self.config.guild(ctx.guild).function_calling_default_location.set([latitude, longitude])
        embed = discord.Embed(
            title="Location now set to:",
            description=f"{latitude}, {longitude}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    async def toggle_function_helper(self, ctx: commands.Context, tool_names: list, embed_title: str):
        enabled_tools: list = await self.config.guild(ctx.guild).function_calling_functions()

        if tool_names[0] not in enabled_tools:
            enabled_tools.extend(tool_names)
        else:
            for tool in tool_names:
                enabled_tools.remove(tool)

        await self.config.guild(ctx.guild).function_calling_functions.set(enabled_tools)

        embed = discord.Embed(
            title=f"{embed_title} function calling now set to:",
            description=f"{tool_names[0] in enabled_tools}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="search")
    async def toggle_search_function(self, ctx: commands.Context):
        """ Enable/disable searching/scraping the Internet using Serper.dev """
        if (not (await self.bot.get_shared_api_tokens("serper")).get("api_key")):
            return await ctx.send(f"Serper.dev key not set! Set it using `{ctx.clean_prefix}set api serper api_key,APIKEY`.")

        from aiuser.functions.search.tool_call import SearchToolCall

        # Check mutual exclusivity: if OpenRouter web search is enabled, disable it
        if await self.config.guild(ctx.guild).openrouter_web_search_enabled():
            await self.config.guild(ctx.guild).openrouter_web_search_enabled.set(False)
            await ctx.send("⚠️ OpenRouter web search has been disabled (mutual exclusivity with local Serper.dev search).")

        tool_names = [SearchToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Search")

    @functions.command(name="scrape")
    async def toggle_scrape_function(self, ctx: commands.Context):
        """
        Enable/disable the functionality for the LLM to open URLs in messages

        (May not be called if the link generated an Discord embed)
        """
        from aiuser.functions.scrape.tool_call import ScrapeToolCall

        # Check mutual exclusivity: if OpenRouter web fetch is enabled, disable it
        if await self.config.guild(ctx.guild).openrouter_web_fetch_enabled():
            await self.config.guild(ctx.guild).openrouter_web_fetch_enabled.set(False)
            await ctx.send("⚠️ OpenRouter web fetch has been disabled (mutual exclusivity with local scrape).")

        tool_names = [ScrapeToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Scrape")

    @functions.command(name="weather")
    async def toggle_weather_function(self, ctx: commands.Context):
        """ Enable/disable a group of functions to getting weather using Open-Meteo

            See [Open-Meteo terms](https://open-meteo.com/en/terms) for their free API
        """
        from aiuser.functions.weather.tool_call import (
            IsDaytimeToolCall,
            LocalWeatherToolCall,
            LocationWeatherToolCall,
        )

        tool_names = [IsDaytimeToolCall.function_name,
                      LocalWeatherToolCall.function_name, LocationWeatherToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Weather")

    @functions.command(name="noresponse")
    async def toggle_ignore_function(self, ctx: commands.Context):
        """
        Enable/disable the functionality for the LLM to choose to not respond and ignore messages.

        Temperamental, may require additional prompting to work better.
        """
        from aiuser.functions.noresponse.tool_call import NoResponseToolCall

        tool_names = [NoResponseToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "No response")

    @functions.command(name="wolframalpha")
    async def toggle_wolfram_alpha_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to ask Wolfram Alpha about math, exchange rates, or the weather."""
        from aiuser.functions.wolframalpha.tool_call import WolframAlphaFunctionCall

        if (not (await self.bot.get_shared_api_tokens("wolfram_alpha")).get("app_id")):
            return await ctx.send(f"Wolfram Alpha app id not set! Set it using `{ctx.clean_prefix}set api wolfram_alpha app_id,APPID`.")

        tool_names = [WolframAlphaFunctionCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Wolfram Alpha")

    @functions.command(name="timeoutuser")
    async def toggle_discord_user_timeout_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to timeout users."""
        from aiuser.functions.user_timeout.tool_call import TimeoutUserToolCall

        tool_names = [TimeoutUserToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Timeout User")

    @functions.command(name="changeusernickname")
    async def toggle_discord_user_change_nickname_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to change user nicknames."""
        from aiuser.functions.user_change_nickname.tool_call import ChangeUserNicknameToolCall

        tool_names = [ChangeUserNicknameToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Change User Nickname")

    @functions.command(name="emojireaction")
    async def toggle_discord_emoji_reaction_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to react to messages with an emoji."""
        from aiuser.functions.discord.tool_call import ReactToMessageToolCall

        tool_names = [ReactToMessageToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "React To Message")

    @functions.command(name="ttsmessage")
    async def toggle_discord_tts_message_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to send tts messages."""
        from aiuser.functions.discord.tool_call import SendTtsMessageToolCall

        tool_names = [SendTtsMessageToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Send TTS Message")

    @functions.command(name="pinmessage")
    async def toggle_discord_pin_message_function(self, ctx: commands.Context):
        """ Enable/disable the functionality for the LLM to pin messages."""
        from aiuser.functions.discord.tool_call import PinMessageToolCall

        tool_names = [PinMessageToolCall.function_name]

        await self.toggle_function_helper(ctx, tool_names, "Pin Message")

    # ─── OpenRouter Server Tools ───────────────────────────────────────────────

    async def _toggle_openrouter_boolean(self, ctx: commands.Context, config_key: str, tool_display_name: str):
        """Toggle an OpenRouter server tool boolean config key."""
        current_value = not await getattr(self.config.guild(ctx.guild), config_key)()
        await getattr(self.config.guild(ctx.guild), config_key).set(current_value)
        embed = discord.Embed(
            title=f"OpenRouter {tool_display_name} now set to:",
            description=f"{current_value}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="or_web_search")
    async def toggle_or_web_search(self, ctx: commands.Context):
        """ Toggle OpenRouter web search server tool

        Enables/disables the openrouter:web_search server tool.
        Mutual exclusive with local Serper.dev search - enabling this
        will disable the local search function.
        """
        currently_enabled = await self.config.guild(ctx.guild).openrouter_web_search_enabled()
        if not currently_enabled:
            # Check mutual exclusivity: disable local Serper.dev search if enabling
            from aiuser.functions.search.tool_call import SearchToolCall
            enabled_tools = await self.config.guild(ctx.guild).function_calling_functions()
            if SearchToolCall.function_name in enabled_tools:
                enabled_tools.remove(SearchToolCall.function_name)
                await self.config.guild(ctx.guild).function_calling_functions.set(enabled_tools)
                await ctx.send("⚠️ Local Serper.dev search has been disabled (mutual exclusivity with OpenRouter web search).")

        await self._toggle_openrouter_boolean(ctx, "openrouter_web_search_enabled", "Web Search")

    @functions.command(name="or_web_fetch")
    async def toggle_or_web_fetch(self, ctx: commands.Context):
        """ Toggle OpenRouter web fetch server tool

        Enables/disables the openrouter:web_fetch server tool.
        Mutual exclusive with local scrape - enabling this
        will disable the local scrape function.
        """
        currently_enabled = await self.config.guild(ctx.guild).openrouter_web_fetch_enabled()
        if not currently_enabled:
            # Check mutual exclusivity: disable local scrape if enabling
            from aiuser.functions.scrape.tool_call import ScrapeToolCall
            enabled_tools = await self.config.guild(ctx.guild).function_calling_functions()
            if ScrapeToolCall.function_name in enabled_tools:
                enabled_tools.remove(ScrapeToolCall.function_name)
                await self.config.guild(ctx.guild).function_calling_functions.set(enabled_tools)
                await ctx.send("⚠️ Local scrape function has been disabled (mutual exclusivity with OpenRouter web fetch).")

        await self._toggle_openrouter_boolean(ctx, "openrouter_web_fetch_enabled", "Web Fetch")

    @functions.command(name="or_image_gen")
    async def toggle_or_image_gen(self, ctx: commands.Context):
        """ Toggle OpenRouter image generation server tool

        Enables/disables the openrouter:image_generation server tool.
        No local counterpart, no mutual exclusivity concerns.
        """
        await self._toggle_openrouter_boolean(ctx, "openrouter_image_generation_enabled", "Image Generation")

    @functions.command(name="or_pdf_parsing")
    async def toggle_or_pdf_parsing(self, ctx: commands.Context):
        """ Toggle OpenRouter PDF parsing

        Enables/disables PDF parsing via OpenRouter's server-side file-parser plugin.
        When enabled, PDF links in messages will be downloaded and sent to the LLM
        for parsing. PDF attachments are also processed.
        """
        await self._toggle_openrouter_boolean(ctx, "openrouter_pdf_parsing_enabled", "PDF Parsing")

    # ─── OpenRouter Config Commands ────────────────────────────────────────────

    @functions.command(name="or_web_search_config")
    async def config_or_web_search(self, ctx: commands.Context, *params):
        """ Configure OpenRouter web search parameters

        Set parameters as key=value pairs. If no arguments provided, shows current config.
        Parameters: engine, max_results, max_total_results, search_context_size,
        allowed_domains, excluded_domains

        Examples:
        {ctx.clean_prefix}functions or_web_search_config engine=exa max_results=10
        {ctx.clean_prefix}functions or_web_search_config allowed_domains='["arxiv.org","nature.com"]'
        """
        config_key = "openrouter_web_search_parameters"
        tool_type = OpenRouterToolType.WEB_SEARCH

        if not params:
            # Show current config
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            params_obj = deserialize_parameters(current_json, tool_type)
            embed = discord.Embed(
                title="OpenRouter Web Search Configuration",
                color=await ctx.embed_color(),
            )
            for field_name, field_value in params_obj.__dict__.items():
                if field_value is not None:
                    embed.add_field(name=field_name, value=f"`{field_value}`", inline=True)
            if not embed.fields:
                embed.description = "No custom parameters set. Using OpenRouter defaults."
            embed.set_footer(text="Use key=value arguments to set parameters")
            return await ctx.send(embed=embed)

        # Parse key=value pairs and update config
        current_json = await getattr(self.config.guild(ctx.guild), config_key)()
        params_obj = deserialize_parameters(current_json, tool_type)
        params_dict = params_obj.__dict__.copy()

        for param in params:
            if "=" not in param:
                return await ctx.send(f"Invalid parameter format: `{param}`. Use `key=value`.")
            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in params_dict:
                return await ctx.send(f"Unknown parameter: `{key}`. Valid: {', '.join(params_dict.keys())}.")

            # Parse value types
            try:
                if key in ("max_results", "max_total_results"):
                    params_dict[key] = int(value)
                elif key in ("allowed_domains", "excluded_domains"):
                    params_dict[key] = json.loads(value) if value.lower() != "null" else None
                elif key == "user_location":
                    params_dict[key] = json.loads(value) if value.lower() != "null" else None
                elif value.lower() == "null":
                    params_dict[key] = None
                else:
                    params_dict[key] = value
            except (json.JSONDecodeError, ValueError):
                return await ctx.send(f"Invalid value for `{key}`: `{value}`.")

        params_obj = WebSearchParameters(**params_dict)
        serialized = serialize_parameters(params_obj)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title="OpenRouter Web Search Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="or_web_fetch_config")
    async def config_or_web_fetch(self, ctx: commands.Context, *params):
        """ Configure OpenRouter web fetch parameters

        Set parameters as key=value pairs. If no arguments provided, shows current config.
        Parameters: engine, max_uses, max_content_tokens, allowed_domains, blocked_domains

        Examples:
        {ctx.clean_prefix}functions or_web_fetch_config engine=firecrawl max_uses=5
        """
        config_key = "openrouter_web_fetch_parameters"
        tool_type = OpenRouterToolType.WEB_FETCH

        if not params:
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            params_obj = deserialize_parameters(current_json, tool_type)
            embed = discord.Embed(
                title="OpenRouter Web Fetch Configuration",
                color=await ctx.embed_color(),
            )
            for field_name, field_value in params_obj.__dict__.items():
                if field_value is not None:
                    embed.add_field(name=field_name, value=f"`{field_value}`", inline=True)
            if not embed.fields:
                embed.description = "No custom parameters set. Using OpenRouter defaults."
            embed.set_footer(text="Use key=value arguments to set parameters")
            return await ctx.send(embed=embed)

        current_json = await getattr(self.config.guild(ctx.guild), config_key)()
        params_obj = deserialize_parameters(current_json, tool_type)
        params_dict = params_obj.__dict__.copy()

        for param in params:
            if "=" not in param:
                return await ctx.send(f"Invalid parameter format: `{param}`. Use `key=value`.")
            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in params_dict:
                return await ctx.send(f"Unknown parameter: `{key}`. Valid: {', '.join(params_dict.keys())}.")

            try:
                if key in ("max_uses", "max_content_tokens"):
                    params_dict[key] = int(value)
                elif key in ("allowed_domains", "blocked_domains"):
                    params_dict[key] = json.loads(value) if value.lower() != "null" else None
                elif value.lower() == "null":
                    params_dict[key] = None
                else:
                    params_dict[key] = value
            except (json.JSONDecodeError, ValueError):
                return await ctx.send(f"Invalid value for `{key}`: `{value}`.")

        params_obj = WebFetchParameters(**params_dict)
        serialized = serialize_parameters(params_obj)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title="OpenRouter Web Fetch Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="or_image_gen_config")
    async def config_or_image_gen(self, ctx: commands.Context, *params):
        """ Configure OpenRouter image generation parameters

        Set parameters as key=value pairs. If no arguments provided, shows current config.
        Parameters: model, quality, size, aspect_ratio, background, output_format,
        output_compression, moderation

        Examples:
        {ctx.clean_prefix}functions or_image_gen_config model=openai/dall-e-3 quality=high
        {ctx.clean_prefix}functions or_image_gen_config size=1024x1024 output_format=png
        """
        config_key = "openrouter_image_generation_parameters"
        tool_type = OpenRouterToolType.IMAGE_GENERATION

        if not params:
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            params_obj = deserialize_parameters(current_json, tool_type)
            embed = discord.Embed(
                title="OpenRouter Image Generation Configuration",
                color=await ctx.embed_color(),
            )
            for field_name, field_value in params_obj.__dict__.items():
                if field_value is not None:
                    embed.add_field(name=field_name, value=f"`{field_value}`", inline=True)
            if not embed.fields:
                embed.description = "No custom parameters set. Using OpenRouter defaults."
            embed.set_footer(text="Use key=value arguments to set parameters")
            return await ctx.send(embed=embed)

        current_json = await getattr(self.config.guild(ctx.guild), config_key)()
        params_obj = deserialize_parameters(current_json, tool_type)
        params_dict = params_obj.__dict__.copy()

        for param in params:
            if "=" not in param:
                return await ctx.send(f"Invalid parameter format: `{param}`. Use `key=value`.")
            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in params_dict:
                return await ctx.send(f"Unknown parameter: `{key}`. Valid: {', '.join(params_dict.keys())}.")

            try:
                if key == "output_compression":
                    params_dict[key] = int(value)
                elif value.lower() == "null":
                    params_dict[key] = None
                else:
                    params_dict[key] = value
            except ValueError:
                return await ctx.send(f"Invalid value for `{key}`: `{value}`.")

        params_obj = ImageGenerationParameters(**params_dict)
        serialized = serialize_parameters(params_obj)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title="OpenRouter Image Generation Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="or_pdf_parsing_config")
    async def config_or_pdf_parsing(self, ctx: commands.Context, *params):
        """ Configure OpenRouter PDF parsing engine

        Set parameters as key=value pairs. If no arguments provided, shows current config.
        Parameters: engine (cloudflare-ai, mistral-ocr, or native)

        Examples:
        {ctx.clean_prefix}functions or_pdf_parsing_config engine=cloudflare-ai
        {ctx.clean_prefix}functions or_pdf_parsing_config engine=mistral-ocr
        """
        from aiuser.types.openrouter_types import PdfParsingParameters

        config_key = "openrouter_pdf_parsing_parameters"
        tool_type = OpenRouterToolType.PDF_PARSING

        if not params:
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            params_obj = deserialize_parameters(current_json, tool_type)
            embed = discord.Embed(
                title="OpenRouter PDF Parsing Configuration",
                color=await ctx.embed_color(),
            )
            for field_name, field_value in params_obj.__dict__.items():
                if field_value is not None:
                    embed.add_field(name=field_name, value=f"`{field_value}`", inline=True)
            if not embed.fields:
                embed.description = "No custom engine set. Using cloudflare-ai (free)."
            embed.add_field(
                name="Available Engines",
                value="`cloudflare-ai` (free), `mistral-ocr` (paid), `native` (model-native only)",
                inline=False,
            )
            embed.set_footer(text="Use key=value arguments to set parameters")
            return await ctx.send(embed=embed)

        current_json = await getattr(self.config.guild(ctx.guild), config_key)()
        params_obj = deserialize_parameters(current_json, tool_type)
        params_dict = params_obj.__dict__.copy()

        for param in params:
            if "=" not in param:
                return await ctx.send(f"Invalid parameter format: `{param}`. Use `key=value`.")
            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in params_dict:
                return await ctx.send(f"Unknown parameter: `{key}`. Valid: {', '.join(params_dict.keys())}.")

            try:
                if value.lower() == "null":
                    params_dict[key] = None
                elif key == "engine" and value not in ("cloudflare-ai", "mistral-ocr", "native"):
                    return await ctx.send(
                        f"Invalid engine: `{value}`. Valid: cloudflare-ai, mistral-ocr, native."
                    )
                else:
                    params_dict[key] = value
            except ValueError:
                return await ctx.send(f"Invalid value for `{key}`: `{value}`.")

        params_obj = PdfParsingParameters(**params_dict)
        serialized = serialize_parameters(params_obj)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title="OpenRouter PDF Parsing Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

