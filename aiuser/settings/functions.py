
import json
import logging
from dataclasses import fields

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

    @functions.command(name="generate_image")
    async def toggle_generate_image(self, ctx: commands.Context):
        """ Toggle direct image generation function calling

        Enables/disables the local generate_image function call.
        When enabled, the LLM can call this function to generate images
        directly via the configured OpenRouter model.

        Note: You must configure a model using `[p]aiuser functions generate_image_config`
        before the function will work.
        """
        from aiuser.functions.generate_image.tool_call import GenerateImageToolCall

        tool_names = [GenerateImageToolCall.function_name]
        await self.toggle_function_helper(ctx, tool_names, "Generate Image")

    @functions.command(name="generate_image_config")
    async def config_generate_image(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure direct image generation parameters using JSON

        Sets the model and image_size that will be used for direct image generation.
        The prompt and aspect_ratio are passed by the LLM, everything else is
        configured here.

        To reset parameters to default, use `{ctx.clean_prefix}functions generate_image_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions generate_image_config show`

        Example command:
        `{ctx.clean_prefix}functions generate_image_config ```json\n{"model": "google/gemini-3.1-flash-image-preview", "image_size": "2K"}\n``` `

        Valid fields: model, image_size
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="direct_image_generation_parameters",
            tool_type="direct_image_generation",
            param_display_name="Direct Image Generation",
            example_config={
                "model": "google/gemini-3.1-flash-image-preview",
                "image_size": "2K",
            },
            json_block=json_block,
        )

    @functions.command(name="edit_image")
    async def toggle_edit_image(self, ctx: commands.Context):
        """ Toggle direct image edit function calling

        Enables/disables the local edit_image function call.
        When enabled, the LLM can call this function to edit existing images
        directly via the configured OpenRouter model.

        Note: You must configure a model using `[p]aiuser functions edit_image_config`
        before the function will work.
        """
        from aiuser.functions.edit_image.tool_call import EditImageToolCall

        tool_names = [EditImageToolCall.function_name]
        await self.toggle_function_helper(ctx, tool_names, "Edit Image")

    @functions.command(name="edit_image_config")
    async def config_edit_image(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure direct image edit parameters using JSON

        Sets the model and image_size that will be used for direct image editing.
        The prompt and image_to_edit are passed by the LLM, everything else is
        configured here.

        To reset parameters to default, use `{ctx.clean_prefix}functions edit_image_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions edit_image_config show`

        Example command:
        `{ctx.clean_prefix}functions edit_image_config ```json\n{"model": "google/gemini-3.1-flash-image-preview", "image_size": "2K"}\n``` `

        Valid fields: model, image_size
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="direct_image_edit_parameters",
            tool_type="direct_image_edit",
            param_display_name="Edit Image",
            example_config={
                "model": "google/gemini-3.1-flash-image-preview",
                "image_size": "2K",
            },
            json_block=json_block,
        )

    @functions.command(name="or_pdf_parsing")
    async def toggle_or_pdf_parsing(self, ctx: commands.Context):
        """ Toggle OpenRouter PDF parsing

        Enables/disables PDF parsing via OpenRouter's server-side file-parser plugin.
        When enabled, PDF links in messages will be downloaded and sent to the LLM
        for parsing. PDF attachments are also processed.
        """
        await self._toggle_openrouter_boolean(ctx, "openrouter_pdf_parsing_enabled", "PDF Parsing")

    # ─── OpenRouter Config Commands ────────────────────────────────────────────

    async def _handle_or_json_config(
        self,
        ctx: commands.Context,
        config_key: str,
        tool_type: OpenRouterToolType,
        param_display_name: str,
        example_config: dict,
        *,
        json_block: str,
        extra_help: str = "",
    ):
        """Shared helper for OpenRouter *config commands using JSON.

        Matches the pattern of `[p]aiuser response parameters`:
        - No args or 'show'/'list' → shows current config + example
        - 'reset'/'clear' → resets to defaults
        - JSON code block → sets parameters
        """
        from aiuser.types.openrouter_types import _PARAM_CLASS_MAP

        param_class = _PARAM_CLASS_MAP.get(tool_type)

        if not json_block or json_block in ('show', 'list'):
            # Show current config
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            params_obj = deserialize_parameters(current_json, tool_type)

            embed = discord.Embed(
                title=f"OpenRouter {param_display_name} Configuration",
                color=await ctx.embed_color(),
            )

            # Current settings
            current_fields = {f.name: getattr(params_obj, f.name) for f in fields(param_class)}
            non_null = {k: v for k, v in current_fields.items() if v is not None}
            if non_null:
                for field_name, field_value in non_null.items():
                    embed.add_field(name=field_name, value=f"`{field_value}`", inline=True)
            else:
                embed.add_field(name="Current", value="No custom parameters set. Using OpenRouter defaults.", inline=False)

            # Example reference
            example_json = json.dumps(example_config, indent=2)
            embed.add_field(
                name="Reference Example (JSON)",
                value=f"```json\n{example_json}\n```",
                inline=False,
            )

            # Usage instructions
            tool_config_name = tool_type if isinstance(tool_type, str) else tool_type.value.replace(':', '_')
            usage = (
                f"• Set: `{ctx.clean_prefix}functions {tool_config_name}_config "
                f"```json\n{'{...}'}\n``` `\n"
                f"• Reset: `{ctx.clean_prefix}functions {tool_config_name}_config reset`"
            )
            if extra_help:
                usage += f"\n{extra_help}"
            embed.add_field(name="Usage", value=usage, inline=False)

            return await ctx.send(embed=embed)

        if json_block in ('reset', 'clear'):
            await getattr(self.config.guild(ctx.guild), config_key).set(None)
            return await ctx.send(f"OpenRouter {param_display_name} configuration reset to defaults.")

        # Parse JSON block
        raw = json_block
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return await ctx.send(":warning: Invalid JSON format!")

        if not isinstance(data, dict):
            return await ctx.send(":warning: Expected a JSON object.")

        # Filter to only valid fields
        valid_fields = {f.name for f in fields(param_class)}
        unknown_keys = [k for k in data if k not in valid_fields]
        if unknown_keys:
            await ctx.send(
                f":warning: Unknown key(s): {', '.join(f'`{k}`' for k in unknown_keys)}. "
                f"Valid keys: {', '.join(f'`{k}`' for k in valid_fields)}."
            )
            # Still proceed with valid keys only
            data = {k: v for k, v in data.items() if k in valid_fields}

        # Rebuild using the dataclass (validates types implicitly)
        try:
            params_obj = param_class(**{k: v for k, v in data.items() if k in valid_fields})
        except TypeError as e:
            return await ctx.send(f":warning: Invalid parameter values: {e}")

        serialized = serialize_parameters(params_obj)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title=f"OpenRouter {param_display_name} Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="or_web_search_config")
    async def config_or_web_search(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure OpenRouter web search parameters using JSON

        To reset parameters to default, use `{ctx.clean_prefix}functions or_web_search_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions or_web_search_config show`

        Example command:
        `{ctx.clean_prefix}functions or_web_search_config ```json\n{"engine": "exa", "max_results": 10}\n``` `

        Valid fields: engine, max_results, max_total_results, search_context_size,
        user_location, allowed_domains, excluded_domains
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="openrouter_web_search_parameters",
            tool_type=OpenRouterToolType.WEB_SEARCH,
            param_display_name="Web Search",
            example_config={
                "engine": "exa",
                "max_results": 10,
                "max_total_results": 100,
                "search_context_size": "medium",
                "user_location": None,
                "allowed_domains": ["arxiv.org", "nature.com"],
                "excluded_domains": None,
            },
            json_block=json_block,
        )

    @functions.command(name="or_web_fetch_config")
    async def config_or_web_fetch(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure OpenRouter web fetch parameters using JSON

        To reset parameters to default, use `{ctx.clean_prefix}functions or_web_fetch_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions or_web_fetch_config show`

        Example command:
        `{ctx.clean_prefix}functions or_web_fetch_config ```json\n{"engine": "firecrawl", "max_uses": 5}\n``` `

        Valid fields: engine, max_uses, max_content_tokens, allowed_domains, blocked_domains
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="openrouter_web_fetch_parameters",
            tool_type=OpenRouterToolType.WEB_FETCH,
            param_display_name="Web Fetch",
            example_config={
                "engine": "firecrawl",
                "max_uses": 5,
                "max_content_tokens": 8000,
                "allowed_domains": None,
                "blocked_domains": ["example.com"],
            },
            json_block=json_block,
        )

    @functions.command(name="or_image_gen_config")
    async def config_or_image_gen(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure OpenRouter image generation parameters using JSON

        To reset parameters to default, use `{ctx.clean_prefix}functions or_image_gen_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions or_image_gen_config show`

        Example command:
        `{ctx.clean_prefix}functions or_image_gen_config ```json\n{"model": "openai/dall-e-3", "quality": "high"}\n``` `

        Valid fields: model, quality, size, aspect_ratio, background, output_format,
        output_compression, moderation
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="openrouter_image_generation_parameters",
            tool_type=OpenRouterToolType.IMAGE_GENERATION,
            param_display_name="Image Generation",
            example_config={
                "model": "openai/dall-e-3",
                "quality": "high",
                "size": "1024x1024",
                "aspect_ratio": None,
                "background": None,
                "output_format": "png",
                "output_compression": None,
                "moderation": None,
            },
            json_block=json_block,
        )

    @functions.command(name="or_pdf_parsing_config")
    async def config_or_pdf_parsing(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure OpenRouter PDF parsing engine using JSON

        To reset parameters to default, use `{ctx.clean_prefix}functions or_pdf_parsing_config reset`
        To show current parameters, use `{ctx.clean_prefix}functions or_pdf_parsing_config show`

        Example command:
        `{ctx.clean_prefix}functions or_pdf_parsing_config ```json\n{"engine": "cloudflare-ai"}\n``` `

        Valid fields: engine (cloudflare-ai, mistral-ocr, or native)
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="openrouter_pdf_parsing_parameters",
            tool_type=OpenRouterToolType.PDF_PARSING,
            param_display_name="PDF Parsing",
            example_config={
                "engine": "cloudflare-ai",
            },
            json_block=json_block,
            extra_help="Available engines: `cloudflare-ai` (free), `mistral-ocr` (paid), `native` (model-native only)",
        )

