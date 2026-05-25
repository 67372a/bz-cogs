
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
from aiuser.utils.utilities import get_enabled_tools

logger = logging.getLogger("red.bz_cogs.aiuser")


class FunctionCallingSettings(MixinMeta):
    @aiuser.group(invoke_without_command=True)
    @checks.is_owner()
    async def functions(self, ctx: commands.Context):
        """ Settings to manage function calling

            (All subcommands are per server)
        """
        # Check for help flag: pass through to Red's help system
        content = ctx.message.content.strip()
        if content.endswith(('--help', 'help')):
            return await ctx.send_help(ctx.command)

        enabled = await self.config.guild(ctx.guild).function_calling()
        enabled_functions = await get_enabled_tools(self.config, ctx)

        embed = discord.Embed(
            title="Function Calling Settings",
            color=await ctx.embed_color(),
        )

        embed.add_field(name="Enabled", value=f"`{enabled}`", inline=False)

        if enabled_functions:
            func_names = "\n".join(f"• `{t.function_name}`" for t in enabled_functions)
            embed.add_field(name=f"Enabled Functions ({len(enabled_functions)})", value=func_names, inline=False)
        else:
            embed.add_field(name="Enabled Functions", value="*None*", inline=False)

        # Generic web function backends
        web_search_backend = await self.config.guild(ctx.guild).web_search_backend()
        web_fetch_backend = await self.config.guild(ctx.guild).web_fetch_backend()
        web_answer_backend = await self.config.guild(ctx.guild).web_answer_backend()

        web_info_parts = []
        if web_search_backend:
            web_info_parts.append(f"Web Search: `{web_search_backend}`")
        if web_fetch_backend:
            web_info_parts.append(f"Web Fetch: `{web_fetch_backend}`")
        if web_answer_backend:
            web_info_parts.append(f"Web Answer: `{web_answer_backend}`")
        if web_info_parts:
            embed.add_field(name="Web Function Backends", value="\n".join(web_info_parts), inline=False)

        # OpenRouter server tools
        or_search = await self.config.guild(ctx.guild).openrouter_web_search_enabled()
        or_fetch = await self.config.guild(ctx.guild).openrouter_web_fetch_enabled()
        or_image = await self.config.guild(ctx.guild).openrouter_image_generation_enabled()
        or_pdf = await self.config.guild(ctx.guild).openrouter_pdf_parsing_enabled()

        or_tools = []
        if or_search:
            or_tools.append("Web Search (server)")
        if or_fetch:
            or_tools.append("Web Fetch (server)")
        if or_image:
            or_tools.append("Image Gen")
        if or_pdf:
            or_tools.append("PDF Parsing")
        or_status = "\n".join(f"• `{t}`" for t in or_tools) if or_tools else "*None*"
        embed.add_field(name="OpenRouter Server Tools", value=or_status, inline=False)

        # Location
        location = await self.config.guild(ctx.guild).function_calling_default_location()
        if location:
            embed.add_field(name="Location", value=f"`{location[0]}, {location[1]}`", inline=False)

        max_rounds = await self.config.guild(ctx.guild).max_tool_rounds()
        embed.add_field(name="Max Tool Rounds", value=f"`{max_rounds}`", inline=False)

        # Attach Files config
        attach_max_files = await self.config.guild(ctx.guild).attach_files_max_files()
        attach_max_size = await self.config.guild(ctx.guild).attach_files_max_file_size_mb()
        embed.add_field(
            name="Attach Files Config",
            value=f"Max files: `{attach_max_files}` • Max size: `{attach_max_size}` MB",
            inline=False,
        )

        # Mermaid Diagram config
        mermaid_enabled = await self.config.guild(ctx.guild).function_calling_functions()
        if "create_mermaid_diagram" in mermaid_enabled:
            mermaid_theme = await self.config.guild(ctx.guild).mermaid_diagram_theme()
            embed.add_field(
                name="Mermaid Diagram Config",
                value=f"Theme: `{mermaid_theme}`",
                inline=False,
            )

        embed.set_footer(text=f"Use {ctx.clean_prefix}aiuser functions --help to see all subcommands")
        await ctx.send(embed=embed)

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

    @functions.command(name="maxrounds")
    async def set_max_tool_rounds(self, ctx: commands.Context, rounds: int):
        """Set the maximum number of tool-calling rounds.

        Controls how many sequential LLM calls can be made with tool execution
        between them. Default is 2 (one tool call + one response).
        Set to 1 to disable tool execution (single call only).
        Set higher (e.g. 5) for chained/sequential tool calls.

        **Arguments**
            - `rounds` number of rounds (1–10)
        """
        if rounds < 1 or rounds > 10:
            return await ctx.send("Please provide a value between 1 and 10.")
        await self.config.guild(ctx.guild).max_tool_rounds.set(rounds)
        embed = discord.Embed(
            title="Max Tool Rounds now set to:",
            description=f"`{rounds}`",
            color=await ctx.embed_color(),
        )
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

    # ─── Deprecated: legacy search command ─────────────────────────────────

    @functions.command(name="search")
    async def toggle_search_function(self, ctx: commands.Context):
        """⚠️ DEPRECATED — use `web_search` instead.

        This command will auto-migrate your settings to the new generic web_search
        system using the Serper.dev backend. No data is lost.
        """
        await ctx.send("⚠️ The `search` command is deprecated. Migrating to generic `web_search` with `serper` backend...")
        await self.config.guild(ctx.guild).web_search_backend.set("serper")
        return await self._toggle_generic_web_function(ctx, "web_search", "Web Search")

    # ─── Deprecated: legacy scrape command ─────────────────────────────────

    @functions.command(name="scrape")
    async def toggle_scrape_function(self, ctx: commands.Context):
        """⚠️ DEPRECATED — use `web_fetch` instead.

        This command will auto-migrate your settings to the new generic web_fetch
        system using the scrape backend. No data is lost.
        """
        await ctx.send("⚠️ The `scrape` command is deprecated. Migrating to generic `web_fetch` with `scrape` backend...")
        await self.config.guild(ctx.guild).web_fetch_backend.set("scrape")
        return await self._toggle_generic_web_function(ctx, "web_fetch", "Web Fetch")

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

    @functions.command(name="attach_files")
    async def toggle_attach_files_function(self, ctx: commands.Context):
        """Enable/disable the functionality for the LLM to attach code/text files to responses.

        When enabled, the LLM can output code and text files as downloadable
        Discord attachments instead of pasting code inline in messages.
        Configure limits with `attach_files_max_files` and `attach_files_max_file_size`.
        """
        from aiuser.functions.attach_files.tool_call import AttachFilesToolCall

        tool_names = [AttachFilesToolCall.function_name]
        await self.toggle_function_helper(ctx, tool_names, "Attach Files")

    @functions.command(name="attach_files_max_files")
    async def set_attach_files_max_files(self, ctx: commands.Context, max_files: int = 0):
        """Set the maximum number of files the LLM can attach per call.

        **Arguments**
            - `max_files` maximum files per call (1–25, default: 10)

        Use without arguments to view the current setting.
        """
        if max_files == 0:
            current = await self.config.guild(ctx.guild).attach_files_max_files()
            embed = discord.Embed(
                title="Attach Files — Max Files",
                description=f"`{current}` file(s) per call",
                color=await ctx.embed_color(),
            )
            embed.add_field(
                name="Set",
                value=f"`{ctx.clean_prefix}aiuser functions attach_files_max_files <1-25>`",
                inline=False,
            )
            return await ctx.send(embed=embed)

        if max_files < 1 or max_files > 25:
            return await ctx.send("Please provide a value between 1 and 25.")

        await self.config.guild(ctx.guild).attach_files_max_files.set(max_files)
        embed = discord.Embed(
            title="Attach Files — Max Files",
            description=f"Now set to: `{max_files}` file(s) per call",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="attach_files_max_file_size")
    async def set_attach_files_max_file_size(self, ctx: commands.Context, max_size_mb: int = 0):
        """Set the maximum file size (in MB) for each attached file.

        **Arguments**
            - `max_size_mb` maximum size in megabytes (1–25, default: 25)

        Use without arguments to view the current setting.
        """
        if max_size_mb == 0:
            current = await self.config.guild(ctx.guild).attach_files_max_file_size_mb()
            embed = discord.Embed(
                title="Attach Files — Max File Size",
                description=f"`{current}` MB per file",
                color=await ctx.embed_color(),
            )
            embed.add_field(
                name="Set",
                value=f"`{ctx.clean_prefix}aiuser functions attach_files_max_file_size <1-25>`",
                inline=False,
            )
            return await ctx.send(embed=embed)

        if max_size_mb < 1 or max_size_mb > 25:
            return await ctx.send("Please provide a value between 1 and 25 MB.")

        await self.config.guild(ctx.guild).attach_files_max_file_size_mb.set(max_size_mb)
        embed = discord.Embed(
            title="Attach Files — Max File Size",
            description=f"Now set to: `{max_size_mb}` MB per file",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="mermaid_diagram")
    async def toggle_mermaid_diagram_function(self, ctx: commands.Context):
        """Enable/disable the LLM's ability to create Mermaid diagram images.

        When enabled, the LLM can render Mermaid diagrams (flowcharts, sequence
        diagrams, class diagrams, etc.) as PNG images attached to responses.
        Uses the 'merm' library for local rendering — no external API needed.
        """
        from aiuser.functions.mermaid.tool_call import MermaidDiagramToolCall
        tool_names = [MermaidDiagramToolCall.function_name]
        await self.toggle_function_helper(ctx, tool_names, "Mermaid Diagram")

    @functions.command(name="mermaid_diagram_theme")
    async def set_mermaid_diagram_theme(self, ctx: commands.Context, theme: str = ""):
        """Set the Mermaid diagram theme.

        Available themes: default, dark, forest, neutral
        Default: dark

        **Arguments**
            - `theme` the theme name (default: dark)

        Use without arguments to view the current setting.
        """
        VALID_THEMES = {"default", "dark", "forest", "neutral"}

        if not theme:
            current = await self.config.guild(ctx.guild).mermaid_diagram_theme()
            embed = discord.Embed(
                title="Mermaid Diagram — Theme",
                description=f"`{current}`",
                color=await ctx.embed_color(),
            )
            embed.add_field(
                name="Available Themes",
                value=", ".join(f"`{t}`" for t in sorted(VALID_THEMES)),
                inline=False,
            )
            embed.add_field(
                name="Set",
                value=f"`{ctx.clean_prefix}aiuser functions mermaid_diagram_theme <theme>`",
                inline=False,
            )
            return await ctx.send(embed=embed)

        theme = theme.lower().strip()
        if theme not in VALID_THEMES:
            return await ctx.send(
                f"Invalid theme. Available themes: {', '.join(sorted(VALID_THEMES))}"
            )

        await self.config.guild(ctx.guild).mermaid_diagram_theme.set(theme)
        embed = discord.Embed(
            title="Mermaid Diagram — Theme",
            description=f"Now set to: `{theme}`",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

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
        """⚠️ DEPRECATED — use `web_search` with `web_search_backend exa` instead.

        This command will auto-migrate your settings to the new generic web_search
        system using the Exa backend. No data is lost.
        """
        await ctx.send("⚠️ The `or_web_search` command is deprecated. Migrating to generic `web_search` with `exa` backend...")
        await self.config.guild(ctx.guild).openrouter_web_search_enabled.set(False)
        await self.config.guild(ctx.guild).web_search_backend.set("exa")
        return await self._toggle_generic_web_function(ctx, "web_search", "Web Search (Exa)")

    @functions.command(name="or_web_fetch")
    async def toggle_or_web_fetch(self, ctx: commands.Context):
        """⚠️ DEPRECATED — use `web_fetch` with `web_fetch_backend exa` instead.

        This command will auto-migrate your settings to the new generic web_fetch
        system using the Exa backend. No data is lost.
        """
        await ctx.send("⚠️ The `or_web_fetch` command is deprecated. Migrating to generic `web_fetch` with `exa` backend...")
        await self.config.guild(ctx.guild).openrouter_web_fetch_enabled.set(False)
        await self.config.guild(ctx.guild).web_fetch_backend.set("exa")
        return await self._toggle_generic_web_function(ctx, "web_fetch", "Web Fetch (Exa)")

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

    @functions.command(name="generate_image_prompt")
    async def config_generate_image_prompt(self, ctx: commands.Context, *, prompt: str = ""):
        """ Set/show/reset the system prompt for image generation

        This system prompt is sent to the image generation model alongside
        the user's prompt. Useful for setting art style, quality expectations,
        or behavioral guidelines. (Separate from the chat persona prompt.)

        To view the current prompt, use without arguments.
        To reset/clear, use `{ctx.clean_prefix}aiuser functions generate_image_prompt reset`
        To set, provide the prompt text directly.

        **Arguments**
            - `prompt` The prompt text to set. Use "reset" to clear it.
        """
        if not prompt:
            # Show current prompt
            current = await self.config.guild(ctx.guild).direct_image_generation_system_prompt()
            if current:
                embed = discord.Embed(
                    title="Image Generation System Prompt",
                    description=current,
                    color=await ctx.embed_color(),
                )
                embed.add_field(name="Reset", value=f"`{ctx.clean_prefix}aiuser functions generate_image_prompt reset`", inline=False)
            else:
                embed = discord.Embed(
                    title="Image Generation System Prompt",
                    description="*No custom system prompt set. The model will receive only the user's prompt.*",
                    color=await ctx.embed_color(),
                )
            return await ctx.send(embed=embed)

        if prompt.lower() in ("reset", "clear", "none"):
            await self.config.guild(ctx.guild).direct_image_generation_system_prompt.set(None)
            return await ctx.send("Image generation system prompt has been reset (cleared).")

        await self.config.guild(ctx.guild).direct_image_generation_system_prompt.set(prompt)
        embed = discord.Embed(
            title="Image Generation System Prompt Updated",
            description=prompt,
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="generate_image_config")
    async def config_generate_image(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure direct image generation parameters using JSON

        Sets the model and image_size that will be used for direct image generation.
        The prompt and aspect_ratio are passed by the LLM, everything else is
        configured here.

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions generate_image_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions generate_image_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions generate_image_config ```json\n{"model": "google/gemini-3.1-flash-image-preview", "image_size": "2K", "reasoning_effort": "medium", "temperature": 0.8, "top_p": 0.95, "service_tier": "flex"}\n``` `

        Valid fields: model, image_size, reasoning_effort, temperature, top_p, service_tier
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="direct_image_generation_parameters",
            tool_type="direct_image_generation",
            param_display_name="Direct Image Generation",
            example_config={
                "model": "google/gemini-3.1-flash-image-preview",
                "image_size": "2K",
                "reasoning_effort": "medium",
                "temperature": 0.8,
                "top_p": 0.95,
                "service_tier": "flex",
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

    @functions.command(name="edit_image_prompt")
    async def config_edit_image_prompt(self, ctx: commands.Context, *, prompt: str = ""):
        """ Set/show/reset the system prompt for image editing

        This system prompt is sent to the image edit model alongside
        the edit prompt. Useful for guiding how edits are applied,
        preserving aspects of the original, or setting style constraints.
        (Separate from the chat persona prompt.)

        To view the current prompt, use without arguments.
        To reset/clear, use `{ctx.clean_prefix}aiuser functions edit_image_prompt reset`
        To set, provide the prompt text directly.

        **Arguments**
            - `prompt` The prompt text to set. Use "reset" to clear it.
        """
        if not prompt:
            # Show current prompt
            current = await self.config.guild(ctx.guild).direct_image_edit_system_prompt()
            if current:
                embed = discord.Embed(
                    title="Image Edit System Prompt",
                    description=current,
                    color=await ctx.embed_color(),
                )
                embed.add_field(name="Reset", value=f"`{ctx.clean_prefix}aiuser functions edit_image_prompt reset`", inline=False)
            else:
                embed = discord.Embed(
                    title="Image Edit System Prompt",
                    description="*No custom system prompt set. The model will receive only the user's prompt.*",
                    color=await ctx.embed_color(),
                )
            return await ctx.send(embed=embed)

        if prompt.lower() in ("reset", "clear", "none"):
            await self.config.guild(ctx.guild).direct_image_edit_system_prompt.set(None)
            return await ctx.send("Image edit system prompt has been reset (cleared).")

        await self.config.guild(ctx.guild).direct_image_edit_system_prompt.set(prompt)
        embed = discord.Embed(
            title="Image Edit System Prompt Updated",
            description=prompt,
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    @functions.command(name="edit_image_config")
    async def config_edit_image(self, ctx: commands.Context, *, json_block: str = ""):
        """ Configure direct image edit parameters using JSON

        Sets the model and image_size that will be used for direct image editing.
        The prompt and image_to_edit are passed by the LLM, everything else is
        configured here.

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions edit_image_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions edit_image_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions edit_image_config ```json\n{"model": "google/gemini-3.1-flash-image-preview", "image_size": "2K", "reasoning_effort": "medium", "temperature": 0.8, "top_p": 0.95, "service_tier": "flex"}\n``` `

        Valid fields: model, image_size, reasoning_effort, temperature, top_p, service_tier
        """
        await self._handle_or_json_config(
            ctx=ctx,
            config_key="direct_image_edit_parameters",
            tool_type="direct_image_edit",
            param_display_name="Edit Image",
            example_config={
                "model": "google/gemini-3.1-flash-image-preview",
                "image_size": "2K",
                "reasoning_effort": "medium",
                "temperature": 0.8,
                "top_p": 0.95,
                "service_tier": "flex",
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
                f"• Set: `{ctx.clean_prefix}aiuser functions {tool_config_name}_config "
                f"```json\n{'{...}'}\n``` `\n"
                f"• Reset: `{ctx.clean_prefix}aiuser functions {tool_config_name}_config reset`"
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

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions or_web_search_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions or_web_search_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions or_web_search_config ```json\n{"engine": "exa", "max_results": 10}\n``` `

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

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions or_web_fetch_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions or_web_fetch_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions or_web_fetch_config ```json\n{"engine": "firecrawl", "max_uses": 5}\n``` `

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

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions or_image_gen_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions or_image_gen_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions or_image_gen_config ```json\n{"model": "openai/dall-e-3", "quality": "high"}\n``` `

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

        To reset parameters to default, use `{ctx.clean_prefix}aiuser functions or_pdf_parsing_config reset`
        To show current parameters, use `{ctx.clean_prefix}aiuser functions or_pdf_parsing_config show`

        Example command:
        `{ctx.clean_prefix}aiuser functions or_pdf_parsing_config ```json\n{"engine": "cloudflare-ai"}\n``` `

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

    # ─── Generic Web Function Helpers ────────────────────────────────────────

    async def _toggle_generic_web_function(self, ctx: commands.Context, function_name: str, display_name: str):
        """Toggle a generic web function in the function_calling_functions list."""
        from aiuser.functions.web_search.tool_call import WebSearchToolCall
        from aiuser.functions.web_fetch.tool_call import WebFetchToolCall
        from aiuser.functions.web_answer.tool_call import WebAnswerToolCall

        name_to_class = {
            "web_search": WebSearchToolCall,
            "web_fetch": WebFetchToolCall,
            "web_answer": WebAnswerToolCall,
        }
        tool_class = name_to_class.get(function_name)
        if not tool_class:
            return await ctx.send(f":warning: Unknown function: `{function_name}`")

        tool_names = [tool_class.function_name]
        await self.toggle_function_helper(ctx, tool_names, display_name)

    async def _handle_web_config(
        self,
        ctx: commands.Context,
        config_key: str,
        param_display_name: str,
        valid_keys: list,
        example_config: dict,
        *,
        json_block: str,
    ):
        """Shared helper for web_*_config commands.

        - No args or 'show'/'list' → shows current config + example
        - 'reset'/'clear' → resets to defaults
        - JSON code block → sets config (unrecognized keys warn but are saved)
        """
        if not json_block or json_block in ("show", "list"):
            current_json = await getattr(self.config.guild(ctx.guild), config_key)()
            current = json.loads(current_json) if current_json else {}

            embed = discord.Embed(
                title=f"Web {param_display_name} Configuration",
                color=await ctx.embed_color(),
            )

            if current:
                for k, v in current.items():
                    embed.add_field(name=k, value=f"`{v}`", inline=True)
            else:
                embed.add_field(name="Current", value="No custom parameters set. Using defaults.", inline=False)

            example_json = json.dumps(example_config, indent=2)
            embed.add_field(
                name="Reference Example (JSON)",
                value=f"```json\n{example_json}\n```",
                inline=False,
            )

            config_cmd = config_key.replace("_", " ")
            usage = (
                f"• Set: `{ctx.clean_prefix}aiuser functions {config_cmd} "
                f"```json\n{{...}}\n``` `\n"
                f"• Reset: `{ctx.clean_prefix}aiuser functions {config_cmd} reset`"
            )
            embed.add_field(name="Usage", value=usage, inline=False)
            return await ctx.send(embed=embed)

        if json_block in ("reset", "clear"):
            await getattr(self.config.guild(ctx.guild), config_key).set(None)
            return await ctx.send(f"Web {param_display_name} configuration reset to defaults.")

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

        unknown_keys = [k for k in data if k not in valid_keys]
        if unknown_keys:
            await ctx.send(
                f":information_source: Unrecognized key(s): {', '.join(f'`{k}`' for k in unknown_keys)}. "
                f"These will be saved but may be ignored by the current backend. "
                f"Recognized keys: {', '.join(f'`{k}`' for k in valid_keys)}."
            )

        serialized = json.dumps(data)
        await getattr(self.config.guild(ctx.guild), config_key).set(serialized)

        embed = discord.Embed(
            title=f"Web {param_display_name} Configuration Updated",
            description=f"Parameters saved: {serialized}",
            color=await ctx.embed_color(),
        )
        await ctx.send(embed=embed)

    # ─── Generic Web Search ──────────────────────────────────────────────────

    @functions.command(name="web_search")
    async def toggle_web_search(self, ctx: commands.Context):
        """Toggle the generic web_search function.

        Before enabling, set a backend with `web_search_backend` (exa/serper).
        The LLM will only pass a query; all other parameters are owner-configured.
        """
        await self._toggle_generic_web_function(ctx, "web_search", "Web Search")

    @functions.command(name="web_search_backend")
    async def set_web_search_backend(self, ctx: commands.Context, backend: str = ""):
        """Set the web_search backend provider.

        **Arguments**
        - `backend`: `exa` (requires Exa API key), `serper` (requires Serper.dev key)

        Example: `{ctx.clean_prefix}aiuser functions web_search_backend exa`
        """
        valid_backends = ["exa", "serper"]
        if not backend or backend not in valid_backends:
            return await ctx.send(
                f"Please specify a backend: {', '.join(f'`{b}`' for b in valid_backends)}.\n"
                f"Example: `{ctx.clean_prefix}aiuser functions web_search_backend exa`"
            )

        if backend == "exa":
            if not (await self.bot.get_shared_api_tokens("exa")).get("api_key"):
                return await ctx.send(
                    f"Exa API key not set! Set it using `{ctx.clean_prefix}set api exa api_key,YOUR_KEY`."
                )
        elif backend == "serper":
            if not (await self.bot.get_shared_api_tokens("serper")).get("api_key"):
                return await ctx.send(
                    f"Serper.dev key not set! Set it using `{ctx.clean_prefix}set api serper api_key,APIKEY`."
                )

        # Disable OpenRouter web search if switching to local backend
        if await self.config.guild(ctx.guild).openrouter_web_search_enabled():
            await self.config.guild(ctx.guild).openrouter_web_search_enabled.set(False)
            await ctx.send("⚠️ OpenRouter web search has been disabled.")

        await self.config.guild(ctx.guild).web_search_backend.set(backend)
        embed = discord.Embed(
            title="Web Search Backend",
            description=f"Now set to: `{backend}`",
            color=await ctx.embed_color(),
        )
        embed.add_field(
            name="Next Step",
            value=f"Enable with `{ctx.clean_prefix}aiuser functions web_search`",
            inline=False,
        )
        await ctx.send(embed=embed)

    @functions.command(name="web_search_config")
    async def config_web_search(self, ctx: commands.Context, *, json_block: str = ""):
        """Configure web_search backend parameters using JSON.

        Any JSON fields are accepted — each backend picks the fields it cares about.
        Unrecognized fields are saved but may be silently ignored.

        **Exa recognized keys**: `num_results` (int), `type` (auto/fast/deep-lite/deep/deep-reasoning),
        `include_domains` (list), `exclude_domains` (list),
        `start_published_date` (str YYYY-MM-DD), `end_published_date` (str YYYY-MM-DD)

        **Serper recognized keys**: *(none — no configurable parameters)*

        Example: `{ctx.clean_prefix}aiuser functions web_search_config ```json\n{"num_results": 10, "type": "auto"}\n``` `
        """
        await self._handle_web_config(
            ctx=ctx,
            config_key="web_search_config",
            param_display_name="Search",
            valid_keys=["num_results", "type", "include_domains", "exclude_domains",
                        "start_published_date", "end_published_date"],
            example_config={"num_results": 10, "type": "auto"},
            json_block=json_block,
        )

    # ─── Generic Web Fetch ───────────────────────────────────────────────────

    @functions.command(name="web_fetch")
    async def toggle_web_fetch(self, ctx: commands.Context):
        """Toggle the generic web_fetch function.

        Before enabling, set a backend with `web_fetch_backend` (exa/scrape).
        The LLM will only pass URLs; all other parameters are owner-configured.
        """
        await self._toggle_generic_web_function(ctx, "web_fetch", "Web Fetch")

    @functions.command(name="web_fetch_backend")
    async def set_web_fetch_backend(self, ctx: commands.Context, backend: str = ""):
        """Set the web_fetch backend provider.

        **Arguments**
        - `backend`: `exa` (requires Exa API key), `scrape` (direct scraping, no API key needed)

        Example: `{ctx.clean_prefix}aiuser functions web_fetch_backend exa`
        """
        valid_backends = ["exa", "scrape"]
        if not backend or backend not in valid_backends:
            return await ctx.send(
                f"Please specify a backend: {', '.join(f'`{b}`' for b in valid_backends)}.\n"
                f"Example: `{ctx.clean_prefix}aiuser functions web_fetch_backend exa`"
            )

        if backend == "exa":
            if not (await self.bot.get_shared_api_tokens("exa")).get("api_key"):
                return await ctx.send(
                    f"Exa API key not set! Set it using `{ctx.clean_prefix}set api exa api_key,YOUR_KEY`."
                )

        # Disable OpenRouter web fetch if switching to local backend
        if await self.config.guild(ctx.guild).openrouter_web_fetch_enabled():
            await self.config.guild(ctx.guild).openrouter_web_fetch_enabled.set(False)
            await ctx.send("⚠️ OpenRouter web fetch has been disabled.")

        await self.config.guild(ctx.guild).web_fetch_backend.set(backend)
        embed = discord.Embed(
            title="Web Fetch Backend",
            description=f"Now set to: `{backend}`",
            color=await ctx.embed_color(),
        )
        embed.add_field(
            name="Next Step",
            value=f"Enable with `{ctx.clean_prefix}aiuser functions web_fetch`",
            inline=False,
        )
        await ctx.send(embed=embed)

    @functions.command(name="web_fetch_config")
    async def config_web_fetch(self, ctx: commands.Context, *, json_block: str = ""):
        """Configure web_fetch backend parameters using JSON.

        Any JSON fields are accepted — each backend picks the fields it cares about.
        Unrecognized fields are saved but may be silently ignored.

        **Exa recognized keys**: `text` (bool), `summary` (bool)

        **Scrape recognized keys**: *(none — no configurable parameters)*

        Example: `{ctx.clean_prefix}aiuser functions web_fetch_config ```json\n{"text": true}\n``` `
        """
        await self._handle_web_config(
            ctx=ctx,
            config_key="web_fetch_config",
            param_display_name="Fetch",
            valid_keys=["text", "summary"],
            example_config={"text": True},
            json_block=json_block,
        )

    # ─── Generic Web Answer ──────────────────────────────────────────────────

    @functions.command(name="web_answer")
    async def toggle_web_answer(self, ctx: commands.Context):
        """Toggle the generic web_answer function.

        Before enabling, set a backend with `web_answer_backend` (only `exa` currently).
        The LLM will only pass a question; all other parameters are owner-configured.
        """
        await self._toggle_generic_web_function(ctx, "web_answer", "Web Answer")

    @functions.command(name="web_answer_backend")
    async def set_web_answer_backend(self, ctx: commands.Context, backend: str = ""):
        """Set the web_answer backend provider.

        **Arguments**
        - `backend`: `exa` (requires Exa API key)

        Example: `{ctx.clean_prefix}aiuser functions web_answer_backend exa`
        """
        valid_backends = ["exa"]
        if not backend or backend not in valid_backends:
            return await ctx.send(
                f"Please specify a backend: {', '.join(f'`{b}`' for b in valid_backends)}.\n"
                f"Example: `{ctx.clean_prefix}aiuser functions web_answer_backend exa`"
            )

        if backend == "exa":
            if not (await self.bot.get_shared_api_tokens("exa")).get("api_key"):
                return await ctx.send(
                    f"Exa API key not set! Set it using `{ctx.clean_prefix}set api exa api_key,YOUR_KEY`."
                )

        await self.config.guild(ctx.guild).web_answer_backend.set(backend)
        embed = discord.Embed(
            title="Web Answer Backend",
            description=f"Now set to: `{backend}`",
            color=await ctx.embed_color(),
        )
        embed.add_field(
            name="Next Step",
            value=f"Enable with `{ctx.clean_prefix}aiuser functions web_answer`",
            inline=False,
        )
        await ctx.send(embed=embed)

    @functions.command(name="web_answer_config")
    async def config_web_answer(self, ctx: commands.Context, *, json_block: str = ""):
        """Configure web_answer backend parameters using JSON.

        Any JSON fields are accepted — each backend picks the fields it cares about.
        Unrecognized fields are saved but may be silently ignored.

        **Exa recognized keys**: `text` (bool — include full citation text)

        Example: `{ctx.clean_prefix}aiuser functions web_answer_config ```json\n{"text": false}\n``` `
        """
        await self._handle_web_config(
            ctx=ctx,
            config_key="web_answer_config",
            param_display_name="Answer",
            valid_keys=["text"],
            example_config={"text": False},
            json_block=json_block,
        )

    # ─── Legacy Migration ────────────────────────────────────────────────────

    async def _migrate_legacy_function_names(self):
        """Migrate legacy function_calling_functions entries to generic names.

        Called during cog load. Converts old function names and enables
        appropriate backends so existing setups continue working.
        """
        for guild_id in await self.config.all_guilds():
            guild_config = self.config.guild_from_id(guild_id)
            enabled = await guild_config.function_calling_functions()
            if not enabled:
                continue

            modified = False

            # Migrate search_google → web_search + serper backend
            if "search_google" in enabled:
                enabled.remove("search_google")
                if "web_search" not in enabled:
                    enabled.append("web_search")
                modified = True
                backend = await guild_config.web_search_backend()
                if not backend:
                    await guild_config.web_search_backend.set("serper")

            # Migrate open_url → web_fetch + scrape backend
            if "open_url" in enabled:
                enabled.remove("open_url")
                if "web_fetch" not in enabled:
                    enabled.append("web_fetch")
                modified = True
                backend = await guild_config.web_fetch_backend()
                if not backend:
                    await guild_config.web_fetch_backend.set("scrape")

            if modified:
                await guild_config.function_calling_functions.set(enabled)
                logger.info("Migrated legacy function names for guild %s to generic web_* functions", guild_id)
