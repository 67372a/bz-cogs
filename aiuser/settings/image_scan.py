import importlib
import logging

import discord
from redbot.core import checks, commands

from aiuser.config.constants import OPENROUTER_URL
from aiuser.config.models import VISION_SUPPORTED_MODELS
from aiuser.types.abc import MixinMeta, aiuser
from aiuser.types.enums import ScanImageMode

logger = logging.getLogger("red.bz_cogs.aiuser")


class ImageScanSettings(MixinMeta):
    @aiuser.group()
    @checks.is_owner()
    async def imagescan(self, _):
        """ Change the image scan setting

            Go [here](https://github.com/zhaobenny/bz-cogs/tree/main/aiuser#image-scanning-%EF%B8%8F) for more info.

            (All subcommands are per server)
        """
        pass

    @imagescan.command(name="toggle")
    async def image_scanning(self, ctx: commands.Context):
        """ Toggle image scanning """
        value = not (await self.config.guild(ctx.guild).scan_images())
        await self.config.guild(ctx.guild).scan_images.set(value)
        embed = discord.Embed(
            title="Scanning Images for this server now set to:",
            description=f"`{value}`",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @imagescan.command(name="maxsize")
    async def image_maxsize(self, ctx: commands.Context, size: float):
        """ Set max download size in Megabytes for image scanning """
        await self.config.guild(ctx.guild).max_image_size.set(size * 1024 * 1024)
        embed = discord.Embed(
            title="Max download size to scan images now set to:",
            description=f"`{size:.2f}` MB",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @imagescan.command(name="resolution", aliases=["maxpixels"])
    async def image_resolution(self, ctx: commands.Context, resolution: float):
        """ Set max image resolution in Megapixels (MP) for image scanning. 
        
        Images larger than this will be scaled down. 
        Set to 0 to reset to defaults (16.8MP for LLM, 1MP for others).
        """
        if resolution == 0:
            await self.config.guild(ctx.guild).max_image_pixels.set(None)
            embed = discord.Embed(
                title="Max resolution for scanning images reset to default.",
                color=await ctx.embed_color())
            return await ctx.send(embed=embed)

        if resolution < 0.01:
             return await ctx.send("Please set a reasonable resolution (in MP).")
            
        pixels = int(resolution * 1000000)
        
        await self.config.guild(ctx.guild).max_image_pixels.set(pixels)
        embed = discord.Embed(
            title="Max resolution for scanning images now set to:",
            description=f"`{resolution:.2f}` MP ({pixels:,} pixels)",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @imagescan.command(name="mode")
    async def image_mode(self, ctx: commands.Context, mode: str):  # Modify the parameter type
        """ Set method for scanning images


            **Arguments**
            - `mode` One of the following: `local`, `ai-horde`, `supported-llm`
        """
        values = [m.value for m in ScanImageMode]

        if mode not in values:
            await ctx.send(":warning: Invalid mode.")

        if mode == "list" or mode not in values:
            embed = discord.Embed(
                title="Valid modes:",
                description="\n".join(values),
                color=await ctx.embed_color())
            return await ctx.send(embed=embed)

        mode = ScanImageMode(mode)
        if mode == ScanImageMode.LOCAL:
            try:
                importlib.import_module("pytesseract")
                importlib.import_module("torch")
                importlib.import_module("transformers")
                await self.config.guild(ctx.guild).scan_images_mode.set(ScanImageMode.LOCAL.value)
                embed = discord.Embed(title="Scanning Images for this server now set to", color=await ctx.embed_color())
                embed.add_field(
                    name="❗ __WILL CAUSE HEAVY CPU LOAD__ ❗", value=f"`{mode.value}`", inline=False)
                return await ctx.send(embed=embed)
            except Exception:
                logger.error(
                    "Image processing dependencies import failed. ", exc_info=True)
                await self.config.guild(ctx.guild).scan_images_mode.set(ScanImageMode.AI_HORDE.value)
                return await ctx.send(":warning: Local image processing dependencies not available. Please install them (see cog README.md) to use this feature locally.")
        elif mode == ScanImageMode.AI_HORDE:
            await self.config.guild(ctx.guild).scan_images_mode.set(ScanImageMode.AI_HORDE.value)
            embed = discord.Embed(title="Scanning Images for this server now set to", description=f"`{mode.value}`", color=await ctx.embed_color())
            embed.add_field(
                name="👁️ __PRIVACY WARNING__",
                value="This will send image attachments to a random volunteer worker machine for processing!",
                inline=False)
            if (await self.bot.get_shared_api_tokens('ai-horde')).get("api_key"):
                key_description = "Key set."
            else:
                key_description = f"No key set. \n Request will be lower priority.\n  \
                                Create one [here](https://stablehorde.net/#:~:text=0%20alchemy%20forms.-,Usage,-First%20Register%20an)\
                                and set it with `{ctx.clean_prefix}set api ai-horde api_key,API_KEY`"
            embed.add_field(
                name="AI Horde API key:",
                value=key_description,
                inline=False)
            embed.add_field(
                name="Reminder",
                value="AI Horde is a crowdsourced volunteer service. \n Please contribute back if heavily used. See [here](https://stablehorde.net/)",
                inline=False)
            return await ctx.send(embed=embed)
        elif mode == ScanImageMode.LLM:
            await self.config.guild(ctx.guild).scan_images_mode.set(ScanImageMode.LLM.value)
            embed = discord.Embed(title="Scanning Images for this server now set to", description=f"`{mode.value}`", color=await ctx.embed_color())

            embed.add_field(
                name="👁️ __PRIVACY WARNING__",
                value="This will send image attachments to the specified endpoint for processing!",
                inline=False)
            
            model = await self.config.guild(ctx.guild).model()
            if model not in VISION_SUPPORTED_MODELS:
                embed.add_field(
                    name=":warning: Unvalidated Model",
                    value=f"Set image scanning model to `{model}` but it has not been validated for image scanning.",
                    inline=False)
                            
            await self.config.guild(ctx.guild).scan_images_model.set(model)
            return await ctx.send(embed=embed)
        
    @imagescan.group(name="urls")
    async def image_urls(self, _):
        """ Configure scanning images from URLs in messages

            (All subcommands are per server)
        """
        pass

    @image_urls.command(name="toggle")
    async def image_url_scanning(self, ctx: commands.Context):
        """ Toggle scanning images from URLs in message text """
        value = not (await self.config.guild(ctx.guild).openrouter_image_parsing_enabled())
        await self.config.guild(ctx.guild).openrouter_image_parsing_enabled.set(value)
        embed = discord.Embed(
            title="Scanning Images from URLs now set to:",
            description=f"`{value}`",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @image_urls.command(name="maxsize")
    async def image_url_maxsize(self, ctx: commands.Context, size: float):
        """ Set max download size in Megabytes for image URLs """
        await self.config.guild(ctx.guild).openrouter_image_parsing_max_size.set(size * 1024 * 1024)
        embed = discord.Embed(
            title="Max download size for image URLs now set to:",
            description=f"`{size:.2f}` MB",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @imagescan.command(name="model")
    async def image_model(self, ctx: commands.Context, model_name: str):
        """ Set the specific LLM used in the `supported-llm` mode


        **Arguments**
            - `model_name` Name of a compatible model
        """
        custom_endpoint = await self.config.custom_openai_endpoint()
        warning_message = None

        if (not custom_endpoint or custom_endpoint.startswith(OPENROUTER_URL)):
            await ctx.message.add_reaction("🔄")
            models = [model.id for model in (await self.openai_client.models.list()).data]
            models = list(set(models) & set(VISION_SUPPORTED_MODELS))  # only show supported models
            await ctx.message.remove_reaction("🔄", ctx.me)

            if model_name not in models:
                warning_message = "⚠️ Model has not been validated for image scanning."


        await self.config.guild(ctx.guild).scan_images_model.set(model_name)
        embed = discord.Embed(
            title="LLM for scanning images now set to:",
            description=f"`{model_name}`",
            color=await ctx.embed_color())
        if warning_message:
            embed.set_footer(text=warning_message)
        return await ctx.send(embed=embed)
