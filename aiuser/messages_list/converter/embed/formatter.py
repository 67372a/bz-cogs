from discord import Message

from aiuser.types.abc import MixinMeta
from aiuser.config.constants import URL_PATTERN
from aiuser.utils.utilities import contains_youtube_link
from aiuser.functions.scrape.tool_call import ScrapeToolCall
from aiuser.messages_list.converter.embed.youtube import format_youtube_embed
from xml.sax.saxutils import escape
from aiuser.messages_list.converter.helpers import (
    _get_msg_header,
    first_rich_embed,
)


async def format_embed_content(cog: MixinMeta, message: Message):
    yt_api_key = (await cog.bot.get_shared_api_tokens("youtube")).get("api_key")
    if (yt_api_key and contains_youtube_link(message.content)):
        return await format_youtube_embed(yt_api_key, message)
    elif (URL_PATTERN.search(message.content) and ScrapeToolCall.function_name in await cog.config.guild(message.guild).function_calling_functions()):
        return None
    else:
        # Rich embeds only — Discord link-preview unfurls are excluded so the
        # serialized form cannot flip when previews arrive asynchronously.
        embed = first_rich_embed(message)
        if embed is None:
            return None
        title = escape(embed.title or "")
        desc = escape(embed.description or "")

        xml_content = f'<embed title="{title}">{desc}</embed>'

        return f'{_get_msg_header(message)}{xml_content}</message>'
     
    
async def format_bot_embed_content(cog: MixinMeta, message: Message):
    # This handles the bot's own internal "Thought" or "Response" embeds
    # Wrap in XML metadata so the LLM can correlate with message IDs and timestamps
    embed = first_rich_embed(message)
    if embed is None:
        return None
    description = embed.description
    if not description:
        return None
    return f'{_get_msg_header(message)}{description}</message>'
