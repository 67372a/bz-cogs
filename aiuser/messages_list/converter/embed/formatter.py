
from discord import Message

from aiuser.types.abc import MixinMeta
from aiuser.config.constants import URL_PATTERN
from aiuser.utils.utilities import contains_youtube_link
from aiuser.functions.scrape.tool_call import ScrapeToolCall
from aiuser.messages_list.converter.embed.youtube import format_youtube_embed
from xml.sax.saxutils import escape
from aiuser.messages_list.converter.helpers import _get_msg_header


async def format_embed_content(cog: MixinMeta, message: Message):
    yt_api_key = (await cog.bot.get_shared_api_tokens("youtube")).get("api_key")
    if (yt_api_key and contains_youtube_link(message.content)):
        return await format_youtube_embed(yt_api_key, message)
    elif (URL_PATTERN.search(message.content) and ScrapeToolCall.function_name in await cog.config.guild(message.guild).function_calling_functions()):
        return None
    else:
        title = escape(message.embeds[0].title or "")
        desc = escape(message.embeds[0].description or "")
        
        xml_content = f'<embed title="{title}">{desc}</embed>'
        
        return f'{_get_msg_header(message)}{xml_content}</message>'
    
async def format_bot_embed_content(cog: MixinMeta, message: Message):
    # This handles the bot's own internal "Thought" or "Response" embeds
    # We generally want to return raw text here so the LLM sees its own past thoughts as text
    return message.embeds[0].description
