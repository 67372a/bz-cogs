import logging
from datetime import timezone
from io import BytesIO, TextIOWrapper
import base64
from xml.sax.saxutils import escape
from discord import Message, MessageType

from aiuser.config.constants import URL_PATTERN

logger = logging.getLogger("red.bz_cogs.aiuser")

def _get_msg_header(message: Message) -> str:
    """Helper to generate standard XML message header"""
    # Escape quotes and special chars to prevent XML breakage
    author_name = escape(message.author.name)
    display_name = escape(message.author.display_name)
    
    reply_info = ""
    if message.reference and isinstance(getattr(message.reference, 'resolved', None), Message):
        reply_message = message.reference.resolved
        reply_author_name = escape(reply_message.author.name)
        reply_author_displayname = escape(reply_message.author.display_name)
        reply_info = (f' reply_to_id="{reply_message.id}" '
                      f'reply_target_id="{reply_message.author.id}" '
                      f'reply_target_username="{reply_author_name}" '
                      f'reply_target_displayname="{reply_author_displayname}"')
    elif message.reference and getattr(message.reference, 'message_id', None):
        reply_info = f' reply_to_id="{message.reference.message_id}"'

    timestamp = message.created_at.astimezone(timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')
    return (f'<message id="{message.id}" '
            f'timestamp="{timestamp}" '
            f'author_id="{message.author.id}" '
            f'username="{author_name}" '
            f'displayname="{display_name}"{reply_info}>')

def format_text_content(message: Message):
    if message.type == MessageType.new_member:
        return f'{_get_msg_header(message)}Joined the server.</message>'
    
    if not message.content or message.content == "" or message.content.isspace():
        return None
        
    content = mention_to_text(message)
    
    # If it's the bot's own message, we return just the content 
    # TODO: determine if should apply XML to responses
    if message.author.id == message.guild.me.id:
        return content

    # Standard User Message
    return f'{_get_msg_header(message)}{content}</message>'

def format_embed_text_content(message: Message):
    content = mention_to_text(message)
    content = URL_PATTERN.sub("", content)
    if not content or content == "" or content.isspace():
        return None
    
    if message.author.id == message.guild.me.id:
        return content
        
    return f'{_get_msg_header(message)}{content}</message>'


_QUOTE_ESCAPES = {'"': "&" + "quot;"}


def format_generic_image(message: Message):
    title = f' title="{escape(message.attachments[0].title, _QUOTE_ESCAPES)}"' if message.attachments[0].title else ""
    desc = f' description="{escape(message.attachments[0].description, _QUOTE_ESCAPES)}"' if message.attachments[0].description else ""
    filename = escape(message.attachments[0].filename, _QUOTE_ESCAPES)

    xml_content = f'<image filename="{filename}"{title}{desc}/>'

    if message.author.id == message.guild.me.id:
        return f"Sent {xml_content}"
        
    return f'{_get_msg_header(message)}{xml_content}</message>'

def format_generic_document(message: Message):
    filename = escape(message.attachments[0].filename, _QUOTE_ESCAPES)
    xml_content = f'<file filename="{filename}"/>'
    
    if message.author.id == message.guild.me.id:
        return f"Sent {xml_content}"
        
    return f'{_get_msg_header(message)}{xml_content}</message>'

async def format_binary_document(message: Message):
    attachment = message.attachments[0]

    content = []

    with BytesIO() as buffer: # Use BytesIO as a context manager
        await attachment.save(buffer)

        buffer.seek(0)  # Reset buffer pointer to the beginning for reading
        binary_data = buffer.read()

    base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
    mime_type = get_mime_type(attachment.content_type) 

    file_data = f"data:{mime_type};base64,{base64_encoded_data}"

    content.append(
        {
            "type": "file", 
            "file": 
            {
                "filename": attachment.filename,
                "file_data": file_data
            }
        })
    
    if message.content != "":
        content.append({"type": "text", "text": format_text_content(message)})
        
    return content

async def format_text_document(message: Message):
    attachment = message.attachments[0]
    content = []
    
    if message.content != "":
        content.append({"type": "text", "text": format_text_content(message)})

    with BytesIO() as buffer:
        await attachment.save(buffer)
        buffer.seek(0)
        text_data = TextIOWrapper(buffer, encoding='utf-8').read()

    # Sanitizing content to avoid breaking XML structure if it contains tags
    # Though LLMs are usually smart enough to figure it out, explicit CDATA or escaping is safer
    # For simplicity here, we assume standard text, but escaping is recommended:
    text_data = escape(text_data) 
    
    filename = escape(attachment.filename, _QUOTE_ESCAPES)
    
    # We use a custom tag structure here
    doc_xml = (f'{_get_msg_header(message)}'
               f'<document filename="{filename}">'
               f'{text_data}'
               f'</document>'
               f'</message>')

    content.append({"type": "text", "text": doc_xml})
    return content

async def format_sticker_content(message: Message):
    try:
        sticker = await message.stickers[0].fetch()
        sticker_name = escape(sticker.name)
        desc_attr = f' description="{escape(sticker.description)}"' if getattr(sticker, "description", "") else ""
        
        xml_content = f'<sticker name="{sticker_name}"{desc_attr}/>'
    except Exception:
        sticker_name = escape(message.stickers[0].name)
        xml_content = f'<sticker name="{sticker_name}"/>'

    if message.author.id == message.guild.me.id:
        return f"Sent {xml_content}"

    return f'{_get_msg_header(message)}{xml_content}</message>'


def mention_to_text(message: Message) -> str:
    """
    Converts mentions to text
    """
    content = message.content
    mentions = message.mentions + message.role_mentions + message.channel_mentions

    if not mentions:
        return content

    for mentioned in mentions:
        if mentioned in message.channel_mentions:
            content = content.replace(mentioned.mention, f'#{mentioned.name}')
        elif mentioned in message.role_mentions:
            content = content.replace(mentioned.mention, f'@{mentioned.name}')
        else:
            content = content.replace(mentioned.mention, f'@{mentioned.name}')

    return content

def get_mime_type(content_type_string):
    """
    Parses a content type string and returns just the main type/subtype,
    stripping away any parameters like 'charset'.

    Args:
        content_type_string (str): The full content type string,
            e.g., "text/plain; charset=utf-8", "application/json",
            "  image/jpeg ; quality=0.8  ".
            Can be None or an empty string.

    Returns:
        str: The main type/subtype (e.g., "text/plain", "application/json", "image/jpeg").
             Returns an empty string if the input is None, empty, or contains only whitespace,
             or if the part before the first semicolon is empty after stripping.
    """
    if not content_type_string:
        return ""

    # Strip leading/trailing whitespace from the whole string first
    stripped_content_type = content_type_string.strip()

    if not stripped_content_type:
        return ""

    # Split the string at the first semicolon
    # The `maxsplit=1` argument ensures we only split on the first occurrence
    parts = stripped_content_type.split(';', 1)

    # The main content type is the first part.
    # Strip any whitespace from this part as well (e.g., "text/plain " -> "text/plain")
    main_type = parts[0].strip()

    return main_type