import logging
from io import BytesIO, TextIOWrapper
import base64

from discord import Message, MessageType

from aiuser.config.constants import URL_PATTERN

logger = logging.getLogger("red.bz_cogs.aiuser")


def format_text_content(message: Message):
    if message.type == MessageType.new_member:
        return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] joined the server as a new member.'
    if not message.content or message.content == "" or message.content.isspace():
        return None
    content = mention_to_text(message)
    if message.author.id == message.guild.me.id:
        return f'{content}'
    return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] said "{content}"'


def format_embed_text_content(message: Message):
    content = mention_to_text(message)
    content = URL_PATTERN.sub("", content)
    if not content or content == "" or content.isspace():
        return None
    if message.author.id == message.guild.me.id:
        return f'{content}'
    return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] said "{content}"'


def format_generic_image(message: Message):
    title = f', title "{message.attachments[0].title}"' if message.attachments[0].title else ""
    description = f', description "{message.attachments[0].description}"' if message.attachments[0].description else ""

    if message.author.id == message.guild.me.id:
        return f'An image with the filename "{message.attachments[0].filename}{title}{description}."'
    return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] sent an image with the filename "{message.attachments[0].filename}"{title}{description}."'

def format_generic_document(message: Message):
    title = f', title "{message.attachments[0].title}"' if message.attachments[0].title else ""
    description = f', description "{message.attachments[0].description}"' if message.attachments[0].description else ""

    if message.author.id == message.guild.me.id:
        return f'A document with the filename "{message.attachments[0].filename}{title}{description}."'
    return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] sent a document with the filename "{message.attachments[0].filename}"{title}{description}."'

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

    with BytesIO() as buffer: # Use BytesIO as a context manager
        await attachment.save(buffer)

        buffer.seek(0)  # Reset buffer pointer to the beginning for reading
        text_data = TextIOWrapper(buffer, encoding='utf-8').read()

    mime_type = get_mime_type(attachment.content_type)

    title = f', title "{message.attachments[0].title}"' if message.attachments[0].title else ""
    description = f', description "{message.attachments[0].description}"' if message.attachments[0].description else ""
    document_content = f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] sent a document with the filename "{message.attachments[0].filename}{title}{description}."\n<DOCUMENT_START>{text_data}<DOCUMENT_END>'

    content.append(
        {
            "type": "text", 
            "text": document_content
        })
    return content

async def format_sticker_content(message: Message):
    try:
        sticker = await message.stickers[0].fetch()
        description = getattr(sticker,"description","")
        description_text = f' and description "{description}"' if description else ""
        return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] sent a sticker with name "{sticker.name}"{description_text}.'
    except Exception:
        sticker_name = message.stickers[0].name
        return f'[MESSAGE_ID={message.id} TIMESTAMP={message.created_at.isoformat()} USER_ID={message.author.id} USERNAME="{message.author.name}" NICKNAME="{message.author.name}"] sent a sticker with name "{sticker_name}"'


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