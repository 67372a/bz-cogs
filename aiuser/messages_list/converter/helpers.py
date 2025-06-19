import logging
from io import BytesIO, TextIOWrapper
import base64

from discord import Message, MessageType

from aiuser.config.constants import URL_PATTERN

logger = logging.getLogger("red.bz_cogs.aiuser")


def format_text_content(message: Message):
    if message.type == MessageType.new_member:
        return f'User "{message.author.name}" with display name "{message.author.display_name}" has joined the server. Their Discord ID is {message.author.id}'
    if not message.content or message.content == "" or message.content.isspace():
        return None
    content = mention_to_text(message)
    if message.author.id == message.guild.me.id:
        return f'{content}'
    return f'User "{message.author.name}" with display name "{message.author.display_name}" said: {content}'


def format_embed_text_content(message: Message):
    content = mention_to_text(message)
    content = URL_PATTERN.sub("", content)
    if not content or content == "" or content.isspace():
        return None
    if message.author.id == message.guild.me.id:
        return f'{content}'
    return f'User "{message.author.name}" with display name "{message.author.display_name}" said: {content}'


def format_generic_image(message: Message):
    if message.author.id == message.guild.me.id:
        return f'[Image: "{message.attachments[0].filename}"]'
    return f'User "{message.author.name}" with display name "{message.author.display_name}" sent: [Image: "{message.attachments[0].filename}"]'

def format_generic_document(message: Message):
    if message.author.id == message.guild.me.id:
        return f'[Document: "{message.attachments[0].filename}"]'
    return f'User "{message.author.name}" with display name "{message.author.display_name}" sent: [Document: "{message.attachments[0].filename}"]'

async def format_binary_document(message: Message):
    attachment = message.attachments[0]

    content = []
    if message.content != "":
        content.append({"type": "text", "text": format_text_content(message)})

    with BytesIO() as buffer: # Use BytesIO as a context manager
        await attachment.save(buffer)
        logger.info(f"Attachment '{attachment.filename}' saved to buffer.")

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
    return content

async def format_text_document(message: Message):
    attachment = message.attachments[0]

    content = []
    if message.content != "":
        content.append({"type": "text", "text": format_text_content(message)})

    with BytesIO() as buffer: # Use BytesIO as a context manager
        await attachment.save(buffer)
        logger.info(f"Attachment '{attachment.filename}' saved to buffer.")

        buffer.seek(0)  # Reset buffer pointer to the beginning for reading
        text_data = TextIOWrapper(buffer, encoding='utf-8').read()

    mime_type = get_mime_type(attachment.content_type)

    document_content = f'User "{message.author.name}" with display name "{message.author.display_name}" sent: [Document: "{message.attachments[0].filename}" Type: "{mime_type}"\n<DOCUMENT_START>{text_data}<DOCUMENT_END>]'

    content.append(
        {
            "type": "text", 
            "text": document_content
        })
    return content

async def format_sticker_content(message: Message):
    try:
        sticker = await message.stickers[0].fetch()
        description = sticker.description or ""
        description_text = f' and description "{description}"' if description else ""
        return f'User "{message.author.name}" with display name "{message.author.display_name}" sent: [Sticker with name "{sticker.name}"{description_text}]'
    except Exception:
        sticker_name = message.stickers[0].name
        return f'User "{message.author.name}" with display name "{message.author.display_name}" sent: [Sticker with name "{sticker_name}"]'


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