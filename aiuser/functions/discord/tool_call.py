import logging

import discord
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.utils.utilities import to_thread, get_guild_emoji_map

logger = logging.getLogger("red.bz_cogs.aiuser")

#
# React to Message ToolCall
#

reactToMessageToolCallSchema = ToolCallSchema(
    function=Function(
        name="react_to_message",
        description="Reacts to a recent message with a specified emoji. Reaction emoji are used to indicate how one feels about a message.",
        parameters=Parameters(
            properties={
                "message_id": {
                    "type": "string",
                    "description": "The ID of the specific recent message to react to.",
                },
                "emoji": {
                    "type": "string",
                    "description": "The name of a server emoji, all lowercase.",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for reacting to the message.",
                },
            },
            required=["message_id", "emoji", "reason"],
        ),
    )
)


class ReactToMessageToolCall(ToolCall):
    schema = reactToMessageToolCallSchema
    function_name = reactToMessageToolCallSchema.function.name

    async def _handle(self, arguments: dict):
        guild = self.ctx.guild
        channel = self.ctx.channel
        emoji_name = arguments["emoji"].strip().replace(":","").lower()

        if not channel.permissions_for(guild.me).add_reactions:
            return "Error: I do not have the `Add Reactions` permission."

        try:
            message_id = int(arguments["message_id"])
        except (ValueError, TypeError):
            return "Error: Invalid message ID provided."

        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            return f"Error: Message with ID `{message_id}` not found in this channel."
        except discord.HTTPException as e:
            logger.exception(f"Failed to fetch message {message_id} in {guild.name}")
            return f"An unexpected error occurred while fetching the message: {e}"

        emoji_map = await get_guild_emoji_map(self.ctx)
        emoji = emoji_map.get(emoji_name, None)

        if emoji:
            try:
                await message.add_reaction(emoji)
                return f"Successfully reacted to message `{message_id}` with {emoji}."
            except discord.HTTPException as e:
                logger.exception(f"Failed to react to message {message_id} in {guild.name}")
                return f"An unexpected error occurred while reacting: {e}"
        else:
            return f"Error: Emoji with name `{emoji_name}` not found."

#
# Pin Message ToolCall
#

pinMessageToolCallSchema = ToolCallSchema(
    function=Function(
        name="pin_message",
        description="Pins a recent message in the current channel. Pinned messages are accessible via dedicated menu by users. The purpose of pinning is to track important messages, importance can be informational or due to what users might find entertaining.",
        parameters=Parameters(
            properties={
                "message_id": {
                    "type": "string",
                    "description": "The ID of the specific recent message to pin.",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for pinning the message.",
                },
            },
            required=["message_id", "reason"],
        ),
    )
)


class PinMessageToolCall(ToolCall):
    schema = pinMessageToolCallSchema
    function_name = pinMessageToolCallSchema.function.name

    async def _handle(self, arguments: dict):
        guild = self.ctx.guild
        channel = self.ctx.channel
        reason = arguments.get("reason", "Pinned by AI.")

        if not channel.permissions_for(guild.me).manage_messages:
            return "Error: I do not have the `Manage Messages` permission."

        try:
            message_id = int(arguments["message_id"])
        except (ValueError, TypeError):
            return "Error: Invalid message ID provided."

        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            return f"Error: Message with ID `{message_id}` not found in this channel."
        except discord.HTTPException as e:
            logger.exception(f"Failed to fetch message {message_id} in {guild.name}")
            return f"An unexpected error occurred while fetching the message: {e}"

        if message.pinned:
            return f"Error: Message `{message_id}` is already pinned."

        try:
            await message.pin(reason=reason)
            return f"Successfully pinned message `{message_id}`."
        except discord.HTTPException as e:
            logger.exception(f"Failed to pin message {message_id} in {guild.name}")
            return f"An unexpected error occurred: {e}"

#
# Send TTS Message ToolCall
#

sendTtsMessageToolCallSchema = ToolCallSchema(
    function=Function(
        name="send_tts_message",
        description="Makes the response message you send a text to speech (TTS) message. Use sparingly.",
        parameters=Parameters(
            properties={
                "reason": {
                    "type": "string",
                    "description": "The reason for applying text to speech (TTS) to the message.",
                },
            },
            required=["reason"],
        ),
    )
)


class SendTtsMessageToolCall(ToolCall):
    schema = sendTtsMessageToolCallSchema
    function_name = sendTtsMessageToolCallSchema.function.name

    async def _handle(self, arguments: dict):
        guild = self.ctx.guild
        channel = self.ctx.channel

        if not channel.permissions_for(guild.me).send_tts_messages:
            return "Error: I do not have the `Send TTS Messages` permission."

        return "TTS will be applied to your response message."
        