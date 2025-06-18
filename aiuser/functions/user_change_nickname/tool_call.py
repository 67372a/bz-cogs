import logging

import discord
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from asyncio import to_thread

logger = logging.getLogger("red.bz_cogs.aiuser")

changeUserNickNameToolCallSchema = ToolCallSchema(
        function=Function(
            name="change_user_nickname",
            description="Changes the nickname of a user on the server.",
            parameters=Parameters(
                properties={
                    "username": {
                        "type": "string",
                        "description": "The actual account username of the user whose nickname is to be changed.",
                    },
                    "nickname": {
                        "type": "string",
                        "description": "The new nickname for the user. Must be 32 characters or less. Set to an empty string to remove the nickname.",
                    },
                },
                required=["username", "nickname"],
            ),
        )
    )


class ChangeUserNicknameToolCall(ToolCall):
    schema = changeUserNickNameToolCallSchema
    function_name = schema.function.name

    async def _handle(self, arguments: dict):
        guild = self.ctx.guild
        username = arguments["username"]
        nickname = arguments["nickname"]

        if not guild.me.guild_permissions.manage_nicknames:
            return "Error: I do not have the `Manage Nicknames` permission."
        
        # Use a non-blocking method to search members
        user = await to_thread(_search_members, guild, username.strip().lower())

        if user is None:
            return "Error: Could not resolve user ID from username provided."
        else:
            user_id = user.id

        member = guild.get_member(user_id)
        if not member:
            return f"Error: User with ID `{user_id}` not found in this server."

        if member.top_role >= guild.me.top_role or guild.owner == member:
            return f"Error: I cannot change the nickname of {member.mention} due to role hierarchy."

        if len(nickname) > 32:
            return "Error: The requested nickname is longer than the 32-character limit."

        try:
            old_name = member.display_name
            await member.edit(nick=nickname or None)
            new_name = nickname or member.name
            logger.info(
                f"Changed nickname for {member.name} (ID: {user_id}) from '{old_name}' to '{new_name}' in {guild.name}"
            )
            return f"Successfully changed nickname for {old_name} to `{new_name}`."
        except discord.HTTPException as e:
            logger.exception(f"Failed to change nickname for user {user_id} in {guild.name}")
            return f"An unexpected error occurred: {e}"
        
def _search_members(guild: discord.Guild, query: str):
    """ Blocking search function to be run in a separate thread """
    for member in guild.members:
        if (
            query == member.name.strip().lower()
        ):
            logger.info(f"member found={member}")
            return member
    return None