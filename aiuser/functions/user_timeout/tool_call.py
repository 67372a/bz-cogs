import datetime
import logging

import discord
from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from asyncio import to_thread

logger = logging.getLogger("red.bz_cogs.aiuser")

timeoutUserToolCallSchema = ToolCallSchema(
    function=Function(
        name="timeout_user",
        description="Temporarily times out a user for three seconds, preventing them from sending messages or joining voice channels.",
        parameters=Parameters(
            properties={
                "username": {
                    "type": "string",
                    "description": "The actual account username of the user whose nickname is to be changed.",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason the user is being timed out",
                },
            },
            required=["username", "reason"],
        ),
    )
)

class TimeoutUserToolCall(ToolCall):
    schema = timeoutUserToolCallSchema
    function_name = timeoutUserToolCallSchema.function.name

    async def _handle(self, arguments: dict):
        guild = self.ctx.guild
        username = arguments["username"]
        reason = arguments.get("reason", "Timed out by AI.")

        if not guild.me.guild_permissions.moderate_members:
            return "Error: I do not have the `Moderate Members` permission."
        
        # Use a non-blocking method to search members
        user_id = await to_thread(_search_members, guild, username.strip().lower())

        if user_id is None:
            return "Error: Could not resolve user ID from username provided."
        
        member = guild.get_member(user_id)
        if not member:
            return f"Error: User with ID `{user_id}` not found in this server."

        if member.is_timed_out():
            return f"Error: {member.mention} is already timed out."

        if member.top_role >= guild.me.top_role or guild.owner == member:
            return f"Error: I cannot time out {member.mention} as they have a higher role in the hierarchy than I do."

        duration = datetime.timedelta(seconds=3)
        try:
            await member.timeout(duration, reason=reason)
            logger.info(
                f"Timed out {member.name} (ID: {user_id}) for 3 seconds in {guild.name}. Reason: {reason}"
            )
            return f"Successfully timed out {member.mention} for 3 seconds."
        except discord.HTTPException as e:
            logger.exception(f"Failed to time out user {user_id} in {guild.name}")
            return f"An unexpected error occurred: {e}"
        
def _search_members(guild: discord.Guild, query: str):
    """ Blocking search function to be run in a separate thread """
    for member in guild.members:
        if (
            query == member.name.strip().lower()
        ):
            return member
    return None