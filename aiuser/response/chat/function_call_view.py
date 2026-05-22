"""Discord View with buttons for inspecting function call inputs and outputs.

Provides two buttons on the function call notification embed:
- 📋 View Inputs: Sends an ephemeral message with all function call names
  and their full input arguments.
- 📄 View Outputs: Sends an ephemeral message with all function call
  results (only shown after completion/failure).
"""

import json
import logging
from typing import Any, Dict, List, Optional

import discord

logger = logging.getLogger("red.bz_cogs.aiuser")

# Discord embed description character limit
EMBED_DESCRIPTION_MAX_CHARS = 4096


class FunctionCallView(discord.ui.View):
    """A View attached to the function call notification embed.

    Uses a class-level cache to store tool call data (inputs/outputs)
    keyed by message ID, so button callbacks can retrieve the data
    when a user clicks.
    """

    # Class-level cache: message_id -> {"inputs": [...], "outputs": [...]}
    # Each input is {"name": str, "args": str}
    # Each output is {"name": str, "result": str}
    _data_cache: Dict[int, Dict[str, List[Dict[str, str]]]] = {}

    def __init__(self, message_id: int, has_outputs: bool = False):
        super().__init__(timeout=None)
        self.target_message_id = message_id

        # Remove the outputs button if outputs aren't available yet
        if not has_outputs:
            self.remove_item(self.view_outputs)

    @discord.ui.button(
        label="View Inputs",
        style=discord.ButtonStyle.secondary,
        emoji="📋",
    )
    async def view_inputs(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        """Send an ephemeral message with all function call inputs."""
        data = self._data_cache.get(self.target_message_id)
        inputs = data.get("inputs", []) if data else []

        if not inputs:
            return await interaction.response.send_message(
                "No function call data available.", ephemeral=True
            )

        embed = discord.Embed(
            title="📋 Function Call Inputs",
            color=0x5865F2,
        )

        description_parts = []
        for item in inputs:
            name = item["name"]
            args = item["args"]
            # Truncate individual args to avoid hitting the 4096 limit
            if len(args) > 500:
                args = args[:497] + "..."
            description_parts.append(f"**{name}**\n```json\n{args}\n```")

        description = "\n".join(description_parts)
        if len(description) > EMBED_DESCRIPTION_MAX_CHARS:
            description = description[: EMBED_DESCRIPTION_MAX_CHARS - 20] + "\n\n...(truncated)"

        embed.description = description
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(
        label="View Outputs",
        style=discord.ButtonStyle.secondary,
        emoji="📄",
    )
    async def view_outputs(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        """Send an ephemeral message with all function call outputs."""
        data = self._data_cache.get(self.target_message_id)
        outputs = data.get("outputs", []) if data else []

        if not outputs:
            return await interaction.response.send_message(
                "No function call outputs available yet.", ephemeral=True
            )

        embed = discord.Embed(
            title="📄 Function Call Outputs",
            color=0x57F287,
        )

        description_parts = []
        for item in outputs:
            name = item["name"]
            result = item["result"]
            # Truncate individual results
            if len(result) > 500:
                result = result[:497] + "..."
            description_parts.append(f"**{name}**\n```\n{result}\n```")

        description = "\n".join(description_parts)
        if len(description) > EMBED_DESCRIPTION_MAX_CHARS:
            description = description[: EMBED_DESCRIPTION_MAX_CHARS - 20] + "\n\n...(truncated)"

        embed.description = description
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @classmethod
    def store_inputs(cls, message_id: int, inputs: List[Dict[str, str]]):
        """Store tool call inputs for a given embed message.

        Args:
            message_id: The Discord message ID of the notification embed.
            inputs: List of {"name": str, "args": str} dicts.
        """
        if message_id not in cls._data_cache:
            cls._data_cache[message_id] = {"inputs": [], "outputs": []}
        cls._data_cache[message_id]["inputs"] = inputs

    @classmethod
    def store_outputs(cls, message_id: int, outputs: List[Dict[str, str]]):
        """Store tool call outputs for a given embed message.

        Args:
            message_id: The Discord message ID of the notification embed.
            outputs: List of {"name": str, "result": str} dicts.
        """
        if message_id not in cls._data_cache:
            cls._data_cache[message_id] = {"inputs": [], "outputs": []}
        cls._data_cache[message_id]["outputs"] = outputs

    @classmethod
    def cleanup(cls, message_id: int):
        """Remove cached data for a message."""
        cls._data_cache.pop(message_id, None)
