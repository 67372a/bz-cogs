"""Discord Views for inspecting function call data and model reasoning.

Provides buttons on the function call notification embed:
- 📋 View Inputs: Sends an ephemeral message with all function call names
  and their full input arguments.
- 🧠 View Reasoning: Sends an ephemeral message with the LLM reasoning
  for that specific tool-calling step.
- 📄 View Outputs: Sends an ephemeral message with all function call
  results (only shown after completion/failure).

Provides a button on the response embed:
- 🧠 View Reasoning: Sends an ephemeral message with per-step reasoning
  sections (Round 1, Round 2, ..., Final Response).

When content exceeds Discord's embed description limit (4096 characters),
it is sent as an attached .txt file instead.
"""

import io
import logging
from typing import Any, Dict, List, Optional

import discord

logger = logging.getLogger("red.bz_cogs.aiuser")

# Discord embed description character limit
EMBED_DESCRIPTION_MAX_CHARS = 4096


async def _send_content_or_file(
    interaction: discord.Interaction,
    title: str,
    content: str,
    color: int,
    file_name: str = "content.txt",
):
    """Send content as an embed if it fits, otherwise as a file attachment.

    If *content* fits within the Discord embed description limit it is sent
    as a normal embed.  When the content exceeds the limit a short embed
    noting the size is sent together with a ``discord.File`` attachment
    containing the full text.

    Args:
        interaction: The Discord interaction to respond to.
        title: Embed title.
        content: The full text content to display.
        color: Embed colour value.
        file_name: Filename for the attached file (default ``content.txt``).
    """
    if len(content) <= EMBED_DESCRIPTION_MAX_CHARS:
        embed = discord.Embed(title=title, description=content, color=color)
        await interaction.response.send_message(embed=embed, ephemeral=True)
    else:
        embed = discord.Embed(
            title=title,
            description=(
                f"Content is {len(content):,} characters — "
                "too large for a single embed. See attached file."
            ),
            color=color,
        )
        file_bytes = content.encode("utf-8")
        file = discord.File(io.BytesIO(file_bytes), filename=file_name)
        await interaction.response.send_message(embed=embed, file=file, ephemeral=True)


class FunctionCallView(discord.ui.View):
    """A View attached to the function call notification embed.

    Uses a class-level cache to store tool call data (inputs/outputs/reasoning)
    keyed by message ID, so button callbacks can retrieve the data
    when a user clicks.
    """

    # Class-level cache: message_id -> {"inputs": [...], "outputs": [...], "reasoning": str}
    # Each input is {"name": str, "args": str}
    # Each output is {"name": str, "result": str}
    _data_cache: Dict[int, Dict[str, Any]] = {}

    def __init__(
        self,
        message_id: int,
        has_outputs: bool = False,
        has_reasoning: bool = False,
    ):
        super().__init__(timeout=None)
        self.target_message_id = message_id

        # Remove buttons whose data isn't available yet
        if not has_outputs:
            self.remove_item(self.view_outputs)
        if not has_reasoning:
            self.remove_item(self.view_reasoning)

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

        description_parts = []
        for item in inputs:
            name = item["name"]
            args = item["args"]
            description_parts.append(f"**{name}**\n```json\n{args}\n```")

        description = "\n".join(description_parts)
        await _send_content_or_file(
            interaction,
            title="📋 Function Call Inputs",
            content=description,
            color=0x5865F2,
            file_name="inputs.txt",
        )

    @discord.ui.button(
        label="View Reasoning",
        style=discord.ButtonStyle.secondary,
        emoji="🧠",
    )
    async def view_reasoning(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        """Send ephemeral messages with the full LLM reasoning for this step."""
        data = self._data_cache.get(self.target_message_id)
        reasoning = data.get("reasoning", "") if data else ""

        if not reasoning:
            return await interaction.response.send_message(
                "No reasoning available for this step.", ephemeral=True
            )

        await _send_content_or_file(
            interaction,
            title="🧠 Model Reasoning",
            content=reasoning,
            color=0xFEE75C,
            file_name="reasoning.txt",
        )

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

        description_parts = []
        for item in outputs:
            name = item["name"]
            result = item["result"]
            description_parts.append(f"**{name}**\n```\n{result}\n```")

        description = "\n".join(description_parts)
        await _send_content_or_file(
            interaction,
            title="📄 Function Call Outputs",
            content=description,
            color=0x57F287,
            file_name="outputs.txt",
        )

    @classmethod
    def store_inputs(cls, message_id: int, inputs: List[Dict[str, str]]):
        """Store tool call inputs for a given embed message.

        Args:
            message_id: The Discord message ID of the notification embed.
            inputs: List of {"name": str, "args": str} dicts.
        """
        if message_id not in cls._data_cache:
            cls._data_cache[message_id] = {"inputs": [], "outputs": [], "reasoning": ""}
        cls._data_cache[message_id]["inputs"] = inputs

    @classmethod
    def store_outputs(cls, message_id: int, outputs: List[Dict[str, str]]):
        """Store tool call outputs for a given embed message.

        Args:
            message_id: The Discord message ID of the notification embed.
            outputs: List of {"name": str, "result": str} dicts.
        """
        if message_id not in cls._data_cache:
            cls._data_cache[message_id] = {"inputs": [], "outputs": [], "reasoning": ""}
        cls._data_cache[message_id]["outputs"] = outputs

    @classmethod
    def store_reasoning(cls, message_id: int, reasoning: str):
        """Store LLM reasoning for a given embed message.

        Args:
            message_id: The Discord message ID of the notification embed.
            reasoning: The reasoning text from the LLM for this step.
        """
        if message_id not in cls._data_cache:
            cls._data_cache[message_id] = {"inputs": [], "outputs": [], "reasoning": ""}
        cls._data_cache[message_id]["reasoning"] = reasoning

    @classmethod
    def cleanup(cls, message_id: int):
        """Remove cached data for a message."""
        cls._data_cache.pop(message_id, None)


class ResponseView(discord.ui.View):
    """A View attached to the bot's response embed for inspecting reasoning.

    Uses a class-level cache to store per-step reasoning (list of strings)
    keyed by the response message ID. Each string is the reasoning from
    one round of the tool-calling loop.
    """

    # Class-level cache: message_id -> List[str] (per-step reasoning)
    _reasoning_cache: Dict[int, List[str]] = {}

    def __init__(self, message_id: int, has_reasoning: bool = False):
        super().__init__(timeout=None)
        self.target_message_id = message_id

        if not has_reasoning:
            self.remove_item(self.view_reasoning)

    @discord.ui.button(
        label="View Reasoning",
        style=discord.ButtonStyle.secondary,
        emoji="🧠",
    )
    async def view_reasoning(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        """Send ephemeral messages with full per-step reasoning sections."""
        steps = self._reasoning_cache.get(self.target_message_id, [])

        if not steps:
            return await interaction.response.send_message(
                "No reasoning available.", ephemeral=True
            )

        # Build per-step sections (no truncation)
        description_parts = []
        for i, step_reasoning in enumerate(steps):
            if i < len(steps) - 1:
                label = f"Round {i + 1}"
            else:
                if len(steps) == 1:
                    label = None
                else:
                    label = "Final Response"

            if label:
                description_parts.append(f"**{label}**\n{step_reasoning}")
            else:
                description_parts.append(step_reasoning)

        full_description = "\n\n".join(description_parts)
        await _send_content_or_file(
            interaction,
            title="🧠 Model Reasoning",
            content=full_description,
            color=0xFEE75C,
            file_name="reasoning.txt",
        )

    @classmethod
    def store_reasoning_steps(cls, message_id: int, steps: List[str]):
        """Store per-step reasoning for a response message.

        Args:
            message_id: The Discord message ID of the response embed.
            steps: List of reasoning strings, one per loop round.
        """
        cls._reasoning_cache[message_id] = steps

    @classmethod
    def cleanup(cls, message_id: int):
        """Remove cached data for a message."""
        cls._reasoning_cache.pop(message_id, None)
