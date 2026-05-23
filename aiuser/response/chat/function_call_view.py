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
"""

import json
import logging
from typing import Any, Dict, List, Optional

import discord

logger = logging.getLogger("red.bz_cogs.aiuser")

# Discord embed description character limit
EMBED_DESCRIPTION_MAX_CHARS = 4096


def _split_text_to_embed_chunks(text: str, max_chars: int = EMBED_DESCRIPTION_MAX_CHARS) -> List[str]:
    """Split text into chunks that fit within Discord embed description limits.

    Attempts to split on paragraph boundaries (\\n\\n) first, then line breaks (\\n),
    then falls back to hard character splitting. Each chunk is at most *max_chars*
    characters so it can be placed in an embed description without truncation.

    Args:
        text: The full text to split.
        max_chars: Maximum characters per chunk (default: 4096).

    Returns:
        A list of text chunks, each <= max_chars characters.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Try to find a good split point within the character limit
        segment = remaining[:max_chars]

        # Prefer splitting on paragraph boundaries
        split_pos = segment.rfind("\n\n")
        if split_pos > max_chars // 4:
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip("\n")
            continue

        # Fall back to line breaks
        split_pos = segment.rfind("\n")
        if split_pos > max_chars // 4:
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos + 1:]
            continue

        # Last resort: hard split at max_chars
        chunks.append(segment)
        remaining = remaining[max_chars:]

    return chunks


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

        embed = discord.Embed(
            title="📋 Function Call Inputs",
            color=0x5865F2,
        )

        description_parts = []
        for item in inputs:
            name = item["name"]
            args = item["args"]
            description_parts.append(f"**{name}**\n```json\n{args}\n```")

        description = "\n".join(description_parts)
        if len(description) > EMBED_DESCRIPTION_MAX_CHARS:
            description = description[: EMBED_DESCRIPTION_MAX_CHARS - 20] + "\n\n...(truncated)"

        embed.description = description
        await interaction.response.send_message(embed=embed, ephemeral=True)

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

        chunks = _split_text_to_embed_chunks(reasoning)
        # First chunk uses interaction.response (required for ephemeral)
        embed = discord.Embed(
            title="🧠 Model Reasoning",
            description=chunks[0],
            color=0xFEE75C,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

        # Remaining chunks sent via followup (ephemeral by default after initial ephemeral response)
        for chunk in chunks[1:]:
            embed = discord.Embed(
                title="🧠 Model Reasoning (continued)",
                description=chunk,
                color=0xFEE75C,
            )
            await interaction.followup.send(embed=embed, ephemeral=True)

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
        chunks = _split_text_to_embed_chunks(full_description)

        # First chunk uses interaction.response (required for ephemeral)
        embed = discord.Embed(
            title="🧠 Model Reasoning",
            description=chunks[0],
            color=0xFEE75C,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

        # Remaining chunks sent via followup
        for chunk in chunks[1:]:
            embed = discord.Embed(
                title="🧠 Model Reasoning (continued)",
                description=chunk,
                color=0xFEE75C,
            )
            await interaction.followup.send(embed=embed, ephemeral=True)

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
