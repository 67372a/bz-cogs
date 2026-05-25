import logging
import os
from typing import Dict, List, Optional, Tuple

from redbot.core import Config, commands

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema
from aiuser.utils.utilities import to_thread

logger = logging.getLogger("red.bz_cogs.aiuser")

DEFAULT_MERMAID_THEME = "dark"

mermaid_diagram_tool_call_schema = ToolCallSchema(
    function=Function(
        name="create_mermaid_diagram",
        description=(
            "Renders a Mermaid diagram to a PNG image and attaches it to your response. "
            "Use this tool when the user asks for a diagram, flowchart, sequence diagram, "
            "class diagram, state diagram, ER diagram, Gantt chart, pie chart, mindmap, "
            "or any other visual diagram. Pass valid Mermaid syntax in the 'code' parameter."
        ),
        parameters=Parameters(
            properties={
                "filename": {
                    "type": "string",
                    "description": (
                        "The desired filename including the .png extension "
                        "(e.g. 'flowchart.png', 'architecture.png')."
                    ),
                },
                "code": {
                    "type": "string",
                    "description": (
                        "The complete Mermaid diagram syntax to render. "
                        "Supports flowchart, sequence, class, state, ER, gantt, "
                        "pie, mindmap, and gitgraph diagram types."
                    ),
                },
            },
            required=["filename", "code"],
        ),
    )
)


class MermaidDiagramToolCall(ToolCall):
    """Render Mermaid diagrams to PNG/SVG images as Discord attachments.

    Uses the 'merm' library for pure-Python Mermaid parsing and SVG
    generation. PNG output is preferred and requires 'cairosvg' with the
    system 'cairo' library. When cairo is unavailable, falls back to SVG
    output automatically.

    Rendered images are stored internally and collected by the pipeline
    to be sent as Discord file attachments.

    The diagram theme is configurable per-guild via the
    'mermaid_diagram_theme' config setting (default: 'dark').
    """

    schema = mermaid_diagram_tool_call_schema
    function_name = mermaid_diagram_tool_call_schema.function.name

    def __init__(self, config: Config, ctx: commands.Context):
        super().__init__(config, ctx)
        self.generated_images: List[Dict] = []

    def get_generated_images(self) -> List[Dict]:
        """Return stored rendered images and clear the internal list.

        Each entry has:
            - bytes: PNG or SVG image bytes.
            - filename: The filename for the Discord attachment.

        Returns:
            List of image dicts ready to be sent as Discord attachments.
        """
        images = self.generated_images[:]
        self.generated_images.clear()
        return images

    async def _get_theme(self) -> str:
        """Return the configured Mermaid theme for this guild."""
        try:
            theme = await self.config.guild(self.ctx.guild).mermaid_diagram_theme()
            if theme and theme.strip():
                return theme.strip()
        except Exception:
            pass
        return DEFAULT_MERMAID_THEME

    async def _handle(self, arguments: dict) -> str:
        filename = arguments.get("filename", "").strip()
        code = arguments.get("code", "").strip()

        if not filename:
            return "Error: No filename provided. Supply a filename including the .png extension."
        if not code:
            return "Error: No Mermaid code provided. Supply the diagram syntax in the 'code' parameter."

        # Sanitize filename: strip directory components, keep only basename
        filename = os.path.basename(filename)
        if not filename or filename.startswith("."):
            return "Error: Invalid filename after sanitization."

        theme = await self._get_theme()

        # Render diagram — try PNG first, fall back to SVG if cairo is missing
        try:
            image_bytes, extension = await _render_mermaid(code, theme=theme)
        except MermaidRenderError as e:
            return f"Error: Failed to render Mermaid diagram — {e}"

        if not image_bytes or len(image_bytes) < 100:
            return "Error: Rendered image is too small or empty."

        # Ensure filename has the correct extension
        if not filename.lower().endswith(f".{extension}"):
            # Replace wrong extension or add the correct one
            base = os.path.splitext(filename)[0]
            filename = f"{base}.{extension}"

        self.generated_images.append({
            "bytes": image_bytes,
            "filename": filename,
        })

        return (
            f"Successfully created Mermaid diagram: `{filename}`. "
            "The image is attached to this response."
        )


class MermaidRenderError(Exception):
    """Raised when Mermaid rendering fails."""
    pass


@to_thread(timeout=30)
def _render_mermaid_sync(code: str, theme: str = "dark") -> Tuple[bytes, str]:
    """Synchronous Mermaid rendering (runs in thread executor).

    Tries to render as PNG (requires cairosvg + cairo). If cairo is not
    available, falls back to SVG output.

    Args:
        code: The Mermaid diagram syntax to render.
        theme: The Mermaid theme to use (default, dark, forest, neutral).

    Returns:
        Tuple of (image_bytes, extension) where extension is 'png' or 'svg'.

    Raises:
        MermaidRenderError: If rendering fails.
    """
    from merm import render_diagram

    # Generate SVG first (always works)
    try:
        svg = render_diagram(code, theme=theme)
    except Exception as e:
        raise MermaidRenderError(str(e))

    # Try to convert to PNG
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="#f0f0f0")
        return png_bytes, "png"
    except ImportError:
        logger.info(
            "cairosvg/cairo not available, falling back to SVG output for Mermaid diagrams"
        )
    except Exception as e:
        logger.warning(
            "cairosvg PNG conversion failed (%s), falling back to SVG output", e
        )

    # Fallback: return SVG bytes
    return svg.encode("utf-8"), "svg"


async def _render_mermaid(code: str, theme: str = "dark") -> Tuple[bytes, str]:
    """Render Mermaid code to image bytes using the merm library.

    Args:
        code: The Mermaid diagram syntax to render.
        theme: The Mermaid theme to use.

    Returns:
        Tuple of (image_bytes, extension).

    Raises:
        MermaidRenderError: If rendering fails.
    """
    try:
        return await _render_mermaid_sync(code, theme=theme)
    except MermaidRenderError:
        raise
    except Exception as e:
        raise MermaidRenderError(str(e))
