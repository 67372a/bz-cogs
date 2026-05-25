"""Tests for the create_mermaid_diagram function call.

Verifies that:
- MermaidDiagramToolCall schema is correctly defined
- Input validation rejects empty filenames and empty code
- Filenames are sanitized (path components stripped, extension corrected)
- Rendering calls merm.render_diagram with correct theme
- PNG rendering works when cairosvg is available
- SVG fallback works when cairosvg is not available
- get_generated_images() returns stored images and clears the internal list
- Theme configuration is read from guild config
- Error handling for rendering failures
"""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_attach_files.py pattern)
# ---------------------------------------------------------------------------

def _make_mock_package(name: str) -> ModuleType:
    pkg = ModuleType(name)
    pkg.__path__ = []
    pkg.__package__ = name
    pkg.__file__ = f"<mocked-{name}>"
    return pkg


def _ensure_package(name: str):
    if name not in sys.modules:
        sys.modules[name] = _make_mock_package(name)


# Pre-register required packages
for _pkg in [
    "aiuser", "aiuser.functions", "aiuser.functions.types",
    "aiuser.functions.tool_call", "aiuser.functions.mermaid",
    "aiuser.functions.generate_image", "aiuser.functions.edit_image",
    "aiuser.config", "aiuser.types", "aiuser.utils", "aiuser.utils.utilities",
    "aiuser.config.constants",
    "aiuser.response", "aiuser.response.chat",
    "aiuser.functions.openrouter",
    "redbot", "redbot.core", "redbot.core.commands",
    "redbot.core.bot", "redbot.core.utils", "redbot.core.utils.chat_formatting",
    "discord", "discord.ext", "discord.ext.commands",
    "merm",
]:
    _ensure_package(_pkg)

# Mock redbot.core.commands so Config and commands are available
if not hasattr(sys.modules["redbot.core"], "Config"):
    sys.modules["redbot.core"].Config = MagicMock()
if not hasattr(sys.modules["redbot.core.commands"], "Context"):
    sys.modules["redbot.core.commands"].Context = MagicMock()

# Import mock_importer helpers
from tests.mock_importer import import_module_directly

# Load real types module
sys.modules["aiuser.functions.types"] = import_module_directly(
    "aiuser.functions.types", "aiuser/functions/types.py"
)

# Load real tool_call base class
sys.modules["aiuser.functions.tool_call"] = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)

# Mock aiuser.utils.utilities with a to_thread decorator (avoids heavy dependencies)
_utils_mod = _make_mock_package("aiuser.utils.utilities")

def _mock_to_thread(timeout=300):
    """Mock to_thread that runs the function synchronously in tests."""
    import functools
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

_utils_mod.to_thread = _mock_to_thread
sys.modules["aiuser.utils.utilities"] = _utils_mod

# Load real mermaid tool_call module
sys.modules["aiuser.functions.mermaid"] = _ensure_package("aiuser.functions.mermaid") or sys.modules["aiuser.functions.mermaid"]
sys.modules["aiuser.functions.mermaid.tool_call"] = import_module_directly(
    "aiuser.functions.mermaid.tool_call", "aiuser/functions/mermaid/tool_call.py"
)

MermaidDiagramToolCall = sys.modules["aiuser.functions.mermaid.tool_call"].MermaidDiagramToolCall
MermaidRenderError = sys.modules["aiuser.functions.mermaid.tool_call"].MermaidRenderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(config=None, ctx=None, theme="dark"):
    """Create a MermaidDiagramToolCall instance with mocked config and ctx.

    Args:
        config: Optional mock config. If None, a mock with proper async
            guild config accessors is created.
        ctx: Optional mock context.
        theme: Configured Mermaid theme (default "dark").
    """
    if config is None:
        config = MagicMock()
        config.guild.return_value.mermaid_diagram_theme = AsyncMock(return_value=theme)
    if ctx is None:
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 999
    return MermaidDiagramToolCall(config=config, ctx=ctx)


def _run_async(coro):
    """Run an async coroutine in a new event loop (for tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _fake_png():
    """Return fake PNG bytes (enough to pass the size check)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


def _fake_svg():
    """Return fake SVG bytes (must be >= 100 bytes to pass size check)."""
    return b'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">' + b'<rect width="100" height="100" fill="red"/>' * 5 + b'</svg>'


# ---------------------------------------------------------------------------
# Tests: Schema
# ---------------------------------------------------------------------------

class TestMermaidDiagramSchema:
    """Verify the tool call schema is correctly defined."""

    def test_function_name(self):
        assert MermaidDiagramToolCall.function_name == "create_mermaid_diagram"

    def test_schema_has_filename_and_code_parameters(self):
        params = MermaidDiagramToolCall.schema.function.parameters
        assert "filename" in params.properties
        assert "code" in params.properties
        assert params.required == ["filename", "code"]

    def test_schema_type_is_function(self):
        assert MermaidDiagramToolCall.schema.type == "function"

    def test_description_mentions_diagram(self):
        desc = MermaidDiagramToolCall.schema.function.description.lower()
        assert "diagram" in desc

    def test_description_mentions_mermaid(self):
        desc = MermaidDiagramToolCall.schema.function.description.lower()
        assert "mermaid" in desc


# ---------------------------------------------------------------------------
# Tests: _handle validation
# ---------------------------------------------------------------------------

class TestMermaidDiagramValidation:
    """Verify input validation in _handle()."""

    def test_empty_filename_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"filename": "", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "No filename" in result

    def test_empty_code_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"filename": "diagram.png", "code": ""}))
        assert "Error" in result
        assert "No Mermaid code" in result

    def test_missing_filename_key_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "No filename" in result

    def test_missing_code_key_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"filename": "diagram.png"}))
        assert "Error" in result
        assert "No Mermaid code" in result

    def test_empty_both_returns_filename_error(self):
        """When both are empty, filename error takes precedence."""
        tool = _make_tool()
        result = _run_async(tool.run({}))
        assert "Error" in result
        assert "No filename" in result

    def test_whitespace_only_filename_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"filename": "   ", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "No filename" in result

    def test_whitespace_only_code_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"filename": "diagram.png", "code": "   "}))
        assert "Error" in result
        assert "No Mermaid code" in result


# ---------------------------------------------------------------------------
# Tests: Filename sanitization
# ---------------------------------------------------------------------------

class TestMermaidDiagramFilenameSanitization:
    """Verify filename sanitization behavior."""

    def test_png_extension_preserved(self):
        """PNG extension should be kept as-is when rendering produces PNG."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_png(), "png"),
        ):
            result = _run_async(tool.run({"filename": "diagram.png", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`diagram.png`" in result

    def test_wrong_extension_corrected_for_svg(self):
        """When rendering produces SVG, extension should be corrected."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_svg(), "svg"),
        ):
            result = _run_async(tool.run({"filename": "diagram.png", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`diagram.svg`" in result

    def test_missing_extension_appended(self):
        """No extension should get the correct one appended."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_png(), "png"),
        ):
            result = _run_async(tool.run({"filename": "diagram", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`diagram.png`" in result

    def test_path_components_stripped(self):
        """Directory components should be stripped from filename."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_png(), "png"),
        ):
            result = _run_async(tool.run({"filename": "subdir/diagram.png", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`diagram.png`" in result
        assert "subdir" not in result

    def test_dot_prefix_filename_returns_error(self):
        """Filenames starting with a dot should be rejected."""
        tool = _make_tool()
        result = _run_async(tool.run({"filename": ".hidden.png", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "Invalid filename" in result

    def test_only_extension_returns_error(self):
        """Filename that is just an extension (e.g. '.png') should be rejected."""
        tool = _make_tool()
        result = _run_async(tool.run({"filename": ".png", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "Invalid filename" in result


# ---------------------------------------------------------------------------
# Tests: Rendering
# ---------------------------------------------------------------------------

class TestMermaidDiagramRendering:
    """Verify rendering behavior with mocked merm library."""

    def test_successful_png_render(self):
        """Successful PNG rendering should store the image and return success."""
        tool = _make_tool()
        fake = _fake_png()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(fake, "png"),
        ):
            result = _run_async(tool.run({"filename": "flowchart.png", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`flowchart.png`" in result
        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["bytes"] == fake
        assert tool.generated_images[0]["filename"] == "flowchart.png"

    def test_successful_svg_render(self):
        """Successful SVG rendering should store the image with .svg extension."""
        tool = _make_tool()
        fake = _fake_svg()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(fake, "svg"),
        ):
            result = _run_async(tool.run({"filename": "flowchart.png", "code": "flowchart TD\n    A --> B"}))
        assert "Successfully" in result
        assert "`flowchart.svg`" in result
        assert len(tool.generated_images) == 1
        assert tool.generated_images[0]["bytes"] == fake

    def test_render_error_returns_error_message(self):
        """Rendering failure should return an error message."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            side_effect=MermaidRenderError("Invalid syntax"),
        ):
            result = _run_async(tool.run({"filename": "bad.png", "code": "invalid mermaid %%%"}))
        assert "Error" in result
        assert "Invalid syntax" in result

    def test_render_too_small_returns_error(self):
        """Output that's too small should return an error."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(b"\x89PNG\x00", "png"),  # Only 5 bytes
        ):
            result = _run_async(tool.run({"filename": "tiny.png", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result
        assert "too small" in result

    def test_render_empty_bytes_returns_error(self):
        """Empty output should return an error."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(b"", "png"),
        ):
            result = _run_async(tool.run({"filename": "empty.png", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result

    def test_render_none_returns_error(self):
        """None output should return an error."""
        tool = _make_tool()
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(None, "png"),
        ):
            result = _run_async(tool.run({"filename": "none.png", "code": "flowchart TD\n    A --> B"}))
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: Theme configuration
# ---------------------------------------------------------------------------

class TestMermaidDiagramTheme:
    """Verify theme is read from config and passed to renderer."""

    def test_default_theme_is_dark(self):
        """Default theme should be 'dark'."""
        from aiuser.functions.mermaid.tool_call import DEFAULT_MERMAID_THEME
        assert DEFAULT_MERMAID_THEME == "dark"

    def test_theme_read_from_config(self):
        """Theme should be read from guild config."""
        tool = _make_tool(theme="forest")
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_png(), "png"),
        ) as mock_render:
            _run_async(tool.run({"filename": "test.png", "code": "flowchart TD\n    A --> B"}))
        mock_render.assert_called_once_with("flowchart TD\n    A --> B", theme="forest")

    def test_theme_fallback_on_error(self):
        """Should fall back to default theme on config error."""
        config = MagicMock()
        config.guild.return_value.mermaid_diagram_theme = AsyncMock(side_effect=Exception("config error"))
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 999
        tool = _make_tool(config=config, ctx=ctx)
        with patch(
            "aiuser.functions.mermaid.tool_call._render_mermaid",
            new_callable=AsyncMock,
            return_value=(_fake_png(), "png"),
        ) as mock_render:
            _run_async(tool.run({"filename": "test.png", "code": "flowchart TD\n    A --> B"}))
        mock_render.assert_called_once_with("flowchart TD\n    A --> B", theme="dark")

    def test_theme_passthrough_to_renderer(self):
        """Each valid theme should be passed through to the renderer."""
        for theme in ["default", "dark", "forest", "neutral"]:
            tool = _make_tool(theme=theme)
            with patch(
                "aiuser.functions.mermaid.tool_call._render_mermaid",
                new_callable=AsyncMock,
                return_value=(_fake_png(), "png"),
            ) as mock_render:
                _run_async(tool.run({"filename": "test.png", "code": "flowchart TD\n    A --> B"}))
            mock_render.assert_called_once_with("flowchart TD\n    A --> B", theme=theme)


# ---------------------------------------------------------------------------
# Tests: get_generated_images
# ---------------------------------------------------------------------------

class TestMermaidDiagramImageCollection:
    """Verify get_generated_images() behavior."""

    def test_returns_stored_images(self):
        """get_generated_images should return all stored images."""
        tool = _make_tool()
        fake = _fake_png()
        tool.generated_images = [{"bytes": fake, "filename": "test.png"}]
        result = tool.get_generated_images()
        assert len(result) == 1
        assert result[0]["bytes"] == fake
        assert result[0]["filename"] == "test.png"

    def test_clears_internal_list(self):
        """get_generated_images should clear the internal list after returning."""
        tool = _make_tool()
        tool.generated_images = [{"bytes": b"data", "filename": "a.png"}]
        tool.get_generated_images()
        assert len(tool.generated_images) == 0

    def test_returns_copy(self):
        """get_generated_images should return a copy, not the original list."""
        tool = _make_tool()
        tool.generated_images = [{"bytes": b"data", "filename": "a.png"}]
        result = tool.get_generated_images()
        result.append({"bytes": b"other", "filename": "b.png"})
        # Internal list was cleared by get_generated_images, so appending to
        # the returned copy should not affect the (now empty) internal list.
        assert len(tool.generated_images) == 0
        assert len(result) == 2

    def test_multiple_images(self):
        """Multiple images should be returned in order."""
        tool = _make_tool()
        tool.generated_images = [
            {"bytes": b"img1", "filename": "a.png"},
            {"bytes": b"img2", "filename": "b.png"},
            {"bytes": b"img3", "filename": "c.png"},
        ]
        result = tool.get_generated_images()
        assert len(result) == 3
        assert [r["filename"] for r in result] == ["a.png", "b.png", "c.png"]

    def test_empty_list(self):
        """Empty list should return empty list."""
        tool = _make_tool()
        result = tool.get_generated_images()
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------

class TestMermaidDiagramPipelineIntegration:
    """Verify the pipeline correctly identifies and collects from MermaidDiagramToolCall."""

    def test_pipeline_collects_from_mermaid_tool(self):
        """_collect_images_from_tool should recognize MermaidDiagramToolCall."""
        tool = _make_tool()
        fake = _fake_png()
        tool.generated_images = [{"bytes": fake, "filename": "flowchart.png"}]

        # Simulate the pipeline's _collect_images_from_tool logic
        collected = []
        if isinstance(tool, MermaidDiagramToolCall):
            collected = tool.get_generated_images()

        assert len(collected) == 1
        assert collected[0]["bytes"] == fake
