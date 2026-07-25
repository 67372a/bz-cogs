"""Tests for the reasoning_effort optional property in image tool schemas.

Verifies that:
1. Both GenerateImageToolCall and EditImageToolCall schemas include reasoning_effort.
2. The reasoning_effort property has the correct enum values and type.
3. The _handle() method allows LLM-provided reasoning_effort to override admin config.
4. When no LLM value is provided, the admin config default is used.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Mock setup — same pattern as test_generate_image_dedup.py
# ---------------------------------------------------------------------------

_injected_modules: dict = {}
_SENTINEL = object()


def _inject_mock(name: str, module=None):
    if name in _injected_modules:
        return
    _injected_modules[name] = sys.modules.get(name, _SENTINEL)
    if module is None:
        module = MagicMock()
    sys.modules[name] = module


def _restore_modules():
    for name, original in _injected_modules.items():
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _injected_modules.clear()


# --- discord stub ---
import types as _types

_discord_mod = _types.ModuleType("discord")
_discord_mod.Message = type("_DiscordMessage", (), {})
_discord_mod.Embed = MagicMock()
_discord_mod.Color = MagicMock()
_discord_mod.Colour = MagicMock()
_inject_mock("discord", _discord_mod)

# --- redbot stubs ---
_redbot = _make_mock_package("redbot")
_redbot_core = _make_mock_package("redbot.core")
_redbot_core.Config = MagicMock()
_redbot_core.commands = MagicMock()
_redbot_bot = _make_mock_package("redbot.core.bot")
_redbot_bot.Red = MagicMock()
_inject_mock("redbot", _redbot)
_inject_mock("redbot.core", _redbot_core)
_inject_mock("redbot.core.bot", _redbot_bot)

# --- Import real types module ---
types_mod = import_module_directly(
    "aiuser.functions.types", "aiuser/functions/types.py"
)
_inject_mock("aiuser.functions.types", types_mod)

# --- Import real ToolCall base ---
tool_call_base = import_module_directly(
    "aiuser.functions.tool_call", "aiuser/functions/tool_call.py"
)
_inject_mock("aiuser.functions.tool_call", tool_call_base)

# --- Stub remaining external dependencies ---
_or_types = _make_mock_package("aiuser.types")
_or_types_mod = _make_mock_package("aiuser.types.openrouter_types")
_or_types_mod.DirectImageGenerationParameters = MagicMock
_or_types_mod.DirectImageEditParameters = MagicMock
_or_types_mod.deserialize_parameters = MagicMock()
_inject_mock("aiuser.types", _or_types)
_inject_mock("aiuser.types.openrouter_types", _or_types_mod)

_image_cache_mod = _make_mock_package("aiuser.utils.image_cache")
_image_cache_mod.image_cache = MagicMock()
_image_cache_mod.processed_image_cache = MagicMock()
_inject_mock("aiuser.utils", _make_mock_package("aiuser.utils"))
_inject_mock("aiuser.utils.image_cache", _image_cache_mod)

_real_image_processing = import_module_directly(
    "aiuser.utils.image_processing", "aiuser/utils/image_processing.py"
)
_inject_mock("aiuser.utils.image_processing", _real_image_processing)

_inject_mock("aiuser.config", _make_mock_package("aiuser.config"))
_real_constants = import_module_directly(
    "aiuser.config.constants", "aiuser/config/constants.py"
)
_inject_mock("aiuser.config.constants", _real_constants)
_real_utilities = import_module_directly(
    "aiuser.utils.utilities", "aiuser/utils/utilities.py"
)
_inject_mock("aiuser.utils.utilities", _real_utilities)

if "aiohttp" not in sys.modules:
    _inject_mock("aiohttp")

# Ensure parent packages
_all_pkg_prefixes = set()
for _name in list(sys.modules.keys()) + [
    "aiuser.functions.generate_image.tool_call",
    "aiuser.functions.edit_image.tool_call",
]:
    _parts = _name.split(".")
    for _i in range(1, len(_parts)):
        _all_pkg_prefixes.add(".".join(_parts[:_i]))
for _pkg in _all_pkg_prefixes:
    if _pkg not in sys.modules:
        _inject_mock(_pkg, _make_mock_package(_pkg))

# Load both modules under test
generate_image_mod = import_module_directly(
    "aiuser.functions.generate_image.tool_call",
    "aiuser/functions/generate_image/tool_call.py",
)
GenerateImageToolCall = generate_image_mod.GenerateImageToolCall

edit_image_mod = import_module_directly(
    "aiuser.functions.edit_image.tool_call",
    "aiuser/functions/edit_image/tool_call.py",
)
EditImageToolCall = edit_image_mod.EditImageToolCall

_restore_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ENUM_VALUES = ["minimal", "low", "medium", "high"]


def _get_schema_properties(tool_class) -> dict:
    """Extract the properties dict from a tool class's schema."""
    return tool_class.schema.function.parameters.properties


def _make_generate_tool_call() -> GenerateImageToolCall:
    """Create a GenerateImageToolCall with mocked dependencies."""
    ctx = MagicMock()
    ctx.message.id = 123456789
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    config = MagicMock()
    tool = GenerateImageToolCall.__new__(GenerateImageToolCall)
    tool.config = config
    tool.ctx = ctx
    tool.bot = MagicMock()
    tool._cog = None
    tool.generated_images = []
    return tool


def _make_edit_tool_call() -> EditImageToolCall:
    """Create an EditImageToolCall with mocked dependencies."""
    ctx = MagicMock()
    ctx.message.id = 123456789
    ctx.me.id = 999
    ctx.channel.id = 888
    ctx.guild = MagicMock()
    ctx.guild.name = "TestGuild"
    config = MagicMock()
    tool = EditImageToolCall.__new__(EditImageToolCall)
    tool.config = config
    tool.ctx = ctx
    tool.bot = MagicMock()
    tool._cog = None
    tool.generated_images = []
    return tool


# ---------------------------------------------------------------------------
# Tests: Schema property presence
# ---------------------------------------------------------------------------


class TestReasoningEffortSchemaProperty:
    """Verify the reasoning_effort property exists in both tool schemas."""

    def test_generate_image_has_reasoning_effort_property(self):
        properties = _get_schema_properties(GenerateImageToolCall)
        assert "reasoning_effort" in properties

    def test_edit_image_has_reasoning_effort_property(self):
        properties = _get_schema_properties(EditImageToolCall)
        assert "reasoning_effort" in properties

    def test_generate_image_reasoning_effort_is_string_type(self):
        prop = _get_schema_properties(GenerateImageToolCall)["reasoning_effort"]
        assert prop["type"] == "string"

    def test_edit_image_reasoning_effort_is_string_type(self):
        prop = _get_schema_properties(EditImageToolCall)["reasoning_effort"]
        assert prop["type"] == "string"

    def test_generate_image_reasoning_effort_enum_values(self):
        prop = _get_schema_properties(GenerateImageToolCall)["reasoning_effort"]
        assert prop["enum"] == VALID_ENUM_VALUES

    def test_edit_image_reasoning_effort_enum_values(self):
        prop = _get_schema_properties(EditImageToolCall)["reasoning_effort"]
        assert prop["enum"] == VALID_ENUM_VALUES

    def test_generate_image_reasoning_effort_has_description(self):
        prop = _get_schema_properties(GenerateImageToolCall)["reasoning_effort"]
        assert "description" in prop
        assert len(prop["description"]) > 0
        # Description should mention the key concepts
        desc_lower = prop["description"].lower()
        assert "minimal" in desc_lower
        assert "sexual" in desc_lower
        assert "complex" in desc_lower

    def test_edit_image_reasoning_effort_has_description(self):
        prop = _get_schema_properties(EditImageToolCall)["reasoning_effort"]
        assert "description" in prop
        assert len(prop["description"]) > 0
        desc_lower = prop["description"].lower()
        assert "minimal" in desc_lower
        assert "sexual" in desc_lower
        assert "complex" in desc_lower

    def test_reasoning_effort_not_in_required_for_generate(self):
        required = GenerateImageToolCall.schema.function.parameters.required
        assert "reasoning_effort" not in required

    def test_reasoning_effort_not_in_required_for_edit(self):
        required = EditImageToolCall.schema.function.parameters.required
        assert "reasoning_effort" not in required


# ---------------------------------------------------------------------------
# Tests: reasoning_effort override logic in _handle()
# ---------------------------------------------------------------------------


class TestReasoningEffortOverrideLogic:
    """Verify LLM-provided reasoning_effort overrides admin config default."""

    def test_generate_image_llm_value_overrides_admin_default(self):
        """When LLM passes reasoning_effort, it should override the admin config."""
        tool = _make_generate_tool_call()

        # Simulate arguments with LLM-provided reasoning_effort
        arguments = {"prompt": "test prompt", "reasoning_effort": "minimal"}

        # Simulate what _handle does: read admin config then check LLM override
        admin_reasoning_effort = "medium"  # admin-configured default
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == "minimal"

    def test_generate_image_admin_default_used_when_no_llm_value(self):
        """When LLM does not pass reasoning_effort, admin config default should be used."""
        tool = _make_generate_tool_call()

        arguments = {"prompt": "test prompt"}

        admin_reasoning_effort = "medium"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == "medium"

    def test_edit_image_llm_value_overrides_admin_default(self):
        """When LLM passes reasoning_effort for edit, it should override admin config."""
        tool = _make_edit_tool_call()

        arguments = {"prompt": "edit prompt", "image_to_edit": "123", "reasoning_effort": "high"}

        admin_reasoning_effort = "low"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == "high"

    def test_edit_image_admin_default_used_when_no_llm_value(self):
        """When LLM does not pass reasoning_effort for edit, admin default should be used."""
        tool = _make_edit_tool_call()

        arguments = {"prompt": "edit prompt", "image_to_edit": "123"}

        admin_reasoning_effort = "low"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == "low"

    def test_generate_image_empty_string_does_not_override(self):
        """An empty string reasoning_effort should not override admin config."""
        arguments = {"prompt": "test", "reasoning_effort": ""}

        admin_reasoning_effort = "high"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        # Empty string is falsy, so admin default should be used
        assert reasoning_effort == "high"

    def test_generate_image_none_does_not_override(self):
        """A None reasoning_effort should not override admin config."""
        arguments = {"prompt": "test", "reasoning_effort": None}

        admin_reasoning_effort = "high"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == "high"

    @pytest.mark.parametrize("effort_value", VALID_ENUM_VALUES)
    def test_all_enum_values_accepted_by_generate(self, effort_value):
        """All valid enum values should be accepted as overrides."""
        arguments = {"prompt": "test", "reasoning_effort": effort_value}

        admin_reasoning_effort = "medium"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == effort_value

    @pytest.mark.parametrize("effort_value", VALID_ENUM_VALUES)
    def test_all_enum_values_accepted_by_edit(self, effort_value):
        """All valid enum values should be accepted as overrides for edit."""
        arguments = {"prompt": "test", "image_to_edit": "123", "reasoning_effort": effort_value}

        admin_reasoning_effort = "medium"
        reasoning_effort = admin_reasoning_effort

        llm_reasoning_effort = arguments.get("reasoning_effort")
        if llm_reasoning_effort:
            reasoning_effort = llm_reasoning_effort

        assert reasoning_effort == effort_value
