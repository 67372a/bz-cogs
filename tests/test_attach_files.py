"""Tests for the attach_files function call.

Verifies that:
- AttachFilesToolCall schema is correctly defined
- File validation rejects empty filenames, empty content, and oversized files
- Filenames are sanitized (path components stripped)
- Content is encoded to UTF-8 bytes correctly
- get_attached_files() returns stored files and clears the internal list
- _collect_attached_files_from_tool in LLMPipeline collects files from AttachFilesToolCall
- Multiple files can be attached in a single call
- Error messages are returned for invalid inputs
"""

import asyncio
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_tool_call_context.py pattern)
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
    "aiuser.functions.tool_call", "aiuser.functions.attach_files",
    "aiuser.functions.generate_image", "aiuser.functions.edit_image",
    "aiuser.config", "aiuser.types", "aiuser.utils", "aiuser.response",
    "aiuser.response.chat",
    "aiuser.functions.openrouter",
    "redbot", "redbot.core", "redbot.core.commands",
    "redbot.core.bot", "redbot.core.utils", "redbot.core.utils.chat_formatting",
    "discord", "discord.ext", "discord.ext.commands",
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

# Load real attach_files tool_call module
sys.modules["aiuser.functions.attach_files"] = _ensure_package("aiuser.functions.attach_files") or sys.modules["aiuser.functions.attach_files"]
sys.modules["aiuser.functions.attach_files.tool_call"] = import_module_directly(
    "aiuser.functions.attach_files.tool_call", "aiuser/functions/attach_files/tool_call.py"
)

AttachFilesToolCall = sys.modules["aiuser.functions.attach_files.tool_call"].AttachFilesToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(config=None, ctx=None):
    """Create an AttachFilesToolCall instance with mocked config and ctx."""
    if config is None:
        config = MagicMock()
    if ctx is None:
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 999
    return AttachFilesToolCall(config=config, ctx=ctx)


def _run_async(coro):
    """Run an async coroutine in a new event loop (for tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests: Schema
# ---------------------------------------------------------------------------

class TestAttachFilesSchema:
    """Verify the tool call schema is correctly defined."""

    def test_function_name(self):
        assert AttachFilesToolCall.function_name == "attach_files"

    def test_schema_has_files_parameter(self):
        params = AttachFilesToolCall.schema.function.parameters
        assert "files" in params.properties
        assert params.required == ["files"]

    def test_files_is_array_type(self):
        files_prop = AttachFilesToolCall.schema.function.parameters.properties["files"]
        assert files_prop["type"] == "array"

    def test_files_items_require_filename_and_content(self):
        files_prop = AttachFilesToolCall.schema.function.parameters.properties["files"]
        items = files_prop["items"]
        assert items["type"] == "object"
        assert set(items["required"]) == {"filename", "content"}

    def test_schema_type_is_function(self):
        assert AttachFilesToolCall.schema.type == "function"

    def test_description_mentions_attachments(self):
        desc = AttachFilesToolCall.schema.function.description.lower()
        assert "attach" in desc


# ---------------------------------------------------------------------------
# Tests: _handle validation
# ---------------------------------------------------------------------------

class TestAttachFilesValidation:
    """Verify input validation in _handle()."""

    def test_empty_files_list_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": []}))
        assert "Error" in result
        assert "No files" in result

    def test_missing_files_key_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({}))
        assert "Error" in result

    def test_empty_filename_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "", "content": "hello"}]}))
        assert "Error" in result
        assert "Empty filename" in result

    def test_empty_content_still_allowed(self):
        """Empty string content should be allowed (valid but empty file)."""
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "empty.txt", "content": ""}]}))
        assert "Successfully attached" in result
        assert "`empty.txt`" in result

    def test_none_content_returns_error(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "test.py", "content": None}]}))
        assert "Error" in result
        assert "Empty content" in result

    def test_oversized_file_returns_error(self):
        tool = _make_tool()
        # Create content larger than 25MB
        large_content = "x" * (25 * 1024 * 1024 + 1)
        result = _run_async(tool.run({"files": [{"filename": "huge.txt", "content": large_content}]}))
        assert "Error" in result
        assert "exceeds" in result

    def test_too_many_files_returns_error(self):
        tool = _make_tool()
        files = [{"filename": f"file{i}.txt", "content": "content"} for i in range(11)]
        result = _run_async(tool.run({"files": files}))
        assert "Error" in result
        assert "Too many files" in result


# ---------------------------------------------------------------------------
# Tests: Filename sanitization
# ---------------------------------------------------------------------------

class TestFilenameSanitization:
    """Verify filenames are properly sanitized."""

    def test_path_components_stripped(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "../../etc/passwd", "content": "data"}]}))
        # os.path.basename("../../etc/passwd") == "passwd" which is valid
        assert "Successfully attached" in result
        assert "`passwd`" in result

    def test_dotfile_rejected(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": ".hidden", "content": "data"}]}))
        assert "Error" in result
        assert "Invalid filename" in result

    def test_windows_path_stripped(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "C:\\Users\\test\\file.py", "content": "print()"}]}))
        assert "Successfully attached" in result
        assert "`file.py`" in result

    def test_whitespace_trimmed(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "  my_file.py  ", "content": "code"}]}))
        assert "Successfully attached" in result
        assert "`my_file.py`" in result


# ---------------------------------------------------------------------------
# Tests: Successful file attachment
# ---------------------------------------------------------------------------

class TestAttachFilesSuccess:
    """Verify successful file attachment behavior."""

    def test_single_file_attached(self):
        tool = _make_tool()
        result = _run_async(tool.run({"files": [{"filename": "main.py", "content": "print('hello')"}]}))
        assert "Successfully attached 1 file(s)" in result
        assert "`main.py`" in result

    def test_multiple_files_attached(self):
        tool = _make_tool()
        files = [
            {"filename": "main.py", "content": "import os"},
            {"filename": "utils.py", "content": "def helper(): pass"},
            {"filename": "config.json", "content": '{"key": "value"}'},
        ]
        result = _run_async(tool.run({"files": files}))
        assert "Successfully attached 3 file(s)" in result
        assert "`main.py`" in result
        assert "`utils.py`" in result
        assert "`config.json`" in result

    def test_stored_files_have_correct_format(self):
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "test.py", "content": "print('hi')"}]}))
        files = tool.get_attached_files()
        assert len(files) == 1
        assert files[0]["filename"] == "test.py"
        assert files[0]["bytes"] == "print('hi')".encode("utf-8")
        assert isinstance(files[0]["bytes"], bytes)

    def test_utf8_content_encoded_correctly(self):
        tool = _make_tool()
        content = "Hello 世界! 🎉 café"
        _run_async(tool.run({"files": [{"filename": "unicode.txt", "content": content}]}))
        files = tool.get_attached_files()
        assert files[0]["bytes"] == content.encode("utf-8")

    def test_get_attached_files_clears_internal_list(self):
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "a.py", "content": "code"}]}))
        files1 = tool.get_attached_files()
        assert len(files1) == 1
        files2 = tool.get_attached_files()
        assert len(files2) == 0

    def test_get_attached_files_returns_copy(self):
        """Verifies get_attached_files() returns a shallow copy and clears internal list."""
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "a.py", "content": "code"}]}))

        # First call returns the file data and clears the internal list
        files1 = tool.get_attached_files()
        assert len(files1) == 1
        assert files1[0]["filename"] == "a.py"

        # Second call returns empty (internal list was already cleared)
        files2 = tool.get_attached_files()
        assert len(files2) == 0

        # The first returned list is still intact (independent copy)
        assert len(files1) == 1

    def test_partial_success_with_errors(self):
        """Some files valid, some invalid — should attach the valid ones."""
        tool = _make_tool()
        files = [
            {"filename": "good.py", "content": "code"},
            {"filename": "", "content": "bad"},  # invalid
            {"filename": "also_good.js", "content": "var x = 1;"},
        ]
        result = _run_async(tool.run({"files": files}))
        assert "Successfully attached 2 file(s)" in result
        assert "`good.py`" in result
        assert "`also_good.js`" in result
        assert "Error" in result
        attached = tool.get_attached_files()
        assert len(attached) == 2

    def test_mixed_success_and_size_error(self):
        """Valid file plus oversized file."""
        tool = _make_tool()
        large_content = "x" * (25 * 1024 * 1024 + 1)
        files = [
            {"filename": "small.txt", "content": "ok"},
            {"filename": "huge.txt", "content": large_content},
        ]
        result = _run_async(tool.run({"files": files}))
        assert "Successfully attached 1 file(s)" in result
        assert "`small.txt`" in result
        attached = tool.get_attached_files()
        assert len(attached) == 1

    def test_all_files_invalid_returns_error(self):
        tool = _make_tool()
        files = [
            {"filename": "", "content": "bad1"},
            {"filename": "", "content": "bad2"},
        ]
        result = _run_async(tool.run({"files": files}))
        assert "Error" in result
        assert "No files could be attached" in result


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineFileCollection:
    """Verify _collect_attached_files_from_tool works in the pipeline."""

    def _make_pipeline_mock(self):
        """Create a minimal LLMPipeline-like object for testing."""
        from types import ModuleType

        # Load the real pipeline module if not already loaded
        if "aiuser.response.chat.llm_pipeline" not in sys.modules or \
           not hasattr(sys.modules["aiuser.response.chat.llm_pipeline"], "LLMPipeline"):
            # Mock dependencies needed by llm_pipeline
            for dep in [
                "aiuser.functions.openrouter",
                "aiuser.functions.openrouter.web_search",
                "aiuser.functions.openrouter.web_fetch",
                "aiuser.functions.openrouter.image_generation",
                "aiuser.functions.openrouter.pdf_parsing",
                "aiuser.functions.openrouter.image_parsing",
                "aiuser.response.chat.function_call_view",
                "aiuser.response.chat.response",
                "tenacity", "tenacity.retry", "tenacity.stop_after_attempt",
                "tenacity.wait_random_exponential",
                "tenacity.retry_if_exception_type", "tenacity.retry_if_result",
                "httpx", "aiuser.config.models", "aiuser.config.constants",
            ]:
                _ensure_package(dep) if "." in dep else None
                if dep not in sys.modules:
                    sys.modules[dep] = MagicMock()

            # Mock openai with proper types
            _openai_mock = MagicMock()
            _openai_mock.AsyncOpenAI = MagicMock()
            _openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
            _openai_mock.APIConnectionError = type("APIConnectionError", (Exception,), {})
            _openai_mock.InternalServerError = type("InternalServerError", (Exception,), {})
            _openai_mock.APIError = type("APIError", (Exception,), {})
            _openai_mock.APIStatusError = type("APIStatusError", (Exception,), {})

            _chat_mock = MagicMock()
            _chat_mock.ChatCompletion = MagicMock()
            _chat_mock.ChatCompletionMessageToolCall = MagicMock()

            sys.modules["openai"] = _openai_mock
            _ensure_package("openai.types")
            sys.modules["openai.types"] = MagicMock()
            sys.modules["openai.types.chat"] = _chat_mock

            # Mock openrouter package with required classes
            _or_mock = sys.modules.setdefault("aiuser.functions.openrouter", MagicMock())
            _or_mock.OpenRouterWebSearch = MagicMock()
            _or_mock.OpenRouterWebFetch = MagicMock()
            _or_mock.OpenRouterImageGeneration = MagicMock()
            _or_mock.OpenRouterPdfParsing = MagicMock()
            _or_mock.OpenRouterImageParsing = MagicMock()

            # Mock generate_image and edit_image tool_call modules
            _ensure_package("aiuser.functions.generate_image")
            _ensure_package("aiuser.functions.edit_image")
            _gi_mock = MagicMock()
            _gi_mock.GenerateImageToolCall = type("GenerateImageToolCall", (), {})
            sys.modules["aiuser.functions.generate_image.tool_call"] = _gi_mock
            _ei_mock = MagicMock()
            _ei_mock.EditImageToolCall = type("EditImageToolCall", (), {})
            sys.modules["aiuser.functions.edit_image.tool_call"] = _ei_mock

            # Mock aiuser.config.models with required constants
            _config_models_mock = MagicMock()
            _config_models_mock.UNSUPPORTED_LOGIT_BIAS_MODELS = set()
            _config_models_mock.VISION_SUPPORTED_MODELS = set()
            sys.modules["aiuser.config.models"] = _config_models_mock

            # Mock aiuser.config.constants
            _config_constants_mock = MagicMock()
            _config_constants_mock.REGEX_RUN_TIMEOUT = 5
            _config_constants_mock.OPENROUTER_URL = "https://openrouter.ai/api/v1"
            _config_constants_mock.YOUTUBE_URL_PATTERN = r".*"
            sys.modules["aiuser.config.constants"] = _config_constants_mock

            # Mock aiuser.messages_list and its submodules
            _ensure_package("aiuser.messages_list")
            sys.modules["aiuser.messages_list.entry"] = MagicMock()
            sys.modules["aiuser.messages_list.messages"] = MagicMock()

            # Mock aiuser.types.abc (MixinMeta)
            _ensure_package("aiuser.types")
            sys.modules["aiuser.types.abc"] = MagicMock()

            # Mock aiuser.types.enums (OpenRouterToolType)
            _enums_mock = MagicMock()
            _enums_mock_val = MagicMock()
            _enums_mock_val.WEB_SEARCH.value = "web_search"
            _enums_mock_val.WEB_FETCH.value = "web_fetch"
            _enums_mock_val.IMAGE_GENERATION.value = "image_generation"
            _enums_mock.OpenRouterToolType = _enums_mock_val
            sys.modules["aiuser.types.enums"] = _enums_mock

            # Mock aiuser.utils.utilities
            _ensure_package("aiuser.utils")
            _utils_mock = MagicMock()
            _utils_mock.get_enabled_tools = MagicMock(return_value=[])
            sys.modules["aiuser.utils.utilities"] = _utils_mock

            sys.modules["aiuser.response.chat.function_call_view"] = MagicMock()

            sys.modules["aiuser.response.chat.llm_pipeline"] = import_module_directly(
                "aiuser.response.chat.llm_pipeline", "aiuser/response/chat/llm_pipeline.py"
            )

        LLMPipeline = sys.modules["aiuser.response.chat.llm_pipeline"].LLMPipeline

        pipeline = LLMPipeline.__new__(LLMPipeline)
        pipeline.ctx = MagicMock()
        pipeline.ctx.guild.id = 123
        pipeline.msg_list = MagicMock()
        pipeline.enabled_tools = []
        pipeline.collected_images = []
        return pipeline

    def test_collect_attached_files_returns_files(self):
        pipeline = self._make_pipeline_mock()
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "main.py", "content": "print(1)"}]}))
        pipeline.enabled_tools = [tool]

        files = pipeline._collect_attached_files_from_tool("attach_files")
        assert len(files) == 1
        assert files[0]["filename"] == "main.py"
        assert files[0]["bytes"] == b"print(1)"

    def test_collect_attached_files_clears_tool_list(self):
        pipeline = self._make_pipeline_mock()
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "a.py", "content": "x"}]}))
        pipeline.enabled_tools = [tool]

        files1 = pipeline._collect_attached_files_from_tool("attach_files")
        assert len(files1) == 1
        files2 = pipeline._collect_attached_files_from_tool("attach_files")
        assert len(files2) == 0

    def test_collect_attached_files_wrong_name_returns_empty(self):
        pipeline = self._make_pipeline_mock()
        tool = _make_tool()
        _run_async(tool.run({"files": [{"filename": "a.py", "content": "x"}]}))
        pipeline.enabled_tools = [tool]

        files = pipeline._collect_attached_files_from_tool("other_function")
        assert len(files) == 0

    def test_collect_attached_files_no_matching_tool_returns_empty(self):
        pipeline = self._make_pipeline_mock()
        pipeline.enabled_tools = []
        files = pipeline._collect_attached_files_from_tool("attach_files")
        assert len(files) == 0

    def test_collect_attached_files_multiple_files(self):
        pipeline = self._make_pipeline_mock()
        tool = _make_tool()
        _run_async(tool.run({"files": [
            {"filename": "a.py", "content": "a"},
            {"filename": "b.py", "content": "b"},
            {"filename": "c.py", "content": "c"},
        ]}))
        pipeline.enabled_tools = [tool]

        files = pipeline._collect_attached_files_from_tool("attach_files")
        assert len(files) == 3
        names = [f["filename"] for f in files]
        assert names == ["a.py", "b.py", "c.py"]
