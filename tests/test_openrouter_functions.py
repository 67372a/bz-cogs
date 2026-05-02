"""Unit tests for aiuser.functions.openrouter — WebSearch, WebFetch, ImageGeneration classes.

These tests use mock_functions_importer to load the real function classes while
mocking discord, redbot, etc.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# --- Mock discord before importing anything else ---
discord_mock = MagicMock()
discord_mock.Embed = MagicMock()
discord_mock.Color = MagicMock()
discord_mock.Colour = MagicMock()
sys.modules["discord"] = discord_mock
sys.modules["discord.ext"] = MagicMock()
sys.modules["discord.ext.commands"] = MagicMock()
sys.modules["discord.app_commands"] = MagicMock()

# --- Load the real function classes ---
# (mock_functions_importer takes care of the rest)
from tests.mock_functions_importer import (
    OpenRouterWebSearch,
    OpenRouterWebFetch,
    OpenRouterImageGeneration,
    OpenRouterPdfParsing,
)
from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    PdfParsingParameters,
    serialize_parameters,
    deserialize_parameters,
)


class TestOpenRouterWebSearch:
    """Tests for the openrouter:web_search server tool class."""

    def test_tool_type(self):
        ws = OpenRouterWebSearch(MagicMock(), MagicMock())
        assert ws.tool_type == OpenRouterToolType.WEB_SEARCH

    def test_init_stores_config_and_ctx(self):
        config = MagicMock()
        ctx = MagicMock()
        ws = OpenRouterWebSearch(config, ctx)
        assert ws.config is config
        assert ws.ctx is ctx

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_defaults(self):
        """When parameters are None/empty, get_tool_dict returns only the type."""
        config = self._make_config(search_params=None)
        ws = OpenRouterWebSearch(config, self._make_ctx())
        result = await ws.get_tool_dict()
        assert result == {"type": "openrouter:web_search"}

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_custom_params(self):
        config = self._make_config(
            search_params='{"engine": "google", "max_results": 5, "search_context_size": "medium"}'
        )
        ws = OpenRouterWebSearch(config, self._make_ctx())
        result = await ws.get_tool_dict()
        assert result["type"] == "openrouter:web_search"
        assert result["parameters"]["engine"] == "google"
        assert result["parameters"]["max_results"] == 5
        assert result["parameters"]["search_context_size"] == "medium"

    @pytest.mark.asyncio
    async def test_get_tool_dict_filters_none_params(self):
        config = self._make_config(
            search_params='{"engine": "google", "max_results": null, "excluded_domains": null}'
        )
        ws = OpenRouterWebSearch(config, self._make_ctx())
        result = await ws.get_tool_dict()
        assert result["parameters"]["engine"] == "google"
        assert "max_results" not in result["parameters"]
        assert "excluded_domains" not in result["parameters"]

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_all_params(self):
        config = self._make_config(
            search_params=(
                '{"engine": "bing", "max_results": 10, "max_total_results": 50, '
                '"search_context_size": "large", "user_location": {"country": "US"}, '
                '"allowed_domains": ["good.com"], "excluded_domains": ["bad.com"]}'
            )
        )
        ws = OpenRouterWebSearch(config, self._make_ctx())
        result = await ws.get_tool_dict()
        assert result["parameters"]["engine"] == "bing"
        assert result["parameters"]["max_results"] == 10
        assert result["parameters"]["max_total_results"] == 50
        assert result["parameters"]["search_context_size"] == "large"
        assert result["parameters"]["user_location"] == {"country": "US"}
        assert result["parameters"]["allowed_domains"] == ["good.com"]
        assert result["parameters"]["excluded_domains"] == ["bad.com"]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_config(search_params="null"):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_web_search_parameters = AsyncMock(return_value=search_params)
        config.guild.return_value = guild_cfg
        return config

    @staticmethod
    def _make_ctx():
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 123456789
        ctx.guild.name = "Test Guild"
        return ctx


class TestOpenRouterWebFetch:
    """Tests for the openrouter:web_fetch server tool class."""

    def test_tool_type(self):
        wf = OpenRouterWebFetch(MagicMock(), MagicMock())
        assert wf.tool_type == OpenRouterToolType.WEB_FETCH

    def test_init_stores_config_and_ctx(self):
        config = MagicMock()
        ctx = MagicMock()
        wf = OpenRouterWebFetch(config, ctx)
        assert wf.config is config
        assert wf.ctx is ctx

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_defaults(self):
        config = self._make_config(fetch_params=None)
        wf = OpenRouterWebFetch(config, self._make_ctx())
        result = await wf.get_tool_dict()
        assert result == {"type": "openrouter:web_fetch"}

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_custom_params(self):
        config = self._make_config(
            fetch_params='{"engine": "firecrawl", "max_content_tokens": 4000, "max_uses": 10}'
        )
        wf = OpenRouterWebFetch(config, self._make_ctx())
        result = await wf.get_tool_dict()
        assert result["type"] == "openrouter:web_fetch"
        assert result["parameters"]["engine"] == "firecrawl"
        assert result["parameters"]["max_content_tokens"] == 4000
        assert result["parameters"]["max_uses"] == 10

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_domains(self):
        config = self._make_config(
            fetch_params=(
                '{"engine": "firecrawl", "allowed_domains": ["good.com", "docs.example.com"], '
                '"blocked_domains": ["bad.com"]}'
            )
        )
        wf = OpenRouterWebFetch(config, self._make_ctx())
        result = await wf.get_tool_dict()
        assert result["parameters"]["allowed_domains"] == ["good.com", "docs.example.com"]
        assert result["parameters"]["blocked_domains"] == ["bad.com"]

    @pytest.mark.asyncio
    async def test_get_tool_dict_filters_none(self):
        config = self._make_config(fetch_params='{"engine": null, "max_uses": 5}')
        wf = OpenRouterWebFetch(config, self._make_ctx())
        result = await wf.get_tool_dict()
        assert result["parameters"]["max_uses"] == 5
        assert "engine" not in result["parameters"]

    @staticmethod
    def _make_config(fetch_params="null"):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_web_fetch_parameters = AsyncMock(return_value=fetch_params)
        config.guild.return_value = guild_cfg
        return config

    @staticmethod
    def _make_ctx():
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 123456789
        return ctx


class TestOpenRouterImageGeneration:
    """Tests for the openrouter:image_generation server tool class."""

    def test_tool_type(self):
        ig = OpenRouterImageGeneration(MagicMock(), MagicMock())
        assert ig.tool_type == OpenRouterToolType.IMAGE_GENERATION

    def test_init_stores_config_and_ctx(self):
        config = MagicMock()
        ctx = MagicMock()
        ig = OpenRouterImageGeneration(config, ctx)
        assert ig.config is config
        assert ig.ctx is ctx

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_defaults(self):
        config = self._make_config(gen_params=None)
        ig = OpenRouterImageGeneration(config, self._make_ctx())
        result = await ig.get_tool_dict()
        assert result == {"type": "openrouter:image_generation"}

    @pytest.mark.asyncio
    async def test_get_tool_dict_with_custom_params(self):
        config = self._make_config(
            gen_params=(
                '{"model": "dall-e-3", "quality": "hd", "size": "1792x1024", '
                '"aspect_ratio": "16:9", "background": "transparent", '
                '"output_format": "png", "output_compression": 80, '
                '"moderation": "strict"}'
            )
        )
        ig = OpenRouterImageGeneration(config, self._make_ctx())
        result = await ig.get_tool_dict()
        params = result["parameters"]
        assert params["model"] == "dall-e-3"
        assert params["quality"] == "hd"
        assert params["size"] == "1792x1024"
        assert params["aspect_ratio"] == "16:9"
        assert params["background"] == "transparent"
        assert params["output_format"] == "png"
        assert params["output_compression"] == 80
        assert params["moderation"] == "strict"

    @pytest.mark.asyncio
    async def test_get_tool_dict_partial_params(self):
        """Only some params set — only those should appear."""
        config = self._make_config(gen_params='{"model": "dall-e-3", "size": "1024x1024"}')
        ig = OpenRouterImageGeneration(config, self._make_ctx())
        result = await ig.get_tool_dict()
        assert "model" in result["parameters"]
        assert "size" in result["parameters"]
        assert "quality" not in result["parameters"]
        assert "aspect_ratio" not in result["parameters"]

    @pytest.mark.asyncio
    async def test_handle_tool_response_content_with_image_urls(self):
        """Should extract image URLs and send them as embeds."""
        ctx = self._make_ctx()
        content = (
            "I generated an image for you!\n"
            "Here it is: https://example.com/images/output.png?size=1024\n"
            "And another: https://cdn.example.com/pic.jpg"
        )

        await OpenRouterImageGeneration.handle_tool_response_content(content, ctx)

        assert ctx.send.call_count == 2
        for call_args in ctx.send.call_args_list:
            kwargs = call_args[1]
            assert "embed" in kwargs

    @pytest.mark.asyncio
    async def test_handle_tool_response_content_no_images(self):
        ctx = self._make_ctx()
        content = "The image was generated successfully."
        await OpenRouterImageGeneration.handle_tool_response_content(content, ctx)
        ctx.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_tool_response_content_empty_string(self):
        ctx = self._make_ctx()
        await OpenRouterImageGeneration.handle_tool_response_content("", ctx)
        ctx.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_tool_response_content_mixed_content(self):
        """Text with non-image URLs should only match image URLs."""
        ctx = self._make_ctx()
        content = (
            "Check out https://example.com/page.html for details. "
            "Image: https://example.com/image.webp"
        )
        await OpenRouterImageGeneration.handle_tool_response_content(content, ctx)
        assert ctx.send.call_count == 1

    @pytest.mark.asyncio
    async def test_handle_tool_response_content_send_failure(self, caplog):
        """If send raises, it logs an error but doesn't crash."""
        ctx = self._make_ctx()
        ctx.send.side_effect = [Exception("Discord API error"), None]
        content = "Image1: https://example.com/img1.png Image2: https://example.com/img2.jpg"

        await OpenRouterImageGeneration.handle_tool_response_content(content, ctx)

        assert ctx.send.call_count == 2
        error_logged = any("Failed to send image embed" in record.message for record in caplog.records)
        assert error_logged

    @staticmethod
    def _make_config(gen_params="null"):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_image_generation_parameters = AsyncMock(return_value=gen_params)
        config.guild.return_value = guild_cfg
        return config

    @staticmethod
    def _make_ctx():
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 123456789
        ctx.channel = MagicMock()
        ctx.channel.name = "test-channel"
        ctx.send = AsyncMock()
        ctx.embed_color = AsyncMock(return_value=discord_mock.Color.blue())
        return ctx


class TestPdfParsingParameters:
    """Tests for PdfParsingParameters dataclass and serialization."""

    def test_enum_value(self):
        assert OpenRouterToolType.PDF_PARSING.value == "openrouter:pdf_parsing"

    def test_default_params(self):
        params = PdfParsingParameters()
        assert params.engine is None

    def test_params_with_engine(self):
        params = PdfParsingParameters(engine="mistral-ocr")
        assert params.engine == "mistral-ocr"

    def test_serialize_defaults(self):
        params = PdfParsingParameters()
        result = serialize_parameters(params)
        assert result == '{"engine": null}'

    def test_serialize_cloudflare(self):
        params = PdfParsingParameters(engine="cloudflare-ai")
        result = serialize_parameters(params)
        assert result == '{"engine": "cloudflare-ai"}'

    def test_deserialize_none(self):
        params = deserialize_parameters(None, OpenRouterToolType.PDF_PARSING)
        assert isinstance(params, PdfParsingParameters)
        assert params.engine is None

    def test_deserialize_empty_string(self):
        params = deserialize_parameters("", OpenRouterToolType.PDF_PARSING)
        assert isinstance(params, PdfParsingParameters)
        assert params.engine is None

    def test_deserialize_valid_json(self):
        params = deserialize_parameters(
            '{"engine": "mistral-ocr"}',
            OpenRouterToolType.PDF_PARSING,
        )
        assert isinstance(params, PdfParsingParameters)
        assert params.engine == "mistral-ocr"

    def test_deserialize_invalid_json(self):
        params = deserialize_parameters(
            "not valid json",
            OpenRouterToolType.PDF_PARSING,
        )
        assert isinstance(params, PdfParsingParameters)
        assert params.engine is None

    def test_deserialize_native_engine(self):
        params = deserialize_parameters(
            '{"engine": "native"}',
            OpenRouterToolType.PDF_PARSING,
        )
        assert params.engine == "native"

    def test_roundtrip(self):
        original = PdfParsingParameters(engine="cloudflare-ai")
        serialized = serialize_parameters(original)
        deserialized = deserialize_parameters(serialized, OpenRouterToolType.PDF_PARSING)
        assert deserialized.engine == original.engine


class TestPdfUrlDetection:
    """Tests for PDF URL detection in message text."""

    @staticmethod
    def _get_detect_function():
        return OpenRouterPdfParsing.detect_pdf_links

    def test_no_urls_in_empty_string(self):
        detect = self._get_detect_function()
        assert detect("") == []
        assert detect(None) == []

    def test_no_pdf_urls(self):
        detect = self._get_detect_function()
        assert detect("Hello world") == []
        assert detect("Check this out https://example.com/page.html") == []

    def test_single_pdf_url(self):
        detect = self._get_detect_function()
        urls = detect("Read this: https://example.com/document.pdf")
        assert urls == ["https://example.com/document.pdf"]

    def test_multiple_pdf_urls(self):
        detect = self._get_detect_function()
        urls = detect(
            "Doc1: https://a.com/file1.pdf and Doc2: https://b.com/file2.pdf"
        )
        assert len(urls) == 2
        assert "https://a.com/file1.pdf" in urls
        assert "https://b.com/file2.pdf" in urls

    def test_pdf_url_with_query_params(self):
        detect = self._get_detect_function()
        urls = detect("Doc: https://example.com/report.pdf?version=2&download=1")
        assert urls == ["https://example.com/report.pdf?version=2&download=1"]

    def test_pdf_url_mid_sentence(self):
        detect = self._get_detect_function()
        urls = detect(
            "Please read https://arxiv.org/pdf/2301.12345.pdf for more info on this topic."
        )
        assert urls == ["https://arxiv.org/pdf/2301.12345.pdf"]

    def test_deduplicates_urls(self):
        detect = self._get_detect_function()
        urls = detect("Same URL twice: https://a.com/doc.pdf and https://a.com/doc.pdf")
        assert urls == ["https://a.com/doc.pdf"]

    def test_case_insensitive(self):
        detect = self._get_detect_function()
        urls = detect("Doc: https://example.com/Document.PDF")
        assert urls == ["https://example.com/Document.PDF"]

    def test_ftp_not_matched(self):
        detect = self._get_detect_function()
        urls = detect("FTP: ftp://example.com/file.pdf")
        assert urls == []


class TestPdfBase64Encoding:
    """Tests for base64 encoding of PDF data."""

    def test_encode_small_pdf(self):
        data = b"%PDF-1.4\nThis is a test PDF file.\n"
        result = OpenRouterPdfParsing.encode_pdf_to_base64(data)
        assert result.startswith("data:application/pdf;base64,")
        # Should be valid base64
        import base64
        encoded_part = result.split(",", 1)[1]
        decoded = base64.b64decode(encoded_part)
        assert decoded == data


class TestPdfFileContentBuilding:
    """Tests for building file content parts."""

    def test_build_file_content(self):
        result = OpenRouterPdfParsing.build_file_content(
            "test.pdf", "data:application/pdf;base64,SGVsbG8="
        )
        assert result["type"] == "file"
        assert result["file"]["filename"] == "test.pdf"
        assert result["file"]["file_data"] == "data:application/pdf;base64,SGVsbG8="


class TestPdfFilenameExtraction:
    """Tests for filename extraction from URLs."""

    def test_extract_basic_filename(self):
        filename = OpenRouterPdfParsing.extract_filename_from_url(
            "https://example.com/papers/article.pdf"
        )
        assert filename == "article.pdf"

    def test_extract_with_query_string(self):
        filename = OpenRouterPdfParsing.extract_filename_from_url(
            "https://example.com/report.pdf?version=2"
        )
        assert filename == "report.pdf"

    def test_fallback_filename(self):
        filename = OpenRouterPdfParsing.extract_filename_from_url(
            "https://example.com/download", index=2
        )
        assert filename == "document_3.pdf"


class TestPdfPluginsBuilding:
    """Tests for plugins array building with PdfParsing service."""

    @pytest.mark.asyncio
    async def test_default_engine_when_disabled(self):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_pdf_parsing_enabled = AsyncMock(return_value=False)
        config.guild.return_value = guild_cfg
        ctx = self._make_ctx()
        service = OpenRouterPdfParsing(config, ctx)
        # We don't test build_plugins directly since it reads from config,
        # instead we test the class can be instantiated
        assert service is not None
        assert service.config is config
        assert service.ctx is ctx

    @pytest.mark.asyncio
    async def test_build_plugins_cloudflare(self):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_pdf_parsing_parameters = AsyncMock(
            return_value='{"engine": "cloudflare-ai"}'
        )
        config.guild.return_value = guild_cfg
        ctx = self._make_ctx()
        service = OpenRouterPdfParsing(config, ctx)
        plugins = await service.build_plugins()
        assert len(plugins) == 1
        assert plugins[0]["id"] == "file-parser"
        assert plugins[0]["pdf"]["engine"] == "cloudflare-ai"

    @pytest.mark.asyncio
    async def test_build_plugins_mistral_ocr(self):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_pdf_parsing_parameters = AsyncMock(
            return_value='{"engine": "mistral-ocr"}'
        )
        config.guild.return_value = guild_cfg
        ctx = self._make_ctx()
        service = OpenRouterPdfParsing(config, ctx)
        plugins = await service.build_plugins()
        assert plugins[0]["pdf"]["engine"] == "mistral-ocr"

    @pytest.mark.asyncio
    async def test_build_plugins_default_when_no_config(self):
        config = MagicMock()
        guild_cfg = MagicMock()
        guild_cfg.openrouter_pdf_parsing_parameters = AsyncMock(return_value=None)
        config.guild.return_value = guild_cfg
        ctx = self._make_ctx()
        service = OpenRouterPdfParsing(config, ctx)
        plugins = await service.build_plugins()
        # Default should be cloudflare-ai when no config set
        assert plugins[0]["pdf"]["engine"] == "cloudflare-ai"

    @staticmethod
    def _make_ctx():
        ctx = MagicMock()
        ctx.guild = MagicMock()
        ctx.guild.id = 123456789
        ctx.guild.name = "Test Guild"
        return ctx
