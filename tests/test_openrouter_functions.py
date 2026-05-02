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
)
from aiuser.types.enums import OpenRouterToolType


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