"""Unit tests for aiuser.types.openrouter_types — pure data/utility functions.

These tests use direct module loading (via mock_importer) to avoid triggering
the full aiuser/__init__.py dependency chain.
"""

import json
import pytest

from tests.mock_importer import (
    OpenRouterToolType,
    WebSearchParameters,
    WebFetchParameters,
    ImageGenerationParameters,
    build_openrouter_tool_dict,
    serialize_parameters,
    deserialize_parameters,
)


class TestWebSearchParameters:
    def test_default_values(self):
        params = WebSearchParameters()
        assert params.engine is None
        assert params.max_results is None
        assert params.max_total_results is None
        assert params.search_context_size is None
        assert params.user_location is None
        assert params.allowed_domains is None
        assert params.excluded_domains is None

    def test_custom_values(self):
        params = WebSearchParameters(
            engine="google",
            max_results=5,
            max_total_results=100,
            search_context_size="medium",
            user_location={"country": "US", "region": "California"},
            allowed_domains=["example.com"],
            excluded_domains=["bad.example.com"],
        )
        assert params.engine == "google"
        assert params.max_results == 5
        assert params.max_total_results == 100
        assert params.search_context_size == "medium"
        assert params.user_location == {"country": "US", "region": "California"}
        assert params.allowed_domains == ["example.com"]
        assert params.excluded_domains == ["bad.example.com"]


class TestWebFetchParameters:
    def test_default_values(self):
        params = WebFetchParameters()
        assert params.engine is None
        assert params.max_uses is None
        assert params.max_content_tokens is None
        assert params.allowed_domains is None
        assert params.blocked_domains is None

    def test_custom_values(self):
        params = WebFetchParameters(
            engine="firecrawl",
            max_uses=10,
            max_content_tokens=4000,
            allowed_domains=["docs.example.com"],
            blocked_domains=["spam.com"],
        )
        assert params.engine == "firecrawl"
        assert params.max_uses == 10
        assert params.max_content_tokens == 4000
        assert params.allowed_domains == ["docs.example.com"]
        assert params.blocked_domains == ["spam.com"]


class TestImageGenerationParameters:
    def test_default_values(self):
        params = ImageGenerationParameters()
        assert params.model is None
        assert params.quality is None
        assert params.size is None
        assert params.aspect_ratio is None
        assert params.background is None
        assert params.output_format is None
        assert params.output_compression is None
        assert params.moderation is None

    def test_custom_values(self):
        params = ImageGenerationParameters(
            model="dall-e-3",
            quality="hd",
            size="1792x1024",
            aspect_ratio="16:9",
            background="transparent",
            output_format="png",
            output_compression=80,
            moderation="strict",
        )
        assert params.model == "dall-e-3"
        assert params.quality == "hd"
        assert params.size == "1792x1024"
        assert params.aspect_ratio == "16:9"
        assert params.background == "transparent"
        assert params.output_format == "png"
        assert params.output_compression == 80
        assert params.moderation == "strict"


class TestBuildOpenRouterToolDict:
    def test_build_web_search_basic(self):
        result = build_openrouter_tool_dict(
            OpenRouterToolType.WEB_SEARCH,
            {"engine": "google", "max_results": 5},
        )
        assert result == {
            "type": "openrouter:web_search",
            "parameters": {"engine": "google", "max_results": 5},
        }

    def test_build_web_fetch_basic(self):
        result = build_openrouter_tool_dict(
            OpenRouterToolType.WEB_FETCH,
            {"engine": "firecrawl", "max_content_tokens": 3000},
        )
        assert result == {
            "type": "openrouter:web_fetch",
            "parameters": {"engine": "firecrawl", "max_content_tokens": 3000},
        }

    def test_build_image_generation_basic(self):
        result = build_openrouter_tool_dict(
            OpenRouterToolType.IMAGE_GENERATION,
            {"model": "dall-e-3", "size": "1024x1024"},
        )
        assert result == {
            "type": "openrouter:image_generation",
            "parameters": {"model": "dall-e-3", "size": "1024x1024"},
        }

    def test_filters_none_values(self):
        result = build_openrouter_tool_dict(
            OpenRouterToolType.WEB_SEARCH,
            {"engine": "google", "max_results": None, "search_context_size": None},
        )
        assert result == {
            "type": "openrouter:web_search",
            "parameters": {"engine": "google"},
        }

    def test_empty_parameters_omitted(self):
        """When all parameters are None, the 'parameters' key should be omitted."""
        result = build_openrouter_tool_dict(
            OpenRouterToolType.WEB_SEARCH,
            {"engine": None, "max_results": None},
        )
        assert result == {"type": "openrouter:web_search"}
        assert "parameters" not in result

    def test_round_trip_with_build(self):
        d = build_openrouter_tool_dict(
            OpenRouterToolType.WEB_SEARCH,
            {"engine": "google", "max_results": 10},
        )
        assert d["type"] == "openrouter:web_search"
        assert d["parameters"]["engine"] == "google"
        assert d["parameters"]["max_results"] == 10


class TestSerializeParameters:
    def test_serialize_web_search(self):
        params = WebSearchParameters(engine="google", max_results=10)
        result = serialize_parameters(params)
        data = json.loads(result)
        assert data["engine"] == "google"
        assert data["max_results"] == 10

    def test_serialize_image_generation(self):
        params = ImageGenerationParameters(model="dall-e-3", size="1024x1024")
        result = serialize_parameters(params)
        data = json.loads(result)
        assert data["model"] == "dall-e-3"
        assert data["size"] == "1024x1024"
        assert data["quality"] is None


class TestDeserializeParameters:
    def test_deserialize_web_search(self):
        json_str = '{"engine": "google", "max_results": 5}'
        result = deserialize_parameters(json_str, OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine == "google"
        assert result.max_results == 5

    def test_deserialize_web_fetch(self):
        json_str = '{"engine": "firecrawl", "max_content_tokens": 4000}'
        result = deserialize_parameters(json_str, OpenRouterToolType.WEB_FETCH)
        assert isinstance(result, WebFetchParameters)
        assert result.engine == "firecrawl"
        assert result.max_content_tokens == 4000

    def test_deserialize_image_generation(self):
        json_str = '{"model": "dall-e-3", "size": "1024x1024", "quality": "hd"}'
        result = deserialize_parameters(json_str, OpenRouterToolType.IMAGE_GENERATION)
        assert isinstance(result, ImageGenerationParameters)
        assert result.model == "dall-e-3"
        assert result.size == "1024x1024"
        assert result.quality == "hd"

    def test_deserialize_none_returns_defaults(self):
        result = deserialize_parameters(None, OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine is None
        assert result.max_results is None

    def test_deserialize_empty_string_returns_defaults(self):
        result = deserialize_parameters("", OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine is None

    def test_deserialize_invalid_json_returns_defaults(self):
        result = deserialize_parameters("not valid json", OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine is None

    def test_deserialize_non_dict_json_returns_defaults(self):
        result = deserialize_parameters('["list", "not", "dict"]', OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine is None

    def test_deserialize_filters_unknown_fields(self):
        """Extra fields in JSON should be silently ignored."""
        json_str = '{"engine": "google", "max_results": 5, "unknown_field": "should_be_ignored"}'
        result = deserialize_parameters(json_str, OpenRouterToolType.WEB_SEARCH)
        assert isinstance(result, WebSearchParameters)
        assert result.engine == "google"
        assert result.max_results == 5
        with pytest.raises(AttributeError):
            _ = result.unknown_field

    def test_deserialize_round_trip(self):
        """Serialize then deserialize should produce the same values for known fields."""
        original = WebSearchParameters(engine="bing", max_results=20)
        serialized = serialize_parameters(original)
        restored = deserialize_parameters(serialized, OpenRouterToolType.WEB_SEARCH)
        assert restored.engine == original.engine
        assert restored.max_results == original.max_results