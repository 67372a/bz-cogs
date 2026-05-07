import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

from aiuser.types.enums import OpenRouterToolType


@dataclass
class WebSearchParameters:
    engine: Optional[str] = None
    max_results: Optional[int] = None
    max_total_results: Optional[int] = None
    search_context_size: Optional[str] = None
    user_location: Optional[Dict[str, str]] = None
    allowed_domains: Optional[List[str]] = None
    excluded_domains: Optional[List[str]] = None


@dataclass
class WebFetchParameters:
    engine: Optional[str] = None
    max_uses: Optional[int] = None
    max_content_tokens: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None


@dataclass
class ImageGenerationParameters:
    model: Optional[str] = None
    quality: Optional[str] = None
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    background: Optional[str] = None
    output_format: Optional[str] = None
    output_compression: Optional[int] = None
    moderation: Optional[str] = None


@dataclass
class PdfParsingParameters:
    engine: Optional[str] = None  # "cloudflare-ai", "mistral-ocr", or "native"


@dataclass
class DirectImageGenerationParameters:
    """Preconfigured parameters for the local generate_image ToolCall.

    These are set by server admins. The model and image_size are fixed
    across all generations, while the LLM passes prompt and aspect_ratio
    dynamically in the function call.
    """
    model: Optional[str] = None       # e.g. "google/gemini-3.1-flash-image-preview"
    image_size: Optional[str] = None  # "1K", "2K", "4K", "0.5K"


@dataclass
class DirectImageEditParameters:
    """Preconfigured parameters for the local edit_image ToolCall.

    These are set by server admins. The model and image_size are fixed
    across all edits, while the LLM passes prompt, image_to_edit, and
    aspect_ratio dynamically in the function call.
    """
    model: Optional[str] = None       # e.g. "google/gemini-3.1-flash-image-preview"
    image_size: Optional[str] = None  # "1K", "2K", "4K", "0.5K"


_PARAM_CLASS_MAP = {
    OpenRouterToolType.WEB_SEARCH: WebSearchParameters,
    OpenRouterToolType.WEB_FETCH: WebFetchParameters,
    OpenRouterToolType.IMAGE_GENERATION: ImageGenerationParameters,
    OpenRouterToolType.PDF_PARSING: PdfParsingParameters,
}

# Additional string-keyed entries for local ToolCalls that don't use OpenRouterToolType
_PARAM_CLASS_MAP["direct_image_generation"] = DirectImageGenerationParameters
_PARAM_CLASS_MAP["direct_image_edit"] = DirectImageEditParameters


def build_openrouter_tool_dict(tool_type: OpenRouterToolType, parameters: dict) -> dict:
    """
    Build a raw tool dict for the OpenAI tools array with type 'openrouter:...'.

    Args:
        tool_type: The OpenRouterToolType enum value.
        parameters: Dictionary of parameter key-value pairs. None values are filtered out.

    Returns:
        A dict like {"type": "openrouter:web_search", "parameters": {...}}
    """
    filtered_params = {k: v for k, v in parameters.items() if v is not None}
    result: Dict[str, Any] = {"type": tool_type.value}
    if filtered_params:
        result["parameters"] = filtered_params
    return result


def serialize_parameters(params) -> str:
    """Serialize a parameters dataclass to a JSON string."""
    return json.dumps(asdict(params))


def deserialize_parameters(json_str: Optional[str], tool_type):
    """
    Deserialize a JSON string into the appropriate parameters dataclass.

    Args:
        json_str: JSON string to deserialize, or None.
        tool_type: The OpenRouterToolType (or string key) to determine which
            dataclass to use. Supports both OpenRouterToolType enum values and
            string keys like "direct_image_generation".

    Returns:
        An instance of the appropriate parameters dataclass with default values
        if json_str is None or invalid.
    """
    param_class = _PARAM_CLASS_MAP.get(tool_type, WebSearchParameters)
    if not json_str:
        return param_class()
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return param_class()
        # Filter to only valid fields for the dataclass
        valid_fields = {f.name for f in fields(param_class)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return param_class(**filtered_data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return param_class()