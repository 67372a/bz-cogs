from enum import Enum, auto


class ScanImageMode(Enum):
    LOCAL = "local"
    AI_HORDE = "ai-horde"
    LLM = "supported-llm"

class MentionType(Enum):
    SERVER = auto()
    USER = auto()
    ROLE = auto()
    CHANNEL = auto()

class OpenRouterToolType(Enum):
    WEB_SEARCH = "openrouter:web_search"
    WEB_FETCH = "openrouter:web_fetch"
    IMAGE_GENERATION = "openrouter:image_generation"
    PDF_PARSING = "openrouter:pdf_parsing"


class ServiceTier(Enum):
    """Service tier for OpenAI/OpenRouter API requests.

    Controls cost/latency tradeoffs. ``DEFAULT`` omits the parameter entirely
    (the API uses its own default tier). ``FLEX`` trades speed for lower cost.
    ``PRIORITY`` trades cost for lower latency.

    References:
        - https://openrouter.ai/docs/features/service-tiers
        - https://developers.openai.com/api-reference/resources/chat/subresources/completions/methods/create
    """
    DEFAULT = None
    FLEX = "flex"
    PRIORITY = "priority"
