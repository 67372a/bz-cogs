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
