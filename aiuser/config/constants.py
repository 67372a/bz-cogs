import re

IMAGE_REQUEST_CHECK_PROMPT = "As an AI, named {botname}, you are tasked to analyze messages directed towards you. Your role is to identify whether each specific message is asking you to send a picture of yourself or not. Messages can be phrased in a variety of ways, so you should look for key contextual clues such as requests for images, photographs, selfies, or other synonyms, but make sure it's specifically asking for a picture of 'you'. If the message explicitly requests a picture of {botname}, you are to respond with 'True'. If the message doesn't solicit a picture of 'you', then respond with 'False'."
IMAGE_REQUEST_REPLY_PROMPT = "You sent the picture above. Respond accordingly."
IMAGE_REQUEST_AIHORDE_URL = "https://aihorde.net/api"

RANDOM_MESSAGE_TASK_RETRY_SECONDS = 33 * 60

GROK_PRIMARY_TRIGGERS = ["grok", "gork"]
GROK_SECONDARY_TRIGGERS = ["true", "explain", "confirm"]
GROK_MAX_WORDS = 25

# regex patterns
URL_PATTERN = re.compile(r"(https?://\S+)")
YOUTUBE_URL_PATTERN = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(.+)")
YOUTUBE_VIDEO_ID_PATTERN = re.compile(
    r"(?:youtube(?:-nocookie)?\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/|v\/|t\/\S*?\/?)([a-zA-Z0-9_-]{11})")
SINGULAR_MENTION_PATTERN = re.compile(r"^<@!?&?(\d+)>$")
REGEX_RUN_TIMEOUT = 10

# Reserved tag names of the Semantic XML context format. Tags with these names
# must never be interpreted from user input (injection) nor shown in output (leak).
XML_RESERVED_TAG_NAMES = ("message", "image", "file", "sticker", "document")
# Matches opening, closing, and self-closing tags of the reserved schema.
# `[^>]` spans newlines, so multi-line attribute lists are covered.
XML_RESERVED_TAG_PATTERN = re.compile(
    r"</?(?:" + "|".join(XML_RESERVED_TAG_NAMES) + r")\b[^>]*>",
    re.IGNORECASE,
)


OPENROUTER_URL = "https://openrouter.ai/api/"

XML_SYSTEM_PROMPT_APPENDIX = (
    "IMPORTANT CONTEXTUAL INSTRUCTIONS:\n"
    "1. The chat history below is formatted in Semantic XML. "
    "All messages (including your own past responses) are wrapped in <message> tags "
    "containing metadata (id, timestamp, author_id, username, displayname, and reply_to_* attributes "
    "identifying the message and author being replied to). "
    "You can identify your own past messages by matching author_id against your own bot ID.\n"
    "2. Media attachments are represented by placeholder tags like <image/>, <file/>, or <sticker/> "
    "with filename/title/description attributes. A placeholder tag means the media was NOT provided "
    "to you — do not pretend you can see its contents. Text documents may appear inline as "
    "<document filename=\"...\">content</document>.\n"
    "3. You are part of this chat. Do not output these XML tags yourself. "
    "Respond naturally with text, inclusive of markdown and emojis as is relevant, adhering to the persona described above."
)

MATH_NOTATION_INSTRUCTIONS = (
    "MATH NOTATION:\n"
    "Do NOT use LaTeX formatting ($...$, $$...$$, \\frac, \\sum, etc.) in your responses. "
    "Instead, use plain Unicode math symbols where possible: ×, ÷, ≥, ≤, ≈, √, ², ³, α, β, π, "
    "→, ←, ∫, Σ, etc. For fractions, write them as (numerator)/(denominator). "
    "For complex mathematical expressions that cannot be expressed in Unicode, "
    "use a fenced code block (```...```)."
)

CITATION_INSTRUCTIONS = (
    "CITATION REQUIREMENTS:\n"
    "When you use information from web_search or web_fetch tools, you MUST include "
    "markdown-based source citations for factual claims derived from those results.\n\n"
    "Format: Use numbered footnotes — `[^1]`, `[^2]` — inline after sourced claims. "
    "List the source URLs at the end of your message:\n"
    "  `[^1]: https://source-url.com`\n"
    "  `[^2]: https://another-source.org`\n\n"
    "Rules:\n"
    "- Only cite information actually retrieved via the tools, not your own knowledge.\n"
    "- Use only URLs returned by the tools. If no URL was provided, write:\n"
    "  `[^N]: Retrieved via web search — no direct URL provided`\n"
    "- Place the source list at the very end of your message, after a blank line.\n"
    "- Group citations like `[^1][^2]` when multiple sources support the same claim.\n\n"
    "Example:\n"
    "  The project reached $10M in funding last week[^1] and plans to launch in Q3[^2].\n\n"
    "  [^1]: https://techcrunch.com/example\n"
    "  [^2]: https://theverge.com/example"
)