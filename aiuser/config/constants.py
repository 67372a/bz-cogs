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


OPENROUTER_URL = "https://openrouter.ai/api/"

XML_SYSTEM_PROMPT_APPENDIX = (
    "IMPORTANT CONTEXTUAL INSTRUCTIONS:\n"
    "1. The chat history below is formatted in Semantic XML. "
    "User messages are wrapped in <message> tags containing metadata (id, author, time, and reply targets).\n"
    "2. Media attachments are represented by tags like <image/>, <file/>, or <sticker/>.\n"
    "3. You are part of this chat. Do not output these XML tags yourself. "
    "Respond naturally in with text, inclusive of markdown and emojis as is relevant, adhering to the persona described above."
)

OPENROUTER_CITATION_INSTRUCTIONS = (
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