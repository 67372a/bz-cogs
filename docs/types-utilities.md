# Types & Utilities

## 1. Core Types

### 1.1 `MixinMeta` ([`aiuser/types/abc.py`](../aiuser/types/abc.py:26))

Abstract base class defining the shared interface for all mixins composing the `AIUser` cog:

```python
class MixinMeta(ABC):
    bot: Red
    config: Config
    cached_options: dict
    override_prompt_start_time: dict[int, datetime]
    cached_messages: Cache[int, MessageEntry]
    ignore_regex: dict[int, re.Pattern]
    channels_whitelist: dict[int, list[int]]
    openai_client: AsyncOpenAI
    optindefault: dict[int, bool]
    generating_channels: set[int]
    message_queues: dict[int, Queue]
    processing_tasks: dict[int, Task]
    backfill_anchors: dict[int, discord.Message]
```

### 1.2 `CompositeMetaClass` ([`aiuser/types/abc.py`](../aiuser/types/abc.py:22))

Merges `type(commands.Cog)` and `type(ABC)` to support multiple mixin inheritance with ABC.

### 1.3 `MessageEntry` ([`aiuser/messages_list/entry.py`](../aiuser/messages_list/entry.py:5))

Frozen dataclass for ChatML-format messages:

```python
@dataclass(frozen=True)
class MessageEntry:
    role: Literal['user', 'assistant', 'system', 'tool']
    content: Union[str, list]
    name: str = None
    tool_calls: list = field(default_factory=list)
    tool_call_id: int = None
    reasoning_details: list = None
```

### 1.4 `ResponsePart` ([`aiuser/response/chat/llm_pipeline.py`](../aiuser/response/chat/llm_pipeline.py:41))

A segment of the bot's response to send to Discord:

```python
@dataclass
class ResponsePart:
    type: str  # "text" or "text_and_images"
    content: str
    images: List[Dict] = field(default_factory=list)
```

### 1.5 `ToolCall` ([`aiuser/functions/tool_call.py`](../aiuser/functions/tool_call.py:8))

Abstract base for function-calling tools:

```python
class ToolCall:
    schema: ToolCallSchema = None
    function_name: str = None

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx
        self.bot: Red = ctx.bot

    def run(self, arguments: dict):  # calls self._handle(arguments)
    def _handle(arguments: dict):     # implemented by subclasses
```

### 1.6 Schema Types ([`aiuser/functions/types.py`](../aiuser/functions/types.py))

```python
@dataclass(frozen=True)
class Parameters:
    properties: dict
    required: list = field(default_factory=list)
    type: str = "object"

@dataclass(frozen=True)
class Function:
    name: str
    description: str
    parameters: Parameters

@dataclass(frozen=True)
class ToolCallSchema:
    function: Function
    type: str = "function"
```

---

## 2. Enums ([`aiuser/types/enums.py`](../aiuser/types/enums.py))

| Enum | Values | Purpose |
|------|--------|---------|
| `ScanImageMode` | `LOCAL`, `AI_HORDE`, `LLM` | Image scanning backends |
| `MentionType` | `SERVER`, `USER`, `ROLE`, `CHANNEL` | Discord mention types |
| `OpenRouterToolType` | `WEB_SEARCH`, `WEB_FETCH`, `IMAGE_GENERATION`, `PDF_PARSING` | OpenRouter server tool types |

### OpenRouterToolType Values

| Constant | Value | Purpose |
|----------|-------|---------|
| `WEB_SEARCH` | `"openrouter:web_search"` | Server-side web search |
| `WEB_FETCH` | `"openrouter:web_fetch"` | Server-side URL fetching |
| `IMAGE_GENERATION` | `"openrouter:image_generation"` | Server-side image generation |
| `PDF_PARSING` | `"openrouter:pdf_parsing"` | PDF parsing plugin |

---

## 3. Type Aliases ([`aiuser/types/types.py`](../aiuser/types/types.py))

```python
COMPATIBLE_CHANNELS = Union[
    discord.TextChannel,
    discord.VoiceChannel,
    discord.StageChannel,
    discord.ForumChannel
]
COMPATIBLE_MENTIONS = Union[discord.Member, discord.Role, COMPATIBLE_CHANNELS]
```

---

## 4. Caching

### 4.1 `Cache` ([`aiuser/utils/cache.py`](../aiuser/utils/cache.py:1))

Simple dict-based LRU cache:

```python
class Cache(dict):
    def __init__(self, limit: int):
        # Maintains insertion order via self.keys list
        # On set: move to end or evict oldest
        # On get: move to end (LRU behavior)
        # Used for: cached_messages (limit=100)
```

### 4.2 Emoji Cache ([`aiuser/utils/utilities.py`](../aiuser/utils/utilities.py:29))

TTL-based cache (10 minutes) for guild emoji maps:
- Structure: `{guild_id: {emoji_name_lowercase: str(emoji)}}`
- Max 100 entries
- Used by [`get_guild_emoji_map()`](../aiuser/utils/utilities.py:47) and [`resolve_emojis_for_discord()`](../aiuser/utils/utilities.py:162)

### 4.3 Image Cache ([`aiuser/utils/image_cache.py`](../aiuser/utils/image_cache.py))

Separately caches generated image data for reuse.

---

## 5. Utility Functions ([`aiuser/utils/utilities.py`](../aiuser/utils/utilities.py))

### 5.1 `to_thread(timeout=300)`

Decorator that runs a synchronous function in an executor with a configurable timeout. Used for CPU-intensive regex operations.

### 5.2 `format_variables(ctx, text)`

Inserts dynamic variables into prompt/response text:
`{botname}`, `{botdisplayname}`, `{botowner}`, `{authorname}`, `{authordisplayname}`, `{authortoprole}`, `{authormention}`, `{serveremojis}`, `{servername}`, `{channelname}`, `{channeltopic}`, `{currentdate}`, `{currentweekday}`, `{currenttime}`, `{randomnumber}`

### 5.3 `get_enabled_tools(config, ctx)`

Dynamic tool discovery:
1. Imports all `aiuser/functions/*/tool_call.py` modules
2. Collects all `ToolCall.__subclasses__()`
3. Filters against `function_calling_functions` config list
4. Returns list of instantiated tool objects

### 5.4 `resolve_emojis_for_discord(ctx, text)`

Converts `:emoji_name:` shortcodes to full Discord emoji format (`<:name:id>` or `<a:name:id>` for animated) using cached guild emoji map.

### 5.5 `escape_unescaped_backticks(text)`

Safely escapes backticks in text for Discord markdown. Handles already-escaped backticks correctly:
- `` ` `` → `` \` ``
- `` \` `` → `` \` `` (no change)
- `` \\` `` → `` \\\` `` (backtick was not escaped)

### 5.6 `collapse_lines(text, replacement)`

Collapses multiple consecutive newlines into the specified replacement string.

### 5.7 Endpoint Detection

| Function | Returns |
|----------|---------|
| `is_using_openai_endpoint(client)` | `True` if base URL starts with `https://api.openai.com/` |
| `is_using_openrouter_endpoint(client)` | `True` if base URL starts with OpenRouter URL |
| `contains_youtube_link(content)` | `True` if content matches YouTube URL pattern |

### 5.8 `is_embed_valid(message)`

Checks if a Discord message has a valid embed (has title AND description).

---

## 6. Constants ([`aiuser/config/constants.py`](../aiuser/config/constants.py))

| Constant | Value | Purpose |
|----------|-------|---------|
| `OPENROUTER_URL` | `"https://openrouter.ai/api/"` | OpenRouter base URL |
| `RANDOM_MESSAGE_TASK_RETRY_SECONDS` | `1980` (33 min) | Random message interval |
| `IMAGE_REQUEST_AIHORDE_URL` | `"https://aihorde.net/api"` | AI Horde endpoint |
| `GROK_MAX_WORDS` | `25` | Max words for grok trigger |
| `REGEX_RUN_TIMEOUT` | `10` | Timeout for regex operations |

### Regex Patterns

| Pattern | File:Line | Matches |
|---------|-----------|---------|
| `URL_PATTERN` | `constants.py:14` | `https?://\S+` |
| `YOUTUBE_URL_PATTERN` | `constants.py:15` | YouTube and youtu.be links |
| `YOUTUBE_VIDEO_ID_PATTERN` | `constants.py:16` | YouTube video IDs |
| `SINGULAR_MENTION_PATTERN` | `constants.py:18` | `^<@!?&?(\d+)>$` (just a mention) |
| `EMOJI_PATTERN` | `utilities.py:22` | `:emoji_name:` shortcodes |
| `BACKTICK_PATTERN` | `utilities.py:24` | Backtick escape detection |

---

## 7. Dashboard Integration ([`aiuser/dashboard/base.py`](../aiuser/dashboard/base.py))

Registers three web dashboard pages via Red's `on_dashboard_cog_add` listener:

| Page | File | Purpose |
|------|------|---------|
| `main` | [`main_page.py`](../aiuser/dashboard/main_page.py) | Guild settings overview |
| `opt_consent` | [`consent_page.py`](../aiuser/dashboard/consent_page.py) | User opt-in/out management |
| `bot_owner_server_config` | [`owner_config_page.py`](../aiuser/dashboard/owner_config_page.py) | Bot-owner global settings |

---

## 8. OpenRouter Types ([`aiuser/types/openrouter_types.py`](../aiuser/types/openrouter_types.py))

Contains dataclasses for OpenRouter tool parameter serialization:

| Type | Used By |
|------|---------|
| `WebSearchParameters` | `OpenRouterWebSearch` |
| `WebFetchParameters` | `OpenRouterWebFetch` |
| `ImageGenerationParameters` | `OpenRouterImageGeneration` |
| `DirectImageGenerationParameters` | `GenerateImageToolCall` |

**Functions**: `serialize_parameters()` / `deserialize_parameters()` for JSON round-tripping, and `build_openrouter_tool_dict()` to construct the tool dict from a type enum and parameter dict.

---

## 9. Test Infrastructure ([`tests/`](../tests/))

| File | Purpose |
|------|---------|
| [`conftest.py`](../tests/conftest.py) | pytest fixtures |
| [`mock_importer.py`](../tests/mock_importer.py) | Cog mock helper |
| [`mock_functions_importer.py`](../tests/mock_functions_importer.py) | Functions mock helper |
| [`test_openrouter_functions.py`](../tests/test_openrouter_functions.py) | OpenRouter tool tests |
| [`test_openrouter_types.py`](../tests/test_openrouter_types.py) | Type serialization tests |
