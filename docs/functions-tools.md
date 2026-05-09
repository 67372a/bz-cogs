# Function Calling System

## 1. Architecture

### 1.1 Base Classes

| Class | File | Purpose |
|-------|------|---------|
| [`ToolCall`](../aiuser/functions/tool_call.py:8) | Abstract base for all local tools. Has `schema`, `function_name`, `run()`, `_handle()` |
| [`ToolCallSchema`](../aiuser/functions/types.py:20) | Frozen dataclass: `{type: "function", function: Function}` |
| [`Function`](../aiuser/functions/types.py:13) | Frozen dataclass: `{name, description, parameters: Parameters}` |
| [`Parameters`](../aiuser/functions/types.py:6) | Frozen dataclass: `{properties: dict, required: list, type: "object"}` |

### 1.2 Tool Discovery

[`get_enabled_tools()`](../aiuser/utils/utilities.py:145):
1. Scans all subdirectories in `aiuser/functions/`
2. Imports each `*.tool_call` module (triggers `ToolCall.__subclasses__()` registration)
3. Reads `function_calling_functions` config list
4. Instantiates only those `ToolCall` subclasses whose `function_name` is in the enabled list

### 1.3 Tool Execution

In [`LLMPipeline.run_tool()`](../aiuser/response/chat/llm_pipeline.py:604):
1. Matches `tool_name` against `tool_obj.function_name`
2. Calls `tool_obj.run(arguments)` (which calls `_handle(arguments)`)
3. Adds `"request"` key to arguments for pipeline context
4. Returns string result, or error message on exception

Tool results are added to message history via [`add_tool_result()`](../aiuser/messages_list/messages.py:258) as `role: "tool"` entries.

---

## 2. Available Local Tools

### 2.1 Discord-Native Actions

#### `react_to_message`
- **File**: [`aiuser/functions/discord/tool_call.py`](../aiuser/functions/discord/tool_call.py:39)
- **Parameters**: `message_id`, `emoji` (server emoji name), `reason`
- **Action**: Fetches the message, looks up emoji from guild cache, adds reaction
- **Permission check**: `add_reactions` permission

#### `pin_message`
- **File**: [`aiuser/functions/discord/tool_call.py`](../aiuser/functions/discord/tool_call.py)
- **Parameters**: `message_id`, `reason`
- **Action**: Pins the message in the current channel
- **Note**: Discord's 50-pin limit applies

#### `send_tts_message`
- **File**: [`aiuser/functions/discord/tool_call.py`](../aiuser/functions/discord/tool_call.py)
- **Action**: Flags the next output as TTS (Text-to-Speech)

#### `change_user_nickname`
- **File**: [`aiuser/functions/user_change_nickname/tool_call.py`](../aiuser/functions/user_change_nickname/tool_call.py)
- **Parameters**: `username`, `nickname`

#### `timeout_user`
- **File**: [`aiuser/functions/user_timeout/tool_call.py`](../aiuser/functions/user_timeout/tool_call.py)
- **Parameters**: `username`, `reason`
- **Action**: Applies a brief 3-second timeout (symbolic)

### 2.2 Media & Creativity

#### `generate_image`
- **File**: [`aiuser/functions/generate_image/tool_call.py`](../aiuser/functions/generate_image/tool_call.py:44)
- **Parameters**: `prompt` (detailed narrative), `aspect_ratio` (14 options from 1:1 to 8:1), optional `reference_messages` (Discord message links for reference images)
- **Backend**: Direct OpenRouter API call via existing `AsyncOpenAI` client
- **Configurable**: Model, image_size via server settings
- **Output**: Stores generated images in `self.generated_images` for pipeline to send

#### `edit_image`
- **File**: [`aiuser/functions/edit_image/tool_call.py`](../aiuser/functions/edit_image/tool_call.py)
- **Action**: Downloads → edits → re-encodes existing images

### 2.3 Utility & Searches

#### `open_url`
- **Files**: [`functions/scrape/tool_call.py`](../aiuser/functions/scrape/tool_call.py), [`functions/scrape/scrape.py`](../aiuser/functions/scrape/scrape.py)
- **Backend**: `trafilatura` + `BeautifulSoup` for web scraping

#### `search_google`
- **Files**: [`functions/search/tool_call.py`](../aiuser/functions/search/tool_call.py), [`functions/search/query.py`](../aiuser/functions/search/query.py)
- **Backend**: Serper API

#### `get_weather`
- **Files**: [`functions/weather/tool_call.py`](../aiuser/functions/weather/tool_call.py), [`functions/weather/query.py`](../aiuser/functions/weather/query.py)
- **Parameters**: `location`, `days`
- **Also**: `get_local_weather(days)`, `is_daytime_local()` using configured default location

#### `wolfram_alpha`
- **Files**: [`functions/wolframalpha/tool_call.py`](../aiuser/functions/wolframalpha/tool_call.py), [`functions/wolframalpha/query.py`](../aiuser/functions/wolframalpha/query.py)
- **Purpose**: Mathematical and factual queries

### 2.4 Emergent Behavior

#### `do_not_respond`
- **File**: [`aiuser/functions/noresponse/tool_call.py`](../aiuser/functions/noresponse/tool_call.py)
- **Parameters**: `reason`, `respond`
- **Purpose**: Allows the LLM to explicitly choose not to respond to a message

---

## 3. OpenRouter Server Tools

These are NOT `ToolCall` subclasses. They are dict structures passed to the LLM API for **server-side execution**. OpenRouter handles the execution and returns results in the response.

| Tool | File | Type Value | Config Flag |
|------|------|------------|-------------|
| [`OpenRouterWebSearch`](../aiuser/functions/openrouter/web_search.py) | `"openrouter:web_search"` | `openrouter_web_search_enabled` |
| [`OpenRouterWebFetch`](../aiuser/functions/openrouter/web_fetch.py) | `"openrouter:web_fetch"` | `openrouter_web_fetch_enabled` |
| [`OpenRouterImageGeneration`](../aiuser/functions/openrouter/image_generation.py:21) | `"openrouter:image_generation"` | `openrouter_image_generation_enabled` |
| [`OpenRouterPdfParsing`](../aiuser/functions/openrouter/pdf_parsing.py) | PDF parsing plugins | `openrouter_pdf_parsing_enabled` |
| [`OpenRouterImageParsing`](../aiuser/functions/openrouter/image_parsing.py) | Server-side image analysis | `openrouter_image_parsing_enabled` |

### 3.1 OpenRouter Tool Structure

Each tool provides a `get_tool_dict()` method that returns a dict like:
```python
{"type": "openrouter:web_search", "parameters": {...}}
```

Parameters are loaded from guild config (customizable per-server) and deserialized via [`deserialize_parameters()`](../aiuser/types/openrouter_types.py).

### 3.2 Image Generation Tool Handling

`OpenRouterImageGeneration.handle_tool_response_content()`:
1. Extracts image URLs from the LLM response text using multiple regex strategies (file extensions + CDN path patterns)
2. Downloads images and sends them as Discord embeds
3. Has a transparent mode where the tool executes without requiring `tool_calls` in the API response

### 3.3 PDF Parsing

[`OpenRouterPdfParsing.build_plugins()`](../aiuser/functions/openrouter/pdf_parsing.py) builds the OpenRouter `plugins` array for PDF file parsing. Default engine is `cloudflare-ai` when the tool is disabled.

### 3.4 Web Search Output Format

When web search/fetch are enabled, the system appends [`OPENROUTER_CITATION_INSTRUCTIONS`](../aiuser/config/constants.py:33) to the system prompt, requiring the LLM to format citations as numbered footnotes `[^1]`, `[^2]` with source URLs at the end.

---

## 4. Function Configuration

**Command**: `[p]aiuser functions`
**File**: [`aiuser/settings/functions.py`](../aiuser/settings/functions.py:23)

| Subcommand | Purpose |
|------------|---------|
| `toggle` | Enable/disable function calling |
| `list` | Show available functions |
| `add` | Enable specific functions |
| `remove` | Disable specific functions |
| `location` | Set default lat/lng for weather tool |
| OpenRouter subcommands | Configure web search, web fetch, image gen, PDF parsing parameters |

Default location: `[49.24966, -123.11934]` (Vancouver, BC)
