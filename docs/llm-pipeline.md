# LLM Pipeline & Response System

## 1. Context Building: `MessagesList`

[`MessagesList`](../aiuser/messages_list/messages.py:49) converts Discord chat history into ChatML format for LLM consumption.

### 1.1 Initialization ([`_init()`](../aiuser/messages_list/messages.py:77))

1. Resolves model and token limit from guild config (or [`MODELS_LIMITS`](../aiuser/config/models.py:119) lookup)
2. Sets up `tiktoken` encoding for token counting
3. Adds the initiating message (unless skipped for backfill)
4. Picks persona prompt via hierarchy: **Member → Role → Channel → Guild → Global → `DEFAULT_PROMPT`**
5. Formats dynamic variables via [`format_variables()`](../aiuser/utils/utilities.py:69):

   | Variable | Example |
   |----------|---------|
   | `{botname}` | Bot's current username |
   | `{botdisplayname}` | Bot's display name |
   | `{botowner}` | Bot owner's username |
   | `{authorname}` | Message author's username |
   | `{authordisplayname}` | Author's display name |
   | `{authortoprole}` | Author's highest role |
   | `{authormention}` | Author mention string |
   | `{serveremojis}` | All server emojis as string |
   | `{servername}` | Guild name |
   | `{channelname}` | Channel name |
   | `{channeltopic}` | Channel topic/description |
   | `{currentdate}` | Current date (YYYY/MM/DD) |
   | `{currentweekday}` | Current day name |
   | `{currenttime}` | Current time (HH:MM) |
   | `{randomnumber}` | Random 0–100 |

6. Appends `XML_SYSTEM_PROMPT_APPENDIX` — instructs the LLM on the Semantic XML message format:

   > *"The chat history below is formatted in Semantic XML. User messages are wrapped in `<message>` tags containing metadata (id, author, time, and reply targets). Media attachments are represented by tags like `<image/>`, `<file/>`, or `<sticker/>."*

7. Optionally appends `CITATION_INSTRUCTIONS` when web search/fetch tools are enabled
8. Adds optional prefill prompt (member → role → channel → guild hierarchy)

### 1.2 History Collection ([`add_history()`](../aiuser/messages_list/messages.py:265))

Reads messages **before** and **after** the trigger, respecting:
- `messages_backread` count (default: 20)
- `messages_backread_seconds` time gap (default: 7200s)
- Token budget from model limit
- Filters: self-opt-in/out, ignore regex, bot thought embeds, opt-in embeds

### 1.3 Backfill System

The [`backfill`](../aiuser/settings/base.py:53) command sets a one-shot anchor message. [`add_backfill_history()`](../aiuser/messages_list/messages.py:283) loads messages chronologically from anchor → trigger, inserting between system prompt and trigger message.

### 1.4 Message Converter

[`MessageConverter`](../aiuser/messages_list/converter/converter.py:31) converts Discord messages to `MessageEntry` objects:

| Content Type | Handling |
|--------------|----------|
| Plain text | Wrapped in XML-like `<message>` tags with metadata |
| Image URLs | Downloaded, base64-encoded as `image_url` content parts |
| PDF URLs | Fetched and parsed |
| File attachments | Images scanned (OCR/BLIP/LLM), documents parsed (PDF text extraction, text files), videos described, audio transcribed |
| Stickers | Converted to `<sticker/>` tag |
| Embeds | Parsed from embed title/description/fields |
| YouTube links | YouTube transcript fetched and injected |
| Bot's own responses | Special formatting via `format_bot_embed_content` |

---

## 2. Tool-Calling Loop Pipeline

[`LLMPipeline`](../aiuser/response/chat/llm_pipeline.py:91) is the core interaction engine.
Instead of a fixed two-phase approach, the pipeline runs a **loop with a configurable
maximum number of tool-calling rounds** (`max_tool_rounds`, default: `2`, which preserves
the existing two-call behavior for backward compatibility).

### 2.1 Loop Flow

```
round = 0
response_text = ""
reasoning = ""
collected_images = []

while round < max_tool_rounds:
    round += 1
    │
    ├── Build API call with tools + tool_choice (see §2.2)
    ├── Call LLM with tenacity retries (see §2.4)
    ├── Add assistant response to message history
    │
    ├── IF no tool_calls:
    │      Append content → response_text
    │      Append reasoning → final_reasoning
    │      BREAK (LLM chose to respond directly)
    │
    └── IF has tool_calls:
           ├── Send early pre-text to Discord (first round only, if complete)
           ├── Split: local tool calls vs. OpenRouter server tools
           ├── Execute local tools in parallel (asyncio.gather)
           ├── Collect generated images from tool executions
           ├── Add tool results to message history (multimodal content parts
           │   for image-generating tools)
           ├── Add synthetic tool results for OpenRouter server-side tools
           └── Continue to next round

return final_text, final_reasoning, collected_images
```

### 2.2 `tool_choice` Per Round

| Round | `tool_choice` | Rationale |
|-------|--------------|-----------|
| 1 to N-1 | `"auto"` | LLM may call tools OR respond directly — the loop naturally terminates when `tool_calls` is empty |
| N (last round) | `"none"` + system prompt injection (see §2.3) | Attempt to force a final text response without further tool calls |

Tools are included in **every** API call because some providers reject calls with an
empty or missing `tools` array.

### 2.3 Stopping Tool Calls on the Final Round

Because `tool_choice="none"` can be unreliable across providers/proxies, a
**layered defense** is applied on the last round:

1. **`tool_choice="none"`** — Set per the OpenAI spec. Works on most providers.

2. **User-role stop instruction** — A temporary `user`-role message is appended
   before the final call (stripped afterward — not persisted to `MessagesList`).
   A `user` role is used instead of `system` to avoid breaking provider-level
   system-prompt caching (e.g., Google's context cache):

   > *"You have reached the maximum number of tool-calling rounds. Based on all
   > the information you have gathered, provide your final, complete response now.
   > Do NOT call any more tools — respond directly to the user."*

3. **Graceful forced-break** — If the LLM returns `tool_calls` despite the above,
   they are executed, results are added to history, and the loop is force-broken.
   The accumulated `response_text` and tool results form the final answer. A
   warning is logged so operators can detect models that ignore `tool_choice="none"`.

### 2.4 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `max_tool_rounds` | `2` | Maximum LLM calls in the tool-calling loop. Set to `1` to disable tools entirely (one call, no execution). Set higher (e.g., `5`) for chained/sequential tool calls. |

Added to [`DEFAULT_GUILD`](../aiuser/config/defaults.py:63) and exposed via the
`[p]functions` settings group.

### 2.5 Text Completeness Heuristic

[`_is_text_incomplete()`](../aiuser/response/chat/llm_pipeline.py:139) checks if tool-call
prefacing text appears interrupted. Applied on the **first round only** — if the pre-text
before tool calls looks complete, it is sent to Discord early. On subsequent rounds,
pre-text is accumulated internally and only delivered as part of the final combined message.
Checks:
1. Trailing punctuation (`,`, `;`, `-`, `–`)
2. Trailing dash (`--` or ` -`)
3. Ends with a "bridging word" (articles, conjunctions, prepositions — 80+ word set)
4. Trailing whitespace

### 2.6 Retry & Error Handling

[`_create_completion_with_retry()`](../aiuser/response/chat/llm_pipeline.py:374):

| Parameter | Value |
|-----------|-------|
| Wait strategy | `wait_random_exponential(min=1, max=5)` |
| Max attempts | 4 |
| Retry triggers | `RateLimitError`, `APIConnectionError`, `InternalServerError` |

Each LLM call within the loop uses the same retry logic independently.

[`is_response_unsatisfactory()`](../aiuser/response/chat/llm_pipeline.py:55) also retries on: empty content with `content_filter` or `error` finish_reason.

Error reactions:
| Error | Reaction |
|-------|----------|
| `ReadTimeout` | 💤 "aiuser request timed out" |
| `RateLimitError` | 💤 "aiuser request ratelimited" |
| `APIConnectionError` | ⚠️ "aiuser could not connect to LLM API" |
| `APIStatusError` | ⚠️ "aiuser LLM API error (Status NNN)" |
| Any other | ⚠️ "aiuser request failed" |
---

## 3. Response Dispatch

### 3.1 Dispatcher ([`dispatch_response()`](../aiuser/response/dispatcher.py:16))

Routes based on context:
1. If no messages_list and not an interaction → checks [`is_image_request()`](../aiuser/response/is_image_request.py) (LLM-based classification)
2. Otherwise builds `MessagesList` → routes to chat response

### 3.2 Chat Response Flow ([`create_chat_response()`](../aiuser/response/chat/response.py:110))

```
pipeline.run() completes
    │
    ├── No tools used: send reasoning → clean text → send response → done
    │
    └── Tools used:
         ├── Pre-text complete? → send early output (first round only)
         └── Final round complete → clean final text + images → send combined message
```

### 3.3 Response Cleaning

[`remove_patterns_from_response()`](../aiuser/response/chat/response.py:28) applies regex cleanup:
- Strips XML tags (`<message>`, `</message>`, `<image/>`, `<file/>`, `<sticker/>`)
- Replaces `{botname}` and `{botdisplayname}` with actual values
- Expands `{authorname}` placeholders against up to 20 recent authors
- Patterns from [`DEFAULT_REMOVE_PATTERNS`](../aiuser/config/defaults.py:29)

Additional processing:
- [`collapse_lines()`] — collapses multi-newlines
- [`escape_unescaped_backticks()`](../aiuser/utils/utilities.py:189) — safe backtick escaping for Discord markdown
- [`resolve_emojis_for_discord()`](../aiuser/utils/utilities.py:162) — converts `:emoji_name:` to full Discord format using cached guild emoji map

### 3.4 Response Delivery

| Condition | Behavior |
|-----------|----------|
| Text > 4096 chars | Paginated into embed pages with `X of N` footer |
| Can reply + heuristic passes | Replies to triggering message (25% chance, or if last message was bot) |
| Via slash command | Uses `interaction.followup.send()` |
| Default | Sends new message with embed |

Reasoning is always sent first as a "Thoughts" spoiler embed: `"||{reasoning}||"`.

All responses use `AllowedMentions(everyone=False, roles=False, users=[author])`.
