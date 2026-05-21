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

## 2. Two-Phase LLM Pipeline

[`LLMPipeline`](../aiuser/response/chat/llm_pipeline.py:91) is the core interaction engine.

### 2.1 Phase 1: First LLM Call ([`phase1()`](../aiuser/response/chat/llm_pipeline.py:389))

```
1. Get custom parameters (logit_bias, model-specific kwargs)
2. Setup tools (local ToolCall subclasses + OpenRouter server tools)
3. Inject cached PDF annotations
4. Build API call with tools, tool_choice="auto"
5. Set safety settings to BLOCK_NONE for all categories
6. Call LLM with tenacity retries
7. Add assistant response to message history
8. Return (pre_text, reasoning, has_tools, is_incomplete)
```

### 2.2 Phase 2: Tool Execution + Second Call ([`phase2()`](../aiuser/response/chat/llm_pipeline.py:435))

```
1. Split tool calls into local vs. OpenRouter server tools
2. Execute local tools in parallel (asyncio.gather)
3. Collect generated images from tool executions
4. Inject images into message history for vision context
5. Second LLM call with tool_choice="none"
6. Process OpenRouter server tool results
7. Handle Gemini "replacement pattern" detection
8. Combine reasoning from both calls
9. Build ResponsePart objects
10. Return (final_text, final_reasoning, images)
```

### 2.3 Text Completeness Heuristic

[`_is_text_incomplete()`](../aiuser/response/chat/llm_pipeline.py:131) checks if tool-call prefacing text appears interrupted:
1. Trailing punctuation (`,`, `;`, `-`, `–`)
2. Trailing dash (`--` or ` -`)
3. Ends with a "bridging word" (articles, conjunctions, prepositions — 80+ word set)
4. Trailing whitespace

### 2.4 Retry & Error Handling

[`_create_completion_with_retry()`](../aiuser/response/chat/llm_pipeline.py:374):

| Parameter | Value |
|-----------|-------|
| Wait strategy | `wait_random_exponential(min=1, max=5)` |
| Max attempts | 4 |
| Retry triggers | `RateLimitError`, `APIConnectionError`, `InternalServerError` |

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
Phase 1 complete
    │
    ├── No tools: send reasoning → clean text → send response → done
    │
    └── Has tools:
         ├── Pre-text complete? → send early output
         └── Phase 2 → clean final text + images → send combined message
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
