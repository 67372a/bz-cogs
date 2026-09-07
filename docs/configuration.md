# Configuration System

## 1. Settings Architecture

The [`Settings`](../aiuser/settings/base.py:33) class uses the same composite pattern as the main cog, merging **9 settings sub-mixins**:

| Mixin | File | Command Group | Key Features |
|-------|------|---------------|--------------|
| `PromptSettings` | [`settings/prompt.py`](../aiuser/settings/prompt.py) | `[p]aiuser prompt` | Custom prompts, presets (cynical/neutral/tsundere), prompt length |
| `ImageScanSettings` | [`settings/image_scan.py`](../aiuser/settings/image_scan.py) | `[p]aiuser imagescan` | Toggle scanning, mode (LLM/AI Horde/Local), model selection |
| `ImageRequestSettings` | [`settings/image_request.py`](../aiuser/settings/image_request.py) | `[p]aiuser imagerequest` | Toggle, endpoint, trigger words, SD prompt, parameters |
| `HistorySettings` | [`settings/history.py`](../aiuser/settings/history.py) | `[p]aiuser history` | Backread count/time, forget, backfill, opt-in embed |
| `ResponseSettings` | [`settings/response.py`](../aiuser/settings/response.py) | `[p]aiuser response` | Model, parameters (logit_bias, temperature), reply percent |
| `TriggerSettings` | [`settings/triggers.py`](../aiuser/settings/triggers.py) | `[p]aiuser triggers` | Reply-to-mentions, conversation percent/time, grok, always-reply-on-words |
| `OwnerSettings` | [`settings/owner.py`](../aiuser/settings/owner.py) | `[p]aiuserowner` | OpenAI endpoint, request timeout, ratelimit reset, global text prompt |
| `RandomMessageSettings` | [`settings/random_message.py`](../aiuser/settings/random_message.py) | `[p]aiuser randommessage` | Toggle, percent, topics |
| `FunctionCallingSettings` | [`settings/functions.py`](../aiuser/settings/functions.py) | `[p]aiuser functions` | Toggle, enable/disable functions, location, OpenRouter tool parameters |

---

## 2. Config Hierarchy

Settings cascade from most-specific to least-specific scope:

```
Member → Role (highest) → Channel → Guild → Global
```

For prompts, this means:
1. Member-specific prompt (if set)
2. Role-specific prompt (highest role with one)
3. Channel-specific prompt
4. Guild-specific prompt
5. Global custom prompt
6. `DEFAULT_PROMPT`: *"You are {botname}. You are in a Discord text channel. Respond to anything... You are forbidden from saying you're an AI or a bot."*

The same hierarchy applies to `reply_percent` and `prefill_prompt`.

---

## 3. Default Values

### Global ([`DEFAULT_GLOBAL`](../aiuser/config/defaults.py:51))

| Setting | Default |
|---------|---------|
| `custom_openai_endpoint` | `None` (uses official OpenAI) |
| `openai_endpoint_request_timeout` | 60 seconds |
| `optout` / `optin` | `[]` (empty lists) |
| `ratelimit_reset` | `1990-01-01 00:01:00` (never locked) |
| `max_random_prompt_length` | 200 |
| `max_prompt_length` | 200 |

### Guild ([`DEFAULT_GUILD`](../aiuser/config/defaults.py:63))

| Setting | Default | Purpose |
|---------|---------|---------|
| `model` | `"google/gemini-2.5-flash"` | LLM model for chat |
| `reply_percent` | `0.01` (1%) | Base reply chance |
| `messages_backread` | `20` | Messages to scan for context |
| `messages_backread_seconds` | `7200` (2 hours) | Max gap between messages |
| `messages_min_length` | `2` | Min chars to trigger |
| `reply_to_mentions_replies` | `True` | Always reply when pinged |
| `function_calling` | `False` | Enable/disable tools |
| `scan_images` | `False` | Enable image scanning |
| `scan_images_mode` | `"supported-llm"` | Scanning backend |
| `conversation_reply_percent` | `0` | Continue conversation chance (0=disabled) |
| `conversation_reply_time` | `20` | Seconds to track conversation |
| `image_requests` | `False` | Enable image generation |
| `image_requests_endpoint` | `"dall-e-2"` | Default image provider |

### Other Scopes

| Scope | File | Fields |
|-------|------|--------|
| `DEFAULT_CHANNEL` | [`config/defaults.py:127`](../aiuser/config/defaults.py:127) | `custom_text_prompt: None`, `reply_percent: None` |
| `DEFAULT_ROLE` | [`config/defaults.py:131`](../aiuser/config/defaults.py:131) | `custom_text_prompt: None`, `reply_percent: None` |
| `DEFAULT_MEMBER` | [`config/defaults.py:136`](../aiuser/config/defaults.py:136) | `custom_text_prompt: None`, `reply_percent: None` |

---

## 4. Model Support

### 4.1 Vision Models

[`VISION_SUPPORTED_MODELS`](../aiuser/config/models.py:1) — 110+ models including:
- **OpenAI**: o1, o3, o4-mini, gpt-4.1, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4.5-preview
- **Anthropic**: Claude 3 Haiku/Sonnet/Opus, 3.5 Haiku/Sonnet, 3.7 Sonnet, Claude 4 Sonnet/Opus
- **Google**: Gemini 2.0 Flash/Pro, 2.5 Flash/Pro, 3 Pro preview
- **Meta**: Llama 3.2 Vision (11B/90B), Llama 4 Maverick/Scout
- **Others**: Qwen VL, Mistral Pixtral, Microsoft Phi-4, xAI Grok Vision, and more

### 4.2 Token Limits

[`MODELS_LIMITS`](../aiuser/config/models.py:119) — ~400 entries ranging from:
- **Minimum**: 4096 tokens (e.g., `airoboros-l2-70b`, `command`)
- **Maximum**: 2,000,000 tokens (e.g., `gemini-pro-1.5`, `auto` mode)
- **Common**: 128,000–200,000 (most modern models)

### 4.3 Logit Bias Exclusions

[`UNSUPPORTED_LOGIT_BIAS_MODELS`](../aiuser/config/models.py:113) — models where `logit_bias` is silently removed:
- `openai/o3-minin`, `o3-mini`, `o3-mini-2025-01-31`

---

## 5. Prompt Presets

[`DEFAULT_PRESETS`](../aiuser/config/defaults.py:10):

| Preset | Prompt |
|--------|--------|
| `cynical` | (Default) Respond unhelpfully and cynically. Forbidden from saying you're an AI or bot |
| `neutral` | Standard persona. Forbidden from saying you're an AI or bot |
| `tsundere` | Be a tsundere. Use text emojis. Forbidden from saying you're an AI or bot |

---

## 6. Removal Regex Patterns

[`DEFAULT_REMOVE_PATTERNS`](../aiuser/config/defaults.py:29) strips from LLM output:

```python
r'<message[^>]*>'    # Opening <message> tags
r'</message>'        # Closing </message> tags
r'<image[^>]*/>'     # Self-closing image/file/sticker tags
r'<file[^>]*/>'
r'<sticker[^>]*/>'
```

Supports `{botname}`, `{botdisplayname}`, `{authorname}`, `{authordisplayname}` dynamic placeholders.

---

## 7. Command Reference

| Command | Permission | Description |
|---------|------------|-------------|
| `[p]aiuser add <CHANNEL>` | Bot owner | Whitelist a channel |
| `[p]aiuser remove <CHANNEL>` | Bot owner | Remove from whitelist |
| `[p]aiuser percent <PERCENT>` | Bot owner | Set reply percentage |
| `[p]aiuser optin` | User | Opt into message processing |
| `[p]aiuser optout` | User | Opt out of message processing |
| `[p]aiuser forget` | Admin+ | Clear conversation history for channel |
| `[p]aiuser backfill <ID>` | Admin+ or public | Set context anchor |
| `[p]aiuser config` | Admin | View current guild config |
| `[p]aiuser functions` | Bot owner | Manage function calling |
| `[p]aiuserowner endpoint <URL>` | Bot owner | Set LLM endpoint |

### Owner Commands

- `/chat <text>` — Slash command to talk to the AI (cooldown: 30s global, 5s per user)
- `[p]aiuserowner endpoint <url>` — Set custom OpenAI-compatible endpoint or `"openrouter"` shortcut
- `[p]aiuserowner timeout <seconds>` — Set API request timeout

---

## 8. Prompt Cache Behavior

LLM providers (OpenAI, Gemini via OpenRouter, etc.) discount repeated prompt
**prefixes** — a request only gets cache read-hits when the beginning of the
message list is byte-identical to a recent request. AIUser actively manages
context for this (see [`MessagesList`](../aiuser/messages_list/messages.py)):

**What the cog does to preserve cache hits**

- The system prompt is built from stable variables only; per-request values
  (time, author, random number) are placed in a trailing `user` message.
- A per-channel **history watermark** ([`HistoryWatermark`](../aiuser/messages_list/messages.py))
  pins the oldest message included in context. History is never loaded older
  than the watermark, so the front of the prefix does not drift between
  requests. The watermark only advances when token headroom (70% of the
  token limit) is exhausted, when a genuine conversation time-gap forces a
  semantic reset, or after 2 hours of inactivity.
- Image captions (AI Horde / Local modes) are **pinned per image** in
  [`caption_cache`](../aiuser/utils/image_cache.py) — captions are serialized
  into history and a re-generated caption would diverge the prefix mid-way.
  Caption failures are pinned as a stable placeholder and retried after the
  cache TTL (30 min).
- Tool-calling rounds append messages only; the final-round stop instruction
  is a `user`-role message at the tail.
- A rolling **cache hit ratio** per channel is logged every 20 requests
  (`Prompt cache hit ratio for channel ...`), sourced from provider
  `cached_tokens` usage data.

**Events that intentionally reset the cache prefix (expected full misses)**

| Event | Effect |
|-------|--------|
| `[p]aiuser forget` (channel) or a prompt reset | History window cold-starts from the new start time |
| `[p]aiuser backfill <ID>` | The anchored request uses a different context layout; normal layout resumes on the next request |
| Toggling any function-calling / web tool setting | The system prompt changes (citation instructions are appended when web tools are enabled) |
| User/bot rename, channel topic change, guild emoji changes | Identity strings embedded in the system prompt or message headers change |
| First image trigger in a channel (or mix of image/non-image triggers) | Requests switch between the main model and `scan_images_model` — separate cache namespaces |
| Token headroom exhausted in a long conversation | The history watermark advances (jumps to ≤70% of the token limit) and older history drops from the front |
| Bot restart | In-memory watermark/caption caches reset; provider caches are typically expired anyway |
